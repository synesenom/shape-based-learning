import json
import math

import pytest

from shapeprim.experiment import (
    RunDirectory,
    aggregate_runs,
    format_mean_ci,
    run_id_for,
    set_all_seeds,
    summarize,
    t_critical_95,
    write_json,
)


def test_ci_uses_the_t_distribution_not_the_normal_quantile():
    """With 3 seeds the multiplier is 4.303, not 1.96.

    Using the normal quantile for 3 runs understates the interval by more
    than a factor of two, which is exactly how a nonexistent gap gets
    reported as significant.
    """
    s = summarize([0.90, 0.92, 0.94])
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(0.92)
    assert s["std"] == pytest.approx(0.02)
    assert s["ci95"] == pytest.approx(4.303 * 0.02 / math.sqrt(3))


def test_single_seed_reports_no_interval():
    # A lone run is not an estimate of a distribution; reporting "+/- 0.000"
    # would read as precision rather than as a missing measurement.
    s = summarize([0.9])
    assert s["n"] == 1
    assert math.isnan(s["std"]) and math.isnan(s["ci95"])
    assert "1 seed" in format_mean_ci(s)


def test_empty_summary_is_not_a_crash():
    s = summarize([])
    assert s["n"] == 0 and math.isnan(s["mean"])
    assert format_mean_ci(s) == "n/a"


def test_identical_values_give_zero_width_interval():
    s = summarize([0.5, 0.5, 0.5])
    assert s["ci95"] == pytest.approx(0.0)


def test_format_mean_ci_shape():
    assert format_mean_ci(summarize([0.90, 0.92, 0.94])) == "0.920 +/- 0.050"


@pytest.mark.parametrize("df,expected", [(1, 12.706), (2, 4.303), (9, 2.262), (20, 2.086)])
def test_t_table_values(df, expected):
    assert t_critical_95(df) == pytest.approx(expected)


def test_t_critical_falls_back_for_large_df():
    assert t_critical_95(500) == pytest.approx(1.980)
    assert math.isnan(t_critical_95(0))


def test_run_id_is_stable_and_readable():
    assert run_id_for("cnn", 3, n=25) == "cnn_n-25_seed-3"
    assert run_id_for("gnn_oracle", 0, n=5) == run_id_for("gnn_oracle", 0, n=5)
    # None-valued conditions are omitted rather than rendered as "None".
    assert run_id_for("cnn", 1, n=10, shift=None) == "cnn_n-10_seed-1"


def test_run_directory_round_trip(tmp_path):
    run = RunDirectory(tmp_path, "phase1", "baseline", run_id_for("cnn", 0, n=5))
    assert not run.exists("metrics")
    run.save_config({"lr": 0.001})
    run.save_metrics({"test_acc": 0.93, "seed": 0})
    assert run.exists("metrics")
    assert run.load("metrics")["test_acc"] == 0.93
    assert run.path.relative_to(tmp_path).parts == ("phase1", "baseline", "cnn_n-5_seed-0")


def test_saved_config_records_environment(tmp_path):
    run = RunDirectory(tmp_path, "phase1", "baseline", "r0")
    run.save_config({"lr": 0.001})
    saved = json.loads((run.path / "config.json").read_text())
    assert saved["config"]["lr"] == 0.001
    assert "python" in saved["environment"] and "timestamp_utc" in saved["environment"]


def test_load_missing_returns_none(tmp_path):
    assert RunDirectory(tmp_path, "p", "e", "r").load("metrics") is None


def test_corrupt_metrics_file_loads_as_none(tmp_path):
    run = RunDirectory(tmp_path, "p", "e", "r")
    (run.path / "metrics.json").write_text("{ truncated")
    assert run.load("metrics") is None


def test_write_json_handles_nan_and_tuples(tmp_path):
    # NaN is not valid JSON; it must be written as null rather than
    # producing a file that json.load rejects on the next run.
    path = tmp_path / "m.json"
    write_json(path, {"acc": float("nan"), "range": (0.5, 0.85)})
    loaded = json.loads(path.read_text())
    assert loaded["acc"] is None
    assert loaded["range"] == [0.5, 0.85]
    assert not list(tmp_path.glob("*.tmp"))


def test_aggregate_runs_groups_metrics():
    rows = [
        {"seed": 0, "test_acc": 0.90, "train_seconds": 10.0},
        {"seed": 1, "test_acc": 0.92, "train_seconds": 12.0},
        {"seed": 2, "test_acc": 0.94, "train_seconds": 11.0},
    ]
    agg = aggregate_runs(rows, ["test_acc", "train_seconds", "missing_key"])
    assert agg["n_runs"] == 3 and agg["seeds"] == [0, 1, 2]
    assert agg["test_acc"]["mean"] == pytest.approx(0.92)
    assert "missing_key" not in agg


def test_aggregate_skips_none_values():
    rows = [{"seed": 0, "twin_test_acc": None}, {"seed": 1, "twin_test_acc": 0.5}]
    agg = aggregate_runs(rows, ["twin_test_acc"])
    assert agg["twin_test_acc"]["n"] == 1


def test_set_all_seeds_makes_torch_reproducible():
    import torch

    set_all_seeds(123)
    a = torch.randn(4)
    set_all_seeds(123)
    assert torch.equal(a, torch.randn(4))
