"""Run bookkeeping: seeding, run directories, and aggregation across seeds.

PLAN.md section 4 sets the rules this module enforces mechanically:
3 seeds per run, mean +/- spread reported, and config/seed/metrics saved
to ``results/<phase>/<experiment>/<run_id>/``. Doing this by hand is how
single-seed numbers like ``results/phase1/cnn_vs_gnn.json`` end up being
quoted as findings, so the driver writes a directory per run and an
aggregate per condition, and nothing reports a bare number without its
seed count.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Two-sided 95% Student-t critical values by degrees of freedom (n - 1).
# Hardcoded to avoid a scipy dependency for one lookup. With 3 seeds the
# multiplier is 4.303, not 1.96 -- a distinction worth keeping visible,
# since a 3-seed "95% CI" computed with the normal quantile understates
# the interval by well over a factor of two.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T_95:
        return _T_95[df]
    candidates = [k for k in _T_95 if k <= df]
    return _T_95[max(candidates)] if candidates else 1.960


def set_all_seeds(seed: int) -> None:
    """Seed python, numpy and torch. Call once per run, before model init."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    """mean, std, 95% CI half-width and range for a list of per-seed values."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    if n == 1:
        # One seed is not an estimate of anything. Say so in the record
        # rather than printing "+/- 0.000", which reads like precision.
        return {"n": 1, "mean": vals[0], "std": float("nan"), "ci95": float("nan"),
                "min": vals[0], "max": vals[0], "values": vals}
    sd = stdev(vals)
    return {
        "n": n,
        "mean": mean(vals),
        "std": sd,
        "ci95": t_critical_95(n - 1) * sd / math.sqrt(n),
        "min": min(vals),
        "max": max(vals),
        "values": vals,
    }


def format_mean_ci(summary: Dict[str, Any], digits: int = 3) -> str:
    if not summary or summary.get("n", 0) == 0:
        return "n/a"
    if summary["n"] == 1:
        return f"{summary['mean']:.{digits}f} (1 seed)"
    return f"{summary['mean']:.{digits}f} +/- {summary['ci95']:.{digits}f}"


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Current commit hash, so a results directory can be traced to code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(repo_root),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    return info


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(_jsonable(payload), f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class RunDirectory:
    """``results/<phase>/<experiment>/<run_id>/`` with config + metrics."""

    def __init__(self, results_root: Path, phase: str, experiment: str, run_id: str):
        self.path = Path(results_root) / phase / experiment / run_id
        self.path.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Dict[str, Any], repo_root: Optional[Path] = None) -> None:
        write_json(self.path / "config.json", {"config": config, "environment": environment_info(repo_root)})

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        write_json(self.path / "metrics.json", metrics)

    def save(self, name: str, payload: Any) -> None:
        write_json(self.path / f"{name}.json", payload)

    def exists(self, name: str = "metrics") -> bool:
        return (self.path / f"{name}.json").exists()

    def load(self, name: str = "metrics") -> Optional[dict]:
        p = self.path / f"{name}.json"
        if not p.exists():
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None


def run_id_for(model: str, seed: int, **kwargs) -> str:
    """Stable, readable run id: model, the condition, then the seed."""
    parts = [model]
    for key in sorted(kwargs):
        value = kwargs[key]
        if value is None:
            continue
        parts.append(f"{key}-{value}")
    parts.append(f"seed-{seed}")
    return "_".join(str(p).replace("/", "-").replace(" ", "") for p in parts)


def aggregate_runs(records: Iterable[Dict[str, Any]], metric_keys: Sequence[str]) -> Dict[str, Any]:
    """Group per-seed records into {metric: summary}."""
    records = list(records)
    out: Dict[str, Any] = {"n_runs": len(records), "seeds": [r.get("seed") for r in records]}
    for key in metric_keys:
        values = [r[key] for r in records if key in r and r[key] is not None]
        if values:
            out[key] = summarize(values)
    return out
