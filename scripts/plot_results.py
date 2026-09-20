#!/usr/bin/env python3
"""Plot and tabulate a run's ``summary.json``.

Reads ``results/<phase>/<experiment>/summary.json`` (written by
``run_experiment.py``) and emits, next to it:

- ``learning_curve.png`` when the experiment swept training-set size, or
  ``comparison.png`` for a single condition;
- ``results.md``, the same numbers as a table.

The table is not redundant. Two of the four series colors sit below 3:1
contrast on a white surface, so the palette's relief rule requires either
visible direct labels or a table view; this ships both. It also means the
numbers survive being printed in grayscale, which is where a color-only
figure fails a reviewer.

Usage:
    python scripts/plot_results.py --summary results/phase1/baseline/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shapeprim.experiment import t_critical_95  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Categorical slots 1-4 of the validated reference palette, assigned in
# fixed order and never cycled: pixel models take slots 1-2, graph models
# slots 3-4. Assignment follows the entity, so dropping a model from a plot
# must not repaint the others.
SERIES_COLORS = {
    "cnn": "#2a78d6",
    "cnn_aug": "#eb6834",
    "gnn_oracle": "#1baf7a",
    "gnn_classical": "#eda100",
}
FALLBACK_COLORS = ["#e87ba4", "#008300", "#4a3aa7", "#e34948"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

DISPLAY_NAMES = {
    "cnn": "CNN",
    "cnn_aug": "CNN + augmentation",
    "gnn_oracle": "GNN (oracle)",
    "gnn_classical": "GNN (classical)",
}


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)


def color_for(model: str, index: int) -> str:
    return SERIES_COLORS.get(model, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def _style_axes(ax) -> None:
    """Recessive grid and axes; the data is the only prominent ink."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def _value(summary_entry: Optional[dict], field: str = "mean") -> Optional[float]:
    if not summary_entry:
        return None
    v = summary_entry.get(field)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def _ci(entry: Optional[dict]) -> float:
    v = _value(entry, "ci95")
    return 0.0 if v is None else v


def model_order(conditions: Dict[str, dict]) -> List[str]:
    """Stable model ordering: known series first, then whatever else appears."""
    seen: List[str] = []
    for per_model in conditions.values():
        for name in per_model:
            if name not in seen:
                seen.append(name)
    known = [m for m in SERIES_COLORS if m in seen]
    return known + [m for m in seen if m not in known]


def plot_learning_curve(summary: dict, out_path: Path, metric: str = "test_acc") -> None:
    conditions = summary["conditions"]
    sizes = sorted(int(k) for k in conditions)
    models = model_order(conditions)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    for i, model in enumerate(models):
        xs, ys, los, his = [], [], [], []
        for n in sizes:
            entry = conditions[str(n)].get(model, {}).get(metric)
            mean = _value(entry)
            if mean is None:
                continue
            half = _ci(entry)
            xs.append(n)
            ys.append(mean)
            los.append(max(0.0, mean - half))
            his.append(min(1.0, mean + half))
        if not xs:
            continue
        color = color_for(model, i)
        ax.fill_between(xs, los, his, color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, label=display_name(model))
        # Direct label at the right end: identity is never color-alone, and
        # two of these hues are below the contrast floor on white.
        ax.annotate(
            display_name(model),
            xy=(xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=TEXT_SECONDARY,
            fontsize=8,
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("training examples per class", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("test accuracy", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylim(0, 1.02)
    n_seeds = len(summary.get("seeds", []))
    ax.set_title(
        f"Few-shot learning curves ({n_seeds} seeds, band = 95% CI)",
        color=TEXT_PRIMARY, fontsize=12, pad=12, loc="left",
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    # Room for the direct labels.
    ax.set_xlim(right=sizes[-1] * 2.2)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_single_condition(summary: dict, out_path: Path, metric: str = "test_acc") -> None:
    conditions = summary["conditions"]
    size = sorted(int(k) for k in conditions)[0]
    per_model = conditions[str(size)]
    models = [m for m in model_order(conditions) if _value(per_model.get(m, {}).get(metric)) is not None]

    means = [_value(per_model[m][metric]) for m in models]
    errs = [_ci(per_model[m][metric]) for m in models]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.grid(False, axis="x")

    top = 1.15
    xs = range(len(models))
    for i, (x, mean, err) in enumerate(zip(xs, means, errs)):
        color = color_for(models[i], i)
        ax.bar(x, mean, width=0.58, color=color, zorder=3)
        # A few-seed interval can be wider than the whole accuracy scale.
        # Clip the whisker to the axis instead of letting it run off the
        # figure, and mark that it was clipped so the plot doesn't quietly
        # understate the uncertainty.
        lo = min(err, mean)
        hi = min(err, top - mean)
        ax.errorbar(
            x, mean, yerr=[[lo], [hi]], fmt="none",
            ecolor=TEXT_SECONDARY, elinewidth=1.2, capsize=4, zorder=4,
        )
        clipped = err > hi + 1e-9
        label = f"{mean:.3f}"
        if err > 0:
            label += f"\n±{err:.3f}" + (" (clipped)" if clipped else "")
        ax.annotate(
            label,
            xy=(x, min(mean + err, top)),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            color=TEXT_SECONDARY,
            fontsize=8,
        )

    ax.set_xticks(list(xs))
    ax.set_xticklabels([display_name(m) for m in models], fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("test accuracy", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylim(0, top + 0.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    n_seeds = len(summary.get("seeds", []))
    ax.set_title(
        f"Phase 1 baseline, {size} examples/class ({n_seeds} seeds, bars = 95% CI)",
        color=TEXT_PRIMARY, fontsize=12, pad=12, loc="left",
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def _cell(entry: Optional[dict], digits: int = 3) -> str:
    mean = _value(entry)
    if mean is None:
        return "-"
    n = entry.get("n", 0)
    if n <= 1:
        return f"{mean:.{digits}f} (1 seed)"
    return f"{mean:.{digits}f} ± {entry['ci95']:.{digits}f}"


def write_markdown(summary: dict, out_path: Path) -> None:
    conditions = summary["conditions"]
    sizes = sorted(int(k) for k in conditions)
    models = model_order(conditions)
    seeds = summary.get("seeds", [])

    lines = [
        f"# {summary['config'].get('phase', '?')} / {summary['config'].get('experiment', '?')}",
        "",
        f"Seeds: {seeds}. Test split drawn from fixed seed "
        f"{summary['config'].get('test_seed')}, so the spread is over training "
        "stochasticity and the training draw, not over test sets.",
        "",
        "Intervals are Student-t 95% confidence intervals over seeds "
        f"(t = {t_critical_95(max(len(seeds) - 1, 1)):.3f} at {len(seeds)} seeds, not the "
        "normal-quantile 1.96).",
        "",
    ]

    for n in sizes:
        per_model = conditions[str(n)]
        lines += [
            f"## {n} training examples per class",
            "",
            "| model | test accuracy | val accuracy | tree/arrow_sign | extractor F1 | train (s) | params |",
            "|---|---|---|---|---|---|---|",
        ]
        for model in models:
            agg = per_model.get(model)
            if not agg:
                continue
            params = summary.get("n_parameters", {}).get(model)
            params_text = f"{params:,}" if params else "-"
            lines.append(
                f"| {display_name(model)} | {_cell(agg.get('test_acc'))} | "
                f"{_cell(agg.get('best_val_acc'))} | {_cell(agg.get('twin_test_acc'))} | "
                f"{_cell(agg.get('extractor_f1'))} | {_cell(agg.get('train_seconds'), 0)} | "
                f"{params_text} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", required=True, help="Path to a summary.json")
    parser.add_argument("--metric", default="test_acc")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = REPO_ROOT / summary_path
    with open(summary_path) as f:
        summary = json.load(f)

    out_dir = summary_path.parent
    sizes = sorted(int(k) for k in summary["conditions"])

    if len(sizes) > 1:
        fig_path = out_dir / "learning_curve.png"
        plot_learning_curve(summary, fig_path, metric=args.metric)
    else:
        fig_path = out_dir / "comparison.png"
        plot_single_condition(summary, fig_path, metric=args.metric)

    table_path = out_dir / "results.md"
    write_markdown(summary, table_path)
    print(f"Wrote {fig_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
