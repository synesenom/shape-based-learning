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
    "cnn_aug_strong": "#e87ba4",
    "cnn_probe": "#4a3aa7",
    "cnn_pretrained": "#8a6fd1",
    "st_oracle": "#008300",
    "bag_oracle": "#e34948",
}
FALLBACK_COLORS = ["#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Two-condition series for the shift chart: slots 1 and 2, validated as a
# pair (CVD dE 24.7, normal-vision 33.6, both above 3:1 on the surface).
CONDITION_COLORS = {"in-distribution": "#2a78d6", "shifted": "#eb6834"}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

DISPLAY_NAMES = {
    "cnn": "CNN",
    "cnn_aug": "CNN + aug",
    "cnn_aug_strong": "CNN + strong aug",
    "gnn_oracle": "GNN (oracle)",
    "gnn_classical": "GNN (classical)",
    "cnn_probe": "ImageNet probe",
    "cnn_pretrained": "ImageNet fine-tune",
    "st_oracle": "Set transformer (oracle)",
    "st_classical": "Set transformer (classical)",
    "bag_oracle": "Bag (oracle)",
    "bag_classical": "Bag (classical)",
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


def plot_shift(summary: dict, out_path: Path) -> None:
    """Grouped bars: in-distribution vs shifted accuracy, per model.

    Two series, so a legend is always present and each bar is directly
    labelled -- identity is never carried by colour alone. The drop is
    annotated beneath each pair, because the drop, not the shifted
    accuracy, is what an invariance claim rests on.
    """
    conditions = summary["conditions"]
    size = sorted(int(k) for k in conditions)[0]
    per_model = conditions[str(size)]
    models = [
        m for m in model_order(conditions)
        if _value(per_model.get(m, {}).get("test_acc")) is not None
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.grid(False, axis="x")

    width = 0.36
    gap = 0.02  # surface gap between adjacent fills
    for i, model in enumerate(models):
        agg = per_model[model]
        pairs = [
            ("in-distribution", _value(agg.get("test_acc_indist")), _ci(agg.get("test_acc_indist"))),
            ("shifted", _value(agg.get("test_acc")), _ci(agg.get("test_acc"))),
        ]
        for j, (label, mean, err) in enumerate(pairs):
            if mean is None:
                continue
            x = i + (j - 0.5) * (width + gap)
            ax.bar(x, mean, width=width, color=CONDITION_COLORS[label], zorder=3,
                   label=label if i == 0 else None)
            hi = min(err, 1.15 - mean)
            ax.errorbar(x, mean, yerr=[[min(err, mean)], [hi]], fmt="none",
                        ecolor=TEXT_SECONDARY, elinewidth=1.1, capsize=3, zorder=4)
            ax.annotate(f"{mean:.3f}", xy=(x, min(mean + err, 1.15)), xytext=(0, 5),
                        textcoords="offset points", ha="center", color=TEXT_SECONDARY, fontsize=7.5)

        drop = _value(agg.get("shift_drop"))
        if drop is not None:
            ax.annotate(f"drop {drop:+.3f}", xy=(i, 0), xytext=(0, -28),
                        textcoords="offset points", ha="center",
                        color=TEXT_SECONDARY, fontsize=8)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([display_name(m) for m in models], fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(axis="x", pad=18)
    ax.set_ylabel("test accuracy", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylim(0, 1.27)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    cond = summary.get("condition", {})
    n_seeds = len(summary.get("seeds", []))
    ax.set_title(
        f"Shift: {cond.get('name', '?')} ({n_seeds} seeds, bars = 95% CI)",
        color=TEXT_PRIMARY, fontsize=12, pad=12, loc="left",
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
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

    cond = summary.get("condition", {})
    is_shift = cond.get("is_shift", False)
    if is_shift:
        changed = ", ".join(
            f"`{k}` {v['train']} → {v['test']}" for k, v in cond.get("changed", {}).items()
        )
        lines += [
            f"Shift **{cond.get('name')}**: {changed}.",
            "",
            "Validation follows the *training* distribution: at selection time the "
            "shifted distribution is not available, and selecting on it would leak "
            "the test condition into training.",
            "",
        ]

    for n in sizes:
        per_model = conditions[str(n)]
        lines.append(f"## {n} training examples per class")
        lines.append("")
        if is_shift:
            lines += [
                "| model | in-distribution | shifted | drop | val accuracy | extractor F1 | params |",
                "|---|---|---|---|---|---|---|",
            ]
        else:
            lines += [
                "| model | test accuracy | val accuracy | tree/arrow_sign | extractor F1 | train (s) | params |",
                "|---|---|---|---|---|---|---|",
            ]
        for model in models:
            agg = per_model.get(model)
            if not agg:
                continue
            params = summary.get("n_parameters", {}).get(model)
            params_text = f"{params:,}" if params else "-"
            if is_shift:
                lines.append(
                    f"| {display_name(model)} | {_cell(agg.get('test_acc_indist'))} | "
                    f"{_cell(agg.get('test_acc'))} | {_cell(agg.get('shift_drop'))} | "
                    f"{_cell(agg.get('best_val_acc'))} | {_cell(agg.get('extractor_f1'))} | "
                    f"{params_text} |"
                )
            else:
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

    if summary.get("condition", {}).get("is_shift") and len(sizes) == 1:
        fig_path = out_dir / "shift.png"
        plot_shift(summary, fig_path)
    elif len(sizes) > 1:
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
