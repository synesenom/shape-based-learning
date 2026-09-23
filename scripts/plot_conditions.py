#!/usr/bin/env python3
"""Accuracy vs a binned test condition (viewing angle, rotation, ...).

Reads a ``summary.json`` whose runs carry named extra tests
(``test_acc__<prefix>_<lo>-<hi>``, see ``shift.tests`` in the experiment
configs) and draws accuracy against the bin centre, one figure per
training-set size, as small multiples of at most four series each:

- pixel models;
- graph models on oracle primitives (representation alone);
- graph models on extracted primitives (representation + stage 1),
  with the extractor's F1 per bin as its own panel underneath -- never on
  a second y-axis.

The training range is shaded, so extrapolation is visible at a glance.
Also writes ``<stem>.md`` with every number as a table (two of the four
series colours sit below 3:1 on white, so the table and direct labels are
the relief, not decoration).

Usage:
    python scripts/plot_conditions.py --summary results/phase2/angle_extrapolation/summary.json \
        --prefix angle --train-range 0 30 --xlabel "viewing angle (degrees)"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from plot_results import GRID, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _style_axes, display_name  # noqa: E402

# Validated categorical slots 1-4 (scripts/validate_palette.js: all checks
# pass on #fcfcfb; slots 3-4 below 3:1 contrast -> direct labels + table).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]  # secondary encoding: identity never colour-alone

MORE_NAMES = {
    "cnn_aug_view": "CNN + view aug",
    "cnn_aug_view_rot": "CNN + view/rot aug",
    "gnn_oracle_affine": "GNN (oracle, affine frame)",
    "gnn_classical_affine": "GNN (classical, affine frame)",
    "gnn_learned": "GNN (learned)",
    "gnn_learned_affine": "GNN (learned, affine frame)",
}


def name_of(model: str) -> str:
    return MORE_NAMES.get(model, display_name(model))


def default_groups(models: Sequence[str]) -> List[Tuple[str, List[str]]]:
    pixel = [m for m in models if m.startswith("cnn")]
    oracle = [m for m in models if not m.startswith("cnn") and "oracle" in m]
    extracted = [m for m in models if not m.startswith("cnn") and "oracle" not in m]
    groups = []
    for title, ms in (("Pixel models", pixel), ("Graph models, oracle primitives", oracle),
                      ("Graph models, extracted primitives", extracted)):
        for i in range(0, len(ms), 4):
            groups.append((title if i == 0 else f"{title} (cont.)", ms[i:i + 4]))
    return groups


def bins_for(per_model: Dict[str, dict], prefix: str) -> List[Tuple[str, float, float]]:
    pat = re.compile(rf"^test_acc__({re.escape(prefix)}_(\d+)-(\d+))$")
    found = {}
    for agg in per_model.values():
        for key in agg:
            m = pat.match(key)
            if m:
                found[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return sorted(((k, lo, hi) for k, (lo, hi) in found.items()), key=lambda b: b[1])


def _mean_ci(entry: Optional[dict]) -> Tuple[Optional[float], float]:
    if not entry or entry.get("mean") is None:
        return None, 0.0
    ci = entry.get("ci95")
    return entry["mean"], (ci if isinstance(ci, (int, float)) and ci == ci else 0.0)


def plot_one(per_model, bins, groups, train_range, xlabel, title, out_path, show_f1: bool):
    n_panels = len(groups) + (1 if show_f1 else 0)
    cols = min(n_panels, 2)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 4.5 * rows), dpi=170, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    xs = [(lo + hi) / 2 for _, lo, hi in bins]

    def draw(ax, models, key_fmt, ylabel, label_fn=name_of):
        _style_axes(ax)
        if train_range:
            ax.axvspan(*train_range, color=GRID, alpha=0.6, zorder=0, linewidth=0)
            ax.annotate("trained here", xy=(sum(train_range) / 2, 1.0), ha="center", va="top",
                        color=TEXT_SECONDARY, fontsize=8)
        ends = []
        for i, m in enumerate(models):
            pts = [(x, *_mean_ci(per_model.get(m, {}).get(key_fmt.format(b[0])))) for x, b in zip(xs, bins)]
            pts = [(x, y, c) for x, y, c in pts if y is not None]
            if not pts:
                continue
            px, py, pc = zip(*pts)
            ax.fill_between(px, [max(0, y - c) for y, c in zip(py, pc)], [min(1, y + c) for y, c in zip(py, pc)],
                            color=SLOTS[i], alpha=0.14, linewidth=0, zorder=2)
            label = label_fn(m)
            ax.plot(px, py, color=SLOTS[i], linewidth=2, marker=MARKERS[i], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, label=label)
            ends.append([py[-1], px[-1], label])
        # Direct end labels, nudged apart so coincident series stay legible.
        ends.sort()
        for j in range(1, len(ends)):
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.055)
        for y, x, label in ends:
            ax.annotate(label, xy=(x, min(y, 1.02)), xytext=(6, 0), textcoords="offset points",
                        color=TEXT_SECONDARY, fontsize=7, va="center")
        ax.set_ylim(0, 1.03)
        ax.set_xlim(min(b[1] for b in bins), max(b[2] for b in bins) * 1.45)
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
        leg = ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        for t in leg.get_texts():
            t.set_color(TEXT_SECONDARY)

    flat = [a for row in axes for a in row]
    for ax, (gtitle, models) in zip(flat, groups):
        draw(ax, models, "test_acc__{}", "test accuracy")
        ax.set_title(gtitle, color=TEXT_PRIMARY, fontsize=10, loc="left")
    if show_f1:
        ax = flat[len(groups)]
        # One series per non-oracle extractor (the oracle is 1.0 by
        # definition, and the graph frame does not change extraction).
        chosen = {}
        for m, agg in per_model.items():
            ext = agg.get("extractor_name")
            if ext and ext != "oracle" and any(k.startswith("extractor_f1__") for k in agg):
                chosen.setdefault(ext, m)
        by_model = {m: e for e, m in chosen.items()}
        draw(ax, list(by_model)[:4], "extractor_f1__{}", "extractor F1",
             label_fn=lambda m: f"{by_model[m]} extractor")
        ax.set_title("Primitive extraction F1 (vs ground truth)", color=TEXT_PRIMARY, fontsize=10, loc="left")
    for ax in flat[n_panels:]:
        ax.axis("off")
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def write_table(per_model, bins, n, lines: List[str]) -> None:
    lines.append(f"## {n} training examples per class")
    lines.append("")
    header = "| model | " + " | ".join(b[0] for b in bins) + " |"
    lines += [header, "|---|" + "---|" * len(bins)]
    for m, agg in per_model.items():
        cells = []
        for b in bins:
            mean, ci = _mean_ci(agg.get(f"test_acc__{b[0]}"))
            cells.append("-" if mean is None else f"{mean:.3f} ± {ci:.3f}")
        lines.append(f"| {name_of(m)} | " + " | ".join(cells) + " |")
    f1_models = [m for m, agg in per_model.items() if any(k.startswith("extractor_f1__") for k in agg)]
    if f1_models:
        lines += ["", "Extractor F1 per bin (mean over seeds):", "", header, "|---|" + "---|" * len(bins)]
        for m in f1_models:
            cells = []
            for b in bins:
                mean, _ = _mean_ci(per_model[m].get(f"extractor_f1__{b[0]}"))
                cells.append("-" if mean is None else f"{mean:.3f}")
            lines.append(f"| {name_of(m)} | " + " | ".join(cells) + " |")
    lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--prefix", required=True, help="bin name prefix, e.g. angle or rot")
    ap.add_argument("--train-range", type=float, nargs=2, default=None)
    ap.add_argument("--xlabel", default="condition")
    ap.add_argument("--stem", default=None, help="output file stem (default: <prefix>_curve)")
    args = ap.parse_args()

    path = Path(args.summary)
    path = path if path.is_absolute() else REPO_ROOT / path
    summary = json.loads(path.read_text())
    stem = args.stem or f"{args.prefix}_curve"
    seeds = summary.get("seeds", [])
    lines = [
        f"# {summary['config'].get('phase')} / {summary['config'].get('experiment')}: accuracy by {args.prefix} bin",
        "",
        f"Mean ± Student-t 95% CI over {len(seeds)} seeds. "
        + (f"Training range: {args.train_range[0]:g}-{args.train_range[1]:g}." if args.train_range else ""),
        "",
    ]
    for n_key in sorted(summary["conditions"], key=int):
        per_model = summary["conditions"][n_key]
        bins = bins_for(per_model, args.prefix)
        if not bins:
            continue
        models = list(per_model)
        out = path.parent / (f"{stem}.png" if len(summary["conditions"]) == 1 else f"{stem}_n{n_key}.png")
        plot_one(
            per_model, bins, default_groups(models), args.train_range, args.xlabel,
            f"{summary['config'].get('experiment')}: accuracy vs {args.xlabel}, {n_key}/class "
            f"({len(seeds)} seeds, band = 95% CI)",
            out, show_f1=any("extractor_f1__" in k for a in per_model.values() for k in a),
        )
        print(f"Wrote {out}")
        write_table(per_model, bins, n_key, lines)
    md = path.parent / f"{stem}.md"
    md.write_text("\n".join(lines))
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
