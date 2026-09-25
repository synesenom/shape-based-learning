#!/usr/bin/env python3
"""Accuracy on named test conditions (appearance shifts, real-image OOD sets).

For experiments whose extra test sets are categories rather than bins
(``test_acc__app_texture``, ``test_acc__ood_sketch``, ...). Draws grouped
bars, one panel per model family with at most four models each (a colour
is never reused within a panel), and writes ``<stem>.md`` with every
number plus the extractor F1 per condition where ground truth exists.

Usage:
    python scripts/plot_named_tests.py --summary results/phase3/appearance_shift/summary.json --prefix app
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from plot_results import (  # noqa: E402
    GRID, PANEL_SLOTS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _style_axes, display_name, model_groups,
)


def conditions_for(per_model: Dict[str, dict], prefix: str) -> List[str]:
    tag = f"test_acc__{prefix}_"
    seen: List[str] = []
    for agg in per_model.values():
        for key in agg:
            if key.startswith(tag) and key[len("test_acc__"):] not in seen:
                seen.append(key[len("test_acc__"):])
    return seen


def _mean_ci(entry):
    if not entry or entry.get("mean") is None:
        return None, 0.0
    ci = entry.get("ci95")
    return entry["mean"], (ci if isinstance(ci, (int, float)) and ci == ci else 0.0)


def plot(per_model, conds, prefix, title, out_path):
    groups = model_groups(list(per_model))
    fig, axes = plt.subplots(len(groups), 1, figsize=(max(8.0, 1.3 * len(conds) + 3), 3.4 * len(groups)),
                             dpi=170, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    labels = [c[len(prefix) + 1:].replace("_", " ") for c in conds]
    for ax, (gtitle, models) in zip([a[0] for a in axes], groups):
        _style_axes(ax)
        ax.grid(False, axis="x")
        width = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            xs, ys, errs = [], [], []
            for j, c in enumerate(conds):
                mean, ci = _mean_ci(per_model[m].get(f"test_acc__{c}"))
                if mean is None:
                    continue
                xs.append(j + (i - (len(models) - 1) / 2) * width)
                ys.append(mean)
                errs.append(ci)
            ax.bar(xs, ys, width=width * 0.92, color=PANEL_SLOTS[i], label=display_name(m), zorder=3)
            ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=0.9, capsize=2, zorder=4)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, fontsize=8.5, color=TEXT_SECONDARY)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("test accuracy", color=TEXT_SECONDARY, fontsize=9)
        ax.set_title(gtitle, color=TEXT_PRIMARY, fontsize=10, loc="left")
        leg = ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(1.0, 0.0))
        for t in leg.get_texts():
            t.set_color(TEXT_SECONDARY)
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--prefix", required=True, help="test-name prefix, e.g. app, rev or ood")
    ap.add_argument("--stem", default=None)
    args = ap.parse_args()
    path = Path(args.summary)
    path = path if path.is_absolute() else REPO_ROOT / path
    summary = json.loads(path.read_text())
    stem = args.stem or f"{args.prefix}_conditions"
    seeds = summary.get("seeds", [])
    lines = [f"# {summary['config'].get('phase')} / {summary['config'].get('experiment')}: accuracy by test condition",
             "", f"Mean ± Student-t 95% CI over {len(seeds)} seeds.", ""]
    for n_key in sorted(summary["conditions"], key=int):
        per_model = summary["conditions"][n_key]
        conds = conditions_for(per_model, args.prefix)
        # Config order (e.g. flat first), not alphabetical.
        order = list(((summary.get("config") or {}).get("shift") or {}).get("tests") or {})
        conds.sort(key=lambda c: order.index(c) if c in order else len(order))
        if not conds:
            continue
        out = path.parent / (f"{stem}.png" if len(summary["conditions"]) == 1 else f"{stem}_n{n_key}.png")
        plot(per_model, conds, args.prefix,
             f"{summary['config'].get('experiment')}: {n_key}/class ({len(seeds)} seeds, bars = 95% CI)", out)
        print(f"Wrote {out}")
        header = "| model | " + " | ".join(c[len(args.prefix) + 1:] for c in conds) + " |"
        lines += [f"## {n_key} training examples per class", "", header, "|---|" + "---|" * len(conds)]
        for m, agg in per_model.items():
            cells = []
            for c in conds:
                mean, ci = _mean_ci(agg.get(f"test_acc__{c}"))
                cells.append("-" if mean is None else f"{mean:.3f} ± {ci:.3f}")
            lines.append(f"| {display_name(m)} | " + " | ".join(cells) + " |")
        f1_models = [m for m, a in per_model.items() if any(k.startswith(f"extractor_f1__{args.prefix}_") for k in a)]
        if f1_models:
            lines += ["", "Extractor F1 per condition (mean over seeds):", "", header, "|---|" + "---|" * len(conds)]
            for m in f1_models:
                cells = []
                for c in conds:
                    mean, _ = _mean_ci(per_model[m].get(f"extractor_f1__{c}"))
                    cells.append("-" if mean is None else f"{mean:.3f}")
                lines.append(f"| {display_name(m)} | " + " | ".join(cells) + " |")
        lines.append("")
    md = path.parent / f"{stem}.md"
    md.write_text("\n".join(lines))
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
