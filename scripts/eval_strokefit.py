#!/usr/bin/env python3
"""Stroke-fitter quality on synthetic sketches with known ground truth.

QuickDraw has no primitive annotation, so the stroke fitter's F1 cannot be
measured there. This measures it on the closest proxy that has ground
truth: synthetic objects whose primitives are traced as hand-like strokes
(``extract.strokes.synthetic_strokes``: one stroke per primitive, random
start point, positional jitter, a small closing gap). It is an upper
bound on QuickDraw quality -- people merge parts into one stroke, split
one part across strokes and add details no template has -- and is
reported as such.

Usage:
    python scripts/eval_strokefit.py
"""

import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shapeprim.data.objects import CLASS_NAMES  # noqa: E402
from shapeprim.data.synth_dataset import GenerationConfig, render_sample  # noqa: E402
from shapeprim.experiment import write_json  # noqa: E402
from shapeprim.extract.eval_match import match_primitives  # noqa: E402
from shapeprim.extract.strokes import StrokeFitter, synthetic_strokes  # noqa: E402


def main() -> None:
    fitter = StrokeFitter()
    out = {"n_per_class": 50, "conditions": {}}
    lines = ["# Stroke-fitter F1 on synthetic sketches (QuickDraw proxy)", "",
             "One stroke per ground-truth primitive, traced with jitter; 50 drawings per class, IoU >= 0.5 matching.",
             "An upper bound on QuickDraw quality (see scripts/eval_strokefit.py).", "",
             "| condition | precision | recall | F1 | " + " | ".join(CLASS_NAMES) + " |",
             "|---|---|---|---|" + "---|" * len(CLASS_NAMES)]
    for name, kw in (("frontal", {}), ("view_0-30", {"view_angle_range": (0, 30)}), ("jitter_x2", {"_jitter": 0.04})):
        jitter = kw.pop("_jitter", 0.02)
        cfg = GenerationConfig(image_size=64, **kw)
        per_class, ps, rs, fs = {}, [], [], []
        for c in CLASS_NAMES:
            f = []
            for t in range(50):
                s = render_sample(c, random.Random(t), cfg)
                r = match_primitives(fitter.fit(synthetic_strokes(s.primitives, random.Random(t), jitter), 64),
                                     s.primitives, 64)
                ps.append(r["precision"]); rs.append(r["recall"]); f.append(r["f1"])
            per_class[c] = float(np.mean(f))
            fs += f
        out["conditions"][name] = {"precision": float(np.mean(ps)), "recall": float(np.mean(rs)),
                                   "f1": float(np.mean(fs)), "f1_per_class": per_class}
        lines.append(f"| {name} | {np.mean(ps):.3f} | {np.mean(rs):.3f} | {np.mean(fs):.3f} | "
                     + " | ".join(f"{per_class[c]:.2f}" for c in CLASS_NAMES) + " |")
    d = REPO_ROOT / "results" / "phase3" / "strokefit_proxy"
    write_json(d / "summary.json", out)
    (d / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
