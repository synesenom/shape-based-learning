#!/usr/bin/env python3
"""Classical-extractor quality on clean Phase 1 drawings, by resolution, class and type.

PLAN.md section 5 asks for a near-perfect classical extractor on clean
drawings. This measures it at 64 px (the experiments' resolution) and at
128 px (PLAN.md's nominal resolution), and breaks recall down by
primitive type so the failure mode is on record rather than anecdotal.

Usage:
    python scripts/eval_classical.py
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shapeprim.data.objects import CLASS_NAMES  # noqa: E402
from shapeprim.data.synth_dataset import GenerationConfig, render_sample  # noqa: E402
from shapeprim.experiment import write_json  # noqa: E402
from shapeprim.extract.classical import ClassicalExtractor, ClassicalExtractorV2  # noqa: E402
from shapeprim.extract.eval_match import match_primitives  # noqa: E402

N = 50


def main() -> None:
    out = {"n_per_class": N}
    lines = ["# Classical extractor quality on clean Phase 1 drawings", "",
             f"{N} drawings per class, IoU >= 0.5 same-type matching, no distractors.", ""]
    for ex in (ClassicalExtractor(), ClassicalExtractorV2()):
        for size in (64, 128):
            cfg = GenerationConfig(image_size=size, distractor_prob=0.0)
            per_class = {}
            type_hit, type_tot = defaultdict(int), defaultdict(int)
            p_all, r_all, f_all = [], [], []
            for c in CLASS_NAMES:
                fs = []
                for t in range(N):
                    s = render_sample(c, random.Random(10_000 + t), cfg)
                    res = match_primitives(ex.extract(s.image), s.primitives, size)
                    fs.append(res["f1"]); p_all.append(res["precision"]); r_all.append(res["recall"])
                    matched = {j for _, j, _ in res["matches"]}
                    for j, g in enumerate(s.primitives):
                        type_tot[g.canonical_type] += 1
                        type_hit[g.canonical_type] += j in matched
                per_class[c] = float(np.mean(fs))
                f_all += fs
            key = f"{ex.name}@{size}px"
            out[key] = {
                "precision": float(np.mean(p_all)), "recall": float(np.mean(r_all)), "f1": float(np.mean(f_all)),
                "f1_per_class": per_class,
                "recall_per_type": {t: type_hit[t] / type_tot[t] for t in type_tot},
            }
            r = out[key]
            lines += [f"## {key}: F1 {r['f1']:.3f} (precision {r['precision']:.3f}, recall {r['recall']:.3f})", "",
                      "| " + " | ".join(CLASS_NAMES) + " |", "|" + "---|" * len(CLASS_NAMES),
                      "| " + " | ".join(f"{per_class[c]:.2f}" for c in CLASS_NAMES) + " |", "",
                      "Recall by primitive type: " + ", ".join(f"{t} {v:.2f}" for t, v in sorted(r["recall_per_type"].items())), ""]
            print(key, round(r["f1"], 3), {t: round(v, 2) for t, v in r["recall_per_type"].items()}, flush=True)
    d = REPO_ROOT / "results" / "phase1" / "classical_extractor"
    write_json(d / "summary.json", out)
    (d / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
