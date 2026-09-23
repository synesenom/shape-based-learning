#!/usr/bin/env python3
"""Train the learned primitive extractor and report its F1 per condition.

Usage:
    python scripts/train_learned_extractor.py --config configs/phase2/learned_extractor.yaml

Writes ``<weights>`` (gitignored) and, next to the config's ``report``
path, a JSON with the training log, the primitive-label budget, and
extractor precision/recall/F1 on every evaluation condition -- including
conditions the detector never trained on (the extrapolation angles).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shapeprim.conditions import _apply  # noqa: E402
from shapeprim.data.objects import CLASS_NAMES  # noqa: E402
from shapeprim.data.synth_dataset import GenerationConfig, SynthShapeDataset  # noqa: E402
from shapeprim.experiment import write_json  # noqa: E402
from shapeprim.extract.eval_match import match_primitives  # noqa: E402
from shapeprim.extract.learned import LearnedExtractor, train_detector  # noqa: E402
from shapeprim.extract import ClassicalExtractorV2  # noqa: E402


def f1_report(extractor, cfg: GenerationConfig, classes, n_per_class: int, seed: int) -> dict:
    ds = SynthShapeDataset(classes=classes, n_per_class=n_per_class, cfg=cfg, seed=seed, split="extractor_eval")
    per_class = {c: [] for c in classes}
    ps, rs, fs = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        res = match_primitives(extractor.extract(s.image), s.primitives, cfg.image_size)
        ps.append(res["precision"]); rs.append(res["recall"]); fs.append(res["f1"])
        per_class[s.label].append(res["f1"])
    mean = lambda v: sum(v) / len(v)
    return {
        "precision": mean(ps), "recall": mean(rs), "f1": mean(fs),
        "f1_per_class": {c: mean(v) for c, v in per_class.items()}, "n_samples": len(fs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(REPO_ROOT / args.config))
    torch.set_num_threads(cfg.get("threads", 4))
    classes = cfg.get("classes") or list(CLASS_NAMES)
    base = GenerationConfig.from_dict(cfg.get("generation", {}))
    train_cfg = _apply(base, cfg.get("train_overrides"))

    train = SynthShapeDataset(classes, cfg["n_train_per_class"], train_cfg, seed=cfg.get("seed", 0), split="extractor_train")
    val = SynthShapeDataset(classes, cfg["n_val_per_class"], train_cfg, seed=cfg.get("seed", 0), split="extractor_val")
    print(f"training detector on {len(train)} images ({cfg['n_train_per_class']}/class)", flush=True)
    model, log = train_detector(
        [train[i] for i in range(len(train))], [val[i] for i in range(len(val))],
        image_size=base.image_size, epochs=cfg["epochs"], lr=cfg.get("lr", 2e-3),
        width=cfg.get("width", 32), seed=cfg.get("seed", 0),
    )
    weights = REPO_ROOT / cfg["weights"]
    weights.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "primitive_labelled_images": len(train),
        "primitive_labels": sum(len(train[i].primitives) for i in range(len(train))),
        "train_generation": train_cfg.__dict__,
    }
    torch.save({"state_dict": model.state_dict(), "width": cfg.get("width", 32), "meta": meta}, weights)

    learned = LearnedExtractor(weights, threshold=cfg.get("threshold", 0.3))
    report = {"config": cfg, "train_log": log, "annotation_budget": meta, "conditions": {}}
    for name, overrides in (cfg.get("eval_conditions") or {"train": cfg.get("train_overrides")}).items():
        ecfg = _apply(base, overrides)
        report["conditions"][name] = {
            "learned": f1_report(learned, ecfg, classes, cfg.get("n_eval_per_class", 30), seed=12345),
            "classical_v2": f1_report(ClassicalExtractorV2(), ecfg, classes, cfg.get("n_eval_per_class", 30), seed=12345),
        }
        r = report["conditions"][name]
        print(f"  {name}: learned F1 {r['learned']['f1']:.3f}, classical_v2 F1 {r['classical_v2']['f1']:.3f}", flush=True)
    write_json(REPO_ROOT / cfg["report"], report)


if __name__ == "__main__":
    main()
