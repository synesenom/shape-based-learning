#!/usr/bin/env python3
"""Generate the Phase 1 synthetic dataset and a sample grid for visual check.

Usage:
    python scripts/generate_phase1_data.py --config configs/phase1/data.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shapeprim.data.synth_dataset import GenerationConfig, SynthShapeDataset, generate_dataset  # noqa: E402


def make_sample_grid(classes, cfg: GenerationConfig, seed: int, n_cols: int = 4) -> Image.Image:
    dataset = SynthShapeDataset(classes=classes, n_per_class=n_cols, cfg=cfg, seed=seed)
    size = cfg.image_size
    n_rows = len(classes)
    grid = Image.new("RGB", (n_cols * size, n_rows * size), cfg.background)
    for row, class_name in enumerate(classes):
        for col in range(n_cols):
            sample = dataset[row * n_cols + col]
            grid.paste(sample.image, (col * size, row * size))
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/phase1/data.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Override config's output_dir")
    parser.add_argument("--n-train-per-class", type=int, default=None)
    parser.add_argument("--n-test-per-class", type=int, default=None)
    parser.add_argument(
        "--sample-grid-out",
        type=str,
        default="results/phase1/sample_grid.png",
        help="Where to save the visual sample grid (one row per class).",
    )
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    classes = raw["classes"]
    n_train = args.n_train_per_class or raw["n_train_per_class"]
    n_test = args.n_test_per_class or raw["n_test_per_class"]
    seed = raw["seed"]
    output_dir = REPO_ROOT / (args.output_dir or raw["output_dir"])

    cfg = GenerationConfig.from_dict(raw)

    print(f"Generating train split: {n_train}/class x {len(classes)} classes -> {output_dir/'train'}")
    generate_dataset(output_dir, classes=classes, n_per_class=n_train, cfg=cfg, seed=seed, split="train")

    print(f"Generating test split: {n_test}/class x {len(classes)} classes -> {output_dir/'test'}")
    generate_dataset(output_dir, classes=classes, n_per_class=n_test, cfg=cfg, seed=seed + 1, split="test")

    grid_path = REPO_ROOT / args.sample_grid_out
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_sample_grid(classes, cfg, seed=seed, n_cols=4)
    grid.save(grid_path)
    print(f"Saved sample grid to {grid_path}")


if __name__ == "__main__":
    main()
