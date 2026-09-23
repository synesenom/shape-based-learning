#!/usr/bin/env python3
"""Fill the SAM-fit cache for every Phase 3c image (COCO crops + model-vs-human).

Segmentation is the expensive step (seconds per image on CPU); the
experiments then read primitives from data/cache/samfit/. Images are fed
at the experiment's resolution, since the cache is keyed by pixels.

Usage:
    python scripts/precompute_sam.py [--image-size 128] [--threads 2]
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from shapeprim.data.synth_dataset import GenerationConfig, make_source  # noqa: E402
from shapeprim.extract.sam_fit import SamFitExtractor  # noqa: E402

CLASSES = ["bicycle", "car", "truck", "cat_face"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--sources", nargs="+", default=["mvh_edge", "mvh_silhouette", "mvh_cue", "mvh_sketch", "mvh_stylized", "coco"])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ex = SamFitExtractor()
    for src in args.sources:
        cfg = GenerationConfig(image_size=args.image_size, source=src)
        if src == "coco":
            splits = [("test", 60), ("val", 30), ("train", 100)]
        else:
            splits = [("test", 0)]
        for split, n in splits:
            ds = make_source(CLASSES, n, cfg, seed=0, split=split)
            t0 = time.time()
            counts = []
            for i in range(len(ds)):
                counts.append(len(ex.extract(ds[i].image)))
            print(f"{src}/{split}: {len(ds)} images, {sum(counts) / max(len(counts), 1):.1f} primitives/image, "
                  f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
