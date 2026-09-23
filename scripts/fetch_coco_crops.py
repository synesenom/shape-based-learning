#!/usr/bin/env python3
"""Build the Phase 3c real-photo set: COCO crops of bicycles, cars, trucks, cats.

Reads the COCO 2017 instance annotations (data/coco/annotations/, from
annotations_trainval2017.zip) and, per class, takes non-crowd instances
whose box is at least 96 px on its shorter side. Each crop is the box
grown by 10% and squared, fetched from images.cocodataset.org and saved at
128x128. Training/validation crops come from train2017 and test crops
from val2017, so no photo contributes to both sides.

Writes data/coco/crops/<split>/<class>/<ann_id>.png and data/coco/crops/index.json.

Usage:
    python scripts/fetch_coco_crops.py [--train 130] [--test 60]
"""

import argparse
import io
import json
import random
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data" / "coco"
CLASSES = {2: "bicycle", 3: "car", 8: "truck", 17: "cat_face"}  # experiment names
MIN_SIDE = 96


def candidates(split: str):
    d = json.load(open(ROOT / "annotations" / f"instances_{split}.json"))
    images = {im["id"]: im for im in d["images"]}
    by_class = {c: [] for c in CLASSES.values()}
    for a in d["annotations"]:
        if a["category_id"] in CLASSES and not a["iscrowd"] and min(a["bbox"][2:]) >= MIN_SIDE:
            by_class[CLASSES[a["category_id"]]].append((a, images[a["image_id"]]))
    for v in by_class.values():
        v.sort(key=lambda t: t[0]["id"])
        random.Random(0).shuffle(v)
    return by_class


def crop(a, im, out: Path) -> bool:
    if out.exists():
        return True
    try:
        with urllib.request.urlopen(im["coco_url"], timeout=30) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception as e:  # network hiccup: skip this instance
        print("  skip", a["id"], e)
        return False
    x, y, w, h = a["bbox"]
    side = max(w, h) * 1.1
    cx, cy = x + w / 2, y + h / 2
    box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    # Where the square leaves the photo, PIL pads with black; that applies to
    # every class alike, so it is not a class cue.
    img.crop(tuple(int(round(v)) for v in box)).resize((128, 128), Image.BICUBIC).save(out)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=130, help="train+val crops per class (train2017)")
    ap.add_argument("--test", type=int, default=60, help="test crops per class (val2017)")
    args = ap.parse_args()
    index = {"train": {}, "test": {}}
    for split, coco_split, n in (("train", "train2017", args.train), ("test", "val2017", args.test)):
        for cls, items in candidates(coco_split).items():
            out_dir = ROOT / "crops" / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            kept = []
            for a, im in items:
                if len(kept) >= n:
                    break
                path = out_dir / f"{a['id']}.png"
                if crop(a, im, path):
                    kept.append(str(path.relative_to(ROOT)))
            index[split][cls] = kept
            print(f"{split} {cls}: {len(kept)}", flush=True)
    (ROOT / "crops" / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
