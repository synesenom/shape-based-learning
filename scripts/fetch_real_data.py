#!/usr/bin/env python3
"""Fetch everything Phase 3c needs (idempotent: present files are skipped).

- COCO 2017 instance annotations (annotations_trainval2017.zip, ~250 MB;
  only the two instances JSONs are extracted), then the crops
  (scripts/fetch_coco_crops.py);
- Geirhos et al.'s model-vs-human test sets: sketch, stylized, edge,
  silhouette and cue-conflict (GitHub release v0.1).

Usage:
    python scripts/fetch_real_data.py
"""

import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COCO_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
MVH = "https://github.com/bethgelab/model-vs-human/releases/download/v0.1/{}.tar.gz"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"downloaded {dest}", flush=True)


def main() -> None:
    ann = DATA / "coco" / "annotations"
    if not (ann / "instances_train2017.json").exists():
        download(COCO_ZIP, DATA / "coco" / "annotations.zip")
        with zipfile.ZipFile(DATA / "coco" / "annotations.zip") as z:
            for name in ("annotations/instances_train2017.json", "annotations/instances_val2017.json"):
                z.extract(name, DATA / "coco")
    if not (DATA / "coco" / "crops" / "index.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_coco_crops.py")], check=True)
    for name in ("sketch", "stylized", "edge", "silhouette", "cue-conflict"):
        if (DATA / "mvh" / name).exists():
            continue
        tgz = DATA / "mvh" / f"{name}.tar.gz"
        download(MVH.format(name), tgz)
        with tarfile.open(tgz) as t:
            t.extractall(DATA / "mvh")
    print("real-image data ready")


if __name__ == "__main__":
    main()
