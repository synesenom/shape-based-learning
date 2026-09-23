#!/usr/bin/env python3
"""Download the first few MB of Quick, Draw! simplified strokes per class.

Only a prefix of each (large) ndjson file is fetched with an HTTP range
request; the loader drops the partial last line. ~3 MB is ~4000-8000
recognised drawings per class, far more than the experiments use.

Usage:
    python scripts/fetch_quickdraw.py [--bytes 3000000]
"""

import argparse
import urllib.request
from pathlib import Path

WORDS = ["bicycle", "car", "truck", "house", "cat", "tree", "fish", "snowman", "sun", "face"]
URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"
OUT = Path(__file__).resolve().parents[1] / "data" / "quickdraw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=3_000_000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for w in WORDS:
        dest = OUT / f"{w}.part"
        if dest.exists():
            print(f"{w}: present")
            continue
        req = urllib.request.Request(URL.format(w), headers={"Range": f"bytes=0-{args.bytes}"})
        with urllib.request.urlopen(req) as r:
            dest.write_bytes(r.read())
        print(f"{w}: {dest.stat().st_size} bytes")


if __name__ == "__main__":
    main()
