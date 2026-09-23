"""Phase 3b: QuickDraw sketches as a data source (PLAN.md section 7).

Real people drew these, so they check that results do not depend on the
hand-designed classes. A drawing's strokes are rendered for the pixel
models and fitted to primitives (``extract/strokes.py``) for the graph
models; the fitted primitives play the role ground truth plays for
synthetic data -- there is no human primitive annotation for QuickDraw,
and results say so.

Data: the "simplified" ndjson files of the Quick, Draw! dataset (strokes
in a 256-pixel box), of which only the first few megabytes per class are
needed (``scripts/fetch_quickdraw.py``). Drawings the game did not
recognise are dropped.

Splits are fixed partitions of each class's pool (shuffled once with a
fixed seed): the first 200 drawings are the test region, the next 200 the
validation region, the rest the training pool. The test set ignores the
run seed (as for synthetic data, the spread across seeds is over training
draws); train and validation draws are sampled per seed.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..extract.strokes import StrokeFitter

REPO_ROOT = Path(__file__).resolve().parents[3]
QUICKDRAW_DIR = REPO_ROOT / "data" / "quickdraw"
# Experiment class names -> QuickDraw words. Synthetic names are used
# throughout so a synthetic-trained model can be scored on QuickDraw.
QUICKDRAW_WORD = {"cat_face": "cat"}
TEST_REGION, VAL_REGION = 200, 200
OBJECT_FRACTION = 0.72  # drawing's longer side as a fraction of the canvas


@lru_cache(maxsize=None)
def load_pool(word: str) -> Tuple[Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...], ...]:
    path = QUICKDRAW_DIR / f"{word}.part"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/fetch_quickdraw.py first")
    lines = path.read_text().split("\n")[:-1]  # the last line may be cut mid-record
    pool = []
    for line in lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("recognized"):
            pool.append(tuple((tuple(xs), tuple(ys)) for xs, ys in d["drawing"]))
    random.Random(0).shuffle(pool)
    return tuple(pool)


def place_strokes(drawing, size: int) -> List[np.ndarray]:
    strokes = [np.array(list(zip(xs, ys)), dtype=float) for xs, ys in drawing]
    pts = np.concatenate(strokes)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    scale = OBJECT_FRACTION * size / max(float((hi - lo).max()), 1.0)
    offset = size / 2 - (lo + hi) / 2 * scale
    return [s * scale + offset for s in strokes]


def render_strokes(strokes: Sequence[np.ndarray], size: int, color=(40, 40, 40), background=(255, 255, 255)) -> Image.Image:
    img = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(img)
    width = max(1, round(size / 32))
    for s in strokes:
        pts = [tuple(p) for p in s]
        if len(pts) == 1:
            pts = pts * 2
        draw.line(pts, fill=color, width=width, joint="curve")
    return img


class QuickDrawDataset(Sequence):
    """Same interface as ``SynthShapeDataset``: index -> Sample."""

    def __init__(self, classes: Sequence[str], n_per_class: int, cfg, seed: int = 0, split: str = "train"):
        self.classes = list(classes)
        self.n_per_class = n_per_class
        self.cfg = cfg
        self.seed = seed
        self.split = split
        self.fitter = StrokeFitter()
        self._index: List[Tuple[str, int]] = []
        for c in self.classes:
            pool = load_pool(QUICKDRAW_WORD.get(c, c))
            if split == "test":
                region = list(range(0, TEST_REGION))
                chosen = region[:n_per_class]
            elif split == "val":
                region = list(range(TEST_REGION, TEST_REGION + VAL_REGION))
                chosen = random.Random(f"{seed}|val|{c}").sample(region, n_per_class)
            else:
                region = list(range(TEST_REGION + VAL_REGION, len(pool)))
                chosen = random.Random(f"{seed}|{split}|{c}").sample(region, n_per_class)
            if len(chosen) < n_per_class:
                raise ValueError(f"QuickDraw {c}: only {len(chosen)} drawings in the {split} region")
            self._index += [(c, i) for i in chosen]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int):
        from .synth_dataset import Sample

        c, i = self._index[index]
        drawing = load_pool(QUICKDRAW_WORD.get(c, c))[i]
        size = self.cfg.image_size
        strokes = place_strokes(drawing, size)
        return Sample(
            image=render_strokes(strokes, size),
            primitives=self.fitter.fit(strokes, size),
            label=c,
        )
