"""SAM masks -> primitives (PLAN.md section 7, Phase 3c).

Pipeline per image: segment with SAM, prompted by a regular grid of
points; keep confident, non-duplicate masks of plausible size; fit each
mask with an ellipse, a minimum-area rectangle and a polygon
approximation (triangle when it has three corners), and keep the fit with
the best mask IoU -- exactly the recipe PLAN.md gives for ``sam_fit.py``.

Model choice, forced by compute: the original SAM ViT-B took ~130 s per
image on this project's 4-core CPU with the automatic mask generator. The
extractor uses **SlimSAM-77** (Chen et al. 2024; a structurally pruned and
distilled SAM ViT-B, loaded through ``transformers``), which keeps SAM's
prompt interface at a fraction of the cost. It is still SAM-family
segmentation, but not the full model; results are labelled accordingly.

Masks are cached on disk by a digest of the image pixels, so each photo is
segmented once however many runs, seeds and splits use it
(``scripts/precompute_sam.py`` fills the cache ahead of the experiments).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from ..data.primitives import Primitive, polygon_pose
from .base import PrimitiveExtractor

MODEL_ID = "Zigeng/SlimSAM-uniform-77"
REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "samfit"


def _digest(image: Image.Image) -> str:
    arr = np.asarray(image.convert("RGB"))
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()[:20]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def fit_mask(mask: np.ndarray, color=(40, 40, 40), line_aspect: float = 4.0) -> Optional[Primitive]:
    """Best of ellipse / min-area rectangle / triangle for one binary mask."""
    m8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if len(c) < 5 or cv2.contourArea(c) < 4:
        return None
    h, w = mask.shape
    cands: List[Primitive] = []
    (ex, ey), (d1, d2), ang = cv2.fitEllipse(c)
    if d1 > 0 and d2 > 0:
        cands.append(Primitive(type="ellipse", cx=float(ex), cy=float(ey), width=float(d1), height=float(d2),
                               rotation=math.radians(ang), color=color))
    box = cv2.boxPoints(cv2.minAreaRect(c)).astype(float)
    cx, cy, bw, bh, rot = polygon_pose("quadrilateral", box)
    qtype = "line" if max(bw, bh) / max(min(bw, bh), 1e-6) > line_aspect else "quadrilateral"
    cands.append(Primitive(type=qtype, cx=cx, cy=cy, width=bw, height=bh, rotation=rot, color=color,
                           vertices=[tuple(p) for p in box]))
    peri = cv2.arcLength(c, True)
    tri = cv2.approxPolyDP(cv2.convexHull(c), 0.08 * peri, True).reshape(-1, 2).astype(float)
    if len(tri) == 3:
        edge = [np.linalg.norm(tri[i] - tri[(i + 1) % 3]) for i in range(3)]
        apex = int(np.argmin([abs(edge[i - 1] - edge[i]) for i in range(3)]))
        tri = np.roll(tri, -apex, axis=0)
        tcx, tcy, tw, th, trot = polygon_pose("triangle", tri)
        cands.append(Primitive(type="triangle", cx=tcx, cy=tcy, width=tw, height=th, rotation=trot, color=color,
                               vertices=[tuple(p) for p in tri]))
    best, best_iou = None, -1.0
    for p in cands:
        pm = np.zeros((h, w), np.uint8)
        cv2.fillPoly(pm, [np.round(np.array(p.boundary_points())).astype(np.int32).reshape(-1, 1, 2)], 1)
        v = _iou(pm.astype(bool), mask)
        if v > best_iou:
            best, best_iou = p, v
    return best


class SamFitExtractor(PrimitiveExtractor):
    name = "samfit"

    def __init__(
        self,
        points_per_side: int = 6,
        min_score: float = 0.8,
        min_area_frac: float = 0.004,
        max_area_frac: float = 0.6,
        dedupe_iou: float = 0.7,
        max_primitives: int = 12,
        seg_size: int = 256,
        cache_dir: Path = CACHE_DIR,
    ):
        self.points_per_side = points_per_side
        self.min_score = min_score
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.dedupe_iou = dedupe_iou
        self.max_primitives = max_primitives
        self.seg_size = seg_size
        self.cache_dir = Path(cache_dir)
        self._model = None
        self._processor = None

    # -- segmentation -------------------------------------------------------
    def _load(self):
        if self._model is None:
            os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data" / "hf"))
            from transformers import SamModel, SamProcessor

            self._model = SamModel.from_pretrained(MODEL_ID).eval()
            self._processor = SamProcessor.from_pretrained(MODEL_ID)

    def masks(self, image: Image.Image) -> List[np.ndarray]:
        """Confident, de-duplicated masks at ``seg_size`` resolution."""
        import torch

        self._load()
        img = image.convert("RGB").resize((self.seg_size, self.seg_size), Image.BILINEAR)
        s = self.seg_size
        grid = np.linspace(s / (2 * self.points_per_side), s - s / (2 * self.points_per_side), self.points_per_side)
        points = [[[[float(x), float(y)]] for y in grid for x in grid]]
        with torch.no_grad():
            inp = self._processor(img, input_points=points, return_tensors="pt")
            emb = self._model.get_image_embeddings(inp["pixel_values"])
            out = self._model(image_embeddings=emb, input_points=inp["input_points"], multimask_output=True)
            masks = self._processor.image_processor.post_process_masks(
                out.pred_masks, inp["original_sizes"], inp["reshaped_input_sizes"]
            )[0]  # (P, 3, H, W) bool
        scores = out.iou_scores[0].numpy()  # (P, 3)
        cands = []
        for pi in range(masks.shape[0]):
            k = int(np.argmax(scores[pi]))
            if scores[pi, k] < self.min_score:
                continue
            m = masks[pi, k].numpy().astype(bool)
            frac = m.mean()
            if not (self.min_area_frac <= frac <= self.max_area_frac):
                continue
            cands.append((float(scores[pi, k]), m))
        cands.sort(key=lambda t: -t[0])
        kept: List[np.ndarray] = []
        for _, m in cands:
            if all(_iou(m, k) < self.dedupe_iou for k in kept):
                kept.append(m)
        return kept

    # -- interface ----------------------------------------------------------
    def extract(self, image: Image.Image, ground_truth=None) -> List[Primitive]:
        key = _digest(image)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                return [Primitive.from_dict(d) for d in json.loads(path.read_text())]
            except (json.JSONDecodeError, OSError, KeyError):
                pass
        scale = image.size[0] / self.seg_size
        prims = []
        for m in self.masks(image):
            p = fit_mask(m)
            if p is None:
                continue
            # Back to the input image's pixel frame.
            p = Primitive(
                type=p.type, cx=p.cx * scale, cy=p.cy * scale, width=p.width * scale, height=p.height * scale,
                rotation=p.rotation, color=p.color,
                vertices=[(x * scale, y * scale) for x, y in p.vertices] if p.vertices else None,
            )
            prims.append((p.width * p.height, p))
        prims.sort(key=lambda t: -t[0])
        out = [p for _, p in prims[: self.max_primitives]]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([p.to_dict() for p in out]))
        return out
