"""Match predicted primitives to ground truth by rasterized IoU, and score
precision/recall/F1. Used to check an extractor's quality against oracle
ground truth (PLAN.md section 4: "primitive precision/recall for
non-oracle extractors").
"""

from __future__ import annotations

from typing import List, Tuple, TypedDict

import cv2
import numpy as np

from ..data.primitives import Primitive


def primitive_mask(p: Primitive, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    if p.type == "circle":
        center = (int(round(p.cx)), int(round(p.cy)))
        axes = (max(1, int(round(p.width / 2))), max(1, int(round(p.height / 2))))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    else:
        pts = np.array(p.polygon(), dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask > 0


def iou(p1: Primitive, p2: Primitive, size: int) -> float:
    m1, m2 = primitive_mask(p1, size), primitive_mask(p2, size)
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(m1, m2).sum()
    return float(inter) / float(union)


class MatchResult(TypedDict):
    precision: float
    recall: float
    f1: float
    n_pred: int
    n_gt: int
    matches: List[Tuple[int, int, float]]


def match_primitives(
    pred: List[Primitive],
    gt: List[Primitive],
    size: int,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy same-type IoU matching (highest IoU first, each side used once)."""
    candidates = []
    for i, pp in enumerate(pred):
        for j, gp in enumerate(gt):
            if pp.type != gp.type:
                continue
            val = iou(pp, gp, size)
            if val >= iou_threshold:
                candidates.append((val, i, j))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_pred, matched_gt = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for val, i, j in candidates:
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        matches.append((i, j, val))

    precision = len(matches) / len(pred) if pred else 1.0
    recall = len(matches) / len(gt) if gt else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": len(pred),
        "n_gt": len(gt),
        "matches": matches,
    }
