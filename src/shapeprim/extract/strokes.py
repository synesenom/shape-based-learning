"""Fit primitives to sketch strokes (PLAN.md section 7, Phase 3b).

A QuickDraw drawing is a list of strokes (polylines). People draw circles
as one closed loop, a house as one loop or as several lines, and wheels
that do not quite close. The fitter works stroke by stroke:

1. A stroke whose end returns near its start (gap below ``close_frac``
   of its extent) is **closed**. Candidates are fitted to its filled
   outline -- an ellipse (``cv2.fitEllipse``), a quadrilateral (the
   4-vertex polygon approximation, or the min-area rectangle) and a
   triangle (the 3-vertex approximation, or ``cv2.minEnclosingTriangle``)
   -- and the one with the highest mask IoU wins. If even the best fits
   poorly (IoU below ``min_iou``) the loop is not one primitive, and it is
   decomposed into line segments instead.
2. An **open** stroke is simplified (Douglas-Peucker) and every segment
   long enough to matter becomes a ``line``.

At most ``max_primitives`` are kept, largest first, so a heavily scribbled
drawing does not become a 60-node graph.

There is no human primitive annotation for QuickDraw, so the fitter's F1
cannot be measured on it directly. It is measured instead on *synthetic*
sketches with known ground truth (``synthetic_strokes``: each primitive's
outline traced as a stroke with hand-like jitter and a random start
point), which is the proxy reported with every QuickDraw result.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from ..data.primitives import Primitive, polygon_pose

Stroke = np.ndarray  # (N, 2) float, image coordinates


def _mask_poly(pts: np.ndarray, size: int) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def _line(p0, p1, thickness: float, color) -> Primitive:
    d = np.asarray(p1, float) - np.asarray(p0, float)
    length = float(np.hypot(*d))
    n = np.array([-d[1], d[0]]) / max(length, 1e-6) * thickness / 2
    v = np.array([p0 + n, p1 + n, p1 - n, p0 - n], dtype=float)
    cx, cy, w, h, rot = polygon_pose("line", v)
    return Primitive(type="line", cx=cx, cy=cy, width=w, height=h, rotation=rot, color=color,
                     vertices=[tuple(map(float, p)) for p in v])


def _poly(ptype: str, v: np.ndarray, color) -> Primitive:
    cx, cy, w, h, rot = polygon_pose(ptype, v)
    return Primitive(type=ptype, cx=cx, cy=cy, width=w, height=h, rotation=rot, color=color,
                     vertices=[tuple(map(float, p)) for p in v])


def _apex_first(v: np.ndarray) -> np.ndarray:
    edge = [np.linalg.norm(v[i] - v[(i + 1) % 3]) for i in range(3)]
    apex = int(np.argmin([abs(edge[i - 1] - edge[i]) for i in range(3)]))
    return np.roll(v, -apex, axis=0)


class StrokeFitter:
    name = "strokefit"

    def __init__(
        self,
        close_frac: float = 0.4,
        min_iou: float = 0.6,
        segment_eps_frac: float = 0.06,
        min_segment_frac: float = 0.12,
        max_primitives: int = 16,
        line_thickness_frac: float = 0.04,
    ):
        self.close_frac = close_frac
        self.min_iou = min_iou
        self.segment_eps_frac = segment_eps_frac
        self.min_segment_frac = min_segment_frac
        self.max_primitives = max_primitives
        self.line_thickness_frac = line_thickness_frac

    def fit(self, strokes: Sequence[Stroke], image_size: int, color=(40, 40, 40)) -> List[Primitive]:
        strokes = [np.asarray(s, dtype=float).reshape(-1, 2) for s in strokes if len(s) >= 2]
        if not strokes:
            return []
        allpts = np.concatenate(strokes)
        drawing_extent = float(max(np.ptp(allpts[:, 0]), np.ptp(allpts[:, 1]), 1.0))
        thickness = max(1.5, self.line_thickness_frac * drawing_extent)
        prims: List[Tuple[float, Primitive]] = []
        for s in strokes:
            extent = float(max(np.ptp(s[:, 0]), np.ptp(s[:, 1]), 1e-6))
            gap = float(np.hypot(*(s[0] - s[-1])))
            if len(s) >= 4 and gap <= self.close_frac * extent and extent >= self.min_segment_frac * drawing_extent:
                fitted = self._fit_closed(s, image_size, color)
                if fitted is not None:
                    prims.append((fitted.width * fitted.height, fitted))
                    continue
                s = np.vstack([s, s[:1]])  # decompose the loop, including its closing edge
            for p in self._segments(s, drawing_extent, thickness, color):
                prims.append((p.width * p.height, p))
        prims.sort(key=lambda t: -t[0])
        return [p for _, p in prims[: self.max_primitives]]

    def _fit_closed(self, s: Stroke, size: int, color):
        target = _mask_poly(s, size)
        if target.sum() < 4:
            return None
        cands: List[Primitive] = []
        pts32 = s.astype(np.float32)
        if len(s) >= 5:
            (cx, cy), (d1, d2), ang = cv2.fitEllipse(pts32)
            if d1 > 0 and d2 > 0:
                cands.append(Primitive(type="ellipse", cx=float(cx), cy=float(cy), width=float(d1),
                                       height=float(d2), rotation=math.radians(ang), color=color))
        hull = cv2.convexHull(pts32).reshape(-1, 2)
        peri = cv2.arcLength(hull.reshape(-1, 1, 2), True)
        approx = {}
        for frac in (0.02, 0.04, 0.06, 0.1):
            a = cv2.approxPolyDP(hull.reshape(-1, 1, 2), frac * peri, True).reshape(-1, 2).astype(float)
            approx.setdefault(len(a), a)
        quad = approx.get(4, cv2.boxPoints(cv2.minAreaRect(pts32)).astype(float))
        cands.append(_poly("quadrilateral", quad, color))
        if 3 in approx:
            tri = approx[3]
        else:
            _, t = cv2.minEnclosingTriangle(pts32.reshape(-1, 1, 2))
            tri = t.reshape(-1, 2).astype(float)
        cands.append(_poly("triangle", _apex_first(tri), color))
        scored = []
        for c in cands:
            m = _mask_poly(np.array(c.boundary_points()), size)
            scored.append((_iou(m, target), c))
        best_iou, best = max(scored, key=lambda t: t[0])
        return best if best_iou >= self.min_iou else None

    def _segments(self, s: Stroke, drawing_extent: float, thickness: float, color) -> List[Primitive]:
        eps = self.segment_eps_frac * drawing_extent
        simp = cv2.approxPolyDP(s.astype(np.float32).reshape(-1, 1, 2), eps, False).reshape(-1, 2).astype(float)
        out = []
        for a, b in zip(simp[:-1], simp[1:]):
            if np.hypot(*(b - a)) >= self.min_segment_frac * drawing_extent:
                out.append(_line(a, b, thickness, color))
        return out


def synthetic_strokes(primitives: Sequence[Primitive], rng: random.Random, jitter: float = 0.02) -> List[Stroke]:
    """Trace ground-truth primitives as hand-like strokes (for measuring F1).

    Closed shapes become one loop starting at a random point with small
    positional jitter and a small closing gap; lines become one stroke
    along their long axis.
    """
    strokes: List[Stroke] = []
    for p in primitives:
        scale = max(p.width, p.height)
        j = jitter * scale
        if p.canonical_type == "line":
            v = np.array(p.polygon())
            a, b = (v[0] + v[3]) / 2, (v[1] + v[2]) / 2
            if np.hypot(*(v[1] - v[0])) < np.hypot(*(v[3] - v[0])):
                a, b = (v[0] + v[1]) / 2, (v[2] + v[3]) / 2
            pts = np.linspace(a, b, 8)
        else:
            outline = np.array(p.boundary_points(48))
            if p.canonical_type != "ellipse":
                # Densify polygon edges.
                dense = []
                for k in range(len(outline)):
                    q0, q1 = outline[k], outline[(k + 1) % len(outline)]
                    for t in np.linspace(0, 1, 8, endpoint=False):
                        dense.append(q0 + t * (q1 - q0))
                outline = np.array(dense)
            start = rng.randrange(len(outline))
            outline = np.roll(outline, -start, axis=0)
            keep = int(len(outline) * rng.uniform(0.94, 1.0))
            pts = outline[: max(keep, 4)]
        noise = np.array([[rng.gauss(0, j), rng.gauss(0, j)] for _ in range(len(pts))])
        strokes.append(np.asarray(pts, float) + noise)
    return strokes
