"""Classical (OpenCV) primitive extractor.

Pipeline: threshold against the background -> connected components ->
polygon approximation + ellipse fit per component -> classify by vertex
count / roundness / aspect ratio.

Every primitive is rendered with a white outline (see data/primitives.py),
which is the same color as the background. That is what makes this
approach work at all: a part fully enclosed by another same-colored part
(e.g. a window inside a car body) is still separated from it by a ring of
background-colored pixels, so a plain foreground/background threshold
followed by connected-component labeling already isolates one component
per primitive -- no shape-decomposition algorithm is needed for primitives
that touch or overlap, only for primitives that are unioned with *no*
outline between them (which never happens here, since every primitive
draws its own complete outline).

Known limitation: at the acute angle where a thin part (a line) meets the
edge of another part it's attached to, the outline stroke can taper to a
sub-pixel gap in the rasterized image, letting the two same-colored fills
touch directly and merge into one blob. Template geometry mitigates this
(see the "person" and "bicycle" templates in data/objects.py) by keeping
such joins shallow, but classes built from several thin articulated parts
meeting a body at a shallow angle -- currently "bicycle" and "person" --
still measure well below the ~0.8+ F1 (vs. oracle ground truth, IoU>=0.5)
the other eight Phase 1 classes reach. This is a real gap between this
extractor and the oracle upper bound, not a bug to paper over; per
PLAN.md section 4, that gap is exactly what the oracle-vs-classical
comparison in stage-2 experiments is meant to surface.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..data.primitives import OUTLINE_WIDTH, Primitive
from .base import PrimitiveExtractor

Point = Tuple[float, float]

# The outline is rendered in the background color so it separates touching
# same-colored primitives (see module docstring), but that also means the
# foreground mask excludes the outline ring -- every detected blob is
# systematically smaller than the true primitive by about the outline's
# width (empirically ~2*outline_width-1 px total, e.g. a true 60px circle's
# fill-only contour measures ~55px at outline_width=3). Growing the mask to
# recover this was tried and rejected: dilating a *thin* shape (a line just
# a few px thick) rounds its ends enough to make it fit an ellipse well,
# misclassifying lines as circles. Instead, classification and position/
# rotation are computed from the true (un-grown) contour -- shrinking a
# shape symmetrically doesn't move its center or rotation -- and only
# width/height get an additive correction afterward, sized per-primitive
# (see `_size_recovery_pad`): primitives.py itself caps the outline width
# adaptively for thin shapes, so a flat correction sized for OUTLINE_WIDTH
# would over-pad a thin line far more than it was ever shrunk by.


def _size_recovery_pad(width: float, height: float) -> float:
    """primitives.py caps the outline width for thin shapes (so it can't
    swallow a thin line's fill), so a flat correction sized for the default
    OUTLINE_WIDTH overcorrects thin lines. Scale the pad down when the
    detected shape is itself thin, using the detected size as a stand-in
    for the true size (close enough: the shrink is only a few pixels)."""
    thinnest = min(width, height)
    full_pad = 2 * OUTLINE_WIDTH - 1
    return min(full_pad, max(1.0, thinnest / 2))


class ClassicalExtractor(PrimitiveExtractor):
    name = "classical"

    def __init__(
        self,
        background: Tuple[int, int, int] = (255, 255, 255),
        bg_diff_threshold: int = 60,
        min_area: float = 12.0,
        line_aspect_ratio: float = 3.0,
        ellipse_fit_tolerance: float = 0.18,
        min_solidity: float = 0.85,
    ):
        self.background = background
        self.bg_diff_threshold = bg_diff_threshold
        self.min_area = min_area
        self.line_aspect_ratio = line_aspect_ratio
        self.ellipse_fit_tolerance = ellipse_fit_tolerance
        self.min_solidity = min_solidity

    def extract(self, image: Image.Image, ground_truth: Optional[List[Primitive]] = None) -> List[Primitive]:
        arr = np.asarray(image.convert("RGB"))
        mask = self._foreground_mask(arr)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        primitives: List[Primitive] = []
        for label in range(1, n_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue
            comp_mask = np.where(labels == label, np.uint8(255), np.uint8(0))
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            prim = self._classify_contour(contour, arr, comp_mask)
            if prim is not None:
                primitives.append(prim)
        return primitives

    def _foreground_mask(self, arr: np.ndarray) -> np.ndarray:
        bg = np.array(self.background, dtype=np.int16)
        diff = np.abs(arr.astype(np.int16) - bg).sum(axis=2)
        return np.where(diff > self.bg_diff_threshold, np.uint8(255), np.uint8(0))

    def _sample_color(self, comp_mask: np.ndarray, arr: np.ndarray) -> Tuple[int, int, int]:
        ys, xs = np.nonzero(comp_mask)
        pixels = arr[ys, xs]
        colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
        mode_color = colors[np.argmax(counts)]
        return tuple(int(c) for c in mode_color)

    def _classify_contour(self, contour: np.ndarray, arr: np.ndarray, comp_mask: np.ndarray) -> Optional[Primitive]:
        area = cv2.contourArea(contour)
        if area <= 0:
            return None
        color = self._sample_color(comp_mask, arr)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        if solidity >= self.min_solidity and len(contour) >= 5:
            ellipse_fit = self._ellipse_fit_score(contour, area)
            if ellipse_fit is not None and ellipse_fit <= self.ellipse_fit_tolerance:
                x, y, w, h = cv2.boundingRect(contour)
                w, h = max(w, 1), max(h, 1)
                pad = _size_recovery_pad(w, h)
                return Primitive(
                    type="circle",
                    cx=x + w / 2,
                    cy=y + h / 2,
                    width=float(w + pad),
                    height=float(h + pad),
                    rotation=0.0,
                    color=color,
                )

        perimeter = cv2.arcLength(contour, True)
        approx = self._approx_polygon(contour, perimeter)

        if len(approx) == 3:
            cx, cy, w, h, rot = self._triangle_params(approx)
            pad = _size_recovery_pad(w, h)
            return Primitive(
                type="triangle", cx=cx, cy=cy, width=w + pad, height=h + pad, rotation=rot, color=color,
            )

        if len(approx) == 4:
            cx, cy, w, h, rot = self._rect_params(approx)
            pad = _size_recovery_pad(w, h)
            w, h = w + pad, h + pad
            ptype = "line" if _elongation(w, h) > self.line_aspect_ratio else "rectangle"
            return Primitive(type=ptype, cx=cx, cy=cy, width=w, height=h, rotation=rot, color=color)

        # Fallback: neither round nor a clean triangle/quad. Rare on clean
        # synthetic drawings; best-effort as a min-area rectangle so the
        # extractor always returns *something* for every foreground blob.
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        cx, cy, w, h, rot = self._rect_params(box)
        pad = _size_recovery_pad(w, h)
        w, h = w + pad, h + pad
        ptype = "line" if _elongation(w, h) > self.line_aspect_ratio else "rectangle"
        return Primitive(type=ptype, cx=cx, cy=cy, width=w, height=h, rotation=rot, color=color)

    def _ellipse_fit_score(self, contour: np.ndarray, area: float) -> Optional[float]:
        (_, _), (d1, d2), _ = cv2.fitEllipse(contour)
        ellipse_area = math.pi * (d1 / 2) * (d2 / 2)
        if ellipse_area <= 0:
            return None
        return abs(area - ellipse_area) / ellipse_area

    def _approx_polygon(self, contour: np.ndarray, perimeter: float) -> np.ndarray:
        # A triangle can spuriously approximate to 4 vertices at a small
        # epsilon (two near-duplicate points at a slightly rounded apex)
        # before a larger epsilon cleanly collapses it to 3. Scan every
        # candidate epsilon and prefer any 3-vertex result over a 4-vertex
        # one, instead of stopping at the first epsilon that hits either.
        fracs = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1)
        by_len = {}
        for frac in fracs:
            approx = cv2.approxPolyDP(contour, frac * perimeter, True).reshape(-1, 2)
            by_len.setdefault(len(approx), approx)
        for n in (3, 4):
            if n in by_len:
                return by_len[n]
        return cv2.approxPolyDP(contour, fracs[-1] * perimeter, True).reshape(-1, 2)

    def _triangle_params(self, verts: np.ndarray) -> Tuple[float, float, float, float, float]:
        v = verts.astype(float)
        edge_len = [np.linalg.norm(v[i] - v[(i + 1) % 3]) for i in range(3)]
        # The apex is the vertex whose two adjacent edges are most nearly
        # equal in length (our triangles are isosceles: apex + symmetric base).
        apex_idx = int(np.argmin([abs(edge_len[i - 1] - edge_len[i]) for i in range(3)]))
        apex = v[apex_idx]
        base = [v[j] for j in range(3) if j != apex_idx]
        base_mid = (base[0] + base[1]) / 2
        width = float(np.linalg.norm(base[0] - base[1]))

        d = apex - base_mid
        height = float(np.linalg.norm(d))
        if height < 1e-6:
            return float(base_mid[0]), float(base_mid[1]), max(width, 1.0), 1.0, 0.0
        rotation = math.atan2(d[0], -d[1])
        cx = base_mid[0] + (height / 2) * math.sin(rotation)
        cy = base_mid[1] - (height / 2) * math.cos(rotation)
        return float(cx), float(cy), max(width, 1.0), max(height, 1.0), rotation

    def _rect_params(self, verts: np.ndarray) -> Tuple[float, float, float, float, float]:
        v = verts.astype(float)
        cx, cy = v.mean(axis=0)
        edge0 = v[1] - v[0]
        edge1 = v[2] - v[1]
        width = float(np.linalg.norm(edge0))
        height = float(np.linalg.norm(edge1))
        rotation = math.atan2(edge0[1], edge0[0]) if width > 0 else 0.0
        return float(cx), float(cy), max(width, 1.0), max(height, 1.0), rotation


def _elongation(width: float, height: float) -> float:
    lo = max(min(width, height), 1e-6)
    return max(width, height) / lo
