"""Phase 2 viewpoint transforms: rotation, shear, squash and perspective.

PLAN.md section 6 wants drawings "seen from different angles": a bicycle
at an angle is two ellipses and a skewed triangle. The transforms here act
on the *primitives*, not on the rendered pixels -- every primitive's
geometry is pushed through the same planar map and then drawn -- so the
ground truth stays exact by construction, exactly as in Phase 1:

- a circle becomes the ellipse fitted to its transformed outline (exact
  for affine maps, and for a perspective map as long as the circle stays
  in front of the camera, since a projected circle is a conic);
- a rectangle becomes a general quadrilateral with its exact vertices;
- a triangle stays a triangle (projective maps preserve lines);
- a line stays a thin quadrilateral, typed ``line``.

The map is a 3x3 homography in object-centred coordinates whose unit is
the object's size, composed as roll . perspective . shear . squash:

- **perspective** ("viewing angle"): the drawing's plane is tilted by
  ``view_angle`` about an in-plane axis with uniformly random direction,
  and seen through a pinhole camera ``camera_distance`` object-sizes away.
  0 degrees is a frontal view; 70 degrees is steep.
- **rotation**: in-plane roll. Capped by the caller: past 90 degrees
  ``tree`` becomes ``arrow_sign`` (they are exact vertical flips).
- **shear** and **squash**: the affine part, a horizontal shear and an
  anisotropic stretch along a random direction.

After the map, the object is rescaled and re-centred to the size and
position sampled for it before the transform, so viewpoint is not
confounded with object size or position (both are Phase 1's shift axes).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import cv2
import numpy as np

from .primitives import CANONICAL_TYPE, Primitive, polygon_pose

PointMap = Callable[[np.ndarray], np.ndarray]


@dataclass
class ViewSample:
    """The transform drawn for one image, recorded for analysis."""

    view_angle: float = 0.0  # degrees
    azimuth: float = 0.0  # degrees, direction of the tilt axis
    rotation: float = 0.0  # degrees
    shear: float = 0.0
    squash: float = 1.0
    squash_axis: float = 0.0  # degrees

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _signed(rng: random.Random, lo: float, hi: float) -> float:
    """Magnitude uniform in [lo, hi], random sign."""
    mag = rng.uniform(lo, hi)
    return mag if rng.random() < 0.5 else -mag


def sample_view(
    rng: random.Random,
    rotation_range: Sequence[float] = (0.0, 0.0),
    shear_range: Sequence[float] = (0.0, 0.0),
    squash_range: Sequence[float] = (1.0, 1.0),
    view_angle_range: Sequence[float] = (0.0, 0.0),
) -> ViewSample:
    """Draw one viewpoint. Ranges are magnitudes; signs/axes are random."""
    return ViewSample(
        view_angle=rng.uniform(*view_angle_range),
        azimuth=rng.uniform(0.0, 360.0),
        rotation=_signed(rng, *rotation_range),
        shear=_signed(rng, *shear_range),
        squash=rng.uniform(*squash_range),
        squash_axis=rng.uniform(0.0, 180.0),
    )


def _rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def view_homography(view: ViewSample, camera_distance: float = 2.5) -> np.ndarray:
    """3x3 homography (object-centred, unit = object size) for ``view``."""
    # Affine part: squash along a random axis, then shear, then roll.
    q = _rot2(math.radians(view.squash_axis))
    squash = q @ np.diag([view.squash, 1.0 / view.squash]) @ q.T
    shear = np.array([[1.0, view.shear], [0.0, 1.0]])
    affine = shear @ squash

    # Perspective: tilt the plane about the in-plane axis u by the viewing
    # angle (Rodrigues), then project with a pinhole camera at distance d.
    theta = math.radians(view.view_angle)
    phi = math.radians(view.azimuth)
    u = np.array([math.cos(phi), math.sin(phi), 0.0])
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    r3 = np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)
    d = camera_distance
    persp = np.array(
        [
            [r3[0, 0], r3[0, 1], 0.0],
            [r3[1, 0], r3[1, 1], 0.0],
            [r3[2, 0] / d, r3[2, 1] / d, 1.0],
        ]
    )

    roll = np.eye(3)
    roll[:2, :2] = _rot2(math.radians(view.rotation))

    a3 = np.eye(3)
    a3[:2, :2] = affine
    return roll @ persp @ a3


def apply_homography(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    w = homog[:, 2:3]
    if np.any(w <= 1e-6):
        raise ValueError("homography maps a point behind the camera; lower view_angle or raise camera_distance")
    return homog[:, :2] / w


def transform_primitive(p: Primitive, point_map: PointMap) -> Primitive:
    """Push one primitive through ``point_map`` (N x 2 -> N x 2).

    Output vocabulary is Phase 2's: ellipse, triangle, quadrilateral, line.
    """
    ctype = CANONICAL_TYPE[p.type]
    if ctype == "ellipse":
        pts = point_map(np.array(p.boundary_points(), dtype=np.float64))
        (cx, cy), (d1, d2), angle = cv2.fitEllipse(pts.astype(np.float32))
        return Primitive(
            type="ellipse", cx=float(cx), cy=float(cy), width=max(float(d1), 1e-3), height=max(float(d2), 1e-3),
            rotation=math.radians(angle), color=p.color, is_distractor=p.is_distractor,
        )
    verts = point_map(np.array(p.polygon(), dtype=np.float64))
    out_type = {"triangle": "triangle", "quadrilateral": "quadrilateral", "line": "line"}[ctype]
    cx, cy, w, h, rot = polygon_pose(out_type, verts)
    return Primitive(
        type=out_type, cx=cx, cy=cy, width=w, height=h, rotation=rot,
        color=p.color, is_distractor=p.is_distractor,
        vertices=[(float(x), float(y)) for x, y in verts],
    )


def all_points(primitives: Sequence[Primitive]) -> np.ndarray:
    return np.array([pt for p in primitives for pt in p.boundary_points()], dtype=np.float64)


def apply_view(
    primitives: List[Primitive],
    view: ViewSample,
    center: Tuple[float, float],
    obj_size: float,
    image_size: int,
    camera_distance: float = 2.5,
) -> List[Primitive]:
    """Transform an object's primitives and restore its sampled size/position.

    The object is mapped into object-centred units, transformed, then
    scaled so its bounding box's longer side equals ``obj_size`` again and
    centred on ``center`` (clamped so the whole object stays on canvas).
    """
    h = view_homography(view, camera_distance)
    cx0, cy0 = center

    def to_view(pts: np.ndarray) -> np.ndarray:
        q = (pts - np.array([cx0, cy0])) / obj_size
        return apply_homography(h, q)

    viewed = [transform_primitive(p, to_view) for p in primitives]
    pts = all_points(viewed)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))
    scale = obj_size / span
    mid = (lo + hi) / 2
    half = (hi - lo) * scale / 2
    tx = min(max(cx0, half[0]), image_size - half[0])
    ty = min(max(cy0, half[1]), image_size - half[1])

    def place(q: np.ndarray) -> np.ndarray:
        return (q - mid) * scale + np.array([tx, ty])

    return [transform_primitive(p, place) for p in viewed]
