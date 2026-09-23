"""Convert a list of primitives into a translation/scale-invariant graph.

Nodes: one primitive each -- one-hot type, size, aspect ratio, rotation
(sin/cos), position, all normalized to the object's own bounding box.

Edges: fully connected and directed (both i->j and j->i, N*(N-1) edges
total) since several edge features are directional (dx/dy, above/below/
left/right). Features: relative position, distance, size ratio, relative
angle, and above/below/left/right/inside flags.

Normalizing every feature by the object's own bounding box (center and
half-diagonal) is what makes the graph translation- and scale-invariant:
shifting or uniformly scaling every primitive in an object leaves the
graph's features unchanged (see tests/test_graph_build.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..data.primitives import CANONICAL_TYPE, CANONICAL_TYPES, Primitive

# One-hot over the four canonical slots (circle/ellipse, triangle,
# rectangle/quadrilateral, line). Phase 1's four types map one-to-one onto
# these slots in their original order, so Phase 1 graphs are unchanged.
NUM_NODE_FEATURES = len(CANONICAL_TYPES) + 1 + 1 + 2 + 2  # type + size + aspect + rot(sin,cos) + pos(x,y)
NUM_TYPES = len(CANONICAL_TYPES)
# Column positions inside a node feature vector, for models that read
# individual features (the bag ablation reads size and aspect only).
SIZE_INDEX = NUM_TYPES
ASPECT_INDEX = NUM_TYPES + 1
NUM_EDGE_FEATURES = 11  # dx, dy, distance, size_ratio, sin(angle), cos(angle), above, below, left, right, inside

_TYPE_INDEX = {t: CANONICAL_TYPES.index(c) for t, c in CANONICAL_TYPE.items()}
FRAMES = ("bbox", "affine")
_EPS = 1e-6


@dataclass
class ShapeGraph:
    node_features: np.ndarray  # (N, NUM_NODE_FEATURES)
    node_types: List[str]
    edge_index: np.ndarray  # (2, E): row 0 = source node, row 1 = destination node
    edge_features: np.ndarray  # (E, NUM_EDGE_FEATURES)
    label: Optional[str] = None

    @property
    def num_nodes(self) -> int:
        return len(self.node_types)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


def _object_frame(primitives: List[Primitive]) -> Tuple[float, float, float]:
    """Center (cx, cy) and half-diagonal scale of the union bbox of all primitives."""
    x0s, y0s, x1s, y1s = zip(*(p.bbox() for p in primitives))
    x0, y0, x1, y1 = min(x0s), min(y0s), max(x1s), max(y1s)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    scale = 0.5 * math.hypot(x1 - x0, y1 - y0)
    return cx, cy, max(scale, _EPS)


def _node_feature_vector(p: Primitive, cx: float, cy: float, scale: float) -> np.ndarray:
    type_onehot = np.zeros(NUM_TYPES, dtype=np.float32)
    type_onehot[_TYPE_INDEX[p.type]] = 1.0

    size = math.sqrt(max(p.width * p.height, 0.0)) / scale
    aspect = p.width / max(p.height, _EPS)
    rot_sin, rot_cos = math.sin(p.rotation), math.cos(p.rotation)
    pos_x, pos_y = (p.cx - cx) / scale, (p.cy - cy) / scale

    return np.concatenate([type_onehot, [size, aspect, rot_sin, rot_cos, pos_x, pos_y]]).astype(np.float32)


def _edge_feature_vector(pi: Primitive, pj: Primitive, scale: float) -> np.ndarray:
    dx = (pj.cx - pi.cx) / scale
    dy = (pj.cy - pi.cy) / scale
    distance = math.hypot(dx, dy)

    size_i = math.sqrt(max(pi.width * pi.height, 0.0))
    size_j = math.sqrt(max(pj.width * pj.height, 0.0))
    size_ratio = math.log(max(size_j, _EPS) / max(size_i, _EPS))

    angle = math.atan2(dy, dx)
    angle_sin, angle_cos = math.sin(angle), math.cos(angle)

    above = 1.0 if pj.cy < pi.cy else 0.0
    below = 1.0 if pj.cy > pi.cy else 0.0
    left = 1.0 if pj.cx < pi.cx else 0.0
    right = 1.0 if pj.cx > pi.cx else 0.0
    inside = 1.0 if _is_inside(pj, pi) else 0.0

    return np.array(
        [dx, dy, distance, size_ratio, angle_sin, angle_cos, above, below, left, right, inside],
        dtype=np.float32,
    )


def _is_inside(inner: Primitive, outer: Primitive, containment_threshold: float = 0.8) -> bool:
    """True if `inner`'s bbox is mostly contained within `outer`'s bbox."""
    ix0, iy0, ix1, iy1 = inner.bbox()
    ox0, oy0, ox1, oy1 = outer.bbox()
    inter_x0, inter_y0 = max(ix0, ox0), max(iy0, oy0)
    inter_x1, inter_y1 = min(ix1, ox1), min(iy1, oy1)
    inter_area = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
    inner_area = max((ix1 - ix0) * (iy1 - iy0), _EPS)
    return (inter_area / inner_area) >= containment_threshold


def affine_canonical(primitives: List[Primitive]) -> List[Primitive]:
    """Whiten the object: map it so its second-moment matrix is the identity.

    The "built-in invariance" option of PLAN.md section 6. The object's
    covariance is taken over the union of its primitives' outlines (points
    weighted by the outline sampling, which is dense and uniform enough
    for a frame estimate) and the map is its symmetric inverse square
    root. If the object is transformed by any affine map A, the whitened
    result differs from the untransformed one by a rotation only
    (W' A = R W for some orthogonal R), so shear and anisotropic scale --
    and, to first order, perspective foreshortening -- are removed, while
    in-plane orientation is kept. Keeping orientation is deliberate: a
    fully rotation-canonical frame would map tree onto arrow_sign.
    """
    from ..data.transforms import all_points, transform_primitive

    pts = all_points(primitives)
    if len(pts) < 3:
        return primitives
    mu = pts.mean(axis=0)
    cov = np.cov((pts - mu).T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-6 * max(evals.max(), 1e-6))
    w = evecs @ np.diag(evals ** -0.5) @ evecs.T

    def whiten(p: np.ndarray) -> np.ndarray:
        return (p - mu) @ w.T

    return [transform_primitive(p, whiten) for p in primitives]


def build_graph(primitives: List[Primitive], label: Optional[str] = None, frame: str = "bbox") -> ShapeGraph:
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; known: {FRAMES}")
    if frame == "affine" and primitives:
        primitives = affine_canonical(primitives)
    if not primitives:
        return ShapeGraph(
            node_features=np.zeros((0, NUM_NODE_FEATURES), dtype=np.float32),
            node_types=[],
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_features=np.zeros((0, NUM_EDGE_FEATURES), dtype=np.float32),
            label=label,
        )

    cx, cy, scale = _object_frame(primitives)
    node_features = np.stack([_node_feature_vector(p, cx, cy, scale) for p in primitives])
    node_types = [p.type for p in primitives]

    n = len(primitives)
    src, dst, edge_feats = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            edge_feats.append(_edge_feature_vector(primitives[i], primitives[j], scale))

    edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    edge_features = np.stack(edge_feats) if edge_feats else np.zeros((0, NUM_EDGE_FEATURES), dtype=np.float32)

    return ShapeGraph(
        node_features=node_features,
        node_types=node_types,
        edge_index=edge_index,
        edge_features=edge_features,
        label=label,
    )
