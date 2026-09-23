"""Geometric primitive dataclass, polygon math, and rendering.

Phase 1 vocabulary: circle, triangle, rectangle, line. Every primitive is
described by a center, a width/height (in pixels), and a rotation (radians,
clockwise in image coordinates since y grows downward).

Phase 2 (viewpoint transforms) relaxes the circle to an ``ellipse`` (a
circle is the special case width == height) and the rectangle to a general
``quadrilateral``. A transformed polygon is no longer determined by
(center, width, height, rotation), so polygons may carry their exact
``vertices``; the pose parameters are then a summary of those vertices
(see ``polygon_pose``) and the vertices are what gets drawn and scored.

``CANONICAL_TYPE`` folds the two vocabularies onto four slots
(circle -> ellipse, rectangle -> quadrilateral). Graph features and
extractor matching use it, so Phase 1 data produces exactly the one-hot
encoding it always did while Phase 2 types land in the same slots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import ImageDraw

# The Phase 1 rendering vocabulary. Order matters: distractor sampling
# draws from it, so extending it would change every Phase 1 image.
PRIMITIVE_TYPES = ("circle", "triangle", "rectangle", "line")
ALL_PRIMITIVE_TYPES = PRIMITIVE_TYPES + ("ellipse", "quadrilateral")

# Four canonical slots shared by both phases (graph one-hot, matching).
CANONICAL_TYPES = ("ellipse", "triangle", "quadrilateral", "line")
CANONICAL_TYPE = {
    "circle": "ellipse",
    "ellipse": "ellipse",
    "triangle": "triangle",
    "rectangle": "quadrilateral",
    "quadrilateral": "quadrilateral",
    "line": "line",
}

ELLIPSE_POLY_POINTS = 64

Color = Tuple[int, int, int]

# Every primitive is stroked with this outline on top of its fill. Without
# it, a part fully contained in another same-colored part (e.g. a car's
# window inside its body) would leave no pixel evidence at all -- the fill
# is identical to what's underneath, so there is no edge for a classical
# contour-based extractor to find. The outline is a fixed rendering
# convention, not semantic data, so it isn't a dataclass field.
OUTLINE_COLOR: Color = (255, 255, 255)
OUTLINE_WIDTH = 3


@dataclass
class Primitive:
    type: str
    cx: float
    cy: float
    width: float
    height: float
    rotation: float = 0.0
    color: Color = (40, 40, 40)
    is_distractor: bool = False
    # Exact polygon vertices, for polygons whose shape is not determined by
    # the pose parameters (a rectangle under perspective). For an ellipse
    # this stays None: its pose parameters describe it exactly.
    vertices: Optional[List[Tuple[float, float]]] = None

    @property
    def canonical_type(self) -> str:
        return CANONICAL_TYPE[self.type]

    def __post_init__(self) -> None:
        if self.type not in ALL_PRIMITIVE_TYPES:
            raise ValueError(f"unknown primitive type: {self.type!r}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("primitive width/height must be positive")

    def bbox(self) -> Tuple[float, float, float, float]:
        """Axis-aligned bounding box (x0, y0, x1, y1) in image coordinates."""
        pts = self._local_points()
        if pts is None and self.rotation != 0.0 and self.width != self.height:
            pts = self.boundary_points()
        if pts is None:
            return (
                self.cx - self.width / 2,
                self.cy - self.height / 2,
                self.cx + self.width / 2,
                self.cy + self.height / 2,
            )
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def polygon(self) -> list[Tuple[float, float]]:
        """Absolute-coordinate polygon vertices (triangle/rectangle/line)."""
        pts = self._local_points()
        if pts is None:
            raise ValueError(f"{self.type} has no polygon representation")
        return pts

    def boundary_points(self, n: int = ELLIPSE_POLY_POINTS) -> list[Tuple[float, float]]:
        """Polygon outline in absolute coordinates; ellipses are sampled."""
        pts = self._local_points()
        if pts is not None:
            return pts
        hw, hh = self.width / 2, self.height / 2
        return [
            _rotate_translate(hw * math.cos(t), hh * math.sin(t), self.cx, self.cy, self.rotation)
            for t in (2 * math.pi * k / n for k in range(n))
        ]

    def _local_points(self) -> list[Tuple[float, float]] | None:
        if self.vertices is not None:
            return [tuple(v) for v in self.vertices]
        hw, hh = self.width / 2, self.height / 2
        if self.type in ("rectangle", "line", "quadrilateral"):
            local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        elif self.type == "triangle":
            local = [(0.0, -hh), (hw, hh), (-hw, hh)]
        else:
            return None
        return [_rotate_translate(x, y, self.cx, self.cy, self.rotation) for x, y in local]

    def _outline_width(self) -> int:
        # A fixed-width outline drawn on a thin shape (e.g. a 2px-thick line)
        # can fully overwrite the fill, leaving no fill pixel at all -- and
        # since the outline is white, the shape would vanish into the white
        # background. Cap the outline so at least ~half the thinnest
        # dimension stays fill-colored.
        thinnest = min(self.width, self.height)
        return max(1, min(OUTLINE_WIDTH, int(thinnest // 4)))

    def draw(self, draw: ImageDraw.ImageDraw) -> None:
        outline_width = self._outline_width()
        if self.type == "ellipse" or (self.type == "circle" and self.rotation != 0.0):
            draw.polygon(self.boundary_points(), fill=self.color, outline=OUTLINE_COLOR, width=outline_width)
        elif self.type == "circle":
            bbox = (
                self.cx - self.width / 2,
                self.cy - self.height / 2,
                self.cx + self.width / 2,
                self.cy + self.height / 2,
            )
            draw.ellipse(bbox, fill=self.color, outline=OUTLINE_COLOR, width=outline_width)
        else:
            draw.polygon(self.polygon(), fill=self.color, outline=OUTLINE_COLOR, width=outline_width)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "color": list(self.color),
            "is_distractor": self.is_distractor,
            **({"vertices": [list(v) for v in self.vertices]} if self.vertices is not None else {}),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Primitive":
        return cls(
            type=d["type"],
            cx=d["cx"],
            cy=d["cy"],
            width=d["width"],
            height=d["height"],
            rotation=d.get("rotation", 0.0),
            color=tuple(d.get("color", (40, 40, 40))),
            is_distractor=d.get("is_distractor", False),
            vertices=[tuple(v) for v in d["vertices"]] if d.get("vertices") is not None else None,
        )


def _rotate_translate(x: float, y: float, cx: float, cy: float, angle: float) -> Tuple[float, float]:
    ca, sa = math.cos(angle), math.sin(angle)
    rx = x * ca - y * sa
    ry = x * sa + y * ca
    return (cx + rx, cy + ry)


def polygon_pose(ptype: str, vertices) -> Tuple[float, float, float, float, float]:
    """(cx, cy, width, height, rotation) summarising a polygon's vertices.

    Matches the conventions of the untransformed shapes, so an identity
    transform returns the original pose exactly:

    - triangle: vertex 0 is the apex; width = base length, height = apex to
      base midpoint, rotation = direction from base midpoint to apex
      (0 = pointing up), center = halfway between base midpoint and apex.
    - quadrilateral / rectangle / line: vertices in drawing order; width =
      mean length of edges 0 and 2, height = mean of edges 1 and 3,
      rotation = direction of edge 0 (averaged with the reversed edge 2),
      center = vertex mean.
    """
    v = [tuple(map(float, p)) for p in vertices]
    if ptype == "triangle":
        apex, b1, b2 = v[0], v[1], v[2]
        bm = ((b1[0] + b2[0]) / 2, (b1[1] + b2[1]) / 2)
        dx, dy = apex[0] - bm[0], apex[1] - bm[1]
        height = math.hypot(dx, dy)
        width = math.hypot(b1[0] - b2[0], b1[1] - b2[1])
        rotation = math.atan2(dx, -dy)
        return ((apex[0] + bm[0]) / 2, (apex[1] + bm[1]) / 2, max(width, 1e-3), max(height, 1e-3), rotation)
    if len(v) != 4:
        raise ValueError(f"expected 4 vertices for {ptype}, got {len(v)}")
    e = [(v[(i + 1) % 4][0] - v[i][0], v[(i + 1) % 4][1] - v[i][1]) for i in range(4)]
    width = (math.hypot(*e[0]) + math.hypot(*e[2])) / 2
    height = (math.hypot(*e[1]) + math.hypot(*e[3])) / 2
    ax, ay = e[0][0] - e[2][0], e[0][1] - e[2][1]
    rotation = math.atan2(ay, ax)
    cx = sum(p[0] for p in v) / 4
    cy = sum(p[1] for p in v) / 4
    return cx, cy, max(width, 1e-3), max(height, 1e-3), rotation
