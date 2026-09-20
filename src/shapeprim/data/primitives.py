"""Geometric primitive dataclass, polygon math, and rendering.

Phase 1 vocabulary: circle, triangle, rectangle, line. Every primitive is
described by a center, a width/height (in pixels), and a rotation (radians,
clockwise in image coordinates since y grows downward). This schema is
shared with later phases, which will relax the circle's width==height
constraint (ellipse) and generalize rectangles to quadrilaterals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

from PIL import ImageDraw

PRIMITIVE_TYPES = ("circle", "triangle", "rectangle", "line")

Color = Tuple[int, int, int]


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

    def __post_init__(self) -> None:
        if self.type not in PRIMITIVE_TYPES:
            raise ValueError(f"unknown primitive type: {self.type!r}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("primitive width/height must be positive")

    def bbox(self) -> Tuple[float, float, float, float]:
        """Axis-aligned bounding box (x0, y0, x1, y1) in image coordinates."""
        pts = self._local_points()
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

    def _local_points(self) -> list[Tuple[float, float]] | None:
        hw, hh = self.width / 2, self.height / 2
        if self.type == "rectangle" or self.type == "line":
            local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        elif self.type == "triangle":
            local = [(0.0, -hh), (hw, hh), (-hw, hh)]
        else:
            return None
        return [_rotate_translate(x, y, self.cx, self.cy, self.rotation) for x, y in local]

    def draw(self, draw: ImageDraw.ImageDraw) -> None:
        if self.type == "circle":
            bbox = (
                self.cx - self.width / 2,
                self.cy - self.height / 2,
                self.cx + self.width / 2,
                self.cy + self.height / 2,
            )
            draw.ellipse(bbox, fill=self.color)
        else:
            draw.polygon(self.polygon(), fill=self.color)

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
        )


def _rotate_translate(x: float, y: float, cx: float, cy: float, angle: float) -> Tuple[float, float]:
    ca, sa = math.cos(angle), math.sin(angle)
    rx = x * ca - y * sa
    ry = x * sa + y * ca
    return (cx + rx, cy + ry)
