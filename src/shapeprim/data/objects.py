"""Class templates: each object class is a fixed arrangement of primitives.

Coordinates are normalized to the object's own bounding box ([0, 1] x
[0, 1], y growing downward) so templates can be scaled/positioned onto a
canvas of any size. ``width``/``height`` are likewise fractions of the
object's bounding-box side.

The ``tree`` and ``arrow_sign`` templates are deliberately built from the
same two primitive types (rectangle + triangle) in swapped arrangement
(trunk-below-crown vs. rectangle-above-triangle), and ``car``/``truck``
share rectangle+circle vocabulary with different counts. Both pairs exist
to test whether a classifier uses arrangement, not just which shapes are
present (see PLAN.md section 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class PrimitiveSpec:
    """Nominal (un-jittered) placement of one part within an object."""

    type: str
    cx: float
    cy: float
    width: float
    height: float
    rotation: float = 0.0
    name: str = ""


@dataclass
class ClassTemplate:
    name: str
    parts: List[PrimitiveSpec] = field(default_factory=list)


PI = math.pi

CLASS_TEMPLATES: dict[str, ClassTemplate] = {
    "bicycle": ClassTemplate(
        "bicycle",
        [
            PrimitiveSpec("circle", 0.22, 0.75, 0.32, 0.32, name="wheel_rear"),
            PrimitiveSpec("circle", 0.78, 0.75, 0.32, 0.32, name="wheel_front"),
            # Frame triangle, apex pointing down (rotation=PI) like a real bike
            # frame: the wide base is up top (seat/head tube), the apex is the
            # bottom bracket, a single low point between the wheels.
            PrimitiveSpec("triangle", 0.5, 0.45, 0.3, 0.32, rotation=PI, name="frame"),
            # Chainstay + fork: lines from the bottom-bracket apex down to each
            # wheel hub -- the feature that reads as "bike" rather than "triangle + circles".
            PrimitiveSpec("line", 0.36, 0.68, 0.313, 0.05, rotation=2.678, name="chainstay"),
            PrimitiveSpec("line", 0.64, 0.68, 0.313, 0.05, rotation=0.464, name="fork"),
            PrimitiveSpec("line", 0.62, 0.26, 0.14, 0.05, rotation=-0.2, name="handlebar"),
        ],
    ),
    "car": ClassTemplate(
        "car",
        [
            PrimitiveSpec("rectangle", 0.5, 0.55, 0.88, 0.34, name="body"),
            PrimitiveSpec("rectangle", 0.34, 0.4, 0.22, 0.18, name="window_left"),
            PrimitiveSpec("rectangle", 0.66, 0.4, 0.22, 0.18, name="window_right"),
            PrimitiveSpec("circle", 0.28, 0.78, 0.2, 0.2, name="wheel_left"),
            PrimitiveSpec("circle", 0.72, 0.78, 0.2, 0.2, name="wheel_right"),
        ],
    ),
    "truck": ClassTemplate(
        "truck",
        [
            PrimitiveSpec("rectangle", 0.76, 0.5, 0.3, 0.4, name="cab"),
            PrimitiveSpec("rectangle", 0.35, 0.55, 0.5, 0.3, name="cargo"),
            PrimitiveSpec("circle", 0.25, 0.8, 0.18, 0.18, name="wheel_1"),
            PrimitiveSpec("circle", 0.55, 0.8, 0.18, 0.18, name="wheel_2"),
            PrimitiveSpec("circle", 0.85, 0.8, 0.18, 0.18, name="wheel_3"),
        ],
    ),
    "cat_face": ClassTemplate(
        "cat_face",
        [
            PrimitiveSpec("circle", 0.5, 0.55, 0.6, 0.6, name="head"),
            PrimitiveSpec("triangle", 0.28, 0.2, 0.22, 0.3, rotation=-0.35, name="ear_left"),
            PrimitiveSpec("triangle", 0.72, 0.2, 0.22, 0.3, rotation=0.35, name="ear_right"),
        ],
    ),
    "house": ClassTemplate(
        "house",
        [
            PrimitiveSpec("rectangle", 0.5, 0.62, 0.7, 0.5, name="body"),
            PrimitiveSpec("triangle", 0.5, 0.28, 0.8, 0.3, name="roof"),
            PrimitiveSpec("rectangle", 0.5, 0.8, 0.14, 0.24, name="door"),
            PrimitiveSpec("rectangle", 0.28, 0.58, 0.14, 0.14, name="window"),
        ],
    ),
    "tree": ClassTemplate(
        "tree",
        [
            PrimitiveSpec("rectangle", 0.5, 0.76, 0.16, 0.4, name="trunk"),
            PrimitiveSpec("triangle", 0.5, 0.34, 0.56, 0.5, name="crown"),
        ],
    ),
    "arrow_sign": ClassTemplate(
        "arrow_sign",
        [
            PrimitiveSpec("rectangle", 0.5, 0.28, 0.4, 0.28, name="sign"),
            PrimitiveSpec("triangle", 0.5, 0.68, 0.4, 0.4, rotation=PI, name="pointer"),
        ],
    ),
    "snowman": ClassTemplate(
        "snowman",
        [
            PrimitiveSpec("circle", 0.5, 0.78, 0.5, 0.5, name="bottom"),
            PrimitiveSpec("circle", 0.5, 0.45, 0.36, 0.36, name="middle"),
            PrimitiveSpec("circle", 0.5, 0.2, 0.24, 0.24, name="top"),
        ],
    ),
    "person": ClassTemplate(
        "person",
        [
            PrimitiveSpec("circle", 0.5, 0.18, 0.26, 0.26, name="head"),
            PrimitiveSpec("rectangle", 0.5, 0.52, 0.3, 0.4, name="body"),
            PrimitiveSpec("line", 0.28, 0.45, 0.28, 0.05, rotation=0.5, name="arm_left"),
            PrimitiveSpec("line", 0.72, 0.45, 0.28, 0.05, rotation=-0.5, name="arm_right"),
            PrimitiveSpec("line", 0.4, 0.85, 0.32, 0.05, rotation=1.3, name="leg_left"),
            PrimitiveSpec("line", 0.6, 0.85, 0.32, 0.05, rotation=1.84, name="leg_right"),
        ],
    ),
    "fish": ClassTemplate(
        "fish",
        [
            PrimitiveSpec("circle", 0.42, 0.5, 0.6, 0.36, name="body"),
            PrimitiveSpec("triangle", 0.82, 0.5, 0.28, 0.34, rotation=PI / 2, name="tail"),
        ],
    ),
}

CLASS_NAMES = tuple(CLASS_TEMPLATES.keys())
