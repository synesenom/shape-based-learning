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
            # Chainstay + fork: lines from the bottom-bracket apex to just
            # inside each wheel's rim (not the wheel's center -- a line
            # plunging deep into a circle's interior meets it at a near-
            # tangent angle that can leave a 1px gap in the outline stroke,
            # merging the two into one blob; see the person template).
            PrimitiveSpec("line", 0.4248, 0.6476, 0.168, 0.05, rotation=2.678, name="chainstay"),
            PrimitiveSpec("line", 0.5752, 0.6476, 0.168, 0.05, rotation=0.4636, name="fork"),
            PrimitiveSpec("line", 0.62, 0.26, 0.22, 0.05, rotation=-0.2, name="handlebar"),
        ],
    ),
    "car": ClassTemplate(
        "car",
        [
            PrimitiveSpec("rectangle", 0.5, 0.55, 0.8, 0.36, name="body"),
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
            PrimitiveSpec("rectangle", 0.5, 0.76, 0.2, 0.4, name="trunk"),
            PrimitiveSpec("triangle", 0.5, 0.34, 0.56, 0.5, name="crown"),
        ],
    ),
    # The exact vertical flip of `tree`: the same two parts with the same
    # dimensions, the triangle below and pointing down. The first version
    # (DATASET_VERSION 1) used different part sizes (0.4x0.28 sign, 0.4x0.4
    # pointer), which let a bag of part sizes -- no positions at all --
    # separate the pair perfectly, so it was not a relation twin.
    "arrow_sign": ClassTemplate(
        "arrow_sign",
        [
            PrimitiveSpec("rectangle", 0.5, 0.24, 0.2, 0.4, name="sign"),
            PrimitiveSpec("triangle", 0.5, 0.66, 0.56, 0.5, rotation=PI, name="pointer"),
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
            # Limbs are shortened so their inner end just touches the body's
            # edge instead of plunging deep into its interior: a deep
            # penetration makes the join between limb and body nearly
            # tangent, which can leave a 1px gap in the outline stroke at
            # that acute angle and let the two same-colored fills merge
            # into one blob (breaking classical extraction).
            PrimitiveSpec("line", 0.2623, 0.4404, 0.2398, 0.05, rotation=0.5, name="arm_left"),
            PrimitiveSpec("line", 0.7377, 0.4404, 0.2398, 0.05, rotation=-0.5, name="arm_right"),
            PrimitiveSpec("line", 0.4007, 0.8524, 0.3149, 0.05, rotation=1.3, name="leg_left"),
            PrimitiveSpec("line", 0.5993, 0.8525, 0.3149, 0.05, rotation=1.84, name="leg_right"),
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

# Bumped whenever a template changes what gets rendered. It is part of the
# extraction-cache key, so a template fix can never be served stale
# primitives from a cache filled under the old geometry.
#   1: original templates.
#   2: arrow_sign made the exact vertical flip of tree (a true relation twin).
DATASET_VERSION = 2


# ---------------------------------------------------------------------------
# Novel compositions (PLAN.md section 5, "hold out variants during training")
#
# Each class gets held-out variants that change the *composition* -- a part
# added or removed -- while keeping the arrangement that defines the class.
# Training always uses the base template; the novel-composition test set is
# rendered from these. A variant must still be unambiguous to a person
# (a car with three windows is a car), and must not turn into another
# class: no variant of `tree` or `arrow_sign` changes which part sits on
# top, and no truck variant loses the cab that separates it from a car.


def _variant(base: str, name: str, drop: tuple = (), add: tuple = ()) -> ClassTemplate:
    parts = [p for p in CLASS_TEMPLATES[base].parts if p.name not in drop]
    return ClassTemplate(f"{base}:{name}", parts + list(add))


NOVEL_VARIANTS: dict[str, list[ClassTemplate]] = {
    "bicycle": [
        _variant("bicycle", "no_handlebar", drop=("handlebar",)),
        _variant("bicycle", "seat", add=(PrimitiveSpec("line", 0.38, 0.26, 0.14, 0.05, name="seat"),)),
    ],
    "car": [
        _variant(
            "car",
            "three_windows",
            drop=("window_left", "window_right"),
            add=(
                PrimitiveSpec("rectangle", 0.27, 0.4, 0.16, 0.18, name="window_1"),
                PrimitiveSpec("rectangle", 0.5, 0.4, 0.16, 0.18, name="window_2"),
                PrimitiveSpec("rectangle", 0.73, 0.4, 0.16, 0.18, name="window_3"),
            ),
        ),
        _variant("car", "one_window", drop=("window_right",)),
    ],
    "truck": [
        _variant("truck", "two_wheels", drop=("wheel_2",)),
        _variant(
            "truck",
            "cab_window",
            add=(PrimitiveSpec("rectangle", 0.8, 0.42, 0.12, 0.12, name="cab_window"),),
        ),
    ],
    "cat_face": [
        _variant(
            "cat_face",
            "eyes",
            add=(
                PrimitiveSpec("circle", 0.38, 0.5, 0.1, 0.1, name="eye_left"),
                PrimitiveSpec("circle", 0.62, 0.5, 0.1, 0.1, name="eye_right"),
            ),
        ),
        _variant(
            "cat_face",
            "nose",
            add=(PrimitiveSpec("triangle", 0.5, 0.64, 0.1, 0.08, rotation=PI, name="nose"),),
        ),
    ],
    "house": [
        _variant(
            "house",
            "two_windows",
            add=(PrimitiveSpec("rectangle", 0.72, 0.58, 0.14, 0.14, name="window_right"),),
        ),
        _variant("house", "no_window", drop=("window",)),
    ],
    "tree": [
        _variant(
            "tree",
            "two_tier",
            drop=("crown",),
            add=(
                PrimitiveSpec("triangle", 0.5, 0.42, 0.6, 0.36, name="crown_low"),
                PrimitiveSpec("triangle", 0.5, 0.2, 0.44, 0.3, name="crown_high"),
            ),
        ),
    ],
    # Mirrors tree's two-tier variant, so the novel pair stays a relation
    # twin: identical part counts and sizes, flipped arrangement.
    "arrow_sign": [
        _variant(
            "arrow_sign",
            "two_tier",
            drop=("pointer",),
            add=(
                PrimitiveSpec("triangle", 0.5, 0.58, 0.6, 0.36, rotation=PI, name="pointer_high"),
                PrimitiveSpec("triangle", 0.5, 0.8, 0.44, 0.3, rotation=PI, name="pointer_low"),
            ),
        ),
    ],
    "snowman": [
        _variant("snowman", "hat", add=(PrimitiveSpec("rectangle", 0.5, 0.06, 0.18, 0.1, name="hat"),)),
        _variant(
            "snowman",
            "arms",
            add=(
                PrimitiveSpec("line", 0.25, 0.43, 0.2, 0.04, rotation=-0.4, name="arm_left"),
                PrimitiveSpec("line", 0.75, 0.43, 0.2, 0.04, rotation=0.4, name="arm_right"),
            ),
        ),
    ],
    "person": [
        _variant("person", "no_arms", drop=("arm_left", "arm_right")),
        _variant("person", "hat", add=(PrimitiveSpec("rectangle", 0.5, 0.03, 0.2, 0.06, name="hat"),)),
    ],
    "fish": [
        _variant("fish", "eye", add=(PrimitiveSpec("circle", 0.26, 0.45, 0.08, 0.08, name="eye"),)),
        _variant(
            "fish",
            "fin",
            add=(PrimitiveSpec("triangle", 0.42, 0.28, 0.18, 0.14, name="fin"),),
        ),
    ],
}
