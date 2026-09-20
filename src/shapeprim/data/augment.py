"""Image augmentation for the CNN baseline (PLAN.md section 4, WP0).

The point of this module is fairness, not accuracy. The primitive graph is
translation- and scale-invariant *by construction* (``graph/build.py``
normalizes every feature by the object's own bounding box), so a CNN
trained without augmentation is not a control for "does structure help" --
it is a control for "does *anything* that knows about translation help".
Any position/scale result measured against an un-augmented CNN is an
augmentation result wearing a representation result's clothes. So
augmentation is a first-class, config-driven experimental condition here,
reported with and without, rather than a training detail.

Two rendering-specific details that a default torchvision pipeline gets
wrong on this data:

- **Fill color.** ``RandomAffine``'s default ``fill=0`` paints the
  revealed corners black. On a white-background drawing that is not a
  neutral transformation: it stamps a high-contrast frame onto the image
  whose shape encodes the sampled transform, which the CNN can read as a
  feature. The fill must be the generator's background color.
- **Horizontal flip is off by default.** It is not label-preserving for
  every class here: ``fish`` points its tail one way, ``truck`` has its
  cab on one side, and a flip turns one canonical pose into a pose that
  never occurs in the training distribution. It stays available as a flag
  for experiments that want it, but it is not part of "standard
  augmentation" for this dataset.

Rotation is likewise off by default, and for a sharper reason: a 180°
rotation maps ``tree`` onto ``arrow_sign`` (the two classes that exist
precisely to test whether arrangement is used). Rotating past ~90° makes
those two classes genuinely ambiguous, so ``degrees`` is capped and the
caller has to opt in. See PLAN.md section 8, "rotation symmetry breaks
some classes".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

MAX_SAFE_DEGREES = 90.0


@dataclass
class AugmentConfig:
    """Random-affine augmentation for image models.

    Defaults are "off": an ``AugmentConfig()`` is the identity, so a config
    file that says nothing about augmentation gets none.
    """

    enabled: bool = False
    translate: float = 0.0  # max shift as a fraction of image size, both axes
    scale_range: Tuple[float, float] = (1.0, 1.0)
    degrees: float = 0.0
    shear: float = 0.0
    hflip_prob: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0

    def __post_init__(self) -> None:
        self.scale_range = tuple(self.scale_range)
        if not 0.0 <= self.translate <= 1.0:
            raise ValueError(f"translate must be in [0, 1], got {self.translate}")
        if len(self.scale_range) != 2 or self.scale_range[0] <= 0 or self.scale_range[0] > self.scale_range[1]:
            raise ValueError(f"scale_range must be (lo, hi) with 0 < lo <= hi, got {self.scale_range}")
        if self.degrees < 0:
            raise ValueError(f"degrees must be >= 0, got {self.degrees}")
        if self.degrees > MAX_SAFE_DEGREES:
            # Not a style preference: past 90 degrees the tree/arrow_sign
            # pair (same primitives, swapped arrangement) becomes genuinely
            # ambiguous, so the label noise this introduces would be
            # confounded with whatever the experiment is measuring.
            raise ValueError(
                f"degrees={self.degrees} exceeds {MAX_SAFE_DEGREES}: rotation that large maps "
                "tree onto arrow_sign and makes those labels ambiguous (PLAN.md section 8). "
                "Cap the range or drop those classes explicitly."
            )
        if not 0.0 <= self.hflip_prob <= 1.0:
            raise ValueError(f"hflip_prob must be in [0, 1], got {self.hflip_prob}")

    @property
    def is_identity(self) -> bool:
        """True if this config would leave every image untouched."""
        return not self.enabled or (
            self.translate == 0.0
            and tuple(self.scale_range) == (1.0, 1.0)
            and self.degrees == 0.0
            and self.shear == 0.0
            and self.hflip_prob == 0.0
            and self.brightness == 0.0
            and self.contrast == 0.0
        )

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AugmentConfig":
        if not d:
            return cls()
        d = dict(d)
        if "scale_range" in d and d["scale_range"] is not None:
            d["scale_range"] = tuple(d["scale_range"])
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown AugmentConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "translate": self.translate,
            "scale_range": list(self.scale_range),
            "degrees": self.degrees,
            "shear": self.shear,
            "hflip_prob": self.hflip_prob,
            "brightness": self.brightness,
            "contrast": self.contrast,
        }


def build_transform(cfg: AugmentConfig, background: Sequence[int] = (255, 255, 255)):
    """Build a torchvision transform for ``cfg``, or None if it's the identity.

    Returns a callable PIL image -> PIL image. torch is imported lazily so
    that data-only code paths (and tests that never touch torch) keep
    working without it.
    """
    if cfg.is_identity:
        return None

    from torchvision import transforms

    fill = [int(c) for c in background]

    ops = []
    if cfg.hflip_prob > 0:
        ops.append(transforms.RandomHorizontalFlip(p=cfg.hflip_prob))

    needs_affine = (
        cfg.translate > 0 or tuple(cfg.scale_range) != (1.0, 1.0) or cfg.degrees > 0 or cfg.shear > 0
    )
    if needs_affine:
        ops.append(
            transforms.RandomAffine(
                degrees=cfg.degrees,
                translate=(cfg.translate, cfg.translate) if cfg.translate > 0 else None,
                scale=tuple(cfg.scale_range) if tuple(cfg.scale_range) != (1.0, 1.0) else None,
                shear=cfg.shear if cfg.shear > 0 else None,
                fill=fill,  # background, not black -- see module docstring
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )

    if cfg.brightness > 0 or cfg.contrast > 0:
        ops.append(transforms.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast))

    return transforms.Compose(ops) if ops else None


# Named presets, so experiment configs can say `augment: standard` instead
# of repeating magic numbers. "standard" is deliberately matched to the
# generator's own nuisance ranges (object_scale_range 0.5-0.85,
# position_jitter 0.08) with headroom, so it covers in-distribution
# variation without inventing a new distribution.
PRESETS = {
    "none": AugmentConfig(),
    "standard": AugmentConfig(enabled=True, translate=0.15, scale_range=(0.8, 1.2)),
    # For the position/scale-shift experiments (WP1/E3): augmentation wide
    # enough to cover the shifted test condition.
    "strong": AugmentConfig(enabled=True, translate=0.30, scale_range=(0.6, 1.5)),
    # Opt-in rotation, capped below the tree/arrow_sign ambiguity point.
    "standard_rot": AugmentConfig(enabled=True, translate=0.15, scale_range=(0.8, 1.2), degrees=30.0),
}


def resolve_augment(spec) -> AugmentConfig:
    """Accept a preset name, a dict, an AugmentConfig, or None."""
    if spec is None:
        return AugmentConfig()
    if isinstance(spec, AugmentConfig):
        return spec
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"unknown augmentation preset {spec!r}; known: {sorted(PRESETS)}")
        # Return a copy so callers can't mutate the shared preset.
        return AugmentConfig(**PRESETS[spec].to_dict())
    if isinstance(spec, dict):
        return AugmentConfig.from_dict(spec)
    raise TypeError(f"cannot interpret augmentation spec of type {type(spec).__name__}")
