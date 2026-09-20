"""Render class templates into images with ground-truth primitives.

Each rendered sample = a class template placed on a canvas with:
  - object-level scale and position jitter (so objects aren't always the
    same size/position — needed for the position/scale-shift test in
    PLAN.md section 5),
  - small per-part size/position jitter,
  - optional random distractor shapes.

``render_sample`` is the single source of truth: the returned primitive
list is exactly what was drawn, in absolute pixel coordinates, so ground
truth always matches the image by construction.
"""

from __future__ import annotations

import csv
import json
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from .objects import CLASS_NAMES, CLASS_TEMPLATES, ClassTemplate
from .primitives import PRIMITIVE_TYPES, Primitive

DEFAULT_COLOR = (40, 40, 40)
BACKGROUND = (255, 255, 255)


@dataclass
class GenerationConfig:
    image_size: int = 128
    object_scale_range: Tuple[float, float] = (0.5, 0.85)
    position_jitter: float = 0.08
    part_size_jitter: float = 0.08
    part_pos_jitter: float = 0.02
    color: Tuple[int, int, int] = DEFAULT_COLOR
    color_jitter: int = 15
    background: Tuple[int, int, int] = BACKGROUND
    distractor_prob: float = 0.0
    max_distractors: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationConfig":
        d = dict(d)
        if "object_scale_range" in d:
            d["object_scale_range"] = tuple(d["object_scale_range"])
        if "color" in d:
            d["color"] = tuple(d["color"])
        if "background" in d:
            d["background"] = tuple(d["background"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Sample:
    image: Image.Image
    primitives: List[Primitive]
    label: str


def _stable_seed(*parts) -> int:
    """Deterministic seed derivation (Python's str hash is randomized per-process)."""
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(s)


def _jitter_color(base: Tuple[int, int, int], amount: int, rng: random.Random) -> Tuple[int, int, int]:
    if amount <= 0:
        return base
    shift = rng.randint(-amount, amount)
    return tuple(max(0, min(255, c + shift)) for c in base)


def render_sample(
    class_name: str,
    rng: random.Random,
    cfg: GenerationConfig,
    template: Optional[ClassTemplate] = None,
) -> Sample:
    """Render one instance of ``class_name``, returning image + ground truth."""
    template = template or CLASS_TEMPLATES[class_name]
    size = cfg.image_size

    obj_scale = rng.uniform(*cfg.object_scale_range)
    obj_size = obj_scale * size

    max_jitter = cfg.position_jitter * size
    half = obj_size / 2
    lo, hi = half, size - half
    cx = size / 2 + rng.uniform(-max_jitter, max_jitter)
    cy = size / 2 + rng.uniform(-max_jitter, max_jitter)
    cx = min(max(cx, lo), hi) if lo <= hi else size / 2
    cy = min(max(cy, lo), hi) if lo <= hi else size / 2

    instance_color = _jitter_color(cfg.color, cfg.color_jitter, rng)

    primitives: List[Primitive] = []
    for spec in template.parts:
        w = spec.width * obj_size * (1 + rng.uniform(-cfg.part_size_jitter, cfg.part_size_jitter))
        h = spec.height * obj_size * (1 + rng.uniform(-cfg.part_size_jitter, cfg.part_size_jitter))
        px = cx + (spec.cx - 0.5) * obj_size + rng.uniform(-cfg.part_pos_jitter, cfg.part_pos_jitter) * obj_size
        py = cy + (spec.cy - 0.5) * obj_size + rng.uniform(-cfg.part_pos_jitter, cfg.part_pos_jitter) * obj_size
        primitives.append(
            Primitive(
                type=spec.type,
                cx=px,
                cy=py,
                width=w,
                height=h,
                rotation=spec.rotation,
                color=instance_color,
                is_distractor=False,
            )
        )

    # Template parts must draw in template order (container before contained
    # detail, e.g. a car body before its windows) so a part fully inside
    # another isn't erased. Only distractors -- which never contain or are
    # contained by the object -- get a random front/behind placement.
    behind: List[Primitive] = []
    front: List[Primitive] = []
    if cfg.distractor_prob > 0 and rng.random() < cfg.distractor_prob:
        n_distractors = rng.randint(1, max(1, cfg.max_distractors))
        for _ in range(n_distractors):
            d = _random_distractor(size, rng, cfg)
            (front if rng.random() < 0.5 else behind).append(d)

    draw_order = behind + primitives + front

    image = Image.new("RGB", (size, size), cfg.background)
    draw = ImageDraw.Draw(image)
    for p in draw_order:
        p.draw(draw)

    return Sample(image=image, primitives=draw_order, label=class_name)


def _random_distractor(size: int, rng: random.Random, cfg: GenerationConfig) -> Primitive:
    ptype = rng.choice(PRIMITIVE_TYPES)
    w = rng.uniform(0.08, 0.18) * size
    h = rng.uniform(0.08, 0.18) * size if ptype != "line" else w * rng.uniform(0.1, 0.2)
    margin = max(w, h) / 2
    px = rng.uniform(margin, size - margin)
    py = rng.uniform(margin, size - margin)
    rotation = rng.uniform(0, 2 * 3.141592653589793) if ptype in ("triangle", "rectangle", "line") else 0.0
    color = _jitter_color(cfg.color, cfg.color_jitter, rng)
    return Primitive(type=ptype, cx=px, cy=py, width=w, height=h, rotation=rotation, color=color, is_distractor=True)


class SynthShapeDataset(Sequence):
    """In-memory / render-on-the-fly dataset of (image, primitives, label).

    Deterministic given (classes, n_per_class, seed): index ``i`` always
    renders the same sample. Does not require torch.
    """

    def __init__(
        self,
        classes: Sequence[str] = CLASS_NAMES,
        n_per_class: int = 100,
        cfg: Optional[GenerationConfig] = None,
        seed: int = 0,
    ):
        self.classes = list(classes)
        self.n_per_class = n_per_class
        self.cfg = cfg or GenerationConfig()
        self.seed = seed

    def __len__(self) -> int:
        return len(self.classes) * self.n_per_class

    def _index_to_class_and_seed(self, index: int) -> Tuple[str, int]:
        if index < 0:
            index += len(self)
        if not (0 <= index < len(self)):
            raise IndexError(index)
        class_idx, within = divmod(index, self.n_per_class)
        class_name = self.classes[class_idx]
        sample_seed = _stable_seed(self.seed, class_name, within)
        return class_name, sample_seed

    def __getitem__(self, index: int) -> Sample:
        class_name, sample_seed = self._index_to_class_and_seed(index)
        rng = random.Random(sample_seed)
        return render_sample(class_name, rng, self.cfg)


def generate_dataset(
    output_dir: str | Path,
    classes: Sequence[str] = CLASS_NAMES,
    n_per_class: int = 1000,
    cfg: Optional[GenerationConfig] = None,
    seed: int = 0,
    split: str = "train",
) -> Path:
    """Render a full split to disk: <output_dir>/<split>/<class>/<i>.png (+.json), and a manifest.csv."""
    output_dir = Path(output_dir)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    dataset = SynthShapeDataset(classes=classes, n_per_class=n_per_class, cfg=cfg, seed=seed)
    manifest_rows = []

    for class_name in classes:
        (split_dir / class_name).mkdir(parents=True, exist_ok=True)

    for i in range(len(dataset)):
        class_name, sample_seed = dataset._index_to_class_and_seed(i)
        within = i % n_per_class
        sample = dataset[i]

        img_path = split_dir / class_name / f"{within:05d}.png"
        json_path = split_dir / class_name / f"{within:05d}.json"

        sample.image.save(img_path)
        with open(json_path, "w") as f:
            json.dump(
                {
                    "label": sample.label,
                    "image_size": dataset.cfg.image_size,
                    "primitives": [p.to_dict() for p in sample.primitives],
                },
                f,
                indent=2,
            )

        manifest_rows.append(
            {
                "path": str(img_path.relative_to(output_dir)),
                "json": str(json_path.relative_to(output_dir)),
                "label": sample.label,
                "split": split,
            }
        )

    manifest_path = output_dir / f"manifest_{split}.csv"
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "json", "label", "split"])
        if write_header:
            writer.writeheader()
        writer.writerows(manifest_rows)

    return split_dir
