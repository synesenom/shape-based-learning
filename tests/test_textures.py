import colorsys
import random

import numpy as np
import pytest

from shapeprim.data.synth_dataset import GenerationConfig, SynthShapeDataset, render_sample
from shapeprim.data.textures import TEXTURE_KINDS, make_texture, palette_color


def test_appearance_does_not_touch_geometry():
    flat = SynthShapeDataset(n_per_class=2, cfg=GenerationConfig(image_size=64, view_angle_range=(0, 30)), seed=0, split="t")
    styled = SynthShapeDataset(
        n_per_class=2,
        cfg=GenerationConfig(image_size=64, view_angle_range=(0, 30), texture=list(TEXTURE_KINDS),
                             palette="unseen", clutter=True, occlusion_range=(0.1, 0.4), noise_std=5.0),
        seed=0, split="t",
    )
    for i in range(len(flat)):
        a, b = flat[i], styled[i]
        assert [p.to_dict() for p in a.primitives] == [p.to_dict() for p in b.primitives]
        assert not np.array_equal(np.asarray(a.image), np.asarray(b.image))


def test_appearance_off_is_phase1_rendering():
    cfg = GenerationConfig(image_size=64)
    assert not cfg.appearance_enabled
    a = render_sample("car", random.Random(1), cfg)
    b = render_sample("car", random.Random(1), GenerationConfig.from_dict({"image_size": 64, "texture": "flat"}))
    assert np.array_equal(np.asarray(a.image), np.asarray(b.image))


@pytest.mark.parametrize("palette,lo,hi", [("seen", 0.0, 0.5), ("unseen", 0.5, 1.0)])
def test_palettes_are_disjoint_hue_halves(palette, lo, hi):
    rng = random.Random(0)
    for _ in range(200):
        r, g, b = palette_color(rng, palette)
        h, _, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        assert lo - 0.01 <= h <= hi + 0.01


@pytest.mark.parametrize("kind", ("flat",) + TEXTURE_KINDS)
def test_texture_shapes(kind):
    t = make_texture(kind, 32, (100, 50, 20), random.Random(0), np.random.default_rng(0))
    assert t.shape == (32, 32, 3) and t.dtype == np.uint8


def test_occlusion_covers_part_of_the_object():
    cfg0 = GenerationConfig(image_size=64, palette="seen")
    cfg1 = GenerationConfig(image_size=64, palette="seen", occlusion_range=(0.3, 0.3))
    diffs = []
    for t in range(10):
        a = np.asarray(render_sample("house", random.Random(t), cfg0).image).astype(int)
        b = np.asarray(render_sample("house", random.Random(t), cfg1).image).astype(int)
        diffs.append((np.abs(a - b).sum(axis=2) > 0).mean())
    assert np.mean(diffs) > 0.02
