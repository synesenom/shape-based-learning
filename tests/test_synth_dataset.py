import json
import random

import pytest

from shapeprim.data.objects import CLASS_NAMES, CLASS_TEMPLATES
from shapeprim.data.synth_dataset import (
    GenerationConfig,
    SynthShapeDataset,
    generate_dataset,
    render_sample,
)


def test_render_sample_image_size():
    cfg = GenerationConfig(image_size=96)
    sample = render_sample("house", random.Random(0), cfg)
    assert sample.image.size == (96, 96)


def test_render_sample_background_corner_is_background_color():
    cfg = GenerationConfig(image_size=128, object_scale_range=(0.4, 0.5), position_jitter=0.0)
    sample = render_sample("snowman", random.Random(1), cfg)
    assert sample.image.getpixel((0, 0)) == cfg.background


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_ground_truth_primitives_match_rendered_pixels(class_name):
    """Every primitive's fill color must appear near its own center: the
    core check that the JSON ground truth matches the image. A small
    neighborhood (not the single rounded-center pixel) is checked because a
    thin rotated primitive's exact geometric center can rasterize onto its
    outline stroke by a fraction of a pixel."""
    cfg = GenerationConfig(image_size=128, distractor_prob=0.0)
    radius = 2
    for trial in range(5):
        sample = render_sample(class_name, random.Random(trial), cfg)
        for p in sample.primitives:
            cx, cy = int(round(p.cx)), int(round(p.cy))
            found = False
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x = min(max(cx + dx, 0), cfg.image_size - 1)
                    y = min(max(cy + dy, 0), cfg.image_size - 1)
                    if sample.image.getpixel((x, y)) == tuple(p.color):
                        found = True
                        break
                if found:
                    break
            assert found, (class_name, p.type, cx, cy)


def test_num_primitives_matches_template_when_no_distractors():
    cfg = GenerationConfig(distractor_prob=0.0)
    for class_name in CLASS_NAMES:
        sample = render_sample(class_name, random.Random(0), cfg)
        assert len(sample.primitives) == len(CLASS_TEMPLATES[class_name].parts)
        assert all(not p.is_distractor for p in sample.primitives)


def test_distractors_are_flagged_and_added_on_top_of_template_parts():
    cfg = GenerationConfig(distractor_prob=1.0, max_distractors=2)
    sample = render_sample("tree", random.Random(0), cfg)
    n_template = len(CLASS_TEMPLATES["tree"].parts)
    distractors = [p for p in sample.primitives if p.is_distractor]
    assert len(sample.primitives) > n_template
    assert 1 <= len(distractors) <= cfg.max_distractors


def test_rendering_is_reproducible_given_same_seed():
    cfg = GenerationConfig(distractor_prob=0.2)
    s1 = render_sample("car", random.Random(42), cfg)
    s2 = render_sample("car", random.Random(42), cfg)
    assert s1.image.tobytes() == s2.image.tobytes()
    assert [p.to_dict() for p in s1.primitives] == [p.to_dict() for p in s2.primitives]


def test_dataset_is_deterministic_across_instances():
    ds1 = SynthShapeDataset(classes=["house", "fish"], n_per_class=3, seed=7)
    ds2 = SynthShapeDataset(classes=["house", "fish"], n_per_class=3, seed=7)
    assert len(ds1) == 6
    for i in range(len(ds1)):
        a, b = ds1[i], ds2[i]
        assert a.label == b.label
        assert a.image.tobytes() == b.image.tobytes()


def test_dataset_length_and_indexing():
    classes = ["house", "fish", "tree"]
    ds = SynthShapeDataset(classes=classes, n_per_class=4, seed=0)
    assert len(ds) == 12
    labels = [ds[i].label for i in range(len(ds))]
    assert labels == ["house"] * 4 + ["fish"] * 4 + ["tree"] * 4
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_generate_dataset_writes_expected_files(tmp_path):
    classes = ["house", "fish"]
    cfg = GenerationConfig(image_size=32)
    split_dir = generate_dataset(tmp_path, classes=classes, n_per_class=3, cfg=cfg, seed=0, split="train")

    for class_name in classes:
        pngs = sorted((split_dir / class_name).glob("*.png"))
        jsons = sorted((split_dir / class_name).glob("*.json"))
        assert len(pngs) == 3
        assert len(jsons) == 3

    manifest = tmp_path / "manifest_train.csv"
    assert manifest.exists()
    lines = manifest.read_text().strip().splitlines()
    assert len(lines) == 1 + 3 * len(classes)  # header + rows

    with open(split_dir / "house" / "00000.json") as f:
        data = json.load(f)
    assert data["label"] == "house"
    assert data["image_size"] == 32
    assert len(data["primitives"]) >= 2
