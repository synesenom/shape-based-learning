import random

import pytest

from shapeprim.data.synth_dataset import GenerationConfig, render_sample
from shapeprim.extract.classical import ClassicalExtractor
from shapeprim.extract.eval_match import match_primitives
from shapeprim.data.primitives import Primitive

# Classes where classical extraction reliably separates every part from the
# next (see the "Known limitation" note in extract/classical.py).
RELIABLE_CLASSES = ("car", "truck", "house", "tree", "arrow_sign", "snowman", "fish", "cat_face")
# Classes with several thin articulated parts meeting a body at a shallow
# angle, where classical extraction is known to under-perform.
WEAK_CLASSES = ("bicycle", "person")


def _mean_f1(class_name, n_trials=20):
    cfg = GenerationConfig(image_size=128, distractor_prob=0.0)
    extractor = ClassicalExtractor()
    f1s = []
    for trial in range(n_trials):
        sample = render_sample(class_name, random.Random(trial), cfg)
        pred = extractor.extract(sample.image)
        res = match_primitives(pred, sample.primitives, cfg.image_size)
        f1s.append(res["f1"])
    return sum(f1s) / len(f1s)


@pytest.mark.parametrize("class_name", RELIABLE_CLASSES)
def test_classical_extractor_near_perfect_on_reliable_classes(class_name):
    assert _mean_f1(class_name) >= 0.65


@pytest.mark.parametrize("class_name", WEAK_CLASSES)
def test_classical_extractor_still_extracts_something_on_weak_classes(class_name):
    # Not near-perfect (documented limitation), but must not silently return
    # nothing or crash -- this is a floor, not a quality bar.
    assert _mean_f1(class_name) >= 0.2


def test_extract_single_clean_circle():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (128, 128), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    Primitive(type="circle", cx=64, cy=64, width=50, height=50, color=(30, 30, 30)).draw(draw)

    pred = ClassicalExtractor().extract(img)
    assert len(pred) == 1
    assert pred[0].type == "circle"
    assert abs(pred[0].cx - 64) < 3
    assert abs(pred[0].cy - 64) < 3
    assert abs(pred[0].width - 50) < 6


def test_extract_single_clean_triangle():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (128, 128), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    Primitive(type="triangle", cx=64, cy=64, width=50, height=40, color=(30, 30, 30)).draw(draw)

    pred = ClassicalExtractor().extract(img)
    assert len(pred) == 1
    assert pred[0].type == "triangle"


def test_extract_empty_image_returns_no_primitives():
    from PIL import Image

    img = Image.new("RGB", (128, 128), (255, 255, 255))
    assert ClassicalExtractor().extract(img) == []


def test_extract_recovers_contained_part():
    """A part fully inside another same-colored part must still be found
    (this is the entire reason primitives are outlined)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (128, 128), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    Primitive(type="rectangle", cx=64, cy=64, width=80, height=80, color=(30, 30, 30)).draw(draw)
    Primitive(type="rectangle", cx=50, cy=50, width=16, height=16, color=(30, 30, 30)).draw(draw)

    pred = ClassicalExtractor().extract(img)
    assert len(pred) == 2
    assert sorted(p.type for p in pred) == ["rectangle", "rectangle"]
