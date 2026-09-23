import random

import numpy as np
import pytest
import torch

from shapeprim.conditions import resolve_condition
from shapeprim.data.objects import CLASS_NAMES
from shapeprim.data.synth_dataset import GenerationConfig, render_sample
from shapeprim.extract.classical import ClassicalExtractorV2
from shapeprim.extract.eval_match import match_primitives
from shapeprim.extract.learned import CenterNetSmall, decode, encode_targets

RELIABLE = ("car", "truck", "house", "tree", "arrow_sign", "snowman", "fish")


def _f1(extractor, cfg, classes, n=10):
    f = []
    for c in classes:
        for t in range(n):
            s = render_sample(c, random.Random(t), cfg)
            f.append(match_primitives(extractor.extract(s.image), s.primitives, cfg.image_size)["f1"])
    return float(np.mean(f))


def test_classical_v2_on_moderate_viewpoints():
    cfg = GenerationConfig(image_size=64, view_angle_range=(0, 30))
    assert _f1(ClassicalExtractorV2(), cfg, RELIABLE) >= 0.75


def test_learned_target_encoding_round_trips():
    cfg = GenerationConfig(image_size=64, view_angle_range=(0, 30))
    for t in range(20):
        s = render_sample(CLASS_NAMES[t % 10], random.Random(t), cfg)
        tg = encode_targets(s.primitives, 64)
        hm = torch.from_numpy(tg["heat"]).clamp(1e-4, 1 - 1e-4)
        out = torch.cat([torch.log(hm / (1 - hm)), torch.from_numpy(tg["reg"])], 0)
        assert match_primitives(decode(out), s.primitives, 64)["f1"] == pytest.approx(1.0)


def test_detector_output_shape():
    out = CenterNetSmall(width=8)(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 10, 32, 32)


def test_shift_block_extra_tests():
    cond = resolve_condition(
        {
            "image_size": 64,
            "shift": {
                "name": "angle",
                "train": {"view_angle_range": [0, 30]},
                "test": {"view_angle_range": [30, 50]},
                "tests": {"a50": {"view_angle_range": [50, 70]}},
            },
        }
    )
    assert cond.is_shift and cond.extra_tests["a50"].view_angle_range == (50.0, 70.0)
