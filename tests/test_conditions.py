import pytest

from shapeprim.conditions import Condition, base_generation_config, resolve_condition


def test_no_shift_block_gives_identical_distributions():
    cond = resolve_condition({"image_size": 64, "experiment": "baseline"})
    assert not cond.is_shift
    assert cond.train_cfg == cond.test_cfg
    assert cond.describe() == {"name": "baseline", "is_shift": False}


def test_shift_applies_overrides_to_each_side():
    cfg = {
        "image_size": 64,
        "position_jitter": 0.08,
        "shift": {
            "name": "position_scale",
            "train": {"object_scale_range": [0.4, 0.55], "position_jitter": 0.03},
            "test": {"object_scale_range": [0.7, 0.9], "position_jitter": 0.25},
        },
    }
    cond = resolve_condition(cfg)
    assert cond.is_shift
    assert cond.name == "position_scale"
    assert cond.train_cfg.object_scale_range == (0.4, 0.55)
    assert cond.test_cfg.object_scale_range == (0.7, 0.9)
    # Base settings not mentioned by the shift survive on both sides.
    assert cond.train_cfg.image_size == 64 and cond.test_cfg.image_size == 64


def test_describe_lists_only_what_changed():
    cfg = {
        "image_size": 64,
        "shift": {"name": "scale", "train": {"object_scale_range": [0.4, 0.5]}},
    }
    changed = resolve_condition(cfg).describe()["changed"]
    assert set(changed) == {"object_scale_range"}
    assert changed["object_scale_range"] == {"train": (0.4, 0.5), "test": (0.5, 0.85)}


def test_one_sided_shift_is_allowed():
    """Overriding only the train side is a legitimate shift."""
    cond = resolve_condition({"shift": {"name": "s", "train": {"distractor_prob": 0.0}, "test": {"distractor_prob": 0.5}}})
    assert cond.is_shift


def test_shift_that_changes_nothing_is_rejected():
    # Silently running an in-distribution experiment under a shift name
    # would report "no degradation" for entirely the wrong reason.
    with pytest.raises(ValueError, match="leaves the train and test distributions identical"):
        resolve_condition({"shift": {"name": "noop", "train": {"position_jitter": 0.1}, "test": {"position_jitter": 0.1}}})


def test_unknown_override_key_is_rejected():
    with pytest.raises(ValueError, match="not GenerationConfig fields"):
        resolve_condition({"shift": {"name": "typo", "test": {"positon_jitter": 0.25}}})


def test_base_config_reads_experiment_keys():
    cfg = base_generation_config({"image_size": 96, "distractor_prob": 0.3, "epochs": 12, "seeds": [0]})
    assert cfg.image_size == 96
    assert cfg.distractor_prob == 0.3


def test_condition_equality_is_by_value_not_identity():
    base = base_generation_config({"image_size": 64})
    other = base_generation_config({"image_size": 64})
    assert not Condition("c", base, other).is_shift


def test_shifted_test_set_actually_renders_differently():
    """The shift must reach the pixels, not just the config object.

    If the two test distributions ever silently coincide, every shift
    experiment reports a drop of exactly 0.000 and looks like evidence for
    invariance. That failure is indistinguishable from a real result, so it
    is checked directly.
    """
    import numpy as np

    from shapeprim.data.synth_dataset import SynthShapeDataset

    cfg = {
        "image_size": 64,
        "shift": {
            "name": "position_scale",
            "train": {"object_scale_range": [0.40, 0.55], "position_jitter": 0.03},
            "test": {"object_scale_range": [0.70, 0.90], "position_jitter": 0.25},
        },
    }
    cond = resolve_condition(cfg)
    # Same seed and split on both sides: paired samples that differ only in
    # the distribution parameters.
    indist = SynthShapeDataset(classes=["house"], n_per_class=5, cfg=cond.train_cfg, seed=12345, split="test")
    shifted = SynthShapeDataset(classes=["house"], n_per_class=5, cfg=cond.test_cfg, seed=12345, split="test")

    def width(ds, i):
        mask = np.asarray(ds[i].image).sum(axis=2) < 700
        xs = np.nonzero(mask)[1]
        return xs.max() - xs.min() + 1

    for i in range(5):
        assert not np.array_equal(np.asarray(indist[i].image), np.asarray(shifted[i].image))
        # The scale shift is roughly a doubling; require a clear margin.
        assert width(shifted, i) > 1.3 * width(indist, i)
