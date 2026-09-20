"""The train/val/test splits must not overlap, or every number is inflated."""

import numpy as np

from shapeprim.data.synth_dataset import GenerationConfig, SynthShapeDataset

CFG = GenerationConfig(image_size=48, distractor_prob=0.0)


def _fingerprints(split: str, seed: int = 0, n: int = 6):
    ds = SynthShapeDataset(classes=["house", "tree"], n_per_class=n, cfg=CFG, seed=seed, split=split)
    return {np.asarray(ds[i].image).tobytes() for i in range(len(ds))}


def test_splits_from_the_same_seed_are_disjoint():
    train, val, test = _fingerprints("train"), _fingerprints("val"), _fingerprints("test")
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_split_is_reproducible():
    assert _fingerprints("train", seed=0) == _fingerprints("train", seed=0)


def test_different_seeds_give_different_training_draws():
    assert _fingerprints("train", seed=0) != _fingerprints("train", seed=1)


def test_empty_split_reproduces_legacy_seeding():
    """Data generated before `split` existed must be unchanged.

    Committed artifacts (results/phase1/sample_grid.png) and the old
    results file were produced without a split, so the default path has to
    stay bit-identical or they silently stop corresponding to the code.
    """
    legacy = SynthShapeDataset(classes=["fish"], n_per_class=3, cfg=CFG, seed=7)
    explicit = SynthShapeDataset(classes=["fish"], n_per_class=3, cfg=CFG, seed=7, split="")
    for i in range(len(legacy)):
        assert np.array_equal(np.asarray(legacy[i].image), np.asarray(explicit[i].image))


def test_split_labels_are_preserved():
    ds = SynthShapeDataset(classes=["house", "tree"], n_per_class=4, cfg=CFG, seed=3, split="val")
    assert [ds[i].label for i in range(len(ds))] == ["house"] * 4 + ["tree"] * 4


def test_split_parameter_fixes_the_old_cross_seed_leak():
    """Documents the failure the `split` parameter exists to prevent.

    The previous convention drew train from `seed` and test from `seed + 1`.
    Run the sweep over seeds 0, 1, 2 and every test image of run 0 is a
    training image of run 1: invisible within any single run, and fatal
    across the multi-seed sweep the protocol requires.
    """
    cfg = GenerationConfig(image_size=64, distractor_prob=0.0)
    train_of_run1 = SynthShapeDataset(classes=["house"], n_per_class=150, cfg=cfg, seed=1)
    test_of_run0 = SynthShapeDataset(classes=["house"], n_per_class=40, cfg=cfg, seed=1)

    train_imgs = {np.asarray(train_of_run1[i].image).tobytes() for i in range(len(train_of_run1))}
    test_imgs = {np.asarray(test_of_run0[i].image).tobytes() for i in range(len(test_of_run0))}
    assert len(train_imgs & test_imgs) == len(test_imgs), "expected the documented total leak"

    # With named splits the same base seeds share nothing.
    train_split = SynthShapeDataset(classes=["house"], n_per_class=150, cfg=cfg, seed=1, split="train")
    test_split = SynthShapeDataset(classes=["house"], n_per_class=40, cfg=cfg, seed=0, split="test")
    a = {np.asarray(train_split[i].image).tobytes() for i in range(len(train_split))}
    b = {np.asarray(test_split[i].image).tobytes() for i in range(len(test_split))}
    assert not (a & b)
