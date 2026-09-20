import numpy as np
import pytest
from PIL import Image

from shapeprim.data.augment import (
    MAX_SAFE_DEGREES,
    AugmentConfig,
    build_transform,
    resolve_augment,
)


def test_default_config_is_identity():
    cfg = AugmentConfig()
    assert cfg.is_identity
    assert build_transform(cfg) is None


def test_enabled_but_empty_is_still_identity():
    # `enabled: true` with no actual transform configured must not silently
    # build a no-op pipeline that costs time per sample.
    assert AugmentConfig(enabled=True).is_identity


def test_rejects_rotation_that_confuses_the_relation_twins():
    # A 180-degree rotation maps tree onto arrow_sign, the exact pair the
    # dataset uses to test whether arrangement is used. Allowing it would
    # inject label noise into the measurement it is meant to support.
    with pytest.raises(ValueError, match="tree onto arrow_sign"):
        AugmentConfig(enabled=True, degrees=180.0)
    # At the cap it is still allowed.
    AugmentConfig(enabled=True, degrees=MAX_SAFE_DEGREES)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"translate": 1.5},
        {"translate": -0.1},
        {"scale_range": (1.5, 0.5)},
        {"scale_range": (0.0, 1.0)},
        {"degrees": -5.0},
        {"hflip_prob": 1.5},
    ],
)
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        AugmentConfig(enabled=True, **kwargs)


def test_from_dict_rejects_unknown_keys():
    # A typo in a YAML config must fail loudly rather than silently
    # training without the augmentation the experiment claims to use.
    with pytest.raises(ValueError, match="unknown AugmentConfig keys"):
        AugmentConfig.from_dict({"enabled": True, "translation": 0.2})


def test_resolve_accepts_preset_dict_and_instance():
    assert resolve_augment("standard").enabled
    assert resolve_augment(None).is_identity
    assert resolve_augment({"enabled": True, "translate": 0.1}).translate == 0.1
    cfg = AugmentConfig(enabled=True, translate=0.2)
    assert resolve_augment(cfg) is cfg


def test_resolve_preset_returns_a_copy():
    # Mutating a resolved preset must not poison later lookups.
    first = resolve_augment("standard")
    first.translate = 0.99
    assert resolve_augment("standard").translate != 0.99


def test_unknown_preset_name_raises():
    with pytest.raises(ValueError, match="unknown augmentation preset"):
        resolve_augment("aggressive")


def test_to_dict_round_trip():
    cfg = AugmentConfig(enabled=True, translate=0.15, scale_range=(0.8, 1.2), degrees=10.0)
    assert AugmentConfig.from_dict(cfg.to_dict()) == cfg


def _corner_pixels(img: Image.Image, k: int = 3):
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    return np.concatenate([arr[:k, :k].reshape(-1, 3), arr[-k:, -k:].reshape(-1, 3)])


def test_affine_fills_with_background_not_black():
    """The revealed corners must be background, not torchvision's default 0.

    A black fill stamps a high-contrast frame onto every augmented image
    whose shape encodes the sampled transform -- a feature the CNN can read
    that has nothing to do with the object.
    """
    cfg = AugmentConfig(enabled=True, translate=0.3, scale_range=(0.5, 0.6))
    transform = build_transform(cfg, background=(255, 255, 255))
    assert transform is not None

    img = Image.new("RGB", (64, 64), (255, 255, 255))
    for _ in range(20):
        out = transform(img)
        assert out.size == (64, 64)
        assert _corner_pixels(out).min() > 200, "revealed area is not background-colored"


def test_augmentation_actually_moves_the_image():
    cfg = AugmentConfig(enabled=True, translate=0.3)
    transform = build_transform(cfg, background=(255, 255, 255))
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    img.paste((0, 0, 0), (28, 28, 36, 36))

    base = np.asarray(img, dtype=np.int16)
    assert any(np.abs(np.asarray(transform(img), dtype=np.int16) - base).sum() > 0 for _ in range(20))


def test_non_white_background_is_respected():
    transform = build_transform(
        AugmentConfig(enabled=True, translate=0.4), background=(10, 20, 30)
    )
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    for _ in range(10):
        corners = _corner_pixels(transform(img))
        assert np.abs(corners - np.array([10, 20, 30])).max() <= 12
