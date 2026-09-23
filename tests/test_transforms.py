import math
import random

import numpy as np
import pytest

from shapeprim.data.objects import CLASS_NAMES
from shapeprim.data.primitives import Primitive, polygon_pose
from shapeprim.data.synth_dataset import GenerationConfig, SynthShapeDataset, render_sample
from shapeprim.data.transforms import ViewSample, apply_homography, transform_primitive, view_homography
from shapeprim.extract.eval_match import iou, primitive_mask
from shapeprim.graph.build import affine_canonical, build_graph

VIEW_CFG = dict(view_angle_range=(0, 70), rotation_range=(0, 45), shear_range=(0, 0.3), squash_range=(0.8, 1.2))


def _identity(pts):
    return np.asarray(pts, dtype=float)


@pytest.mark.parametrize("ptype", ["triangle", "rectangle", "line", "circle"])
def test_identity_transform_preserves_the_primitive(ptype):
    p = Primitive(type=ptype, cx=50, cy=60, width=30, height=20 if ptype != "circle" else 30, rotation=0.4 if ptype != "circle" else 0.0)
    q = transform_primitive(p, _identity)
    assert iou(p, q, 128) > 0.97
    if ptype != "circle":
        assert (q.cx, q.cy, q.width, q.height) == pytest.approx((p.cx, p.cy, p.width, p.height), abs=1e-6)
        assert math.remainder(q.rotation - p.rotation, 2 * math.pi) == pytest.approx(0.0, abs=1e-6)


def test_frontal_view_is_the_identity_homography():
    h = view_homography(ViewSample())
    assert np.allclose(h, np.eye(3))


def test_circle_under_perspective_becomes_a_foreshortened_ellipse():
    c = Primitive(type="circle", cx=0, cy=0, width=0.4, height=0.4)
    h = view_homography(ViewSample(view_angle=60, azimuth=0))  # tilt about the x axis
    e = transform_primitive(c, lambda pts: apply_homography(h, pts))
    assert e.type == "ellipse"
    minor, major = sorted([e.width, e.height])
    # Foreshortening by roughly cos(60) = 0.5 along the tilted direction.
    assert 0.4 < minor / major < 0.6


def test_rectangle_under_perspective_is_a_quadrilateral_with_exact_vertices():
    r = Primitive(type="rectangle", cx=0, cy=0, width=0.5, height=0.3)
    h = view_homography(ViewSample(view_angle=50, azimuth=30))
    q = transform_primitive(r, lambda pts: apply_homography(h, pts))
    assert q.type == "quadrilateral" and len(q.vertices) == 4
    assert np.allclose(q.polygon(), apply_homography(h, np.array(r.polygon())))


def test_polygon_pose_triangle_convention():
    # Apex straight up: rotation 0; apex to the right: rotation +pi/2.
    assert polygon_pose("triangle", [(0, -1), (1, 1), (-1, 1)])[4] == pytest.approx(0.0)
    assert polygon_pose("triangle", [(1, 0), (-1, 1), (-1, -1)])[4] == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("seed", range(30))
def test_ground_truth_matches_rendered_pixels_under_transforms(seed):
    cfg = GenerationConfig(image_size=128, **VIEW_CFG)
    s = render_sample(CLASS_NAMES[seed % 10], random.Random(seed), cfg)
    fg = np.abs(np.asarray(s.image).astype(int) - 255).sum(axis=2) > 60
    gt = np.zeros_like(fg)
    for p in s.primitives:
        gt |= primitive_mask(p, 128)
    # Every drawn pixel belongs to some ground-truth primitive (the white
    # outlines make the converse loose, so it is only a floor).
    assert (fg & ~gt).sum() <= 0.03 * fg.sum()
    assert (fg & gt).sum() >= 0.35 * gt.sum()


def test_view_types_are_the_phase2_vocabulary():
    cfg = GenerationConfig(image_size=64, view_angle_range=(10, 20))
    types = {p.type for i in range(20) for p in SynthShapeDataset(n_per_class=2, cfg=cfg, seed=0)[i].primitives}
    assert types <= {"ellipse", "triangle", "quadrilateral", "line"}


def test_view_disabled_leaves_phase1_data_unchanged():
    a = SynthShapeDataset(n_per_class=3, cfg=GenerationConfig(image_size=64), seed=0, split="test")
    b = SynthShapeDataset(n_per_class=3, cfg=GenerationConfig.from_dict({"image_size": 64, "view_angle_range": [0, 0]}), seed=0, split="test")
    for i in range(len(a)):
        assert np.array_equal(np.asarray(a[i].image), np.asarray(b[i].image))


def test_affine_frame_removes_affine_maps_up_to_rotation():
    s = render_sample("house", random.Random(0), GenerationConfig(image_size=128))
    a = np.array([[1.3, 0.5], [-0.2, 0.7]])
    moved = [transform_primitive(p, lambda pts: np.asarray(pts) @ a.T + 10) for p in s.primitives]
    c1 = np.array([[p.cx, p.cy] for p in affine_canonical(s.primitives)])
    c2 = np.array([[p.cx, p.cy] for p in affine_canonical(moved)])
    d1 = np.linalg.norm(c1[:, None] - c1[None], axis=-1)
    d2 = np.linalg.norm(c2[:, None] - c2[None], axis=-1)
    assert np.allclose(d1, d2, atol=0.02)


def test_graph_one_hot_is_shared_across_vocabularies():
    circle = build_graph([Primitive(type="circle", cx=0, cy=0, width=4, height=4)])
    ellipse = build_graph([Primitive(type="ellipse", cx=0, cy=0, width=4, height=4)])
    assert np.array_equal(circle.node_features[0, :4], ellipse.node_features[0, :4])
