import numpy as np
import pytest

from shapeprim.data.primitives import Primitive
from shapeprim.graph.build import NUM_EDGE_FEATURES, NUM_NODE_FEATURES, build_graph


def _sample_primitives():
    return [
        Primitive(type="circle", cx=20, cy=20, width=10, height=10, color=(1, 1, 1)),
        Primitive(type="rectangle", cx=50, cy=25, width=20, height=8, color=(1, 1, 1)),
        Primitive(type="triangle", cx=35, cy=60, width=15, height=12, rotation=0.3, color=(1, 1, 1)),
    ]


def test_empty_primitive_list():
    g = build_graph([])
    assert g.num_nodes == 0
    assert g.node_features.shape == (0, NUM_NODE_FEATURES)
    assert g.edge_index.shape == (2, 0)
    assert g.edge_features.shape == (0, NUM_EDGE_FEATURES)


def test_single_primitive_has_no_edges():
    g = build_graph([Primitive(type="circle", cx=0, cy=0, width=4, height=4, color=(1, 1, 1))])
    assert g.num_nodes == 1
    assert g.num_edges == 0
    assert g.node_features.shape == (1, NUM_NODE_FEATURES)


def test_fully_connected_directed_edge_count():
    prims = _sample_primitives()
    g = build_graph(prims, label="widget")
    assert g.num_nodes == 3
    assert g.num_edges == 3 * 2  # N*(N-1)
    assert g.label == "widget"
    assert g.node_types == ["circle", "rectangle", "triangle"]


def test_node_type_onehot_matches_primitive_type():
    from shapeprim.data.primitives import PRIMITIVE_TYPES

    prims = _sample_primitives()
    g = build_graph(prims)
    for i, p in enumerate(prims):
        onehot = g.node_features[i, : len(PRIMITIVE_TYPES)]
        assert onehot.sum() == 1
        assert PRIMITIVE_TYPES[int(np.argmax(onehot))] == p.type


def test_translation_invariance():
    prims = _sample_primitives()
    shifted = [
        Primitive(type=p.type, cx=p.cx + 100, cy=p.cy - 40, width=p.width, height=p.height,
                  rotation=p.rotation, color=p.color)
        for p in prims
    ]
    g1 = build_graph(prims)
    g2 = build_graph(shifted)
    np.testing.assert_allclose(g1.node_features, g2.node_features, atol=1e-5)
    np.testing.assert_allclose(g1.edge_features, g2.edge_features, atol=1e-5)


def test_scale_invariance():
    prims = _sample_primitives()
    factor = 3.0
    scaled = [
        Primitive(type=p.type, cx=p.cx * factor, cy=p.cy * factor, width=p.width * factor,
                  height=p.height * factor, rotation=p.rotation, color=p.color)
        for p in prims
    ]
    g1 = build_graph(prims)
    g2 = build_graph(scaled)
    np.testing.assert_allclose(g1.node_features, g2.node_features, atol=1e-4)
    np.testing.assert_allclose(g1.edge_features, g2.edge_features, atol=1e-4)


def test_above_below_left_right_flags_are_consistent():
    left = Primitive(type="circle", cx=0, cy=0, width=4, height=4, color=(1, 1, 1))
    right = Primitive(type="circle", cx=10, cy=0, width=4, height=4, color=(1, 1, 1))
    g = build_graph([left, right])
    # edge_index columns: (0,1) = left->right, (1,0) = right->left
    assert list(g.edge_index[:, 0]) == [0, 1]
    left_to_right = g.edge_features[0]
    right_to_left = g.edge_features[1]
    # feature order: dx, dy, distance, size_ratio, sin, cos, above, below, left, right, inside
    assert left_to_right[9] == 1.0  # right is to the right of left
    assert left_to_right[8] == 0.0
    assert right_to_left[8] == 1.0  # left is to the left of right
    assert right_to_left[9] == 0.0


def test_inside_flag_for_contained_primitive():
    # edge i->j's "inside" flag means: is the destination (j) inside the source (i)?
    outer = Primitive(type="rectangle", cx=0, cy=0, width=40, height=40, color=(1, 1, 1))
    inner = Primitive(type="rectangle", cx=2, cy=2, width=10, height=10, color=(1, 1, 1))
    g = build_graph([outer, inner])
    assert list(g.edge_index[:, 0]) == [0, 1]  # outer(0) -> inner(1)
    outer_to_inner = g.edge_features[0]
    inner_to_outer = g.edge_features[1]
    assert outer_to_inner[10] == 1.0  # inner (destination) is inside outer (source)
    assert inner_to_outer[10] == 0.0  # outer (destination) is not inside inner (source)
