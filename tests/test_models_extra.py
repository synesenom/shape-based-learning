import torch

from shapeprim.data.objects import CLASS_NAMES, CLASS_TEMPLATES, NOVEL_VARIANTS
from shapeprim.data.synth_dataset import GenerationConfig, SynthShapeDataset
from shapeprim.data.torch_datasets import GraphClassificationDataset, collate_graphs
from shapeprim.extract.oracle import OracleExtractor
from shapeprim.graph.build import NUM_NODE_FEATURES
from shapeprim.models.bag import bag_features, BagClassifier
from shapeprim.models.cnn import build_cnn
from shapeprim.models.settransformer import SetTransformerClassifier, to_padded


def _batch(n=6, classes=("tree", "arrow_sign", "car")):
    ds = GraphClassificationDataset(OracleExtractor(), classes=classes, n_per_class=n, cfg=GenerationConfig(image_size=64))
    return collate_graphs([ds[i] for i in range(len(ds))])


def test_bag_ignores_positions_and_edges():
    nf, ei, ef, nb, ng, _ = _batch()
    model = BagClassifier(num_classes=3).eval()
    moved = nf.clone()
    moved[:, -2:] += torch.randn_like(moved[:, -2:])  # positions
    moved[:, -4:-2] = torch.randn_like(moved[:, -4:-2])  # rotation
    with torch.no_grad():
        a = model(nf, ei, ef, nb, ng)
        b = model(moved, ei[:, :0], ef[:0], nb, ng)
    assert torch.allclose(a, b, atol=1e-6)


def test_bag_counts_types():
    nf, _, _, nb, ng, labels = _batch(n=1, classes=("car",))
    feats = bag_features(nf, nb, ng).view(ng, -1, 3)
    # car: 2 circles, 3 rectangles
    assert feats[0, 0, 0].item() == 2 and feats[0, 2, 0].item() == 3


def test_settransformer_shape_and_permutation_invariance():
    torch.manual_seed(0)
    nf, ei, ef, nb, ng, _ = _batch()
    model = SetTransformerClassifier(NUM_NODE_FEATURES, num_classes=3).eval()
    with torch.no_grad():
        out = model(nf, ei, ef, nb, ng)
        assert out.shape == (ng, 3)
        # Permute nodes within the first graph.
        n0 = int((nb == 0).sum())
        perm = torch.arange(nf.size(0))
        perm[:n0] = torch.randperm(n0)
        out2 = model(nf[perm], ei, ef, nb, ng)
    assert torch.allclose(out, out2, atol=1e-5)


def test_to_padded_roundtrip():
    nf = torch.arange(10.0).view(5, 2)
    nb = torch.tensor([0, 0, 1, 1, 1])
    padded, mask = to_padded(nf, nb, 2)
    assert padded.shape == (2, 3, 2)
    assert mask.tolist() == [[False, False, True], [False, False, False]]
    assert torch.equal(padded[~mask], nf)


def test_cnn_stem_stride_two():
    m = build_cnn(10, stem_stride=2)
    assert m(torch.rand(2, 3, 64, 64)).shape == (2, 10)


def test_every_class_has_novel_variants_that_differ_from_base():
    for name in CLASS_NAMES:
        assert NOVEL_VARIANTS[name], name
        base = sorted(p.name for p in CLASS_TEMPLATES[name].parts)
        for v in NOVEL_VARIANTS[name]:
            assert sorted(p.name for p in v.parts) != base, v.name


def test_novel_template_set_renders_variants_and_base_is_unchanged():
    base_cfg = GenerationConfig(image_size=64)
    novel_cfg = GenerationConfig(image_size=64, template_set="novel")
    base = SynthShapeDataset(classes=["car"], n_per_class=8, cfg=base_cfg, seed=0)
    novel = SynthShapeDataset(classes=["car"], n_per_class=8, cfg=novel_cfg, seed=0)
    assert all(len(base[i].primitives) == 5 for i in range(8))
    assert any(len(novel[i].primitives) != 5 for i in range(8))


import pytest
from shapeprim.graph.build import NUM_EDGE_FEATURES
from shapeprim.models.gnn import GNNClassifier


@pytest.mark.parametrize("pooling", ["mean", "max", "attention"])
def test_gnn_pooling_shapes_and_permutation_invariance(pooling):
    torch.manual_seed(0)
    nf, ei, ef, nb, ng, _ = _batch()
    model = GNNClassifier(NUM_NODE_FEATURES, NUM_EDGE_FEATURES, num_classes=3, hidden_dim=16, num_layers=2, pooling=pooling).eval()
    with torch.no_grad():
        out = model(nf, ei, ef, nb, ng)
        assert out.shape == (ng, 3)
        # Reverse the node order of the whole batch, remapping edges.
        n = nf.size(0)
        perm = torch.arange(n - 1, -1, -1)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(n)
        out2 = model(nf[perm], inv[ei], ef, nb[perm], ng)
    assert torch.allclose(out, out2, atol=1e-5)


@pytest.mark.parametrize("pooling", ["mean", "max", "attention"])
def test_gnn_pooling_empty_graph(pooling):
    model = GNNClassifier(NUM_NODE_FEATURES, NUM_EDGE_FEATURES, num_classes=3, hidden_dim=16, pooling=pooling).eval()
    out = model(torch.zeros(0, NUM_NODE_FEATURES), torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, NUM_EDGE_FEATURES), torch.zeros(0, dtype=torch.long), 2)
    assert out.shape == (2, 3) and torch.isfinite(out).all()
