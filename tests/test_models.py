import torch

from shapeprim.data.objects import CLASS_NAMES
from shapeprim.graph.build import NUM_EDGE_FEATURES, NUM_NODE_FEATURES
from shapeprim.models.cnn import build_cnn
from shapeprim.models.gnn import GNNClassifier


def test_cnn_forward_shape():
    model = build_cnn(num_classes=len(CLASS_NAMES), pretrained=False)
    x = torch.rand(4, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (4, len(CLASS_NAMES))


def test_cnn_handles_small_images():
    # from-scratch stem (3x3 stride-1, no maxpool) must survive a 32x32 input
    model = build_cnn(num_classes=len(CLASS_NAMES), pretrained=False)
    x = torch.rand(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, len(CLASS_NAMES))


def test_gnn_forward_shape():
    model = GNNClassifier(NUM_NODE_FEATURES, NUM_EDGE_FEATURES, num_classes=len(CLASS_NAMES), hidden_dim=16, num_layers=2)
    num_nodes = 7
    node_features = torch.rand(num_nodes, NUM_NODE_FEATURES)
    # two graphs: first 4 nodes, last 3 nodes, fully connected within each
    node_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    src, dst = [], []
    for g, n in [(0, 4), (1, 3)]:
        base = 0 if g == 0 else 4
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(base + i)
                    dst.append(base + j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_features = torch.rand(edge_index.shape[1], NUM_EDGE_FEATURES)

    logits = model(node_features, edge_index, edge_features, node_batch, num_graphs=2)
    assert logits.shape == (2, len(CLASS_NAMES))


def test_gnn_handles_graph_with_no_edges():
    model = GNNClassifier(NUM_NODE_FEATURES, NUM_EDGE_FEATURES, num_classes=len(CLASS_NAMES), hidden_dim=16, num_layers=2)
    node_features = torch.rand(1, NUM_NODE_FEATURES)
    node_batch = torch.tensor([0])
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    edge_features = torch.zeros(0, NUM_EDGE_FEATURES)
    logits = model(node_features, edge_index, edge_features, node_batch, num_graphs=1)
    assert logits.shape == (1, len(CLASS_NAMES))


def test_gnn_is_permutation_invariant_within_a_graph():
    torch.manual_seed(0)
    model = GNNClassifier(NUM_NODE_FEATURES, NUM_EDGE_FEATURES, num_classes=len(CLASS_NAMES), hidden_dim=16, num_layers=2)
    model.eval()

    node_features = torch.rand(3, NUM_NODE_FEATURES)
    node_batch = torch.tensor([0, 0, 0])
    src = [0, 0, 1, 1, 2, 2]
    dst = [1, 2, 0, 2, 0, 1]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_features = torch.rand(6, NUM_EDGE_FEATURES)

    logits1 = model(node_features, edge_index, edge_features, node_batch, num_graphs=1)

    # new_node_features[k] = old node `perm[k]`; edges keep the same physical
    # attributes (edge_features unchanged) but their endpoints are relabeled
    # to the old nodes' new positions.
    perm = torch.tensor([2, 0, 1])
    inv_perm = torch.argsort(perm)
    node_features_p = node_features[perm]
    src_p = inv_perm[torch.tensor(src)]
    dst_p = inv_perm[torch.tensor(dst)]
    edge_index_p = torch.stack([src_p, dst_p])

    logits2 = model(node_features_p, edge_index_p, edge_features, node_batch, num_graphs=1)

    assert torch.allclose(logits1, logits2, atol=1e-5)
