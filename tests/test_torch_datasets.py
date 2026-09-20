import torch
from torch.utils.data import DataLoader

from shapeprim.data.objects import CLASS_NAMES
from shapeprim.data.synth_dataset import GenerationConfig
from shapeprim.data.torch_datasets import (
    CLASS_TO_IDX,
    GraphClassificationDataset,
    ImageClassificationDataset,
    collate_graphs,
)
from shapeprim.extract.oracle import OracleExtractor
from shapeprim.graph.build import NUM_EDGE_FEATURES, NUM_NODE_FEATURES


def test_class_to_idx_covers_all_classes():
    assert set(CLASS_TO_IDX.keys()) == set(CLASS_NAMES)
    assert sorted(CLASS_TO_IDX.values()) == list(range(len(CLASS_NAMES)))


def test_image_dataset_shapes():
    cfg = GenerationConfig(image_size=48)
    ds = ImageClassificationDataset(classes=["house", "fish"], n_per_class=3, cfg=cfg, seed=0)
    assert len(ds) == 6
    image, label = ds[0]
    assert image.shape == (3, 48, 48)
    assert image.dtype == torch.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert isinstance(label, int)


def test_image_dataset_with_dataloader_batches():
    cfg = GenerationConfig(image_size=32)
    ds = ImageClassificationDataset(classes=["house", "fish"], n_per_class=4, cfg=cfg, seed=0)
    loader = DataLoader(ds, batch_size=3)
    images, labels = next(iter(loader))
    assert images.shape == (3, 3, 32, 32)
    assert labels.shape == (3,)


def test_graph_dataset_with_oracle():
    cfg = GenerationConfig(image_size=48, distractor_prob=0.0)
    ds = GraphClassificationDataset(OracleExtractor(), classes=["house"], n_per_class=2, cfg=cfg, seed=0)
    graph, label = ds[0]
    assert graph.num_nodes == 4  # house template: body, roof, door, window
    # Indexed against this dataset's own class list, so a single-class
    # dataset emits 0 -- not CLASS_TO_IDX["house"] (4), which would run
    # past the output layer of a classifier built for these classes.
    assert label == 0
    assert ds.class_to_idx == {"house": 0}


def test_full_vocabulary_dataset_matches_the_global_index():
    cfg = GenerationConfig(image_size=32, distractor_prob=0.0)
    ds = GraphClassificationDataset(OracleExtractor(), n_per_class=1, cfg=cfg, seed=0)
    assert ds.class_to_idx == CLASS_TO_IDX


def test_collate_graphs_batches_variable_sized_graphs():
    cfg = GenerationConfig(image_size=48, distractor_prob=0.0)
    ds = GraphClassificationDataset(OracleExtractor(), classes=["tree", "house"], n_per_class=2, cfg=cfg, seed=0)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_graphs)
    node_features, edge_index, edge_features, node_batch, num_graphs, labels = next(iter(loader))

    assert num_graphs == 4
    assert labels.shape == (4,)
    assert node_features.shape[1] == NUM_NODE_FEATURES
    assert edge_features.shape[1] == NUM_EDGE_FEATURES
    assert node_batch.shape[0] == node_features.shape[0]
    assert edge_index.max().item() < node_features.shape[0]
    # each graph contributes N*(N-1) edges; node counts per graph come from node_batch
    counts = torch.bincount(node_batch, minlength=num_graphs)
    expected_edges = int((counts * (counts - 1)).sum().item())
    assert edge_index.shape[1] == expected_edges


def test_subset_labels_are_indexed_against_the_subset():
    """A dataset restricted to a subset must emit 0..k-1, not global indices.

    Indexing against the full 10-class vocabulary made a 3-class run emit
    label 4 into a 3-output classifier, which surfaces as
    "IndexError: Target 4 is out of bounds" inside the loss rather than
    anywhere near the cause.
    """
    from shapeprim.data.torch_datasets import GraphClassificationDataset, ImageClassificationDataset
    from shapeprim.extract.oracle import OracleExtractor

    subset = ["house", "tree", "arrow_sign"]
    cfg = GenerationConfig(image_size=32, distractor_prob=0.0)

    images = ImageClassificationDataset(classes=subset, n_per_class=2, cfg=cfg, seed=0, split="train")
    labels = {images[i][1] for i in range(len(images))}
    assert labels == {0, 1, 2}

    graphs = GraphClassificationDataset(
        OracleExtractor(), classes=subset, n_per_class=2, cfg=cfg, seed=0, split="train"
    )
    assert {graphs[i][1] for i in range(len(graphs))} == {0, 1, 2}


def test_image_and_graph_datasets_agree_on_labels():
    from shapeprim.data.torch_datasets import GraphClassificationDataset, ImageClassificationDataset
    from shapeprim.extract.oracle import OracleExtractor

    subset = ["fish", "snowman"]
    cfg = GenerationConfig(image_size=32, distractor_prob=0.0)
    images = ImageClassificationDataset(classes=subset, n_per_class=3, cfg=cfg, seed=1, split="train")
    graphs = GraphClassificationDataset(
        OracleExtractor(), classes=subset, n_per_class=3, cfg=cfg, seed=1, split="train"
    )
    assert [images[i][1] for i in range(len(images))] == [graphs[i][1] for i in range(len(graphs))]
