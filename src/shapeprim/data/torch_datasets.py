"""torch.utils.data.Dataset wrappers around SynthShapeDataset.

``ImageClassificationDataset`` feeds the CNN baseline raw pixels.
``GraphClassificationDataset`` runs a primitive extractor on the rendered
image and turns the result into a ``ShapeGraph`` for the GNN. Both share
the same underlying rendering (same seed -> same image), so a CNN run and
a GNN run trained with matching seeds see the same images.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..extract.base import PrimitiveExtractor
from ..graph.build import NUM_EDGE_FEATURES, NUM_NODE_FEATURES, ShapeGraph, build_graph
from .objects import CLASS_NAMES
from .synth_dataset import GenerationConfig, SynthShapeDataset

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        classes: Sequence[str] = CLASS_NAMES,
        n_per_class: int = 100,
        cfg: Optional[GenerationConfig] = None,
        seed: int = 0,
    ):
        self.inner = SynthShapeDataset(classes=classes, n_per_class=n_per_class, cfg=cfg, seed=seed)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.inner[idx]
        arr = np.asarray(sample.image.convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (C, H, W)
        return image, CLASS_TO_IDX[sample.label]


class GraphClassificationDataset(Dataset):
    def __init__(
        self,
        extractor: PrimitiveExtractor,
        classes: Sequence[str] = CLASS_NAMES,
        n_per_class: int = 100,
        cfg: Optional[GenerationConfig] = None,
        seed: int = 0,
    ):
        self.inner = SynthShapeDataset(classes=classes, n_per_class=n_per_class, cfg=cfg, seed=seed)
        self.extractor = extractor

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> Tuple[ShapeGraph, int]:
        sample = self.inner[idx]
        primitives = self.extractor.extract(sample.image, ground_truth=sample.primitives)
        graph = build_graph(primitives, label=sample.label)
        return graph, CLASS_TO_IDX[sample.label]


def collate_graphs(batch: List[Tuple[ShapeGraph, int]]):
    """Merge a list of (ShapeGraph, label) into one disjoint-union batch.

    Returns (node_features, edge_index, edge_features, node_batch, num_graphs, labels).
    ``node_batch[i]`` is the index of the graph node ``i`` belongs to -- the
    standard "batch vector" trick for pooling variable-sized graphs without
    padding.
    """
    graphs, labels = zip(*batch)
    num_graphs = len(graphs)

    node_feat_parts, batch_parts = [], []
    src_parts, dst_parts, edge_feat_parts = [], [], []
    offset = 0
    for gi, g in enumerate(graphs):
        n = g.num_nodes
        if n > 0:
            node_feat_parts.append(torch.from_numpy(g.node_features))
            batch_parts.append(torch.full((n,), gi, dtype=torch.long))
        if g.num_edges > 0:
            src_parts.append(torch.from_numpy(g.edge_index[0]) + offset)
            dst_parts.append(torch.from_numpy(g.edge_index[1]) + offset)
            edge_feat_parts.append(torch.from_numpy(g.edge_features))
        offset += n

    node_features = torch.cat(node_feat_parts, dim=0).float() if node_feat_parts else torch.zeros(0, NUM_NODE_FEATURES)
    node_batch = torch.cat(batch_parts, dim=0) if batch_parts else torch.zeros(0, dtype=torch.long)
    if src_parts:
        edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)
        edge_features = torch.cat(edge_feat_parts, dim=0).float()
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_features = torch.zeros(0, NUM_EDGE_FEATURES)

    return node_features, edge_index, edge_features, node_batch, num_graphs, torch.tensor(labels, dtype=torch.long)
