"""Bag-of-primitives ablation (PLAN.md section 4): no positions, no relations.

The model sees, per primitive type, how many primitives of that type the
extractor found and their mean size and mean log-aspect ratio -- and
nothing else. It consumes the same batches as the GNN (``collate_graphs``)
and throws away positions, rotations and every edge, so any accuracy it
reaches is accuracy that does not need arrangement. The gap between it and
the GNN on the relation-twin pair (tree vs arrow_sign) is the evidence for
H2 ("relations matter").

Size and aspect are read from the graph's node features, which are already
normalised by the object's bounding box, so the bag is translation- and
scale-invariant exactly like the graph models.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..graph.build import NUM_TYPES, SIZE_INDEX, ASPECT_INDEX

BAG_FEATURES_PER_TYPE = 3  # count, mean size, mean log-aspect


def bag_features(node_features: torch.Tensor, node_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """(num_graphs, NUM_TYPES * 3) orderless summary of each graph's nodes."""
    out = node_features.new_zeros(num_graphs, NUM_TYPES, BAG_FEATURES_PER_TYPE)
    if node_features.size(0) == 0:
        return out.flatten(1)
    types = node_features[:, :NUM_TYPES]  # one-hot
    size = node_features[:, SIZE_INDEX : SIZE_INDEX + 1]
    log_aspect = torch.log(node_features[:, ASPECT_INDEX : ASPECT_INDEX + 1].clamp(min=1e-3))
    per_node = torch.stack([types, types * size, types * log_aspect], dim=-1)  # (N, T, 3)
    out.index_add_(0, node_batch, per_node)
    counts = out[:, :, 0:1]
    means = out[:, :, 1:] / counts.clamp(min=1.0)
    return torch.cat([counts, means], dim=-1).flatten(1)


class BagClassifier(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 64, dropout: float = 0.1, **_ignored):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(NUM_TYPES * BAG_FEATURES_PER_TYPE, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, node_features, edge_index, edge_features, node_batch, num_graphs):
        return self.mlp(bag_features(node_features, node_batch, num_graphs))
