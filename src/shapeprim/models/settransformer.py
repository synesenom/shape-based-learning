"""Set transformer over primitives (PLAN.md section 4).

A second relational architecture so that conclusions about the primitive
representation do not hinge on one GNN. Nodes attend to each other with
multi-head self-attention (SAB blocks, Lee et al. 2019) and a learned seed
pools them by attention (PMA). Relations are not given as edges here: each
node carries its bbox-normalised position, and attention has to discover
the pairwise structure itself. Edge features are accepted and ignored so
the model consumes the same batches as the GNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def to_padded(node_features: torch.Tensor, node_batch: torch.Tensor, num_graphs: int):
    """Disjoint-union nodes -> (B, N_max, F) with a True-means-padding mask."""
    feat_dim = node_features.size(-1)
    if node_features.size(0) == 0:
        return node_features.new_zeros(num_graphs, 1, feat_dim), torch.ones(
            num_graphs, 1, dtype=torch.bool, device=node_features.device
        )
    counts = torch.bincount(node_batch, minlength=num_graphs)
    n_max = max(int(counts.max().item()), 1)
    # Position of each node within its own graph.
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(node_features.size(0), device=node_features.device) - starts[node_batch]
    padded = node_features.new_zeros(num_graphs, n_max, feat_dim)
    padded[node_batch, within] = node_features
    mask = torch.ones(num_graphs, n_max, dtype=torch.bool, device=node_features.device)
    mask[node_batch, within] = False
    return padded, mask


class SetTransformerClassifier(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_dim, num_heads, dim_feedforward=2 * hidden_dim,
                    dropout=dropout, batch_first=True, norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.seed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.1)
        self.pool = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, node_features, edge_index, edge_features, node_batch, num_graphs):
        x, pad = to_padded(node_features, node_batch, num_graphs)
        empty = pad.all(dim=1)
        if empty.any():
            # An empty graph (extractor found nothing) would make attention
            # softmax over nothing; give it one zero "node" to attend to.
            pad = pad.clone()
            pad[empty, 0] = False
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, src_key_padding_mask=pad)
        seed = self.seed.expand(num_graphs, -1, -1)
        pooled, _ = self.pool(seed, h, h, key_padding_mask=pad)
        return self.head(pooled.squeeze(1))
