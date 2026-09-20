"""Message-passing GNN classifier over ShapeGraph batches.

Plain PyTorch (no PyTorch Geometric dependency) -- our graphs are small
(a handful of nodes, fully connected) so hand-rolled scatter/index_add
message passing is simple and fast enough.

Each layer is a simplified GINE (Graph Isomorphism Network with Edge
features): a message from node i to node j is an MLP over
[node_i_features, edge_ij_features], messages are summed per destination
node, and combined with the destination's own features through a second
MLP (with a learned self-weight, as in GIN). Followed by global mean
pooling and an MLP head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GINELayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return F.relu(self.update_mlp((1 + self.eps) * x))
        src, dst = edge_index[0], edge_index[1]
        messages = self.message_mlp(torch.cat([x[src], edge_attr], dim=-1))
        aggregated = torch.zeros_like(x).index_add_(0, dst, messages)
        return F.relu(self.update_mlp((1 + self.eps) * x + aggregated))


def _mean_pool(x: torch.Tensor, node_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    dim = x.size(-1)
    pooled = torch.zeros(num_graphs, dim, device=x.device, dtype=x.dtype)
    if x.size(0) == 0:
        return pooled
    pooled.index_add_(0, node_batch, x)
    counts = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)
    counts.index_add_(0, node_batch, torch.ones(x.size(0), device=x.device, dtype=x.dtype))
    return pooled / counts.clamp(min=1).unsqueeze(-1)


class GNNClassifier(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)
        self.layers = nn.ModuleList([GINELayer(hidden_dim, edge_in_dim, hidden_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        x = F.relu(self.input_proj(node_features)) if node_features.size(0) > 0 else node_features.new_zeros(0, self.input_proj.out_features)
        for layer in self.layers:
            x = x + layer(x, edge_index, edge_features)
        pooled = _mean_pool(x, node_batch, num_graphs)
        return self.head(pooled)
