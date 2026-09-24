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

Pooling is configurable (``pooling``):

- ``mean`` (the default, and what every result before this option used):
  average of node embeddings. An extra or missing part shifts the whole
  average, which is what the novel-composition test punished (0.49
  accuracy with oracle primitives against 0.86 for the set transformer).
- ``max``: per-feature maximum over nodes. A part that adds nothing new
  leaves it unchanged, so one extra window or a hat should matter less.
- ``attention``: gated attention pooling (Li et al. 2016; Ilse et al.
  2018): a learned score per node, softmax over the graph, weighted sum.
  The model can learn to down-weight parts that do not identify the class.
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


def _max_pool(x: torch.Tensor, node_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    pooled = torch.full((num_graphs, x.size(-1)), float("-inf"), device=x.device, dtype=x.dtype)
    if x.size(0) > 0:
        pooled = pooled.scatter_reduce(0, node_batch.unsqueeze(-1).expand_as(x), x, reduce="amax", include_self=True)
    # A graph with no nodes (the extractor found nothing) pools to zeros.
    return torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)


def _attention_pool(x, scores, node_batch, num_graphs) -> torch.Tensor:
    """Softmax of ``scores`` within each graph, then a weighted sum."""
    if x.size(0) == 0:
        return x.new_zeros(num_graphs, x.size(-1))
    s = scores.squeeze(-1)
    smax = torch.full((num_graphs,), float("-inf"), device=x.device, dtype=x.dtype)
    smax = smax.scatter_reduce(0, node_batch, s, reduce="amax", include_self=True)
    w = torch.exp(s - smax[node_batch])
    denom = torch.zeros(num_graphs, device=x.device, dtype=x.dtype).index_add_(0, node_batch, w)
    w = w / denom[node_batch].clamp(min=1e-12)
    return torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype).index_add_(0, node_batch, x * w.unsqueeze(-1))


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
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in ("mean", "max", "attention"):
            raise ValueError(f"unknown pooling {pooling!r}; known: mean, max, attention")
        self.pooling = pooling
        if pooling == "attention":
            self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
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
        if self.pooling == "max":
            pooled = _max_pool(x, node_batch, num_graphs)
        elif self.pooling == "attention":
            pooled = _attention_pool(x, self.gate(x), node_batch, num_graphs) if x.size(0) else x.new_zeros(num_graphs, x.size(-1))
        else:
            pooled = _mean_pool(x, node_batch, num_graphs)
        return self.head(pooled)
