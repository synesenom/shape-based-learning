"""Generic training/evaluation loop shared by every model.

Different models consume different batch shapes (a CNN wants an image
tensor, a GNN wants the disjoint-union graph tensors from
``collate_graphs``), so each model brings a small ``forward_fn(model,
batch, device) -> (logits, labels)`` adapter; the training loop itself
doesn't know or care what kind of model it's driving.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ForwardFn = Callable[[torch.nn.Module, tuple, str], Tuple[torch.Tensor, torch.Tensor]]


def cnn_forward(model: torch.nn.Module, batch, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    return model(images), labels


def gnn_forward(model: torch.nn.Module, batch, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    node_features, edge_index, edge_features, node_batch, num_graphs, labels = batch
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    edge_features = edge_features.to(device)
    node_batch = node_batch.to(device)
    labels = labels.to(device)
    logits = model(node_features, edge_index, edge_features, node_batch, num_graphs)
    return logits, labels


@torch.no_grad()
def evaluate_classifier(model: torch.nn.Module, forward_fn: ForwardFn, loader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        logits, labels = forward_fn(model, batch, device)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def train_classifier(
    model: torch.nn.Module,
    forward_fn: ForwardFn,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str = "cpu",
    weight_decay: float = 0.0,
) -> List[dict]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, labels = forward_fn(model, batch, device)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
        train_loss = total_loss / n if n else 0.0
        test_acc = evaluate_classifier(model, forward_fn, test_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_acc": test_acc})
    return history
