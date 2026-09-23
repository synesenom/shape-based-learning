"""Generic training/evaluation loop shared by every model.

Different models consume different batch shapes (a CNN wants an image
tensor, a GNN wants the disjoint-union graph tensors from
``collate_graphs``), so each model brings a small ``forward_fn(model,
batch, device) -> (logits, labels)`` adapter; the training loop itself
doesn't know or care what kind of model it's driving.

Protocol (WP0, experimental hygiene). The loop trains on ``train_loader``
and selects on ``val_loader``, and it never sees the test set. That
separation is the whole point: the previous version of this file evaluated
the *test* set after every epoch and reported the last epoch's number, so
every architecture and learning-rate decision made while watching that
curve leaked test information into the reported result, and a run that
happened to end on a bad epoch (the CNN's accuracy fell to 0.18 and 0.61
at two points in ``results/phase1/cnn_vs_gnn.json``) reported that dip as
its score. Here the best-on-validation weights are restored at the end and
the caller evaluates the test set exactly once.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

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


def resolve_device(device: Optional[str] = None) -> str:
    """'auto' / None -> cuda if available else cpu."""
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class TrainResult:
    history: List[dict] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = -1
    epochs_run: int = 0
    train_seconds: float = 0.0
    stopped_early: bool = False

    @property
    def final_train_loss(self) -> float:
        return self.history[-1]["train_loss"] if self.history else float("nan")


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module, forward_fn: ForwardFn, loader: DataLoader, device: str = "cpu"
) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        logits, labels = forward_fn(model, batch, device)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


@torch.no_grad()
def evaluate_loss_and_accuracy(
    model: torch.nn.Module, forward_fn: ForwardFn, loader: DataLoader, device: str = "cpu"
) -> Tuple[float, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for batch in loader:
        logits, labels = forward_fn(model, batch, device)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    if not total:
        return float("nan"), 0.0
    return loss_sum / total, correct / total


def build_optimizer(
    model: torch.nn.Module, name: str = "adam", lr: float = 1e-3, weight_decay: float = 0.0, momentum: float = 0.9
) -> torch.optim.Optimizer:
    name = name.lower()
    # Frozen parameters (a linear probe's backbone) are not handed to the
    # optimizer: weight decay would otherwise still shrink them.
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
    raise ValueError(f"unknown optimizer {name!r}; known: adam, adamw, sgd")


def _lr_scale(epoch: int, epochs: int, schedule: str, warmup_epochs: int, min_lr_fraction: float) -> float:
    """Multiplier on the base LR for ``epoch`` (0-indexed)."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        # Linear warm-up from one fraction-step up to 1.0; starting at
        # exactly 0 would waste the first epoch entirely.
        return float(epoch + 1) / float(warmup_epochs + 1)
    schedule = (schedule or "none").lower()
    if schedule in ("none", "constant"):
        return 1.0
    progress_span = max(epochs - warmup_epochs, 1)
    progress = min(max(epoch - warmup_epochs, 0) / progress_span, 1.0)
    if schedule == "cosine":
        return min_lr_fraction + (1.0 - min_lr_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "step":
        # Standard 3-stage decay at 50% and 75% of the post-warmup budget.
        if progress < 0.5:
            return 1.0
        return 0.1 if progress < 0.75 else 0.01
    raise ValueError(f"unknown schedule {schedule!r}; known: none, cosine, step")


def train_classifier(
    model: torch.nn.Module,
    forward_fn: ForwardFn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str = "cpu",
    weight_decay: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    schedule: str = "cosine",
    warmup_epochs: int = 0,
    min_lr_fraction: float = 0.01,
    early_stopping_patience: Optional[int] = None,
    grad_clip: Optional[float] = None,
    restore_best: bool = True,
    verbose: bool = False,
) -> TrainResult:
    """Train on ``train_loader``, select on ``val_loader``.

    The returned model has the best-validation weights loaded (unless
    ``restore_best`` is False). The test set is the caller's business and
    must be touched exactly once, after this returns.
    """
    device = resolve_device(device)
    model.to(device)
    opt = build_optimizer(model, optimizer, lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda e: _lr_scale(e, epochs, schedule, warmup_epochs, min_lr_fraction)
    )

    result = TrainResult()
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_since_improvement = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            logits, labels = forward_fn(model, batch, device)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
        train_loss = total_loss / n if n else 0.0

        val_loss, val_acc = evaluate_loss_and_accuracy(model, forward_fn, val_loader, device)
        current_lr = opt.param_groups[0]["lr"]
        scheduler.step()

        result.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            }
        )
        result.epochs_run = epoch + 1

        if val_acc > result.best_val_acc:
            result.best_val_acc = val_acc
            result.best_epoch = epoch
            epochs_since_improvement = 0
            if restore_best:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_since_improvement += 1

        if verbose:
            print(
                f"    epoch {epoch:3d}  train_loss {train_loss:.4f}  val_acc {val_acc:.4f}  lr {current_lr:.2e}",
                flush=True,
            )

        if early_stopping_patience is not None and epochs_since_improvement >= early_stopping_patience:
            result.stopped_early = True
            break

    result.train_seconds = time.time() - t0
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return result
