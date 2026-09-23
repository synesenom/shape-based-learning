"""Evaluation utilities: per-class accuracy, extractor quality, and tables.

``train.evaluate_classifier`` gives overall accuracy; this module adds the
per-class breakdown, the extractor precision/recall/F1 that PLAN.md
section 4 requires for every non-oracle extractor, and the comparison
tables. Reporting extraction quality next to downstream accuracy is what
separates "the representation is bad" from "stage 1 failed to recover it"
-- without it, a low GNN number is uninterpretable.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .data.objects import CLASS_NAMES
from .experiment import format_mean_ci
from .extract.eval_match import match_primitives
from .train import ForwardFn


@torch.no_grad()
def per_class_accuracy(
    model: torch.nn.Module,
    forward_fn: ForwardFn,
    loader: DataLoader,
    device: str = "cpu",
    classes: Sequence[str] = CLASS_NAMES,
) -> Dict[str, float]:
    """Accuracy per class name.

    ``classes`` must be the same ordered list the dataset's label indices
    were built from; passing a subset that doesn't match the dataset's
    ordering silently mislabels every row.
    """
    model.eval()
    classes = list(classes)
    correct = {name: 0 for name in classes}
    total = {name: 0 for name in classes}
    for batch in loader:
        logits, labels = forward_fn(model, batch, device)
        preds = logits.argmax(dim=-1)
        for label, pred in zip(labels.tolist(), preds.tolist()):
            name = classes[label]
            total[name] += 1
            correct[name] += int(label == pred)
    return {name: (correct[name] / total[name] if total[name] else 0.0) for name in classes}


@torch.no_grad()
def confusion_pairs(
    model: torch.nn.Module,
    forward_fn: ForwardFn,
    loader: DataLoader,
    device: str = "cpu",
    classes: Sequence[str] = CLASS_NAMES,
) -> Dict[str, Dict[str, int]]:
    """Confusion counts as {true_class: {predicted_class: count}}."""
    model.eval()
    classes = list(classes)
    matrix = {t: {p: 0 for p in classes} for t in classes}
    for batch in loader:
        logits, labels = forward_fn(model, batch, device)
        preds = logits.argmax(dim=-1)
        for label, pred in zip(labels.tolist(), preds.tolist()):
            matrix[classes[label]][classes[pred]] += 1
    return matrix


@torch.no_grad()
def accuracy_on_classes(
    model: torch.nn.Module,
    forward_fn: ForwardFn,
    loader: DataLoader,
    subset: Sequence[str],
    device: str = "cpu",
    classes: Sequence[str] = CLASS_NAMES,
) -> float:
    """Accuracy restricted to samples whose true label is in ``subset``.

    This is the tree-vs-arrow_sign measurement (PLAN.md section 5): the
    model still chooses among all classes, but only the relation-twin
    samples are scored.
    """
    model.eval()
    classes = list(classes)
    wanted = {classes.index(name) for name in subset}
    correct, total = 0, 0
    for batch in loader:
        logits, labels = forward_fn(model, batch, device)
        preds = logits.argmax(dim=-1)
        for label, pred in zip(labels.tolist(), preds.tolist()):
            if label in wanted:
                total += 1
                correct += int(label == pred)
    return correct / total if total else 0.0


@torch.no_grad()
def predictions(model: torch.nn.Module, forward_fn: ForwardFn, loader: DataLoader, device: str = "cpu") -> List[int]:
    """Predicted class index per test sample, in dataset order (loader must not shuffle)."""
    model.eval()
    out: List[int] = []
    for batch in loader:
        logits, _ = forward_fn(model, batch, device)
        out += logits.argmax(dim=-1).tolist()
    return out


def evaluate_extractor(
    dataset,
    iou_threshold: float = 0.5,
    max_samples: Optional[int] = None,
    per_class: bool = True,
) -> Dict[str, Any]:
    """Precision/recall/F1 of a GraphClassificationDataset's extractor.

    Takes the dataset rather than an extractor so it reuses the extraction
    cache: the numbers reported here are computed on exactly the primitives
    the classifier was trained on, not on a fresh extraction that might
    differ.
    """
    n = len(dataset)
    indices = range(n) if max_samples is None or max_samples >= n else range(0, n, max(1, n // max_samples))

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    by_class: Dict[str, List[float]] = {}
    n_pred_total, n_gt_total = 0, 0

    image_size = dataset.inner.cfg.image_size
    for i in indices:
        pred, gt, label = dataset.primitives_at(i)
        res = match_primitives(pred, gt, image_size, iou_threshold=iou_threshold)
        precisions.append(res["precision"])
        recalls.append(res["recall"])
        f1s.append(res["f1"])
        n_pred_total += res["n_pred"]
        n_gt_total += res["n_gt"]
        if per_class:
            by_class.setdefault(label, []).append(res["f1"])

    out: Dict[str, Any] = {
        "extractor": getattr(dataset.extractor, "name", type(dataset.extractor).__name__),
        "n_samples": len(f1s),
        "iou_threshold": iou_threshold,
        "precision": mean(precisions) if precisions else float("nan"),
        "recall": mean(recalls) if recalls else float("nan"),
        "f1": mean(f1s) if f1s else float("nan"),
        "mean_primitives_predicted": n_pred_total / len(f1s) if f1s else float("nan"),
        "mean_primitives_ground_truth": n_gt_total / len(f1s) if f1s else float("nan"),
    }
    if per_class:
        out["f1_per_class"] = {k: mean(v) for k, v in by_class.items()}
    return out


def format_comparison_table(results: Dict[str, dict]) -> str:
    """Single-run table. ``results`` maps model name to a metrics dict."""
    header = f"{'model':<18}{'test acc':>10}{'val acc':>10}{'train time (s)':>16}"
    lines = [header, "-" * len(header)]
    for name, r in results.items():
        lines.append(
            f"{name:<18}{r.get('test_acc', float('nan')):>10.3f}"
            f"{r.get('best_val_acc', float('nan')):>10.3f}{r.get('train_seconds', 0.0):>16.1f}"
        )
    return "\n".join(lines)


def format_aggregate_table(aggregates: Dict[str, dict], metric: str = "test_acc", extra: Sequence[str] = ()) -> str:
    """Multi-seed table: mean +/- 95% CI per model.

    ``aggregates`` maps a model name to the output of
    ``experiment.aggregate_runs``.
    """
    cols = [metric, *extra]
    width = 24
    header = f"{'model':<18}{'seeds':>7}" + "".join(f"{c:>{width}}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, agg in aggregates.items():
        row = f"{name:<18}{agg.get('n_runs', 0):>7}"
        for c in cols:
            row += f"{format_mean_ci(agg.get(c, {})):>{width}}"
        lines.append(row)
    return "\n".join(lines)
