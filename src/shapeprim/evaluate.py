"""Evaluation utilities: per-class accuracy and a comparison table/report.

``train.evaluate_classifier`` gives overall accuracy; this module adds the
per-class breakdown and a human-readable comparison across several trained
models, per PLAN.md section 4's evaluation rules (accuracy per model,
saved alongside config/metrics).
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .data.objects import CLASS_NAMES
from .train import ForwardFn


@torch.no_grad()
def per_class_accuracy(
    model: torch.nn.Module, forward_fn: ForwardFn, loader: DataLoader, device: str = "cpu"
) -> Dict[str, float]:
    model.eval()
    correct = {name: 0 for name in CLASS_NAMES}
    total = {name: 0 for name in CLASS_NAMES}
    for batch in loader:
        logits, labels = forward_fn(model, batch, device)
        preds = logits.argmax(dim=-1)
        for label, pred in zip(labels.tolist(), preds.tolist()):
            name = CLASS_NAMES[label]
            total[name] += 1
            correct[name] += int(label == pred)
    return {name: (correct[name] / total[name] if total[name] else 0.0) for name in CLASS_NAMES}


def format_comparison_table(results: Dict[str, dict]) -> str:
    """``results`` maps a model name to {"final_acc": float, "train_seconds": float, ...}."""
    header = f"{'model':<16}{'test acc':>10}{'train time (s)':>16}"
    lines = [header, "-" * len(header)]
    for name, r in results.items():
        lines.append(f"{name:<16}{r['final_acc']:>10.3f}{r.get('train_seconds', 0.0):>16.1f}")
    return "\n".join(lines)
