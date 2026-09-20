#!/usr/bin/env python3
"""Train the CNN baseline and the GNN classifier (oracle + classical
extractors) on the same Phase 1 data and report test accuracy.

Usage:
    python scripts/compare_cnn_gnn.py --config configs/phase1/baseline_experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shapeprim.data.objects import CLASS_NAMES  # noqa: E402
from shapeprim.data.synth_dataset import GenerationConfig  # noqa: E402
from shapeprim.data.torch_datasets import GraphClassificationDataset, ImageClassificationDataset, collate_graphs  # noqa: E402
from shapeprim.evaluate import format_comparison_table, per_class_accuracy  # noqa: E402
from shapeprim.extract.classical import ClassicalExtractor  # noqa: E402
from shapeprim.extract.oracle import OracleExtractor  # noqa: E402
from shapeprim.graph.build import NUM_EDGE_FEATURES, NUM_NODE_FEATURES  # noqa: E402
from shapeprim.models.cnn import build_cnn  # noqa: E402
from shapeprim.models.gnn import GNNClassifier  # noqa: E402
from shapeprim.train import cnn_forward, gnn_forward, train_classifier  # noqa: E402


def run_cnn(exp_cfg: dict, model_cfg: dict, gen_cfg: GenerationConfig) -> dict:
    train_ds = ImageClassificationDataset(n_per_class=exp_cfg["n_train_per_class"], cfg=gen_cfg, seed=exp_cfg["seed"])
    test_ds = ImageClassificationDataset(n_per_class=exp_cfg["n_test_per_class"], cfg=gen_cfg, seed=exp_cfg["seed"] + 1)
    train_loader = DataLoader(train_ds, batch_size=exp_cfg["batch_size"], shuffle=True, num_workers=exp_cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=exp_cfg["batch_size"], num_workers=exp_cfg["num_workers"])

    model = build_cnn(num_classes=len(CLASS_NAMES), pretrained=model_cfg["pretrained"])
    t0 = time.time()
    history = train_classifier(
        model, cnn_forward, train_loader, test_loader,
        epochs=exp_cfg["epochs"], lr=model_cfg["lr"], weight_decay=model_cfg["weight_decay"],
    )
    train_seconds = time.time() - t0
    per_class = per_class_accuracy(model, cnn_forward, test_loader)
    return {"history": history, "final_acc": history[-1]["test_acc"], "train_seconds": train_seconds, "per_class_acc": per_class}


def run_gnn(extractor, exp_cfg: dict, model_cfg: dict, gen_cfg: GenerationConfig) -> dict:
    train_ds = GraphClassificationDataset(extractor, n_per_class=exp_cfg["n_train_per_class"], cfg=gen_cfg, seed=exp_cfg["seed"])
    test_ds = GraphClassificationDataset(extractor, n_per_class=exp_cfg["n_test_per_class"], cfg=gen_cfg, seed=exp_cfg["seed"] + 1)
    train_loader = DataLoader(
        train_ds, batch_size=exp_cfg["batch_size"], shuffle=True, num_workers=exp_cfg["num_workers"], collate_fn=collate_graphs
    )
    test_loader = DataLoader(
        test_ds, batch_size=exp_cfg["batch_size"], num_workers=exp_cfg["num_workers"], collate_fn=collate_graphs
    )

    model = GNNClassifier(
        NUM_NODE_FEATURES, NUM_EDGE_FEATURES, num_classes=len(CLASS_NAMES),
        hidden_dim=model_cfg["hidden_dim"], num_layers=model_cfg["num_layers"], dropout=model_cfg["dropout"],
    )
    t0 = time.time()
    history = train_classifier(
        model, gnn_forward, train_loader, test_loader,
        epochs=exp_cfg["epochs"], lr=model_cfg["lr"], weight_decay=model_cfg["weight_decay"],
    )
    train_seconds = time.time() - t0
    per_class = per_class_accuracy(model, gnn_forward, test_loader)
    return {"history": history, "final_acc": history[-1]["test_acc"], "train_seconds": train_seconds, "per_class_acc": per_class}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1/baseline_experiment.yaml")
    parser.add_argument("--cnn-config", default="configs/model/cnn.yaml")
    parser.add_argument("--gnn-config", default="configs/model/gnn.yaml")
    parser.add_argument("--out", default="results/phase1/cnn_vs_gnn.json")
    args = parser.parse_args()

    with open(REPO_ROOT / args.config) as f:
        exp_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / args.cnn_config) as f:
        cnn_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / args.gnn_config) as f:
        gnn_cfg = yaml.safe_load(f)

    torch.manual_seed(exp_cfg["seed"])

    gen_cfg = GenerationConfig(image_size=exp_cfg["image_size"], distractor_prob=exp_cfg["distractor_prob"])

    results = {}

    print("Training CNN...")
    results["cnn"] = run_cnn(exp_cfg, cnn_cfg, gen_cfg)
    print(f"  test acc = {results['cnn']['final_acc']:.3f}")

    print("Training GNN (oracle extractor)...")
    results["gnn_oracle"] = run_gnn(OracleExtractor(), exp_cfg, gnn_cfg, gen_cfg)
    print(f"  test acc = {results['gnn_oracle']['final_acc']:.3f}")

    print("Training GNN (classical extractor)...")
    results["gnn_classical"] = run_gnn(ClassicalExtractor(), exp_cfg, gnn_cfg, gen_cfg)
    print(f"  test acc = {results['gnn_classical']['final_acc']:.3f}")

    print()
    print(format_comparison_table(results))

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": exp_cfg, "results": results}, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
