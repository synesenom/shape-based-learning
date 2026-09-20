#!/usr/bin/env python3
"""Run a Phase 1 experiment: models x conditions x seeds, with a real protocol.

Replaces the single-seed, test-set-monitoring ``compare_cnn_gnn.py`` path.
What this driver guarantees (PLAN.md section 4; WP0 in
``docs/research_plan.md``):

- **Three splits from disjoint random streams.** ``train`` and ``val`` are
  drawn per run seed; ``test`` is drawn from a *fixed* seed shared by every
  run, so per-seed variation reflects training stochasticity and the
  training draw, not a different test set each time.
- **Model selection on validation only.** The test set is evaluated exactly
  once per run, after the best-validation weights are restored.
- **Every run on disk** under ``results/<phase>/<experiment>/<run_id>/``
  with its config, environment, git commit, seed and metrics.
- **Aggregates across seeds** with mean and a Student-t 95% interval.
- **Extractor precision/recall/F1** reported next to every GNN accuracy.

Usage:
    python scripts/run_experiment.py --config configs/phase1/baseline_experiment.yaml
    python scripts/run_experiment.py --config configs/phase1/learning_curve.yaml
    python scripts/run_experiment.py --config ... --resume      # skip finished runs
    python scripts/run_experiment.py --config ... --dry-run     # list the runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shapeprim.data.augment import resolve_augment  # noqa: E402
from shapeprim.data.objects import CLASS_NAMES  # noqa: E402
from shapeprim.data.synth_dataset import GenerationConfig  # noqa: E402
from shapeprim.data.torch_datasets import (  # noqa: E402
    GraphClassificationDataset,
    ImageClassificationDataset,
    collate_graphs,
)
from shapeprim.evaluate import (  # noqa: E402
    accuracy_on_classes,
    confusion_pairs,
    evaluate_extractor,
    format_aggregate_table,
    per_class_accuracy,
)
from shapeprim.experiment import (  # noqa: E402
    RunDirectory,
    aggregate_runs,
    run_id_for,
    set_all_seeds,
    write_json,
)
from shapeprim.extract.classical import ClassicalExtractor  # noqa: E402
from shapeprim.extract.oracle import OracleExtractor  # noqa: E402
from shapeprim.graph.build import NUM_EDGE_FEATURES, NUM_NODE_FEATURES  # noqa: E402
from shapeprim.models.cnn import build_cnn  # noqa: E402
from shapeprim.models.gnn import GNNClassifier  # noqa: E402
from shapeprim.train import (  # noqa: E402
    cnn_forward,
    evaluate_classifier,
    gnn_forward,
    resolve_device,
    train_classifier,
)

# The relation-twin pair: same primitive types, swapped arrangement. Tracked
# separately on every run because overall accuracy hides it (2 of 10 classes).
RELATION_TWINS = ("tree", "arrow_sign")

EXTRACTORS = {"oracle": OracleExtractor, "classical": ClassicalExtractor}

# Metrics aggregated across seeds for the summary table.
AGGREGATE_KEYS = (
    "test_acc",
    "best_val_acc",
    "twin_test_acc",
    "train_seconds",
    "epochs_run",
    "extractor_f1",
)


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_generation_config(cfg: dict) -> GenerationConfig:
    """Generation config from the experiment config's own keys.

    Reads every GenerationConfig field the experiment config mentions, so a
    condition can vary scale, jitter or distractors without a code change.
    The previous script hardcoded two fields and silently dropped the rest.
    """
    return GenerationConfig.from_dict(cfg)


def make_loaders_image(
    cfg: dict, gen_cfg: GenerationConfig, classes: List[str], seed: int, n_train: int, augment
) -> Dict[str, DataLoader]:
    workers = cfg.get("num_workers", 0)
    common = dict(classes=classes, cfg=gen_cfg)
    # Augmentation on the training split only: an augmented validation set
    # measures a different distribution than the one selection is meant to
    # estimate, and an augmented test set is not the test set.
    train_ds = ImageClassificationDataset(
        **common, n_per_class=n_train, seed=seed, split="train", augment=augment
    )
    val_ds = ImageClassificationDataset(
        **common, n_per_class=cfg["n_val_per_class"], seed=seed, split="val", augment=None
    )
    test_ds = ImageClassificationDataset(
        **common, n_per_class=cfg["n_test_per_class"], seed=cfg["test_seed"], split="test", augment=None
    )
    return {
        "train": DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=workers),
        "val": DataLoader(val_ds, batch_size=cfg["batch_size"], num_workers=workers),
        "test": DataLoader(test_ds, batch_size=cfg["batch_size"], num_workers=workers),
    }


def make_loaders_graph(
    cfg: dict, gen_cfg: GenerationConfig, classes: List[str], seed: int, n_train: int, extractor_name: str
) -> tuple[Dict[str, DataLoader], Dict[str, GraphClassificationDataset]]:
    workers = cfg.get("num_workers", 0)
    cache_dir = REPO_ROOT / cfg["cache_dir"] if cfg.get("cache_dir") else None
    extractor = EXTRACTORS[extractor_name]()
    common = dict(classes=classes, cfg=gen_cfg, cache_dir=cache_dir)

    datasets = {
        "train": GraphClassificationDataset(extractor, **common, n_per_class=n_train, seed=seed, split="train"),
        "val": GraphClassificationDataset(
            extractor, **common, n_per_class=cfg["n_val_per_class"], seed=seed, split="val"
        ),
        "test": GraphClassificationDataset(
            extractor, **common, n_per_class=cfg["n_test_per_class"], seed=cfg["test_seed"], split="test"
        ),
    }
    # Extract once, in this process, before any DataLoader worker forks.
    for ds in datasets.values():
        ds.prewarm(verbose=cfg.get("verbose", False))

    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=cfg["batch_size"], shuffle=True, num_workers=workers, collate_fn=collate_graphs
        ),
        "val": DataLoader(datasets["val"], batch_size=cfg["batch_size"], num_workers=workers, collate_fn=collate_graphs),
        "test": DataLoader(
            datasets["test"], batch_size=cfg["batch_size"], num_workers=workers, collate_fn=collate_graphs
        ),
    }
    return loaders, datasets


def train_params(model_cfg: dict, cfg: dict) -> dict:
    """Training hyperparameters, model config overriding experiment defaults."""
    return {
        "epochs": model_cfg.get("epochs", cfg["epochs"]),
        "lr": model_cfg["lr"],
        "weight_decay": model_cfg.get("weight_decay", 0.0),
        "optimizer": model_cfg.get("optimizer", "adam"),
        "momentum": model_cfg.get("momentum", 0.9),
        "schedule": model_cfg.get("schedule", cfg.get("schedule", "cosine")),
        "warmup_epochs": model_cfg.get("warmup_epochs", cfg.get("warmup_epochs", 0)),
        "min_lr_fraction": model_cfg.get("min_lr_fraction", 0.01),
        "early_stopping_patience": model_cfg.get(
            "early_stopping_patience", cfg.get("early_stopping_patience")
        ),
        "grad_clip": model_cfg.get("grad_clip", cfg.get("grad_clip")),
    }


def run_one(
    spec: dict,
    cfg: dict,
    model_configs: Dict[str, dict],
    classes: List[str],
    seed: int,
    n_train: int,
    device: str,
) -> Dict[str, Any]:
    """One (model, condition, seed) run. Returns the metrics record."""
    kind = spec["kind"]
    gen_cfg = build_generation_config(cfg)
    set_all_seeds(seed)

    extractor_report: Optional[dict] = None

    if kind == "cnn":
        model_cfg = dict(model_configs["cnn"])
        model_cfg.update(spec.get("model_overrides", {}))
        augment = resolve_augment(spec.get("augment"))
        loaders = make_loaders_image(cfg, gen_cfg, classes, seed, n_train, augment)
        model = build_cnn(num_classes=len(classes), pretrained=model_cfg.get("pretrained", False))
        forward_fn = cnn_forward
        augment_record = augment.to_dict()
    elif kind == "gnn":
        model_cfg = dict(model_configs["gnn"])
        model_cfg.update(spec.get("model_overrides", {}))
        extractor_name = spec["extractor"]
        loaders, datasets = make_loaders_graph(cfg, gen_cfg, classes, seed, n_train, extractor_name)
        model = GNNClassifier(
            NUM_NODE_FEATURES,
            NUM_EDGE_FEATURES,
            num_classes=len(classes),
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
        )
        forward_fn = gnn_forward
        augment_record = None
        # Extraction quality on the same primitives the model is scored on.
        extractor_report = evaluate_extractor(
            datasets["test"], max_samples=cfg.get("extractor_eval_samples", 200)
        )
    else:
        raise ValueError(f"unknown model kind {kind!r}; known: cnn, gnn")

    n_params = sum(p.numel() for p in model.parameters())
    params = train_params(model_cfg, cfg)
    result = train_classifier(
        model,
        forward_fn,
        loaders["train"],
        loaders["val"],
        device=device,
        verbose=cfg.get("verbose", False),
        **params,
    )

    # The single, final look at the test set, on best-validation weights.
    test_acc = evaluate_classifier(model, forward_fn, loaders["test"], device)
    per_class = per_class_accuracy(model, forward_fn, loaders["test"], device, classes=classes)
    twins = [c for c in RELATION_TWINS if c in classes]
    twin_acc = (
        accuracy_on_classes(model, forward_fn, loaders["test"], twins, device, classes=classes)
        if len(twins) == len(RELATION_TWINS)
        else None
    )

    record: Dict[str, Any] = {
        "model": spec["name"],
        "kind": kind,
        "seed": seed,
        "n_train_per_class": n_train,
        "test_acc": test_acc,
        "best_val_acc": result.best_val_acc,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "train_seconds": result.train_seconds,
        "n_parameters": n_params,
        "twin_test_acc": twin_acc,
        "per_class_acc": per_class,
        "confusion": confusion_pairs(model, forward_fn, loaders["test"], device, classes=classes),
        "history": result.history,
        "augment": augment_record,
        "train_params": params,
        "device": device,
    }
    if extractor_report is not None:
        record["extractor"] = extractor_report
        record["extractor_f1"] = extractor_report["f1"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/phase1/baseline_experiment.yaml")
    parser.add_argument("--cnn-config", default="configs/model/cnn.yaml")
    parser.add_argument("--gnn-config", default="configs/model/gnn.yaml")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--resume", action="store_true", help="Skip runs that already have metrics.json")
    parser.add_argument("--dry-run", action="store_true", help="List the runs without training")
    parser.add_argument("--device", default=None, help="Override the config's device")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override the config's seeds")
    parser.add_argument("--models", nargs="+", default=None, help="Run only these model names")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(REPO_ROOT / args.config)
    cfg.setdefault("phase", "phase1")
    cfg.setdefault("experiment", Path(args.config).stem)
    cfg.setdefault("test_seed", 12345)
    cfg.setdefault("n_val_per_class", cfg.get("n_test_per_class", 40))
    cfg.setdefault("num_workers", 0)
    cfg.setdefault("cache_dir", "data/cache")
    if args.verbose:
        cfg["verbose"] = True

    model_configs = {
        "cnn": load_yaml(REPO_ROOT / args.cnn_config),
        "gnn": load_yaml(REPO_ROOT / args.gnn_config),
    }

    classes = cfg.get("classes") or list(CLASS_NAMES)
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [0, 1, 2])
    device = resolve_device(args.device or cfg.get("device", "auto"))

    specs = cfg["models"]
    if args.models:
        specs = [s for s in specs if s["name"] in args.models]
        if not specs:
            raise SystemExit(f"no models named {args.models} in {args.config}")

    sweep = cfg.get("sweep") or {}
    n_train_values = sweep.get("n_train_per_class") or [cfg["n_train_per_class"]]

    plan = [
        (spec, n_train, seed) for spec in specs for n_train in n_train_values for seed in seeds
    ]
    print(
        f"{cfg['phase']}/{cfg['experiment']}: {len(plan)} runs "
        f"({len(specs)} models x {len(n_train_values)} sizes x {len(seeds)} seeds) on {device}"
    )
    if args.dry_run:
        for spec, n_train, seed in plan:
            print(f"  {run_id_for(spec['name'], seed, n=n_train)}")
        return

    results_root = REPO_ROOT / args.results_root
    records: List[Dict[str, Any]] = []

    for i, (spec, n_train, seed) in enumerate(plan, 1):
        run_id = run_id_for(spec["name"], seed, n=n_train)
        run_dir = RunDirectory(results_root, cfg["phase"], cfg["experiment"], run_id)

        if args.resume and run_dir.exists("metrics"):
            cached = run_dir.load("metrics")
            if cached:
                print(f"[{i}/{len(plan)}] {run_id}: cached, test acc = {cached.get('test_acc', float('nan')):.3f}")
                records.append(cached)
                continue

        print(f"[{i}/{len(plan)}] {run_id} ...", flush=True)
        record = run_one(spec, cfg, model_configs, classes, seed, n_train, device)
        run_dir.save_config(
            {"experiment": cfg, "model_spec": spec, "seed": seed, "n_train_per_class": n_train, "classes": classes},
            repo_root=REPO_ROOT,
        )
        run_dir.save_metrics(record)
        records.append(record)

        extra = f", extractor F1 = {record['extractor_f1']:.3f}" if "extractor_f1" in record else ""
        print(
            f"      test acc = {record['test_acc']:.3f} (val {record['best_val_acc']:.3f}, "
            f"best epoch {record['best_epoch']}, {record['train_seconds']:.0f}s){extra}",
            flush=True,
        )

    # Aggregate per (model, n_train).
    summary: Dict[str, Any] = {
        "config": cfg,
        "classes": classes,
        "seeds": seeds,
        "conditions": {},
        # Parameter counts belong next to the accuracies: the CNN carries
        # ~11.2M parameters against the GNN's ~57k, and a sample-efficiency
        # claim that doesn't state that gap invites the obvious objection.
        "n_parameters": {r["model"]: r["n_parameters"] for r in records},
    }
    for n_train in n_train_values:
        per_model = {}
        for spec in specs:
            rows = [r for r in records if r["model"] == spec["name"] and r["n_train_per_class"] == n_train]
            if rows:
                per_model[spec["name"]] = aggregate_runs(rows, AGGREGATE_KEYS)
        summary["conditions"][str(n_train)] = per_model

        print()
        print(f"n_train_per_class = {n_train}")
        print(format_aggregate_table(per_model, metric="test_acc", extra=["twin_test_acc", "extractor_f1"]))

    out_path = results_root / cfg["phase"] / cfg["experiment"] / "summary.json"
    write_json(out_path, summary)
    print(f"\nSaved {len(records)} runs and the summary under {out_path.parent}")


if __name__ == "__main__":
    main()
