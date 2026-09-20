import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shapeprim.data.synth_dataset import GenerationConfig
from shapeprim.data.torch_datasets import GraphClassificationDataset
from shapeprim.evaluate import (
    accuracy_on_classes,
    confusion_pairs,
    evaluate_extractor,
    format_aggregate_table,
    per_class_accuracy,
)
from shapeprim.experiment import aggregate_runs
from shapeprim.extract.classical import ClassicalExtractor
from shapeprim.extract.oracle import OracleExtractor

CLASSES = ["a", "b", "c"]


class Oracleish(nn.Module):
    """Predicts the label encoded in the input, except for class index 1."""

    def forward(self, x):
        preds = x.argmax(dim=-1)
        preds = torch.where(preds == 1, torch.zeros_like(preds), preds)  # always wrong on "b"
        return nn.functional.one_hot(preds, num_classes=len(CLASSES)).float()


def _forward(model, batch, device):
    x, y = batch
    return model(x.to(device)), y.to(device)


def _loader():
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    x = nn.functional.one_hot(y, num_classes=len(CLASSES)).float()
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_per_class_accuracy_uses_the_given_class_list():
    acc = per_class_accuracy(Oracleish(), _forward, _loader(), classes=CLASSES)
    assert acc == {"a": 1.0, "b": 0.0, "c": 1.0}


def test_accuracy_on_subset_scores_only_those_classes():
    assert accuracy_on_classes(Oracleish(), _forward, _loader(), ["b"], classes=CLASSES) == 0.0
    assert accuracy_on_classes(Oracleish(), _forward, _loader(), ["a", "c"], classes=CLASSES) == 1.0
    # The subset restricts scoring, not the model's choices: it still
    # predicts over all classes.
    assert accuracy_on_classes(Oracleish(), _forward, _loader(), ["a", "b"], classes=CLASSES) == 0.5


def test_confusion_records_where_errors_go():
    matrix = confusion_pairs(Oracleish(), _forward, _loader(), classes=CLASSES)
    assert matrix["b"]["a"] == 2
    assert matrix["b"]["b"] == 0
    assert matrix["a"]["a"] == 2


def test_oracle_extractor_scores_perfect_f1():
    ds = GraphClassificationDataset(
        OracleExtractor(),
        classes=["house", "tree"],
        n_per_class=4,
        cfg=GenerationConfig(image_size=64, distractor_prob=0.0),
        seed=0,
        split="test",
    )
    report = evaluate_extractor(ds)
    assert report["extractor"] == "oracle"
    assert report["f1"] == pytest.approx(1.0)
    assert report["precision"] == pytest.approx(1.0)
    assert report["recall"] == pytest.approx(1.0)
    assert report["mean_primitives_predicted"] == report["mean_primitives_ground_truth"]


def test_classical_extractor_reports_a_real_gap():
    """The extractor report must surface the oracle-vs-classical gap.

    Without this number next to the accuracy, a low GNN score can't be
    attributed to the representation rather than to stage 1.
    """
    ds = GraphClassificationDataset(
        ClassicalExtractor(),
        classes=["house", "tree"],
        n_per_class=4,
        cfg=GenerationConfig(image_size=128, distractor_prob=0.0),
        seed=0,
        split="test",
    )
    report = evaluate_extractor(ds)
    assert 0.0 < report["f1"] <= 1.0
    assert set(report["f1_per_class"]) == {"house", "tree"}


def test_extractor_eval_reuses_the_cache(tmp_path):
    ds = GraphClassificationDataset(
        OracleExtractor(),
        classes=["tree"],
        n_per_class=3,
        cfg=GenerationConfig(image_size=48, distractor_prob=0.0),
        seed=0,
        split="test",
        cache_dir=tmp_path,
    )
    ds.prewarm()
    hits_before = ds.cache.hits
    evaluate_extractor(ds)
    assert ds.cache.hits > hits_before


def test_extractor_eval_subsamples():
    ds = GraphClassificationDataset(
        OracleExtractor(),
        classes=["tree"],
        n_per_class=20,
        cfg=GenerationConfig(image_size=48, distractor_prob=0.0),
        seed=0,
        split="test",
    )
    assert evaluate_extractor(ds, max_samples=5)["n_samples"] <= 6


def test_aggregate_table_renders_multi_seed_rows():
    rows = [{"seed": s, "test_acc": a} for s, a in zip(range(3), [0.9, 0.92, 0.94])]
    table = format_aggregate_table({"cnn": aggregate_runs(rows, ["test_acc"])})
    assert "cnn" in table and "+/-" in table and "3" in table
