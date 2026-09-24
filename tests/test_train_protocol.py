"""The training protocol: validation selection, early stopping, schedules.

These are the WP0 guarantees the reported numbers depend on. The failure
mode they guard against is silent: a loop that selects on the test set, or
that returns whatever weights the last epoch happened to leave behind,
still produces a plausible-looking accuracy.
"""

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shapeprim.train import (
    _lr_scale,
    build_optimizer,
    evaluate_classifier,
    evaluate_loss_and_accuracy,
    resolve_device,
    train_classifier,
)


def _forward(model, batch, device):
    x, y = batch
    return model(x.to(device)), y.to(device)


def _separable_loaders(n=64, d=4, classes=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, classes, (n,), generator=g)
    x = torch.zeros(n, d)
    x[torch.arange(n), y] = 5.0  # trivially separable
    x += 0.01 * torch.randn(n, d, generator=g)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=16, shuffle=True), DataLoader(ds, batch_size=16)


def test_returned_model_has_the_best_validation_weights():
    """After training, the model must score exactly best_val_acc on val.

    This is the contract that makes a reported test number meaningful: the
    weights evaluated on test are the ones selected on validation, not the
    last epoch's.
    """
    torch.manual_seed(0)
    train_loader, val_loader = _separable_loaders()
    model = nn.Linear(4, 2)
    result = train_classifier(
        model, _forward, train_loader, val_loader, epochs=8, lr=0.05, schedule="cosine"
    )
    assert evaluate_classifier(model, _forward, val_loader) == pytest.approx(result.best_val_acc)
    assert 0 <= result.best_epoch < result.epochs_run


def test_history_records_validation_not_test():
    torch.manual_seed(0)
    train_loader, val_loader = _separable_loaders()
    result = train_classifier(nn.Linear(4, 2), _forward, train_loader, val_loader, epochs=3, lr=0.05)
    assert len(result.history) == 3
    for row in result.history:
        assert {"epoch", "train_loss", "val_loss", "val_acc", "lr"} <= set(row)
        assert "test_acc" not in row


def test_early_stopping_triggers_when_validation_stalls():
    torch.manual_seed(0)
    train_loader, val_loader = _separable_loaders()
    result = train_classifier(
        nn.Linear(4, 2), _forward, train_loader, val_loader,
        epochs=50, lr=0.2, early_stopping_patience=2,
    )
    assert result.stopped_early
    assert result.epochs_run < 50


def test_no_early_stopping_runs_the_full_budget():
    torch.manual_seed(0)
    train_loader, val_loader = _separable_loaders()
    result = train_classifier(nn.Linear(4, 2), _forward, train_loader, val_loader, epochs=5, lr=0.05)
    assert result.epochs_run == 5
    assert not result.stopped_early


def test_restore_best_can_be_disabled():
    torch.manual_seed(0)
    train_loader, val_loader = _separable_loaders()
    model = nn.Linear(4, 2)
    result = train_classifier(
        model, _forward, train_loader, val_loader, epochs=6, lr=0.05, restore_best=False
    )
    assert result.best_epoch >= 0  # still tracked, just not restored


def test_warmup_ramps_then_schedule_takes_over():
    assert _lr_scale(0, 10, "cosine", 2, 0.01) == pytest.approx(1 / 3)
    assert _lr_scale(1, 10, "cosine", 2, 0.01) == pytest.approx(2 / 3)
    # First post-warmup epoch is at full LR.
    assert _lr_scale(2, 10, "cosine", 2, 0.01) == pytest.approx(1.0)


def test_cosine_decays_monotonically_to_the_floor():
    values = [_lr_scale(e, 10, "cosine", 0, 0.01) for e in range(11)]
    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.01)
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))


def test_step_schedule_drops_at_half_and_three_quarters():
    assert _lr_scale(0, 20, "step", 0, 0.01) == 1.0
    assert _lr_scale(9, 20, "step", 0, 0.01) == 1.0
    assert _lr_scale(10, 20, "step", 0, 0.01) == pytest.approx(0.1)
    assert _lr_scale(15, 20, "step", 0, 0.01) == pytest.approx(0.01)


def test_constant_schedule_is_flat():
    assert {_lr_scale(e, 10, "none", 0, 0.01) for e in range(10)} == {1.0}


def test_unknown_schedule_raises():
    with pytest.raises(ValueError, match="unknown schedule"):
        _lr_scale(0, 10, "exponential", 0, 0.01)


@pytest.mark.parametrize("name", ["adam", "adamw", "sgd"])
def test_optimizers_build(name):
    opt = build_optimizer(nn.Linear(2, 2), name, lr=0.01, weight_decay=1e-4)
    assert opt.param_groups[0]["lr"] == 0.01


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(nn.Linear(2, 2), "lbfgs")


def test_schedule_is_applied_to_the_optimizer():
    torch.manual_seed(0)
    train_loader, val_loader = _separable_loaders()
    result = train_classifier(
        nn.Linear(4, 2), _forward, train_loader, val_loader, epochs=6, lr=0.1, schedule="cosine", warmup_epochs=2
    )
    lrs = [row["lr"] for row in result.history]
    assert lrs[0] < lrs[2]  # warm-up ramps up
    assert lrs[-1] < lrs[2]  # then decays


def test_evaluate_loss_and_accuracy_agrees_with_accuracy():
    torch.manual_seed(0)
    _, val_loader = _separable_loaders()
    model = nn.Linear(4, 2)
    loss, acc = evaluate_loss_and_accuracy(model, _forward, val_loader)
    assert acc == pytest.approx(evaluate_classifier(model, _forward, val_loader))
    assert math.isfinite(loss)


def test_resolve_device_maps_auto():
    assert resolve_device("auto") in ("cpu", "cuda")
    assert resolve_device(None) in ("cpu", "cuda")
    assert resolve_device("cpu") == "cpu"


def test_early_stopping_waits_for_min_steps():
    """Patience in epochs is meaningless at a few steps per epoch.

    With a constant (never-improving) model, patience-1 stopping fires after
    two epochs without min_steps; with min_steps it must run until the step
    floor is reached.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from shapeprim.train import cnn_forward, train_classifier

    torch.manual_seed(0)
    x = torch.zeros(8, 3, 4, 4)
    y = torch.zeros(8, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)  # 2 steps per epoch
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))
    fast = train_classifier(model, cnn_forward, loader, loader, epochs=50, lr=0.0, early_stopping_patience=1)
    slow = train_classifier(model, cnn_forward, loader, loader, epochs=50, lr=0.0, early_stopping_patience=1, min_steps=20)
    assert fast.stopped_early and fast.steps < 20
    assert slow.stopped_early and slow.steps >= 20


def test_precise_bn_reestimates_trainable_stats_and_skips_frozen():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from shapeprim.train import cnn_forward, recalibrate_batchnorm

    x = torch.randn(16, 3, 4, 4) * 3 + 5
    loader = DataLoader(TensorDataset(x, torch.zeros(16, dtype=torch.long)), batch_size=8)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.BatchNorm2d(2), torch.nn.Flatten(), torch.nn.Linear(32, 2))
    assert recalibrate_batchnorm(net, cnn_forward, loader, "cpu")
    with torch.no_grad():
        feats = net[0](x)
    assert torch.allclose(net[1].running_mean, feats.mean(dim=(0, 2, 3)), atol=1e-4)
    assert net[1].momentum == 0.1  # restored
    for p in net[1].parameters():
        p.requires_grad = False
    before = net[1].running_mean.clone()
    assert not recalibrate_batchnorm(net, cnn_forward, DataLoader(TensorDataset(x * 0, torch.zeros(16, dtype=torch.long)), batch_size=8), "cpu")
    assert torch.equal(net[1].running_mean, before)


def test_max_evals_caps_validation_passes():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from shapeprim.train import cnn_forward, train_classifier

    x = torch.randn(8, 3, 4, 4)
    loader = DataLoader(TensorDataset(x, torch.zeros(8, dtype=torch.long)), batch_size=4)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))
    res = train_classifier(model, cnn_forward, loader, loader, epochs=100, lr=1e-3, max_evals=10)
    assert len(res.history) == 10 and res.epochs_run == 100 and res.steps == 200
    assert res.history[-1]["epoch"] == 99
