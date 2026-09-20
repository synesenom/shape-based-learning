"""Experimental conditions: train-on-A / test-on-B generation configs.

A shift experiment (PLAN.md section 5, "position/scale shift"; WP0 item 4
and WP1.4 in ``docs/research_plan.md``) trains on one data distribution
and tests on another. Everything about that is config, not code: a
``shift`` block names overrides for the training distribution and the test
distribution, both applied on top of the experiment's base generation
settings.

Two decisions are baked in here because getting either wrong quietly
invalidates the measurement.

**Validation follows the training distribution, never the test one.** At
model-selection time an experimenter does not have access to the shifted
distribution -- that is the whole premise of a shift test. Selecting on
shifted validation data leaks the test condition into training and turns a
generalization measurement into a weak form of training on the target.

**The in-distribution test set is kept and reported too.** A shifted
accuracy on its own is uninterpretable: 0.7 means one thing if the model
scores 0.71 in-distribution and something else entirely if it scores 1.0.
The quantity of interest is the drop, so both are measured and the
difference is recorded.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from .data.synth_dataset import GenerationConfig

# Keys of the experiment config that are GenerationConfig fields; anything
# else in the experiment config (epochs, seeds, ...) is not generation.
GENERATION_FIELDS = set(GenerationConfig.__dataclass_fields__)


def base_generation_config(cfg: Dict[str, Any]) -> GenerationConfig:
    """The experiment's base generation config."""
    return GenerationConfig.from_dict(cfg)


def _apply(base: GenerationConfig, overrides: Optional[Dict[str, Any]]) -> GenerationConfig:
    if not overrides:
        return base
    unknown = set(overrides) - GENERATION_FIELDS
    if unknown:
        # A typo here would silently leave the distribution unshifted, and
        # the experiment would report "no degradation" for the wrong reason.
        raise ValueError(
            f"shift overrides are not GenerationConfig fields: {sorted(unknown)}; "
            f"known fields: {sorted(GENERATION_FIELDS)}"
        )
    merged = dict(asdict(base))
    merged.update(overrides)
    return GenerationConfig.from_dict(merged)


class Condition:
    """Resolved generation configs for one experimental condition.

    ``train_cfg`` covers the train and validation splits; ``test_cfg``
    covers the primary test split. When they are equal the experiment is
    in-distribution and ``is_shift`` is False.
    """

    def __init__(self, name: str, train_cfg: GenerationConfig, test_cfg: GenerationConfig):
        self.name = name
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def is_shift(self) -> bool:
        return asdict(self.train_cfg) != asdict(self.test_cfg)

    def describe(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "is_shift": self.is_shift}
        if self.is_shift:
            train, test = asdict(self.train_cfg), asdict(self.test_cfg)
            out["changed"] = {
                k: {"train": train[k], "test": test[k]} for k in train if train[k] != test[k]
            }
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Condition(name={self.name!r}, is_shift={self.is_shift})"


def resolve_condition(cfg: Dict[str, Any]) -> Condition:
    """Build the condition described by an experiment config.

    With no ``shift`` block the train and test distributions are identical.
    With one, ``shift.train`` and ``shift.test`` are overrides applied to
    the base config.
    """
    base = base_generation_config(cfg)
    shift = cfg.get("shift")
    if not shift:
        return Condition(cfg.get("experiment", "in_distribution"), base, base)

    name = shift.get("name", "shift")
    train_cfg = _apply(base, shift.get("train"))
    test_cfg = _apply(base, shift.get("test"))
    if asdict(train_cfg) == asdict(test_cfg):
        raise ValueError(
            f"shift {name!r} leaves the train and test distributions identical; "
            "either give it real overrides or drop the shift block"
        )
    return Condition(name, train_cfg, test_cfg)
