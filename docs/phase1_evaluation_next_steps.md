# Phase 1: what's left to actually compare CNN vs. GNN

> **Superseded in part.** `scripts/compare_cnn_gnn.py` has been replaced by
> `scripts/run_experiment.py`, which runs the same comparison under the
> protocol in `docs/experiment_protocol.md` (train/val/test splits, model
> selection on validation, three seeds, augmented and un-augmented pixel
> baselines, extractor F1 reported alongside accuracy). Experiment 1 below
> is now a config (`configs/phase1/learning_curve.yaml`) rather than work
> to be built. Experiments 2-4 are still outstanding, and
> `docs/research_plan.md` places all four inside a larger plan.

The first comparison trained the CNN baseline and the GNN (oracle and
classical extractors) on clean, in-distribution Phase 1 data. All three
reached ~100% test accuracy (CNN 100.0%, GNN+oracle 100.0%, GNN+classical
99.5%), with the GNN training ~100x faster on CPU.

That result is expected, not a finding: PLAN.md section 5 says up front
that on clean in-distribution data both approaches saturate, so accuracy
alone here doesn't test the hypotheses. It's the four experiments below
that are supposed to separate them.

Those numbers are also not trustworthy as stated: they come from a single
seed, and the "test accuracy" is the last epoch of a curve that was itself
monitored on the test set every epoch. `results/phase1/baseline/` holds the
replacement measurement.

## 1. Few-shot learning curves

**Question:** does the primitive/graph representation need fewer examples
per class (H1)?

**Built.** `configs/phase1/learning_curve.yaml` sweeps
`n_train_per_class` over 5, 10, 25, 50, 100, 250 for all four models at 3
seeds each; `scripts/plot_results.py` draws the curve with confidence
bands and writes the matching table.

```bash
python scripts/run_experiment.py --config configs/phase1/learning_curve.yaml
python scripts/plot_results.py --summary results/phase1/learning_curve/summary.json
```

**Caveat to carry into the write-up:** the validation set stays at 40 per
class while training shrinks to 5, so at the small end every model selects
on more data than it trains on. The gaps stay comparable; the absolute
few-shot numbers are optimistic for all models.

## 2. Position / scale shift

**Question:** does the primitive model generalize better when objects
move or resize between train and test (part of H3, the viewpoint
hypothesis, tested here without rotation)?

**What to build:** two `GenerationConfig` variants -- train with objects
small and centered (e.g. `object_scale_range=(0.4, 0.55)`,
`position_jitter=0.03`), test with objects large or off-center (e.g.
`object_scale_range=(0.7, 0.9)`, `position_jitter=0.25`). The driver
already passes every `GenerationConfig` field straight through from the
experiment config, so this needs the two YAML configs and a train-on-A /
test-on-B option in `run_experiment.py`, not new model code.

**Now also required:** run it against `cnn_aug` (and a `strong` preset
matched to the shift), not only the un-augmented CNN. The primitive graph
is invariant by construction, so a gap measured against an un-augmented
pixel model is an augmentation result, not a representation result.

**Expected result to look for:** the primitive graph is already built
translation/scale-invariant by construction (`graph/build.py` normalizes
every feature by the object's own bounding box -- verified by
`test_scale_invariance`/`test_translation_invariance` in
`tests/test_graph_build.py`). So the *oracle* GNN should be close to
unaffected by this shift. If it degrades anyway, the invariance claim or
its test is wrong. The CNN, with no such invariance built in, is expected
to degrade more.

## 3. Novel primitive compositions

**Question:** does the model generalize to part arrangements it never
saw during training?

**What to build:** this needs new template *variants*, which don't exist
yet -- `data/objects.py` currently has one fixed arrangement per class.
Add 1-2 alternate `ClassTemplate`s per class where feasible (e.g. a
`car_3window` variant with an extra window, a `house_2window` variant),
generate training data from only the base variant, and test on the
alternate. This is the piece of the plan requiring the most new code
before it can run.

## 4. Relation test: tree vs. arrow_sign, with a bag-of-shapes ablation

**Question:** does the classifier actually use spatial arrangement, or
just which shapes are present (H2)?

**What to build:**
- `models/bag.py`: an ablation classifier with the same primitive-type
  vocabulary as the GNN but *no* position/relation features -- per
  PLAN.md section 4, counts and average size per primitive type only, no
  edges. A small MLP over a fixed-size `[count, mean_size, mean_aspect]
  x 4 types` vector is enough; `PRIMITIVE_TYPES` in `data/primitives.py`
  already gives the type ordering to build it against, and the dataset
  wrapper can reuse `GraphClassificationDataset`'s extractor plumbing
  since a bag is just a different pooling of the same primitive list.
- ~~An eval script ... reports accuracy restricted to `{tree, arrow_sign}`~~
  **Built:** every run already reports `twin_test_acc`, the accuracy
  restricted to the `{tree, arrow_sign}` pair while the model still chooses
  among all classes (`evaluate.accuracy_on_classes`). What remains is
  `models/bag.py` itself, plus registering it as a `kind` in the driver.

**Expected result to look for:** the GNN should hold up on the
tree/arrow_sign pair since arrangement is in its edge features; the bag
ablation, having no positions at all, should be near chance on that pair
specifically (~50%) despite doing fine on the other 8 classes where shape
*identity* alone is enough to tell classes apart. That gap is the
concrete evidence for "relations matter," not just an assertion.

## Order of work

1 is now a config sweep that can be run as-is. 4 needs `models/bag.py`,
which is small, since the twin-pair measurement already exists. 2 needs
two new configs plus a train-on-A/test-on-B option in the driver. 3 is the
largest lift since it requires new template variants per class. PLAN.md's
Phase 1 "done when" checklist (section 5) isn't met until all four have
run -- and `docs/research_plan.md` argues that meeting it is necessary but
not sufficient for a paper, because on this benchmark the oracle GNN is
handed the generator's own latent variables.
