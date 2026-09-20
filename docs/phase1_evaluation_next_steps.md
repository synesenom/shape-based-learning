# Phase 1: what's left to actually compare CNN vs. GNN

`scripts/compare_cnn_gnn.py` trains the CNN baseline and the GNN (oracle
and classical extractors) on clean, in-distribution Phase 1 data. All
three reach ~100% test accuracy (CNN 100.0%, GNN+oracle 100.0%,
GNN+classical 99.5%), with the GNN training ~100x faster on CPU.

That result is expected, not a finding: PLAN.md section 5 says up front
that on clean in-distribution data both approaches saturate, so accuracy
alone here doesn't test the hypotheses. It's the four experiments below
that are supposed to separate them. None are built yet.

## 1. Few-shot learning curves

**Question:** does the primitive/graph representation need fewer examples
per class (H1)?

**What to build:** vary `n_train_per_class` in
`configs/phase1/baseline_experiment.yaml` over 5, 10, 25, 50, 100, 1000
and rerun all three models at each size, 3 seeds per point (PLAN.md
section 4's evaluation rule). Plot test accuracy vs. `n_train_per_class`
for CNN, GNN+oracle, GNN+classical on one chart.

**Reuse:** `run_cnn` / `run_gnn` in `scripts/compare_cnn_gnn.py` already
take `n_train_per_class` from the experiment config -- this is a sweep
over that config, not new model code. Only new work: a sweep driver and a
plotting/reporting step (`scripts/` per PLAN.md's structure).

## 2. Position / scale shift

**Question:** does the primitive model generalize better when objects
move or resize between train and test (part of H3, the viewpoint
hypothesis, tested here without rotation)?

**What to build:** two `GenerationConfig` variants -- train with objects
small and centered (e.g. `object_scale_range=(0.4, 0.55)`,
`position_jitter=0.03`), test with objects large or off-center (e.g.
`object_scale_range=(0.7, 0.9)`, `position_jitter=0.25`). Both configs
already exist as `GenerationConfig` fields in `data/synth_dataset.py`;
this needs two new YAML configs (e.g.
`configs/phase1/shift_train.yaml`, `configs/phase1/shift_test.yaml`) and
a training/eval script that trains on one and evaluates on the other.

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
- An eval script that trains all three models (CNN, GNN, bag) on the
  full 10-class set as usual, then reports accuracy restricted to just
  `{tree, arrow_sign}` (two classes built from the same rectangle +
  triangle, swapped top/bottom -- see `data/objects.py`).

**Expected result to look for:** the GNN should hold up on the
tree/arrow_sign pair since arrangement is in its edge features; the bag
ablation, having no positions at all, should be near chance on that pair
specifically (~50%) despite doing fine on the other 8 classes where shape
*identity* alone is enough to tell classes apart. That gap is the
concrete evidence for "relations matter," not just an assertion.

## Order of work

1 and 4 need no new template/data code and are the cheapest to run next
(1 is a config sweep, 4 needs `models/bag.py`, which is small). 2 needs
two new configs plus a train-on-A/test-on-B script. 3 is the largest
lift since it requires new template variants per class. PLAN.md's Phase 1
"done when" checklist (section 5) isn't met until all four have run.
