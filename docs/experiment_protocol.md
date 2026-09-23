# Experiment protocol (WP0)

How experiments are run in this repository, what the protocol guarantees,
and which caveats have to travel with the numbers. This is the WP0
deliverable from `docs/research_plan.md`; it exists so that a result can
be read without reverse-engineering the driver.

## Running an experiment

```bash
python scripts/run_experiment.py --config configs/phase1/baseline_experiment.yaml
python scripts/plot_results.py --summary results/phase1/baseline/summary.json
```

Useful flags: `--resume` (skip runs that already have a `metrics.json`),
`--dry-run` (list the runs), `--models cnn gnn_oracle` (subset),
`--seeds 0 1 2 3 4`, `--device cpu|cuda|auto`.

Each run writes `results/<phase>/<experiment>/<run_id>/` containing
`config.json` (the full experiment config, plus Python/torch versions, the
platform and the git commit) and `metrics.json` (accuracy, per-class
accuracy, confusion counts, the relation-twin accuracy, the training
history, the extractor report, parameter count, wall-clock). The driver
also writes one `summary.json` per experiment with per-condition
aggregates.

## What the protocol guarantees

**1. Three splits from disjoint random streams.** `SynthShapeDataset`
takes a `split` that participates in the per-sample seed derivation, so
`train`, `val` and `test` built from the same base seed cannot overlap.
The previous convention (`seed` for train, `seed + 1` for test) made the
test set of seed 0 the training set of seed 1, which is invisible in any
single run and fatal across a multi-seed sweep.

**2. Model selection on validation, test touched once.** `train_classifier`
takes a validation loader and never sees test. It tracks the best
validation accuracy, keeps those weights, and restores them before
returning; the driver then evaluates test exactly once. The earlier loop
evaluated the *test* set every epoch and reported the last epoch, so every
decision made while looking at that curve leaked test information, and a
run that ended on a bad epoch reported the dip as its score.

**3. The training curve is stabilised, not just longer.** Cosine decay
with two warm-up epochs, gradient clipping, and early stopping on
validation. The first recorded comparison used bare Adam at 1e-3 with no
schedule and its accuracy collapsed to 0.18 and 0.61 at two points before
recovering; a last-epoch number from that curve measures luck.

**3b. A floor in optimizer steps.** Every run trains for at least 500
optimizer steps (`min_steps`; small training sets get more epochs to reach
it) and early stopping may not fire before them. Patience counted in
epochs means very different amounts of training at 5 and at 1000 examples
per class: at 25/class a patience of 8 epochs is 64 steps, shorter than
the noisy start of an augmented CNN, and it ended runs at 0.43
in-distribution accuracy while the learning rate was still near its peak.
Runs recorded before this rule that stopped earlier were moved to
`results/<phase>/superseded_patience8/` and re-run
(`scripts/supersede_short_runs.py`); runs that trained past the floor are
unaffected by it.

**4. The same protocol on both sides.** The CNN and the GNN get the same
schedule, warm-up, clipping, stopping rule and selection criterion. An
advantage produced by tuning one side is exactly the artefact this
comparison exists to rule out. Any per-model tuning budget spent later
must be spent on both and written down.

**5. Augmentation is an experimental condition, not a detail.** The
primitive graph is translation- and scale-invariant by construction, so an
un-augmented CNN is not a control for "does structure help". Every
experiment reports the pixel baseline with and without random-affine
augmentation covering the generator's own position and scale ranges.

**6. Extraction quality is reported next to accuracy.** Every GNN run
reports its extractor's precision, recall and F1 (and per-class F1)
against ground truth, computed on the same cached primitives the model was
scored on. Without it, a low GNN number cannot be attributed to the
representation rather than to stage 1.

**7. Seeds and intervals.** Three seeds minimum, aggregated as mean with a
Student-t 95% interval. At three seeds the multiplier is 4.303, not 1.96;
using the normal quantile understates the interval by more than a factor
of two. A single-seed result is recorded as "1 seed" with no interval
rather than as "± 0.000".

## Caveats that must travel with the numbers

- **The validation set is oversized in the few-shot regime.**
  `n_val_per_class` is fixed (40) while training shrinks to 5 per class, so
  at the small end a model selects on eight times more data than it trains
  on. That is not a realistic few-shot setting. It is applied identically
  to every model, so the *gaps* remain comparable, but the absolute
  few-shot numbers are optimistic for all models and should be described
  that way.
- **The test set is fixed across seeds** (drawn from `test_seed`, default
  12345). Per-seed spread therefore covers training stochasticity and the
  training draw, not test-set variation. This narrows the intervals
  relative to also resampling test, and the claim the intervals support is
  correspondingly narrower.
- **Parameter counts are not matched.** ResNet-18 carries about 11.2M
  parameters against the GNN's roughly 57k. Until the matched-capacity
  baseline in WP2 exists, no sample-efficiency claim here is clean.
- **Rotation augmentation is capped at 90 degrees.** Past that, a rotated
  `tree` is an `arrow_sign`: the two classes exist precisely to test
  whether arrangement is used, and rotating them into each other would
  inject label noise into that measurement. `AugmentConfig` refuses larger
  values rather than silently degrading the twin test.
- **Caching assumes determinism.** Extracted primitives are cached by a
  digest of (classes, count, generation config, seed, split, extractor). If
  an extractor is ever made stochastic, that digest stops being sufficient
  and the cache must key on its seed too.

## Shift experiments (train on A, test on B)

A `shift` block in the experiment config names overrides for the training
distribution and the test distribution, both applied on top of the base
generation settings:

```yaml
shift:
  name: position_scale
  train: {object_scale_range: [0.40, 0.55], position_jitter: 0.03}
  test:  {object_scale_range: [0.70, 0.90], position_jitter: 0.25}
```

Three rules are enforced in code rather than left to discipline
(`src/shapeprim/conditions.py`):

- **Validation follows the training distribution.** At selection time the
  shifted distribution is not available -- that is the premise of a shift
  test. Selecting on shifted validation data leaks the test condition into
  training and turns a generalization measurement into a weak form of
  training on the target.
- **The in-distribution test set is measured too.** A shifted accuracy
  alone is uninterpretable: 0.70 means one thing against an
  in-distribution 0.71 and something else against 1.00. Runs report
  `test_acc` (shifted), `test_acc_indist`, and `shift_drop`, and the drop
  is the quantity an invariance claim rests on. Both test sets are drawn
  from the same seed and split, so they differ only in the distribution
  parameters -- paired samples, which tightens the estimate of the drop.
- **A shift that changes nothing is an error.** An override block that
  leaves the two distributions identical, or that names a field
  `GenerationConfig` does not have, raises rather than silently reporting
  "no degradation" for the wrong reason.

The pixel baseline must be run with augmentation matched to the shift
(the `strong` preset for position/scale), not only un-augmented. The
primitive graph is invariant by construction, so a gap measured only
against an un-augmented CNN is an augmentation result wearing a
representation result's clothes.

## Adding a condition

A new experimental condition is a YAML file, not code. Any
`GenerationConfig` field named in the experiment config is passed through,
so a position/scale-shift condition is a pair of configs differing in
`object_scale_range` and `position_jitter`. Models are declared in the
config's `models:` list (`kind: cnn` with an `augment` preset, or
`kind: gnn` with an `extractor`), and `sweep.n_train_per_class` turns any
config into a learning-curve sweep.
