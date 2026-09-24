# Phase 1 results: clean 2D drawings

PLAN.md section 5. Ten classes built from circles, triangles, rectangles
and lines, rendered flat on white at 64 px, dataset version 2 (`arrow_sign`
is the exact vertical flip of `tree`). Every number below is 3 seeds, mean
± Student-t 95% CI. The protocol is in
[`docs/experiment_protocol.md`](../../docs/experiment_protocol.md):
- selection on validation, with the test set touched once;
- a 500-step training floor, with validation at most ~60 times per run;
- precise BatchNorm;
- the same schedule for every model.

Runs recorded before the last three rules existed are kept under
`superseded_*/` and were re-run.

**Models.**
- Pixel models:
  - ResNet-18 from scratch, with and without random-affine augmentation (and a "strong" preset covering the shift range);
  - ImageNet ResNet-18 fine-tuned;
  - a frozen ImageNet linear probe.
- Primitive models:
  - GINE-style GNN (mean pooling; max pooling as an ablation);
  - set transformer;
  - bag-of-primitives ablation (per-type counts and sizes, no positions or relations).
- Extractors: oracle (the generator's own primitives) and classical (OpenCV).

## The "done when" check

| PLAN.md criterion | Outcome |
|---|---|
| Sample grids look right; tests verify primitives match pixels | **Met.** `sample_grid.png`, `sample_grid_novel.png`; `tests/test_synth_dataset.py`, `tests/test_transforms.py` |
| Primitive model (oracle and classical) beats the CNN on few-shot and position/scale shift | **Partly met.** Only against the *un-augmented* CNN. With oracle primitives the GNN ties the augmented and pretrained CNNs on few-shot and wins the shift. With classical extraction it loses both to augmented CNNs. |
| Bag ablation fails on tree vs arrow sign while the GNN succeeds | **Met.** Bag 0.475–0.500 on the pair at every training size; GNN 1.000. |
| Classical extractor near-perfect on clean drawings | **Not met.** F1 0.78 at 64 px, 0.83 at 128 px. It misses thin lines (recall 0.17–0.30) where they merge into the part they touch. See `classical_extractor/`. |

Decision (recorded with a Fable-subagent review): proceed to Phase 2.
The infrastructure gate is met. The hypothesis gate failing *is* the
Phase 1 result: `docs/research_plan.md` §6 pre-registered exactly this
outcome ("a pretrained linear probe matches the oracle GNN on few-shot:
pivot to the shift and composition axes"). Extraction quality is treated
as a variable in Phases 2–3 (`classical_v2`, the learned detector), not
hand-tuned further here.

## Few-shot learning curves (`learning_curve/`)

Test accuracy by training examples per class:

| model | 5 | 10 | 25 | 50 | 100 | 1000 |
|---|---|---|---|---|---|---|
| CNN | 0.553 ± 0.084 | 0.838 ± 0.023 | 0.969 ± 0.031 | 0.999 | 0.999 | 1.000 |
| CNN + aug | **0.999** ± 0.002 | 0.998 | 1.000 | 1.000 | 0.999 | 1.000 |
| ImageNet fine-tune | **1.000** | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| ImageNet probe | 0.996 | 0.999 | 0.999 | 1.000 | 0.999 | 1.000 |
| GNN (oracle) | **0.999** ± 0.001 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| GNN (oracle, max pool) | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| Set transformer (oracle) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| GNN (classical) | 0.884 ± 0.039 | 0.927 ± 0.024 | 0.964 ± 0.042 | 0.986 | 0.989 | 0.999 |
| Bag (oracle) | 0.898 | 0.899 | 0.895 | 0.899 | 0.899 | 0.900 |

H1 (sample efficiency) holds only against a CNN trained without
augmentation. Standard augmentation alone takes a from-scratch ResNet to
0.999 at 5 examples per class. Caveat: the validation set (40/class) is
larger than the training set at the few-shot end, for every model alike.

## Position/scale shift (`shift_position_scale_fewshot/`, `shift_position_scale/`)

Train on small, centred objects; test on large, off-centre ones.

| model | shifted, 25/class | shifted, 150/class | drop at 25 |
|---|---|---|---|
| GNN / set transformer / max-pool GNN (oracle) | **1.000** | **1.000** | 0.000 |
| CNN + strong aug | 0.936 ± 0.005 | 0.936 ± 0.036 | 0.062 |
| ImageNet probe | 0.864 ± 0.073 | 0.893 ± 0.121 | 0.133 |
| GNN (classical) | 0.843 ± 0.066 | 0.879 ± 0.038 | 0.075 |
| CNN + aug | 0.726 ± 0.045 | 0.930 ± 0.112 | 0.271 |
| CNN | 0.206 ± 0.053 | 0.227 ± 0.115 | 0.784 |
| Bag (oracle) | 0.900 | 0.901 | 0.000 |

The oracle graph is invariant by construction: every feature is
normalised by the object's own bounding box. Augmentation that covers the
test range closes most of the gap. With a real extractor, the ceiling is
extraction quality (F1 0.79), not the shift.

## Novel compositions (`novel_composition/`, `novel_composition_pooling/`)

Train on each class's base template; test on held-out variants that add
or remove a part (a car with three windows, a snowman with a hat, a cat
with eyes). All models score ≥ 0.99 in-distribution, except the bag
(0.90).

| model | held-out variants | drop |
|---|---|---|
| ImageNet probe | **0.980** ± 0.018 | 0.020 |
| CNN + aug | 0.911 ± 0.082 | 0.087 |
| CNN | 0.902 ± 0.208 | 0.096 |
| Set transformer (oracle) | 0.863 ± 0.077 | 0.137 |
| Bag (oracle) | 0.736 ± 0.101 | 0.165 |
| GNN (oracle, max pool) | 0.695 ± 0.220 | 0.305 |
| GNN (classical) | 0.694 ± 0.012 | 0.293 |
| GNN (oracle, attention pool) | 0.527 ± 0.311 | 0.473 |
| **GNN (oracle, mean pool)** | **0.490 ± 0.170** | 0.510 |

This is the clearest loss for the primitive model. The GNN has learned a
part-*type* shortcut: a snowman with arms (lines) is called a bicycle,
and a snowman with a hat (a rectangle) a truck. It does so under every
readout. Max pooling fixes the variants that repeat an existing part type
(house 0.46 → 0.93, arrow sign 0.33 → 1.00). Attention pooling does not
help. The set transformer, with the same inputs, is far more robust
(0.86), so the failure is specific to the GNN's inductive bias, not to
the primitive representation.

## Relation test (every experiment)

Tree vs arrow sign is the same two parts in a flipped arrangement.
- Every model except the bag reaches 1.000 on the pair at ≥ 25
  examples/class.
- The bag stays at chance (0.475–0.500) at every size.
- At 5/class the plain CNN is at 0.60 and the classical GNN at 0.84,
  while the oracle GNN is at 1.000.

Relations are necessary for this pair, and every relational model uses
them.

## What this means for the hypotheses

- **H1 (sample efficiency):** not supported against augmented or pretrained pixel models on clean data.
- **H2 (relations matter):** supported. The bag fails exactly on the relation twins.
- **H3 (viewpoint), position/scale part:** supported for oracle primitives, but largely matched by augmentation. Phase 2 tests the harder part (rotation, perspective).
- **Composition:** the pixel models generalise better. The GNN's part-type shortcut is a new failure mode to carry into later phases.

## Cost

| | parameters | train time at 150/class (CPU) |
|---|---|---|
| GNN | 57,549 | seconds |
| CNN | 11.2 M | 5–8 min |
