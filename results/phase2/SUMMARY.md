# Phase 2 results: viewpoint transformations

PLAN.md section 6. The Phase 1 drawings are seen through a planar map
applied to the primitives themselves, so the ground truth stays exact
(`src/shapeprim/data/transforms.py`). The map is one of:
- a perspective tilt, with "viewing angle" running from 0° (frontal) to 70°;
- an in-plane rotation;
- shear or squash.

Circles become ellipses and rectangles become quadrilaterals. Setup:
- 64 px images, 3 seeds, mean ± Student-t 95% CI;
- the Phase 1 protocol: 500-step floor, precise BN, validation on the
  training distribution.

**Two ways to handle viewpoint** (PLAN.md's key design decision):
- *learned* invariance: the bbox-normalised graph, where the model has to
  learn viewpoint from examples;
- *built-in* invariance: `graph_frame: affine`. Each object's second-moment
  matrix is whitened, which removes shear and squash exactly and
  perspective foreshortening to first order. Orientation is kept on
  purpose, so tree and arrow sign stay distinct.

**Extractors:**

| Extractor | What it is | F1 frontal | 0–30° | 30–50° | 50–70° |
|---|---|---|---|---|---|
| oracle | the true primitives | 1.000 | 1.000 | 1.000 | 1.000 |
| `classical_v2` | OpenCV, rotated ellipses, exact quadrilateral vertices | 0.821 | 0.806 | 0.720 | 0.543 |
| learned | a small class-agnostic CenterNet, trained on 0–30° only (6,000 primitive-labelled images, 22,800 primitive labels) | 1.000 | 0.999 | 0.971 | 0.390 |

The learned detector's full report is in `learned_extractor/`.

## Done-when check

| PLAN.md criterion | Outcome |
|---|---|
| Accuracy-vs-angle plots for all models | **Met.** `angle_extrapolation/angle_curve.png`, `full_range/angle_curve_n*.png`, `relation_rotation/rot_curve.png` |
| Clear whether the advantage comes from the representation or is lost in extraction | **Met: both, cleanly separated.** With oracle primitives the graph extrapolates almost perfectly. With either real extractor the advantage is lost, because the extractors degrade exactly where the pixel models do. |

## Angle extrapolation: train 0–30°, test up to 70° (`angle_extrapolation/`, 150/class)

| model | 30–40° | 40–50° | 50–60° | 60–70° |
|---|---|---|---|---|
| **GNN (oracle, affine frame)** | 0.999 | 0.998 | **0.997** | **0.996** |
| GNN (oracle) | 0.998 | 0.994 | 0.986 | 0.933 |
| Set transformer (oracle) | 0.998 | 0.984 | 0.945 | 0.866 |
| Bag (oracle) | 0.898 | 0.895 | 0.895 | 0.880 |
| CNN | 0.999 | 0.991 | 0.912 | 0.602 |
| CNN + view aug (shear + perspective) | 0.998 | 0.988 | 0.899 | 0.593 |
| CNN + aug | 0.996 | 0.963 | 0.820 | 0.522 |
| ImageNet probe | 0.994 | 0.952 | 0.696 | 0.381 |
| GNN (classical, affine frame) | 0.941 | 0.888 | 0.778 | 0.616 |
| GNN (classical) | 0.934 | 0.864 | 0.696 | 0.459 |
| GNN (learned) | 0.987 | 0.833 | 0.359 | 0.133 |

- **The representation extrapolates.** With the affine frame the oracle
  GNN is flat to 70°, while every pixel model falls to 0.38–0.60.
  Built-in invariance beats learned invariance: 0.996 against 0.933 at
  60–70°.
- **Extraction takes the advantage away.**
  - `classical_v2` in the affine frame ends level with the CNN (0.616 against 0.602).
  - The learned detector matches the oracle inside its training range but
    collapses past 50° (F1 0.16 at 60–70°). The GNN on its output then
    falls below every pixel model.
  - The affine frame helps the classical extractor (+0.16 at 60–70°). It
    can't help the learned one, whose primitives are simply missing.

## Full-range training and few-shot at angles (`full_range/`)

Train and test on all viewing angles 0–70°:

| model | 5/class | 10 | 25 | 150 | 60–70° bin at 5/class |
|---|---|---|---|---|---|
| **GNN (oracle, affine frame)** | **1.000** | 1.000 | 0.999 | 1.000 | **1.000** |
| GNN (oracle) | 0.990 | 0.995 | 1.000 | 1.000 | 0.954 |
| Set transformer (oracle) | 0.993 | 0.994 | 0.999 | 0.999 | 0.972 |
| CNN + view aug | 0.960 | 0.977 | 0.994 | 0.997 | 0.815 |
| CNN + aug | 0.941 | 0.968 | 0.986 | 0.995 | 0.729 |
| ImageNet probe | 0.884 | 0.927 | 0.964 | 0.985 | 0.529 |
| CNN | 0.441 | 0.687 | 0.893 | 0.996 | 0.260 |
| GNN (classical, affine frame) | 0.655 | 0.743 | 0.806 | 0.944 | 0.413 |
| GNN (learned) | 0.770 | 0.803 | 0.826 | 0.851 | 0.184 |
| Bag (oracle) | 0.891 | 0.893 | 0.896 | 0.895 | 0.886 |

Under viewpoint change, PLAN.md's H1 (sample efficiency) *does* hold
with oracle primitives. At 5 examples per class the affine-frame GNN is
at 1.000 against 0.960 for the best augmented CNN; in the steepest bin
the gap is 1.000 against 0.815. With 150 examples per class the pixel
models catch up (0.997).

With extracted primitives, the graph models trail every augmented pixel
model at every size. Caveat: the learned detector was trained only on
0–30°, so in this experiment it runs outside its training range at steep
angles, and its curve plateaus at 0.85.

## Relation test under rotation (`relation_rotation/`)

Train with in-plane rotation up to 30° (plus viewing angle 0–30°); test
on 30–60° and 60–90°. Rotation is capped at 90° because tree and arrow
sign are exact vertical flips: at 180° they are identical (PLAN.md §8).

| model | 30–60° | 60–90° | tree/arrow at 60–90° |
|---|---|---|---|
| GNN (oracle) | 0.948 | **0.807** ± 0.057 | **0.78** |
| GNN (oracle, affine frame) | 0.943 | 0.802 ± 0.113 | 0.68 |
| Set transformer (oracle) | 0.959 | 0.782 ± 0.201 | 0.66 |
| CNN + view/rotation aug (±45°) | **0.966** | 0.727 ± 0.142 | 0.54 |
| GNN (classical) | 0.900 | 0.621 ± 0.175 | 0.43 |
| CNN | 0.715 | 0.253 ± 0.076 | 0.00 |
| Bag (oracle) | 0.899 | 0.899 | 0.49 |

- **Rotation is the hardest transform for every model.** The graph is
  deliberately *not* rotation-invariant: absolute orientation separates
  the twins.
- **Oracle primitives help at the largest rotations**, but the intervals
  are wide.
- **The plain CNN swaps the twins.** It calls every rotated tree an arrow
  sign and vice versa (0.00 on the pair), which is the known symmetry
  limit reached early.
- **The affine frame does not help against rotation**, as designed: it
  removes shear and squash, not orientation.

## What this means for the hypotheses

- **H3 (viewpoint):** supported for the representation. The oracle
  affine-frame GNN extrapolates from 0–30° to 70° with no loss, while
  pixel models, even with perspective augmentation, lose 40–60 points.
  Not supported for the pipeline: both extractors break down under the
  same transforms.
- **H1 (sample efficiency) under viewpoint:** supported with oracle
  primitives (1.000 against 0.960 at 5/class), not with extraction.
- **Built-in vs learned invariance:** built-in wins wherever the
  transform is affine-like. Under rotation it neither helps nor hurts.
- **Where the effort should go:** extraction under viewpoint change. The
  learned detector's collapse past its training range repeats the pixel
  models' failure one stage earlier. Phase 3a asks the same question for
  appearance.
