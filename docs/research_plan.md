# Research plan: from "shape-primitive recognition works" to a publishable paper

Status of this document: a plan, not a result. It was written after reading
PLAN.md, the code under `src/shapeprim`, `docs/phase1_evaluation_next_steps.md`
and `results/phase1/cnn_vs_gnn.json`, and after a literature sweep
(`docs/literature_review.md`). Real-image data (PLAN.md Phase 3c) is out of
scope here by request.

---

## 1. Where the project stands

What exists and works:

- A deterministic synthetic generator: 10 classes, each a fixed template of
  4 primitive types (circle, triangle, rectangle, line), with object
  scale/position jitter, part jitter, colour jitter and optional distractors.
  Ground-truth primitives match the pixels by construction (tested).
- Two extractors: oracle (ground truth) and classical (OpenCV threshold +
  connected components + polygon/ellipse fit). Classical reaches F1 >= 0.65
  on 8 classes and only >= 0.2 on `bicycle` and `person`.
- A bbox-normalised, fully connected primitive graph with 10 node and 11
  edge features; translation and scale invariance is verified by unit tests.
- A GINE-style GNN and a ResNet-18 baseline, a shared training loop, and one
  comparison run: all three models reach ~100 % on clean in-distribution
  data (CNN 100 %, GNN+oracle 100 %, GNN+classical 99.5 %).

What the existing docs already list as missing (Phase 1 "done when"):
learning curves, position/scale shift, novel compositions, the bag-of-shapes
ablation. Those are necessary but, as argued below, not sufficient.

---

## 2. What is missing for a publishable paper

The items are ordered by how likely they are to sink a submission.

### 2.1 The main result is currently true by construction

The synthetic classes are *defined* as primitive templates, and the graph
features are normalised by the object's own bounding box. Therefore:

- The oracle GNN is handed the exact latent variables the generator sampled
  from. Beating a pixel model from 5 examples/class with that input is not
  evidence about representations; it is evidence that the generator is
  invertible.
- Position/scale invariance is a property of `graph/build.py`, not something
  the model learns. Experiment 2 in `docs/phase1_evaluation_next_steps.md`
  cannot fail for the oracle GNN, and the doc says so.
- The four hypotheses (H1-H4) all predict "primitive model wins". A paper
  needs conditions under which it loses, and a boundary between the two.

A reviewer will write: "the benchmark is generated from the representation
being evaluated". The fix is not to abandon synthetic data (control is the
point) but to change the *question* from "does structure win" to "under
which measurable conditions does structure win, and where does it stop".
Section 3 does this.

### 2.2 The pixel baseline is too weak to make any claim against

PLAN.md §4 specifies "ResNet-18, from scratch and ImageNet-pretrained, with
standard augmentation". The code has neither augmentation nor a pretrained
run in the comparison. Concretely:

- No data augmentation at all (`ImageClassificationDataset`). A CNN with
  random affine augmentation will very likely erase the position/scale
  result on its own. If the H3-position result only holds against an
  un-augmented CNN, it is an augmentation result, not a representation
  result.
- No validation split. `final_acc` is the last-epoch *test* accuracy and the
  test set is monitored every epoch; hyperparameters were implicitly tuned on
  it. Any few-shot claim made this way is inadmissible.
- Training is unstable (test accuracy drops to 18 % at epoch 1 and 61 % at
  epoch 11 in `cnn_vs_gnn.json`): Adam at 1e-3 with no schedule on 1500
  images. The CNN numbers currently measure training hygiene as much as the
  model.
- The CNN never sees the object's bounding box, while the GNN gets it for
  free through normalisation. A "CNN on bbox-cropped input" baseline
  separates "normalisation helps" from "structure helps".
- Missing baselines a reviewer will ask for: parameter- or compute-matched
  small CNN (ResNet-18 has 11 M parameters, the GNN ~50 k); an ImageNet-
  pretrained backbone and/or CLIP linear probe for the few-shot regime
  (that is what a practitioner would actually use with 10 examples/class);
  a small ViT; a Relation Network (CNN features + relational module) as the
  natural midpoint between the two extremes; a set model over primitives
  *without* edges (DeepSets) and *with* attention (set transformer, already
  in PLAN.md) so that conclusions do not hinge on one GNN.

### 2.3 Extraction quality is an accident, not a variable

The whole argument rests on stage 1. Today there are two points on the
extractor-quality axis (oracle = perfect, classical = whatever OpenCV gives),
and the classical one is class-dependent (bicycle/person fail). Missing:

- A parametric noise model on top of the oracle: drop a primitive with
  probability p_drop, add a spurious one with p_add, mistype with p_type,
  perturb pose with sigma, merge two touching parts. This turns extraction
  quality into a dial and lets the paper draw the accuracy-vs-extractor-F1
  curve instead of two anecdotes.
- The learned extractor (PLAN.md `extract/learned.py`) does not exist. It is
  the only extractor that could plausibly be used outside this repo, and
  its own data cost (primitive-level labels) has to be accounted for.
- Extraction precision/recall is never reported next to classification
  accuracy in the results file, although PLAN.md §4 requires it.

### 2.4 The dataset cannot support the generalisation claims

- 10 classes, one template each, no variants. The novel-composition test
  needs held-out arrangements; with one template per class there is nothing
  to hold out. Statistical power for a 10-class, 3-seed study is low.
- No object-level rotation is generated at all, so "viewpoint" (H3) is
  untested even in its weakest form. Note the graph is *not* rotation
  invariant (absolute rot sin/cos, absolute dx/dy, above/below flags), so
  rotation is the first place the oracle GNN can genuinely fail.
- No appearance variation beyond a +/-15 colour jitter: H4 is untestable.
- The distractors are unrelated shapes; there is no occlusion, no touching
  parts, no clutter that a real extractor would struggle with.

### 2.5 Rigour and reporting

- 3 seeds with mean +/- std is the PLAN.md rule; the results file has one.
- No hyperparameter search on either side; equal tuning budget must be
  documented.
- No cost accounting: wall-clock, parameters, and above all *annotation*
  cost. The GNN pipeline's few-shot advantage is partly paid for with
  primitive-level supervision (oracle) that the CNN never gets. The paper
  must state label budgets per model.
- No analysis of *why*: which edges the GNN uses on the tree/arrow pair;
  whether the CNN uses arrangement at all (part-shuffled or "Frankenstein"
  images in the style of Baker et al. 2018 / Malhotra et al. 2022).

### 2.6 Positioning

PLAN.md §9 lists 11 references and misses the lines of work most likely to
be raised: relation networks and Sort-of-CLEVR (a GNN/relational module
over object features vs a CNN), neuro-symbolic pipelines with a perception
stage feeding a reasoner (NS-VQA, NS-CL) and studies of how perception
errors propagate, object-centric representation learning and its measured
effect on OOD generalisation (Dittadi et al. 2022), pictorial structures and
attributed relational graphs (the 1970s-2000s version of exactly this
pipeline), sketch-graph recognition (SketchGNN etc.) and Ellis et al. 2018
who infer primitive programs from drawings. See `docs/literature_review.md`.
The "generic primitives, not semantic parts" novelty is real but narrow;
the defensible novelty is the *controlled tradeoff study*, which nobody in
that list has done.

### 2.7 Engineering gaps (small, but blocking)

Sweep driver, results directory convention from PLAN.md §4, plotting,
caching of extracted graphs (the classical extractor is re-run every epoch),
GPU support (`device` is hard-coded to CPU in the script), and
`GenerationConfig` in `compare_cnn_gnn.py` ignoring `configs/phase1/data.yaml`.

---

## 3. The research question

### 3.1 Motivation from practice

There is a large family of real recognition problems where objects are, by
definition, arrangements of a small shared vocabulary of geometric symbols:
engineering and process diagrams (P&ID, circuit schematics, floor plans),
road signs and pictograms, icons and logos, whiteboard and children's
sketches, musical and chemical notation. In all of them, a team building a
recogniser faces the same decision: train an end-to-end image classifier on
whatever labelled examples exist, or build a two-stage system (symbol
detector -> structured reasoner) that is more data-efficient and more
interpretable but fails whenever the detector does. Practitioners lack any
quantitative guidance for that decision. The folklore says "structure helps
when data is scarce and hurts when perception is noisy", and nobody has
measured where the crossover lies.

### 3.2 The question

> **For objects defined by the spatial arrangement of a small, shared
> vocabulary of geometric primitives, under what combination of (a)
> labelled examples per class, (b) primitive-extraction error, and (c)
> train-test shift does an explicit primitive-graph recogniser outperform an
> end-to-end image classifier, and where is the break-even?**

The deliverable is a **break-even map**: on axes of examples-per-class and
extractor F1, for each shift condition, the contour where the structured
pipeline and the best pixel baseline tie, plus the mechanism behind its
shape. This converts the four PLAN.md hypotheses from four "we win" claims
into the axes of one falsifiable surface:

| PLAN.md hypothesis | Becomes |
|---|---|
| H1 sample efficiency | the examples-per-class axis |
| H2 relations matter | the diagnostic that explains *why* the surface has the shape it has (bag ablation, relation-twin classes, part-shuffling) |
| H3 viewpoint | one family of shift conditions (position/scale, rotation, novel composition) |
| H4 appearance | the second family of shift conditions, and the argument that appearance shift is *relocated* into stage 1 rather than removed |

### 3.3 Pre-registered predictions (to be written down before running)

Stating these in advance is what makes the study a test rather than a demo.

- **P1.** With oracle primitives the GNN needs at least 5x fewer examples
  than the strongest pixel baseline (augmented, pretrained) to reach 95 %
  on the base task. *Falsified if* a pretrained linear probe matches it.
- **P2.** The advantage shrinks monotonically with extraction error, and
  break-even lies at extractor F1 between 0.7 and 0.85 for the 25-
  examples-per-class regime. *Falsified if* break-even is above 0.95
  (structure never pays without a near-perfect extractor) or below 0.5.
- **P3.** Random-affine augmentation closes the position/scale gap
  entirely; the gap persists under rotation only if the graph is made
  rotation-canonical, and persists under novel compositions regardless.
- **P4.** The bag model is near chance on relation-twin pairs and fine
  elsewhere; the CNN is fine on twins in-distribution but degrades far more
  than the GNN on part-shuffled images (it uses arrangement weakly).
- **P5.** Under appearance shift, oracle-GNN accuracy is unchanged (it is
  blind to appearance by construction) and the *entire* effect appears in
  extractor F1. Whether the pipeline wins is then decided by whether a
  class-agnostic extractor is easier to make style-robust than a
  class-specific classifier. Prediction: yes, because the extractor's
  training signal is shared across all classes.

---

## 4. Work packages

Estimates assume one person with a single GPU; the GNN side is CPU-cheap.

### WP0. Experimental hygiene (1 week)

Blocks every later result.

1. Train/val/test split with a fixed validation seed; model selection and
   early stopping on val only; test evaluated once per run.
2. CNN training: cosine or step schedule, warm-up, weight decay sweep; SGD
   and Adam both tried; document the budget.
3. Standard augmentation for the CNN (random affine: translate, scale,
   optionally rotate; horizontal flip *off*, since it changes some classes)
   as a config flag, with results reported with and without.
4. Sweep driver that writes `results/<phase>/<experiment>/<run_id>/`
   (config, seed, metrics, extractor P/R/F1) per PLAN.md §4, and one plotting
   script for learning curves and shift tables.
5. Cache extracted graphs to disk keyed by (generator config, seed, index,
   extractor name); GPU device flag.
6. 3 seeds minimum, 5 for any headline number; report mean +/- 95 % CI.

### WP1. Benchmark v2: a primitive grammar, not ten templates (2-3 weeks)

Goal: enough classes and controlled variation to support compositional and
shift claims, while keeping the interpretable 10.

1. **Two class families.** Keep the 10 hand-designed classes (Family A,
   interpretable figures). Add Family B: N (aim 40-60) procedurally
   generated classes, each defined by a small relational program over 2-6
   primitives (e.g. "A above B, C inside A, B same-size-as C"). Programs are
   sampled with constraints that guarantee every class has at least one
   **relation twin**: another class with the identical multiset of primitive
   types but a different arrangement. Twins generalise the tree/arrow_sign
   trick to the whole benchmark and make H2 testable with statistical power.
2. **Variants per class** for the novel-composition split: each program
   admits a family of instantiations (part counts, optional parts, allowed
   arrangement perturbations); hold out a subset of instantiations per class
   as in CLEVR-CoGenT.
3. **Nuisance dials as generator parameters**, each off by default and each
   with a paired ground truth: object rotation (with a documented cap for
   symmetric twins), stroke-only vs filled rendering, stroke width, colour
   palette, texture fill, background clutter, partial occlusion, blur/noise.
4. **A shift battery** built from the dials, always as train-config /
   test-config pairs: position-scale, rotation-extrapolation (train 0-30°,
   test 30-90°), style (filled -> outline; flat -> textured), clutter,
   occlusion, novel composition, novel part count.
5. Sample grids and the existing pixel-vs-ground-truth tests for every dial.

### WP2. Models and baselines (2 weeks)

Pixel side:
- ResNet-18 from scratch, with and without augmentation (WP0).
- ResNet-18 on bbox-cropped input (gives the CNN the normalisation the graph
  has).
- Parameter-matched small CNN (~50-100 k params).
- ImageNet-pretrained ResNet-18 fine-tune and a frozen CLIP/DINO linear
  probe: the honest few-shot baseline.
- Small ViT (optional, if compute allows).
- Relation Network (Santoro et al. 2017) over CNN feature-map cells: the
  midpoint that has relations but no explicit primitives.

Primitive side:
- GINE GNN (exists).
- Set transformer over nodes (PLAN.md), to show results are not GINE-specific.
- Bag ablation (`models/bag.py`, PLAN.md).
- Ordered-list MLP over primitives sorted by a fixed rule: tests whether
  permutation-invariant structure matters or any symbol list would do.

Graph ablations (config flags in `graph/build.py`): no edge features; no
bbox normalisation (does the model *learn* invariance?); k-NN instead of
full connectivity; rotation-canonical frame (principal axis) for the
rotation experiments; drop the `inside` flag.

### WP3. Extraction as an experimental variable (2 weeks)

1. **Noisy oracle**: a wrapper extractor with parameters p_drop, p_add,
   p_type, sigma_pose, p_merge. Report the resulting F1 against ground truth
   with the existing `eval_match`, so every run is placed on the F1 axis.
2. **Calibrate** the noise model against the classical extractor's empirical
   error profile per class (what it drops, what it mistypes), and verify that
   the noisy oracle at matched F1 reproduces the classical extractor's
   downstream accuracy. This is what licenses reading the classical point off
   the curve.
3. **Learned extractor** (`extract/learned.py`): a small CenterNet-style
   detector (heatmap per type + pose regression) trained on synthetic
   images. Train it once, class-agnostically; report F1 in-distribution and
   under every shift in WP1. Also measure how many primitive-annotated images
   it needs (its own learning curve): this number is the hidden cost of the
   pipeline and belongs in the paper.
4. Keep the classical extractor as the "no training data at all" point.

### WP4. Core experiments (3-4 weeks, mostly compute)

- **E1 Learning curves.** n in {5, 10, 25, 50, 100, 250, 1000}/class, all
  models, both families, in-distribution. Report examples-to-95 % per model.
- **E2 Relation diagnostics.** Accuracy on twin pairs vs non-twins for every
  model; part-shuffled and part-swapped test images for the pixel models;
  edge-feature ablation for the graph models; per-edge attribution on the
  twins (gradient x edge-feature or a simple GNNExplainer).
- **E3 Shift battery.** Every train/test pair from WP1.4, all models, at
  n = 25 and n = 1000. Pixel models with and without augmentation that
  matches the shift (so the reader can see which gaps are augmentation
  artefacts).
- **E4 Break-even map (the headline).** n x extractor-F1 grid (noisy oracle)
  x shift condition; plus the classical and learned extractors placed as
  points. Output: for each shift, the contour where GNN = best pixel model,
  with CIs from seeds.
- **E5 Architecture check.** Repeat E1 and E4 at reduced resolution with the
  set transformer; confirm the contour moves little.
- **E6 Cost accounting.** For every cell: parameters, train wall-clock,
  inference wall-clock, class labels used, primitive labels used.

### WP5. Analysis and writing (2-3 weeks)

- Failure atlas: where extraction breaks (thin parts, touching parts,
  occlusion) and what it does downstream; what the CNN confuses under shift.
- The relocation argument for H4: show accuracy-vs-shift for the extractor
  alone next to the classifier alone.
- Limitations section written first: synthetic; primitives chosen to match
  the generator; annotation cost of the learned extractor; rotation
  symmetry; scope of "objects defined by arrangement".
- Target venues: TMLR (welcomes careful tradeoff studies and nuanced
  outcomes), NeurIPS Datasets & Benchmarks (if Family B is released as a
  benchmark), or a CogSci / CCN paper if the framing leans on
  recognition-by-components.

### WP6 (optional, recommended). A real-data anchor without photographs

Hand-drawn sketches are the cheapest real data in which primitive
decomposition is natural, and PLAN.md already lists them (Phase 3b).
Quick, Draw! ships stroke sequences, so a stroke-to-primitive fitter gives a
non-oracle, non-synthetic extractor whose F1 can be estimated on a small
hand-labelled subset. Experiments: synthetic-to-sketch transfer, few-shot on
sketches, and placing the sketch extractor on the E4 map. Even a small
version of this turns "synthetic study" into "synthetic study whose
break-even prediction was checked once on real drawings".

---

## 5. Order of work and milestones

1. WP0 -> re-run the existing comparison with val split, augmentation, 3
   seeds. *Milestone: an honest Phase 1 table.* Expect the position/scale gap
   to shrink or vanish for the augmented CNN; that is the first real finding.
2. WP3.1-3.2 (noisy oracle + calibration) -> E4 on the existing 10 classes.
   *Milestone: a first break-even curve.* This is the earliest point at which
   the paper's core figure exists; if the curve is uninteresting (break-even
   only at F1 > 0.95) the plan needs revisiting before investing in WP1.
3. WP1 (grammar, twins, dials) and WP2 (baselines) in parallel.
4. WP3.3 learned extractor, then E1-E3, E5, E6 on Family A + B.
5. WP5 writing, WP6 if time allows.

## 6. What would make this not worth publishing

Writing this down so the decision at milestone 2 is explicit:

- Break-even requires F1 > 0.95 under every condition: then structure only
  pays with an oracle, and the paper becomes a negative result (still
  publishable at TMLR, but framed differently).
- A pretrained linear probe matches the oracle GNN on few-shot: then H1 is an
  artefact of training from scratch, and the study should pivot to the shift
  and composition axes only.
- The noisy-oracle calibration fails (noise at matched F1 does not reproduce
  the classical extractor's downstream accuracy): then the F1 axis is not a
  sufficient statistic and the map has to be drawn per error *type*.
