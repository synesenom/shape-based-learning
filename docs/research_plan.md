# Research plan: from "shape-primitive recognition works" to a publishable paper

**Target venue: Nature Machine Intelligence** (section 7), with TMLR as the
graceful landing if a gate in section 5 fails.

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
be raised (details and corrected citations in `docs/literature_review.md`):

- **Object-centric plus relational models on synthetic shape tasks**
  (Webb, Sinha & Cohen 2021; Mondal, Webb & Cohen 2023; Webb et al. 2023,
  OCRA). These already report sample-efficiency and OOD gains of an
  extract-then-relate pipeline over CNN and transformer pixel models. This
  is the work a reviewer will cite against H1 and H3 as stated.
- **Neuro-symbolic VQA** (Yi et al. 2018 NS-VQA; Mao et al. 2019 NS-CL;
  Amizadeh et al. 2020 on disentangling perception from reasoning) and
  **relation networks** (Santoro et al. 2017): the same two-stage idea, with
  the effect of perception errors discussed but never mapped.
- **Object-centric representation robustness** (Dittadi et al. 2022): the
  most careful existing study of whether object-centric representations
  buy OOD generalisation; mixed results, and no explicit relations.
- **Pictorial structures and attributed relational graphs** (Fischler &
  Elschlager 1973; Felzenszwalb & Huttenlocher 2005; Bunke & Allermann 1983;
  Sanfeliu & Fu 1983): the pre-deep-learning version of exactly this
  pipeline, which the paper must acknowledge as lineage.
- **Sketch-graph recognition** (SketchGNN; Multi-Graph Transformer;
  Sketchformer) and **primitive/program inference from drawings** (Ellis et
  al. 2018; CSGNet): the closest engineering precedents, and the source of
  the real-data anchor in WP6.

Against that background, the "generic primitives, not semantic parts"
novelty in PLAN.md is real but narrow: it distinguishes the project from
CompositionalNets and part-based robustness work, not from slot-based or
neuro-symbolic pipelines. The defensible novelty is the *controlled
tradeoff study*: extraction quality as an experimental dial, break-even
against strong pixel baselines, relation-twin classes with causal
diagnostics, and an honest annotation budget. None of the works above does
that.

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

The sharpest version of that decision, and the one the paper should open
on, is the **new symbol set**. A notation standard is revised, a client
uses house conventions, a new schematic family appears, and the team has
five examples of each symbol and a deadline. This is a genuine few-shot
regime with real money attached, not a benchmark contrivance. It is also
exactly the regime where the two approaches are predicted to diverge and
where nobody can currently say which to choose.

### 3.1b The question behind the question: structure given vs structure learned

The practical decision is an instance of an older one. A recogniser can be
*handed* a vocabulary of parts, or made to *discover* one from examples.
This project measures the exchange rate between the two: how good a
supplied vocabulary has to be, and how scarce the data has to be, before
being handed the vocabulary wins. The oracle extractor is the limiting
case of a perfect innate vocabulary; the classical and learned extractors
are degraded versions of it.

That framing has an obvious biological echo, and the echo is worth stating
carefully because it is easy to overclaim. What the developmental
literature does and does not support:

- **Supported: architectural priors and pre-natal structuring.**
  Orientation-selective cells are present before patterned visual
  experience, and spontaneous retinal waves structure the visual system
  before the eyes open (Ackman, Burbridge & Crair 2012, *Nature*) -- an
  internally generated pre-training signal. Newborns minutes old orient
  toward a top-heavy three-blob configuration (Goren, Sarty & Wu 1975;
  Johnson & Morton 1991), which is a configural bias rather than a face
  detector. Face-selective and number-selective units emerge in *untrained*
  randomly initialised networks (Baek, Song & Paik 2021, *Nature
  Communications*; Kim, Jang & Paik 2021, *Science Advances*), showing
  selectivity can fall out of wiring statistics alone.
- **Supported: experience then refines it.** Kittens reared seeing only
  one orientation go behaviourally blind to the other (Blakemore & Cooper
  1970, *Nature* 228:477). Late-sighted patients recover acuity quickly but
  struggle with object integration (Project Prakash; Ostrovsky et al.
  2009, *Psychological Science*).
- **Not supported: a stored library of shape templates.** No evidence
  places geons, or any primitive alphabet, in cortex at birth. Biederman
  (1987) proposed geons as universal, and their innateness remains an open
  question rather than an established one.
- **The honest counterweight.** The shape bias in children -- generalising
  a new word by shape rather than colour or texture -- appears around 24
  months and tracks vocabulary growth (Landau, Smith & Jones 1988). It
  looks learned. So "shape matters" is itself acquired, even if the
  machinery that makes it quickly acquirable is not.

**How this may and may not be used in the paper.** It motivates, and it
belongs in the introduction and discussion. It is *not* a result: this
project has no human or animal data and must claim nothing about brains.
The defensible sentence is that a supplied primitive vocabulary is worth a
measurable quantity of training data under stated conditions. That is a
claim about learning systems, and it lets the reader draw the biological
inference without the paper asserting it. Reviewers punish the overreach
harder than they reward the framing.

### 3.2 The question

**Target venue: Nature Machine Intelligence**, with TMLR as the graceful
landing (section 7). That target sets the shape of the claim. A measured
tradeoff surface is a good TMLR paper; it is not an NMI paper. What raises
it is a *predictive* claim: a rule derived under controlled conditions
that correctly forecasts an outcome in a domain it was not fitted on.

> **For objects defined by the spatial arrangement of a small, shared
> vocabulary of geometric primitives, can the point at which an explicit
> primitive-graph recogniser overtakes an end-to-end image classifier be
> predicted in advance, from two quantities a practitioner can measure
> before training either model: the number of labelled examples per class
> and the quality of the primitive extractor?**

The study has three parts, and the third is what distinguishes it:

1. **Measure.** On synthetic data where extraction quality is a controlled
   dial, map the break-even surface over (examples per class) x (extractor
   F1) x (shift condition).
2. **Compress.** Fit a compact description of the crossover -- the
   smallest expression in those two measurable quantities that locates it
   within its confidence interval. A rule travels; a table does not.
3. **Predict, then test.** For a real domain, measure *only* the extractor
   F1 and the available dataset size. Register the predicted winner and
   the predicted margin. Then train both models and report whether the
   prediction held. The registration happens before the downstream
   training runs, and the registered file is committed with a timestamp.

Part 3 is the paper's backbone. Reporting a synthetic surface and some
real numbers side by side is a benchmark study. Forecasting the real
outcome from the synthetic law, and being right, is a result about how
structured and end-to-end recognition trade off in general. If the
forecast fails, that is reportable too, and it tells you the surface is
not governed by those two quantities alone, which is itself informative.

This converts the four PLAN.md hypotheses from four "we win" claims
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
- **P6 (the out-of-domain forecast).** The crossover fitted on synthetic
  data predicts the winner on each real domain, and the predicted margin
  falls inside the measured confidence interval. *Falsified if* the
  predicted winner is wrong on any domain, or the margin is outside its
  interval on most. This prediction is registered per domain, in a
  committed file, before the downstream models are trained.

---

## 4. Work packages

Estimates assume one person with a single GPU; the GNN side is CPU-cheap.

### WP0. Experimental hygiene (1 week) -- BUILT

Blocks every later result. Protocol and caveats:
`docs/experiment_protocol.md`. Implementation:
`scripts/run_experiment.py`, `scripts/plot_results.py`,
`src/shapeprim/experiment.py`, `src/shapeprim/data/augment.py`.

1. **Done.** Train/val/test drawn from disjoint streams (`split`
   participates in the per-sample seed derivation), model selection and
   early stopping on validation, best-validation weights restored, test
   evaluated once per run.
2. **Done.** Cosine and step schedules, warm-up, gradient clipping, and a
   choice of Adam/AdamW/SGD. The budget is an explicit config
   (`configs/phase1/tuning.yaml`): an identical four-trial grid over
   (optimizer, learning rate, weight decay) for each side, selected on
   validation.
3. **Done.** `AugmentConfig` with presets, reported with and without. Fill
   colour is the generator background rather than black, and rotation past
   90 degrees is refused because it maps `tree` onto `arrow_sign`.
4. **Done.** Sweep driver writing `results/<phase>/<experiment>/<run_id>/`
   with config, environment, git commit, seed, metrics and extractor
   P/R/F1, plus a plotting/table script.
5. **Done.** Extraction cached by a digest of (classes, count, generation
   config, seed, split, extractor), persisted to disk; `--device` flag with
   `auto` resolution.
6. **Done.** Three seeds by default, aggregated with a Student-t 95%
   interval (4.303 at three seeds, not 1.96); a single-seed result is
   labelled as such rather than given a zero-width interval.

Found while building it: the dataset wrappers indexed labels against the
global 10-class vocabulary, so any experiment on a class subset emitted
labels past the classifier's output layer. Fixed, with regression tests.

Still open in WP0's spirit, deferred to the work packages that need them:
a train-on-A / test-on-B option in the driver (WP1.4's shift battery), and
the matched-capacity baseline that makes the 11.2M-vs-57k parameter gap
defensible (WP2).

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
- Optional but closest competitor: Slot Attention extractor with the same
  GNN/set-transformer head (Locatello et al. 2020; Mondal et al. 2023).
  Learned objects instead of geometric primitives, no primitive labels
  needed; if it matches the oracle-primitive pipeline, the "generic
  primitives" claim loses its force and the paper must say so.

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

### WP5. The crossover rule (1-2 weeks, after E4)

Turn the measured surface into something that can be carried to a new
domain. This is the step that separates the NMI version from the TMLR
version, and it must be done *before* any real-domain model is trained.

1. **Fit.** Find the smallest expression in (examples per class, extractor
   F1) that locates the crossover inside its seed-derived confidence
   interval across the shift conditions. Candidates in increasing
   complexity: a constant F1 threshold; a threshold linear in log n; a
   two-term form with a shift-condition offset. Prefer the simplest that
   fits, and report the residuals of the ones rejected.
2. **Cross-validate within the synthetic world first.** Fit on Family A
   (the 10 hand-designed classes), predict Family B (the procedural
   grammar), and vice versa. A rule that cannot cross that much smaller
   gap will not cross a real one, and this check is cheap.
3. **State the rule's domain of validity** explicitly: primitive
   vocabulary size, parts per object, class count, what "extractor F1"
   was measured against. A rule quoted outside its range is worse than no
   rule.
4. **Write the registration protocol** used in WP6: what is measured on a
   new domain, how the prediction is computed, what counts as a hit.

### WP6. Real-domain validation (4-6 weeks) -- CORE, not optional

The single largest lever on the venue. One real domain checked against a
registered prediction is worth more than any amount of additional
synthetic conditions. Three domains, in ascending order of cost and of
stakes; two are the minimum for the claim that the rule travels.

1. **Hand-drawn sketches (Quick, Draw!).** Cheapest real data where
   primitive decomposition is natural, and PLAN.md already lists it
   (Phase 3b). Stroke sequences give a non-oracle, non-synthetic extractor
   (stroke-to-primitive fitting) whose F1 is estimable on a small
   hand-labelled subset. Ten classes overlap the existing templates. Run
   this first: it is the cheapest possible test of whether the rule
   travels at all, and it is the bridge to the developmental framing in
   section 3.1b.
2. **Icons and logos under style shift (Icons-50, LLD).** Born-digital
   geometry, and the same icon rendered by different vendors is a
   ready-made appearance shift with no rendering tricks. Tests H4 on real
   data.
3. **Engineering or process diagrams (floor plans, P&ID, circuits) -- the
   headline application.** Highest stakes, and the domain the introduction
   is motivated by. Meaning lives entirely in the arrangement of a shared
   symbol vocabulary, and symbol detection followed by graph reasoning is
   already the industrial pipeline, so the comparison is the one
   practitioners actually face. This is where the new-symbol-set framing
   and the annotation-cost accounting land hardest, because someone is
   paying for those labels. Most expensive: annotation is scarce and
   licensing varies, so scope it early.

For each domain, in this order, and the order is the method:

- Measure the extractor's F1 on a hand-labelled subset, and the available
  examples per class. Nothing else.
- Compute and **register** the predicted winner and margin from the WP5
  rule. Commit the registration file; its timestamp is the evidence.
- Only then train the pixel and graph models and report the outcome
  against the registration.

Also required here, and cheap: the **annotation-cost accounting** that
makes the result actionable. Primitive labels needed to reach a given
extractor F1, against class labels needed for the pixel model to match.
That converts the rule from a curiosity into a budgeting tool.

### WP7. Analysis and writing (3-4 weeks)

- Failure atlas: where extraction breaks (thin parts, touching parts,
  occlusion) and what it does downstream; what the CNN confuses under shift.
- The relocation argument for H4: show accuracy-vs-shift for the extractor
  alone next to the classifier alone.
- The figure the paper is remembered by: the crossover surface with the
  real domains plotted as points, each showing predicted against measured.
- Limitations written first: synthetic origin of the rule; primitives
  chosen to match the generator; annotation cost of the learned extractor;
  rotation symmetry; scope of "objects defined by arrangement"; and the
  domains where the rule is *not* claimed to hold (natural photographs).

---

## 5. Order of work and milestones

1. WP0 -> re-run the comparison with val split, augmentation, 5 seeds.
   *Milestone: an honest Phase 1 table.* Done; see
   `docs/experiment_protocol.md`. Expect the position/scale gap to shrink
   or vanish for the augmented CNN.
2. WP3.1-3.2 (noisy oracle + calibration) -> E4 on the existing 10 classes.
   *Milestone: a first break-even curve.* Earliest point at which the
   paper's core figure exists. **Gate:** if break-even only occurs above
   F1 0.95, the NMI framing is not available and the project should
   re-target before investing in WP1 and WP6.
3. WP1 (grammar, twins, dials) and WP2 (baselines) in parallel.
4. WP3.3 learned extractor, then E1-E3, E5, E6 on Family A + B.
5. WP5 rule fitting. **Gate:** if no simple form fits, or the Family A ->
   Family B cross-prediction fails, the predictive claim is unavailable
   and the paper becomes the TMLR tradeoff study.
6. WP6 real domains, registrations first. *Milestone: the first
   out-of-domain forecast, hit or miss.*
7. WP7 writing.

The two gates exist so that the expensive half (WP1, WP6) is only paid for
once the cheap half has shown the claim is reachable.

## 6. What would make this not worth publishing

Writing this down so the decisions at the gates are explicit:

- Break-even requires F1 > 0.95 under every condition: then structure only
  pays with an oracle, and the paper becomes a negative result (still
  publishable at TMLR, but framed differently).
- A pretrained linear probe matches the oracle GNN on few-shot: then H1 is an
  artefact of training from scratch, and the study should pivot to the shift
  and composition axes only.
- The noisy-oracle calibration fails (noise at matched F1 does not reproduce
  the classical extractor's downstream accuracy): then the F1 axis is not a
  sufficient statistic and the map has to be drawn per error *type*.
- No simple form fits the crossover, or the Family A -> Family B
  cross-prediction misses: the predictive claim is gone and only the
  tradeoff study remains.

## 7. Venue strategy and the graceful landing

**Target: Nature Machine Intelligence.** What that venue needs, beyond
rigour, is a result that changes what practitioners do and that is shown
to hold beyond the setting it was derived in. Three things carry it:

1. a rule, not a table, for when structure pays;
2. registered out-of-domain forecasts on at least two real domains;
3. annotation-cost accounting that makes the rule a budgeting tool.

The nearest published comparator is Madan et al. 2022 in the same venue,
on CNN generalisation to out-of-distribution category-viewpoint
combinations. That is a controlled study with a general claim and no
real-world deployment, which is the right calibration for what is
achievable here.

**The landing, if a gate fails.** Each fallback is a real paper, so no
outcome wastes the work:

| If | Then |
|---|---|
| The rule fits and the forecasts hit | NMI submission as planned |
| The rule fits, forecasts miss on some domains | NMI is a stretch; the honest version (what the rule fails to capture) is a strong TMLR paper |
| No simple rule, but the surface is well measured | TMLR tradeoff study, the original plan |
| Family B is the most reusable artefact | NeurIPS Datasets and Benchmarks, in addition |
| The whole advantage is an augmentation artefact | A short, useful negative-result paper, and worth writing |

**What the target does not change.** Nothing about the protocol, the
baselines, or the honesty of the reporting is adjusted to suit a venue.
The framing follows the result. Aiming higher changes what is *built*
(the rule, the real domains, the registrations), never what is *claimed*
about what was measured.
