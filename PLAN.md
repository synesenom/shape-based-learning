# PLAN.md — Shape-Primitive Object Recognition

> **Status:**
> - Phases 1 and 2 are done ([results](results/phase1/SUMMARY.md), [results](results/phase2/SUMMARY.md)).
> - Phase 3a is half done; 3b and 3c are implemented but not run.
> - The remaining work, how to resume, and open caveats are in [`docs/continuation_plan.md`](docs/continuation_plan.md).

## 1. Goal

Test whether classifying objects from **generic geometric primitives and their spatial relations** generalizes better than a standard end-to-end CNN.

The pipeline has two stages:

```
pixels ──► [Stage 1: primitive extractor] ──► set/graph of shapes ──► [Stage 2: relational classifier] ──► class
            (circle, ellipse, triangle,          (type, position,         (GNN / set transformer)
             quadrilateral, line + pose)          size, rotation, relations)
```

The project grows in **three phases**, each harder than the last:

| Phase | Data | Main question |
|---|---|---|
| 1 | Clean, flat 2D drawings | Does the shape-graph representation work at all? |
| 2 | Same drawings under rotation, skew, and perspective | Does the model learn that two ellipses + a triangle is still a bike? |
| 3 | Textures, then real images | Does it survive realistic appearance? |

### Hypotheses

- **H1 (sample efficiency):** The primitive model needs fewer training examples per class.
- **H2 (relations matter):** Removing or shuffling spatial relations hurts the primitive model a lot.
- **H3 (viewpoint):** The primitive model generalizes to unseen viewing angles better than a CNN.
- **H4 (appearance):** The primitive model is more robust to texture, color, and style changes.

### Novelty angle

Most recent part-based work (e.g. Sitawarin et al. 2022, CompositionalNets) uses **semantic parts** (wheel, head). This project uses **generic primitives shared across all classes**, closer to Biederman's geons, with modern tools.

---

## 2. Tech stack

- Python 3.11, PyTorch, torchvision
- PyTorch Geometric (GNN classifier)
- OpenCV + scikit-image (contour extraction, primitive fitting)
- NumPy, Pillow / pycairo (rendering)
- YAML configs, TensorBoard or W&B, pytest
- Phase 3: `segment-anything` (SAM), QuickDraw, real-image datasets

---

## 3. Repository structure

```
shape-primitives/
├── PLAN.md
├── configs/
│   ├── phase1/  phase2/  phase3/
│   └── model/            # cnn.yaml, gnn.yaml, settransformer.yaml, bag.yaml
├── src/shapeprim/
│   ├── data/
│   │   ├── primitives.py      # primitive dataclasses + rendering
│   │   ├── objects.py         # class templates built from primitives
│   │   ├── synth_dataset.py   # images + ground-truth primitives
│   │   ├── transforms.py      # phase 2: rotation, skew, perspective
│   │   ├── textures.py        # phase 3a: textures, backgrounds
│   │   └── real.py            # phase 3b/3c: QuickDraw + real images
│   ├── extract/
│   │   ├── oracle.py          # ground-truth primitives
│   │   ├── classical.py       # OpenCV contour + shape fitting
│   │   ├── learned.py         # small primitive detector
│   │   └── sam_fit.py         # phase 3c: SAM masks → primitives
│   ├── graph/build.py         # primitives → graph
│   ├── models/                # cnn.py, gnn.py, settransformer.py, bag.py
│   ├── train.py
│   └── evaluate.py
├── scripts/                   # generate data, run experiments, make report
├── tests/
└── results/phase1/  phase2/  phase3/
```

---

## 4. Shared components (built once, reused in every phase)

### Models

- **CNN baseline:** ResNet-18, from scratch and ImageNet-pretrained, with standard augmentation.
- **Primitive model:** extractor → graph → relational classifier.
  - `gnn.py`: message-passing GNN with edge features (e.g. 3× GINE) + pooling + MLP.
  - `settransformer.py`: transformer over nodes, as a second architecture so conclusions don't depend on one model.
  - `bag.py` (ablation): counts and sizes of each primitive type, **no positions or relations**.

### Graph representation (`graph/build.py`)

- **Nodes:** one-hot type, size, aspect ratio, rotation (sin/cos), position normalized to the object's bounding box.
- **Edges:** fully connected; relative position (dx, dy), distance, size ratio, relative angle, above/below/left/right/inside flags.
- All features are relative to the whole object, so the graph is translation- and scale-invariant.

### Extractors (same interface: `extract(image) -> list[Primitive]`)

- **Oracle:** returns ground truth. Upper bound for stage 2; separates extraction errors from classification errors.
- **Classical:** OpenCV contours → polygon approximation + ellipse fitting → classify by vertex count and circularity.
- **Learned:** small detector trained on synthetic images to output primitive type + pose.
- **SAM-fit** (phase 3 only): SAM masks → best-fitting primitive per mask.

### Evaluation rules

- 3 seeds per run, report mean ± std.
- Always report: in-distribution accuracy, accuracy under each test condition, few-shot learning curves (5, 10, 25, 50, 100, 1000 examples/class), and primitive precision/recall for non-oracle extractors.
- Save config, seed, and metrics to `results/<phase>/<experiment>/<run_id>/`.

---

## 5. Phase 1 — Clean 2D drawings

**Goal:** show the representation works when shapes are obvious.

### Data

Primitives: `circle`, `triangle`, `rectangle`, `line`. Flat colors, white background, 128×128, object centered-ish.

About 10 classes, defined as primitive templates with small variation:

| Class | Rough composition |
|---|---|
| bicycle | 2 circles + triangle frame + lines |
| car | large rectangle + small rectangles (windows) + 2 circles below |
| truck | 2 rectangles + 3 circles |
| cat face | circle + 2 triangles on top |
| house | rectangle + triangle on top + small rectangles |
| tree | rectangle trunk + triangle crown |
| arrow sign | triangle **below** rectangle (same shapes as tree, different arrangement) |
| snowman | 3 stacked circles |
| person | circle head + rectangle body + lines |
| fish | circle/ellipse body + triangle tail |

The **tree vs arrow sign** and **car vs truck** pairs are there on purpose: they force the model to use arrangement, not just which shapes appear.

Variation: position jitter, scale, small part-size jitter, optional distractor shapes. 1,000 training images per class, 200 per class per test set.

### Important: what "winning" means here

On clean in-distribution data, the CNN will also reach close to 100%. So in Phase 1 the primitive model should win on:

- **Few-shot:** 5–25 examples per class.
- **Position/scale shift:** train with objects small and centered, test with objects large or off-center.
- **Novel compositions:** hold out variants during training (e.g. car with 3 windows) and test on them.
- **Relation test:** tree vs arrow sign accuracy; the bag ablation should fail here.

### Extractors

Oracle and classical. The classical extractor should be near-perfect on clean drawings; if it isn't, fix it before moving on.

### Done when

- Sample grids look right and tests verify JSON primitives match the rendered image.
- Primitive model (oracle and classical) beats the CNN on few-shot and position/scale shift.
- Bag ablation fails on tree vs arrow sign while the GNN succeeds.

---

## 6. Phase 2 — Viewpoint transformations

**Goal:** emulate photos taken from different angles. The model should learn that a bike seen at an angle (two **ellipses** + a skewed triangle) is still a bike.

### Data

Apply transformations to Phase 1 drawings (`transforms.py`), rendering the ground-truth primitives through the same transform:

- **Rotation** (in-plane)
- **Skew / shear** (affine)
- **Perspective** (homography), controlled by a "viewing angle" parameter from 0° (front) to ~70° (steep)
- Optional: non-uniform scaling (squash/stretch)

Under these transforms, circles become ellipses and rectangles become general quadrilaterals. So the primitive vocabulary grows to: `ellipse` (circle is a special case), `triangle`, `quadrilateral`, `line`. Each primitive now carries aspect ratio and orientation.

Optional stretch: build a few classes from simple 3D primitives (cylinders, boxes, cones) and render them with a lightweight renderer for true viewpoint changes, including self-occlusion.

### Key design decision

There are two ways the primitive model can handle viewpoint. Test both:

1. **Learned invariance:** raw graph features; the classifier learns from examples at many angles.
2. **Built-in invariance:** graph features made more affine-invariant (e.g. positions normalized by the object's principal axes, ellipses described relative to each other instead of absolutely).

### Experiments

- **Angle extrapolation (main test):** train on viewing angles 0–30°, test on 30–50° and 50–70°. Plot accuracy vs angle for CNN and primitive models.
- **Full-range training:** train on all angles; compare sample efficiency.
- **Few-shot at angles:** 5–25 examples per class across angles.
- **Relation test under transforms:** tree vs arrow sign under rotation (note: a 180° rotation makes them identical, so cap rotation or treat this as a known limit).

### Extractors

Oracle, classical (now fitting ellipses and quadrilaterals), and learned. The learned detector should be trained on a range of angles and tested on the extrapolation angles too.

### Done when

- Accuracy-vs-angle plots exist for all models.
- It's clear whether the primitive model's advantage comes from the representation (oracle) or is lost in extraction (classical/learned).

---

## 7. Phase 3 — Texture and real images

**Goal:** move from clean shapes to realistic appearance. Done in three steps so each change can be measured separately.

### 3a. Textured synthetic

Take Phase 2 data and add:

- Procedural textures on each primitive (noise, stripes, dots, photo patches)
- Unseen color palettes
- Cluttered backgrounds
- Occlusion (random shapes covering 10–40% of the object)
- Noise and blur

Train on flat, test on textured (and vice versa). This is where H4 is tested cleanly, since ground truth still exists.

### 3b. Sketches (QuickDraw)

- ~10 QuickDraw classes that decompose well into primitives (bicycle, car, truck, house, cat, tree, fish, snowman, sun, face).
- Fit primitives to strokes or to rasterized images.
- Few-shot comparison, plus cross-domain: train on synthetic, test on QuickDraw (and vice versa).

Real people drew these, so this checks that results don't depend on hand-designed classes.

### 3c. Real images

- A small set of real photos for matching classes (e.g. PartImageNet or COCO crops of bicycles, cars, trucks, cats, houses).
- Extract primitives with SAM + shape fitting (`sam_fit.py`): ellipse fit, minimum-area rectangle, polygon approximation; keep the best fit by IoU.
- Evaluate on photos, ImageNet-Sketch, and Stylized-ImageNet / cue-conflict images for matching classes (Geirhos's `model-vs-human` toolkit).

### Expected result

The primitive model probably loses on raw accuracy with real photos but wins on sketches, stylized images, and few-shot. Report failure examples honestly, especially where extraction breaks.

### Done when

- Results tables for 3a, 3b, and 3c.
- A figure showing the same object across all three phases with CNN vs primitive predictions.

---

## 8. Risks

- **Phase 1 too easy for both models:** use few-shot, shift, and composition tests to separate them; add distractors if needed.
- **Stage 1 becomes the bottleneck in Phases 2–3:** expected. Oracle results still answer "is the representation good?"
- **Unfair comparison:** document augmentation and what each extractor saw during training.
- **Rotation symmetry breaks some classes:** cap rotation ranges or treat as a documented limit.

---

## 9. Related work

- Biederman (1987) — Recognition-by-Components
- Marr & Nishihara (1978) — 3D shape representations
- Felzenszwalb et al. (2010) — Deformable Part Models
- Sabour, Frosst & Hinton (2017) — Capsule Networks; Hinton (2021) — GLOM
- Lake et al. (2015) — Bayesian Program Learning
- George et al. (2017) — Recursive Cortical Network
- Geirhos et al. (2019) — texture bias in ImageNet CNNs
- Baker et al. (2018); Malhotra et al. (2022/23) — CNNs ignore relational shape
- Kortylewski et al. (2020) — CompositionalNets
- Sitawarin et al. (2022) — Part-based models improve adversarial robustness

---

## 10. Notes for Claude Code

- Start with setup + shared components, then Phase 1. Do not start a phase until the previous "done when" passes.
- Stop and show results at each "done when" check.
- Write tests for data generation, transforms, and graph building before training.
- Keep settings in `configs/`; no hard-coded hyperparameters.
- Do quick small runs first (fewer classes, 64×64) to check the pipeline end to end, then scale up.
