# Shape-primitive object recognition

Does recognising objects from **generic geometric primitives and their
spatial relations** generalise better than an end-to-end CNN?

```
pixels ──► primitive extractor ──► graph of shapes ──► relational classifier ──► class
           (oracle / classical /     (type, size, pose,   (GNN / set transformer /
            learned / SAM-fit)        pairwise relations)   bag ablation)
```

The plan is in [`PLAN.md`](PLAN.md). The project runs in three phases,
each harder than the one before:

| Phase | Data | Question | Status |
|---|---|---|---|
| 1 | Clean, flat 2D drawings (10 classes) | Does the shape-graph representation work at all? | **done**: [results](results/phase1/SUMMARY.md) |
| 2 | The same drawings under rotation, shear and perspective | Is a bike at an angle still two ellipses and a triangle? | **done**: [results](results/phase2/SUMMARY.md) |
| 3 | Textures, QuickDraw sketches, real photos | Does it survive realistic appearance? | experiments running |

The hypotheses are sample efficiency (H1), relations matter (H2),
viewpoint robustness (H3) and appearance robustness (H4). The Phase 1 CNN
baseline is intentionally strong: augmented, with an ImageNet-pretrained
fine-tune and a linear probe. A win measured against a weak baseline
would not mean anything.

## Quick start

```bash
# Python 3.10+; the CPU build of PyTorch is enough for everything here
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

python -m pytest -q                                  # the test suite (also runs in CI)
python scripts/generate_phase1_data.py               # render the dataset + a sample grid
python scripts/run_experiment.py --config configs/phase1/baseline_experiment.yaml
python scripts/plot_results.py --summary results/phase1/baseline/summary.json
```

`run_experiment.py` takes these flags:

| Flag | What it does |
|---|---|
| `--dry-run` | List the runs without training |
| `--resume` | Skip runs that already finished |
| `--models ...` | Run only the named models |
| `--seeds ...` | Override the seeds |
| `--device cpu\|cuda\|auto` | Pick the device |

## Repository layout

```
src/shapeprim/
  data/        primitives.py   primitive dataclass + rendering
               objects.py      the 10 class templates + held-out composition variants
               synth_dataset.py deterministic generator (train/val/test from disjoint streams)
               augment.py      CNN augmentation presets
               torch_datasets.py image / graph datasets, extraction cache
  extract/     oracle.py       ground-truth primitives (upper bound for stage 2)
               classical.py    OpenCV contours -> polygon / ellipse fits
               eval_match.py   IoU matching -> extractor precision / recall / F1
  graph/       build.py        primitives -> bbox-normalised, fully connected graph
  models/      cnn.py          ResNet-18 (scratch, ImageNet fine-tune, linear probe)
               gnn.py          GINE-style message passing
               settransformer.py  attention over primitives (second architecture)
               bag.py          ablation: primitive counts and sizes, no positions
  train.py, evaluate.py, experiment.py, conditions.py
configs/       model/*.yaml, phase1/*.yaml  every hyperparameter lives here
scripts/       run_experiment.py, plot_results.py, generate_phase1_data.py
results/       one directory per experiment: summary.json, results.md, figures
docs/          protocol, research plan, literature review
```

## Experimental protocol

The full protocol is in
[`docs/experiment_protocol.md`](docs/experiment_protocol.md). In short:

- **Splits:** train, validation and test come from disjoint random streams.
- **Selection:** models are selected on validation, and the test set is touched exactly once per run.
- **Seeds:** at least 3, reported as mean ± Student-t 95% CI.
- **Equal treatment:** CNN and GNN get the same schedule, stopping rule and tuning budget.
- **Augmentation:** the pixel baseline is always reported with and without it.
- **Extractor quality:** precision/recall/F1 is reported next to every graph-model accuracy.
- **Records:** every run writes its config, git commit, seed and metrics to
  `results/<phase>/<experiment>/<run_id>/`.
- **Training floor and BN:** at least 500 optimizer steps per run, precise
  BatchNorm before every validation pass, and validation at most ~60 times
  per run.

The whole remaining plan runs with `scripts/run_all.sh`: one job at a
time, resumable, with results committed and pushed every 15 minutes.

## Phase 1 at a glance

- **Classes:** bicycle, car, truck, cat face, house, tree, arrow sign,
  snowman, person, fish, each built from circles, triangles, rectangles and
  lines.
- **Relation twins:** `tree` and `arrow_sign` use identical parts in a
  flipped arrangement, so only the arrangement separates them. From dataset
  version 2 on, `arrow_sign` is the exact vertical flip of `tree`.
- **Experiments:** few-shot learning curves (5–1000 examples/class), a
  position/scale shift, novel compositions (held-out part variants) and the
  bag-of-primitives relation ablation.

**Phase 1 in one paragraph** ([full results](results/phase1/SUMMARY.md)).
With oracle primitives, the GNN matches augmented and ImageNet-pretrained
CNNs on few-shot learning: all are ≥ 0.996 at 5 examples per class, and
only the un-augmented CNN is left behind at 0.55. It is exactly invariant
to position and scale shift (1.000 against 0.936 for the best augmented
CNN), and the bag ablation shows relations are needed for the tree/arrow
twins (bag 0.50, GNN 1.00). It loses clearly on novel part compositions
(0.49 against 0.90–0.98), where it falls back on a part-type shortcut.
With the classical extractor (F1 ~0.8; it misses thin lines), every
primitive-model advantage shrinks or reverses.

The earlier baseline table in `results/phase1/baseline/` was measured on
dataset version 1, before the twin fix, and is superseded.

**Phase 2 in one paragraph** ([full results](results/phase2/SUMMARY.md)).
Trained on viewing angles 0–30°, the oracle GNN with an affine-normalised
graph stays at 0.996 out to 70°, while every pixel model, including one
trained with perspective augmentation, falls to 0.38–0.60. It is also
more sample-efficient under viewpoint change (1.000 against 0.960 at 5
examples per class). The advantage is lost in extraction, because both
the classical extractor and the learned detector break down at steep
angles: the learned detector's F1 falls from 1.0 to 0.16.

![Phase 1 sample grid](results/phase1/sample_grid.png)

## Further reading

- [`PLAN.md`](PLAN.md): the phased project plan.
- [`docs/research_plan.md`](docs/research_plan.md): from "it works" to a
  publishable, falsifiable claim.
- [`docs/literature_review.md`](docs/literature_review.md): positioning
  against part-based, neuro-symbolic and sketch-graph work.
