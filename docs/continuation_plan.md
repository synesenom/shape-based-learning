# Continuation plan

Where the project stands after the first execution pass of
[`PLAN.md`](../PLAN.md), and exactly what remains. Written so work can resume
in a fresh clone (or a new repository) with no other context.

## 1. State at hand-off

| PLAN step | Status | Evidence |
|---|---|---|
| Setup + shared components | done | PRs #4, #7 |
| Phase 1: clean drawings | done | [`results/phase1/SUMMARY.md`](../results/phase1/SUMMARY.md) |
| Phase 2: viewpoint | done | [`results/phase2/SUMMARY.md`](../results/phase2/SUMMARY.md) |
| Phase 3a: appearance | **half done** | flat→textured complete (24 runs); textured→flat 1/18 runs; no write-up yet |
| Phase 3b: QuickDraw | code done, smoke-tested end to end | not run |
| Phase 3c: real photos | code done, smoke-tested end to end | not run; 144 of ~1,300 images segmented (cached in `results/phase3/samfit_cache/`) |
| Final cross-phase figure | not started | needs a script (see §3, step 7) |

**Findings so far, in one line each:**

- **Clean drawings:** with oracle primitives the graph model ties augmented
  and pretrained CNNs on few-shot, and is invariant to position/scale. It
  *loses* on novel part compositions, because it learns a part-type shortcut.
  Relations are needed for the tree/arrow twins.
- **Viewpoint:** with oracle primitives and the affine-whitened graph it
  extrapolates from 0–30° to 70° with no loss (0.996, where pixel models
  score 0.38–0.60). The advantage is lost in extraction.
- **Appearance (flat→textured):** the GNN on a class-agnostic,
  appearance-trained detector scores 0.83 on all appearance changes
  combined, against at most 0.28 for pixel models. The reverse direction is
  pending.

## 2. How to resume

```bash
git clone <repo> && cd <repo>
scripts/setup_env.sh --data        # CPU PyTorch, package + extras, tests, Phase 3b/3c data
git checkout -b <working-branch>   # run_all.sh pushes checkpoints to the checked-out branch
scripts/run_all.sh                 # resumes; add --no-push to keep results local
```

`run_all.sh` is idempotent:
- it skips every run whose `results/.../metrics.json` is committed;
- it does not retrain a detector whose weights are committed;
- it re-fetches missing data.

It runs **one training job at a time**. Two PyTorch jobs on 4 cores made
each ~10× slower. It commits and pushes `results/` every 15 minutes and
after every step.

**Operational lessons** (each one cost hours in the first pass):

- After a machine restart, check `ps` for a still-running `run_all.sh`
  before launching another. Two queues on one machine ran 8 hours at a
  tenth of the speed.
- Never `pkill -f <pattern>` from a shell whose own command line contains
  the pattern: it kills the shell. Kill by PID.
- Never let `data/` become a tracked path (a committed symlink once
  replaced the real directory). `/data` is ignored as file *or* directory.
- A restart can stop the queue silently. Check progress (`tail
  logs/run_all.log`) at least hourly.
- On a private repository, the GitHub Actions workflow uses your Actions
  minutes (~1 minute per push).

## 3. Remaining work, in order

CPU estimates are for 4 cores, no GPU. Every step ends with a
`results/phase3/...` write-up, a PR, and a README status update.

### Step 5: finish Phase 3a (~1.5 h CPU)

1. `appearance_reverse` resumes automatically (17 runs left). It trains the
   pixel models on textured, cluttered, "seen"-palette data and tests on
   flat drawings and the unseen palette. This is the fairness check for the
   3a headline: the pixel models get appearance-varied training data too.
2. Write `results/phase3/SUMMARY_3a.md`: the appearance-shift table (plot
   and table already in `results/phase3/appearance_shift/app_conditions.*`),
   the reverse table, the detector-F1 table, and the H4 verdict. Keep two
   caveats:
   - the detector saw textured/cluttered images during its label-free
     training;
   - the first detector version (clutter on every image) failed on plain
     backgrounds and is kept in `superseded_clutter_always/`.
3. Done when: both directions reported, with extractor F1 per condition
   next to accuracy.

### Step 6: Phase 3b, QuickDraw (~6 h CPU)

1. `scripts/eval_strokefit.py` measures stroke-fitter F1 on synthetic
   sketches with known ground truth (0.91 frontal in development). QuickDraw
   itself has no primitive labels, so this proxy is the only extractor
   number; state it as an upper bound.
2. `quickdraw_fewshot`: 10 QuickDraw classes, 5–500 examples/class; CNNs vs
   stroke-fit graph models.
3. `quickdraw_cross` and `quickdraw_cross_reverse`: synthetic (outline
   rendering) ↔ QuickDraw, on the 8 shared classes.
4. Report failure examples: single-stroke houses lose the roof, car bodies
   come out as triangles, cats are often drawn as whole bodies, not faces.
   The per-sample `predictions` in each `metrics.json` identify them.
5. Consider trimming the 500/class point to 250 if CPU is tight (saves
   ~2 h). The few-shot claims live at 5–25.
6. Done when: few-shot table, cross-domain table (both directions), and the
   failure examples.

### Step 7: Phase 3c, real photos, and the final figure (~5 h CPU)

1. `scripts/precompute_sam.py --threads 4`: SlimSAM-77 segmentation of the
   remaining ~1,150 images (~4 s each on 4 cores). Results cache to the
   committed `results/phase3/samfit_cache/`.
2. `real_images`:
   - classes: bicycle, car, truck, cat;
   - training data: COCO crops (train2017), with 10 and 100 per class;
   - tests: COCO val2017 crops, plus Geirhos's model-vs-human sketch,
     stylized, edge, silhouette and cue-conflict images of the same classes.
3. Caveats to keep:
   - the ImageNet-pretrained rows have seen these categories;
   - cue-conflict is scored on shape and restricted to texture classes
     among the four;
   - no primitive ground truth for photos, so no extractor F1.
4. **Final figure** (PLAN.md §7): the same object across all phases, with
   CNN vs primitive-model predictions. It needs a new script,
   `scripts/make_cross_phase_figure.py`:
   - Phases 1, 2 and 3a share geometry: the same `test_seed`, split and
     index give the same object instance. The view and appearance random
     draws happen after the geometry. So pick a test index and render it
     frontal, at 50–70°, and fully styled.
   - Read the CNN and GNN predictions for that index from the runs'
     `metrics.json` (`predictions` key). For Phase 1, use
     `novel_composition`'s `test_indist` predictions: every `cnn_aug` and
     `gnn_oracle` seed there has them, so no re-run is needed.
   - Add one QuickDraw drawing and one COCO crop of the same class, with
     their predictions.
5. Done when: result tables for 3a, 3b and 3c, and the cross-phase figure
   (PLAN.md §7 "done when").

## 4. Known caveats and open decisions

These are recorded, not fixed. Each is a candidate follow-up.

1. **Learned detector range in `full_range`.** The Phase 2 detector was
   trained on 0–30° only, so in the full-range experiment it runs outside
   its range at steep angles (plateau at 0.85). Optional: train a 0–70°
   detector with `configs/phase2/learned_extractor.yaml` and
   `train_overrides.view_angle_range: [0, 70]`, then re-run the
   `gnn_learned*` rows of `full_range` (~40 min).
2. **Novel-composition shortcut.** The GNN classifies by which part types
   are present (a snowman with arms is called a bicycle). Candidate fixes to
   test:
   - random part dropout/insertion during training;
   - training on some variants.

   The set transformer (0.86) suggests the readout and inductive bias
   matter.
3. **Few-shot validation sets are larger than the training sets** (40 per
   class for validation at 5 per class for training) for every model.
   Absolute few-shot numbers are optimistic; the gaps are comparable.
4. **Parameter counts are not matched** (ResNet-18 at 11.2 M, GNN at 57 k).
   A small matched CNN is WP2 in the research plan; not done.
5. **SlimSAM-77, not full SAM ViT-B**, for CPU cost (~130 s per image with
   ViT-B). Re-run `precompute_sam.py` with a different model id on a GPU
   machine if that matters.
6. **Research-plan work packages not in PLAN.md are not started**
   ([`research_plan.md`](research_plan.md)):
   - WP3.1–3.2: the noisy-oracle F1 dial and break-even surface;
   - WP5: the crossover rule;
   - WP6: registered out-of-domain forecasts;
   - WP1 Family B: procedural classes with relation twins.

   Phases 2–3 already show that extraction quality decides the outcome,
   which is exactly what WP3's noisy-oracle sweep would quantify. It is the
   natural next experiment after this plan.

## 5. Where things live

- Configs: `configs/phase{1,2,3}/*.yaml`, `configs/model/*.yaml`.
  Every experiment is a config.
- Results:
  - `results/<phase>/<experiment>/`: `summary.json`, `results.md` and
    figures, plus one directory per run with `metrics.json` (including
    per-sample predictions) and `config.json` (including the git commit);
  - `results/<phase>/SUMMARY.md`: the phase write-ups;
  - `results/<phase>/superseded_*/`: runs recorded under an earlier
    protocol, kept for audit.
- Protocol: [`experiment_protocol.md`](experiment_protocol.md), rules 1–7
  plus 3b–3d added during Phase 1.
- Large downloads are not committed:
  - `data/` (QuickDraw, COCO, model-vs-human, Hugging Face cache);
  - regenerate with `scripts/setup_env.sh --data`.
