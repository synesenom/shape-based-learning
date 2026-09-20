# Literature review: has "primitive graph vs CNN" been done?

Scope: four search sweeps (classical/cognitive part-based recognition;
modern relational and neuro-symbolic vision; sketches, diagrams and
primitive inference; compositional generalisation, invariance and
methodology), run by search agents and then curated by hand. Entries
marked come from memory rather than from a search hit and must be
checked before they go into a paper. Several agent attributions were wrong
and have been corrected here; anything the agents returned that could not
be pinned down has been dropped.

## 1. Verdict

**The pipeline is not new. The proposed study is.**

- "Extract parts or objects, then reason over their relations, and compare
  with a pixel model" has been done in at least five separate literatures:
  pictorial structures and attributed relational graphs (1973-2010),
  relation networks and neuro-symbolic VQA on CLEVR (2017-2019),
  object-centric slot models with relational heads on abstract-reasoning
  tasks (2021-2023), sketch-graph networks (2019-2022), and graphics-program
  inference from drawings (2018-2019).
- The closest single line of work is Webb / Mondal / Cohen's object-centric
  relational abstraction on synthetic shape tasks, which reports exactly the
  kind of sample-efficiency and OOD advantage PLAN.md's H1 and H3 predict,
  using slot attention instead of geometric primitives. A reviewer who knows
  that work will not accept H1-H3 as findings in themselves.
- What has **not** been done, as far as these sweeps and my own knowledge go:
  1. Treating extraction quality as a *controlled experimental variable* and
     mapping the break-even between the structured pipeline and the best
     pixel baseline over (examples per class) x (extractor F1) x (shift type).
     The neuro-symbolic literature talks about perception-error propagation
     but does not measure a tradeoff surface.
  2. Doing that with *generic geometric primitives shared across classes*
     (Biederman's programme) rather than semantic parts, learned slots, or
     class-specific strokes; and with a *class-agnostic* extractor whose own
     data cost is accounted for.
  3. Reporting the annotation budget honestly: class labels for the pixel
     model vs class labels plus primitive labels for the pipeline.
  4. The "relation twin" design (identical part multiset, different
     arrangement) at benchmark scale, with a bag-of-parts ablation and
     part-shuffling diagnostics for the pixel models, to give a causal rather
     than correlational answer to "does the model use arrangement".

The PLAN.md novelty sentence ("generic primitives, not semantic parts") is
true but narrow. The defensible contribution is the controlled tradeoff
study, the benchmark that supports it, and the practical decision rule it
yields.

## 2. Closest prior work

| Work | What it did | What it leaves open for this project |
|---|---|---|
| Fischler & Elschlager 1973; Felzenszwalb & Huttenlocher 2005 (pictorial structures); Felzenszwalb et al. 2010 (DPM) | Objects as parts plus spring-like spatial relations; learned appearance per part; strong pre-deep detection | Parts are class-specific templates, not a shared primitive vocabulary; no data-efficiency curves vs end-to-end nets; no extractor-noise analysis |
| Bunke & Allermann 1983; Sanfeliu & Fu 1983 (attributed relational graphs) | Structural pattern recognition by inexact graph matching over primitive-and-relation graphs; the 1980s version of this pipeline | No learning of the classifier, no comparison with pixel models, no controlled synthetic benchmark |
| Santoro et al. 2017, Relation Networks (NIPS); Sort-of-CLEVR | Pairwise relational module over CNN feature cells beats a plain CNN on relational questions | Objects are feature-map cells, not primitives; VQA not classification; no extraction stage to be noisy |
| Yi et al. 2018 NS-VQA (NeurIPS); Mao et al. 2019 NS-CL (ICLR) | Scene parser -> symbolic scene graph -> program executor; near-perfect CLEVR accuracy and strong data efficiency | Reasoning is a program over attributes, not recognition from arrangement; the effect of parser errors is reported anecdotally, not mapped |
| Amizadeh et al. 2020, "Neuro-symbolic visual reasoning: disentangling visual from reasoning" (ICML) | Explicit separation of perception and reasoning; evaluates reasoning with oracle vs learned perception | VQA again; one extractor-quality point, no curve |
| Webb, Sinha & Cohen 2021 (ESBN, ICLR); Mondal, Webb & Cohen 2023 "Learning to reason over visual objects" (ICLR); Webb et al. 2023 OCRA (NeurIPS) | Slot-attention object extraction plus a relational module on SVRT, ART, CLEVR-ART; reports sample-efficiency and OOD gains over CNN/transformer pixel models | Objects are slots, not geometric primitives; tasks are abstract rules (same/different, RMTS) rather than object categories defined by arrangement; extractor quality is not a controlled variable |
| Dittadi et al. 2022 "Generalization and robustness implications in object-centric learning" (ICML) | Systematic study of whether object-centric representations improve downstream robustness under shifts; mixed results | Unsupervised slot models only; no explicit relations; no learning-curve x extractor-quality interaction |
| Kim, Ricci & Serre 2018 "Not-so-CLEVR" (Interface Focus); Fleuret et al. 2011 SVRT (PNAS) | Feedforward CNNs strain on same-different and relational tasks over random shapes | Diagnoses CNN weakness; does not build or evaluate the structured alternative |
| Yang et al. 2021 SketchGNN (TOG); Xu et al. 2021 Multi-Graph Transformer (TNNLS); Sketchformer (Ribeiro et al. 2020 CVPR); Sketch-R2CNN | Stroke-graph / stroke-sequence networks for sketch recognition and segmentation, often beating raster CNNs | Strokes are given by the input device, not extracted; no data-efficiency curves; no OOD/composition splits |
| Ellis et al. 2018 "Learning to infer graphics programs from hand-drawn images" (NeurIPS); Sharma et al. 2018 CSGNet (CVPR); Tian et al. 2019 shape programs (ICLR) | Infer primitives and programs from drawings/shapes | Inference, not recognition; no comparison with a pixel classifier on generalisation |
| Lake et al. 2015 BPL (Science), Omniglot | Stroke-program representation gives one-shot character learning that beat deep nets of the time | Characters, not objects; the comparison has since been contested by strong meta-learning baselines |
| Kortylewski et al. 2020/21 CompositionalNets (CVPR/IJCV); Sitawarin et al. 2023 part-based robustness (ICLR) | Semantic-part compositional models improve occlusion and adversarial robustness on real images | Semantic parts; PLAN.md already positions against these |
| Sabour, Frosst & Hinton 2017 capsules (NIPS); Hinton 2021 GLOM | Part-whole hierarchies with pose; viewpoint-generalisation claims | Claims were not borne out at scale; useful as a baseline or as motivation, not as competition |

## 3. Supporting literature by theme

### 3.1 Cognitive-science motivation (why generic primitives)
- Biederman 1987, Recognition-by-Components (Psych. Review).
- Marr & Nishihara 1978, generalized cylinders (Proc. R. Soc. B).
- Hummel & Biederman 1992, JIM network: dynamic binding of parts to relations (Psych. Review).
- Biederman & Gerhardstein 1993, viewpoint invariance conditions (JEP:HPP).
- Tanaka & Farah 1993, parts and wholes (QJEP).
- Smith 2003, "Learning to recognize objects" — children recognise shape caricatures built from 2-4 geometric volumes in the right arrangement (Psych. Science).
- Lake, Ullman, Tenenbaum & Gershman 2017, compositionality as a route to data efficiency (BBS).
- Battaglia et al. 2018, relational inductive biases and graph networks (arXiv).

### 3.2 Evidence that pixel models under-use global arrangement (motivates H2 diagnostics)
- Geirhos et al. 2019, texture bias (ICLR).
- Brendel & Bethge 2019, BagNet: bag-of-local-features approximates ImageNet CNNs (ICLR).
- Baker, Lu, Erlikhman & Kellman 2018, CNNs do not classify by global shape (PLOS Comput. Biol.); Baker & Elder 2022, configural shape (iScience).
- Malhotra, Dujmović & Bowers 2022 "Feature blindness" (PLOS Comput. Biol.); Malhotra et al. 2023 on relational shape (JEP:General). Already in PLAN.md.
- Madan et al. 2022, CNN generalisation to OOD category-viewpoint combinations (Nat. Mach. Intell.).
- Kim, Ricci & Serre 2018; Fleuret et al. 2011 (above).
- Vaishnav et al. 2022, computational demands of visual reasoning (Neural Computation).

### 3.3 Invariance is mostly learned from augmentation (what the CNN baseline must get)
- Azulay & Weiss 2019, poor generalisation to small transformations (JMLR).
- Kauderer-Abrams 2017, quantifying translation invariance; augmentation, not architecture, drives it (arXiv).
- Biscione & Bowers 2021, CNNs are not translation invariant but can learn to be (JMLR).
- Cohen & Welling 2016, group-equivariant CNNs (ICML); Sosnovik et al. 2020, scale-equivariant steerable networks (ICLR): the architectural alternative to explicit primitives for H3.
- Radford et al. 2021, CLIP (ICML): the pretrained few-shot baseline a practitioner would actually use.

### 3.4 Compositional generalisation benchmarks and methodology
- Johnson et al. 2017, CLEVR and the CoGenT split (CVPR).
- Wiedemer et al. 2023, compositional generalisation from first principles (NeurIPS): conditions under which held-out compositions are learnable; useful for designing the Family B grammar.
- Andreas et al. 2016, Neural Module Networks and the SHAPES dataset (CVPR).
- Nie et al. 2020, Bongard-LOGO (NeurIPS): programmatically generated shape concepts; few-shot concept learning.
- Stammer, Schramowski & Kersting 2021, CLEVR-Hans (CVPR): neuro-symbolic confounder benchmark.
- Barrett et al. 2018 PGM (ICML); Zhang et al. 2019 RAVEN (CVPR): relational shape reasoning benchmarks.
- Matthey et al. 2017, dSprites (DeepMind).
- Bouthillier et al. 2021, "Accounting for variance in ML benchmarks" (MLSys): seeds and CIs.

### 3.5 Object-centric learning as an alternative extractor
- Locatello et al. 2020, Slot Attention (NeurIPS); Burgess et al. 2019 MONet; Greff et al. 2019 IODINE (ICML); Engelcke et al. 2020 GENESIS (ICLR).
- Dittadi et al. 2022 (above).
- Kipf et al. 2018, Neural Relational Inference (ICML): unsupervised edge inference; relevant if edges are to be learned rather than hand-featured.

### 3.6 Sketches, diagrams and vector primitives (real-data anchor candidates)
- Eitz, Hays & Alexa 2012, TU-Berlin sketch dataset (SIGGRAPH). [DATASET]
- Google Quick, Draw! — 50 M drawings, stroke sequences. [DATASET, has strokes]
- Sangkloy et al. 2016, Sketchy database (TOG). [DATASET]
- Lake et al. 2015, Omniglot with stroke data. [DATASET, has strokes]
- Kembhavi et al. 2016, AI2D diagrams with parse graphs (ECCV). [DATASET, has element/relation graphs]
- Hendrycks & Dietterich 2018, Icons-50 (arXiv v1 of the corruptions paper). [DATASET, style shift across vendors]
- Stallkamp et al. 2012, GTSRB traffic signs (Neural Networks). [DATASET]
- Yu et al. 2017, Sketch-a-Net (IJCV); Ha & Eck 2018, Sketch-RNN (ICLR).
- Carlier et al. 2020, DeepSVG (NeurIPS); Reddy et al. 2021, Im2Vec (CVPR); Dominici et al. 2020, PolyFit (TOG): raster-to-vector, i.e. candidate learned extractors for icons.
- Zeng et al. 2019, floor plan recognition (ICCV); line-segment floor-plan GNN 2023 (arXiv); Paliwal et al. 2021 Digitize-PID: engineering-drawing symbol detection followed by graph reasoning, the industrial version of this pipeline.

## 4. Baselines a reviewer will demand (with the motivating paper)

1. ResNet-18 with random-affine augmentation matched to the tested shift (Azulay & Weiss; Kauderer-Abrams; Biscione & Bowers).
2. ResNet-18 on bounding-box-cropped input (gives the pixel model the normalisation the graph gets for free; no paper, plain fairness).
3. ImageNet-pretrained ResNet-18 fine-tune and a frozen CLIP or DINO linear probe for the few-shot regime (Radford et al. 2021).
4. Parameter- and compute-matched small CNN (Bouthillier et al. 2021 on fair benchmarking).
5. Relation Network over CNN feature cells (Santoro et al. 2017): relations without explicit primitives.
6. Slot Attention plus relational head (Locatello et al. 2020; Mondal et al. 2023): learned objects instead of geometric primitives. Optional but it is the closest competitor.
7. Group- or scale-equivariant CNN (Cohen & Welling 2016; Sosnovik et al. 2020) for the rotation/scale experiments. Optional.
8. DeepSets / bag-of-primitives and an ordered-list MLP over primitives: isolates the value of relations and of permutation invariance (Battaglia et al. 2018).
9. Set transformer over primitives (already in PLAN.md) so results are not GINE-specific.

## 5. Candidate real-world domains (for framing, and for WP6)

Ranked by naturalness of primitive decomposition, public data, stakes.

1. **Engineering and process diagrams** (P&ID, circuit schematics, floor plans): meaning is entirely in symbol arrangement; industrial pipelines already are detect-then-graph; public data is thin but exists (AI2D for science diagrams, floor-plan datasets, Digitize-PID). High stakes, best fit for the introduction's motivating example.
2. **Hand-drawn sketches** (Quick, Draw!, TU-Berlin, Omniglot): stroke data gives a non-oracle, non-synthetic extractor; 10 of the project's classes already exist in Quick, Draw!. Best fit for a cheap real-data anchor (WP6).
3. **Road signs and pictograms** (GTSRB): defined by regulation as primitive compositions; "learn from the specification drawing, recognise in the wild" is a compelling few-shot story but needs photographs, so out of scope here.
4. **Icons and logos** (Icons-50, LLD): style shift across vendors is a ready-made H4 test; low stakes.

## 6. Agent attributions corrected or dropped

- Bongard-LOGO is Nie et al. 2020, not Ellis et al.
- CLEVR-CoGenT is part of Johnson et al. 2017, not a separate 2018 paper.
- "Neuro-symbolic visual reasoning: disentangling visual from reasoning" is Amizadeh et al. 2020, not Yi et al.
- "Understanding the computational demands underlying visual reasoning" is Vaishnav et al. 2022, not Tacchetti.
- Multi-Graph Transformer is Xu et al.; Sketchformer is Ribeiro et al. 2020; CompositionalNets is Kortylewski et al.
- Dropped as unverifiable or irrelevant: a 2026 arXiv "symbolic grounding bottleneck" preprint, a medial-axis 3D paper, a many-objective feature-selection paper offered as "methodology", a 2024 PAC-learnability preprint, and a scene-graph survey with a wrong URL.
