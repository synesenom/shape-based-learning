# phase1 / shift_position_scale_fewshot

Seeds: [0, 1, 2]. Test split drawn from fixed seed 12345, so the spread is over training stochasticity and the training draw, not over test sets.

Intervals are Student-t 95% confidence intervals over seeds (t = 4.303 at 3 seeds, not the normal-quantile 1.96).

Shift **position_scale**: `object_scale_range` [0.4, 0.55] → [0.7, 0.9], `position_jitter` 0.03 → 0.25.

Validation follows the *training* distribution: at selection time the shifted distribution is not available, and selecting on it would leak the test condition into training.

## 25 training examples per class

| model | in-distribution | shifted | drop | val accuracy | extractor F1 | params |
|---|---|---|---|---|---|---|
| CNN | 0.978 ± 0.015 | 0.199 ± 0.063 | 0.779 ± 0.060 | 0.986 ± 0.007 | - | 11,173,962 |
| CNN + aug | 0.760 ± 0.728 | 0.360 ± 0.455 | 0.400 ± 0.526 | 0.757 ± 0.726 | - | 11,173,962 |
| GNN (oracle) | 1.000 ± 0.001 | 1.000 ± 0.001 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 57,549 |
| GNN (classical) | 0.902 ± 0.022 | 0.854 ± 0.133 | 0.048 ± 0.147 | 0.916 ± 0.022 | 0.793 ± 0.000 | 57,549 |
| CNN + strong aug | 0.864 ± 0.355 | 0.651 ± 0.494 | 0.213 ± 0.139 | 0.867 ± 0.351 | - | 11,173,962 |
| ImageNet probe | 0.997 ± 0.007 | 0.864 ± 0.073 | 0.133 ± 0.067 | 0.998 ± 0.007 | - | 11,181,642 |
| Set transformer (oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 89,290 |
| Bag (oracle) | 0.894 ± 0.019 | 0.894 ± 0.019 | 0.000 ± 0.000 | 0.908 ± 0.020 | 1.000 ± 0.000 | 5,642 |
