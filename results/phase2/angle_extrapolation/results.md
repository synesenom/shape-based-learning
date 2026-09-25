# phase2 / angle_extrapolation

Seeds: [0, 1, 2]. Test split drawn from fixed seed 12345, so the spread is over training stochasticity and the training draw, not over test sets.

Intervals are Student-t 95% confidence intervals over seeds (t = 4.303 at 3 seeds, not the normal-quantile 1.96).

Shift **view_angle**: `view_angle_range` [0.0, 30.0] → [50.0, 70.0].

Validation follows the *training* distribution: at selection time the shifted distribution is not available, and selecting on it would leak the test condition into training.

## 150 training examples per class

| model | in-distribution | shifted | drop | val accuracy | extractor F1 | params |
|---|---|---|---|---|---|---|
| CNN | 1.000 ± 0.000 | 0.756 ± 0.090 | 0.244 ± 0.090 | 1.000 ± 0.000 | - | 11,173,962 |
| CNN + aug | 0.999 ± 0.002 | 0.671 ± 0.169 | 0.328 ± 0.171 | 1.000 ± 0.000 | - | 11,173,962 |
| GNN (oracle) | 1.000 ± 0.000 | 0.960 ± 0.110 | 0.040 ± 0.110 | 1.000 ± 0.000 | 1.000 ± 0.000 | 57,549 |
| GNN (classical) | 0.988 ± 0.009 | 0.579 ± 0.079 | 0.409 ± 0.070 | 0.986 ± 0.016 | 0.503 ± 0.000 | 57,549 |
| ImageNet probe | 0.999 ± 0.002 | 0.532 ± 0.055 | 0.467 ± 0.055 | 1.000 ± 0.000 | - | 11,181,642 |
| Set transformer (oracle) | 1.000 ± 0.000 | 0.905 ± 0.086 | 0.095 ± 0.086 | 1.000 ± 0.000 | 1.000 ± 0.000 | 89,290 |
| Bag (oracle) | 0.900 ± 0.001 | 0.889 ± 0.028 | 0.011 ± 0.027 | 0.905 ± 0.022 | 1.000 ± 0.000 | 5,642 |
| CNN + view aug | 0.999 ± 0.003 | 0.749 ± 0.130 | 0.251 ± 0.128 | 1.000 ± 0.000 | - | 11,173,962 |
| GNN (oracle, affine frame) | 1.000 ± 0.000 | 0.997 ± 0.014 | 0.003 ± 0.014 | 1.000 ± 0.000 | 1.000 ± 0.000 | 57,549 |
| GNN (classical, affine frame) | 0.989 ± 0.003 | 0.688 ± 0.063 | 0.301 ± 0.062 | 0.988 ± 0.013 | 0.503 ± 0.000 | 57,549 |
| GNN (learned) | 0.998 ± 0.004 | 0.243 ± 0.021 | 0.755 ± 0.019 | 1.000 ± 0.000 | 0.379 ± 0.000 | 57,549 |
| GNN (learned, affine frame) | 0.998 ± 0.001 | 0.250 ± 0.052 | 0.748 ± 0.051 | 1.000 ± 0.000 | 0.379 ± 0.000 | 57,549 |
