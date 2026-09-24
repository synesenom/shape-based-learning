# phase1 / shift_position_scale

Seeds: [0, 1, 2]. Test split drawn from fixed seed 12345, so the spread is over training stochasticity and the training draw, not over test sets.

Intervals are Student-t 95% confidence intervals over seeds (t = 4.303 at 3 seeds, not the normal-quantile 1.96).

Shift **position_scale**: `object_scale_range` [0.4, 0.55] → [0.7, 0.9], `position_jitter` 0.03 → 0.25.

Validation follows the *training* distribution: at selection time the shifted distribution is not available, and selecting on it would leak the test condition into training.

## 150 training examples per class

| model | in-distribution | shifted | drop | val accuracy | extractor F1 | params |
|---|---|---|---|---|---|---|
| CNN | 0.997 ± 0.006 | 0.227 ± 0.115 | 0.770 ± 0.109 | 1.000 ± 0.000 | - | 11,173,962 |
| CNN + aug | 0.998 ± 0.005 | 0.930 ± 0.112 | 0.069 ± 0.108 | 1.000 ± 0.000 | - | 11,173,962 |
| GNN (oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 57,549 |
| GNN (classical) | 0.985 ± 0.009 | 0.879 ± 0.038 | 0.106 ± 0.032 | 0.989 ± 0.013 | 0.793 ± 0.000 | 57,549 |
| CNN + strong aug | 0.999 ± 0.004 | 0.936 ± 0.036 | 0.063 ± 0.032 | 1.000 ± 0.000 | - | 11,173,962 |
| ImageNet probe | 1.000 ± 0.000 | 0.893 ± 0.121 | 0.107 ± 0.121 | 1.000 ± 0.000 | - | 11,181,642 |
| Set transformer (oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 89,290 |
| Bag (oracle) | 0.901 ± 0.004 | 0.901 ± 0.004 | 0.000 ± 0.000 | 0.911 ± 0.041 | 1.000 ± 0.000 | 5,642 |
| GNN (oracle, max pool) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 57,549 |
