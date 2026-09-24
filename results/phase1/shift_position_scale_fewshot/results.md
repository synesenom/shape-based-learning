# phase1 / shift_position_scale_fewshot

Seeds: [0, 1, 2]. Test split drawn from fixed seed 12345, so the spread is over training stochasticity and the training draw, not over test sets.

Intervals are Student-t 95% confidence intervals over seeds (t = 4.303 at 3 seeds, not the normal-quantile 1.96).

Shift **position_scale**: `object_scale_range` [0.4, 0.55] → [0.7, 0.9], `position_jitter` 0.03 → 0.25.

Validation follows the *training* distribution: at selection time the shifted distribution is not available, and selecting on it would leak the test condition into training.

## 25 training examples per class

| model | in-distribution | shifted | drop | val accuracy | extractor F1 | params |
|---|---|---|---|---|---|---|
| CNN | 0.989 ± 0.010 | 0.206 ± 0.053 | 0.784 ± 0.053 | 0.993 ± 0.013 | - | 11,173,962 |
| CNN + aug | 0.997 ± 0.006 | 0.726 ± 0.045 | 0.271 ± 0.045 | 1.000 ± 0.000 | - | 11,173,962 |
| GNN (oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 57,549 |
| GNN (classical) | 0.918 ± 0.032 | 0.843 ± 0.066 | 0.075 ± 0.043 | 0.927 ± 0.037 | 0.793 ± 0.000 | 57,549 |
| CNN + strong aug | 0.998 ± 0.005 | 0.936 ± 0.005 | 0.062 ± 0.008 | 1.000 ± 0.000 | - | 11,173,962 |
| ImageNet probe | 0.997 ± 0.007 | 0.864 ± 0.073 | 0.133 ± 0.067 | 0.998 ± 0.007 | - | 11,181,642 |
| Set transformer (oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 89,290 |
| Bag (oracle) | 0.900 ± 0.001 | 0.900 ± 0.001 | 0.000 ± 0.000 | 0.914 ± 0.039 | 1.000 ± 0.000 | 5,642 |
