# phase1 / novel_composition

Seeds: [0, 1, 2]. Test split drawn from fixed seed 12345, so the spread is over training stochasticity and the training draw, not over test sets.

Intervals are Student-t 95% confidence intervals over seeds (t = 4.303 at 3 seeds, not the normal-quantile 1.96).

Shift **novel_composition**: `template_set` base → novel.

Validation follows the *training* distribution: at selection time the shifted distribution is not available, and selecting on it would leak the test condition into training.

## 150 training examples per class

| model | in-distribution | shifted | drop | val accuracy | extractor F1 | params |
|---|---|---|---|---|---|---|
| CNN | 1.000 ± 0.000 | 0.922 ± 0.068 | 0.078 ± 0.068 | 1.000 ± 0.000 | - | 11,173,962 |
| CNN + aug | 1.000 ± 0.000 | 0.951 ± 0.107 | 0.049 ± 0.107 | 1.000 ± 0.000 | - | 11,173,962 |
| GNN (oracle) | 1.000 ± 0.000 | 0.490 ± 0.170 | 0.510 ± 0.170 | 1.000 ± 0.000 | 1.000 ± 0.000 | 57,549 |
| GNN (classical) | 0.987 ± 0.006 | 0.694 ± 0.012 | 0.293 ± 0.019 | 0.998 ± 0.004 | 0.647 ± 0.000 | 57,549 |
| ImageNet probe | 1.000 ± 0.000 | 0.980 ± 0.018 | 0.020 ± 0.018 | 1.000 ± 0.000 | - | 11,181,642 |
| Set transformer (oracle) | 1.000 ± 0.000 | 0.863 ± 0.077 | 0.137 ± 0.077 | 1.000 ± 0.000 | 1.000 ± 0.000 | 89,290 |
| Bag (oracle) | 0.901 ± 0.004 | 0.736 ± 0.101 | 0.165 ± 0.105 | 0.911 ± 0.041 | 1.000 ± 0.000 | 5,642 |
