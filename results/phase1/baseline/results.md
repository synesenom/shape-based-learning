# phase1 / baseline

Seeds: [0, 1, 2]. Test split drawn from fixed seed 12345, so the spread is over training stochasticity and the training draw, not over test sets.

Intervals are Student-t 95% confidence intervals over seeds (t = 4.303 at 3 seeds, not the normal-quantile 1.96).

## 150 training examples per class

| model | test accuracy | val accuracy | tree/arrow_sign | extractor F1 | train (s) | params |
|---|---|---|---|---|---|---|
| CNN | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | - | 604 ± 262 | - |
| CNN + aug | 0.999 ± 0.006 | 1.000 ± 0.000 | 1.000 ± 0.000 | - | 587 ± 139 | - |
| GNN (oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 7 ± 2 | - |
| GNN (classical) | 0.992 ± 0.003 | 0.998 ± 0.000 | 1.000 ± 0.000 | 0.734 ± 0.000 | 18 ± 12 | - |
