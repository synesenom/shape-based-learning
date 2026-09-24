# phase2 / relation_rotation

Seeds: [0, 1, 2]. Test split drawn from fixed seed 12345, so the spread is over training stochasticity and the training draw, not over test sets.

Intervals are Student-t 95% confidence intervals over seeds (t = 4.303 at 3 seeds, not the normal-quantile 1.96).

Shift **rotation**: `rotation_range` [0.0, 30.0] → [60.0, 90.0].

Validation follows the *training* distribution: at selection time the shifted distribution is not available, and selecting on it would leak the test condition into training.

## 150 training examples per class

| model | in-distribution | shifted | drop | val accuracy | extractor F1 | params |
|---|---|---|---|---|---|---|
| CNN | 0.999 ± 0.003 | 0.253 ± 0.076 | 0.746 ± 0.074 | 1.000 ± 0.000 | - | 11,173,962 |
| GNN (oracle) | 1.000 ± 0.000 | 0.807 ± 0.057 | 0.193 ± 0.057 | 1.000 ± 0.000 | 1.000 ± 0.000 | 57,549 |
| GNN (classical) | 0.973 ± 0.014 | 0.621 ± 0.175 | 0.352 ± 0.168 | 0.976 ± 0.004 | 0.687 ± 0.000 | 57,549 |
| Set transformer (oracle) | 0.998 ± 0.009 | 0.782 ± 0.201 | 0.216 ± 0.193 | 1.000 ± 0.000 | 1.000 ± 0.000 | 89,290 |
| Bag (oracle) | 0.898 ± 0.009 | 0.899 ± 0.006 | -0.001 ± 0.003 | 0.906 ± 0.025 | 1.000 ± 0.000 | 5,642 |
| cnn_aug_view_rot | 0.998 ± 0.004 | 0.727 ± 0.142 | 0.270 ± 0.142 | 1.000 ± 0.000 | - | 11,173,962 |
| gnn_oracle_affine | 1.000 ± 0.000 | 0.802 ± 0.113 | 0.198 ± 0.113 | 1.000 ± 0.000 | 1.000 ± 0.000 | 57,549 |
