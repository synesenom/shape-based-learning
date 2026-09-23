# phase1 / novel_composition_pooling

Seeds: [0, 1, 2]. Test split drawn from fixed seed 12345, so the spread is over training stochasticity and the training draw, not over test sets.

Intervals are Student-t 95% confidence intervals over seeds (t = 4.303 at 3 seeds, not the normal-quantile 1.96).

Shift **novel_composition**: `template_set` base → novel.

Validation follows the *training* distribution: at selection time the shifted distribution is not available, and selecting on it would leak the test condition into training.

## 150 training examples per class

| model | in-distribution | shifted | drop | val accuracy | extractor F1 | params |
|---|---|---|---|---|---|---|
| GNN (oracle) | 1.000 ± 0.000 | 0.501 ± 0.194 | 0.499 ± 0.194 | 1.000 ± 0.000 | 1.000 ± 0.000 | 57,549 |
| GNN (classical) | 0.987 ± 0.006 | 0.694 ± 0.012 | 0.293 ± 0.019 | 0.998 ± 0.004 | 0.647 ± 0.000 | 57,549 |
| gnn_oracle_max | 1.000 ± 0.000 | 0.695 ± 0.220 | 0.305 ± 0.220 | 1.000 ± 0.000 | 1.000 ± 0.000 | 57,549 |
| gnn_oracle_attn | 1.000 ± 0.000 | 0.527 ± 0.311 | 0.473 ± 0.311 | 1.000 ± 0.000 | 1.000 ± 0.000 | 61,774 |
| gnn_classical_max | 0.992 ± 0.008 | 0.690 ± 0.150 | 0.302 ± 0.142 | 0.997 ± 0.004 | 0.647 ± 0.000 | 57,549 |
| gnn_classical_attn | 0.990 ± 0.005 | 0.712 ± 0.143 | 0.278 ± 0.147 | 0.998 ± 0.006 | 0.647 ± 0.000 | 61,774 |
