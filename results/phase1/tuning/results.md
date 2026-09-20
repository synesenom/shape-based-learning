# phase1 / tuning

Seeds: [0]. Test split drawn from fixed seed 12345, so the spread is over training stochasticity and the training draw, not over test sets.

Intervals are Student-t 95% confidence intervals over seeds (t = 12.706 at 1 seeds, not the normal-quantile 1.96).

## 150 training examples per class

| model | test accuracy | val accuracy | tree/arrow_sign | extractor F1 | train (s) | params |
|---|---|---|---|---|---|---|
| cnn_adam_lr1e-3 | 1.000 (1 seed) | 1.000 (1 seed) | 1.000 (1 seed) | - | 571 (1 seed) | 11,173,962 |
| cnn_adam_lr3e-4 | 0.995 (1 seed) | 1.000 (1 seed) | 1.000 (1 seed) | - | 510 (1 seed) | 11,173,962 |
| cnn_sgd_lr5e-2 | 1.000 (1 seed) | 1.000 (1 seed) | 1.000 (1 seed) | - | 590 (1 seed) | 11,173,962 |
| cnn_adamw_lr1e-3 | 1.000 (1 seed) | 1.000 (1 seed) | 1.000 (1 seed) | - | 456 (1 seed) | 11,173,962 |
| gnn_adam_lr1e-3 | 1.000 (1 seed) | 1.000 (1 seed) | 1.000 (1 seed) | 1.000 (1 seed) | 7 (1 seed) | 57,549 |
| gnn_adam_lr3e-3 | 1.000 (1 seed) | 1.000 (1 seed) | 1.000 (1 seed) | 1.000 (1 seed) | 6 (1 seed) | 57,549 |
| gnn_sgd_lr5e-2 | 1.000 (1 seed) | 1.000 (1 seed) | 1.000 (1 seed) | 1.000 (1 seed) | 6 (1 seed) | 57,549 |
| gnn_adamw_lr1e-3 | 1.000 (1 seed) | 1.000 (1 seed) | 1.000 (1 seed) | 1.000 (1 seed) | 6 (1 seed) | 57,549 |
