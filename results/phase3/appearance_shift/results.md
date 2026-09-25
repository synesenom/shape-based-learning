# phase3 / appearance_shift

Seeds: [0, 1, 2]. Test split drawn from fixed seed 12345, so the spread is over training stochasticity and the training draw, not over test sets.

Intervals are Student-t 95% confidence intervals over seeds (t = 4.303 at 3 seeds, not the normal-quantile 1.96).

Shift **appearance**: `texture` flat → ['noise', 'stripes', 'dots', 'photo'], `palette` default → unseen, `clutter` False → True, `occlusion_range` [0.0, 0.0] → [0.1, 0.4], `noise_std` 0.0 → 10.0, `blur_radius` 0.0 → 0.5.

Validation follows the *training* distribution: at selection time the shifted distribution is not available, and selecting on it would leak the test condition into training.

## 150 training examples per class

| model | in-distribution | shifted | drop | val accuracy | extractor F1 | params |
|---|---|---|---|---|---|---|
| CNN | 1.000 ± 0.000 | 0.146 ± 0.058 | 0.854 ± 0.058 | 1.000 ± 0.000 | - | 11,173,962 |
| CNN + aug | 0.999 ± 0.002 | 0.115 ± 0.058 | 0.884 ± 0.055 | 1.000 ± 0.000 | - | 11,173,962 |
| GNN (oracle) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 57,549 |
| GNN (classical) | 0.988 ± 0.009 | 0.137 ± 0.007 | 0.851 ± 0.014 | 0.986 ± 0.016 | 0.072 ± 0.000 | 57,549 |
| ImageNet probe | 0.999 ± 0.002 | 0.279 ± 0.042 | 0.720 ± 0.040 | 1.000 ± 0.000 | - | 11,181,642 |
| CNN + colour aug | 0.999 ± 0.001 | 0.133 ± 0.078 | 0.866 ± 0.079 | 1.000 ± 0.000 | - | 11,173,962 |
| GNN (learned) | 0.998 ± 0.004 | 0.137 ± 0.021 | 0.862 ± 0.020 | 1.000 ± 0.000 | 0.122 ± 0.000 | 57,549 |
| GNN (appearance-trained detector) | 0.999 ± 0.001 | 0.834 ± 0.079 | 0.165 ± 0.081 | 1.000 ± 0.000 | 0.889 ± 0.000 | 57,549 |
