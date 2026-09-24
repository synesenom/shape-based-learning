# phase2 / relation_rotation: accuracy by rot bin

Mean ± Student-t 95% CI over 3 seeds. Training range: 0-30.

## 150 training examples per class

| model | rot_00-30 | rot_30-60 | rot_60-90 |
|---|---|---|---|
| CNN | 0.999 ± 0.003 | 0.715 ± 0.063 | 0.253 ± 0.076 |
| CNN + view/rot aug | 0.998 ± 0.004 | 0.966 ± 0.059 | 0.727 ± 0.142 |
| GNN (oracle) | 1.000 ± 0.000 | 0.948 ± 0.011 | 0.807 ± 0.057 |
| GNN (oracle, affine frame) | 1.000 ± 0.000 | 0.943 ± 0.073 | 0.802 ± 0.113 |
| GNN (classical) | 0.973 ± 0.014 | 0.900 ± 0.049 | 0.621 ± 0.175 |
| Set transformer (oracle) | 0.998 ± 0.009 | 0.959 ± 0.095 | 0.782 ± 0.201 |
| Bag (oracle) | 0.898 ± 0.009 | 0.899 ± 0.006 | 0.899 ± 0.006 |

Extractor F1 per bin (mean over seeds):

| model | rot_00-30 | rot_30-60 | rot_60-90 |
|---|---|---|---|
| GNN (oracle) | 1.000 | 1.000 | 1.000 |
| GNN (oracle, affine frame) | 1.000 | 1.000 | 1.000 |
| GNN (classical) | 0.667 | 0.702 | 0.687 |
| Set transformer (oracle) | 1.000 | 1.000 | 1.000 |
| Bag (oracle) | 1.000 | 1.000 | 1.000 |
