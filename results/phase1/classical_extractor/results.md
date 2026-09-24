# Classical extractor quality on clean Phase 1 drawings

50 drawings per class, IoU >= 0.5 same-type matching, no distractors.

## classical@64px: F1 0.780 (precision 0.830, recall 0.756)

| bicycle | car | truck | cat_face | house | tree | arrow_sign | snowman | person | fish |
|---|---|---|---|---|---|---|---|---|---|
| 0.47 | 0.93 | 0.99 | 0.45 | 0.84 | 0.86 | 0.87 | 0.98 | 0.59 | 0.82 |

Recall by primitive type: ellipse 0.96, line 0.17, quadrilateral 0.97, triangle 0.44

## classical@128px: F1 0.830 (precision 0.844, recall 0.822)

| bicycle | car | truck | cat_face | house | tree | arrow_sign | snowman | person | fish |
|---|---|---|---|---|---|---|---|---|---|
| 0.48 | 1.00 | 1.00 | 0.79 | 0.96 | 0.88 | 0.87 | 1.00 | 0.48 | 0.84 |

Recall by primitive type: ellipse 0.95, line 0.24, quadrilateral 0.94, triangle 0.75

## classical_v2@64px: F1 0.818 (precision 0.870, recall 0.793)

| bicycle | car | truck | cat_face | house | tree | arrow_sign | snowman | person | fish |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.93 | 0.99 | 0.64 | 0.88 | 0.88 | 0.89 | 0.98 | 0.61 | 0.88 |

Recall by primitive type: ellipse 0.95, line 0.19, quadrilateral 0.97, triangle 0.59

## classical_v2@128px: F1 0.847 (precision 0.861, recall 0.838)

| bicycle | car | truck | cat_face | house | tree | arrow_sign | snowman | person | fish |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 1.00 | 1.00 | 0.88 | 0.96 | 0.88 | 0.87 | 1.00 | 0.53 | 0.84 |

Recall by primitive type: ellipse 0.95, line 0.30, quadrilateral 0.94, triangle 0.79
