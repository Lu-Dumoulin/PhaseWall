# fitzhugh-nagumo

396 runs on 99 parameter points (eps, delta); analysis 0 min.

Reference labels (runs): uniform 134, dynamic 121, stripes 83, fronts 51, holes 7

## Steady state

24 runs (20 at points whose seeds end in different states) rerun to 2x, 4x the run time: label changed in 0 runs at 2x, 0 runs at 4x; largest change of the field between two snapshots at the longest time: 1.9e+00.

| run | eps | delta | seed | at_boundary | label | label_x2 | change_x2 | label_x4 | change_x4 |
|---|---|---|---|---|---|---|---|---|---|
| 298 | 0.199 | 0.178 | 2 | True | dynamic | dynamic | 0.88 | dynamic | 0.88 |
| 162 | 0.0315 | 0.632 | 2 | True | dynamic | dynamic | 1.91 | dynamic | 1.92 |
| 363 | 0.5 | 0.05 | 3 | True | uniform | uniform | 0 | uniform | 0 |
| 213 | 0.05 | 8 | 1 | True | stripes | stripes | 1e-06 | stripes | 0 |
| 314 | 0.199 | 2.25 | 2 | True | stripes | stripes | 6e-06 | stripes | 0 |
| 143 | 0.0199 | 8 | 3 | True | fronts | fronts | 0 | fronts | 0 |
| 30 | 0.005 | 4.24 | 2 | True | uniform | uniform | 0 | uniform | 0 |
| 349 | 0.316 | 2.25 | 1 | True | uniform | uniform | 0 | uniform | 0 |
| 337 | 0.316 | 0.335 | 1 | True | uniform | uniform | 0 | uniform | 0 |
| 7 | 0.005 | 0.0943 | 3 | True | uniform | uniform | 0 | uniform | 0 |
| 5 | 0.005 | 0.0943 | 1 | True | uniform | uniform | 0 | uniform | 0 |
| 331 | 0.316 | 0.0943 | 3 | True | uniform | uniform | 0 | uniform | 0 |
| 325 | 0.316 | 0.05 | 1 | True | uniform | uniform | 0 | uniform | 0 |
| 8 | 0.005 | 0.178 | 0 | True | dynamic | dynamic | 1.2 | dynamic | 1.19 |
| 196 | 0.05 | 0.632 | 0 | True | uniform | uniform | 0 | uniform | 0 |
| 244 | 0.0792 | 4.24 | 0 | False | stripes | stripes | 0 | stripes | 0 |
| 119 | 0.0199 | 0.178 | 3 | False | dynamic | dynamic | 1.84 | dynamic | 1.84 |
| 69 | 0.0079 | 8 | 1 | True | stripes | stripes | 0 | stripes | 0 |
| 104 | 0.0126 | 8 | 0 | True | fronts | fronts | 0 | fronts | 0 |
| 17 | 0.005 | 0.632 | 1 | True | uniform | uniform | 0 | uniform | 0 |
| 254 | 0.126 | 0.05 | 2 | False | dynamic | dynamic | 1.34 | dynamic | 1.34 |
| 318 | 0.199 | 4.24 | 2 | False | stripes | stripes | 2e-06 | stripes | 0 |
| 361 | 0.5 | 0.05 | 1 | True | uniform | uniform | 0 | uniform | 0 |
| 31 | 0.005 | 4.24 | 3 | True | dynamic | dynamic | 0.434 | dynamic | 0.532 |

## Scores

Each row: a description and a method; agree = points carrying the most common reference label of their group, ari = adjusted Rand index with the reference, pieces = connected pieces on the grid.

| describer | method | groups | noise_points | agree | points | ari | ari_all | pieces | noise_runs |
|---|---|---|---|---|---|---|---|---|---|
| hand | walls | 1 | 0 | 33 | 99 | 0 | 0 | 1 | 0 |
| hand | mixture | 8 | 0 | 90 | 99 | 0.554 | 0.554 | 23 | 0 |
| hand | kmeans | 8 | 0 | 88 | 99 | 0.537 | 0.537 | 22 | 0 |
| hand | hdbscan_10 | 6 | 18 | 75 | 99 | 0.699 | 0.515 | 17 | 0.205 |
| hand | hdbscan_20 | 5 | 22 | 71 | 99 | 0.71 | 0.496 | 16 | 0.23 |
| learned seed 0 | walls | 1 | 0 | 33 | 99 | 0 | 0 | 1 | 0 |
| learned seed 0 | mixture | 9 | 0 | 72 | 99 | 0.237 | 0.237 | 25 | 0 |
| learned seed 0 | kmeans | 10 | 0 | 74 | 99 | 0.238 | 0.238 | 32 | 0 |
| learned seed 0 | hdbscan_10 | 7 | 33 | 62 | 99 | 0.71 | 0.508 | 15 | 0.348 |
| learned seed 0 | hdbscan_20 | 4 | 45 | 28 | 99 | 0.104 | 0.159 | 13 | 0.439 |
| combined seed 0 | walls | 1 | 0 | 33 | 99 | 0 | 0 | 1 | 0 |
| combined seed 0 | mixture | 8 | 0 | 88 | 99 | 0.542 | 0.542 | 23 | 0 |
| combined seed 0 | kmeans | 8 | 0 | 87 | 99 | 0.529 | 0.529 | 21 | 0 |
| combined seed 0 | hdbscan_10 | 6 | 12 | 80 | 99 | 0.734 | 0.598 | 15 | 0.146 |
| combined seed 0 | hdbscan_20 | 5 | 20 | 74 | 99 | 0.739 | 0.531 | 16 | 0.197 |
| learned seed 1 | walls | 1 | 0 | 33 | 99 | 0 | 0 | 1 | 0 |
| learned seed 1 | mixture | 10 | 0 | 74 | 99 | 0.256 | 0.256 | 29 | 0 |
| learned seed 1 | kmeans | 10 | 0 | 74 | 99 | 0.256 | 0.256 | 29 | 0 |
| learned seed 1 | hdbscan_10 | 8 | 44 | 52 | 99 | 0.639 | 0.3 | 20 | 0.437 |
| learned seed 1 | hdbscan_20 | 3 | 44 | 23 | 99 | -0.035 | 0.116 | 11 | 0.427 |
| combined seed 1 | walls | 1 | 0 | 33 | 99 | 0 | 0 | 1 | 0 |
| combined seed 1 | mixture | 10 | 0 | 89 | 99 | 0.513 | 0.513 | 26 | 0 |
| combined seed 1 | kmeans | 6 | 0 | 88 | 99 | 0.569 | 0.569 | 20 | 0 |
| combined seed 1 | hdbscan_10 | 7 | 15 | 77 | 99 | 0.727 | 0.56 | 16 | 0.167 |
| combined seed 1 | hdbscan_20 | 5 | 19 | 75 | 99 | 0.742 | 0.548 | 16 | 0.194 |

Probe (logistic regression of the reference labels, held-out points, balanced accuracy): hand 0.765, learned seed 0 0.698, combined seed 0 0.787, learned seed 1 0.701, combined seed 1 0.787

## Trainings (last epoch)

| name | loss | collapse_loss | alignment | uniformity | effective_rank_h |
|---|---|---|---|---|---|
| learned seed 0 | 2.23 | 5.54 | 0.995 | -2.56 | 34 |
| learned seed 1 | 2.25 | 5.54 | 0.995 | -2.56 | 32.1 |
