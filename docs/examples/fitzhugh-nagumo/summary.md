# fitzhugh-nagumo

396 runs on 99 parameter points (eps, delta); analysis 0 min.

Reference labels (runs): uniform 134, dynamic 121, patterns 90, fronts 51

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
| hand | mixture | 8 | 0 | 91 | 99 | 0.563 | 0.563 | 23 | 0 |
| hand | kmeans | 8 | 0 | 89 | 99 | 0.545 | 0.545 | 22 | 0 |
| hand | hdbscan_10 | 6 | 18 | 75 | 99 | 0.699 | 0.512 | 17 | 0.205 |
| hand | hdbscan_20 | 5 | 22 | 71 | 99 | 0.71 | 0.496 | 16 | 0.23 |
| learned seed 0 | walls | 2 | 0 | 44 | 99 | 0.215 | 0.215 | 3 | 0 |
| learned seed 0 | mixture | 10 | 0 | 90 | 99 | 0.435 | 0.435 | 25 | 0 |
| learned seed 0 | kmeans | 10 | 0 | 90 | 99 | 0.435 | 0.435 | 25 | 0 |
| learned seed 0 | hdbscan_10 | 8 | 14 | 79 | 99 | 0.632 | 0.498 | 21 | 0.116 |
| learned seed 0 | hdbscan_20 | 6 | 38 | 55 | 99 | 0.606 | 0.683 | 15 | 0.366 |
| combined seed 0 | walls | 1 | 0 | 33 | 99 | 0 | 0 | 1 | 0 |
| combined seed 0 | mixture | 9 | 0 | 90 | 99 | 0.493 | 0.493 | 24 | 0 |
| combined seed 0 | kmeans | 8 | 0 | 89 | 99 | 0.559 | 0.559 | 21 | 0 |
| combined seed 0 | hdbscan_10 | 7 | 20 | 73 | 99 | 0.75 | 0.514 | 12 | 0.215 |
| combined seed 0 | hdbscan_20 | 5 | 18 | 75 | 99 | 0.72 | 0.544 | 16 | 0.177 |
| learned seed 1 | walls | 2 | 0 | 42 | 99 | 0.19 | 0.19 | 3 | 0 |
| learned seed 1 | mixture | 10 | 0 | 89 | 99 | 0.42 | 0.42 | 27 | 0 |
| learned seed 1 | kmeans | 10 | 0 | 88 | 99 | 0.432 | 0.432 | 27 | 0 |
| learned seed 1 | hdbscan_10 | 8 | 17 | 74 | 99 | 0.589 | 0.496 | 22 | 0.179 |
| learned seed 1 | hdbscan_20 | 7 | 32 | 62 | 99 | 0.652 | 0.593 | 17 | 0.311 |
| combined seed 1 | walls | 1 | 0 | 33 | 99 | 0 | 0 | 1 | 0 |
| combined seed 1 | mixture | 10 | 0 | 90 | 99 | 0.457 | 0.457 | 25 | 0 |
| combined seed 1 | kmeans | 8 | 0 | 89 | 99 | 0.549 | 0.549 | 22 | 0 |
| combined seed 1 | hdbscan_10 | 6 | 19 | 74 | 99 | 0.75 | 0.522 | 15 | 0.187 |
| combined seed 1 | hdbscan_20 | 5 | 18 | 75 | 99 | 0.72 | 0.544 | 16 | 0.182 |

Probe (logistic regression of the reference labels, held-out points, balanced accuracy): hand 0.930, learned seed 0 0.943, combined seed 0 0.957, learned seed 1 0.937, combined seed 1 0.959

## Trainings (last epoch)

| name | loss | collapse_loss | alignment | uniformity | effective_rank_h |
|---|---|---|---|---|---|
| learned seed 0 | 2.67 | 5.54 | 0.961 | -2.47 | 27 |
| learned seed 1 | 2.67 | 5.54 | 0.959 | -2.48 | 26.3 |
