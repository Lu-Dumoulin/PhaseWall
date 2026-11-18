# gray-scott

2100 runs on 525 parameter points (F, k); analysis 50 min.

Reference labels (runs): uniform (high v) 827, uniform 577, stripes 267, dynamic 239, isolated spots 91, spots 68, holes 31

## Steady state

24 runs (13 at points whose seeds end in different states) rerun to 2x, 4x the run time: label changed in 2 runs at 2x, 3 runs at 4x; largest change of the field between two snapshots at the longest time: 7.5e-01.

| run | F | k | seed | at_boundary | label | label_x2 | change_x2 | label_x4 | change_x4 |
|---|---|---|---|---|---|---|---|---|---|
| 1244 | 0.045 | 0.0663 | 0 | True | isolated spots | isolated spots | 0 | isolated spots | 0 |
| 1309 | 0.0475 | 0.06 | 1 | True | holes | stripes | 0.000114 | holes | 0.00251 |
| 538 | 0.025 | 0.055 | 2 | True | stripes | stripes | 0.000157 | dynamic | 0.299 |
| 2067 | 0.07 | 0.06 | 3 | True | uniform (high v) | uniform (high v) | 4e-06 | uniform (high v) | 4e-06 |
| 556 | 0.025 | 0.0612 | 0 | True | spots | spots | 0 | spots | 0 |
| 554 | 0.025 | 0.06 | 2 | True | spots | dynamic | 0.159 | dynamic | 0.128 |
| 1587 | 0.055 | 0.0675 | 3 | True | uniform | uniform | 0 | uniform | 0 |
| 1246 | 0.045 | 0.0663 | 2 | True | isolated spots | isolated spots | 0 | isolated spots | 0 |
| 634 | 0.0275 | 0.0587 | 2 | True | spots | spots | 0.0726 | spots | 0.0615 |
| 387 | 0.02 | 0.06 | 3 | True | uniform | uniform | 0 | uniform | 0 |
| 735 | 0.03 | 0.0638 | 3 | True | uniform | uniform | 0 | uniform | 0 |
| 1311 | 0.0475 | 0.06 | 3 | True | stripes | stripes | 1e-05 | stripes | 2.6e-05 |
| 1777 | 0.0625 | 0.0488 | 1 | False | uniform (high v) | uniform (high v) | 3e-05 | uniform (high v) | 1.8e-05 |
| 34 | 0.01 | 0.055 | 2 | False | uniform | uniform | 0 | uniform | 0 |
| 1068 | 0.04 | 0.0638 | 0 | False | stripes | stripes | 0.000608 | spots | 0.000103 |
| 1333 | 0.0475 | 0.0675 | 1 | False | isolated spots | isolated spots | 0 | isolated spots | 0 |
| 645 | 0.0275 | 0.0625 | 1 | True | spots | spots | 0 | spots | 0 |
| 365 | 0.02 | 0.0537 | 1 | False | dynamic | dynamic | 0.634 | dynamic | 0.754 |
| 565 | 0.025 | 0.0638 | 1 | False | uniform | uniform | 0 | uniform | 0 |
| 85 | 0.0125 | 0.045 | 1 | False | uniform | uniform | 0 | uniform | 0 |
| 1365 | 0.05 | 0.0512 | 1 | False | uniform (high v) | uniform (high v) | 1e-06 | uniform (high v) | 2e-06 |
| 1707 | 0.06 | 0.0525 | 3 | False | uniform (high v) | uniform (high v) | 1e-05 | uniform (high v) | 1.2e-05 |
| 1916 | 0.065 | 0.0663 | 0 | False | uniform | uniform | 0 | uniform | 0 |
| 156 | 0.0125 | 0.0675 | 0 | False | uniform | uniform | 0 | uniform | 0 |

## Scores

Each row: a description and a method; agree = points carrying the most common reference label of their group, ari = adjusted Rand index with the reference, pieces = connected pieces on the grid.

| describer | method | groups | noise_points | agree | points | ari | ari_all | pieces | noise_runs |
|---|---|---|---|---|---|---|---|---|---|
| hand | walls | 10 | 0 | 482 | 525 | 0.8 | 0.8 | 10 | 0 |
| hand | mixture | 10 | 0 | 515 | 525 | 0.953 | 0.953 | 35 | 0 |
| hand | kmeans | 7 | 0 | 516 | 525 | 0.987 | 0.987 | 23 | 0 |
| hand | hdbscan_10 | 22 | 26 | 490 | 525 | 0.78 | 0.754 | 36 | 0.062 |
| hand | hdbscan_20 | 19 | 42 | 482 | 525 | 0.801 | 0.749 | 38 | 0.102 |
| learned seed 0 | walls | 1 | 0 | 207 | 525 | 0 | 0 | 1 | 0 |
| learned seed 0 | mixture | 10 | 0 | 406 | 525 | 0.451 | 0.451 | 40 | 0 |
| learned seed 0 | kmeans | 10 | 0 | 421 | 525 | 0.483 | 0.483 | 34 | 0 |
| learned seed 0 | hdbscan_10 | 8 | 103 | 376 | 525 | 0.677 | 0.519 | 20 | 0.2 |
| learned seed 0 | hdbscan_20 | 3 | 168 | 331 | 525 | 0.862 | 0.596 | 8 | 0.324 |
| combined seed 0 | walls | 8 | 0 | 503 | 525 | 0.86 | 0.86 | 8 | 0 |
| combined seed 0 | mixture | 10 | 0 | 517 | 525 | 0.758 | 0.758 | 27 | 0 |
| combined seed 0 | kmeans | 4 | 0 | 356 | 525 | 0.535 | 0.535 | 10 | 0 |
| combined seed 0 | hdbscan_10 | 21 | 48 | 476 | 525 | 0.822 | 0.748 | 36 | 0.109 |
| combined seed 0 | hdbscan_20 | 17 | 72 | 453 | 525 | 0.822 | 0.725 | 30 | 0.15 |
| learned seed 1 | walls | 1 | 0 | 207 | 525 | 0 | 0 | 1 | 0 |
| learned seed 1 | mixture | 10 | 0 | 392 | 525 | 0.42 | 0.42 | 32 | 0 |
| learned seed 1 | kmeans | 10 | 0 | 398 | 525 | 0.451 | 0.451 | 35 | 0 |
| learned seed 1 | hdbscan_10 | 12 | 170 | 355 | 525 | 0.859 | 0.493 | 23 | 0.323 |
| learned seed 1 | hdbscan_20 | 6 | 264 | 256 | 525 | 0.933 | 0.455 | 11 | 0.504 |
| combined seed 1 | walls | 8 | 0 | 503 | 525 | 0.86 | 0.86 | 8 | 0 |
| combined seed 1 | mixture | 10 | 0 | 516 | 525 | 0.754 | 0.754 | 32 | 0 |
| combined seed 1 | kmeans | 4 | 0 | 356 | 525 | 0.535 | 0.535 | 10 | 0 |
| combined seed 1 | hdbscan_10 | 21 | 56 | 469 | 525 | 0.827 | 0.742 | 36 | 0.128 |
| combined seed 1 | hdbscan_20 | 16 | 80 | 443 | 525 | 0.845 | 0.717 | 28 | 0.169 |

Probe (logistic regression of the reference labels, held-out points, balanced accuracy): hand 0.962, learned seed 0 0.924, combined seed 0 0.965, learned seed 1 0.917, combined seed 1 0.962

## Trainings (last epoch)

| name | loss | collapse_loss | alignment | uniformity | effective_rank_h |
|---|---|---|---|---|---|
| learned seed 0 | 2.01 | 5.54 | 0.997 | -2.41 | 90.5 |
| learned seed 1 | 2.02 | 5.54 | 0.997 | -2.4 | 91.4 |
