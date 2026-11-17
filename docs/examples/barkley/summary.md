# barkley

660 runs on 165 parameter points (a, b); analysis 14 min.

Reference labels (runs): spirals 363, rest 198, plane waves 99

## Steady state

24 runs (14 at points whose seeds end in different states) rerun to 2x, 4x the run time: label changed in 3 runs at 2x, 6 runs at 4x; largest change of the field between two snapshots at the longest time: 1.9e+00.

| run | a | b | seed | at_boundary | label | label_x2 | change_x2 | label_x4 | change_x4 |
|---|---|---|---|---|---|---|---|---|---|
| 642 | 1 | 0.06 | 2 | True | spirals | spirals | 1.81 | spirals | 1.83 |
| 145 | 0.45 | 0.03 | 1 | True | plane waves | plane waves | 1.18 | plane waves | 1.18 |
| 393 | 0.7 | 0.1 | 1 | True | spirals | spirals | 1.25 | plane waves | 1.25 |
| 626 | 1 | 0.02 | 2 | True | spirals | spirals | 1.92 | spirals | 1.92 |
| 537 | 0.9 | 0.02 | 1 | True | rest | rest | 0 | rest | 0 |
| 335 | 0.65 | 0.06 | 3 | True | spirals | plane waves | 1.35 | plane waves | 1.35 |
| 428 | 0.75 | 0.08 | 0 | True | spirals | spirals | 1.38 | plane waves | 1.4 |
| 525 | 0.85 | 0.1 | 1 | True | spirals | spirals | 1.36 | spirals | 1.35 |
| 45 | 0.35 | 0 | 1 | True | spirals | spirals | 1.15 | spirals | 1.14 |
| 379 | 0.7 | 0.06 | 3 | True | spirals | spirals | 1.4 | plane waves | 1.4 |
| 239 | 0.55 | 0.04 | 3 | True | spirals | plane waves | 1.27 | plane waves | 1.27 |
| 531 | 0.9 | 0 | 3 | True | spirals | spirals | 1.77 | spirals | 1.78 |
| 551 | 0.9 | 0.05 | 3 | False | spirals | spirals | 1.66 | spirals | 1.66 |
| 10 | 0.3 | 0.02 | 2 | False | rest | rest | 0 | rest | 0 |
| 329 | 0.65 | 0.05 | 1 | False | spirals | spirals | 1.27 | spirals | 1.3 |
| 412 | 0.75 | 0.04 | 0 | False | spirals | spirals | 1.48 | spirals | 1.46 |
| 199 | 0.5 | 0.05 | 3 | False | plane waves | plane waves | 1.17 | plane waves | 1.17 |
| 114 | 0.4 | 0.06 | 2 | False | rest | rest | 0 | rest | 0 |
| 174 | 0.45 | 0.1 | 2 | False | rest | rest | 0 | rest | 0 |
| 26 | 0.3 | 0.06 | 2 | False | rest | rest | 0 | rest | 0 |
| 426 | 0.75 | 0.07 | 2 | False | spirals | plane waves | 1.42 | plane waves | 1.42 |
| 534 | 0.9 | 0.01 | 2 | True | spirals | spirals | 1.77 | spirals | 1.75 |
| 601 | 0.95 | 0.07 | 1 | False | spirals | spirals | 1.68 | spirals | 1.67 |
| 49 | 0.35 | 0.01 | 1 | True | plane waves | plane waves | 1.16 | plane waves | 1.16 |

## Scores

Each row: a description and a method; agree = points carrying the most common reference label of their group, ari = adjusted Rand index with the reference, pieces = connected pieces on the grid.

| describer | method | groups | noise_points | agree | points | ari | ari_all | pieces | noise_runs |
|---|---|---|---|---|---|---|---|---|---|
| hand | walls | 2 | 0 | 141 | 165 | 0.672 | 0.672 | 2 | 0 |
| hand | mixture | 10 | 0 | 154 | 165 | 0.435 | 0.435 | 34 | 0 |
| hand | kmeans | 2 | 0 | 140 | 165 | 0.641 | 0.641 | 3 | 0 |
| hand | hdbscan_10 | 8 | 24 | 139 | 165 | 0.825 | 0.61 | 23 | 0.13 |
| hand | hdbscan_20 | 2 | 0 | 140 | 165 | 0.641 | 0.641 | 3 | 0.002 |
| learned seed 0 | walls | 1 | 0 | 99 | 165 | 0 | 0 | 1 | 0 |
| learned seed 0 | mixture | 10 | 0 | 156 | 165 | 0.261 | 0.261 | 23 | 0 |
| learned seed 0 | kmeans | 10 | 0 | 156 | 165 | 0.261 | 0.261 | 24 | 0 |
| learned seed 0 | hdbscan_10 | 2 | 5 | 137 | 165 | 0.666 | 0.623 | 2 | 0.047 |
| learned seed 0 | hdbscan_20 | 3 | 56 | 107 | 165 | 0.962 | 0.502 | 9 | 0.321 |
| combined seed 0 | walls | 2 | 0 | 141 | 165 | 0.672 | 0.672 | 2 | 0 |
| combined seed 0 | mixture | 10 | 0 | 156 | 165 | 0.41 | 0.41 | 35 | 0 |
| combined seed 0 | kmeans | 2 | 0 | 140 | 165 | 0.641 | 0.641 | 3 | 0 |
| combined seed 0 | hdbscan_10 | 7 | 8 | 154 | 165 | 0.925 | 0.883 | 22 | 0.07 |
| combined seed 0 | hdbscan_20 | 2 | 0 | 140 | 165 | 0.641 | 0.641 | 3 | 0 |
| learned seed 1 | walls | 1 | 0 | 99 | 165 | 0 | 0 | 1 | 0 |
| learned seed 1 | mixture | 10 | 0 | 150 | 165 | 0.25 | 0.25 | 26 | 0 |
| learned seed 1 | kmeans | 10 | 0 | 148 | 165 | 0.258 | 0.258 | 24 | 0 |
| learned seed 1 | hdbscan_10 | 4 | 59 | 100 | 165 | 0.803 | 0.408 | 15 | 0.361 |
| learned seed 1 | hdbscan_20 | 2 | 64 | 97 | 165 | 0.913 | 0.447 | 7 | 0.359 |
| combined seed 1 | walls | 2 | 0 | 141 | 165 | 0.672 | 0.672 | 2 | 0 |
| combined seed 1 | mixture | 10 | 0 | 155 | 165 | 0.373 | 0.373 | 36 | 0 |
| combined seed 1 | kmeans | 2 | 0 | 140 | 165 | 0.641 | 0.641 | 3 | 0 |
| combined seed 1 | hdbscan_10 | 9 | 42 | 121 | 165 | 0.84 | 0.611 | 24 | 0.261 |
| combined seed 1 | hdbscan_20 | 2 | 0 | 140 | 165 | 0.641 | 0.641 | 3 | 0 |

Probe (logistic regression of the reference labels, held-out points, balanced accuracy): hand 0.977, learned seed 0 0.913, combined seed 0 0.997, learned seed 1 0.896, combined seed 1 0.997

## Trainings (last epoch)

| name | loss | collapse_loss | alignment | uniformity | effective_rank_h |
|---|---|---|---|---|---|
| learned seed 0 | 2.2 | 5.54 | 0.996 | -2.27 | 43.3 |
| learned seed 1 | 2.23 | 5.54 | 0.994 | -2.24 | 40.4 |
