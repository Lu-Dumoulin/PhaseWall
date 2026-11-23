# barkley

660 runs on 165 parameter points (a, b); analysis 1 min.

Reference labels (runs): spirals 325, rest 198, plane waves 137

## Steady state

24 runs (16 at points whose seeds end in different states) rerun to 2x, 4x the run time: label changed in 0 runs at 2x, 0 runs at 4x; largest change of the field between two snapshots at the longest time: 1.9e+00.

| run | a | b | seed | at_boundary | label | label_x2 | change_x2 | label_x4 | change_x4 |
|---|---|---|---|---|---|---|---|---|---|
| 52 | 0.35 | 0.02 | 0 | True | plane waves | plane waves | 1.06 | plane waves | 1.06 |
| 424 | 0.75 | 0.07 | 0 | True | spirals | spirals | 1.36 | spirals | 1.35 |
| 145 | 0.45 | 0.03 | 1 | True | plane waves | plane waves | 1.18 | plane waves | 1.18 |
| 45 | 0.35 | 0 | 1 | True | spirals | spirals | 1.14 | spirals | 1.14 |
| 627 | 1 | 0.02 | 3 | True | spirals | spirals | 1.92 | spirals | 1.92 |
| 427 | 0.75 | 0.07 | 3 | True | spirals | spirals | 1.41 | spirals | 1.41 |
| 488 | 0.85 | 0.01 | 0 | True | spirals | spirals | 1.74 | spirals | 1.71 |
| 494 | 0.85 | 0.02 | 2 | True | spirals | spirals | 1.71 | spirals | 1.67 |
| 434 | 0.75 | 0.09 | 2 | True | plane waves | plane waves | 1.34 | plane waves | 1.34 |
| 574 | 0.95 | 0 | 2 | True | spirals | spirals | 1.85 | spirals | 1.85 |
| 233 | 0.55 | 0.03 | 1 | True | plane waves | plane waves | 1.31 | plane waves | 1.31 |
| 576 | 0.95 | 0.01 | 0 | True | spirals | spirals | 1.85 | spirals | 1.86 |
| 550 | 0.9 | 0.05 | 2 | False | spirals | spirals | 1.68 | spirals | 1.67 |
| 10 | 0.3 | 0.02 | 2 | False | rest | rest | 0 | rest | 0 |
| 330 | 0.65 | 0.05 | 2 | True | plane waves | plane waves | 1.38 | plane waves | 1.38 |
| 410 | 0.75 | 0.03 | 2 | False | spirals | spirals | 1.51 | spirals | 1.51 |
| 200 | 0.5 | 0.06 | 0 | False | rest | rest | 0 | rest | 0 |
| 115 | 0.4 | 0.06 | 3 | False | rest | rest | 0 | rest | 0 |
| 175 | 0.45 | 0.1 | 3 | False | rest | rest | 0 | rest | 0 |
| 26 | 0.3 | 0.06 | 2 | False | rest | rest | 0 | rest | 0 |
| 425 | 0.75 | 0.07 | 1 | True | spirals | spirals | 1.36 | spirals | 1.34 |
| 534 | 0.9 | 0.01 | 2 | True | spirals | spirals | 1.79 | spirals | 1.77 |
| 602 | 0.95 | 0.07 | 2 | False | spirals | spirals | 1.68 | spirals | 1.67 |
| 49 | 0.35 | 0.01 | 1 | True | plane waves | plane waves | 1.16 | plane waves | 1.16 |

## Scores

Each row: a description and a method; agree = points carrying the most common reference label of their group, ari = adjusted Rand index with the reference, pieces = connected pieces on the grid.

| describer | method | groups | noise_points | agree | points | ari | ari_all | pieces | noise_runs |
|---|---|---|---|---|---|---|---|---|---|
| hand | walls | 2 | 0 | 132 | 165 | 0.582 | 0.582 | 2 | 0 |
| hand | mixture | 10 | 0 | 152 | 165 | 0.462 | 0.462 | 43 | 0 |
| hand | kmeans | 2 | 0 | 131 | 165 | 0.566 | 0.566 | 3 | 0 |
| hand | hdbscan_10 | 10 | 14 | 145 | 165 | 0.747 | 0.635 | 35 | 0.076 |
| hand | hdbscan_20 | 5 | 24 | 134 | 165 | 0.84 | 0.734 | 20 | 0.136 |
| learned seed 0 | walls | 1 | 0 | 90 | 165 | 0 | 0 | 1 | 0 |
| learned seed 0 | mixture | 10 | 0 | 148 | 165 | 0.296 | 0.296 | 21 | 0 |
| learned seed 0 | kmeans | 10 | 0 | 150 | 165 | 0.312 | 0.312 | 21 | 0 |
| learned seed 0 | hdbscan_10 | 3 | 24 | 108 | 165 | 0.411 | 0.511 | 3 | 0.15 |
| learned seed 0 | hdbscan_20 | 2 | 1 | 131 | 165 | 0.581 | 0.572 | 2 | 0.005 |
| combined seed 0 | walls | 2 | 0 | 132 | 165 | 0.582 | 0.582 | 2 | 0 |
| combined seed 0 | mixture | 10 | 0 | 158 | 165 | 0.407 | 0.407 | 35 | 0 |
| combined seed 0 | kmeans | 2 | 0 | 131 | 165 | 0.566 | 0.566 | 3 | 0 |
| combined seed 0 | hdbscan_10 | 8 | 10 | 148 | 165 | 0.811 | 0.727 | 32 | 0.052 |
| combined seed 0 | hdbscan_20 | 4 | 36 | 125 | 165 | 0.883 | 0.76 | 11 | 0.208 |
| learned seed 1 | walls | 1 | 0 | 90 | 165 | 0 | 0 | 1 | 0 |
| learned seed 1 | mixture | 10 | 0 | 143 | 165 | 0.281 | 0.281 | 29 | 0 |
| learned seed 1 | kmeans | 9 | 0 | 145 | 165 | 0.293 | 0.293 | 26 | 0 |
| learned seed 1 | hdbscan_10 | 3 | 24 | 108 | 165 | 0.399 | 0.502 | 4 | 0.158 |
| learned seed 1 | hdbscan_20 | 3 | 37 | 95 | 165 | 0.39 | 0.355 | 4 | 0.224 |
| combined seed 1 | walls | 2 | 0 | 132 | 165 | 0.582 | 0.582 | 2 | 0 |
| combined seed 1 | mixture | 10 | 0 | 158 | 165 | 0.456 | 0.456 | 34 | 0 |
| combined seed 1 | kmeans | 2 | 0 | 131 | 165 | 0.566 | 0.566 | 3 | 0 |
| combined seed 1 | hdbscan_10 | 8 | 8 | 151 | 165 | 0.824 | 0.754 | 33 | 0.047 |
| combined seed 1 | hdbscan_20 | 4 | 34 | 127 | 165 | 0.884 | 0.755 | 13 | 0.203 |

Probe (logistic regression of the reference labels, held-out points, balanced accuracy): hand 0.986, learned seed 0 0.948, combined seed 0 0.993, learned seed 1 0.947, combined seed 1 0.994

## Trainings (last epoch)

| name | loss | collapse_loss | alignment | uniformity | effective_rank_h |
|---|---|---|---|---|---|
| learned seed 0 | 2.34 | 5.54 | 0.982 | -2.25 | 35.3 |
| learned seed 1 | 2.33 | 5.54 | 0.985 | -2.23 | 37.8 |
