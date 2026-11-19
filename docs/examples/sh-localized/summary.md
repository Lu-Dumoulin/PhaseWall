# sh-localized

420 runs on 105 parameter points (a, b); analysis 1 min.

Reference labels (runs): uniform 204, extended 167, localized 49

## Steady state

24 runs (14 at points whose seeds end in different states) rerun to 2x, 4x the run time: label changed in 0 runs at 2x, 0 runs at 4x; largest change of the field between two snapshots at the longest time: 3.6e-05.

| run | a | b | seed | at_boundary | label | label_x2 | change_x2 | label_x4 | change_x4 |
|---|---|---|---|---|---|---|---|---|---|
| 249 | -0.16 | 2 | 1 | True | extended | extended | 0 | extended | 0 |
| 204 | -0.165 | 1.9 | 0 | True | localized | localized | 0 | localized | 0 |
| 246 | -0.16 | 1.9 | 2 | True | localized | localized | 0 | localized | 0 |
| 250 | -0.16 | 2 | 2 | True | extended | extended | 3e-05 | extended | 0 |
| 252 | -0.16 | 2.1 | 0 | True | localized | localized | 1e-05 | localized | 2.7e-05 |
| 257 | -0.16 | 2.2 | 1 | True | extended | extended | 0 | extended | 0 |
| 254 | -0.16 | 2.1 | 2 | True | localized | localized | 0 | localized | 0 |
| 260 | -0.158 | 1.8 | 0 | True | localized | localized | 7e-06 | localized | 3.6e-05 |
| 207 | -0.165 | 1.9 | 3 | True | uniform | uniform | 0 | uniform | 0 |
| 230 | -0.163 | 2 | 2 | True | localized | localized | 0 | localized | 0 |
| 231 | -0.163 | 2 | 3 | True | localized | localized | 0 | localized | 0 |
| 251 | -0.16 | 2 | 3 | True | localized | localized | 0 | localized | 0 |
| 349 | -0.147 | 2 | 1 | False | extended | extended | 0 | extended | 0 |
| 6 | -0.19 | 1.9 | 2 | False | uniform | uniform | 0 | uniform | 0 |
| 203 | -0.165 | 1.8 | 3 | True | uniform | uniform | 0 | uniform | 0 |
| 265 | -0.158 | 1.9 | 1 | True | extended | extended | 0.00129 | extended | 0 |
| 123 | -0.175 | 1.8 | 3 | False | uniform | uniform | 0 | uniform | 0 |
| 70 | -0.182 | 2 | 2 | False | uniform | uniform | 0 | uniform | 0 |
| 107 | -0.177 | 1.9 | 3 | False | uniform | uniform | 0 | uniform | 0 |
| 16 | -0.19 | 2.2 | 0 | False | uniform | uniform | 0 | uniform | 0 |
| 276 | -0.158 | 2.2 | 0 | False | extended | extended | 0 | extended | 0 |
| 342 | -0.147 | 1.8 | 2 | False | extended | extended | 0 | extended | 0 |
| 384 | -0.142 | 1.9 | 0 | False | extended | extended | 1.7e-05 | extended | 4e-06 |
| 30 | -0.188 | 2 | 2 | False | uniform | uniform | 0 | uniform | 0 |

## Scores

Each row: a description and a method; agree = points carrying the most common reference label of their group, ari = adjusted Rand index with the reference, pieces = connected pieces on the grid.

| describer | method | groups | noise_points | agree | points | ari | ari_all | pieces | noise_runs |
|---|---|---|---|---|---|---|---|---|---|
| hand | walls | 3 | 0 | 103 | 105 | 0.96 | 0.96 | 3 | 0 |
| hand | mixture | 7 | 0 | 105 | 105 | 0.799 | 0.799 | 16 | 0 |
| hand | kmeans | 6 | 0 | 105 | 105 | 0.983 | 0.983 | 7 | 0 |
| hand | hdbscan_10 | 3 | 11 | 94 | 105 | 1 | 0.946 | 3 | 0.126 |
| hand | hdbscan_20 | 2 | 3 | 91 | 105 | 0.829 | 0.824 | 2 | 0.076 |
| learned seed 0 | walls | 3 | 0 | 105 | 105 | 1 | 1 | 3 | 0 |
| learned seed 0 | mixture | 6 | 0 | 105 | 105 | 0.868 | 0.868 | 9 | 0 |
| learned seed 0 | kmeans | 3 | 0 | 105 | 105 | 1 | 1 | 3 | 0 |
| learned seed 0 | hdbscan_10 | 4 | 0 | 103 | 105 | 0.819 | 0.819 | 5 | 0.036 |
| learned seed 0 | hdbscan_20 | 4 | 4 | 100 | 105 | 0.857 | 0.809 | 6 | 0.067 |
| combined seed 0 | walls | 3 | 0 | 103 | 105 | 0.96 | 0.96 | 3 | 0 |
| combined seed 0 | mixture | 6 | 0 | 105 | 105 | 0.79 | 0.79 | 18 | 0 |
| combined seed 0 | kmeans | 3 | 0 | 105 | 105 | 1 | 1 | 3 | 0 |
| combined seed 0 | hdbscan_10 | 3 | 17 | 88 | 105 | 1 | 0.826 | 5 | 0.19 |
| combined seed 0 | hdbscan_20 | 3 | 6 | 99 | 105 | 1 | 0.966 | 3 | 0.102 |
| learned seed 1 | walls | 3 | 0 | 105 | 105 | 1 | 1 | 3 | 0 |
| learned seed 1 | mixture | 5 | 0 | 104 | 105 | 0.85 | 0.85 | 6 | 0 |
| learned seed 1 | kmeans | 3 | 0 | 105 | 105 | 1 | 1 | 3 | 0 |
| learned seed 1 | hdbscan_10 | 5 | 0 | 104 | 105 | 0.911 | 0.911 | 6 | 0.048 |
| learned seed 1 | hdbscan_20 | 4 | 0 | 105 | 105 | 0.971 | 0.971 | 4 | 0.012 |
| combined seed 1 | walls | 3 | 0 | 103 | 105 | 0.96 | 0.96 | 3 | 0 |
| combined seed 1 | mixture | 6 | 0 | 104 | 105 | 0.766 | 0.766 | 20 | 0 |
| combined seed 1 | kmeans | 5 | 0 | 105 | 105 | 0.977 | 0.977 | 7 | 0 |
| combined seed 1 | hdbscan_10 | 3 | 17 | 88 | 105 | 1 | 0.826 | 5 | 0.181 |
| combined seed 1 | hdbscan_20 | 3 | 6 | 99 | 105 | 1 | 0.966 | 3 | 0.1 |

Probe (logistic regression of the reference labels, held-out points, balanced accuracy): hand 1.000, learned seed 0 1.000, combined seed 0 1.000, learned seed 1 1.000, combined seed 1 1.000

## Trainings (last epoch)

| name | loss | collapse_loss | alignment | uniformity | effective_rank_h |
|---|---|---|---|---|---|
| learned seed 0 | 4.54 | 5.54 | 0.993 | -0.907 | 4.71 |
| learned seed 1 | 4.55 | 5.54 | 0.992 | -0.911 | 4.48 |
