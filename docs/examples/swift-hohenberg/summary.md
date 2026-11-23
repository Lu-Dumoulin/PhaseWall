# swift-hohenberg

1920 runs on 240 parameter points (eps, g); analysis 2 min.

Reference labels (runs): stripes 1131, hexagons 549, uniform 240

## Steady state

24 runs (12 at points whose seeds end in different states) rerun to 2x, 4x the run time: label changed in 0 runs at 2x, 0 runs at 4x; largest change of the field between two snapshots at the longest time: 1.9e-09.

| run | eps | g | seed | at_boundary | label | label_x2 | change_x2 | label_x4 | change_x4 |
|---|---|---|---|---|---|---|---|---|---|
| 255 | 0.025 | 0.1 | 7 | True | hexagons | hexagons | 0 | hexagons | -0 |
| 1041 | 0.325 | 1 | 1 | True | stripes | stripes | 0 | stripes | 0 |
| 1049 | 0.325 | 1.1 | 1 | True | stripes | stripes | 0 | stripes | 0 |
| 634 | 0.175 | 0.4 | 2 | True | stripes | stripes | 0 | stripes | 0 |
| 1062 | 0.325 | 1.2 | 6 | True | hexagons | hexagons | 0 | hexagons | 0 |
| 1176 | 0.375 | 1.2 | 0 | True | stripes | stripes | 7e-06 | stripes | 0 |
| 947 | 0.275 | 1.3 | 3 | True | hexagons | hexagons | 0 | hexagons | 0 |
| 510 | 0.125 | 0.3 | 6 | True | stripes | stripes | 0 | stripes | 0 |
| 1177 | 0.375 | 1.2 | 1 | True | stripes | stripes | 0 | stripes | 0 |
| 504 | 0.125 | 0.3 | 0 | True | stripes | stripes | 0 | stripes | 0 |
| 1078 | 0.325 | 1.4 | 6 | True | stripes | stripes | 0 | stripes | 0 |
| 1072 | 0.325 | 1.4 | 0 | True | stripes | stripes | 0 | stripes | 0 |
| 1625 | 0.575 | 0.8 | 1 | False | stripes | stripes | 0 | stripes | 0 |
| 31 | -0.075 | 0.3 | 7 | False | uniform | uniform | 0 | uniform | 0 |
| 975 | 0.325 | 0.1 | 7 | False | stripes | stripes | 0 | stripes | 0 |
| 1220 | 0.425 | 0.2 | 4 | False | stripes | stripes | 0 | stripes | 0 |
| 588 | 0.125 | 1.3 | 4 | False | hexagons | hexagons | 0 | hexagons | 0 |
| 334 | 0.025 | 1.1 | 6 | False | hexagons | hexagons | 0 | hexagons | 0 |
| 515 | 0.125 | 0.4 | 3 | False | hexagons | hexagons | 0 | hexagons | 0 |
| 77 | -0.075 | 0.9 | 5 | False | uniform | uniform | 0 | uniform | 0 |
| 1250 | 0.425 | 0.6 | 2 | False | stripes | stripes | 0 | stripes | 0 |
| 1562 | 0.575 | 0 | 2 | False | stripes | stripes | 0 | stripes | 0 |
| 1753 | 0.625 | 0.9 | 1 | False | stripes | stripes | 0 | stripes | 0 |
| 143 | -0.025 | 0.2 | 7 | False | uniform | uniform | 1e-06 | uniform | 0 |

## Scores

Each row: a description and a method; agree = points carrying the most common reference label of their group, ari = adjusted Rand index with the reference, pieces = connected pieces on the grid.

| describer | method | groups | noise_points | agree | points | ari | ari_all | pieces | noise_runs |
|---|---|---|---|---|---|---|---|---|---|
| hand | walls | 3 | 0 | 236 | 240 | 0.942 | 0.942 | 3 | 0 |
| hand | mixture | 10 | 0 | 230 | 240 | 0.382 | 0.382 | 14 | 0 |
| hand | kmeans | 5 | 0 | 235 | 240 | 0.583 | 0.583 | 5 | 0 |
| hand | hdbscan_10 | 7 | 22 | 216 | 240 | 0.772 | 0.659 | 11 | 0.124 |
| hand | hdbscan_20 | 6 | 34 | 205 | 240 | 0.778 | 0.591 | 10 | 0.199 |
| learned seed 0 | walls | 3 | 0 | 237 | 240 | 0.956 | 0.956 | 3 | 0 |
| learned seed 0 | mixture | 10 | 0 | 235 | 240 | 0.304 | 0.304 | 16 | 0 |
| learned seed 0 | kmeans | 2 | 0 | 203 | 240 | 0.602 | 0.602 | 2 | 0 |
| learned seed 0 | hdbscan_10 | 6 | 133 | 106 | 240 | 0.774 | 0.304 | 13 | 0.491 |
| learned seed 0 | hdbscan_20 | 4 | 123 | 117 | 240 | 0.897 | 0.514 | 7 | 0.443 |
| combined seed 0 | walls | 3 | 0 | 238 | 240 | 0.971 | 0.971 | 3 | 0 |
| combined seed 0 | mixture | 9 | 0 | 230 | 240 | 0.547 | 0.547 | 20 | 0 |
| combined seed 0 | kmeans | 5 | 0 | 235 | 240 | 0.577 | 0.577 | 5 | 0 |
| combined seed 0 | hdbscan_10 | 10 | 56 | 182 | 240 | 0.695 | 0.427 | 23 | 0.289 |
| combined seed 0 | hdbscan_20 | 6 | 67 | 172 | 240 | 0.73 | 0.398 | 10 | 0.305 |
| learned seed 1 | walls | 3 | 0 | 237 | 240 | 0.956 | 0.956 | 3 | 0 |
| learned seed 1 | mixture | 10 | 0 | 236 | 240 | 0.263 | 0.263 | 16 | 0 |
| learned seed 1 | kmeans | 8 | 0 | 234 | 240 | 0.33 | 0.33 | 14 | 0 |
| learned seed 1 | hdbscan_10 | 7 | 62 | 175 | 240 | 0.746 | 0.483 | 9 | 0.271 |
| learned seed 1 | hdbscan_20 | 5 | 102 | 137 | 240 | 0.82 | 0.477 | 6 | 0.396 |
| combined seed 1 | walls | 3 | 0 | 236 | 240 | 0.942 | 0.942 | 3 | 0 |
| combined seed 1 | mixture | 10 | 0 | 228 | 240 | 0.341 | 0.341 | 20 | 0 |
| combined seed 1 | kmeans | 5 | 0 | 235 | 240 | 0.572 | 0.572 | 5 | 0 |
| combined seed 1 | hdbscan_10 | 8 | 40 | 198 | 240 | 0.719 | 0.575 | 15 | 0.205 |
| combined seed 1 | hdbscan_20 | 7 | 60 | 179 | 240 | 0.721 | 0.41 | 12 | 0.301 |

Probe (logistic regression of the reference labels, held-out points, balanced accuracy): hand 0.999, learned seed 0 0.998, combined seed 0 0.999, learned seed 1 0.997, combined seed 1 0.999

## Trainings (last epoch)

| name | loss | collapse_loss | alignment | uniformity | effective_rank_h |
|---|---|---|---|---|---|
| learned seed 0 | 2.97 | 5.54 | 0.895 | -2.75 | 26.5 |
| learned seed 1 | 3.06 | 5.54 | 0.883 | -2.71 | 28.7 |
