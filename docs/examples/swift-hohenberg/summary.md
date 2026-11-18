# swift-hohenberg

1920 runs on 240 parameter points (eps, g); analysis 1 min.

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
| hand | walls | 3 | 0 | 235 | 240 | 0.928 | 0.928 | 3 | 0 |
| hand | mixture | 10 | 0 | 230 | 240 | 0.382 | 0.382 | 14 | 0 |
| hand | kmeans | 5 | 0 | 235 | 240 | 0.583 | 0.583 | 5 | 0 |
| hand | hdbscan_10 | 7 | 22 | 216 | 240 | 0.772 | 0.659 | 11 | 0.124 |
| hand | hdbscan_20 | 6 | 34 | 205 | 240 | 0.778 | 0.591 | 10 | 0.199 |
| learned seed 0 | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 |
| learned seed 0 | mixture | 10 | 0 | 191 | 240 | 0.138 | 0.138 | 18 | 0 |
| learned seed 0 | kmeans | 10 | 0 | 182 | 240 | 0.113 | 0.113 | 24 | 0 |
| learned seed 0 | hdbscan_10 | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0.015 |
| learned seed 0 | hdbscan_20 | 2 | 8 | 164 | 240 | 0.366 | 0.329 | 2 | 0.071 |
| combined seed 0 | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 |
| combined seed 0 | mixture | 8 | 0 | 227 | 240 | 0.292 | 0.292 | 12 | 0 |
| combined seed 0 | kmeans | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 |
| combined seed 0 | hdbscan_10 | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0.006 |
| combined seed 0 | hdbscan_20 | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0.007 |
| learned seed 1 | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 |
| learned seed 1 | mixture | 10 | 0 | 197 | 240 | 0.137 | 0.137 | 21 | 0 |
| learned seed 1 | kmeans | 10 | 0 | 199 | 240 | 0.144 | 0.144 | 26 | 0 |
| learned seed 1 | hdbscan_10 | 2 | 1 | 170 | 240 | 0.363 | 0.358 | 2 | 0.03 |
| learned seed 1 | hdbscan_20 | 2 | 19 | 154 | 240 | 0.368 | 0.288 | 2 | 0.122 |
| combined seed 1 | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 |
| combined seed 1 | mixture | 8 | 0 | 232 | 240 | 0.299 | 0.299 | 9 | 0 |
| combined seed 1 | kmeans | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 |
| combined seed 1 | hdbscan_10 | 2 | 12 | 159 | 240 | 0.271 | 0.348 | 2 | 0.056 |
| combined seed 1 | hdbscan_20 | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0.007 |

Probe (logistic regression of the reference labels, held-out points, balanced accuracy): hand 0.999, learned seed 0 0.981, combined seed 0 0.999, learned seed 1 0.979, combined seed 1 0.999

## Trainings (last epoch)

| name | loss | collapse_loss | alignment | uniformity | effective_rank_h |
|---|---|---|---|---|---|
| learned seed 0 | 1.46 | 5.54 | 0.999 | -3.35 | 84.9 |
| learned seed 1 | 1.46 | 5.54 | 0.999 | -3.35 | 81.6 |
