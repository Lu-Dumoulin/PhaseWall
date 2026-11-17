# Hand-made against learned description (Swift-Hohenberg demo)

Walls with the default settings (RATIO = 1.5). Agreement: points that carry the most common reference label of their region. AUC: probability that a link between two reference states jumps more than a link inside one state. NN accuracy: label of the nearest other point in the description (balanced over the three states). jump_*: median jump of the links inside one state, between hexagons and stripes, and between the uniform state and a pattern, divided by the median of all jumps. within_points: share of the variance of the description within parameter points. probe_hex_stripes: ridge probe hexagons against stripes on held-out parameter points (balanced accuracy).

| setting | describer | columns | phases | agree | points | walls | mixed | ratio_range | auc_boundary | nn_accuracy | effective_rank | jump_inner | jump_hex_stripes | jump_uniform_pattern | within_points | probe_hex_stripes | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| periodic box | hand | 28 | 3 | 235 | 240 | 34 | 0 | 1.2-2 | 0.981 | 0.974 | 13.2 | 0.972 | 1.92 | 4.41 | 0.298 | 0.999 | 4.28 |
| periodic box | cnn untrained | 155 | 2 | 171 | 240 | 15 | 0 | 1.2-2 | 0.8 | 0.976 | 68.2 | 0.988 | 1.18 | 1.91 | 0.376 | 0.948 | 0 |
| periodic box | cnn seed 0 | 256 | 2 | 171 | 240 | 15 | 1 | 1.3-2.5 | 0.92 | 0.954 | 85.8 | 0.964 | 1.37 | 2.22 | 0.288 | 0.977 | 168 |
| periodic box | cnn seed 1 | 256 | 2 | 171 | 240 | 15 | 1 | 1.2-2.5 | 0.893 | 0.966 | 82.6 | 0.969 | 1.3 | 2.13 | 0.27 | 0.977 | 264 |
| crop (not periodic) | hand | 24 | 3 | 235 | 240 | 35 | 0 | 1.2-1.5 | 0.989 | 0.986 | 11.6 | 0.971 | 2.04 | 5 | 0.267 | 0.992 | 10.5 |
| crop (not periodic) | cnn untrained | 177 | 2 | 159 | 240 | 11 | 0 | 1.2-1.75 | 0.71 | 0.981 | 108 | 0.995 | 1.06 | 1.52 | 0.566 | 0.935 | 0 |
| crop (not periodic) | cnn seed 0 | 256 | 2 | 171 | 240 | 15 | 0 | 1.3-2.5 | 0.883 | 0.967 | 86.6 | 0.994 | 1.17 | 2.25 | 0.515 | 0.961 | 83.8 |
| crop (not periodic) | cnn seed 1 | 256 | 2 | 171 | 240 | 15 | 0 | 1.2-2.5 | 0.815 | 0.931 | 90.5 | 0.993 | 1.08 | 2.27 | 0.479 | 0.942 | 104 |

Last epoch of each training:

| setting | seed | loss | collapse_loss | alignment | uniformity | effective_rank_h |
|---|---|---|---|---|---|---|
| periodic box | 0 | 1.46 | 5.54 | 0.999 | -3.35 | 84.9 |
| periodic box | 1 | 1.46 | 5.54 | 0.999 | -3.35 | 81.6 |
| crop (not periodic) | 0 | 1.45 | 5.54 | 0.999 | -3.36 | 85.7 |
| crop (not periodic) | 1 | 1.45 | 5.54 | 0.999 | -3.36 | 89.5 |

## Walls against blind grouping

groups: groups found (majority group of each point); agree: points that carry the most common reference label of their group; ari: adjusted Rand index between groups and reference labels of the points; pieces: connected pieces of the groups on the parameter grid (3 for the three reference states); noise_runs: share of runs that HDBSCAN cannot place; k_chosen: k of the mixture (BIC) and of k-means (silhouette); probe_3_states: logistic regression of the three states on held-out points (supervised, balanced accuracy).

| setting | describer | method | groups | noise_points | agree | points | ari | ari_all | pieces | noise_runs | k_chosen | components | probe_3_states |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| periodic box | hand | walls | 3 | 0 | 235 | 240 | 0.928 | 0.928 | 3 | 0 | nan | 7 | 0.999 |
| periodic box | hand | mixture | 8 | 0 | 230 | 240 | 0.384 | 0.384 | 13 | 0 | 10 | 7 | 0.999 |
| periodic box | hand | kmeans | 5 | 0 | 235 | 240 | 0.583 | 0.583 | 5 | 0 | 7 | 7 | 0.999 |
| periodic box | hand | hdbscan_5 | 37 | 42 | 196 | 240 | 0.752 | 0.606 | 50 | 0.198 | nan | 7 | 0.999 |
| periodic box | hand | hdbscan_10 | 7 | 22 | 216 | 240 | 0.772 | 0.659 | 11 | 0.124 | nan | 7 | 0.999 |
| periodic box | hand | hdbscan_20 | 6 | 34 | 205 | 240 | 0.778 | 0.591 | 10 | 0.199 | nan | 7 | 0.999 |
| periodic box | hand | hdbscan_40 | 4 | 20 | 218 | 240 | 0.932 | 0.837 | 5 | 0.108 | nan | 7 | 0.999 |
| periodic box | cnn untrained | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 | nan | 33 | 0.958 |
| periodic box | cnn untrained | mixture | 7 | 0 | 189 | 240 | 0.163 | 0.163 | 19 | 0 | 8 | 33 | 0.958 |
| periodic box | cnn untrained | kmeans | 2 | 0 | 141 | 240 | 0.047 | 0.047 | 2 | 0 | 2 | 33 | 0.958 |
| periodic box | cnn untrained | hdbscan_5 | 4 | 200 | 40 | 240 | 0.905 | 0.296 | 6 | 0.619 | nan | 33 | 0.958 |
| periodic box | cnn untrained | hdbscan_10 | 2 | 199 | 41 | 240 | 1 | 0.413 | 2 | 0.726 | nan | 33 | 0.958 |
| periodic box | cnn untrained | hdbscan_20 | 2 | 200 | 40 | 240 | 1 | 0.408 | 2 | 0.784 | nan | 33 | 0.958 |
| periodic box | cnn untrained | hdbscan_40 | 2 | 203 | 36 | 240 | 0.871 | 0.391 | 2 | 0.827 | nan | 33 | 0.958 |
| periodic box | cnn seed 0 | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 | nan | 30 | 0.981 |
| periodic box | cnn seed 0 | mixture | 10 | 0 | 191 | 240 | 0.138 | 0.138 | 18 | 0 | 10 | 30 | 0.981 |
| periodic box | cnn seed 0 | kmeans | 10 | 0 | 182 | 240 | 0.113 | 0.113 | 24 | 0 | 10 | 30 | 0.981 |
| periodic box | cnn seed 0 | hdbscan_5 | 38 | 160 | 79 | 240 | 0.491 | 0.384 | 39 | 0.58 | nan | 30 | 0.981 |
| periodic box | cnn seed 0 | hdbscan_10 | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0.015 | nan | 30 | 0.981 |
| periodic box | cnn seed 0 | hdbscan_20 | 2 | 8 | 164 | 240 | 0.366 | 0.329 | 2 | 0.071 | nan | 30 | 0.981 |
| periodic box | cnn seed 0 | hdbscan_40 | 2 | 52 | 130 | 240 | 0.396 | 0.233 | 2 | 0.26 | nan | 30 | 0.981 |
| periodic box | cnn seed 1 | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 | nan | 29 | 0.979 |
| periodic box | cnn seed 1 | mixture | 10 | 0 | 197 | 240 | 0.137 | 0.137 | 21 | 0 | 10 | 29 | 0.979 |
| periodic box | cnn seed 1 | kmeans | 10 | 0 | 199 | 240 | 0.144 | 0.144 | 26 | 0 | 10 | 29 | 0.979 |
| periodic box | cnn seed 1 | hdbscan_5 | 42 | 157 | 83 | 240 | 0.472 | 0.364 | 46 | 0.573 | nan | 29 | 0.979 |
| periodic box | cnn seed 1 | hdbscan_10 | 2 | 1 | 170 | 240 | 0.363 | 0.358 | 2 | 0.03 | nan | 29 | 0.979 |
| periodic box | cnn seed 1 | hdbscan_20 | 2 | 19 | 154 | 240 | 0.368 | 0.288 | 2 | 0.122 | nan | 29 | 0.979 |
| periodic box | cnn seed 1 | hdbscan_40 | 2 | 94 | 105 | 240 | 0.463 | 0.213 | 2 | 0.438 | nan | 29 | 0.979 |
| crop (not periodic) | hand | walls | 3 | 0 | 235 | 240 | 0.928 | 0.928 | 3 | 0 | nan | 6 | 0.995 |
| crop (not periodic) | hand | mixture | 9 | 0 | 238 | 240 | 0.482 | 0.482 | 15 | 0 | 10 | 6 | 0.995 |
| crop (not periodic) | hand | kmeans | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 | 2 | 6 | 0.995 |
| crop (not periodic) | hand | hdbscan_5 | 3 | 1 | 236 | 240 | 0.956 | 0.946 | 3 | 0.037 | nan | 6 | 0.995 |
| crop (not periodic) | hand | hdbscan_10 | 3 | 1 | 237 | 240 | 0.97 | 0.961 | 3 | 0.035 | nan | 6 | 0.995 |
| crop (not periodic) | hand | hdbscan_20 | 3 | 1 | 237 | 240 | 0.97 | 0.961 | 3 | 0.037 | nan | 6 | 0.995 |
| crop (not periodic) | hand | hdbscan_40 | 3 | 2 | 237 | 240 | 0.985 | 0.966 | 3 | 0.041 | nan | 6 | 0.995 |
| crop (not periodic) | cnn untrained | walls | 2 | 0 | 159 | 240 | 0.199 | 0.199 | 2 | 0 | nan | 64 | 0.954 |
| crop (not periodic) | cnn untrained | mixture | 5 | 0 | 183 | 240 | 0.197 | 0.197 | 10 | 0 | 6 | 64 | 0.954 |
| crop (not periodic) | cnn untrained | kmeans | 2 | 0 | 143 | 240 | 0.156 | 0.156 | 2 | 0 | 2 | 64 | 0.954 |
| crop (not periodic) | cnn untrained | hdbscan_5 | 7 | 209 | 31 | 240 | 0.506 | 0.24 | 9 | 0.528 | nan | 64 | 0.954 |
| crop (not periodic) | cnn untrained | hdbscan_10 | 2 | 226 | 14 | 240 | 0 | 0.149 | 3 | 0.759 | nan | 64 | 0.954 |
| crop (not periodic) | cnn untrained | hdbscan_20 | 1 | 208 | 30 | 240 | 0 | 0.368 | 1 | 0.774 | nan | 64 | 0.954 |
| crop (not periodic) | cnn untrained | hdbscan_40 | 0 | 240 | 0 | 240 | nan | 0 | 0 | 1 | nan | 64 | 0.954 |
| crop (not periodic) | cnn seed 0 | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 | nan | 30 | 0.962 |
| crop (not periodic) | cnn seed 0 | mixture | 10 | 0 | 192 | 240 | 0.134 | 0.134 | 38 | 0 | 10 | 30 | 0.962 |
| crop (not periodic) | cnn seed 0 | kmeans | 10 | 0 | 192 | 240 | 0.147 | 0.147 | 35 | 0 | 10 | 30 | 0.962 |
| crop (not periodic) | cnn seed 0 | hdbscan_5 | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0.007 | nan | 30 | 0.962 |
| crop (not periodic) | cnn seed 0 | hdbscan_10 | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0.035 | nan | 30 | 0.962 |
| crop (not periodic) | cnn seed 0 | hdbscan_20 | 2 | 7 | 164 | 240 | 0.363 | 0.326 | 3 | 0.095 | nan | 30 | 0.962 |
| crop (not periodic) | cnn seed 0 | hdbscan_40 | 2 | 33 | 142 | 240 | 0.374 | 0.253 | 3 | 0.254 | nan | 30 | 0.962 |
| crop (not periodic) | cnn seed 1 | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 | nan | 31 | 0.96 |
| crop (not periodic) | cnn seed 1 | mixture | 10 | 0 | 196 | 240 | 0.141 | 0.141 | 30 | 0 | 10 | 31 | 0.96 |
| crop (not periodic) | cnn seed 1 | kmeans | 10 | 0 | 185 | 240 | 0.11 | 0.11 | 36 | 0 | 10 | 31 | 0.96 |
| crop (not periodic) | cnn seed 1 | hdbscan_5 | 2 | 3 | 168 | 240 | 0.363 | 0.347 | 2 | 0.074 | nan | 31 | 0.96 |
| crop (not periodic) | cnn seed 1 | hdbscan_10 | 2 | 6 | 165 | 240 | 0.363 | 0.331 | 2 | 0.117 | nan | 31 | 0.96 |
| crop (not periodic) | cnn seed 1 | hdbscan_20 | 2 | 51 | 132 | 240 | 0.399 | 0.236 | 2 | 0.319 | nan | 31 | 0.96 |
| crop (not periodic) | cnn seed 1 | hdbscan_40 | 2 | 156 | 59 | 240 | 0.594 | 0.341 | 10 | 0.611 | nan | 31 | 0.96 |
