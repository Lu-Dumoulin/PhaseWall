# Hand-made against learned description (Swift-Hohenberg demo)

Walls with the default settings (RATIO = 1.5). Agreement: points that carry the most common reference label of their region. AUC: probability that a link between two reference states jumps more than a link inside one state. NN accuracy: label of the nearest other point in the description (balanced over the three states). jump_*: median jump of the links inside one state, between hexagons and stripes, and between the uniform state and a pattern, divided by the median of all jumps. within_points: share of the variance of the description within parameter points. probe_hex_stripes: ridge probe hexagons against stripes on held-out parameter points (balanced accuracy).

| setting | describer | columns | phases | agree | points | walls | mixed | ratio_range | auc_boundary | nn_accuracy | effective_rank | jump_inner | jump_hex_stripes | jump_uniform_pattern | within_points | probe_hex_stripes | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| periodic box | hand | 28 | 3 | 235 | 240 | 34 | 0 | 1.2-2 | 0.981 | 0.974 | 13.2 | 0.972 | 1.92 | 4.41 | 0.298 | 0.999 | 4.25 |
| periodic box | cnn untrained | 174 | 2 | 171 | 240 | 15 | 0 | 1.2-2 | 0.927 | 0.983 | 113 | 0.985 | 1.14 | 2.38 | 0.718 | 0.931 | 0 |
| periodic box | cnn seed 0 | 256 | 3 | 237 | 240 | 36 | 0 | 1.2-1.75 | 0.985 | 0.988 | 29.7 | 0.963 | 1.95 | 4.01 | 0.137 | 0.993 | 65.7 |
| periodic box | cnn seed 1 | 256 | 3 | 235 | 240 | 34 | 2 | 1.2-2 | 0.976 | 0.993 | 33 | 0.957 | 1.97 | 3.94 | 0.137 | 0.996 | 63.9 |
| crop (not periodic) | hand | 24 | 3 | 235 | 240 | 35 | 0 | 1.2-1.5 | 0.989 | 0.986 | 11.6 | 0.971 | 2.04 | 5 | 0.267 | 0.992 | 4.47 |
| crop (not periodic) | cnn untrained | 162 | 2 | 171 | 240 | 15 | 0 | 1.2-1.75 | 0.888 | 0.976 | 97.3 | 0.992 | 1.11 | 2.09 | 0.781 | 0.933 | 0 |
| crop (not periodic) | cnn seed 0 | 256 | 2 | 171 | 240 | 15 | 0 | 1.5-3 | 0.97 | 0.986 | 28.1 | 0.973 | 1.76 | 3.13 | 0.44 | 0.985 | 48.3 |
| crop (not periodic) | cnn seed 1 | 256 | 2 | 171 | 240 | 15 | 0 | 1.5-3 | 0.969 | 0.981 | 33 | 0.983 | 1.65 | 3.02 | 0.446 | 0.986 | 48.1 |

Last epoch of each training:

| setting | seed | loss | collapse_loss | alignment | uniformity | effective_rank_h |
|---|---|---|---|---|---|---|
| periodic box | 0 | 2.97 | 5.54 | 0.895 | -2.74 | 26.5 |
| periodic box | 1 | 3.06 | 5.54 | 0.883 | -2.71 | 28.7 |
| crop (not periodic) | 0 | 2.63 | 5.54 | 0.913 | -2.86 | 26 |
| crop (not periodic) | 1 | 2.49 | 5.54 | 0.916 | -2.95 | 31 |

## Walls against blind grouping

groups: groups found (majority group of each point); agree: points that carry the most common reference label of their group; ari: adjusted Rand index between groups and reference labels of the points; pieces: connected pieces of the groups on the parameter grid (3 for the three reference states); noise_runs: share of runs that HDBSCAN cannot place; k_chosen: k of the mixture (BIC) and of k-means (silhouette); probe_3_states: logistic regression of the three states on held-out points (supervised, balanced accuracy).

| setting | describer | method | groups | noise_points | agree | points | ari | ari_all | pieces | noise_runs | k_chosen | components | probe_3_states |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| periodic box | hand | walls | 3 | 0 | 236 | 240 | 0.942 | 0.942 | 3 | 0 | nan | 7 | 0.999 |
| periodic box | hand | mixture | 8 | 0 | 230 | 240 | 0.384 | 0.384 | 13 | 0 | 10 | 7 | 0.999 |
| periodic box | hand | kmeans | 5 | 0 | 235 | 240 | 0.583 | 0.583 | 5 | 0 | 7 | 7 | 0.999 |
| periodic box | hand | hdbscan_5 | 37 | 42 | 196 | 240 | 0.752 | 0.606 | 50 | 0.198 | nan | 7 | 0.999 |
| periodic box | hand | hdbscan_10 | 7 | 22 | 216 | 240 | 0.772 | 0.659 | 11 | 0.124 | nan | 7 | 0.999 |
| periodic box | hand | hdbscan_20 | 6 | 34 | 205 | 240 | 0.778 | 0.591 | 10 | 0.199 | nan | 7 | 0.999 |
| periodic box | hand | hdbscan_40 | 4 | 20 | 218 | 240 | 0.932 | 0.837 | 5 | 0.108 | nan | 7 | 0.999 |
| periodic box | cnn untrained | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 | nan | 71 | 0.959 |
| periodic box | cnn untrained | mixture | 8 | 0 | 189 | 240 | 0.246 | 0.246 | 28 | 0 | 9 | 71 | 0.959 |
| periodic box | cnn untrained | kmeans | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 | 2 | 71 | 0.959 |
| periodic box | cnn untrained | hdbscan_5 | 13 | 184 | 53 | 240 | 0.33 | 0.262 | 23 | 0.377 | nan | 71 | 0.959 |
| periodic box | cnn untrained | hdbscan_10 | 7 | 204 | 36 | 240 | 0.481 | 0.276 | 11 | 0.457 | nan | 71 | 0.959 |
| periodic box | cnn untrained | hdbscan_20 | 1 | 210 | 30 | 240 | 1 | 0.364 | 1 | 0.544 | nan | 71 | 0.959 |
| periodic box | cnn untrained | hdbscan_40 | 1 | 210 | 30 | 240 | 1 | 0.364 | 1 | 0.636 | nan | 71 | 0.959 |
| periodic box | cnn seed 0 | walls | 3 | 0 | 237 | 240 | 0.956 | 0.956 | 3 | 0 | nan | 7 | 0.998 |
| periodic box | cnn seed 0 | mixture | 10 | 0 | 235 | 240 | 0.304 | 0.304 | 16 | 0 | 10 | 7 | 0.998 |
| periodic box | cnn seed 0 | kmeans | 2 | 0 | 203 | 240 | 0.602 | 0.602 | 2 | 0 | 2 | 7 | 0.998 |
| periodic box | cnn seed 0 | hdbscan_5 | 10 | 153 | 86 | 240 | 0.833 | 0.404 | 13 | 0.471 | nan | 7 | 0.998 |
| periodic box | cnn seed 0 | hdbscan_10 | 6 | 133 | 106 | 240 | 0.774 | 0.304 | 13 | 0.491 | nan | 7 | 0.998 |
| periodic box | cnn seed 0 | hdbscan_20 | 4 | 123 | 117 | 240 | 0.897 | 0.514 | 7 | 0.443 | nan | 7 | 0.998 |
| periodic box | cnn seed 0 | hdbscan_40 | 4 | 100 | 139 | 240 | 0.872 | 0.609 | 8 | 0.367 | nan | 7 | 0.998 |
| periodic box | cnn seed 1 | walls | 3 | 0 | 237 | 240 | 0.956 | 0.956 | 3 | 0 | nan | 7 | 0.997 |
| periodic box | cnn seed 1 | mixture | 10 | 0 | 236 | 240 | 0.263 | 0.263 | 16 | 0 | 10 | 7 | 0.997 |
| periodic box | cnn seed 1 | kmeans | 8 | 0 | 234 | 240 | 0.33 | 0.33 | 14 | 0 | 8 | 7 | 0.997 |
| periodic box | cnn seed 1 | hdbscan_5 | 14 | 110 | 129 | 240 | 0.638 | 0.26 | 17 | 0.37 | nan | 7 | 0.997 |
| periodic box | cnn seed 1 | hdbscan_10 | 7 | 62 | 175 | 240 | 0.746 | 0.483 | 9 | 0.271 | nan | 7 | 0.997 |
| periodic box | cnn seed 1 | hdbscan_20 | 5 | 102 | 137 | 240 | 0.82 | 0.477 | 6 | 0.396 | nan | 7 | 0.997 |
| periodic box | cnn seed 1 | hdbscan_40 | 4 | 86 | 153 | 240 | 0.858 | 0.581 | 6 | 0.35 | nan | 7 | 0.997 |
| crop (not periodic) | hand | walls | 3 | 0 | 238 | 240 | 0.971 | 0.971 | 3 | 0 | nan | 6 | 0.995 |
| crop (not periodic) | hand | mixture | 9 | 0 | 238 | 240 | 0.482 | 0.482 | 15 | 0 | 10 | 6 | 0.995 |
| crop (not periodic) | hand | kmeans | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 | 2 | 6 | 0.995 |
| crop (not periodic) | hand | hdbscan_5 | 3 | 1 | 236 | 240 | 0.956 | 0.946 | 3 | 0.037 | nan | 6 | 0.995 |
| crop (not periodic) | hand | hdbscan_10 | 3 | 1 | 237 | 240 | 0.97 | 0.961 | 3 | 0.035 | nan | 6 | 0.995 |
| crop (not periodic) | hand | hdbscan_20 | 3 | 1 | 237 | 240 | 0.97 | 0.961 | 3 | 0.037 | nan | 6 | 0.995 |
| crop (not periodic) | hand | hdbscan_40 | 3 | 2 | 237 | 240 | 0.985 | 0.966 | 3 | 0.041 | nan | 6 | 0.995 |
| crop (not periodic) | cnn untrained | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 | nan | 57 | 0.954 |
| crop (not periodic) | cnn untrained | mixture | 8 | 0 | 204 | 240 | 0.281 | 0.281 | 63 | 0 | 10 | 57 | 0.954 |
| crop (not periodic) | cnn untrained | kmeans | 10 | 0 | 206 | 240 | 0.207 | 0.207 | 85 | 0 | 10 | 57 | 0.954 |
| crop (not periodic) | cnn untrained | hdbscan_5 | 12 | 166 | 72 | 240 | 0.621 | 0.419 | 21 | 0.329 | nan | 57 | 0.954 |
| crop (not periodic) | cnn untrained | hdbscan_10 | 10 | 190 | 50 | 240 | 0.825 | 0.409 | 11 | 0.419 | nan | 57 | 0.954 |
| crop (not periodic) | cnn untrained | hdbscan_20 | 2 | 203 | 37 | 240 | 1 | 0.326 | 7 | 0.461 | nan | 57 | 0.954 |
| crop (not periodic) | cnn untrained | hdbscan_40 | 4 | 202 | 38 | 240 | 0.963 | 0.329 | 8 | 0.544 | nan | 57 | 0.954 |
| crop (not periodic) | cnn seed 0 | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 | nan | 8 | 0.991 |
| crop (not periodic) | cnn seed 0 | mixture | 10 | 0 | 237 | 240 | 0.282 | 0.282 | 74 | 0 | 10 | 8 | 0.991 |
| crop (not periodic) | cnn seed 0 | kmeans | 8 | 0 | 237 | 240 | 0.314 | 0.314 | 46 | 0 | 8 | 8 | 0.991 |
| crop (not periodic) | cnn seed 0 | hdbscan_5 | 18 | 192 | 48 | 240 | 0.235 | 0.183 | 34 | 0.406 | nan | 8 | 0.991 |
| crop (not periodic) | cnn seed 0 | hdbscan_10 | 11 | 173 | 64 | 240 | 0.438 | 0.247 | 25 | 0.483 | nan | 8 | 0.991 |
| crop (not periodic) | cnn seed 0 | hdbscan_20 | 6 | 60 | 176 | 240 | 0.888 | 0.778 | 16 | 0.209 | nan | 8 | 0.991 |
| crop (not periodic) | cnn seed 0 | hdbscan_40 | 5 | 31 | 206 | 240 | 0.922 | 0.844 | 19 | 0.111 | nan | 8 | 0.991 |
| crop (not periodic) | cnn seed 1 | walls | 2 | 0 | 171 | 240 | 0.364 | 0.364 | 2 | 0 | nan | 9 | 0.991 |
| crop (not periodic) | cnn seed 1 | mixture | 10 | 0 | 231 | 240 | 0.251 | 0.251 | 60 | 0 | 10 | 9 | 0.991 |
| crop (not periodic) | cnn seed 1 | kmeans | 10 | 0 | 237 | 240 | 0.278 | 0.278 | 61 | 0 | 10 | 9 | 0.991 |
| crop (not periodic) | cnn seed 1 | hdbscan_5 | 15 | 208 | 32 | 240 | 0.314 | 0.042 | 19 | 0.462 | nan | 9 | 0.991 |
| crop (not periodic) | cnn seed 1 | hdbscan_10 | 10 | 191 | 48 | 240 | 0.532 | 0.195 | 20 | 0.525 | nan | 9 | 0.991 |
| crop (not periodic) | cnn seed 1 | hdbscan_20 | 7 | 130 | 109 | 240 | 0.876 | 0.514 | 15 | 0.404 | nan | 9 | 0.991 |
| crop (not periodic) | cnn seed 1 | hdbscan_40 | 3 | 7 | 232 | 240 | 0.985 | 0.948 | 3 | 0.045 | nan | 9 | 0.991 |
