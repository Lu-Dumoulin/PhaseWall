# Hand-made against learned description (Swift-Hohenberg demo)

Walls with the default settings (RATIO = 1.5). Agreement: points that carry the most common reference label of their region. AUC: probability that a link between two reference states jumps more than a link inside one state. NN accuracy: label of the nearest other point in the description (balanced over the three states). jump_*: median jump of the links inside one state, between hexagons and stripes, and between the uniform state and a pattern, divided by the median of all jumps. within_points: share of the variance of the description within parameter points. probe_hex_stripes: ridge probe hexagons against stripes on held-out parameter points (balanced accuracy).

| setting | describer | columns | phases | agree | points | walls | mixed | ratio_range | auc_boundary | nn_accuracy | effective_rank | jump_inner | jump_hex_stripes | jump_uniform_pattern | within_points | probe_hex_stripes | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| periodic box | hand | 28 | 3 | 231 | 240 | 39 | 1 | 1.2-2 | 0.967 | 0.979 | 14.2 | 0.962 | 1.81 | 4.24 | 0.268 | 0.989 | 3.73 |
| periodic box | cnn untrained | 155 | 2 | 169 | 240 | 15 | 0 | 1.2-2 | 0.808 | 0.962 | 65.7 | 0.989 | 1.21 | 2.25 | 0.36 | 0.947 | 0 |
| periodic box | cnn seed 0 | 256 | 2 | 169 | 240 | 15 | 3 | 1.2-2 | 0.743 | 0.965 | 67.3 | 0.988 | 1.03 | 1.96 | 0.073 | 0.964 | 125 |
| periodic box | cnn seed 1 | 256 | 2 | 169 | 240 | 15 | 1 | 1.2-2.5 | 0.811 | 0.969 | 81.4 | 0.978 | 1.15 | 2.41 | 0.197 | 0.975 | 137 |
| crop (not periodic) | hand | 24 | 3 | 227 | 240 | 36 | 1 | 1.2-1.5 | 0.97 | 0.974 | 12.6 | 0.975 | 1.85 | 4.83 | 0.258 | 0.99 | 4.15 |
| crop (not periodic) | cnn untrained | 177 | 2 | 159 | 240 | 12 | 0 | 1.2-2 | 0.759 | 0.979 | 103 | 0.995 | 1.11 | 1.79 | 0.565 | 0.956 | 0 |
| crop (not periodic) | cnn seed 0 | 256 | 2 | 169 | 240 | 15 | 0 | 1.2-3 | 0.829 | 0.922 | 90.6 | 0.987 | 1.11 | 2.65 | 0.328 | 0.954 | 65.2 |
| crop (not periodic) | cnn seed 1 | 256 | 2 | 169 | 240 | 15 | 0 | 1.2-3 | 0.781 | 0.92 | 85 | 0.984 | 1.06 | 2.71 | 0.319 | 0.947 | 65.7 |

Last epoch of each training:

| setting | seed | loss | collapse_loss | alignment | uniformity | effective_rank_h |
|---|---|---|---|---|---|---|
| periodic box | 0 | 1.5 | 5.54 | 0.997 | -3.34 | 66.5 |
| periodic box | 1 | 1.47 | 5.54 | 0.997 | -3.35 | 80.5 |
| crop (not periodic) | 0 | 1.46 | 5.54 | 0.996 | -3.36 | 89.5 |
| crop (not periodic) | 1 | 1.46 | 5.54 | 0.996 | -3.35 | 84 |
