# Networks compared on swift-hohenberg (40 epochs, seed 0)

## Label-free diagnostics (last epoch): what a choice without labels can use

| network | channels | hidden | projection | temperature | receptive_field | weights_f | loss / collapse | alignment | uniformity | rank of h | rank of h / dim_h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | (32, 64, 128, 256) | 256 | 64 | 0.2 | 91 | 1173888 | 0.536 | 0.895 | -2.75 | 26.5 | 0.103 |
| 3 blocks | (32, 64, 256) | 256 | 64 | 0.2 | 43 | 804480 | 0.525 | 0.894 | -2.77 | 34.4 | 0.134 |
| 5 blocks | (32, 64, 128, 256, 512) | 512 | 64 | 0.2 | 187 | 4715904 | 0.489 | 0.925 | -2.81 | 29.7 | 0.058 |
| narrow | (8, 16, 32, 64) | 64 | 32 | 0.2 | 91 | 74016 | 0.66 | 0.885 | -2.21 | 13.3 | 0.208 |
| wide h | (32, 64, 128, 512) | 512 | 128 | 0.2 | 91 | 3239808 | 0.542 | 0.896 | -2.71 | 36.8 | 0.072 |
| T = 0.5 | (32, 64, 128, 256) | 256 | 64 | 0.5 | 91 | 1173888 | 0.73 | 0.944 | -2.27 | 23.4 | 0.091 |
| d4 views | (32, 64, 128, 256) | 256 | 64 | 0.2 | 91 | 1173888 | 0.264 | 0.999 | -3.35 | 84.9 | 0.332 |

## Scores against the reference labels (documentation only; not used to choose)

| network | method | groups | agree | ari | pieces | probe |
|---|---|---|---|---|---|---|
| default | walls | 3 | 237 | 0.956 | 3 | 0.998 |
| default | hdbscan_20 | 4 | 117 | 0.897 | 7 | 0.998 |
| 3 blocks | walls | 4 | 234 | 0.886 | 4 | 0.996 |
| 3 blocks | hdbscan_20 | 7 | 137 | 0.688 | 16 | 0.996 |
| 5 blocks | walls | 3 | 234 | 0.914 | 3 | 0.999 |
| 5 blocks | hdbscan_20 | 5 | 142 | 0.762 | 8 | 0.999 |
| narrow | walls | 4 | 235 | 0.899 | 4 | 0.998 |
| narrow | hdbscan_20 | 3 | 233 | 0.985 | 3 | 0.998 |
| wide h | walls | 3 | 235 | 0.928 | 3 | 0.997 |
| wide h | hdbscan_20 | 4 | 117 | 0.916 | 4 | 0.997 |
| T = 0.5 | walls | 4 | 235 | 0.907 | 4 | 0.998 |
| T = 0.5 | hdbscan_20 | 6 | 128 | 0.68 | 11 | 0.998 |
| d4 views | walls | 2 | 171 | 0.364 | 2 | 0.981 |
| d4 views | hdbscan_20 | 2 | 164 | 0.366 | 2 | 0.981 |
