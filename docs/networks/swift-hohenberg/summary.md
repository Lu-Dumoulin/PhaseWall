# Networks compared on swift-hohenberg (40 epochs, seed 0)

## Label-free diagnostics (last epoch): what a choice without labels can use

| network | channels | hidden | projection | temperature | receptive_field | weights_f | loss / collapse | alignment | uniformity | rank of h | rank of h / dim_h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | (32, 64, 128, 256) | 256 | 64 | 0.2 | 91 | 1173888 | 0.264 | 0.999 | -3.35 | 84.9 | 0.332 |
| 3 blocks | (32, 64, 256) | 256 | 64 | 0.2 | 43 | 804480 | 0.265 | 0.999 | -3.35 | 60.7 | 0.237 |
| 5 blocks | (32, 64, 128, 256, 512) | 512 | 64 | 0.2 | 187 | 4715904 | 0.277 | 0.998 | -3.32 | 55.2 | 0.108 |
| narrow | (8, 16, 32, 64) | 64 | 32 | 0.2 | 91 | 74016 | 0.293 | 0.995 | -3.27 | 32.1 | 0.502 |
| wide h | (32, 64, 128, 512) | 512 | 128 | 0.2 | 91 | 3239808 | 0.255 | 0.999 | -3.38 | 109 | 0.213 |
| T = 0.5 | (32, 64, 128, 256) | 256 | 64 | 0.5 | 91 | 1173888 | 0.659 | 1 | -3.29 | 61.9 | 0.242 |

## Scores against the reference labels (documentation only; not used to choose)

| network | method | groups | agree | ari | pieces | probe |
|---|---|---|---|---|---|---|
| default | walls | 2 | 171 | 0.364 | 2 | 0.981 |
| default | hdbscan_20 | 2 | 164 | 0.366 | 2 | 0.981 |
| 3 blocks | walls | 2 | 171 | 0.364 | 2 | 0.99 |
| 3 blocks | hdbscan_20 | 2 | 115 | 0.385 | 2 | 0.99 |
| 5 blocks | walls | 2 | 171 | 0.364 | 2 | 0.932 |
| 5 blocks | hdbscan_20 | 2 | 165 | 0.363 | 2 | 0.932 |
| narrow | walls | 2 | 171 | 0.364 | 2 | 0.979 |
| narrow | hdbscan_20 | 2 | 154 | 0.363 | 2 | 0.979 |
| wide h | walls | 2 | 171 | 0.364 | 2 | 0.983 |
| wide h | hdbscan_20 | 2 | 142 | 0.368 | 2 | 0.983 |
| T = 0.5 | walls | 2 | 171 | 0.364 | 2 | 0.982 |
| T = 0.5 | hdbscan_20 | 2 | 171 | 0.364 | 2 | 0.982 |
