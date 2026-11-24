# 1D complex Ginzburg-Landau: descriptions and methods

504 runs, 126 points, 8 windows per run (last 0.5 of each run). Reference labels are used only to score.

| describer | method | groups | noise_points | agree | points | ari | ari_all | pieces | noise_runs |
|---|---|---|---|---|---|---|---|---|---|
| hand | walls | 4 | 0 | 122 | 126 | 0.804 | 0.804 | 4 | 0 |
| hand | tree | 1 | 0 | 60 | 126 | 0 | 0 | 1 | 0 |
| hand | mixture | 9 | 0 | 120 | 126 | 0.518 | 0.518 | 14 | 0 |
| hand | kmeans | 8 | 0 | 121 | 126 | 0.539 | 0.539 | 13 | 0 |
| hand | hdbscan_10 | 8 | 13 | 110 | 126 | 0.625 | 0.5 | 14 | 0.085 |
| hand | hdbscan_20 | 7 | 10 | 113 | 126 | 0.602 | 0.508 | 15 | 0.087 |
| learned seed 0 | walls | 1 | 0 | 60 | 126 | 0 | 0 | 1 | 0 |
| learned seed 0 | tree | 1 | 0 | 60 | 126 | 0 | 0 | 1 | 0 |
| learned seed 0 | mixture | 10 | 0 | 120 | 126 | 0.302 | 0.302 | 14 | 0 |
| learned seed 0 | kmeans | 10 | 0 | 114 | 126 | 0.273 | 0.273 | 17 | 0 |
| learned seed 0 | hdbscan_10 | 3 | 33 | 86 | 126 | 0.865 | 0.648 | 4 | 0.262 |
| learned seed 0 | hdbscan_20 | 2 | 39 | 84 | 126 | 0.958 | 0.683 | 3 | 0.31 |
| learned seed 1 | walls | 1 | 0 | 60 | 126 | 0 | 0 | 1 | 0 |
| learned seed 1 | tree | 3 | 2 | 100 | 126 | 0.486 | 0.489 | 4 | 0.016 |
| learned seed 1 | mixture | 10 | 0 | 121 | 126 | 0.389 | 0.389 | 16 | 0 |
| learned seed 1 | kmeans | 10 | 0 | 122 | 126 | 0.408 | 0.408 | 15 | 0 |
| learned seed 1 | hdbscan_10 | 5 | 40 | 84 | 126 | 0.895 | 0.593 | 9 | 0.323 |
| learned seed 1 | hdbscan_20 | 2 | 50 | 76 | 126 | 1 | 0.591 | 2 | 0.45 |
| untrained | walls | 2 | 0 | 102 | 126 | 0.677 | 0.677 | 3 | 0 |
| untrained | tree | 1 | 0 | 60 | 126 | 0 | 0 | 1 | 0 |
| untrained | mixture | 3 | 0 | 101 | 126 | 0.509 | 0.509 | 10 | 0 |
| untrained | kmeans | 3 | 0 | 101 | 126 | 0.434 | 0.434 | 10 | 0 |
| untrained | hdbscan_10 | 2 | 15 | 89 | 126 | 0.683 | 0.542 | 2 | 0.181 |
| untrained | hdbscan_20 | 2 | 28 | 77 | 126 | 0.652 | 0.454 | 3 | 0.272 |
| jepa seed 0 | walls | 3 | 0 | 102 | 126 | 0.625 | 0.625 | 3 | 0 |
| jepa seed 0 | tree | 3 | 4 | 118 | 126 | 0.916 | 0.868 | 3 | 0.032 |
| jepa seed 0 | mixture | 9 | 0 | 118 | 126 | 0.379 | 0.379 | 22 | 0 |
| jepa seed 0 | kmeans | 2 | 0 | 102 | 126 | 0.596 | 0.596 | 2 | 0 |
| jepa seed 0 | hdbscan_10 | 5 | 47 | 79 | 126 | 0.805 | 0.349 | 8 | 0.365 |
| jepa seed 0 | hdbscan_20 | 3 | 24 | 102 | 126 | 1 | 0.752 | 4 | 0.226 |
| jepa seed 1 | walls | 3 | 0 | 112 | 126 | 0.739 | 0.739 | 3 | 0 |
| jepa seed 1 | tree | 3 | 0 | 101 | 126 | 0.466 | 0.466 | 3 | 0 |
| jepa seed 1 | mixture | 9 | 0 | 121 | 126 | 0.392 | 0.392 | 20 | 0 |
| jepa seed 1 | kmeans | 2 | 0 | 101 | 126 | 0.569 | 0.569 | 2 | 0 |
| jepa seed 1 | hdbscan_10 | 3 | 15 | 110 | 126 | 0.975 | 0.83 | 4 | 0.143 |
| jepa seed 1 | hdbscan_20 | 3 | 26 | 100 | 126 | 1 | 0.771 | 5 | 0.258 |
| combined seed 0 | walls | 2 | 0 | 103 | 126 | 0.633 | 0.633 | 2 | 0 |
| combined seed 0 | tree | 1 | 0 | 60 | 126 | 0 | 0 | 1 | 0 |
| combined seed 0 | mixture | 9 | 0 | 121 | 126 | 0.477 | 0.477 | 15 | 0 |
| combined seed 0 | kmeans | 7 | 0 | 121 | 126 | 0.682 | 0.682 | 10 | 0 |
| combined seed 0 | hdbscan_10 | 6 | 10 | 115 | 126 | 0.789 | 0.676 | 10 | 0.103 |
| combined seed 0 | hdbscan_20 | 5 | 10 | 113 | 126 | 0.866 | 0.753 | 6 | 0.093 |
| combined jepa seed 0 | walls | 4 | 0 | 123 | 126 | 0.826 | 0.826 | 4 | 0 |
| combined jepa seed 0 | tree | 1 | 0 | 60 | 126 | 0 | 0 | 1 | 0 |
| combined jepa seed 0 | mixture | 10 | 0 | 122 | 126 | 0.481 | 0.481 | 18 | 0 |
| combined jepa seed 0 | kmeans | 8 | 0 | 121 | 126 | 0.539 | 0.539 | 13 | 0 |
| combined jepa seed 0 | hdbscan_10 | 8 | 15 | 110 | 126 | 0.589 | 0.456 | 17 | 0.127 |
| combined jepa seed 0 | hdbscan_20 | 6 | 9 | 115 | 126 | 0.794 | 0.692 | 9 | 0.099 |

Probe (balanced accuracy, held-out points):

| describer | probe |
|---|---|
| hand | 0.954 |
| learned seed 0 | 0.944 |
| learned seed 1 | 0.946 |
| untrained | 0.948 |
| jepa seed 0 | 0.957 |
| jepa seed 1 | 0.963 |
| combined seed 0 | 0.955 |
| combined jepa seed 0 | 0.955 |
