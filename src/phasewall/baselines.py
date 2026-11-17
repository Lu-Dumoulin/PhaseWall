"""Other ways to find the states, for comparison with the walls: blind grouping of the runs
(Gaussian mixture, k-means, HDBSCAN) and a supervised linear probe (logistic regression).
Adapted from hfml (discover.py, Step 5.2, and the probe of the Step 7 evaluation). Needs the
extra `[baselines]` (scikit-learn).

Blind grouping. Each run is the mean of its snapshots (one row per run). The description is
standardised and block-weighted as in walls.feature_matrix, then compressed to the principal
components that explain PCA_VARIANCE of the variance. Then:
- Gaussian mixture, k from 1 to 10, k chosen by the smallest BIC;
- k-means, k from 2 to 10, k chosen by the largest mean silhouette;
- HDBSCAN for minimum cluster sizes HDBSCAN_SIZES (runs it cannot place are noise, -1).
None of them uses the parameter grid: two runs are grouped by their description only, so a
group may be spread over several separate places of the grid; the walls method, which cuts
the grid, gives connected regions by construction.

Probe. Logistic regression of a label from the description of each snapshot, tested on
parameter points held out in GroupKFold folds (no point is in the training and the test of the
same fold); balanced accuracy. It sees the labels, so it is not a way to find the states: it
measures how well a description separates them at best.

Comparison with a labelling of the points (point_scores): the label of a point is the most
common label of its runs; agreement = points whose label is the most common label of their
group (or region), as for the walls; ARI = adjusted Rand index between the two partitions of
the points (on the points that are placed: HDBSCAN noise is left out); ari_all = the same with
noise counted as one more group, so that leaving points out is not rewarded; pieces = connected
pieces of the groups on the parameter graph.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold

from .graph import grid_edges, parameter_nodes, regions
from .walls import feature_matrix

# Choices (as in hfml protocol version 2, except that rows are runs of an ensemble)
PCA_VARIANCE = 0.95
K_MIXTURE = tuple(range(1, 11))
K_KMEANS = tuple(range(2, 11))
N_INIT = 10
HDBSCAN_SIZES = (5, 10, 20, 40)
FOLDS = 5
C_PROBE = 1.0           # inverse strength of the L2 penalty of the logistic regression
SEED = 0


def run_matrix(table: pd.DataFrame, parameters: list[str], columns: list[str] | None = None
               ) -> tuple[np.ndarray, pd.DataFrame]:
    """Mean description of each run (rows of feature_matrix averaged per run) and a table of
    the runs (run, parameters)."""
    X, _ = feature_matrix(table, columns)
    run = table["run"].to_numpy()
    ids = np.unique(run)
    M = np.stack([X[run == r].mean(axis=0) for r in ids])
    runs = table.groupby("run")[parameters].first().loc[ids].reset_index()
    return M, runs


def reduce(X: np.ndarray, variance: float = PCA_VARIANCE) -> np.ndarray:
    """Principal components explaining `variance` of the (centred) columns."""
    return PCA(n_components=variance, svd_solver="full", random_state=SEED).fit_transform(X - X.mean(axis=0))


def fit_mixture(Z: np.ndarray, ks=K_MIXTURE, n_init: int = N_INIT) -> dict:
    """Gaussian mixtures for every k; the one with the smallest BIC."""
    models = {k: GaussianMixture(k, covariance_type="full", n_init=n_init, random_state=SEED).fit(Z) for k in ks}
    bic = {k: float(m.bic(Z)) for k, m in models.items()}
    k = min(bic, key=bic.get)
    return {"k": k, "labels": models[k].predict(Z), "bic": bic}


def fit_kmeans(Z: np.ndarray, ks=K_KMEANS, n_init: int = N_INIT) -> dict:
    """k-means for every k; the one with the largest mean silhouette."""
    labels = {k: KMeans(k, n_init=n_init, random_state=SEED).fit_predict(Z) for k in ks}
    sil = {k: float(silhouette_score(Z, lab, random_state=SEED)) for k, lab in labels.items()}
    k = max(sil, key=sil.get)
    return {"k": k, "labels": labels[k], "silhouette": sil}


def fit_hdbscan(Z: np.ndarray, sizes=HDBSCAN_SIZES) -> dict[int, np.ndarray]:
    """HDBSCAN labels for each minimum cluster size (-1: noise)."""
    return {m: HDBSCAN(min_cluster_size=m, copy=True).fit_predict(Z) for m in sizes}


def n_groups(labels: np.ndarray) -> int:
    return len(set(np.asarray(labels).tolist()) - {-1})


def blind_groups(table: pd.DataFrame, parameters: list[str], columns: list[str] | None = None,
                 n_init: int = N_INIT) -> tuple[pd.DataFrame, dict]:
    """Groups of every run by every method: a table (run, parameters, mixture, kmeans,
    hdbscan_5, ...) and the details (k chosen, BIC, silhouette, number of components)."""
    M, runs = run_matrix(table, parameters, columns)
    Z = reduce(M)
    mix, km, hdb = fit_mixture(Z, n_init=n_init), fit_kmeans(Z, n_init=n_init), fit_hdbscan(Z)
    out = runs.assign(mixture=mix["labels"], kmeans=km["labels"])
    for m, lab in hdb.items():
        out[f"hdbscan_{m}"] = lab
    return out, {"components": Z.shape[1], "mixture": mix, "kmeans": km}


def point_scores(groups: pd.DataFrame, column: str, parameters: list[str], labels: pd.Series) -> dict:
    """Compare a grouping of the runs with labels of the runs (Series indexed by run): at the
    level of the points (majority group and majority label of each point)."""
    g = groups.set_index("run")[column]
    node, points = parameter_nodes(groups[parameters].to_numpy())
    runs = groups["run"].to_numpy()

    def majority(values):
        return pd.Series(values).groupby(node).agg(lambda s: s.value_counts().idxmax()).to_numpy()

    pg = majority(g.loc[runs].to_numpy())
    pl = majority(labels.loc[runs].to_numpy())
    kept = pg != -1
    table = pd.crosstab(pl[kept], pg[kept])
    edges = grid_edges(points)
    same = pg[edges["a"].to_numpy()] == pg[edges["b"].to_numpy()]
    pieces = len(np.unique(regions(len(points), edges, same)[kept]))
    return {"groups": n_groups(pg), "noise_points": int((~kept).sum()), "agree": int(table.max(axis=0).sum()),
            "points": len(points), "ari": float(adjusted_rand_score(pl[kept], pg[kept])),
            "ari_all": float(adjusted_rand_score(pl, pg)),
            "pieces": pieces, "noise_runs": float(np.mean(g.to_numpy() == -1))}


def probe(table: pd.DataFrame, parameters: list[str], labels: pd.Series, columns: list[str] | None = None,
          folds: int = FOLDS, C: float = C_PROBE) -> float:
    """Balanced accuracy of a logistic regression of the run labels (Series indexed by run)
    from the description of each snapshot, on parameter points held out (GroupKFold)."""
    X, _ = feature_matrix(table, columns)
    y = labels.loc[table["run"]].to_numpy()
    node, _ = parameter_nodes(table[parameters].to_numpy())
    pred = np.empty_like(y)
    for train, test in GroupKFold(folds).split(X, y, node):
        model = LogisticRegression(C=C, max_iter=2000).fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))
