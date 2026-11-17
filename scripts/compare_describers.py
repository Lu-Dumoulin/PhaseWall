"""Hand-made against learned description on the Swift-Hohenberg demo, for the whole periodic
box and for a centred crop (not periodic).

Usage: python scripts/compare_describers.py data/demo docs/comparison
(data/demo is written by `phasewall demo data/demo`; needs the extra [nn].)

Describers, on the same rows (the snapshots that have a partner pair_dt earlier):
- hand: describe.py;
- cnn seed 0, cnn seed 1: the encoder trained by phasewall.nn (two training seeds);
- cnn untrained: the same encoder with its random initial weights (baseline: what the
  architecture gives without training).
For each: walls.find_walls with the default settings, compared with the reference labels of the
demo (never seen by the describers): phases, agreement, walls, mixed points, the range of RATIO
that keeps the count, the AUC of the jump J at true boundaries (links between points of
different reference labels against links inside one label), the nearest-neighbour accuracy of
the labels from the description (point means, leave one point out, balanced), and the
effective rank of the description.

Methods (second table). On each description, the walls are compared with blind grouping of the
runs (Gaussian mixture with k by BIC, k-means with k by silhouette, HDBSCAN with minimum sizes
5, 10, 20, 40; phasewall.baselines, adapted from hfml) and with a logistic probe of the three
reference states (supervised: a ceiling, not a method to find states). Scores at the level of
the points: groups, agreement, adjusted Rand index with the reference, connected pieces of the
groups on the parameter grid, share of runs called noise (HDBSCAN).

Outputs: summary.md, comparison.csv, methods.csv, training.csv, phase_maps.png, methods_maps.png,
training.png.
"""

import dataclasses
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from scipy.stats import mannwhitneyu

from phasewall import baselines, report
from phasewall.cli import agreement, node_labels
from phasewall.describe import describe_ensemble
from phasewall.graph import parameter_nodes
from phasewall.io import load_ensemble
from phasewall.nn.data import load_frames
from phasewall.nn.loss import effective_rank
from phasewall.nn.train import Settings, embed, h_table, new_encoder, train
from phasewall.walls import RATIO, feature_matrix, find_walls, phases_against_ratio

warnings.simplefilter("ignore", RuntimeWarning)
CROP_POINTS = 40          # crop of 40 x 40 points = 5 wavelengths (a choice)
SEEDS = (0, 1)            # training seeds
EPOCHS = 40
WORKERS = 8


def nn_accuracy(table: pd.DataFrame, parameters, ref: pd.Series) -> float:
    """Balanced accuracy of the label of the nearest other point (point means of the
    standardised, block-weighted description)."""
    X, _ = feature_matrix(table)
    node, _ = parameter_nodes(table[parameters].to_numpy())
    means = np.stack([X[node == k].mean(axis=0) for k in range(node.max() + 1)])
    d = np.linalg.norm(means[:, None] - means[None], axis=2)
    np.fill_diagonal(d, np.inf)
    pred = ref.to_numpy()[d.argmin(axis=1)]
    truth = ref.to_numpy()
    return float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)]))


def probe(table: pd.DataFrame, parameters, runs: pd.DataFrame, folds: int = 5, lam: float = 1.0) -> float:
    """Ridge probe hexagons against stripes per snapshot (reference label of its run), folds
    made of whole parameter points; balanced accuracy on the held-out points."""
    X, _ = feature_matrix(table)
    lab = runs.set_index("run").reference.loc[table.run].to_numpy()
    m = np.isin(lab, ["hexagons", "stripes"])
    X, y = X[m], np.where(lab[m] == "hexagons", 1.0, -1.0)
    node, _ = parameter_nodes(table[parameters].to_numpy()[m])
    fold = np.random.default_rng(0).integers(folds, size=node.max() + 1)[node]
    pred = np.zeros_like(y)
    for f in range(folds):
        tr, te = fold != f, fold == f
        A = np.c_[X[tr], np.ones(tr.sum())]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.c_[X[te], np.ones(te.sum())] @ w
    return float(np.mean([np.mean(np.sign(pred[y == c]) == c) for c in (-1.0, 1.0)]))


def within_points(table: pd.DataFrame, parameters) -> float:
    """Share of the variance of the description that lies within parameter points."""
    X, _ = feature_matrix(table)
    node, _ = parameter_nodes(table[parameters].to_numpy())
    within = sum(((X[node == k] - X[node == k].mean(axis=0)) ** 2).sum() for k in np.unique(node))
    return float(within / ((X - X.mean(axis=0)) ** 2).sum())


def evaluate(name, setting, table, ens, runs, seconds) -> tuple[dict, object, pd.Series]:
    w = find_walls(table, ens.parameters)
    ref = node_labels(w, runs)
    _, agree = agreement(w, ref)
    curve = phases_against_ratio(table, ens.parameters)
    same = curve[curve.phases == w.n_phases].ratio
    a, b = w.edges.a.to_numpy(), w.edges.b.to_numpy()
    boundary = ref.to_numpy()[a] != ref.to_numpy()[b]
    J = w.edges.jump.to_numpy()
    auc = mannwhitneyu(J[boundary], J[~boundary]).statistic / (boundary.sum() * (~boundary).sum())
    X, names = feature_matrix(table)
    r = ref.to_numpy()
    pattern = {"hexagons", "stripes"}
    hs = np.array([{r[i], r[j]} == pattern for i, j in zip(a, b)])
    up = np.array([("uniform" in {r[i], r[j]}) and r[i] != r[j] for i, j in zip(a, b)])
    med = np.median(J)
    row = {"setting": setting, "describer": name, "columns": len(names), "phases": w.n_phases,
           "agree": agree, "points": len(w.nodes), "walls": int(w.edges.wall.sum()),
           "mixed": int(w.nodes.mixed.sum()),
           "ratio_range": f"{same.min():g}-{same.max():g}" if len(same) else "-",
           "auc_boundary": float(auc), "nn_accuracy": nn_accuracy(table, ens.parameters, ref),
           "effective_rank": effective_rank(torch.from_numpy(X)),
           "jump_inner": float(np.median(J[~boundary]) / med), "jump_hex_stripes": float(np.median(J[hs]) / med),
           "jump_uniform_pattern": float(np.median(J[up]) / med),
           "within_points": within_points(table, ens.parameters),
           "probe_hex_stripes": probe(table, ens.parameters, runs), "seconds": seconds}
    return row, w, ref


def compare_methods(name, setting, table, ens, runs, w) -> tuple[list[dict], pd.DataFrame]:
    """Walls and blind grouping on one description, scored against the reference labels."""
    labels = runs.set_index("run").reference
    t0 = time.time()
    groups, info = baselines.blind_groups(table, ens.parameters)
    seconds = time.time() - t0
    node, _ = parameter_nodes(groups[ens.parameters].to_numpy())
    groups["walls"] = w.nodes["phase"].to_numpy()[node]
    p = baselines.probe(table, ens.parameters, labels)
    rows = []
    for col in ["walls", "mixture", "kmeans", *[c for c in groups if c.startswith("hdbscan_")]]:
        k = {"mixture": info["mixture"]["k"], "kmeans": info["kmeans"]["k"]}.get(col)
        rows.append({"setting": setting, "describer": name, "method": col,
                     **baselines.point_scores(groups, col, ens.parameters, labels),
                     "k_chosen": k, "components": info["components"], "probe_3_states": p,
                     "seconds_grouping": seconds})
    return rows, groups


def main(demo: str, out: str):
    demo, out = Path(demo), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = load_ensemble(demo / "phasewall.toml")
    runs = pd.read_csv(demo / "runs.csv")
    settings = {"periodic box": base, "crop (not periodic)": dataclasses.replace(base, crop=CROP_POINTS * base.dx)}
    rows, maps, logs, method_rows, groupings = [], {}, [], [], {}
    for setting, ens in settings.items():
        t0 = time.time()
        hand, errors = describe_ensemble(ens, workers=WORKERS)
        assert not errors, errors
        t_hand = time.time() - t0
        frames = load_frames(ens, workers=WORKERS)
        keep = set(zip(frames.run, frames.time))
        hand = hand[[k in keep for k in zip(hand.run, hand.time)]].reset_index(drop=True)
        tables = {"hand": (hand, t_hand)}
        enc = new_encoder(frames, Settings(seed=0)).eval()
        tables["cnn untrained"] = (h_table(frames, embed(enc, frames), ens), 0.0)
        for seed in SEEDS:
            t0 = time.time()
            print(f"--- {setting}: training seed {seed}", flush=True)
            enc, log = train(frames, Settings(epochs=EPOCHS, seed=seed))
            tables[f"cnn seed {seed}"] = (h_table(frames, embed(enc, frames), ens), time.time() - t0)
            logs.append(log.assign(setting=setting, seed=seed))
        for name, (table, sec) in tables.items():
            row, w, ref = evaluate(name, setting, table, ens, runs, sec)
            rows.append(row)
            maps[(setting, name)] = (w, ref)
            print(row, flush=True)
            mrows, groups = compare_methods(name, setting, table, ens, runs, w)
            method_rows += mrows
            groupings[(setting, name)] = groups
            print(pd.DataFrame(mrows)[["method", "groups", "agree", "ari", "pieces", "noise_runs"]].round(3)
                  .to_string(index=False), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "comparison.csv", index=False)
    methods = pd.DataFrame(method_rows)
    methods.to_csv(out / "methods.csv", index=False)
    method_figure(groupings, methods, base.parameters, out)
    pd.concat(logs).to_csv(out / "training.csv", index=False)
    figures(maps, pd.concat(logs), out)
    write_summary(res, pd.concat(logs), out, methods)


def method_figure(groupings, methods, parameters, out, describers=("hand", "cnn seed 0"),
                  columns=("walls", "mixture", "kmeans", "hdbscan_10")):
    """Majority group of every point for each method (rows: frame and description)."""
    keys = [(s, d) for s in dict.fromkeys(k[0] for k in groupings) for d in describers]
    fig = Figure(figsize=(3.6 * len(columns), 3.7 * len(keys)), facecolor=report.SURFACE, layout="constrained")
    axes = fig.subplots(len(keys), len(columns))
    x, y = parameters
    for i, (setting, name) in enumerate(keys):
        g = groupings[(setting, name)]
        node, points = parameter_nodes(g[parameters].to_numpy())
        for j, col in enumerate(columns):
            maj = pd.Series(g[col].to_numpy()).groupby(node).agg(lambda v: v.value_counts().idxmax()).to_numpy()
            ranks = pd.Series(maj[maj >= 0]).value_counts().index.tolist()      # largest group first
            classes = np.array([ranks.index(v) if v >= 0 else -1 for v in maj])
            ax = axes[i, j]
            report._points_by_class(ax, points[:, 0], points[:, 1], classes,
                                    lambda c: f"group {c}" if c >= 0 else "noise")
            m = methods[(methods.setting == setting) & (methods.describer == name) & (methods.method == col)].iloc[0]
            report._style(ax, x, y)
            ax.set_title(f"{setting}, {name}: {col}\n{m.groups} groups, {m.agree}/{m.points} agree, "
                         f"ARI {m.ari:.2f}, {m.pieces} pieces", fontsize=8, loc="left", color=report.INK)
    fig.savefig(out / "methods_maps.png", dpi=report.DPI, facecolor=report.SURFACE)


def figures(maps, logs, out):
    names = ["hand", "cnn seed 0", "cnn untrained"]
    settings = list(dict.fromkeys(s for s, _ in maps))
    fig = Figure(figsize=(4.0 * (len(names) + 1), 4.3 * len(settings)), facecolor=report.SURFACE,
                 layout="constrained")
    axes = fig.subplots(len(settings), len(names) + 1)
    for i, setting in enumerate(settings):
        w0, ref = maps[(setting, "hand")]
        report.plot_reference(axes[i, 0], w0, ref)
        axes[i, 0].set_title(f"{setting}: reference labels", fontsize=9, loc="left", color=report.INK)
        for j, name in enumerate(names):
            w, ref = maps[(setting, name)]
            report.plot_phases(axes[i, j + 1], w)
            _, agree = agreement(w, ref)
            axes[i, j + 1].set_title(f"{setting}: {name}\n{w.n_phases} phases, {agree}/{len(w.nodes)} agree",
                                     fontsize=9, loc="left", color=report.INK)
    fig.savefig(out / "phase_maps.png", dpi=report.DPI, facecolor=report.SURFACE)

    fig = Figure(figsize=(9.0, 3.2), facecolor=report.SURFACE, layout="constrained")
    ax1, ax2 = fig.subplots(1, 2)
    for k, ((setting, seed), g) in enumerate(logs.groupby(["setting", "seed"], sort=False)):
        colour = report.PHASE_COLOURS[0 if setting.startswith("periodic") else 1]
        ls = "-" if seed == 0 else "--"
        ax1.plot(g.epoch, g.loss, color=colour, ls=ls, lw=1.6, label=f"{setting}, seed {seed}")
        ax2.plot(g.epoch, g.effective_rank_h, color=colour, ls=ls, lw=1.6)
    ax1.axhline(logs.collapse_loss.iloc[0], color=report.INK_2, lw=0.8, ls=":")
    ax1.text(1, logs.collapse_loss.iloc[0], " collapse ln(2N - 1)", fontsize=7, va="bottom", color=report.INK_2)
    ax1.legend(fontsize=7, frameon=False)
    report._style(ax1, "epoch", "InfoNCE loss")
    report._style(ax2, "epoch", "effective rank of h")
    fig.savefig(out / "training.png", dpi=report.DPI, facecolor=report.SURFACE)


def write_summary(res, logs, out, methods=None):
    last = logs.groupby(["setting", "seed"], sort=False).tail(1)
    lines = ["# Hand-made against learned description (Swift-Hohenberg demo)", "",
             (f"Walls with the default settings (RATIO = {RATIO}). Agreement: points that carry the most "
              "common reference label of their region. AUC: probability that a link between two "
              "reference states jumps more than a link inside one state. NN accuracy: label of the "
              "nearest other point in the description (balanced over the three states). jump_*: median "
              "jump of the links inside one state, between hexagons and stripes, and between the uniform "
              "state and a pattern, divided by the median of all jumps. within_points: share of the "
              "variance of the description within parameter points. probe_hex_stripes: ridge probe "
              "hexagons against stripes on held-out parameter points (balanced accuracy)."), "",
             report._markdown(res), "", "Last epoch of each training:", "",
             report._markdown(last[["setting", "seed", "loss", "collapse_loss", "alignment", "uniformity",
                                    "effective_rank_h"]]), ""]
    if methods is not None:
        lines += ["## Walls against blind grouping", "",
                  ("groups: groups found (majority group of each point); agree: points that carry the most "
                   "common reference label of their group; ari: adjusted Rand index between groups and "
                   "reference labels of the points; pieces: connected pieces of the groups on the "
                   "parameter grid (3 for the three reference states); noise_runs: share of runs that "
                   "HDBSCAN cannot place; k_chosen: k of the mixture (BIC) and of k-means (silhouette); "
                   "probe_3_states: logistic regression of the three states on held-out points "
                   "(supervised, balanced accuracy)."), "",
                  report._markdown(methods.drop(columns=["seconds_grouping"]).round(3)), ""]
    (out / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:3])
