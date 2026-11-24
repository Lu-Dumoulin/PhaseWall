"""Comparison on the 1D example: hand-made, learned (seeds 0 and 1, and the untrained network) and
combined descriptions, each with the walls, the threshold tree, a Gaussian mixture, k-means and
HDBSCAN; also the predictive (JEPA) description, seeds 0 and 1, and its combination; scores against the reference labels (used only here), and the probe of the labels.

    python scripts/oned_compare.py data/oned_cgle docs/oned

The ensemble and its hand-made table come from `python -m phasewall.oned example` and
`describe` (or scripts/oned_example.py). Learned tables are kept in the data folder
(descriptors_cnn_*.csv) and read back on a second run. Outputs (out folder, tracked: small files):
scores.csv, probe.csv, training.csv, methods_maps.png, phase_map_<description>.png,
kymographs.png, summary.md.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from phasewall import baselines, report, trees
from phasewall.describe import combine_tables
from phasewall.oned.jepa1d import h_table_jepa, settings_1d, train_jepa
from phasewall.oned.kymo import load_ensemble1d
from phasewall.oned.nn1d import Settings1D, h_table1d, new_encoder1d, train1d, windows_of
from phasewall.oned.report1d import make_kymographs
from phasewall.scores import point_scores
from phasewall.walls import find_walls

SEEDS = (0, 1)
METHODS = ["walls", "tree", "mixture", "kmeans", "hdbscan_10", "hdbscan_20"]


def learned_tables(ens, folder: Path, workers: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    out, logs, frames = {}, [], None
    names = [(f"learned seed {s}", s) for s in SEEDS] + [("untrained", 0)] + [(f"jepa seed {s}", s) for s in SEEDS]
    for name, seed in names:
        path = folder / f"descriptors_cnn_{name.replace(' ', '_')}.csv"
        if path.exists():
            out[name] = pd.read_csv(path)
            continue
        frames = frames or windows_of(ens, workers)
        t0 = time.time()
        if name.startswith("jepa"):
            enc, log = train_jepa(frames, settings_1d(seed=seed), verbose=False)
            logs.append(log.assign(seed=seed, objective="jepa"))
            print(f"{name}: trained in {time.time() - t0:.0f} s, final loss {log.loss.iloc[-1]:.4f}, "
                  f"h std {log.h_std.iloc[-1]:.4f}", flush=True)
            out[name] = h_table_jepa(enc, frames, ens)
            out[name].to_csv(path, index=False)
            continue
        if name == "untrained":
            enc = new_encoder1d(frames, Settings1D(seed=seed))
        else:
            enc, log = train1d(frames, Settings1D(seed=seed), verbose=False)
            logs.append(log.assign(seed=seed, objective="contrastive"))
            print(f"{name}: trained in {time.time() - t0:.0f} s, final loss {log.loss.iloc[-1]:.3f}", flush=True)
        out[name] = h_table1d(enc, frames, ens)
        out[name].to_csv(path, index=False)
    old = pd.read_csv(folder / "training_cnn.csv") if (folder / "training_cnn.csv").exists() else pd.DataFrame()
    if len(old) and "objective" not in old.columns:
        old["objective"] = "contrastive"
    log = pd.concat([old, *logs]) if logs else old
    if len(log):
        log.to_csv(folder / "training_cnn.csv", index=False)
    return out, log


def groups_of(name: str, table: pd.DataFrame, params: list[str]) -> tuple[pd.DataFrame, object]:
    """Group of every run for every method (one row per run)."""
    w = find_walls(table, params)
    groups, _ = baselines.blind_groups(table, params, sizes=(10, 20))
    key = pd.MultiIndex.from_frame(groups[params].round(9))
    node_of = pd.Series(w.nodes.index, index=pd.MultiIndex.from_frame(w.nodes[params].round(9)))
    pos = node_of.reindex(key).to_numpy()
    groups["walls"] = -1
    ok = ~np.isnan(pos)
    groups.loc[ok, "walls"] = w.nodes.phase.to_numpy()[pos[ok].astype(int)]
    tr = trees.fit_tree(table, params, use_components=name.startswith(("learned", "untrained", "jepa")))
    t_idx = {tuple(r): i for i, r in enumerate(tr.nodes[params].round(9).to_numpy())}
    groups["tree"] = tr.nodes.phase.to_numpy()[[t_idx[tuple(r)] for r in groups[params].round(9).to_numpy()]]
    return groups, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    folder, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ens = load_ensemble1d(folder / "phasewall.toml")
    params = ens.parameters
    labels = ens.runs.set_index("run").reference
    tables = {"hand": pd.read_csv(folder / "descriptors.csv")}
    learned, log = learned_tables(ens, folder, args.workers)
    tables |= learned
    tables["combined seed 0"] = combine_tables(tables["hand"], tables["learned seed 0"])
    tables["combined jepa seed 0"] = combine_tables(tables["hand"], tables["jepa seed 0"])
    rows, probes, groups_all = [], [], {}
    for name, table in tables.items():
        groups, w = groups_of(name, table, params)
        groups_all[name] = groups
        for m in METHODS:
            rows.append({"describer": name, "method": m, **point_scores(groups, m, params, labels)})
        probes.append({"describer": name, "probe": baselines.probe(table, params, labels)})
        ref = report.node_labels(w, ens.runs)
        report.make_phase_map(w, ref).savefig(out / f"phase_map_{name.replace(' ', '_')}.png", dpi=report.DPI,
                                             facecolor=report.SURFACE)
        if name == "hand":
            make_kymographs(w, ens).savefig(out / "kymographs.png", dpi=report.DPI, facecolor=report.SURFACE)
        print(f"{name}: walls {w.n_phases} phases", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "scores.csv", index=False)
    pd.DataFrame(probes).to_csv(out / "probe.csv", index=False)
    if len(log):
        log.to_csv(out / "training.csv", index=False)
    report.make_methods_map(groups_all, params, METHODS, scores).savefig(out / "methods_maps.png", dpi=report.DPI,
                                                                         facecolor=report.SURFACE)
    text = ["# 1D complex Ginzburg-Landau: descriptions and methods", "",
            (f"{len(ens.runs)} runs, {ens.runs.groupby(params).ngroups} points, {ens.windows} windows per run "
             f"(last {ens.share:g} of each run). Reference labels are used only to score."), "",
            report._markdown(scores.round(3)), "", "Probe (balanced accuracy, held-out points):", "",
            report._markdown(pd.DataFrame(probes).round(3)), ""]
    (out / "summary.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
