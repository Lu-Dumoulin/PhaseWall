"""Figures of real numbers for the documentation (docs/documentation/), made from the demo
ensemble and from synthetic tables.

Usage: python scripts/doc_figures.py data/demo docs/documentation/data
(data/demo is written by `phasewall demo data/demo`.)

Each figure replays one step of walls.find_walls with the same functions, so that the
documentation shows what the method actually computed: the Fiedler vector and the
normalised cut of each cut tried, the jumps used by the test, the spread used for the mixed
flag, and the selection bias that the split into halves removes.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_walls import DIM, table

from phasewall import report, walls
from phasewall.cli import node_labels
from phasewall.describe import k_shells
from phasewall.graph import parameter_nodes
from phasewall.io import load_ensemble, read_snapshot, run_snapshots

warnings.simplefilter("ignore", RuntimeWarning)
COL, INK, INK_2, GRID = report.PHASE_COLOURS, report.INK, report.INK_2, report.GRID
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#f0efec", "#b23a39"])
NOISE_SEEDS = 40     # synthetic pure-noise tables for the selection-bias figure (a choice)


def new_figure(w: float, h: float, n: int = 1):
    fig = Figure(figsize=(w, h), facecolor="white", layout="constrained")
    axes = fig.subplots(1, n)
    return fig, np.atleast_1d(axes)


def style(ax, xlabel="", ylabel="", title=""):
    report._style(ax, xlabel, ylabel)
    ax.set_facecolor("white")
    if title:
        ax.set_title(title, color=INK, fontsize=9, loc="left")


# ---------------------------------------------------------------- replay of the cuts
def weights_and_logs(w: walls.Walls):
    """Weights and log jumps exactly as find_walls passes them to split_regions."""
    log_select = np.log(np.maximum(w.edges.jump_select.to_numpy(), w.floor))
    log_test = np.log(np.maximum(w.edges.jump_test.to_numpy(), w.floor))
    weight = np.maximum(np.exp(-(np.exp(log_select - np.median(log_select))) ** 2), walls.MIN_WEIGHT)
    return weight, log_select, log_test


def fiedler(members, a, b, weight):
    """Fiedler vector (random-walk normalisation D^-1/2 v2) and the normalised cut along its
    order, as in walls.best_cut."""
    m = len(members)
    pos = walls._positions(members, a, b)
    inside = (pos[a] >= 0) & (pos[b] >= 0)
    A = np.zeros((m, m))
    A[pos[a[inside]], pos[b[inside]]] = weight[inside]
    A += A.T
    d = A.sum(axis=1)
    dm = 1 / np.sqrt(d)
    val, vec = np.linalg.eigh(np.eye(m) - dm[:, None] * A * dm[None, :])
    f = vec[:, 1] * dm
    order = np.argsort(f)
    ncut = []
    S = np.zeros(m, dtype=bool)
    for k in range(m - 1):
        S[order[k]] = True
        cut = A[S][:, ~S].sum()
        ncut.append(cut / d[S].sum() + cut / (d.sum() - d[S].sum()))
    return f, np.array(ncut), val[:4]


def replay(w: walls.Walls):
    """The cuts tried by find_walls, in order: members, side, Fiedler vector, test result."""
    a, b = w.edges.a.to_numpy(), w.edges.b.to_numpy()
    weight, _, log_test = weights_and_logs(w)
    s = w.settings
    queue, out = [np.arange(len(w.nodes))], []
    while queue:
        members = queue.pop(0)
        parts = walls.pieces(members, a, b)
        if len(parts) > 1:
            queue += parts
            continue
        side = walls.best_cut(members, a, b, weight, s["min_phase"])
        if side is None:
            continue
        t = walls.test_cut(side, members, a, b, log_test)
        f, ncut, val = fiedler(members, a, b, weight)
        accepted = t["p"] < s["alpha"] and t["ratio"] >= s["ratio"]
        out.append({"members": members, "side": side, "fiedler": f, "ncut": ncut, "eig": val, **t,
                    "accepted": accepted})
        if accepted:
            queue += [members[side], members[~side]]
    return out


# ---------------------------------------------------------------- figures
def fig_sk(w, table, names, dx, n, out):
    """Shape of S(k) at the central point of each phase (mean over its snapshots)."""
    edges = k_shells(n, dx)
    centres = np.sqrt(edges[:-1] * edges[1:])
    cols = sorted(c for c in table.columns if c.startswith("u.shape."))
    fig, (ax,) = new_figure(4.6, 2.9)
    node, _ = parameter_nodes(table[w.parameters].to_numpy())
    for ph, nd in report.central_points(w).items():
        s = table.loc[node == nd, cols].mean().to_numpy()
        where = ", ".join(f"{p} = {v:g}" for p, v in zip(w.parameters, w.nodes.loc[nd, w.parameters]))
        ax.plot(centres[: len(s)], s, color=COL[ph], marker=report.PHASE_MARKERS[ph], ms=4, lw=1.6,
                label=f"phase {ph}, {names[ph]} ({where})")
    ax.axvline(1.0, color=INK_2, lw=0.8, ls=":")
    ax.text(1.03, ax.get_ylim()[1] * 0.92, "k = 1", color=INK_2, fontsize=7)
    ax.set_xscale("log")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    style(ax, "|k| (shell centre)", "S(k) / sum of S")
    fig.savefig(out / "sk_phases.pdf")


def fig_amplitude(table, out):
    """Standard deviation of u at g = 0 against eps, with the weakly nonlinear prediction."""
    t = table[np.isclose(table.g, 0.0)]
    m = t.groupby("eps")["u.mom.std"].agg(["mean", "min", "max"]).reset_index()
    fig, (ax,) = new_figure(3.6, 2.7)
    e = np.linspace(0, m.eps.max(), 200)
    ax.plot(e, np.sqrt(2 * e / 3), color=INK_2, lw=1.2, ls="--", label=r"$\sqrt{2\varepsilon/3}$ (theory)")
    ax.errorbar(m.eps, m["mean"], yerr=[m["mean"] - m["min"], m["max"] - m["mean"]], fmt="o", ms=4,
                color=COL[0], ecolor=COL[0], capsize=0, label="runs at g = 0 (mean, range)")
    ax.legend(fontsize=7, frameon=False)
    style(ax, r"$\varepsilon$", "standard deviation of u")
    fig.savefig(out / "amplitude.pdf")


def map_axes(ax, w, values, cmap, norm, title, cut_side=None, members=None):
    x, y = w.parameters
    nodes = w.nodes
    ax.scatter(nodes[x], nodes[y], s=10, c="#e4e3df", linewidths=0, zorder=1)
    sel = nodes.iloc[members] if members is not None else nodes
    sc = ax.scatter(sel[x], sel[y], s=26, c=values, cmap=cmap, norm=norm, linewidths=0, zorder=2)
    if cut_side is not None:
        ax.scatter(sel[x][cut_side], sel[y][cut_side], s=60, facecolors="none", edgecolors=INK,
                   linewidths=0.5, zorder=3)
    style(ax, x, y, title)
    return sc


def fig_fiedler(w, cuts, out):
    """Fiedler vector and normalised cut of the first two cuts tried on the demo."""
    fig, axes = new_figure(10.5, 3.2, 3)
    for ax, (i, c) in zip(axes[:2], enumerate(cuts[:2])):
        f = c["fiedler"] / np.abs(c["fiedler"]).max()
        sc = map_axes(ax, w, f, DIVERGING, TwoSlopeNorm(0, -1, 1),
                      f"Cut {i + 1}: Fiedler vector on {len(c['members'])} points", c["side"], c["members"])
    cb = fig.colorbar(sc, ax=list(axes[:2]), location="right", fraction=0.03, aspect=25)
    cb.set_label("Fiedler vector (scaled)", fontsize=7, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.outline.set_visible(False)
    ax = axes[2]
    for i, c in enumerate(cuts[:2]):
        k = np.arange(1, len(c["ncut"]) + 1)
        ax.plot(k, c["ncut"], color=COL[i], lw=1.5, label=f"cut {i + 1}")
        j = int(np.argmin(c["ncut"][walls.MIN_PHASE - 1:len(c["ncut"]) - walls.MIN_PHASE + 1])) + walls.MIN_PHASE - 1
        ax.plot(k[j], c["ncut"][j], "o", color=COL[i], ms=6)
    ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False)
    style(ax, "points on one side (Fiedler order)", "normalised cut", "Normalised cut along the order")
    fig.savefig(out / "fiedler.pdf")


def fig_tests(w, cuts, out):
    """log J2 of the crossing and inner links of every cut tried on the demo."""
    a, b = w.edges.a.to_numpy(), w.edges.b.to_numpy()
    _, _, log_test = weights_and_logs(w)
    fig, (ax,) = new_figure(6.4, 3.2)
    rng = np.random.default_rng(0)
    for i, c in enumerate(cuts):
        pos = walls._positions(c["members"], a, b)
        inside = np.flatnonzero((pos[a] >= 0) & (pos[b] >= 0))
        cross = np.isin(inside, c["crossing"])
        for k, (sel, colour) in enumerate([(~cross, "#b9b8b3"), (cross, COL[1] if c["accepted"] else INK_2)]):
            v = log_test[inside[sel]] / np.log(10)
            ax.scatter(i + (k - 0.5) * 0.36 + rng.uniform(-0.1, 0.1, len(v)), v, s=6, c=colour, linewidths=0)
            ax.plot([i + (k - 0.5) * 0.36 - 0.14, i + (k - 0.5) * 0.36 + 0.14], [np.median(v)] * 2,
                    color=INK, lw=1.2)
        verdict = "wall" if c["accepted"] else "no wall"
        ax.annotate(f"p = {c['p']:.1g}\nratio {c['ratio']:.2f}\n{verdict}", (i, 1.02), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(cuts)), [f"cut {i + 1}\n{len(c['members'])} pts" for i, c in enumerate(cuts)], fontsize=7)
    ax.scatter([], [], s=12, c="#b9b8b3", label="inner links")
    ax.scatter([], [], s=12, c=COL[1], label="crossing links (wall)")
    ax.scatter([], [], s=12, c=INK_2, label="crossing links (no wall)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    style(ax, "", r"$\log_{10} J_2$")
    fig.savefig(out / "tests.pdf")


def fig_selection(out):
    """First cut on pure noise: tested with the jumps that placed it (J1) or with J2."""
    p_same, p_other, r_same, r_other = [], [], [], []
    for seed in range(NOISE_SEEDS):
        t = table(lambda a, b: np.zeros(DIM), seed=seed)
        w = walls.find_walls(t, ["a", "b"])
        a, b = w.edges.a.to_numpy(), w.edges.b.to_numpy()
        weight, log_select, log_test = weights_and_logs(w)
        members = np.arange(len(w.nodes))
        side = walls.best_cut(members, a, b, weight)
        s1, s2 = walls.test_cut(side, members, a, b, log_select), walls.test_cut(side, members, a, b, log_test)
        p_same.append(s1["p"])
        p_other.append(s2["p"])
        r_same.append(s1["ratio"])
        r_other.append(s2["ratio"])
    fig, axes = new_figure(7.0, 2.8, 2)
    rng = np.random.default_rng(1)
    for ax, (vals, label, line) in zip(axes, [((p_same, p_other), r"$\log_{10} p$", np.log10(walls.ALPHA)),
                                              ((r_same, r_other), "ratio of medians", walls.RATIO)]):
        for k, (v, colour) in enumerate(zip(vals, (COL[1], COL[0]))):
            v = np.log10(v) if "p" in label else np.asarray(v)
            ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=10, c=colour, linewidths=0)
        ax.axhline(line, color=INK, lw=0.9, ls=":")
        ax.set_xticks([0, 1], ["tested with $J_1$\n(placed the cut)", "tested with $J_2$\n(other halves)"],
                      fontsize=7)
        ax.set_xlim(-0.6, 1.6)
        style(ax, "", label)
    axes[0].set_title(f"First cut on pure noise, {NOISE_SEEDS} tables", color=INK, fontsize=9, loc="left")
    axes[0].text(1.55, np.log10(walls.ALPHA), " ALPHA", fontsize=7, color=INK, va="bottom", ha="right")
    axes[1].text(1.55, walls.RATIO, " RATIO", fontsize=7, color=INK, va="bottom", ha="right")
    fig.savefig(out / "selection.pdf")
    n1 = sum(p < walls.ALPHA and r >= walls.RATIO for p, r in zip(p_same, r_same))
    n2 = sum(p < walls.ALPHA and r >= walls.RATIO for p, r in zip(p_other, r_other))
    return n1, n2


def fig_mixed(w, table, out):
    """Robust z of the within-point energy statistic on the demo, with the threshold."""
    X, _ = walls.feature_matrix(table)
    node, _ = parameter_nodes(table[w.parameters].to_numpy())
    run, time = table.run.to_numpy(), table.time.to_numpy(float)
    within = []
    for n in range(len(w.nodes)):
        i = np.flatnonzero(node == n)
        h = walls.halves(run[i], time[i])
        within.append(walls.energy_statistic(X[i[h]], X[i[~h]]))
    within = np.array(within)
    med = np.median(within)
    scale = max(walls.MAD_TO_STD * np.median(np.abs(within - med)), w.floor**2)
    z = (within - med) / scale
    fig, (ax,) = new_figure(3.6, 2.6)
    bins = np.linspace(min(z.min(), -3), max(z.max(), walls.MIXED_K) + 1, 40)
    ax.hist(z, bins=bins, color=COL[0], edgecolor="white", linewidth=0.5)
    ax.axvline(walls.MIXED_K, color=INK, lw=0.9, ls=":")
    ax.text(walls.MIXED_K, ax.get_ylim()[1] * 0.9, " MIXED_K", fontsize=7, color=INK)
    ax.set_yscale("log")
    style(ax, "robust z of the within-point statistic", "points")
    fig.savefig(out / "mixed.pdf")
    return int((z > walls.MIXED_K).sum())


def main(demo: str, out: str):
    demo, out = Path(demo), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ens = load_ensemble(demo / "phasewall.toml")
    t = pd.read_csv(demo / "descriptors.csv")
    w = walls.find_walls(t, ens.parameters)
    labels = node_labels(w, pd.read_csv(demo / "runs.csv"))
    names = {ph: labels[w.nodes.phase == ph].mode().iloc[0] for ph in sorted(w.nodes.phase.unique()) if ph >= 0}
    first = run_snapshots(ens, ens.runs.path.iloc[0])[-1][1]
    n = read_snapshot(first, ens.fields[:1], ens.julia_order)[ens.fields[0].name].shape[0]
    fig_sk(w, t, names, ens.dx, n, out)
    fig_amplitude(t, out)
    cuts = replay(w)
    fig_fiedler(w, cuts, out)
    fig_tests(w, cuts, out)
    n1, n2 = fig_selection(out)
    n_mixed = fig_mixed(w, t, out)
    lines = [f"phases {w.n_phases}; names {names}",
             f"eigenvalues of the first two cuts: {[np.round(c['eig'], 4).tolist() for c in cuts[:2]]}",
             f"cuts: {[(len(c['members']), int(c['side'].sum()), round(c['p'], 12), round(c['ratio'], 2), c['accepted']) for c in cuts]}",
             f"pure noise: first cut passes with J1 in {n1}/{NOISE_SEEDS}, with J2 in {n2}/{NOISE_SEEDS}",
             f"mixed points: {n_mixed}"]
    (out / "numbers.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:3])
