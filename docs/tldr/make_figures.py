"""Figures of the one-page summary (docs/TLDR.pdf) made from the simulations themselves.

Needs the local data (git-ignored): the Swift-Hohenberg demo in data/demo, the Barkley ensemble
and the trainings in data/cache (made by scripts/run_examples.py).
- fig_sh_grid.png: the input, last snapshots of one run at points of the (eps, g) grid;
- fig_sh_phases.png: the output, phases and walls of the learned description (default network,
  seed 0) with the last snapshot of one run at the central point of each phase;
- fig_barkley_mixed.png: two runs of the same parameter point that end in different states
  (rest and spirals).
Usage (from the repository): python docs/tldr/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from phasewall import report
from phasewall.io import load_ensemble, read_snapshot, run_snapshots
from phasewall.walls import find_walls

HERE = Path(__file__).parent
SH = ROOT / "data/demo/phasewall.toml"
BARKLEY = ROOT / "data/cache/ensemble-barkley-797261897c92/phasewall.toml"
DEFAULT = {"channels": [32, 64, 128, 256], "hidden": 256, "projection": 64, "temperature": 0.2,
           "views": "rotzoom", "seed": 0}
CMAP = "RdBu_r"
DPI = 200


def last_field(ens, run: int, name: str = "u") -> np.ndarray:
    path = ens.runs.set_index("run").loc[run, "path"]
    return read_snapshot(run_snapshots(ens, path)[-1][1], ens.fields, ens.julia_order)[name]


def run_at(ens, point: dict, seed: int | None = None) -> int:
    r = ens.runs
    m = np.ones(len(r), bool)
    for k, v in point.items():
        m &= np.isclose(r[k], v)
    if seed is not None:
        m &= r.seed.to_numpy() == seed
    return int(r[m].run.iloc[0])


def thumb(ax, a, lim, frame="#52514e", lw=1.5):
    ax.imshow(a.T, origin="lower", cmap=CMAP, vmin=-lim, vmax=lim, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(frame)
        s.set_linewidth(lw)


def sh_grid(ens):
    eps = [-0.025, 0.075, 0.175, 0.325, 0.475, 0.625]
    gs = [1.3, 0.9, 0.5, 0.1]
    fig, axes = plt.subplots(len(gs), len(eps), figsize=(6.0, 4.2))
    for i, g in enumerate(gs):
        for j, e in enumerate(eps):
            thumb(axes[i, j], last_field(ens, run_at(ens, {"eps": e, "g": g}, 0)), 1.2)
            if i == len(gs) - 1:
                axes[i, j].set_xlabel(f"{e:g}", fontsize=8)
            if j == 0:
                axes[i, j].set_ylabel(f"{g:g}", fontsize=8, rotation=0, labelpad=12, va="center")
    fig.supxlabel("ε (drive)", fontsize=9, y=0.02)
    fig.supylabel("g", fontsize=9, x=0.01)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.11, wspace=0.06, hspace=0.06)
    fig.savefig(HERE / "fig_sh_grid.png", dpi=DPI)


def learned_table():
    for d in sorted((ROOT / "data/cache").glob("learned-*")):
        meta = json.loads((d / "settings.json").read_text())
        st = meta["settings"]
        if all(st.get(k) == v for k, v in DEFAULT.items()) and (d / "table.pkl").exists():
            t = pd.read_pickle(d / "table.pkl")
            if "eps" in t and "g" in t and len(t) == 9600:
                return t
    raise SystemExit("no cached training of the default network on the Swift-Hohenberg demo")


def sh_phases(ens):
    w = find_walls(learned_table(), ["eps", "g"])
    ref = ens.runs.groupby(["eps", "g"]).reference.agg(lambda s: s.value_counts().idxmax())
    fig = plt.figure(figsize=(5.2, 5.6))
    ax = fig.add_axes([0.13, 0.08, 0.84, 0.56])
    report.plot_phases(ax, w)
    ax.set_title("", loc="left")
    ax.set_title("")
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel("ε", fontsize=10)
    ax.set_ylabel("g", fontsize=10)
    centre = report.central_points(w)
    order = sorted(centre, key=lambda ph: float(w.nodes.loc[centre[ph], "eps"]))      # left to right
    xs = np.linspace(0.22, 0.82, len(order))
    for x, ph in zip(xs, order):
        node = w.nodes.loc[centre[ph]]
        e, g = float(node.eps), float(node.g)
        name = ref.get((round(e, 9), round(g, 9)), "")
        a = last_field(ens, run_at(ens, {"eps": e, "g": g}, 0))
        img = OffsetImage(a.T[::-1], cmap=CMAP, zoom=0.9, norm=plt.Normalize(-1.2, 1.2))
        colour = report.PHASE_COLOURS[ph % len(report.PHASE_COLOURS)]
        ab = AnnotationBbox(img, (e, g), xybox=(x, 0.79), boxcoords="figure fraction",
                            bboxprops={"edgecolor": colour, "linewidth": 2.5},
                            arrowprops={"arrowstyle": "-", "color": colour, "lw": 1.5})
        ax.add_artist(ab)
        fig.text(x, 0.925, name, fontsize=10, color=colour, ha="center", fontweight="bold")
    fig.savefig(HERE / "fig_sh_phases.png", dpi=DPI)
    return w


def barkley_mixed(ens):
    """Two runs of one point: rest and spirals (the clearest spirals: most excited area, at most 75%)."""
    r = ens.runs
    by = r.groupby(["a", "b"]).reference.agg(lambda s: tuple(sorted(set(s))))
    best, point = -1.0, None
    for p in by[by == ("rest", "spirals")].index:
        here = r[np.isclose(r.a, p[0]) & np.isclose(r.b, p[1])]
        act = float((last_field(ens, int(here[here.reference == "spirals"].run.iloc[0])) > 0.5).mean())
        if act < 0.75 and act > best:
            best, point = act, p
    runs = r[np.isclose(r.a, point[0]) & np.isclose(r.b, point[1])]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
    for ax, lab in zip(axes, ("rest", "spirals")):
        run = int(runs[runs.reference == lab].run.iloc[0])
        ax.imshow(last_field(ens, run).T, origin="lower", cmap="Blues", vmin=0, vmax=1, interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#52514e")
            sp.set_linewidth(1.5)
        ax.set_title(f"seed {int(r.set_index('run').loc[run, 'seed'])}: {lab}", fontsize=8)
    fig.suptitle(f"same point (a = {point[0]:g}, b = {point[1]:g})", fontsize=8.5, y=0.99)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.03, wspace=0.08)
    fig.savefig(HERE / "fig_barkley_mixed.png", dpi=DPI)


if __name__ == "__main__":
    sh = load_ensemble(SH)
    sh_grid(sh)
    sh_phases(sh)
    barkley_mixed(load_ensemble(BARKLEY))
