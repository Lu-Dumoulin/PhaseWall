"""Snapshot figures of every example for the documentation, from the cached runs of
scripts/run_examples.py (walls on the hand-made description, default settings):

- states.png:   one snapshot per reference state (a run carrying that label, at the point where
                the most runs carry it, nearest the middle of those points);
- phases.png:   one snapshot per phase found (report.make_snapshots: central point of each phase);
- mistakes.csv: every point whose reference label is not the most common label of its region
                (the points not counted in `agree`), with the labels of its runs;
- mistakes.png: the snapshot of each of them, with its reference label and the phase found; when
                there are more than MAX_SHOWN, MAX_SHOWN of them, taken in turn from each kind of
                mistake (reference label, phase found), the most common kinds first.

Usage: python scripts/doc_examples.py [--only swift-hohenberg ...] [--workers 8]
Every figure shows the field the reference labels are made from (FIELD), on one colour scale per
figure. The reference labels are used only to choose and caption the pictures.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent))
from run_examples import JOBS, OUT

from phasewall import pipeline as pl
from phasewall import report
from phasewall.io import read_snapshot, run_snapshots

warnings.simplefilter("ignore", RuntimeWarning)
MAX_SHOWN = 10          # mistakes shown as snapshots when there are at most this many (a choice)
FIELD = {"gray-scott": "v"}     # field shown; default the first field of the ensemble


def last_snapshot(ens, run: int, field) -> np.ndarray:
    runs = ens.runs.set_index("run")
    _, file = run_snapshots(ens, runs.loc[run, "path"])[-1]
    return report._picture(read_snapshot(file, [field], ens.julia_order)[field.name], field)


def panels(images: list[tuple[str, np.ndarray, str]], field, columns: int = 5) -> Figure:
    """A grid of snapshots on one colour scale; each (title, image, frame colour)."""
    lo, hi = min(i.min() for _, i, _ in images), max(i.max() for _, i, _ in images)
    if field.symmetry == "scalar" and lo < 0 < hi:
        cmap, vmin, vmax = report.FIELD, -max(-lo, hi), max(-lo, hi)
    else:
        cmap, vmin, vmax = report.JUMPS, lo, hi if hi > lo else lo + 1.0
    cols = min(columns, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig = Figure(figsize=(2.5 * cols + 0.7, 2.95 * rows), facecolor=report.SURFACE, layout="constrained")
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ax in axes[len(images):]:
        ax.set_visible(False)
    for ax, (title, img, colour) in zip(axes, images):
        im = ax.imshow(img.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(colour)
            side.set_linewidth(3)
        ax.set_title(title, fontsize=7.5, color=report.INK, loc="left")
    cb = fig.colorbar(im, ax=list(axes[:len(images)]), location="right", fraction=0.03, aspect=30)
    cb.set_label({"scalar": field.name, "vector": f"|{field.name}|", "nematic": f"2|{field.name}|"}[field.symmetry],
                 color=report.INK, fontsize=8, rotation=0, labelpad=8)
    cb.ax.tick_params(colors=report.INK_2, labelsize=7)
    cb.outline.set_visible(False)
    return fig


def where(params, values) -> str:
    return ", ".join(f"{p} = {v:.4g}" for p, v in zip(params, values))


def sample(table: pd.DataFrame, n: int = MAX_SHOWN) -> pd.DataFrame:
    """All rows if there are at most n; otherwise n rows taken in turn from each kind of mistake
    (reference label, phase found), the most common kinds first."""
    if len(table) <= n:
        return table
    kinds = sorted(table.groupby(["reference", "phase"]).groups.values(), key=len, reverse=True)
    picks, depth = [], 0
    while len(picks) < n:
        picks += [idx[depth] for idx in kinds if depth < len(idx)][: n - len(picks)]
        depth += 1
    return table.loc[picks]


def run_one(name: str, workers: int) -> dict:
    settings = pl.Settings(source=JOBS[name],
                           describers=pl.Describers(hand=True, learned=False, combined=False),
                           methods=pl.Methods(walls=True, robustness=False, mixture=False, kmeans=False, hdbscan=(),
                                              probe=False),
                           execution=pl.Execution(workers=workers, cache="data/cache"))
    res = pl.run(settings)
    ens, w, params = res.ensemble, res.walls["hand"], list(res.parameters)
    field = next(f for f in ens.fields if f.name == FIELD.get(name, ens.fields[0].name))
    runs = res.runs.assign(reference=res.labels.loc[res.runs.run].to_numpy())
    runs["key"] = runs[params].round(9).apply(tuple, axis=1)
    out = OUT / name

    # one snapshot per reference state
    steps = report._steps(w)
    images = []
    for label, group in runs.groupby("reference"):
        count = group.groupby("key").size()
        best = count[count == count.max()].index
        pts = np.array(best.tolist(), dtype=float) / steps
        centre = np.median(np.array(group.key.tolist(), dtype=float) / steps, axis=0)
        point = best[int(np.argmin(np.linalg.norm(pts - centre, axis=1)))]
        run = int(group[group.key == point].run.iloc[0])
        images.append((f"{label}\n{where(params, point)}", last_snapshot(ens, run, field), report.INK_2))
    panels(images, field).savefig(out / "states.png", dpi=report.DPI, facecolor=report.SURFACE)
    report.make_snapshots(w, ens, field).savefig(out / "phases.png", dpi=report.DPI, facecolor=report.SURFACE)

    # points not counted in `agree`: label differs from the most common label of their region
    point_label = report.node_labels(w, runs)
    region_label = point_label.groupby(w.nodes.region).agg(lambda s: s.value_counts().idxmax())
    wrong = w.nodes[point_label.to_numpy() != region_label.loc[w.nodes.region].to_numpy()]
    rows = []
    for i, node in wrong.iterrows():
        k = tuple(node[params].to_numpy(dtype=float).round(9))
        here = runs[runs.key == k]
        rows.append({**{p: node[p] for p in params}, "reference": point_label.loc[i],
                     "runs": ", ".join(f"{lab} {n}" for lab, n in here.reference.value_counts().items()),
                     "phase": int(node.phase), "phase_label": region_label.loc[node.region],
                     "run_shown": int(here[here.reference == point_label.loc[i]].run.iloc[0])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "mistakes.csv", index=False)
    (out / "mistakes.png").unlink(missing_ok=True)
    if len(table):
        shown = sample(table)
        images = []
        for r in shown.itertuples():
            found = f"phase {r.phase} ({r.phase_label})" if r.phase >= 0 else f"boundary region ({r.phase_label})"
            colour = report.PHASE_COLOURS[r.phase] if 0 <= r.phase < len(report.PHASE_COLOURS) else report.OTHER
            title = (f"{where(params, [getattr(r, p) for p in params])}\nreference: {r.reference}\n"
                     f"runs: {r.runs}\nfound: {found}")
            images.append((title, last_snapshot(ens, r.run_shown, field), colour))
        panels(images, field).savefig(out / "mistakes.png", dpi=report.DPI, facecolor=report.SURFACE)
    print(f"{name}: {runs.reference.nunique()} states, {w.n_phases} phases, {len(table)} mistaken points")
    return {"example": name, "states": runs.reference.nunique(), "phases": w.n_phases, "mistakes": len(table),
            "shown": min(len(table), MAX_SHOWN)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(JOBS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    print(pd.DataFrame([run_one(n, args.workers) for n in args.only]).to_string(index=False))


if __name__ == "__main__":
    main()
