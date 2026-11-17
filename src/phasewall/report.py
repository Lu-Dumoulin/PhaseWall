"""Report of a wall search: tables, a short summary in Markdown and figures.

Files written by write_report into one folder:

- nodes.csv, edges.csv, cuts.csv: the tables of walls.Walls (one row per parameter point,
  per link, per cut tried).
- robustness.csv: number of phases against RATIO (walls.phases_against_ratio).
- summary.md: phases with their extent in parameter space, walls per parameter, cuts,
  mixed points, robustness.
- phase_map.png (scans of two parameters only): left, every link coloured by its jump;
  middle, the phases found with the walls drawn across the cut links; right, reference
  labels when they are given (the demo).
- robustness.png: the robustness curve.
- snapshots.png (when the ensemble is given): one snapshot from inside each phase.

Figures use matplotlib's object interface (Figure, no pyplot), so nothing needs a screen.
Colours: one fixed colour and marker per phase, in the same order every time; boundary
regions (fewer than MIN_PHASE points) in grey.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from .fields import Field
from .io import Ensemble, read_snapshot, run_snapshots
from .walls import Walls

PHASE_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
PHASE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
OTHER = "#8a8985"          # boundary regions, and phases beyond the eighth
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
JUMPS = LinearSegmentedColormap.from_list("jumps", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
FIELD = LinearSegmentedColormap.from_list("field", ["#184f95", "#f0efec", "#b23a39"])
DPI = 150


# ---------------------------------------------------------------- tables and summary
def phase_extent(w: Walls) -> pd.DataFrame:
    """One row per phase: number of points and the range of every parameter."""
    rows = []
    for ph, g in w.nodes[w.nodes.phase >= 0].groupby("phase"):
        row = {"phase": int(ph), "points": len(g), "runs": int(g.runs.sum())}
        for p in w.parameters:
            row[p] = f"{g[p].min():g} to {g[p].max():g}"
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown(df: pd.DataFrame) -> str:
    """A small table in Markdown (pandas' to_markdown needs an extra package)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def summary(w: Walls, curve: pd.DataFrame | None = None, notes: str = "") -> str:
    """The summary of a wall search, in Markdown."""
    s = w.settings
    walls_per_param = w.edges[w.edges.wall].groupby("param").size()
    mixed = w.nodes[w.nodes.mixed]
    parts = [
        "# phasewall result",
        "",
        (f"{w.n_phases} phases on {len(w.nodes)} parameter points ({', '.join(w.parameters)}), "
         f"{int(w.nodes.runs.sum())} runs, {int(w.nodes.snapshots.sum())} snapshots, "
         f"{len(w.features)} descriptors."),
        "",
        (f"Settings: ALPHA = {s.get('alpha')}, RATIO = {s.get('ratio')}, MIN_PHASE = {s.get('min_phase')}. "
         f"Jump floor {w.floor:.3g}, median spread of one point {w.noise:.3g}."),
        "",
        "## Phases",
        "",
        _markdown(phase_extent(w)) if w.n_phases else "No phase.",
        "",
        f"Points in boundary regions (fewer than MIN_PHASE points): {int((w.nodes.phase < 0).sum())}.",
        "",
        "## Walls",
        "",
        f"{int(w.edges.wall.sum())} of {len(w.edges)} links are walls"
        + (": " + ", ".join(f"{n} across {p}" for p, n in walls_per_param.items()) if len(walls_per_param) else "")
        + ".",
        "",
        "Cuts tried, in order (each accepted cut splits a region in two, then both sides are tried again):",
        "",
        _markdown(w.cuts) if len(w.cuts) else "No cut could be tried.",
        "",
        "## Mixed points",
        "",
        "Points whose two halves differ far more than usual (bistability, or runs not yet steady):",
        "",
        _markdown(mixed[[*w.parameters, "runs", "spread", "phase"]]) if len(mixed) else "None.",
        "",
    ]
    if curve is not None:
        parts += ["## Robustness", "",
                  "Number of phases when RATIO changes (a count that holds over a range of ratios is robust):",
                  "", _markdown(curve), ""]
    if notes:
        parts += [notes, ""]
    return "\n".join(parts)


# ---------------------------------------------------------------- figures
def _style(ax, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)


def _title(ax, text: str) -> None:
    ax.set_title(text, color=INK, fontsize=10, loc="left")


def _steps(w: Walls) -> np.ndarray:
    """Typical step of each parameter: the median length of the links along it."""
    pts = w.nodes[w.parameters].to_numpy()
    steps = np.ones(len(w.parameters))
    for k in range(len(w.parameters)):
        e = w.edges[w.edges.axis == k]
        if len(e):
            steps[k] = float(np.median(np.abs(pts[e.b, k] - pts[e.a, k])))
    return steps


def wall_segments(w: Walls) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """For a scan of two parameters: one segment per wall link, through the middle of the
    link and across it, one step of the other parameter long."""
    pts = w.nodes[w.parameters].to_numpy()
    steps = _steps(w)
    segs = []
    for e in w.edges[w.edges.wall].itertuples():
        mid = (pts[e.a] + pts[e.b]) / 2
        other = 1 - e.axis
        lo, hi = mid.copy(), mid.copy()
        lo[other] -= steps[other] / 2
        hi[other] += steps[other] / 2
        segs.append((tuple(lo), tuple(hi)))
    return segs


def _points_by_class(ax, x, y, classes, names) -> None:
    """Scatter the points of each class with its fixed colour and marker; class < 0 grey."""
    for c in sorted(set(classes)):
        m = classes == c
        if 0 <= c < len(PHASE_COLOURS):
            colour, marker = PHASE_COLOURS[c], PHASE_MARKERS[c]
        else:
            colour, marker = OTHER, "."
        ax.scatter(x[m], y[m], s=26, c=colour, marker=marker, linewidths=0, label=names(c), zorder=3)


def _legend(ax) -> None:
    n = len(ax.get_legend_handles_labels()[0])
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=n if n <= 5 else 3, fontsize=8,
                    frameon=False, handletextpad=0.2, columnspacing=0.9)
    for t in leg.get_texts():
        t.set_color(INK)


def plot_jumps(ax, w: Walls) -> None:
    """Every link coloured by its contrast J / noise (log scale)."""
    x, y = w.parameters
    pts = w.nodes[[x, y]].to_numpy()
    lines = [(pts[a], pts[b]) for a, b in zip(w.edges.a, w.edges.b)]
    c = np.log10(w.edges.contrast.to_numpy())
    lc = LineCollection(lines, cmap=JUMPS, norm=Normalize(c.min(), c.max()), linewidths=2.2, capstyle="round")
    lc.set_array(c)
    ax.add_collection(lc)
    ax.scatter(pts[:, 0], pts[:, 1], s=4, c=INK_2, zorder=3, linewidths=0)
    cb = ax.figure.colorbar(lc, ax=ax, location="right", pad=0.02, fraction=0.05, aspect=30, shrink=0.8)
    cb.set_label("log10 jump / median spread", color=INK, fontsize=8)
    cb.ax.tick_params(colors=INK_2, labelsize=7)
    cb.outline.set_visible(False)
    _style(ax, x, y)
    _title(ax, "Jump across every link")


def plot_phases(ax, w: Walls) -> None:
    """Phases found, with the walls and the mixed points."""
    x, y = w.parameters
    n = w.nodes
    _points_by_class(ax, n[x].to_numpy(), n[y].to_numpy(), n.phase.to_numpy(),
                     lambda c: f"phase {c}" if c >= 0 else "boundary")
    ax.add_collection(LineCollection(wall_segments(w), colors=INK, linewidths=1.6, capstyle="round", zorder=4))
    m = n[n.mixed]
    if len(m):
        ax.scatter(m[x], m[y], s=90, facecolors="none", edgecolors=INK, linewidths=1.0, label="mixed", zorder=5)
    ax.plot([], [], color=INK, lw=1.6, label="wall")
    _style(ax, x, y)
    _title(ax, f"Phases found: {w.n_phases}")
    _legend(ax)


def plot_reference(ax, w: Walls, reference: pd.Series) -> None:
    """Reference label of each point (the most common label of its runs), for checking."""
    x, y = w.parameters
    names = list(dict.fromkeys(reference.loc[w.nodes.index]))
    # colour each reference label like the phase that holds most of its points
    best = {lab: int(w.nodes.phase[reference == lab].mode().iloc[0]) for lab in names}
    classes = np.array([best[lab] for lab in reference.loc[w.nodes.index]])
    label_of = {best[lab]: lab for lab in names}
    _points_by_class(ax, w.nodes[x].to_numpy(), w.nodes[y].to_numpy(), classes, lambda c: label_of.get(c, "?"))
    ax.add_collection(LineCollection(wall_segments(w), colors=INK, linewidths=1.0, alpha=0.35, zorder=4))
    _style(ax, x, y)
    _title(ax, "Reference labels (walls found in grey)")
    _legend(ax)


def phase_map_figure(w: Walls, path: str | Path, reference: pd.Series | None = None) -> None:
    """phase_map.png for a scan of two parameters."""
    if len(w.parameters) != 2:
        raise ValueError("a phase map needs a scan of exactly two parameters")
    panels = 3 if reference is not None else 2
    fig = Figure(figsize=(4.4 * panels, 4.6), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, panels)
    plot_jumps(axes[0], w)
    plot_phases(axes[1], w)
    if reference is not None:
        plot_reference(axes[2], w, reference)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)


def robustness_figure(curve: pd.DataFrame, path: str | Path, ratio: float | None = None) -> None:
    """robustness.png: number of phases against RATIO."""
    fig = Figure(figsize=(4.4, 3.0), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    ax.plot(curve.ratio, curve.phases, color=PHASE_COLOURS[0], lw=2, marker="o", ms=6, zorder=3)
    if ratio is not None:
        ax.axvline(ratio, color=INK_2, lw=1, ls=":", zorder=2)
        ax.text(ratio, curve.phases.max() + 0.3, " RATIO used", color=INK_2, fontsize=8, va="bottom")
    ax.set_ylim(0, curve.phases.max() + 1)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax, "RATIO (crossing jump / inner jump needed)", "phases")
    _title(ax, "Number of phases against the threshold")
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)


def _picture(a: np.ndarray, f: Field) -> np.ndarray:
    """What is shown of a field: the field (scalar), |a| (vector) or 2|Q| (nematic)."""
    if f.kind == "scalar":
        return a
    size = np.hypot(a[..., 0], a[..., 1])
    return 2 * size if f.kind == "nematic" else size


def central_points(w: Walls) -> dict[int, int]:
    """For each phase, the node closest to the middle of the phase (parameters scaled by
    their step), as an example of the phase away from its walls."""
    pts = w.nodes[w.parameters].to_numpy() / _steps(w)
    out = {}
    for ph in sorted(w.nodes.phase[w.nodes.phase >= 0].unique()):
        idx = np.flatnonzero(w.nodes.phase.to_numpy() == ph)
        centre = np.median(pts[idx], axis=0)
        out[int(ph)] = int(idx[np.argmin(np.linalg.norm(pts[idx] - centre, axis=1))])
    return out


def snapshots_figure(w: Walls, ens: Ensemble, path: str | Path, field: Field | None = None) -> None:
    """snapshots.png: the last snapshot of one run at the central point of each phase, all
    on one colour scale so that amplitudes can be compared (a uniform state looks flat)."""
    f = field or ens.fields[0]
    picks = central_points(w)
    runs = ens.runs.set_index("run")
    params = runs[ens.parameters].round(9)
    images = {}
    for ph, node in picks.items():
        values = w.nodes.loc[node, ens.parameters].to_numpy(dtype=float).round(9)
        run = params.index[(params.to_numpy() == values).all(axis=1)][0]
        _, file = run_snapshots(ens, runs.loc[run, "path"])[-1]
        images[ph] = (values, _picture(read_snapshot(file, [f], ens.julia_order)[f.name], f))
    if f.kind == "scalar":
        lim = max(np.abs(img).max() for _, img in images.values()) or 1.0
        cmap, vmin, vmax = FIELD, -lim, lim
    else:
        cmap, vmin, vmax = JUMPS, 0.0, max(img.max() for _, img in images.values()) or 1.0
    fig = Figure(figsize=(2.6 * max(len(picks), 1) + 0.6, 3.0), facecolor=SURFACE, layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, max(len(picks), 1)))
    for ax, (ph, (values, img)) in zip(axes, images.items()):
        im = ax.imshow(img.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(PHASE_COLOURS[ph] if ph < len(PHASE_COLOURS) else OTHER)
            side.set_linewidth(3)
        where = ", ".join(f"{p} = {v:g}" for p, v in zip(ens.parameters, values))
        _title(ax, f"phase {ph}\n{where}")
    if images:
        cb = fig.colorbar(im, ax=list(axes), location="right", fraction=0.04, aspect=25)
        cb.set_label({"scalar": f.name, "vector": f"|{f.name}|", "nematic": f"2|{f.name}|"}[f.kind],
                     color=INK, fontsize=8)
        cb.ax.tick_params(colors=INK_2, labelsize=7)
        cb.outline.set_visible(False)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)


# ---------------------------------------------------------------- everything
def write_report(w: Walls, out: str | Path, curve: pd.DataFrame | None = None,
                 reference: pd.Series | None = None, ens: Ensemble | None = None, notes: str = "") -> Path:
    """Write the tables, summary.md and the figures into the folder `out`; returns it.
    reference: a label per node (index of w.nodes), drawn next to the phases found."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    w.nodes.to_csv(out / "nodes.csv", index_label="node")
    w.edges.to_csv(out / "edges.csv", index=False)
    w.cuts.to_csv(out / "cuts.csv", index=False)
    if curve is not None:
        curve.to_csv(out / "robustness.csv", index=False)
        robustness_figure(curve, out / "robustness.png", w.settings.get("ratio"))
    if len(w.parameters) == 2:
        phase_map_figure(w, out / "phase_map.png", reference)
    if ens is not None and w.n_phases:
        snapshots_figure(w, ens, out / "snapshots.png")
    (out / "summary.md").write_text(summary(w, curve, notes))
    return out
