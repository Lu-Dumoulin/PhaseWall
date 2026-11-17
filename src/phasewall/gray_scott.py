"""A second demo ensemble: the Gray-Scott reaction-diffusion model on a grid of (F, k).

    du/dt = Du lap u - u v^2 + F (1 - u)
    dv/dt = Dv lap v + u v^2 - (F + k) v

u is a substrate fed at rate F, v an autocatalyst removed at rate F + k. In the (F, k) plane
the model shows (Pearson, Science 261, 189, 1993): the uniform state u = 1, v = 0, a second
uniform state, spots, stripes and labyrinths, holes (spots of the low-v state in a high-v
background), self-replicating spots, travelling waves and spatio-temporal chaos. Unlike
Swift-Hohenberg it is not a gradient flow: some states never stop moving.

Numerics as in Pearson 1993: Du = 2e-5, Dv = 1e-5, grid spacing 2.5 / 256, forward Euler with
dt = 1 and the five-point Laplacian, periodic box (Du dt / dx^2 = 0.21, below the stability
limit 0.25). Each run starts from u = 1, v = 0 with N_SEEDS squares of side SQUARE points at
random places set to u = 1/2, v = 1/4, plus 1% noise (Pearson used one central square; random
places make the seeds differ). Snapshots of u and v are written every `every` time units after
t_relax, laid out as Julia writes arrays.

reference_label gives an approximate state of a run from direct measurements (mean and spread
of v, connected domains of high and low v, change between the last two snapshots); it is a
coarse version of Pearson's classes, only used to check phasewall, never by it.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .io import write_snapshot
from .parallel import run_jobs

# Settings of the demo (choices)
N = 128                 # grid points per side (box 1.25, half of Pearson's)
DX = 2.5 / 256          # grid spacing (Pearson)
DU, DV = 2e-5, 1e-5     # diffusion coefficients (Pearson)
DT = 1.0                # time step (Pearson)
T_RELAX = 20_000.0      # time before the first saved snapshot (patterns still coarsen at 10 000 near k = 0.0625)
EVERY = 500.0           # time between saved snapshots
N_SAVE = 6              # saved snapshots per run
N_SQUARES, SQUARE = 4, 10    # initial perturbation: squares of 10 x 10 points at random places
F = np.round(np.arange(0.010, 0.0701, 0.005), 4)
K = np.round(np.arange(0.045, 0.0701, 0.0025), 4)
SEEDS = tuple(range(4))
GRID = {"F": F, "k": K}                           # default parameter grid
SETTINGS = {"n": N, "dx": DX, "dt": DT, "t_relax": T_RELAX, "every": EVERY, "n_save": N_SAVE}
N_FIELDS = 2          # fields per snapshot
SECONDS_PER_POINT_STEP = 9.5e-9   # measured on a laptop (Apple M4 Max), one process: for cost estimates
QUICK = {"F": F[::3], "k": K[::3], "n": 64, "t_relax": 5000.0}   # --quick: seconds, too coarse for all states
# reference_label thresholds (choices)
FLAT_STD = 1e-3         # std of v below this: uniform
LOW_V = 0.01            # mean v below this: the trivial uniform state u = 1, v = 0
MOVING = 0.1            # 1 - correlation of v between the last two snapshots above this: dynamic
COMPACT = 2.0           # domains with an aspect ratio below this (and small) are compact: spots or holes
COMPACT_SHARE = 0.7     # spots (or holes) when this share of the domains is compact
ISOLATED = 0.05         # spots covering less than this share of the box: isolated spots
MIN_AREA = 4            # domains smaller than this (points) are ignored


def laplacian(a: np.ndarray, dx: float = DX) -> np.ndarray:
    """Five-point Laplacian, periodic."""
    return (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a) / dx**2


def initial(n: int, seed: int, squares: int = N_SQUARES, side: int = SQUARE) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = np.ones((n, n)), np.zeros((n, n))
    for _ in range(squares):
        i, j = rng.integers(0, n, 2)
        rows, cols = np.arange(i, i + side) % n, np.arange(j, j + side) % n
        u[np.ix_(rows, cols)], v[np.ix_(rows, cols)] = 0.5, 0.25
    u += 0.01 * rng.standard_normal((n, n))
    v += 0.01 * rng.standard_normal((n, n))
    return u, v


def simulate(F: float, k: float, seed: int, n: int = N, dx: float = DX, dt: float = DT, t_relax: float = T_RELAX,
             every: float = EVERY, n_save: int = N_SAVE) -> list[tuple[float, dict[str, np.ndarray]]]:
    """The saved snapshots (time, {"u", "v"} indexed [x, y]) of one run."""
    u, v = initial(n, seed)
    save = {round((t_relax + j * every) / dt): t_relax + j * every for j in range(n_save)}
    out = []
    for step in range(max(save) + 1):
        if step in save:
            out.append((save[step], {"u": u.copy(), "v": v.copy()}))
        uvv = u * v * v
        u, v = (u + dt * (DU * laplacian(u, dx) - uvv + F * (1 - u)),
                v + dt * (DV * laplacian(v, dx) + uvv - (F + k) * v))
    return out


def seconds_per_run(**kw) -> float:
    """Rough time of one run in one process (seconds), from the measured cost per grid point
    and time step."""
    s = SETTINGS | kw
    steps = (s["t_relax"] + s["every"] * (s["n_save"] - 1)) / s["dt"]
    return SECONDS_PER_POINT_STEP * steps * s["n"] ** 2


def compact_share(mask: np.ndarray) -> tuple[int, float]:
    """Number of domains of a mask (at least MIN_AREA points) and the share of them that are
    compact: aspect ratio sqrt(l1 / l2) of the second moments of their points below COMPACT,
    and area below a tenth of the box. Spots and holes are compact; stripes and worms are
    elongated or large."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, 0.0
    idx = np.arange(1, n + 1)
    x, y = np.indices(mask.shape)
    area = ndimage.sum(mask, lab, idx)
    mx, my = ndimage.mean(x, lab, idx), ndimage.mean(y, lab, idx)
    cxx = ndimage.mean(x * x, lab, idx) - mx**2
    cyy = ndimage.mean(y * y, lab, idx) - my**2
    cxy = ndimage.mean(x * y, lab, idx) - mx * my
    tr, det = cxx + cyy, cxx * cyy - cxy**2
    disc = np.sqrt(np.maximum(tr**2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-9)
    keep = area >= MIN_AREA
    if not keep.any():
        return 0, 0.0
    compact = (np.sqrt(l1 / l2) < COMPACT) & (area < 0.1 * mask.size)
    return int(keep.sum()), float(np.mean(compact[keep]))


def reference_label(snaps: list[dict[str, np.ndarray]]) -> str:
    """Approximate state from the last two snapshots of a run: 'uniform', 'uniform (high v)',
    'dynamic', 'spots', 'isolated spots', 'holes' or 'stripes' (stripes, worms, labyrinths)."""
    v0, v = snaps[-2]["v"], snaps[-1]["v"]
    if v.std() < FLAT_STD:
        return "uniform" if v.mean() < LOW_V else "uniform (high v)"
    a, b = v0 - v0.mean(), v - v.mean()
    corr = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if 1 - corr > MOVING:
        return "dynamic"
    # centre the box on the largest v, so that a lone spot is not cut by the (periodic) edges
    i, j = np.unravel_index(np.argmax(v), v.shape)
    v = np.roll(v, (v.shape[0] // 2 - i, v.shape[1] // 2 - j), axis=(0, 1))
    high = v > v.mean()
    n_hi, share_hi = compact_share(high)
    n_lo, share_lo = compact_share(~high)
    if share_hi > COMPACT_SHARE and n_hi >= n_lo:
        return "isolated spots" if high.mean() < ISOLATED else "spots"
    if share_lo > COMPACT_SHARE and n_lo > n_hi:
        return "holes"
    return "stripes"


def _one(args):
    folder, run, F_, k_, seed, kw = args
    snaps = simulate(F_, k_, seed, **kw)
    for t, s in snaps:
        write_snapshot(folder / f"run{run:04d}" / "Data" / f"{round(t):010d}.h5", s)
    return reference_label([s for _, s in snaps])


def make_ensemble(folder: str | Path, F=F, k=K, seeds=SEEDS, workers: int = 1, progress=None, **kw) -> pd.DataFrame:
    """Write the Gray-Scott ensemble into `folder`: one subfolder per run, runs.csv (path, F,
    k, seed, reference label) and phasewall.toml. Returns the table of runs."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(F, k, seeds))
    jobs = [(folder, i, float(a), float(b), int(s), kw) for i, (a, b, s) in enumerate(grid)]
    labels = run_jobs(_one, jobs, workers, progress, "simulate")
    runs = pd.DataFrame({"run": range(len(grid)), "path": [f"run{i:04d}" for i in range(len(grid))],
                         "F": [a for a, _, _ in grid], "k": [b for _, b, _ in grid],
                         "seed": [s for _, _, s in grid], "reference": labels})
    runs.to_csv(folder / "runs.csv", index=False)
    (folder / "phasewall.toml").write_text(f"""# Gray-Scott demo ensemble written by phasewall.gray_scott
[ensemble]
runs = "runs.csv"
parameters = ["F", "k"]
snapshots = "Data/*.h5"
time_regex = '(\\d+)\\.h5$'
time_unit = 1.0
julia_order = true
dx = {kw.get("dx", DX)!r}
periodic = true
last = {kw.get("n_save", N_SAVE)}
pair_dt = {float(kw.get("every", EVERY))!r}

[[fields]]
name = "u"
keys = ["u"]
kind = "scalar"

[[fields]]
name = "v"
keys = ["v"]
kind = "scalar"
""")
    return runs
