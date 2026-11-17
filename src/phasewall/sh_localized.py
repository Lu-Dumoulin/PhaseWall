"""A fourth demo ensemble: localised states of the cubic-quintic Swift-Hohenberg equation,

    du/dt = eps u - (1 + lap)^2 u + b u^3 - u^5,

started from a localised patch of stripes instead of noise. For eps < 0 the uniform state is
stable and coexists with large-amplitude stripes; a front between them is pinned to the pattern
over a range of eps (homoclinic snaking; Burke and Knobloch, Phys. Rev. E 73, 056211, 2006: for
b = 2 in one dimension, -0.675 < eps < -0.604), where a patch stays localised (a dissipative
soliton). Below the range the patch disappears; above, it invades the box. The range moves with
b roughly as b^2, so the scan uses a = eps / b^2 and b (eps = a b^2). Each run starts from a stripe
patch of amplitude 1.4 (the upper branch; a smaller patch decays whatever eps), random radius
(2 to 5 wavelengths), orientation and place.

(With the quadratic-cubic equation of toy.py the pinning range of hexagon patches is narrower
than 0.005 in eps at g = 1: a pilot on 2026-11-17 found no localised state on a grid of 0.005.)

Numerics: pseudo-spectral, semi-implicit with a stabilising constant C (as toy.py), dt = 0.1.
reference_label: 'uniform', 'localized' (patterned share of the box between LOCAL_MIN and
LOCAL_MAX), 'extended'.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import toy
from .io import write_snapshot
from .parallel import run_jobs

N, WAVES = 128, 16       # a box of 16 wavelengths, so that a patch has room (a choice)
DT, C = 0.1, 4.0         # time step and stabilising constant (a choice; dt = 0.25 is unstable with u^5)
T_RELAX = 15_000.0       # fronts near the edges of the pinning range move slowly
EVERY, N_SAVE = 250.0, 6
AMPLITUDE = 1.4          # initial stripe amplitude (upper branch)
A = np.round(np.arange(-0.19, -0.1399, 0.0025), 4)   # eps / b^2
B = np.round(np.arange(1.8, 2.201, 0.1), 3)
SEEDS = tuple(range(4))
GRID = {"a": A, "b": B}
SETTINGS = {"n": N, "waves": WAVES, "dt": DT, "t_relax": T_RELAX, "every": EVERY, "n_save": N_SAVE}
N_FIELDS = 1
SECONDS_PER_POINT_STEP = 2.3e-8    # measured on a laptop (Apple M4 Max), one process
QUICK = {"a": A[::5], "b": B[::2], "n": 64, "waves": 8, "t_relax": 2000.0}
LOCAL_MIN, LOCAL_MAX = 0.02, 0.9     # patterned share of the box for a localised state (a choice)
ENVELOPE = 0.05                      # |u| averaged over a wavelength above this: patterned (a choice)


def seconds_per_run(**kw) -> float:
    s = SETTINGS | kw
    steps = (s["t_relax"] + s["every"] * (s["n_save"] - 1)) / s["dt"]
    return SECONDS_PER_POINT_STEP * steps * s["n"] ** 2


def initial(n: int, waves: int, seed: int, amplitude: float = AMPLITUDE) -> np.ndarray:
    """A stripe patch: amplitude cos(k . x) with |k| = 1 (on the grid) at a random angle,
    inside a smooth disc of random radius (2 to 5 wavelengths) at a random place."""
    rng = np.random.default_rng(seed)
    x = np.arange(n) / n * 2 * np.pi
    X, Y = np.meshgrid(x, x, indexing="ij")
    th = rng.uniform(0, np.pi)
    kx, ky = round(waves * np.cos(th)), round(waves * np.sin(th))
    radius = rng.uniform(2, 5) * n / waves
    cx, cy = rng.integers(0, n, 2)
    dxp = (np.arange(n)[:, None] - cx + n / 2) % n - n / 2
    dyp = (np.arange(n)[None, :] - cy + n / 2) % n - n / 2
    return amplitude * np.cos(kx * X + ky * Y) * np.exp(-(np.hypot(dxp, dyp) / radius) ** 8)


def simulate(a: float, b: float, seed: int, n: int = N, waves: int = WAVES, dt: float = DT,
             t_relax: float = T_RELAX, every: float = EVERY, n_save: int = N_SAVE) -> list[tuple[float, np.ndarray]]:
    """The saved snapshots (time, u indexed [x, y]) of one run; eps = a b^2."""
    eps = a * b * b
    _, k2 = toy.box(n, waves)
    implicit = 1 - dt * (eps - (1 - k2) ** 2 - C)
    uh = np.fft.fft2(initial(n, waves, seed))
    save = {round((t_relax + j * every) / dt): t_relax + j * every for j in range(n_save)}
    out = []
    for step in range(max(save) + 1):
        u = np.fft.ifft2(uh).real
        if step in save:
            out.append((save[step], u.copy()))
        uh = (uh + dt * np.fft.fft2(b * u**3 - u**5 + C * u)) / implicit
    return out


def patterned_share(u: np.ndarray, waves: int = WAVES) -> float:
    """Share of the box where |u| averaged over about one wavelength is above ENVELOPE."""
    size = max(3, round(u.shape[0] / waves))
    return float(np.mean(ndimage.uniform_filter(np.abs(u), size, mode="wrap") > ENVELOPE))


def reference_label(u: np.ndarray, waves: int = WAVES) -> str:
    """'uniform', 'localized' or 'extended'."""
    if u.std() < toy.UNIFORM_STD:
        return "uniform"
    return "localized" if LOCAL_MIN < patterned_share(u, waves) < LOCAL_MAX else "extended"


def _one(args):
    folder, run, a, b, seed, kw = args
    snaps = simulate(a, b, seed, **kw)
    for t, u in snaps:
        write_snapshot(folder / f"run{run:04d}" / "Data" / f"{round(t):010d}.h5", {"u": u})
    return reference_label(snaps[-1][1], kw.get("waves", WAVES))


def make_ensemble(folder: str | Path, a=A, b=B, seeds=SEEDS, workers: int = 1, progress=None, **kw) -> pd.DataFrame:
    """Write the ensemble into `folder` (runs.csv, phasewall.toml, one subfolder per run)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(a, b, seeds))
    jobs = [(folder, i, float(x), float(y), int(s), kw) for i, (x, y, s) in enumerate(grid)]
    labels = run_jobs(_one, jobs, workers, progress, "simulate")
    runs = pd.DataFrame({"run": range(len(grid)), "path": [f"run{i:04d}" for i in range(len(grid))],
                         "a": [x for x, _, _ in grid], "b": [y for _, y, _ in grid],
                         "seed": [s for _, _, s in grid], "reference": labels})
    runs.to_csv(folder / "runs.csv", index=False)
    dx, _ = toy.box(kw.get("n", N), kw.get("waves", WAVES))
    (folder / "phasewall.toml").write_text(
        "# localised (cubic-quintic) Swift-Hohenberg ensemble written by phasewall.sh_localized\n"
        "[ensemble]\n"
        'runs = "runs.csv"\n'
        'parameters = ["a", "b"]\n'
        'snapshots = "Data/*.h5"\n'
        "time_regex = '(\\d+)\\.h5$'\n"
        "time_unit = 1.0\n"
        "julia_order = true\n"
        f"dx = {dx!r}\n"
        "periodic = true\n"
        f"last = {kw.get('n_save', N_SAVE)}\n"
        f"pair_dt = {float(kw.get('every', EVERY))!r}\n\n"
        "[[fields]]\n"
        'name = "u"\n'
        'keys = ["u"]\n'
        'kind = "scalar"\n')
    return runs
