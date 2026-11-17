"""A demo ensemble of excitable waves: the Barkley model on a grid of (a, b).

    du/dt = lap u + (1 / eps) u (1 - u) (u - (v + b) / a)
    dv/dt = u - v

(D. Barkley, Physica D 49, 61, 1991; phase diagram of spiral dynamics in Phys. Rev. Lett. 72,
164, 1994). u is the fast excitation, v the slow recovery (it does not diffuse). For eps = 0.02,
the (a, b) plane holds a medium where waves die (low excitability), rigidly rotating spirals,
meandering spirals, and spiral break-up. Every run starts from a broken wave (an excited half
plane next to a refractory half plane, at a random place and orientation, plus noise), from
which spirals form when the medium supports them. All its sustained states move; it tests the
dynamics descriptors.

Numerics as in Barkley 1991: the five-point Laplacian with forward Euler for diffusion and v,
and Barkley's semi-implicit update of the reaction term, which keeps u in [0, 1]:
u' = u / (1 - h (1 - u)(u - u_th)) if u < u_th, else (u + h u (u - u_th)) / (1 + h u (u - u_th)),
with h = dt / eps and u_th = (v + b) / a (plain forward Euler blows up at dt = 0.02). Periodic
box, dt below dx^2 / 4. Fast to solve (no FFT).

reference_label: 'rest' (no excitation left), 'plane waves' (activity without spiral tips),
'spirals' (at least one tip: a point around which the phase of (u, v) winds by 2 pi); rigid,
meandering and broken spirals are not told apart.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .gray_scott import laplacian
from .io import write_snapshot
from .parallel import run_jobs

EPS = 0.02              # time-scale ratio (Barkley 1994)
N, DX, DT = 128, 0.4, 0.02    # box 51.2; D dt / dx^2 = 0.125 (a choice, stable)
T_RELAX = 500.0         # many spiral rotations
EVERY, N_SAVE = 2.0, 6  # a fraction of a rotation period: rotating spirals decorrelate
A = np.round(np.arange(0.30, 1.001, 0.05), 3)
B = np.round(np.arange(0.0, 0.1001, 0.01), 3)
SEEDS = tuple(range(4))
GRID = {"a": A, "b": B}
SETTINGS = {"n": N, "dx": DX, "dt": DT, "t_relax": T_RELAX, "every": EVERY, "n_save": N_SAVE}
N_FIELDS = 2
SECONDS_PER_POINT_STEP = 6.0e-9    # measured on a laptop (Apple M4 Max), one process
QUICK = {"a": A[::4], "b": B[::3], "n": 64, "t_relax": 200.0}
REST = 0.05             # reference_label: max u below this: rest (a choice)


def initial(n: int, seed: int, a: float) -> tuple[np.ndarray, np.ndarray]:
    """A broken wave: u = 1 on a half plane, v = a / 2 (refractory) on the half plane turned by
    90 degrees, both through a random point at a random angle; plus 1% noise."""
    rng = np.random.default_rng(seed)
    x, y = np.indices((n, n)).astype(float)
    cx, cy = rng.uniform(0, n, 2)
    th = rng.uniform(0, 2 * np.pi)
    s = (x - cx) * np.cos(th) + (y - cy) * np.sin(th)
    t = -(x - cx) * np.sin(th) + (y - cy) * np.cos(th)
    u = ((s > 0) & (np.abs(s) < n / 4)).astype(float)
    v = np.where((t > 0) & (np.abs(t) < n / 4), a / 2, 0.0)
    return u + 0.01 * rng.standard_normal((n, n)), v


def react(u: np.ndarray, uth: np.ndarray, h: float) -> np.ndarray:
    """Barkley's semi-implicit step of du/dt = u (1 - u) (u - uth) / eps (h = dt / eps)."""
    low = u < uth
    out = np.empty_like(u)
    ul, th = u[low], uth[low]
    out[low] = ul / (1 - h * (1 - ul) * (ul - th))
    uh, th = u[~low], uth[~low]
    f = h * uh * (uh - th)
    out[~low] = (uh + f) / (1 + f)
    return out


def simulate(a: float, b: float, seed: int, n: int = N, dx: float = DX, dt: float = DT, t_relax: float = T_RELAX,
             every: float = EVERY, n_save: int = N_SAVE, eps: float = EPS) -> list[tuple[float, dict[str, np.ndarray]]]:
    """The saved snapshots (time, {"u", "v"} indexed [x, y]) of one run."""
    u, v = initial(n, seed, a)
    save = {round((t_relax + j * every) / dt): t_relax + j * every for j in range(n_save)}
    out = []
    for step in range(max(save) + 1):
        if step in save:
            out.append((save[step], {"u": u.copy(), "v": v.copy()}))
        u, v = react(u + dt * laplacian(u, dx), (v + b) / a, dt / eps), v + dt * (u - v)
    return out


def seconds_per_run(**kw) -> float:
    s = SETTINGS | kw
    steps = (s["t_relax"] + s["every"] * (s["n_save"] - 1)) / s["dt"]
    return SECONDS_PER_POINT_STEP * steps * s["n"] ** 2


def tips(u: np.ndarray, v: np.ndarray) -> int:
    """Number of spiral tips: plaquettes of the periodic grid around which the phase
    atan2(v - <v>, u - <u>) winds by +-2 pi."""
    phi = np.arctan2(v - v.mean(), u - u.mean())

    def d(a, b):
        return np.angle(np.exp(1j * (b - a)))
    right, up = np.roll(phi, -1, 0), np.roll(phi, -1, 1)
    diag = np.roll(right, -1, 1)
    wind = d(phi, right) + d(right, diag) + d(diag, up) + d(up, phi)
    return int(np.sum(np.abs(wind) > np.pi))


def reference_label(snaps: list[dict[str, np.ndarray]]) -> str:
    """'rest', 'plane waves' or 'spirals' (see the module docstring)."""
    u, v = snaps[-1]["u"], snaps[-1]["v"]
    if u.max() < REST:
        return "rest"
    return "spirals" if tips(u, v) > 0 else "plane waves"


def _one(args):
    folder, run, a, b, seed, kw = args
    snaps = simulate(a, b, seed, **kw)
    for t, s in snaps:
        write_snapshot(folder / f"run{run:04d}" / "Data" / f"{round(t * 100):010d}.h5", s)
    return reference_label([s for _, s in snaps])


def make_ensemble(folder: str | Path, a=A, b=B, seeds=SEEDS, workers: int = 1, progress=None, **kw) -> pd.DataFrame:
    """Write the Barkley ensemble into `folder` (runs.csv, phasewall.toml, one subfolder per run).
    File names hold the time in hundredths (time_unit = 0.01)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(a, b, seeds))
    jobs = [(folder, i, float(x), float(y), int(s), kw) for i, (x, y, s) in enumerate(grid)]
    labels = run_jobs(_one, jobs, workers, progress, "simulate")
    runs = pd.DataFrame({"run": range(len(grid)), "path": [f"run{i:04d}" for i in range(len(grid))],
                         "a": [x for x, _, _ in grid], "b": [y for _, y, _ in grid],
                         "seed": [s for _, _, s in grid], "reference": labels})
    runs.to_csv(folder / "runs.csv", index=False)
    (folder / "phasewall.toml").write_text(
        "# Barkley demo ensemble written by phasewall.barkley\n"
        "[ensemble]\n"
        'runs = "runs.csv"\n'
        'parameters = ["a", "b"]\n'
        'snapshots = "Data/*.h5"\n'
        "time_regex = '(\\d+)\\.h5$'\n"
        "time_unit = 0.01\n"
        "julia_order = true\n"
        f"dx = {kw.get('dx', DX)!r}\n"
        "periodic = true\n"
        f"last = {kw.get('n_save', N_SAVE)}\n"
        f"pair_dt = {float(kw.get('every', EVERY))!r}\n\n"
        "[[fields]]\n"
        'name = "u"\n'
        'keys = ["u"]\n'
        'kind = "scalar"\n\n'
        "[[fields]]\n"
        'name = "v"\n'
        'keys = ["v"]\n'
        'kind = "scalar"\n')
    return runs
