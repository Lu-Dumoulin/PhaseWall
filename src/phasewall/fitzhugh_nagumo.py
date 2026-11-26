"""A third demo ensemble: the FitzHugh-Nagumo reaction-diffusion model on a grid of (eps, delta).

    du/dt = u - u^3 - v + lap u
    dv/dt = eps (u - a1 v - a0) + delta lap v

in the form studied by Hagberg and Meron (Nonlinearity 7, 805, 1994; Chaos 4, 477, 1994). With
a1 = 2 and a0 = 0 the medium is bistable and symmetric under (u, v) -> (-u, -v): two equivalent
uniform states, up and down. In the scanned (eps, delta) plane (pilot, 2026-11-17): stationary
straight fronts (large eps, small delta), moving domains and front turbulence (small eps and
delta), uniform states (a band in between, up or down at random), Turing labyrinths, rings and
spots (large delta). Unlike Swift-Hohenberg it has states that never stop moving. (With
a0 = -0.1 the down state invades everything at small delta and nothing moves.)

Numerics: pseudo-spectral, semi-implicit (diffusion implicit, reaction explicit), periodic box of
side L = n dx. Each run starts from a smooth random field (Fourier modes up to wavenumber
INIT_K, amplitude INIT_AMP) for u and v = 0, so that domains of both states meet; the seed sets
the field. Snapshots of u and v are written every `every` time units after t_relax, laid out as
Julia writes arrays.

reference_label (approximate, from the last two snapshots of u; only to check phasewall):
'uniform' (up or down, equivalent by symmetry), 'dynamic', 'fronts' (at most two domains),
'patterns' (stationary, three domains or more). Spots, holes and labyrinths are one label: with
a0 = 0, spots of u > 0 and holes of u < 0 are images of each other under the symmetry, and in this
box one run often holds round domains and worms side by side (until 2026-11-25 they were three
labels, set by a threshold on the share of compact domains, which split one state).
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .describe import count_domains
from .io import write_snapshot
from .parallel import run_jobs

# Settings of the demo (choices)
A1, A0 = 2.0, 0.0       # bistable symmetric medium (Hagberg and Meron)
N = 128                 # grid points per side
DX = 0.5                # grid spacing (box 64)
DT = 0.05               # time step
T_RELAX = 3000.0        # time before the first saved snapshot (labyrinths still coarsen at 1000)
EVERY = 50.0            # time between saved snapshots
N_SAVE = 6
INIT_K, INIT_AMP = 6, 0.5    # initial smooth random field: modes up to 6 (2 pi / L), amplitude 0.5
EPS = np.round(np.geomspace(0.005, 0.5, 11), 4)
DELTA = np.round(np.geomspace(0.05, 8.0, 9), 4)
SEEDS = tuple(range(4))
GRID = {"eps": EPS, "delta": DELTA}
SETTINGS = {"n": N, "dx": DX, "dt": DT, "t_relax": T_RELAX, "every": EVERY, "n_save": N_SAVE}
N_FIELDS = 2
SECONDS_PER_POINT_STEP = 3.0e-8    # measured on a laptop (Apple M4 Max), one process: for cost estimates
QUICK = {"eps": EPS[::4], "delta": DELTA[::3], "n": 64, "t_relax": 500.0}
# reference_label thresholds (choices)
FLAT_STD = 1e-3
MOVING = 0.1            # 1 - correlation of u between the last two snapshots above this: dynamic


def wavenumbers(n: int, dx: float) -> np.ndarray:
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    return k[:, None] ** 2 + k[None, :] ** 2


def initial(n: int, dx: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    m = np.hypot(*np.meshgrid(np.fft.fftfreq(n, 1 / n), np.fft.fftfreq(n, 1 / n), indexing="ij"))
    u = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * (m <= INIT_K)).real
    u = INIT_AMP * u / (u.std() or 1.0)
    return u, np.zeros((n, n))


def simulate(eps: float, delta: float, seed: int, n: int = N, dx: float = DX, dt: float = DT,
             t_relax: float = T_RELAX, every: float = EVERY, n_save: int = N_SAVE, a0: float = A0,
             a1: float = A1) -> list[tuple[float, dict[str, np.ndarray]]]:
    """The saved snapshots (time, {"u", "v"} indexed [x, y]) of one run."""
    k2 = wavenumbers(n, dx)
    iu, iv = 1 / (1 + dt * k2), 1 / (1 + dt * delta * k2)
    u, v = initial(n, dx, seed)
    uh, vh = np.fft.fft2(u), np.fft.fft2(v)
    save = {round((t_relax + j * every) / dt): t_relax + j * every for j in range(n_save)}
    out = []
    for step in range(max(save) + 1):
        u, v = np.fft.ifft2(uh).real, np.fft.ifft2(vh).real
        if step in save:
            out.append((save[step], {"u": u.copy(), "v": v.copy()}))
        uh = (uh + dt * np.fft.fft2(u - u**3 - v)) * iu
        vh = (vh + dt * eps * np.fft.fft2(u - a1 * v - a0)) * iv
    return out


def seconds_per_run(**kw) -> float:
    """Rough time of one run in one process (seconds)."""
    s = SETTINGS | kw
    steps = (s["t_relax"] + s["every"] * (s["n_save"] - 1)) / s["dt"]
    return SECONDS_PER_POINT_STEP * steps * s["n"] ** 2


def reference_label(snaps: list[dict[str, np.ndarray]]) -> str:
    """Approximate state of a run from the last two snapshots of u (see the module docstring)."""
    u0, u = snaps[-2]["u"], snaps[-1]["u"]
    if u.std() < FLAT_STD:
        return "uniform"
    a, b = u0 - u0.mean(), u - u.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if den > 0 and 1 - np.sum(a * b) / den > MOVING:
        return "dynamic"
    up = u > 0
    if count_domains(up) + count_domains(~up) <= 2:      # periodic: a band and its complement
        return "fronts"
    return "patterns"


def _one(args):
    folder, run, eps, delta, seed, kw = args
    snaps = simulate(eps, delta, seed, **kw)
    for t, s in snaps:
        write_snapshot(folder / f"run{run:04d}" / "Data" / f"{round(t):010d}.h5", s)
    return reference_label([s for _, s in snaps])


def make_ensemble(folder: str | Path, eps=EPS, delta=DELTA, seeds=SEEDS, workers: int = 1, progress=None,
                  **kw) -> pd.DataFrame:
    """Write the FitzHugh-Nagumo ensemble into `folder` (runs.csv, phasewall.toml, one subfolder
    per run). Returns the table of runs."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(eps, delta, seeds))
    jobs = [(folder, i, float(a), float(b), int(s), kw) for i, (a, b, s) in enumerate(grid)]
    labels = run_jobs(_one, jobs, workers, progress, "simulate")
    runs = pd.DataFrame({"run": range(len(grid)), "path": [f"run{i:04d}" for i in range(len(grid))],
                         "eps": [a for a, _, _ in grid], "delta": [b for _, b, _ in grid],
                         "seed": [s for _, _, s in grid], "reference": labels})
    runs.to_csv(folder / "runs.csv", index=False)
    (folder / "phasewall.toml").write_text(f"""# FitzHugh-Nagumo demo ensemble written by phasewall.fitzhugh_nagumo
[ensemble]
runs = "runs.csv"
parameters = ["eps", "delta"]
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
