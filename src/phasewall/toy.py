"""A demo ensemble with a known answer: the Swift-Hohenberg equation on a grid of parameters.

    du/dt = eps u - (1 + laplacian)^2 u + g u^2 - u^3

is the textbook model of pattern formation (Swift and Hohenberg 1977; Cross and Hohenberg,
Rev. Mod. Phys. 1993). In two dimensions it has three phases: the uniform state u = 0 for
eps < 0, hexagons (spots) for small eps > 0 when g breaks the symmetry u -> -u, and stripes
otherwise. phasewall should find these three phases without being told about them.

The equation is solved pseudo-spectrally on a periodic box of n x n points holding `waves`
wavelengths 2 pi, semi-implicitly: the linear part, minus a stabilising constant C times u,
is implicit, the rest explicit. Each run starts from small random noise (the seed changes
it); after t_relax, n_save snapshots are written every `every` time units, laid out as Julia
writes arrays, like a JLD2 file.

reference_label gives the expected phase of a snapshot from direct measurements (amplitude
and skewness); it is only used to check phasewall, never by it.
"""

import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .io import write_snapshot

# Settings of the demo (choices)
N = 64                  # grid points per side
WAVES = 8               # wavelengths 2 pi across the box
DT = 0.25               # time step
C = 2.0                 # stabilising constant of the semi-implicit scheme
T_RELAX = 500.0         # time before the first saved snapshot
EVERY = 25.0            # time between saved snapshots
N_SAVE = 6              # saved snapshots per run
EPS = np.round(np.arange(-0.075, 0.70, 0.05), 3)   # off eps = 0, where growth is critically slow
G = np.round(np.arange(0.0, 1.401, 0.1), 3)
SEEDS = tuple(range(8))  # runs per point: with 2, the hexagon-stripe wall is too weak for the test
                         # (each half of a point is then one frozen run); found from 6 (2026-11-17)
UNIFORM_STD = 1e-3      # reference_label: below this standard deviation, uniform
HEXAGON_SKEW = 0.4      # reference_label: above this |skewness|, hexagons


def box(n: int = N, waves: int = WAVES) -> tuple[float, np.ndarray]:
    """Grid spacing and |k|^2 of the Fourier modes."""
    dx = waves * 2 * np.pi / n
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    return dx, k[:, None] ** 2 + k[None, :] ** 2


def simulate(eps: float, g: float, seed: int, n: int = N, waves: int = WAVES, dt: float = DT,
             t_relax: float = T_RELAX, every: float = EVERY, n_save: int = N_SAVE) -> list[tuple[float, np.ndarray]]:
    """The saved snapshots (time, u indexed [x, y]) of one run."""
    _, k2 = box(n, waves)
    implicit = 1 - dt * (eps - (1 - k2) ** 2 - C)
    rng = np.random.default_rng(seed)
    uh = np.fft.fft2(0.1 * rng.standard_normal((n, n)))
    save_steps = {round((t_relax + j * every) / dt): t_relax + j * every for j in range(n_save)}
    out = []
    for step in range(max(save_steps) + 1):
        if step in save_steps:
            out.append((save_steps[step], np.fft.ifft2(uh).real))
        u = np.fft.ifft2(uh).real
        uh = (uh + dt * np.fft.fft2(g * u**2 - u**3 + C * u)) / implicit
    return out


def reference_label(u: np.ndarray) -> str:
    """'uniform', 'hexagons' or 'stripes' from the amplitude and the skewness of u."""
    std = u.std()
    if std < UNIFORM_STD:
        return "uniform"
    skew = np.mean((u - u.mean()) ** 3) / std**3
    return "hexagons" if abs(skew) > HEXAGON_SKEW else "stripes"


def _one(args):
    folder, run, eps, g, seed, kw = args
    label = None
    for t, u in simulate(eps, g, seed, **kw):
        write_snapshot(folder / f"run{run:04d}" / "Data" / f"{round(t):010d}.h5", {"u": u})
        label = reference_label(u)
    return label


def make_ensemble(folder: str | Path, eps=EPS, g=G, seeds=SEEDS, workers: int = 1, **kw) -> pd.DataFrame:
    """Write the demo ensemble into `folder`: one subfolder per run with its snapshots,
    runs.csv (path, eps, g, seed, and the reference label of the last snapshot, for checking)
    and phasewall.toml. Returns the table of runs."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(eps, g, seeds))
    jobs = [(folder, i, float(e), float(gg), int(s), kw) for i, (e, gg, s) in enumerate(grid)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            labels = list(pool.map(_one, jobs))
    else:
        labels = [_one(j) for j in jobs]
    runs = pd.DataFrame({"run": range(len(grid)), "path": [f"run{i:04d}" for i in range(len(grid))],
                         "eps": [e for e, _, _ in grid], "g": [gg for _, gg, _ in grid],
                         "seed": [s for _, _, s in grid], "reference": labels})
    runs.to_csv(folder / "runs.csv", index=False)
    dx, _ = box(kw.get("n", N), kw.get("waves", WAVES))
    every = kw.get("every", EVERY)
    (folder / "phasewall.toml").write_text(f"""# Swift-Hohenberg demo ensemble written by phasewall.toy
[ensemble]
runs = "runs.csv"
parameters = ["eps", "g"]
snapshots = "Data/*.h5"
time_regex = '(\\d+)\\.h5$'
time_unit = 1.0
julia_order = true
dx = {dx!r}
periodic = true
last = {kw.get("n_save", N_SAVE)}
pair_dt = {float(every)!r}

[[fields]]
name = "u"
keys = ["u"]
kind = "scalar"
""")
    return runs
