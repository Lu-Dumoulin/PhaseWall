"""A 1D example with a known answer: the complex Ginzburg-Landau equation on a grid of (b, c).

    dA/dt = A + (1 + i b) d_x^2 A - (1 + i c) |A|^2 A

is the normal form of an oscillatory instability (Aranson and Kramer, Rev. Mod. Phys. 2002).
Plane waves are linearly stable when 1 + b c > 0 (Benjamin-Feir-Newell criterion). Started from
small noise on a periodic line, a run ends in one of three regimes, which PhaseWall should find
without being told about them:
- plane waves: |A| uniform, the phase winds at a constant rate (a frozen kymograph of |A|);
- phase turbulence: just past the Benjamin-Feir line, |A| fluctuates a little, the phase
  wanders, but A never vanishes;
- defect turbulence: further, A vanishes at isolated points of the (x, t) plane (phase slips);
  at large |c| this also happens on the stable side of the line, in bursts between laminar
  stretches (spatiotemporal intermittency, counted here with defect turbulence).

The equation is solved pseudo-spectrally with the exponential Euler scheme (the linear part
exactly, the nonlinear term explicitly). After t_relax, `rows` rows are saved every `every` time
units as one kymograph per run (datasets Are and Aim, [x, t], laid out as Julia writes arrays, in
single precision to halve the files).

reference_label measures the regime from the last half of a kymograph (phase slips and the
spread of |A|); it is only used to check PhaseWall, never by it. The grid and the thresholds
come from a pilot (2026-11-23): slips appear for 1 + b c below about -0.6 to -1.2 depending on
b, and at c <= -1.5 for b = 0; phase turbulence has std |A| from 0.01 to 0.09.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from ..parallel import run_jobs
from .kymo import Field1D, write_kymograph

N = 256                 # grid points on the line
LENGTH = 200.0          # length of the periodic line
DT = 0.05               # time step
T_RELAX = 1000.0        # time before the first saved row
EVERY = 0.5             # time between saved rows
ROWS = 800              # saved rows per run (400 time units)
B = np.round(np.arange(0.0, 4.01, 0.5), 3)
C = np.round(np.arange(-0.1, -1.41, -0.1), 3)
SEEDS = tuple(range(4))
GRID = {"b": B, "c": C}
SETTINGS = {"n": N, "length": LENGTH, "dt": DT, "t_relax": T_RELAX, "every": EVERY, "rows": ROWS}
QUICK = {"b": B[::2], "c": C[::3], "n": 128, "length": 100.0, "t_relax": 300.0, "rows": 800}   # windows of 50 rows
SECONDS_PER_POINT_STEP = 3e-8   # measured in the pilot (Apple M4 Max laptop, one process): for cost estimates
DIMS = 1                # a 1D example: kymographs (examples.py, pipeline.py)
FIELDS = [Field1D("A", kind="complex", components=("Are", "Aim"))]
N_FIELDS = 1
PHASE_STD = 0.005       # reference_label: std |A| above this, without slips, is phase turbulence (a choice)


def simulate(b: float, c: float, seed: int, n: int = N, length: float = LENGTH, dt: float = DT,
             t_relax: float = T_RELAX, every: float = EVERY, rows: int = ROWS) -> np.ndarray:
    """The kymograph A[x, t] (complex) of one run started from small noise (seed)."""
    dx = length / n
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    lin = 1 - (1 + 1j * b) * k**2
    expo = np.exp(lin * dt)
    phi = np.where(np.abs(lin * dt) > 1e-8, (expo - 1) / np.where(lin == 0, 1, lin), dt)
    rng = np.random.default_rng(seed)
    ah = np.fft.fft(0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)))
    first, step_every = round(t_relax / dt), round(every / dt)
    out = []
    for step in range(first + step_every * (rows - 1) + 1):
        if step >= first and (step - first) % step_every == 0:
            out.append(np.fft.ifft(ah))
        a = np.fft.ifft(ah)
        ah = expo * ah + phi * np.fft.fft(-(1 + 1j * c) * np.abs(a) ** 2 * a)
    return np.array(out).T


def phase_slips(a: np.ndarray) -> int:
    """Number of points of the (x, t) plane around which the phase of A[x, t] winds (periodic x)."""
    th = np.angle(a)
    th = np.concatenate([th, th[:1]], axis=0)

    def w(d):
        return d - 2 * np.pi * np.round(d / (2 * np.pi))

    turn = (w(th[1:, :-1] - th[:-1, :-1]) + w(th[1:, 1:] - th[1:, :-1]) + w(th[:-1, 1:] - th[1:, 1:])
            + w(th[:-1, :-1] - th[:-1, 1:]))
    return int(np.sum(np.abs(np.rint(turn / (2 * np.pi)))))


def reference_label(a: np.ndarray) -> str:
    """'defect turbulence', 'phase turbulence' or 'plane waves' from the last half of A[x, t]."""
    last = a[:, a.shape[1] // 2:]
    if phase_slips(last) > 0:
        return "defect turbulence"
    return "phase turbulence" if np.abs(last).std() > PHASE_STD else "plane waves"


def seconds_per_run(**kw) -> float:
    s = SETTINGS | kw
    steps = (s["t_relax"] + s["every"] * (s["rows"] - 1)) / s["dt"]
    return SECONDS_PER_POINT_STEP * steps * s["n"]


def _one(args):
    folder, run, b, c, seed, kw = args
    a = simulate(b, c, seed, **kw)
    write_kymograph(folder / f"run{run:04d}" / "kymograph.h5", {"Are": a.real.astype(np.float32),
                                                              "Aim": a.imag.astype(np.float32)})
    return reference_label(a)


def make_ensemble(folder: str | Path, b=B, c=C, seeds=SEEDS, workers: int = 1, progress=None, **kw) -> pd.DataFrame:
    """Write the example into `folder`: one subfolder per run with kymograph.h5, runs.csv (path,
    b, c, seed and the reference label, for checking) and phasewall.toml. Returns the runs."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(b, c, seeds))
    jobs = [(folder, i, float(bb), float(cc), int(s), kw) for i, (bb, cc, s) in enumerate(grid)]
    labels = run_jobs(_one, jobs, workers, progress, "simulate")
    runs = pd.DataFrame({"run": range(len(grid)), "path": [f"run{i:04d}" for i in range(len(grid))],
                         "b": [x for x, _, _ in grid], "c": [y for _, y, _ in grid],
                         "seed": [s for _, _, s in grid], "reference": labels})
    runs.to_csv(folder / "runs.csv", index=False)
    s = SETTINGS | kw
    (folder / "phasewall.toml").write_text(f"""# 1D complex Ginzburg-Landau example written by phasewall.oned.cgle
[kymograph]
runs = "runs.csv"
parameters = ["b", "c"]
file = "kymograph.h5"
julia_order = true
order = "xt"
dx = {s["length"] / s["n"]!r}
dt = {float(s["every"])!r}
t0 = {float(s["t_relax"])!r}
periodic = true
share = 0.5
windows = 8

[[fields]]
name = "A"
kind = "complex"
components = ["Are", "Aim"]
""")
    return runs
