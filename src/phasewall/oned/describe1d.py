"""Hand-made observables of a time window of a kymograph (1D systems).

A window is the array a[x, t, component] of every field over a stretch of time (kymo.py). Each
field gives columns `field.block.name`, as in 2D, so the walls and the other methods read them
unchanged. Every column is unchanged by a shift in x (exactly if x is periodic) and by the
mirror x -> -x with the odd components flipped (kymo.Field1D.odd); for a complex field also by a
global phase A -> A e^{i phi}. tests/test_oned_describe.py checks these symmetries and a known
answer for each block.

Blocks (DEFAULT_BLOCKS gives those of each kind):
- amp: log10 of the variance over x and t (components summed).
- shape: S(k) on shells of |k| (k and -k together), averaged over the window, divided by its sum.
- mom: mean, std, skewness, kurtosis of the value (scalar) or of the size |a| (other kinds).
- topo: intervals per unit length above and below mean + z std, z = -1, 0, 1, averaged over t.
- order: |<a>_x| / <|a|>_x, averaged over t (not for scalars).
- flow: log10 <(d_x a_x)^2>, the compressibility of a velocity or the splay of a polarity.
- defects: one-component polarity: zeros of p per unit length, split by the sign of d_x p
  (plus: p points away on both sides, an aster; minus: towards it). Two components (polarity
  (p_x, p_y), complex A): phase slips, the points of the (x, t) plane around which the angle of
  the field winds by 2 pi, per unit length and time; twist, the mean |d_x angle| (the wavenumber
  of a plane wave), and twist_std, its spread (0 for a plane wave, large in phase turbulence).
- A complex amplitude gets every other block on its size |A|: the phase of a plane wave and its
  wavenumber are not a difference of state, and A itself would make each plane wave look new.
- skw: the space-time spectrum S(k, omega): share of the power that does not move (static),
  log10 of the mean |omega| weighted by the power, travelling share |P+ - P-| / (P+ + P-) (P+:
  power where k omega > 0), spectral entropy over omega.
- corr: correlation of the window with itself `lag` later: log10 of the decorrelation time tau
  (C < 1/e), C at the lag DRIFT_LAG of the window at the same place (still) and after the best
  shift in x (moving), stored as log10(1 - C + DECOR_FLOOR) as in 2D, and log10 of the drift
  speed of that best shift. As in particle image velocimetry, a periodic pattern that moves by
  more than half its wavelength during the lag is seen moving by less (the smallest shift wins).
- kinks: zeros of the sign field (scalar: a minus its mean; velocity and polarity: a_x) in
  space-time: log10 of their density per unit length, log10 of births and deaths per unit
  length and time, mean lifetime of the space-time domains between them (share of the window).
- trend: slope of log variance against log t over the window (and of log(1 + zeros) for the
  kinds with kinks): 0 when steady, negative while a pattern relaxes or coarsens.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal.windows import tukey

from ..describe import (
    DECOR_FLOOR,
    FLAT,
    FLAT_RELATIVE,
    LEVELS,
    TUKEY_ALPHA,
    assemble_table,
    joined_labels,
    k_shells,
    moments,
    wrapped,
)
from ..parallel import Progress, run_jobs
from .kymo import Ensemble1D, Field1D, read_kymograph, window_slices

BLOCKS = ("amp", "shape", "mom", "topo", "order", "flow", "defects", "skw", "corr", "kinks", "trend")
DEFAULT_BLOCKS = {
    "scalar": ("amp", "shape", "mom", "topo", "skw", "corr", "kinks", "trend"),
    "velocity": ("amp", "shape", "mom", "topo", "order", "flow", "skw", "corr", "kinks", "trend"),
    "polarity": ("amp", "shape", "mom", "topo", "order", "flow", "defects", "skw", "corr", "kinks", "trend"),
    "complex": ("amp", "shape", "mom", "topo", "order", "defects", "skw", "corr", "trend"),
}
ALLOWED = {k: set(v) for k, v in DEFAULT_BLOCKS.items()}
BLOCK_HELP = {                                   # one line per block, for the interface and explain()
    "amp": "how strong the pattern is: log10 of the variance over the window",
    "shape": "which length scales are present: S(k) on shells of |k|, averaged over the window, divided by its sum",
    "mom": "distribution of the values (|a| for a vector or complex field): mean, std, skewness, kurtosis",
    "topo": "intervals per unit length above and below the mean plus -1, 0, 1 standard deviations",
    "order": "global alignment |<a>| / <|a|>, averaged over time: 1 when everything points the same way",
    "flow": "compressibility of a velocity, splay of a polarity: log10 of the mean square d_x a_x",
    "defects": "zeros of a one-component polarity by sign; phase slips in (x, t) and twist of a two-component field",
    "skw": "space-time spectrum S(k, omega): static share, mean frequency, travelling share, entropy in time",
    "corr": "correlation over many lags: decorrelation time, change at the same place and moving, drift speed",
    "kinks": "zeros of the field in space-time: density, births, deaths, lifetime of the domains between them",
    "trend": "slope of the variance (and of the zeros) against log t: 0 when steady, negative while relaxing",
}
AMP_FLOOR = -12.0      # log10 variance of a flat field (as in 2D)
STATIC_BINS = 3        # skw: power within 3 frequency bins of omega = 0 counts as static (a choice:
                       # the main lobe of the Tukey window in time)
DRIFT_LAG = 0.125      # corr: the moving correlation is taken at 1/8 of the window (a choice)
MAX_SHIFT_FRACTION = 0.2   # corr, non-periodic x: shifts up to 1/5 of the line (as in 2D)
TAU_LEVEL = np.exp(-1.0)   # corr: decorrelation when C falls below 1/e (a choice)
TWIST_SIZE = 0.1       # defects: twist counts the links where the size is above 1/10 of its mean (a choice:
                       # the angle of a vanishing field is noise)


def blocks_of(f: Field1D) -> tuple[str, ...]:
    """The blocks computed for a field: its own, or the defaults of its kind. ValueError for a
    block that does not fit the kind."""
    chosen = f.blocks if f.blocks is not None else DEFAULT_BLOCKS[f.kind]
    bad = [b for b in chosen if b not in ALLOWED[f.kind]]
    if bad:
        raise ValueError(f"field {f.name}: blocks {bad} do not apply to a {f.kind} in 1D "
                         f"(allowed: {sorted(ALLOWED[f.kind])})")
    if f.kind == "complex" and "kinks" in chosen:
        raise ValueError(f"field {f.name}: a complex field has phase slips (defects), not kinks")
    return tuple(chosen)


# ---------------------------------------------------------------- the window, prepared
def prepared(a: np.ndarray, f: Field1D) -> np.ndarray:
    """a[x, t, component] as float; a relative scalar becomes (a - mean) / mean; a vector or
    complex field whose fluctuations are below FLAT_RELATIVE of its size is replaced by its
    mean (uniform, as in 2D)."""
    a = np.asarray(a, dtype=float)
    a = a[..., None] if a.ndim == 2 else a
    if f.relative:
        m = a.mean()
        a = (a - m) / m if abs(m) > FLAT else a - m
    if f.kind != "scalar":
        size = np.sqrt(np.mean(np.sum(a**2, axis=-1)))
        fluct = np.sqrt(np.mean(np.sum((a - a.mean(axis=(0, 1))) ** 2, axis=-1)))
        if size > 0 and fluct < FLAT_RELATIVE * size:
            a = np.broadcast_to(a.mean(axis=(0, 1)), a.shape).copy()
    return a


def size_of(a: np.ndarray, f: Field1D) -> np.ndarray:
    """The value described by mom and topo: the scalar itself, or the size |a|."""
    return a[..., 0] if f.kind == "scalar" else np.sqrt(np.sum(a**2, axis=-1))


def _log10(x: float, floor: float) -> float:
    return float(np.log10(max(x, floor)))


# ---------------------------------------------------------------- blocks
def amp(a: np.ndarray) -> float:
    return _log10(float(np.sum(a.var(axis=(0, 1)))), 10**AMP_FLOOR)


def shape(a: np.ndarray, dx: float, periodic: bool) -> dict[str, float]:
    """S(k) averaged over the rows, on the shells of describe.k_shells, divided by its sum."""
    n = a.shape[0]
    edges = k_shells(n, dx)
    b = a - a.mean(axis=(0, 1))
    if not periodic:
        w = tukey(n, TUKEY_ALPHA)
        b = b * (w / np.sqrt(np.mean(w**2)))[:, None, None]
    power = np.sum(np.abs(np.fft.fft(b, axis=0)) ** 2, axis=(1, 2))      # summed over t and components
    k = np.abs(2 * np.pi * np.fft.fftfreq(n, d=dx))
    s = np.array([power[(k >= lo) & (k < hi)].sum() for lo, hi in itertools.pairwise(edges)])
    total = s.sum()
    s = s / total if total > FLAT**2 * power.size else np.zeros_like(s)
    return {f"k{i:02d}": float(v) for i, v in enumerate(s)}


def mom(v: np.ndarray, f: Field1D) -> dict[str, float]:
    m, s, sk, ku = moments(v)
    if f.kind != "scalar" and s < FLAT_RELATIVE * abs(m):
        s, sk, ku = 0.0, 0.0, 0.0        # a size that does not vary (as in 2D)
    out = {"mean": m} if not f.relative else {}
    return out | {"std": s, "skew": sk, "kurt": ku}


def intervals(mask: np.ndarray, periodic: bool) -> np.ndarray:
    """Number of intervals of True along x in every row (mask[x, t]); with `periodic`, an
    interval through the end of the line is counted once."""
    m = mask.astype(int)
    starts = np.sum(np.diff(m, axis=0) == 1, axis=0) + m[0]
    if periodic:
        starts = starts - (m[0] & m[-1])
        starts = np.where(m.all(axis=0), 1, starts)
    return starts


def topo(v: np.ndarray, length: float, periodic: bool, uniform: bool) -> dict[str, float]:
    mean, std = v.mean(), v.std()
    out = {}
    for z in LEVELS:
        if std < FLAT or uniform:
            up = down = 0.0
        else:
            up = intervals(v > mean + z * std, periodic).mean()
            down = intervals(v < mean + z * std, periodic).mean()
        out[f"above{z:+.0f}"] = float(up) / length
        out[f"below{z:+.0f}"] = float(down) / length
    return out


def order(a: np.ndarray) -> float:
    mean_vec = np.sqrt(np.sum(a.mean(axis=0) ** 2, axis=-1))           # |<a>_x| per row
    mean_size = np.sqrt(np.sum(a**2, axis=-1)).mean(axis=0)
    ok = mean_size > FLAT
    return float(np.mean(np.where(ok, mean_vec / np.where(ok, mean_size, 1.0), 0.0)))


def d_dx(u: np.ndarray, dx: float, periodic: bool) -> np.ndarray:
    """Derivative along x (axis 0): spectral if periodic, second-order differences otherwise."""
    if periodic:
        k = 2 * np.pi * np.fft.fftfreq(u.shape[0], d=dx)
        shape_k = (-1,) + (1,) * (u.ndim - 1)
        return np.fft.ifft(1j * k.reshape(shape_k) * np.fft.fft(u, axis=0), axis=0).real
    return np.gradient(u, dx, axis=0)


def flow(a: np.ndarray, dx: float, periodic: bool) -> dict[str, float]:
    return {"log10div2": _log10(float(np.mean(d_dx(a[..., 0], dx, periodic) ** 2)), FLAT**2)}


def point_defects(p: np.ndarray, dx: float, periodic: bool) -> dict[str, float]:
    """Zeros of a one-component polarity p[x, t] per unit length, averaged over t: plus where p
    rises through 0 (pointing away on both sides), minus where it falls."""
    s = p > 0
    nxt = np.roll(s, -1, axis=0) if periodic else s[1:]
    cur = s if periodic else s[:-1]
    length = p.shape[0] * dx
    rising = np.sum(~cur & nxt, axis=0).mean() / length
    falling = np.sum(cur & ~nxt, axis=0).mean() / length
    return {"plus": float(rising), "minus": float(falling)}


def phase_slips(a: np.ndarray, dx: float, dt: float, periodic: bool) -> dict[str, float]:
    """Two-component field a[x, t, 2]: winding of its angle around every plaquette of the (x, t)
    grid (each step the smallest turn); slips = plaquettes with |winding| = 1 per unit length and
    time; imbalance = |n+ - n-| / (n+ + n-) (0 without slips; the mirror swaps the two signs);
    twist = mean |d_x angle| over the points where the field is not small."""
    theta = np.arctan2(a[..., 1], a[..., 0])
    if periodic:
        th = np.concatenate([theta, theta[:1]], axis=0)
    else:
        th = theta
    c00, c10, c11, c01 = th[:-1, :-1], th[1:, :-1], th[1:, 1:], th[:-1, 1:]
    turn = (wrapped(c10 - c00, 2 * np.pi) + wrapped(c11 - c10, 2 * np.pi) + wrapped(c01 - c11, 2 * np.pi)
            + wrapped(c00 - c01, 2 * np.pi))
    w = np.rint(turn / (2 * np.pi)).astype(int)
    n_plus, n_minus = int(np.sum(w > 0)), int(np.sum(w < 0))
    area = (th.shape[0] - 1) * dx * (theta.shape[1] - 1) * dt
    size = np.sqrt(np.sum(a**2, axis=-1))
    size = np.concatenate([size, size[:1]], axis=0) if periodic else size
    dth = wrapped(np.diff(th, axis=0), 2 * np.pi) / dx
    link = 0.5 * (size[1:] + size[:-1])                       # size on each link between two points
    big = (link > TWIST_SIZE * size.mean()) if size.mean() > FLAT else np.zeros(dth.shape, bool)
    twist = float(np.abs(dth[big]).mean()) if big.any() else 0.0
    twist_std = float(dth[big].std()) if big.any() else 0.0
    n = n_plus + n_minus
    return {"slips": n / area, "imbalance": abs(n_plus - n_minus) / n if n else 0.0, "twist": twist,
            "twist_std": twist_std}


def skw(a: np.ndarray, dx: float, dt: float, periodic: bool) -> dict[str, float]:
    """Space-time spectrum of the window (fluctuations about its mean, Tukey window in t, and
    in x if not periodic)."""
    nx, nt = a.shape[:2]
    domega = 2 * np.pi / (nt * dt)
    if np.sum(a.var(axis=(0, 1))) < FLAT**2:
        return {"static": 1.0, "log10omega": float(np.log10(domega)), "travelling": 0.0, "entropy_t": 0.0}
    b = a - a.mean(axis=(0, 1))
    wt = tukey(nt, TUKEY_ALPHA)
    b = b * wt[None, :, None]
    if not periodic:
        b = b * tukey(nx, TUKEY_ALPHA)[:, None, None]
    P = np.sum(np.abs(np.fft.fft2(b, axes=(0, 1))) ** 2, axis=-1)
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)[:, None]
    om = 2 * np.pi * np.fft.fftfreq(nt, d=dt)[None, :]
    total = P.sum()
    static = P[:, np.abs(om[0]) < STATIC_BINS * domega].sum() / total
    mean_omega = float((P * np.abs(om)).sum() / total)
    plus, minus = P[(k * om) > 0].sum(), P[(k * om) < 0].sum()
    travelling = abs(plus - minus) / (plus + minus) if plus + minus > FLAT * total else 0.0
    marg = P.sum(axis=0) / total
    marg = marg[marg > 0]
    entropy = float(-(marg * np.log(marg)).sum() / np.log(nt))
    return {"static": float(static), "log10omega": _log10(mean_omega, domega), "travelling": float(travelling),
            "entropy_t": entropy}


def _corr_at(b: np.ndarray, lag: int, shifts: np.ndarray, periodic: bool) -> np.ndarray:
    """Correlation of b[x, t, c] with itself `lag` rows later, shifted by each of `shifts`
    points in x (Pearson over the overlapping part; components summed)."""
    x0, x1 = b[:, : b.shape[1] - lag], b[:, lag:]
    out = []
    for s in shifts:
        if periodic:
            u, v = x0, np.roll(x1, -s, axis=0)
        elif s >= 0:
            u, v = x0[: x0.shape[0] - s], x1[s:]
        else:
            u, v = x0[-s:], x1[: x1.shape[0] + s]
        norm = np.sqrt(np.sum(u**2) * np.sum(v**2))
        if norm < FLAT**2:
            out.append(1.0)
            continue
        out.append(float(np.sum(u * v) / norm))
    return np.array(out)


def corr(a: np.ndarray, dx: float, dt: float, periodic: bool) -> dict[str, float]:
    nx, nt = a.shape[:2]
    b = a - a.mean(axis=(0, 1))
    lags = np.unique(np.round(2.0 ** np.arange(0, np.log2(nt / 2) + 1e-9)).astype(int))
    c = np.array([_corr_at(b, int(L), np.array([0]), periodic)[0] for L in lags])
    below = np.flatnonzero(c < TAU_LEVEL)
    if not len(below):
        tau = lags[-1] * dt                                   # longer than half the window
    elif below[0] == 0:
        tau = lags[0] * dt * c[0] / TAU_LEVEL if c[0] > 0 else lags[0] * dt * 0.5
    else:
        i = below[0]                                          # interpolate in log lag between lags i-1 and i
        f = (c[i - 1] - TAU_LEVEL) / (c[i - 1] - c[i])
        tau = float(np.exp(np.log(lags[i - 1]) + f * (np.log(lags[i]) - np.log(lags[i - 1])))) * dt
    lag = max(1, round(DRIFT_LAG * nt))
    smax = nx // 2 if periodic else max(1, int(MAX_SHIFT_FRACTION * nx))
    shifts = np.arange(-smax, smax + 1)
    cs = _corr_at(b, lag, shifts, periodic)
    tied = np.flatnonzero(cs >= cs.max() - 1e-9)              # a periodic pattern: the smallest shift wins
    best = int(tied[np.argmin(np.abs(shifts[tied]))])
    still = cs[shifts == 0][0]
    speed = abs(shifts[best]) * dx / (lag * dt)
    return {"log10tau": _log10(tau, dt), "still": float(np.log10(max(1 - still, 0) + DECOR_FLOOR)),
            "moving": float(np.log10(max(1 - cs[best], 0) + DECOR_FLOOR)),
            "log10drift": _log10(speed, dx / (nt * dt))}


def sign_field(a: np.ndarray, f: Field1D) -> np.ndarray | None:
    """The field whose zeros are the kinks: a scalar minus its mean, or a_x; None if flat."""
    s = a[..., 0] - a[..., 0].mean() if f.kind == "scalar" else a[..., 0]
    return None if s.std() < FLAT and np.abs(s).max() < FLAT else s


def zero_counts(s: np.ndarray, periodic: bool) -> np.ndarray:
    """Zeros of s[x, t] in every row: sign changes between neighbours along x."""
    pos = s > 0
    if periodic:
        return np.sum(pos != np.roll(pos, -1, axis=0), axis=0)
    return np.sum(pos[1:] != pos[:-1], axis=0)


def domain_lifetimes(pos: np.ndarray, periodic: bool) -> np.ndarray:
    """Number of rows spanned by every connected domain of pos[x, t] and of ~pos (4-neighbour,
    joined through the ends of the line if periodic)."""
    out = []
    for mask in (pos, ~pos):
        lab, n = ndimage.label(mask)
        if n == 0:
            continue
        roots = joined_labels(n, zip(lab[0], lab[-1]) if periodic else [])
        rows = np.broadcast_to(np.arange(mask.shape[1]), mask.shape)
        r = roots[lab[mask]]
        first = pd.Series(rows[mask]).groupby(r).min()
        last = pd.Series(rows[mask]).groupby(r).max()
        out.extend((last - first + 1).tolist())
    return np.array(out, dtype=float)


def kinks(s: np.ndarray | None, dx: float, dt: float, periodic: bool) -> dict[str, float]:
    nx, nt = (s.shape if s is not None else (1, 1))
    length, duration = nx * dx, nt * dt
    floor_density, floor_rate = 1.0 / length, 1.0 / (length * duration)
    if s is None:
        return {"log10zeros": float(np.log10(floor_density)), "log10births": float(np.log10(floor_rate)),
                "log10deaths": float(np.log10(floor_rate)), "lifetime": 1.0}
    z = zero_counts(s, periodic)
    dz = np.diff(z)
    births = np.sum(np.maximum(dz, 0)) / (length * duration)
    deaths = np.sum(np.maximum(-dz, 0)) / (length * duration)
    life = domain_lifetimes(s > 0, periodic)
    return {"log10zeros": _log10(float(z.mean()) / length, floor_density),
            "log10births": _log10(births, floor_rate), "log10deaths": _log10(deaths, floor_rate),
            "lifetime": float(life.mean() / nt) if len(life) else 1.0}


def slope_log(y: np.ndarray, t: np.ndarray) -> float:
    """Least-squares slope of log y against log t (0 if y does not vary)."""
    if np.ptp(y) <= FLAT * max(np.abs(y).max(), FLAT) or np.any(t <= 0):
        return 0.0
    return float(np.polyfit(np.log(t), np.log(y), 1)[0])


def trend(a: np.ndarray, s: np.ndarray | None, times: np.ndarray, periodic: bool, with_zeros: bool
          ) -> dict[str, float]:
    var_t = np.sum(a.var(axis=0), axis=-1) + FLAT**2
    out = {"amp_slope": slope_log(var_t, times)}
    if with_zeros:
        out["zeros_slope"] = slope_log(1.0 + zero_counts(s, periodic), times) if s is not None else 0.0
    return out


# ---------------------------------------------------------------- one window, one run, an ensemble
def field_descriptors(a: np.ndarray, f: Field1D, dx: float, dt: float, times: np.ndarray, periodic: bool
                      ) -> dict[str, float]:
    """The columns of one field over one window: a[x, t(, component)], times of its rows."""
    a = prepared(a, f)
    use = blocks_of(f)
    v = size_of(a, f)
    uniform = f.kind != "scalar" and v.std() < FLAT_RELATIVE * max(abs(v.mean()), FLAT)
    complex_field = f.kind == "complex"
    # a complex amplitude: everything but order and defects is computed on its size |A| (the phase
    # of a plane wave, and its wavenumber, are not a difference of state)
    g = (np.full_like(v, v.mean()) if uniform else v)[..., None] if complex_field else a
    length = a.shape[0] * dx
    out: dict[str, float] = {}
    if "amp" in use:
        out["amp.log10var"] = amp(g)
    if "shape" in use:
        out |= {f"shape.{k}": x for k, x in shape(g, dx, periodic).items()}
    if "mom" in use:
        out |= {f"mom.{k}": x for k, x in mom(v, f).items()}
    if "topo" in use:
        out |= {f"topo.{k}": x for k, x in topo(v, length, periodic, uniform).items()}
    if "order" in use:
        out["order.global"] = order(a)
    if "flow" in use:
        out |= {f"flow.{k}": x for k, x in flow(a, dx, periodic).items()}
    if "defects" in use:
        d = point_defects(a[..., 0], dx, periodic) if a.shape[-1] == 1 else phase_slips(a, dx, dt, periodic)
        out |= {f"defects.{k}": x for k, x in d.items()}
    if "skw" in use:
        out |= {f"skw.{k}": x for k, x in skw(g, dx, dt, periodic).items()}
    if "corr" in use:
        out |= {f"corr.{k}": x for k, x in corr(g, dx, dt, periodic).items()}
    s = sign_field(a, f) if ("kinks" in use or "trend" in use) and not complex_field else None
    if "kinks" in use:
        out |= {f"kinks.{k}": x for k, x in kinks(s, dx, dt, periodic).items()}
    if "trend" in use:
        out |= {f"trend.{k}": x for k, x in trend(g, s, times, periodic, "kinks" in use).items()}
    return {f"{f.name}.{k}": x for k, x in out.items()}


def describe_window(arrays: dict[str, np.ndarray], fields: list[Field1D], dx: float, dt: float,
                    times: np.ndarray, periodic: bool = True) -> dict[str, float]:
    """All columns of one window: arrays[name] indexed [x, t(, component)] over the window."""
    row = {}
    for f in fields:
        row |= field_descriptors(arrays[f.name], f, dx, dt, times, periodic)
    return row


def describe_run(ens: Ensemble1D, run: int, path: str) -> list[dict]:
    """One row per window of a run (kymo.window_slices): run, time (the end of the window),
    then the columns."""
    data = read_kymograph(Path(path) / ens.file, ens.fields, ens.julia_order, ens.order)
    n_t = next(iter(data.values())).shape[1]
    t_all = ens.t0 + ens.dt * np.arange(n_t)
    dx = ens.dx_of(run)
    rows = []
    for sl in window_slices(n_t, ens.share, ens.windows):
        arrays = {name: a[:, sl] for name, a in data.items()}
        rows.append({"run": run, "time": float(t_all[sl][-1])}
                    | describe_window(arrays, ens.fields, dx, ens.dt, t_all[sl], ens.periodic))
    return rows


def _describe_one(args):
    ens, run, path = args
    try:
        return describe_run(ens, run, path), None
    except Exception as err:   # noqa: BLE001 (one broken run must not stop the ensemble)
        return [], f"run {run}: {type(err).__name__}: {err}"


def describe_ensemble(ens: Ensemble1D, workers: int = 1, progress: Progress | None = None
                      ) -> tuple[pd.DataFrame, list[str]]:
    """Descriptors of every run: one row per window (columns run, time, the parameters, then
    the descriptors, as in 2D) and the runs that could not be read. ValueError if the runs
    give different S(k) shells (lines of different lengths or spacings)."""
    for f in ens.fields:
        blocks_of(f)                                        # refuse a wrong block before any reading
    jobs = [(ens, int(r.run), r.path) for r in ens.runs.itertuples()]
    results = run_jobs(_describe_one, jobs, workers, progress, "describe")
    return assemble_table(results, ens.runs, ens.parameters, "lines of different lengths or grid spacings")
