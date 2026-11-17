"""Generic descriptors of a snapshot: numbers that say what a pattern looks like, whatever the
system, unchanged by a rotation, a mirror or a periodic shift of the box.

For every field (arrays [x, y] or [x, y, component], periodic box, grid spacing dx):

- amp:   log10 of the variance of the field (summed over components): how strong the pattern is.
- shape: the structure factor S(k) on shells of |k|, divided by its sum: which length scales
         are present, whatever the amplitude.
- mom:   mean (scalars that are not relative), standard deviation, skewness and excess kurtosis
         of the values (scalar), of |a| (vector) or of the scalar order 2|Q| (nematic).
- order: vectors and nematics: |<a>| / <|a|>, 1 when everything points the same way.
- topo:  the number of connected domains (per unit area, periodic box) where the values
         (as for mom) lie above, and below, the mean plus z standard deviations, for z in
         LEVELS: spots give many small domains, stripes a few long ones; the domains of low
         |p| are the defect cores. Minkowski functionals of the same kind are used for
         patterns and in cosmology.
- flow:  vectors: log10 of the mean square divergence and curl, and the share of the curl:
         splay against bend, asters against vortices.
- dyn:   for a pair of snapshots a time pair_dt apart: the correlation C of the fluctuations
         at the same place (still) and after the best periodic shift (moving), given as the
         decorrelation log10(1 - C + DECOR_FLOOR). A frozen pattern has both at the floor; a
         pattern that travels without changing has still above the floor and moving at it; a
         pattern that changes has both above it. The floor keeps numerical noise in a frozen
         pattern (C = 0.99999) from looking like dynamics.

Column names are field.block.name, for example rho.shape.k07; walls.py weights the blocks.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage

from .fields import Field
from .io import Ensemble, read_snapshot, run_snapshots

FLAT = 1e-6           # fields whose fluctuations are below this are flat (a choice)
SHELL_RATIO = 1.2     # each shell of |k| is 1.2 times wider than the previous one (a choice)
DECOR_FLOOR = 1e-3    # correlations above 1 - DECOR_FLOOR count as frozen (a choice)
LEVELS = (-1.0, 0.0, 1.0)   # topo: levels in standard deviations from the mean (a choice)


# ---------------------------------------------------------------- spectra
def k_shells(n: int, dx: float, ratio: float = SHELL_RATIO) -> np.ndarray:
    """Edges of the shells of |k| for a box of n points: from half the lowest wavenumber
    2 pi / L to pi / dx, each shell `ratio` times wider than the previous one but never
    narrower than 2 pi / L, so that no shell is empty."""
    dk = 2 * np.pi / (n * dx)
    edges = [0.5 * dk]
    while edges[-1] * ratio < np.pi / dx and edges[-1] + dk < np.pi / dx:
        edges.append(max(edges[-1] * ratio, edges[-1] + dk))
    edges.append(np.pi / dx)
    return np.array(edges)


def _components(a: np.ndarray) -> np.ndarray:
    return a[..., None] if a.ndim == 2 else a


def structure_factor(a: np.ndarray, dx: float, edges: np.ndarray) -> np.ndarray:
    """S(k) = dx^2 / (nx ny) <|a_k|^2> averaged over each shell, components summed (so a
    global rotation of the vectors leaves it unchanged); the mean is removed first."""
    f = _components(a)
    nx, ny = f.shape[:2]
    power = sum(np.abs(np.fft.fft2(f[..., c] - f[..., c].mean())) ** 2 for c in range(f.shape[2]))
    k = np.hypot(*np.meshgrid(2 * np.pi * np.fft.fftfreq(nx, dx), 2 * np.pi * np.fft.fftfreq(ny, dx),
                              indexing="ij"))
    shell = np.digitize(k.ravel(), edges) - 1
    inside = (shell >= 0) & (shell < len(edges) - 1)
    n = len(edges) - 1
    total = np.bincount(shell[inside], weights=power.ravel()[inside], minlength=n)
    count = np.bincount(shell[inside], minlength=n)
    return dx**2 / (nx * ny) * total / np.maximum(count, 1)


def div_curl(a: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Divergence and curl of a periodic vector field (spectral derivatives)."""
    nx, ny = a.shape[:2]
    kx = 2j * np.pi * np.fft.fftfreq(nx, dx)[:, None]
    ky = 2j * np.pi * np.fft.fftfreq(ny, dx)[None, :]
    ax, ay = np.fft.fft2(a[..., 0]), np.fft.fft2(a[..., 1])
    return np.fft.ifft2(kx * ax + ky * ay).real, np.fft.ifft2(kx * ay - ky * ax).real


def moments(v: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, standard deviation, skewness, excess kurtosis; 0 for the last two when the
    values do not vary (a choice)."""
    v = np.ravel(v)
    mean, std = float(v.mean()), float(v.std())
    if std < FLAT:
        return mean, std, 0.0, 0.0
    z = (v - mean) / std
    return mean, std, float(np.mean(z**3)), float(np.mean(z**4) - 3)


def count_domains(mask: np.ndarray, periodic: bool = True) -> int:
    """Number of connected domains of True (4-neighbour connectivity); with `periodic`, a
    domain that crosses the edge of the box is counted once."""
    lab, n = ndimage.label(mask)
    if not periodic or n == 0:
        return int(n)
    parent = np.arange(n + 1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in ((lab[0, :], lab[-1, :]), (lab[:, 0], lab[:, -1])):
        for i, j in zip(a, b):
            if i and j:
                ri, rj = root(i), root(j)
                if ri != rj:
                    parent[ri] = rj
    return len({root(i) for i in range(1, n + 1)})


def domain_counts(values: np.ndarray, area: float, periodic: bool = True) -> dict[str, float]:
    """topo block: domains per unit area above and below mean + z std for z in LEVELS."""
    mean, std = values.mean(), values.std()
    out = {}
    for z in LEVELS:
        if std < FLAT:
            up = down = 0
        else:
            up = count_domains(values > mean + z * std, periodic)
            down = count_domains(values < mean + z * std, periodic)
        out[f"above{z:+.0f}"] = up / area
        out[f"below{z:+.0f}"] = down / area
    return out


def _log10(x: float) -> float:
    return float(np.log10(max(x, FLAT**2)))


# ---------------------------------------------------------------- one field
def prepared(a: np.ndarray, f: Field) -> np.ndarray:
    """The array that is described: (a - <a>) / <a> for a relative scalar."""
    if f.relative:
        m = a.mean()
        return (a - m) / m if m != 0 else a - m
    return a


def field_descriptors(a: np.ndarray, f: Field, dx: float, edges: np.ndarray) -> dict[str, float]:
    """Descriptors of one field of one snapshot (see the module docstring)."""
    a = prepared(a, f)
    c = _components(a)
    variance = float(np.sum(c.var(axis=(0, 1))))
    flat = np.sqrt(variance) < FLAT
    out = {f"{f.name}.amp.log10var": _log10(variance)}
    S = np.zeros(len(edges) - 1) if flat else structure_factor(a, dx, edges)
    shape = S / S.sum() if S.sum() > 0 else S
    out |= {f"{f.name}.shape.k{j:02d}": float(s) for j, s in enumerate(shape)}
    if f.kind == "scalar":
        values = a
    else:
        size = np.hypot(c[..., 0], c[..., 1])
        values = 2 * size if f.kind == "nematic" else size
    mean, std, skew, kurt = moments(values)
    area = values.shape[0] * values.shape[1] * dx**2
    out |= {f"{f.name}.topo.{k}": v for k, v in domain_counts(values, area).items()}
    if not f.relative:
        out[f"{f.name}.mom.mean"] = mean
    out |= {f"{f.name}.mom.std": std, f"{f.name}.mom.skew": skew, f"{f.name}.mom.kurt": kurt}
    if f.kind != "scalar":
        size_mean = float(np.hypot(c[..., 0], c[..., 1]).mean())
        mean_vec = float(np.hypot(c[..., 0].mean(), c[..., 1].mean()))
        out[f"{f.name}.order.global"] = mean_vec / size_mean if size_mean > FLAT else 0.0
    if f.kind == "vector":
        d, w = div_curl(a, dx)
        d2, w2 = float(np.mean(d**2)), float(np.mean(w**2))
        out[f"{f.name}.flow.log10div2"] = _log10(d2)
        out[f"{f.name}.flow.log10curl2"] = _log10(w2)
        # 0.5 when neither is present (a choice): no preference for splay or bend
        out[f"{f.name}.flow.curl_share"] = w2 / (d2 + w2) if d2 + w2 > FLAT**2 else 0.5
    return out


def correlations(before: np.ndarray, after: np.ndarray, f: Field) -> tuple[float, float]:
    """Correlation of the fluctuations of one field between two snapshots, at the same place
    (still) and after the periodic shift that maximises it (moving). Both are 1 when either
    snapshot is flat (a choice: nothing changes)."""
    x = _components(prepared(before, f))
    y = _components(prepared(after, f))
    x = x - x.mean(axis=(0, 1))
    y = y - y.mean(axis=(0, 1))
    npts = x[..., 0].size
    if min(np.sum(x**2), np.sum(y**2)) / npts < FLAT**2:
        return 1.0, 1.0
    norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
    cross = sum(np.fft.ifft2(np.conj(np.fft.fft2(x[..., i])) * np.fft.fft2(y[..., i])).real
                for i in range(x.shape[2]))
    return float(np.sum(x * y) / norm), float(cross.max() / norm)


def pair_descriptors(before: np.ndarray, after: np.ndarray, f: Field) -> dict[str, float]:
    """Decorrelation log10(1 - C + DECOR_FLOOR) at the same place and after the best shift."""
    still, moving = correlations(before, after, f)
    return {f"{f.name}.dyn.still": float(np.log10(max(1 - still, 0) + DECOR_FLOOR)),
            f"{f.name}.dyn.moving": float(np.log10(max(1 - moving, 0) + DECOR_FLOOR))}


# ---------------------------------------------------------------- runs and ensembles
def describe_snapshot(snapshot: dict[str, np.ndarray], fields: list[Field], dx: float,
                      before: dict[str, np.ndarray] | None = None) -> dict[str, float]:
    """All descriptors of one snapshot; with `before` (the snapshot pair_dt earlier), also
    the dynamics."""
    n = snapshot[fields[0].name].shape[0]
    edges = k_shells(n, dx)
    out = {}
    for f in fields:
        out |= field_descriptors(snapshot[f.name], f, dx, edges)
        if before is not None:
            out |= pair_descriptors(before[f.name], snapshot[f.name], f)
    return out


def describe_run(ens: Ensemble, run: int, path: str) -> list[dict]:
    """One row per described snapshot of a run: the last `ens.last` snapshots."""
    snaps = run_snapshots(ens, path)
    times = np.array([t for t, _ in snaps])
    rows = []
    for t, p in snaps[-ens.last:]:
        snap = read_snapshot(p, ens.fields, ens.julia_order)
        before = None
        if ens.pair_dt is not None:
            j = np.flatnonzero(np.isclose(times, t - ens.pair_dt, rtol=0, atol=1e-6 * ens.pair_dt))
            if len(j):
                before = read_snapshot(snaps[j[0]][1], ens.fields, ens.julia_order)
        row = {"run": run, "time": t} | describe_snapshot(snap, ens.fields, ens.dx, before)
        rows.append(row)
    return rows


def _describe_one(args):
    ens, run, path = args
    try:
        return describe_run(ens, run, path), None
    except Exception as err:   # noqa: BLE001 (one broken run must not stop the ensemble)
        return [], f"run {run}: {type(err).__name__}: {err}"


def describe_ensemble(ens: Ensemble, workers: int = 1) -> tuple[pd.DataFrame, list[str]]:
    """Descriptors of every run: a table with one row per snapshot (columns run, time, the
    parameters, then the descriptors) and the list of runs that could not be read.

    workers: number of processes reading runs in parallel. Keep it small (2 or 3) when the
    data are read over a network share."""
    jobs = [(ens, int(r.run), r.path) for r in ens.runs.itertuples()]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_describe_one, jobs))
    else:
        results = [_describe_one(j) for j in jobs]
    rows = [row for res, _ in results for row in res]
    errors = [e for _, e in results if e]
    table = pd.DataFrame(rows)
    if len(table):
        params = ens.runs[["run", *ens.parameters]]
        table = params.merge(table, on="run")
        table = table[["run", "time", *ens.parameters,
                       *[c for c in table.columns if c not in ("run", "time", *ens.parameters)]]]
    return table, errors


def descriptor_columns(table: pd.DataFrame) -> list[str]:
    """The descriptor columns of a table (names field.block.name)."""
    return [c for c in table.columns if c.count(".") >= 2]
