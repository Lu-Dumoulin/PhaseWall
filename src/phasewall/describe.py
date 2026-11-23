"""Generic descriptors of a snapshot: numbers that say what a pattern looks like, whatever the
system, unchanged by a rotation or a mirror of the frame (and by a shift, when the frame is a
whole periodic box).

For every field (arrays [x, y] or [x, y, component], grid spacing dx):

- amp:   log10 of the variance of the field (summed over components): how strong the pattern is.
- shape: the structure factor S(k) on shells of |k|, divided by its sum: which length scales
         are present, whatever the amplitude.
- mom:   mean (scalars that are not relative), standard deviation, skewness and excess kurtosis
         of the values (scalar), of |a| (vector) or of the scalar order 2|Q| (nematic).
- order: vectors and nematics: |<a>| / <|a|>, 1 when everything points the same way.
- topo:  the number of connected domains (per unit area) where the values (as for mom) lie
         above, and below, the mean plus z standard deviations, for z in LEVELS: spots give
         many small domains, stripes a few long ones; the domains of low |p| are the defect
         cores. Minkowski functionals of the same kind are used for patterns and in cosmology.
- flow:  vectors: log10 of the mean square divergence and curl, and the share of the curl:
         splay against bend, asters against vortices.
- vortex: velocities: vortex cores by the Okubo-Weiss parameter W = s_n^2 + s_s^2 - omega^2
         (normal and shear strain against vorticity): the share of the area where
         W < -OW_LEVEL std(W) (rotation dominates) and the number of such cores per unit area.
- defects: polarities and nematics: topological defects counted by their winding number, the
         turn of the angle of the field around each 2 x 2 plaquette of grid points (+-1 for a
         polarity, +-1/2 for a nematic), per unit area: `plus` and `minus`.
- frank: polarities and nematics: the Frank energy of the orientation with one elastic constant
         (K = 1), weighted by the size of the order squared so that defect cores stay finite:
         splay m^2 (div n)^2 and bend m^2 |n x curl n|^2, n the direction of the polarity (m = |p|)
         or the director (m = S = 2|Q|). Columns: log10F, log10 of the mean of splay + bend
         (= <|grad a|^2 - |grad |a||^2>: the distortion of the orientation, not of the size);
         splay_share, <splay> / <splay + bend> (0.5 when F is at the floor); and, for a pair of
         snapshots, change = ln F(t) - ln F(t - pair_dt): negative while the orientation relaxes
         or coarsens, near 0 when steady, of either sign when it oscillates (over the snapshots
         of a run). Computed from the components without choosing a sign of n (see frank_density).
- parts: compares the parts of the frame with each other (Binder's block distribution idea,
         Binder 1981): in every square block of side PARTS_SIDE of the frame side (all
         positions; wrapped in a periodic box), six local numbers: log10 of the variance and
         the mean (in units of the variance and the standard deviation of the whole frame),
         skewness and log10 of the kurtosis of the values (as for mom), log10 of a local
         wavenumber sqrt(<|grad a|^2> / var a), and the anisotropy of the structure tensor
         <grad a grad a^T> (about 1 for stripes, about 0 for spots; components summed, so it
         turns with the field). H is the variance of these numbers over the blocks, averaged
         over the six; the column is log10 H. One state everywhere: the blocks differ only by
         finite-size noise and H is small; two states side by side (hexagons next to stripes,
         spots on a uniform background): H is large. Domains of one state as large as the
         blocks (labyrinth stripes) fall in between. A block flat to within PARTS_VAR_FLOOR of
         the frame variance takes the values of the whole frame for its shape numbers (a
         choice: they are not defined there). Not a default block: on the Swift-Hohenberg and
         Gray-Scott demos it marks coexistence well but does not improve the walls (2026-11-18);
         the slope of log H against log l (Binder's test, -2 for one state, 0 for two) was
         tried as a second column and removed: it was noisy and merged phases.
- dyn:   for a pair of snapshots a time pair_dt apart: the correlation C of the fluctuations
         at the same place (still) and after the best shift (moving), given as the
         decorrelation log10(1 - C + DECOR_FLOOR). A frozen pattern has both at the floor; a
         pattern that travels without changing has still above the floor and moving at it; a
         pattern that changes has both above it. The floor keeps numerical noise in a frozen
         pattern (C = 0.99999) from looking like dynamics.

Periodic and non-periodic frames (see frames.py). A whole periodic box is described with
FFTs (S(k), spectral derivatives), domains joined across the edges, and every circular shift
for the moving correlation. A frame that is not periodic (a crop, or a non-periodic box):
- shape: the field is multiplied by a Tukey window (flat in the middle, falling smoothly to
  zero over TUKEY_ALPHA / 2 of the side at each edge), which makes it periodic in effect
  and stops the edges from leaking power to high k; the power is divided by <w^2> so the
  amplitude is kept. Shells with wavelengths longer than side / min_wavelengths are dropped:
  a crop cannot measure them. Measured on 480 crops of broadband random fields (2026-11-17):
  15-30% off for wavelengths side and side / 2, within 5% from side / 3 to side / 11; at
  shorter wavelengths, where the spectrum has fallen below about a tenth of its peak, the
  remaining leakage of the edges grows (10-15% at side / 13).
- flow: second-order centred finite differences, on the interior points.
- topo: domains are not joined across the edges.
- dyn: the frame at t is compared with the frame at t - pair_dt extended by max_shift on
  every side; the correlation is computed for every shift in [-max_shift, max_shift]^2, each
  over the full frame (Pearson correlation of the frame with the window under it), and the
  largest is `moving`; `still` is the one at zero shift. A pattern that moves by more than
  max_shift during pair_dt looks as if it changed.

Which blocks are computed for a field is set by Field.blocks, by default the blocks of its kind
(fields.DEFAULT_BLOCKS: the common ones for every kind, vortex for a velocity, defects and frank for a
polarity or a nematic). Column names are field.block.name, for example rho.shape.k07;
walls.py weights the blocks.
"""

import re

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.signal.windows import tukey

from .fields import Field
from .frames import Frame, frame_for, prepare
from .io import Ensemble, read_snapshot, run_snapshots
from .parallel import Progress, run_jobs

FLAT = 1e-6           # fields whose fluctuations are below this are flat (a choice)
SHELL_RATIO = 1.2     # each shell of |k| is 1.2 times wider than the previous one (a choice)
DECOR_FLOOR = 1e-3    # correlations above 1 - DECOR_FLOOR count as frozen (a choice)
LEVELS = (-1.0, 0.0, 1.0)   # topo: levels in standard deviations from the mean (a choice)
TUKEY_ALPHA = 0.5     # non-periodic S(k): tapered share of the window (a choice: 4x less leakage
                      # than no window on Swift-Hohenberg crops, 2026-11-17)
FLAT_RELATIVE = 1e-3  # vectors and nematics whose fluctuations are below 0.1% of their size are uniform
                      # (a choice: the numerical noise of a uniformly polarised state would otherwise be
                      # described as a pattern, far from the floors of an exactly uniform one)
PARTS_SIDE = 0.5            # parts: block side as a fraction of the frame side (a choice)
PARTS_VAR_FLOOR = 1e-6      # parts: a block whose variance is below this share of the frame
                            # variance is flat (a choice)
PARTS_H_FLOOR = 1e-6        # parts: H below this counts as 0 in log10 H (a choice)
OW_LEVEL = 0.2        # vortex cores: W < -0.2 std(W), the usual threshold of the Okubo-Weiss criterion
                      # (Isern-Fontanet et al. 2003; a choice)


# ---------------------------------------------------------------- spectra
def k_shells(n: int, dx: float, ratio: float = SHELL_RATIO, min_wavelengths: float | None = None
             ) -> np.ndarray:
    """Edges of the shells of |k| for a frame of n points: from half the lowest wavenumber
    2 pi / L (or, with min_wavelengths, from the shell of the wavenumber min_wavelengths
    2 pi / L) to pi / dx, each shell `ratio` times wider than the previous one but never
    narrower than 2 pi / L, so that no shell is empty."""
    dk = 2 * np.pi / (n * dx)
    edges = [0.5 * dk if min_wavelengths is None else (min_wavelengths - 0.5) * dk]
    while edges[-1] * ratio < np.pi / dx and edges[-1] + dk < np.pi / dx:
        edges.append(max(edges[-1] * ratio, edges[-1] + dk))
    edges.append(np.pi / dx)
    return np.array(edges)


def _components(a: np.ndarray) -> np.ndarray:
    return a[..., None] if a.ndim == 2 else a


def window(nx: int, ny: int) -> np.ndarray:
    """Tukey window of a non-periodic frame, scaled so that <w^2> = 1 (the power is kept)."""
    w = np.outer(tukey(nx, TUKEY_ALPHA), tukey(ny, TUKEY_ALPHA))
    return w / np.sqrt(np.mean(w**2))


def structure_factor(a: np.ndarray, dx: float, edges: np.ndarray, periodic: bool = True) -> np.ndarray:
    """S(k) = dx^2 / (nx ny) <|a_k|^2> averaged over each shell, components summed (so a
    global rotation of the vectors leaves it unchanged); the mean is removed first, and a
    non-periodic frame is multiplied by window()."""
    f = _components(a)
    nx, ny = f.shape[:2]
    w = 1.0 if periodic else window(nx, ny)
    power = sum(np.abs(np.fft.fft2(w * (f[..., c] - f[..., c].mean()))) ** 2 for c in range(f.shape[2]))
    k = np.hypot(*np.meshgrid(2 * np.pi * np.fft.fftfreq(nx, dx), 2 * np.pi * np.fft.fftfreq(ny, dx),
                              indexing="ij"))
    shell = np.digitize(k.ravel(), edges) - 1
    inside = (shell >= 0) & (shell < len(edges) - 1)
    n = len(edges) - 1
    total = np.bincount(shell[inside], weights=power.ravel()[inside], minlength=n)
    count = np.bincount(shell[inside], minlength=n)
    return dx**2 / (nx * ny) * total / np.maximum(count, 1)


def gradients(a: np.ndarray, dx: float, periodic: bool = True) -> tuple[np.ndarray, ...]:
    """(d_x a_x, d_y a_x, d_x a_y, d_y a_y) of a vector field: spectral derivatives in a periodic
    frame, second-order centred differences on the interior points otherwise."""
    ax, ay = a[..., 0], a[..., 1]
    if not periodic:
        def ddx(f):
            return (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dx)

        def ddy(f):
            return (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dx)
        return ddx(ax), ddy(ax), ddx(ay), ddy(ay)
    nx, ny = a.shape[:2]
    kx = 2j * np.pi * np.fft.fftfreq(nx, dx)[:, None]
    ky = 2j * np.pi * np.fft.fftfreq(ny, dx)[None, :]
    fx, fy = np.fft.fft2(ax), np.fft.fft2(ay)
    return tuple(np.fft.ifft2(k * f).real for k, f in ((kx, fx), (ky, fx), (kx, fy), (ky, fy)))


def div_curl(a: np.ndarray, dx: float, periodic: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Divergence and curl of a vector field (derivatives as in gradients)."""
    dxax, dyax, dxay, dyay = gradients(a, dx, periodic)
    return dxax + dyay, dxay - dyax


def okubo_weiss(a: np.ndarray, dx: float, periodic: bool = True) -> np.ndarray:
    """W = s_n^2 + s_s^2 - omega^2, with s_n = d_x v_x - d_y v_y, s_s = d_x v_y + d_y v_x,
    omega = d_x v_y - d_y v_x: negative where rotation dominates strain (vortex cores),
    positive where strain dominates. Unchanged by rotations and mirrors of the frame."""
    dxax, dyax, dxay, dyay = gradients(a, dx, periodic)
    return (dxax - dyay) ** 2 + (dxay + dyax) ** 2 - (dxay - dyax) ** 2


def vortex_cores(a: np.ndarray, dx: float, periodic: bool = True) -> dict[str, float]:
    """vortex block: share of the area with W < -OW_LEVEL std(W), and those cores per unit area
    (0 and 0 when W does not vary: no flow, or a uniform one)."""
    w = okubo_weiss(a, dx, periodic)
    sd = w.std()
    if sd < FLAT**2:
        return {"share": 0.0, "cores": 0.0}
    core = w < -OW_LEVEL * sd
    return {"share": float(core.mean()), "cores": count_domains(core, periodic) / (core.size * dx**2)}


def wrapped(d: np.ndarray, period: float) -> np.ndarray:
    """Smallest signed angle difference for angles defined modulo `period`."""
    return d - period * np.round(d / period)


def winding_charges(a: np.ndarray, symmetry: str, periodic: bool = True) -> np.ndarray:
    """Winding number of every 2 x 2 plaquette of grid points: the turn of the angle of the field
    around it over 2 pi. A polarity (angle atan2(a_y, a_x), period 2 pi) gives +-1, a nematic
    (director angle atan2(Q_xy, Q_xx) / 2, period pi) gives +-1/2; most plaquettes give 0.
    Adapted from earlier code by the same author."""
    if symmetry == "nematic":
        theta, period = 0.5 * np.arctan2(a[..., 1], a[..., 0]), np.pi
    else:
        theta, period = np.arctan2(a[..., 1], a[..., 0]), 2 * np.pi
    if periodic:
        t00, t10 = theta, np.roll(theta, -1, axis=0)
        t01, t11 = np.roll(theta, -1, axis=1), np.roll(t10, -1, axis=1)
    else:                                       # the plaquettes inside the frame only
        t00, t10, t01, t11 = theta[:-1, :-1], theta[1:, :-1], theta[:-1, 1:], theta[1:, 1:]
    turn = (wrapped(t10 - t00, period) + wrapped(t11 - t10, period)
            + wrapped(t01 - t11, period) + wrapped(t00 - t01, period))
    return turn / (2 * np.pi)


def defect_counts(a: np.ndarray, symmetry: str, dx: float, periodic: bool = True) -> dict[str, float]:
    """defects block: positive and negative defects per unit area (charge above half of the
    unit charge in size; 0 and 0 for a flat field, whose angle is not defined). Every plaquette
    counts, wherever |a| is small (a choice: in a disordered region the count measures the
    disorder)."""
    if np.sqrt(np.mean(np.sum(a**2, axis=-1))) < FLAT:
        return {"plus": 0.0, "minus": 0.0}
    q = winding_charges(a, symmetry, periodic)
    unit = 0.5 if symmetry == "nematic" else 1.0
    area = q.size * dx**2
    return {"plus": float(np.sum(q > unit / 2)) / area, "minus": float(np.sum(q < -unit / 2)) / area}


def frank_density(a: np.ndarray, symmetry: str, dx: float, periodic: bool = True
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Splay and bend densities m^2 (div n)^2 and m^2 |n x curl n|^2 of a polarity or a nematic
    (frank block, module docstring), on the points where the derivatives are taken. With
    a = (p_x, p_y) or (Q_xx, Q_xy) and theta the angle of n: g = (a_x grad a_y - a_y grad a_x) / |a|
    = m grad theta (polarity: a = m (cos theta, sin theta); nematic: a = |Q| (cos 2 theta,
    sin 2 theta) and S = 2 |Q|), and with c = cos 2 theta, s = sin 2 theta (polarity: from a_x^2 - a_y^2
    and 2 a_x a_y over |a|^2; nematic: a / |a|):
    splay = ((1 - c) g_x^2 + (1 + c) g_y^2 - 2 s g_x g_y) / 2,
    bend  = ((1 + c) g_x^2 + (1 - c) g_y^2 + 2 s g_x g_y) / 2, splay + bend = |g|^2.
    Points where |a| < FLAT count 0 (no orientation)."""
    dxax, dyax, dxay, dyay = gradients(a, dx, periodic)
    if not periodic:
        a = a[1:-1, 1:-1]
    ax, ay = a[..., 0], a[..., 1]
    size2 = ax**2 + ay**2
    ok = size2 > FLAT**2
    size = np.sqrt(np.where(ok, size2, 1.0))
    gx = np.where(ok, (ax * dxay - ay * dxax) / size, 0.0)
    gy = np.where(ok, (ax * dyay - ay * dyax) / size, 0.0)
    if symmetry == "nematic":
        c, s = ax / size, ay / size
    else:
        c, s = (ax**2 - ay**2) / size**2, 2 * ax * ay / size**2
    c, s = np.where(ok, c, 0.0), np.where(ok, s, 0.0)
    splay = 0.5 * ((1 - c) * gx**2 + (1 + c) * gy**2 - 2 * s * gx * gy)
    bend = 0.5 * ((1 + c) * gx**2 + (1 - c) * gy**2 + 2 * s * gx * gy)
    return splay, bend


def frank_energy(a: np.ndarray, f: Field, dx: float, periodic: bool = True) -> tuple[float, float]:
    """(F, mean splay) of one field (prepared as for every block)."""
    splay, bend = frank_density(prepared(a, f), f.symmetry, dx, periodic)
    return float(np.mean(splay + bend)), float(np.mean(splay))


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


# ---------------------------------------------------------------- parts of a frame
def box_means(x: np.ndarray, side: int, periodic: bool = True) -> np.ndarray:
    """Mean of x over every square block of `side` points: one value per block position
    (n x n positions in a periodic frame, wrapping across the edges; (n - side + 1)^2 inside a
    non-periodic one). Sums by a cumulative table, so every block costs the same."""
    if periodic:
        x = np.pad(x, ((0, side - 1), (0, side - 1)), mode="wrap")
    c = np.zeros((x.shape[0] + 1, x.shape[1] + 1))
    c[1:, 1:] = np.cumsum(np.cumsum(x, axis=0), axis=1)
    s = c[side:, side:] - c[:-side, side:] - c[side:, :-side] + c[:-side, :-side]
    return s / side**2


def _parts_numbers(z: np.ndarray, grads: list[tuple[np.ndarray, np.ndarray]], comps: np.ndarray,
                   side: int, periodic: bool, whole: np.ndarray) -> np.ndarray:
    """The six local numbers of the parts block (module docstring) for every block: array
    (6, positions). z: values in units of the frame (mean 0, variance 1); grads: (d_x, d_y) of
    each component; comps: the components (for the variance under the wavenumber); whole: the
    six numbers of the whole frame, used for the shape numbers of flat blocks."""
    m1, m2, m3, m4 = (box_means(z**p, side, periodic) for p in (1, 2, 3, 4))
    var = np.maximum(m2 - m1**2, 0.0)
    c3 = m3 - 3 * m1 * m2 + 2 * m1**3
    c4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
    flat = var < PARTS_VAR_FLOOR
    safe = np.where(flat, 1.0, var)
    skew = np.where(flat, whole[2], c3 / safe**1.5)
    kurt = np.where(flat, whole[3], np.log10(np.maximum(c4 / safe**2, PARTS_VAR_FLOOR)))
    jxx = sum(box_means(gx * gx, side, periodic) for gx, _ in grads)
    jyy = sum(box_means(gy * gy, side, periodic) for _, gy in grads)
    jxy = sum(box_means(gx * gy, side, periodic) for gx, gy in grads)
    cvar = sum(np.maximum(box_means(c * c, side, periodic) - box_means(c, side, periodic) ** 2, 0.0)
               for c in comps)
    trace = jxx + jyy
    still = flat | (trace <= 0) | (cvar <= 0)
    wave = np.where(still, whole[4], 0.5 * np.log10(np.where(still, 1.0, trace) / np.where(still, 1.0, cvar)))
    aniso = np.where(still, whole[5], np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / np.where(still, 1.0, trace))
    return np.stack([np.log10(np.maximum(var, PARTS_VAR_FLOOR)), m1, skew, kurt, wave, aniso]).reshape(6, -1)


def parts(a: np.ndarray, values: np.ndarray, dx: float, periodic: bool = True,
          side: float = PARTS_SIDE) -> dict[str, float]:
    """The parts block of one field (module docstring): {"log10H": ...}.
    a: the prepared field [x, y] or [x, y, component]; values: what mom describes (the field,
    |a| or 2|Q|); side: block side as a fraction of the frame side."""
    c = _components(a)
    comps = [c[..., i] for i in range(c.shape[2])]
    grads = [gradients(np.stack([ci, ci], axis=-1), dx, periodic)[:2] for ci in comps]
    if not periodic:            # finite differences exist on the interior points only
        values, comps = values[1:-1, 1:-1], [ci[1:-1, 1:-1] for ci in comps]
    std = float(values.std())
    if std < FLAT:
        return {"log10H": float(np.log10(PARTS_H_FLOOR))}
    z = (values - values.mean()) / std
    trace = float(sum(np.mean(gx**2 + gy**2) for gx, gy in grads))
    cvar = float(sum(ci.var() for ci in comps))
    _, _, skew, kurt = moments(z)
    whole = np.array([0.0, 0.0, skew, np.log10(max(kurt + 3, PARTS_VAR_FLOOR)),
                      0.5 * np.log10(trace / cvar) if trace > 0 and cvar > 0 else 0.0, 0.0])
    n = max(2, round(side * min(z.shape)))
    numbers = _parts_numbers(z, grads, np.array(comps), n, periodic, whole)
    return {"log10H": float(np.log10(np.mean(np.var(numbers, axis=1)) + PARTS_H_FLOOR))}


# ---------------------------------------------------------------- one field
def prepared(a: np.ndarray, f: Field) -> np.ndarray:
    """The array that is described: (a - <a>) / <a> for a relative scalar; a vector or nematic
    whose fluctuations are below FLAT_RELATIVE of its size (rms of |a|) is replaced by its mean,
    exactly uniform, so that numerical noise is not described as a pattern."""
    if f.relative:
        m = a.mean()
        return (a - m) / m if m != 0 else a - m
    if f.symmetry != "scalar" and a.ndim == 3:
        size = np.sqrt(np.mean(np.sum(a**2, axis=-1)))
        fluct = np.sqrt(np.sum(a.var(axis=(0, 1))))
        if size > 0 and fluct < FLAT_RELATIVE * size:
            return np.broadcast_to(a.mean(axis=(0, 1)), a.shape).copy()
    return a


def field_descriptors(a: np.ndarray, f: Field, dx: float, edges: np.ndarray,
                      periodic: bool = True) -> dict[str, float]:
    """Descriptors of one field of one snapshot: the blocks in f.selected except dyn (see the
    module docstring)."""
    use = set(f.selected)
    a = prepared(a, f)
    c = _components(a)
    variance = float(np.sum(c.var(axis=(0, 1))))
    flat = np.sqrt(variance) < FLAT
    out = {}
    if "amp" in use:
        out[f"{f.name}.amp.log10var"] = _log10(variance)
    if "shape" in use:
        S = np.zeros(len(edges) - 1) if flat else structure_factor(a, dx, edges, periodic)
        shape = S / S.sum() if S.sum() > 0 else S
        out |= {f"{f.name}.shape.k{j:02d}": float(s) for j, s in enumerate(shape)}
    if f.symmetry == "scalar":
        values = a
    else:
        size = np.hypot(c[..., 0], c[..., 1])
        values = 2 * size if f.symmetry == "nematic" else size
    if "topo" in use:
        area = values.shape[0] * values.shape[1] * dx**2
        out |= {f"{f.name}.topo.{k}": v for k, v in domain_counts(values, area, periodic).items()}
    if "mom" in use:
        mean, std, skew, kurt = moments(values)
        if not f.relative:
            out[f"{f.name}.mom.mean"] = mean
        out |= {f"{f.name}.mom.std": std, f"{f.name}.mom.skew": skew, f"{f.name}.mom.kurt": kurt}
    if "order" in use:
        size_mean = float(np.hypot(c[..., 0], c[..., 1]).mean())
        mean_vec = float(np.hypot(c[..., 0].mean(), c[..., 1].mean()))
        out[f"{f.name}.order.global"] = mean_vec / size_mean if size_mean > FLAT else 0.0
    if "flow" in use:
        d, w = div_curl(a, dx, periodic)
        d2, w2 = float(np.mean(d**2)), float(np.mean(w**2))
        out[f"{f.name}.flow.log10div2"] = _log10(d2)
        out[f"{f.name}.flow.log10curl2"] = _log10(w2)
        # 0.5 when neither is present (a choice): no preference for splay or bend
        out[f"{f.name}.flow.curl_share"] = w2 / (d2 + w2) if d2 + w2 > FLAT**2 else 0.5
    if "frank" in use:
        F, splay = frank_energy(a, f, dx, periodic)
        out[f"{f.name}.frank.log10F"] = _log10(F)
        # 0.5 when there is no distortion (a choice): no preference for splay or bend
        out[f"{f.name}.frank.splay_share"] = splay / F if F > FLAT**2 else 0.5
    if "vortex" in use:
        out |= {f"{f.name}.vortex.{k}": v for k, v in vortex_cores(a, dx, periodic).items()}
    if "defects" in use:
        out |= {f"{f.name}.defects.{k}": v for k, v in defect_counts(a, f.symmetry, dx, periodic).items()}
    if "parts" in use:
        out |= {f"{f.name}.parts.{k}": v for k, v in parts(a, values, dx, periodic).items()}
    return out


def correlations(before: np.ndarray, after: np.ndarray, f: Field, shift: int | None = None
                 ) -> tuple[float, float]:
    """Correlation of the fluctuations of one field between two snapshots, at the same place
    (still) and after the shift that maximises it (moving). Both are 1 when either snapshot
    is flat (a choice: nothing changes).

    shift None: periodic frames of the same size, every circular shift is tried.
    shift s: `before` is the frame extended by s points on every side; every shift in
    [-s, s]^2 is tried, and each correlation is the Pearson correlation of `after` with the
    window of `before` under it (normalised over the frame, not over the whole box)."""
    y = _components(prepared(after, f))
    x = _components(prepared(before, f))
    y = y - y.mean(axis=(0, 1))
    npts = y[..., 0].size
    if shift is None:
        x = x - x.mean(axis=(0, 1))
        if min(np.sum(x**2), np.sum(y**2)) / npts < FLAT**2:
            return 1.0, 1.0
        norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
        cross = sum(np.fft.ifft2(np.conj(np.fft.fft2(x[..., i])) * np.fft.fft2(y[..., i])).real
                    for i in range(x.shape[2]))
        return float(np.sum(x * y) / norm), float(cross.max() / norm)
    # windowed (Lewis 1995, fast normalised cross-correlation): sums over every window of the
    # size of the frame by FFT convolution; result shape (2s + 1, 2s + 1), zero shift at [s, s]
    ones = np.ones(y.shape[:2])
    num = sum(fftconvolve(x[..., i], y[::-1, ::-1, i], mode="valid") for i in range(x.shape[2]))
    s1 = [fftconvolve(x[..., i], ones, mode="valid") for i in range(x.shape[2])]
    s2 = [fftconvolve(x[..., i] ** 2, ones, mode="valid") for i in range(x.shape[2])]
    var_x = sum(b - a**2 / npts for a, b in zip(s1, s2))          # sum of squared fluctuations per window
    var_y = float(np.sum(y**2))
    if var_y / npts < FLAT**2 or var_x[shift, shift] / npts < FLAT**2:
        return 1.0, 1.0
    corr = num / np.sqrt(np.maximum(var_x, FLAT**2 * npts) * var_y)
    return float(corr[shift, shift]), float(corr.max())


def pair_descriptors(before: np.ndarray, after: np.ndarray, f: Field, shift: int | None = None
                     ) -> dict[str, float]:
    """Decorrelation log10(1 - C + DECOR_FLOOR) at the same place and after the best shift."""
    still, moving = correlations(before, after, f, shift)
    return {f"{f.name}.dyn.still": float(np.log10(max(1 - still, 0) + DECOR_FLOOR)),
            f"{f.name}.dyn.moving": float(np.log10(max(1 - moving, 0) + DECOR_FLOOR))}


# ---------------------------------------------------------------- frames, runs and ensembles
def describe_snapshot(snapshot: dict[str, np.ndarray], fields: list[Field], dx: float,
                      before: dict[str, np.ndarray] | None = None, periodic: bool = True,
                      shift: int = 0, min_wavelengths: float | None = None) -> dict[str, float]:
    """All descriptors of one frame; with `before` (the frame pair_dt earlier), also the
    dynamics. A non-periodic frame (periodic False) uses the rules of the module docstring;
    then `before` must be extended by `shift` points on every side, and min_wavelengths sets
    the first shell of S(k)."""
    n = snapshot[fields[0].name].shape[0]
    edges = k_shells(n, dx, min_wavelengths=None if periodic else min_wavelengths)
    out = {}
    for f in fields:
        out |= field_descriptors(snapshot[f.name], f, dx, edges, periodic)
        if before is not None and "dyn" in f.selected:
            out |= pair_descriptors(before[f.name], snapshot[f.name], f, None if periodic else shift)
        if before is not None and "frank" in f.selected:
            b = before[f.name]
            if not periodic and shift > 0:                 # the frame inside the extended one
                b = b[shift:-shift, shift:-shift]
            now, earlier = (max(frank_energy(x, f, dx, periodic)[0], FLAT**2) for x in (snapshot[f.name], b))
            out[f"{f.name}.frank.change"] = float(np.log(now) - np.log(earlier))
    return out


def run_frame(ens: Ensemble, shape: tuple[int, ...], dx: float) -> Frame:
    """The frame of one run (see frames.py)."""
    return frame_for(shape, dx, ens.periodic, ens.crop, ens.dx_target, ens.max_shift)


def describe_run(ens: Ensemble, run: int, path: str, dx: float | None = None) -> list[dict]:
    """One row per described snapshot of a run: the last `ens.last` snapshots. dx: grid
    spacing of this run (default ens.dx)."""
    dx = ens.dx if dx is None else dx
    snaps = run_snapshots(ens, path)
    times = np.array([t for t, _ in snaps])
    rows, frame = [], None
    for t, p in snaps[-ens.last:]:
        raw = read_snapshot(p, ens.fields, ens.julia_order)
        frame = frame or run_frame(ens, raw[ens.fields[0].name].shape, dx)
        before = None
        if ens.pair_dt is not None:
            j = np.flatnonzero(np.isclose(times, t - ens.pair_dt, rtol=0, atol=1e-6 * ens.pair_dt))
            if len(j):
                old = read_snapshot(snaps[j[0]][1], ens.fields, ens.julia_order)
                before = prepare(old, ens.fields, frame, margin=not frame.periodic)
        row = {"run": run, "time": t} | describe_snapshot(
            prepare(raw, ens.fields, frame), ens.fields, frame.dx, before, frame.periodic, frame.shift,
            ens.min_wavelengths)
        rows.append(row)
    return rows


def _describe_one(args):
    ens, run, path, dx = args
    try:
        return describe_run(ens, run, path, dx), None
    except Exception as err:   # noqa: BLE001 (one broken run must not stop the ensemble)
        return [], f"run {run}: {type(err).__name__}: {err}"


def describe_ensemble(ens: Ensemble, workers: int = 1, progress: Progress | None = None
                      ) -> tuple[pd.DataFrame, list[str]]:
    """Descriptors of every run: a table with one row per snapshot (columns run, time, the
    parameters, then the descriptors) and the list of runs that could not be read.

    workers: number of processes reading runs in parallel. Keep it small (2 or 3) when the
    data are read over a network share. progress: see parallel.py.

    ValueError if the runs give different S(k) shells (frames of different sizes or grid
    spacings): their columns would not mean the same wavenumbers. Set crop and dx_target."""
    dxs = ens.runs["dx"] if "dx" in ens.runs else [ens.dx] * len(ens.runs)
    jobs = [(ens, int(r.run), r.path, float(d)) for r, d in zip(ens.runs.itertuples(), dxs)]
    results = run_jobs(_describe_one, jobs, workers, progress, "describe")
    shells = {tuple(c for c in rows[0] if ".shape." in c) for rows, _ in results if rows}
    if len(shells) > 1:
        raise ValueError("the runs give different S(k) shells (frames of different sizes or grid "
                         "spacings), so their columns would not mean the same wavenumbers: set "
                         "`crop` and `dx_target` in the configuration so that every run is described "
                         "on the same frame")
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


def combine_tables(hand: pd.DataFrame, learned: pd.DataFrame) -> pd.DataFrame:
    """Hand-made and learned descriptions of the same snapshots side by side: rows matched on
    (run, time), the columns of both kept under their own block names (field.block.name and
    cnn.h.dNNN). walls.feature_matrix weighs every block alike, so the learned block counts as
    one block among the hand-made ones (a choice). Snapshots in only one table are dropped."""
    keep = [c for c in learned.columns if c in ("run", "time") or c not in hand.columns]
    return hand.merge(learned[keep], on=["run", "time"], how="inner")


# ---------------------------------------------------------------- explanation of the columns
_VALUES = {"scalar": "{f}", "vector": "|{f}|", "nematic": "2|{f}| (scalar order S)"}


def explain(column: str, fields: list[Field] | None = None) -> str:
    """A sentence that says what a descriptor column measures, from its name field.block.name
    (and the kind of the field, when `fields` are given): for example "u.topo.below-1" gives
    "domains per unit area where u is below its mean minus 1 standard deviation". Columns of the
    learned description (cnn.h.dNNN), of a table given (table.x.name) and principal components
    (pc.N) are explained too; any other name is returned as it is."""
    parts = column.split(".")
    if len(parts) == 2 and parts[0] == "pc":
        return f"principal component {parts[1]} of the description"
    if len(parts) < 3:
        return column
    f, block, name = parts[0], parts[1], ".".join(parts[2:])
    if f == "cnn" and block == "h":
        return f"component {int(name[1:])} of h, the learned description" if name[1:].isdigit() else column
    if f == "table":
        return f"column {name} of the table of descriptors given"
    kind = next((x.symmetry for x in fields or [] if x.name == f), None)
    v = _VALUES[kind].format(f=f) if kind else f"the values of {f}"
    if block == "amp":
        return f"log10 of the variance of {f} (how strong the pattern is)"
    if block == "shape" and name.startswith("k") and name[1:].isdigit():
        return (f"share of the power of {f} in shell {int(name[1:])} of |k| (S(k) divided by its sum; "
                "shells from long to short wavelengths)")
    if block == "mom":
        what = {"mean": "mean", "std": "standard deviation", "skew": "skewness", "kurt": "excess kurtosis"}
        if name in what:
            return f"{what[name]} of {v}"
    if block == "topo":
        m = re.fullmatch(r"(above|below)([+-]\d+)", name)
        if m:
            z = int(m.group(2))
            level = "its mean" if z == 0 else f"its mean {'plus' if z > 0 else 'minus'} {abs(z)} standard deviation"
            return f"domains per unit area where {v} is {m.group(1)} {level}"
    if block == "order" and name == "global":
        return f"global order of {f}: |<{f}>| / <|{f}|>, 1 when everything points the same way"
    if block == "flow":
        what = {"log10div2": f"log10 of the mean square divergence of {f}",
                "log10curl2": f"log10 of the mean square curl of {f}",
                "curl_share": f"share of the curl in the derivatives of {f}: 0 splay or sources, 1 bend or vortices"}
        if name in what:
            return what[name]
    if block == "frank":
        what = {"log10F": f"log10 of the Frank energy of the orientation of {f} (one constant, weighted by the order "
                          "squared): how distorted the orientation is",
                "splay_share": f"share of splay in the Frank energy of {f}: 0 pure bend, 1 pure splay",
                "change": f"change of the Frank energy of {f} over the lag, ln F(t) - ln F(t - lag): negative while "
                          "it relaxes or coarsens, near 0 when steady"}
        if name in what:
            return what[name]
    if block == "vortex":
        what = {"share": f"share of the area in vortex cores of {f} (Okubo-Weiss W < -0.2 std W)",
                "cores": f"vortex cores of {f} per unit area"}
        if name in what:
            return what[name]
    if block == "defects" and name in ("plus", "minus"):
        return f"{'positive' if name == 'plus' else 'negative'} topological defects of {f} per unit area (winding number)"
    if block == "parts":
        what = {"log10H": f"how much the parts of the frame differ: log10 of the spread of local numbers of {f} "
                          "over blocks of half the side (large when two states sit side by side)"}
        if name in what:
            return what[name]
    if block == "dyn":
        what = {"still": f"change of {f} over the lag, at the same place: log10(1 - C + 0.001)",
                "moving": f"change of {f} over the lag, following the pattern (best shift): log10(1 - C + 0.001)"}
        if name in what:
            return what[name]
    return column
