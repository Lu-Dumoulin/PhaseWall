"""Figures of a 1D analysis: kymographs of one run at the central point of each phase (instead of
snapshots), drawn by report.make_phase_images; report.write_report writes them for a 1D ensemble.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from ..report import central_runs, make_phase_images
from ..walls import Walls
from .kymo import Ensemble1D, Field1D, read_kymograph


def picture(a: np.ndarray, f: Field1D) -> tuple[np.ndarray, str]:
    """What is shown of a field: the field (scalar, one-component vector) or its size (two
    components), with its label."""
    a = a[..., None] if a.ndim == 2 else a
    if a.shape[-1] == 1:
        return a[..., 0], f.name
    return np.sqrt(np.sum(a**2, axis=-1)), f"|{f.name}|"


def make_kymographs(w: Walls, ens: Ensemble1D, field: Field1D | None = None) -> Figure:
    """The described part (the last `share`) of one run at the central point of each phase,
    x across and t up, on one colour scale (report.make_phase_images)."""
    f = field or ens.fields[0]
    runs = ens.runs.set_index("run")
    images, label = {}, f.name
    for ph, (values, run) in central_runs(w, ens.runs, ens.parameters).items():
        a = read_kymograph(Path(runs.loc[run, "path"]) / ens.file, [f], ens.julia_order, ens.order)[f.name]
        n_t = a.shape[1]
        first = n_t - int(np.floor(ens.share * n_t))
        img, label = picture(a[:, first:], f)
        extent = [0, ens.dx_of(int(run)) * img.shape[0], ens.t0 + ens.dt * first, ens.t0 + ens.dt * (n_t - 1)]
        images[ph] = (values, img, extent)
    return make_phase_images(images, ens.parameters, label, signed=True, axes=("x", "t"))
