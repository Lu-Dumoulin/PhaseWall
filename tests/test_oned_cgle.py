"""1D example: the complex Ginzburg-Landau equation, and a whole ensemble described."""

import numpy as np
import pytest

from phasewall.oned import cgle
from phasewall.oned.describe1d import describe_ensemble
from phasewall.oned.kymo import load_ensemble1d

SMALL = {"n": 128, "length": 100.0, "t_relax": 300.0, "rows": 300}   # 150 rows described: 8 windows of 18


@pytest.mark.parametrize(("b", "c", "label"), [(0.0, -0.5, "plane waves"), (3.0, -0.5, "phase turbulence"),
                                               (3.0, -1.1, "defect turbulence")])
def test_three_regimes(b, c, label):
    a = cgle.simulate(b, c, 0, **SMALL)
    assert a.shape == (128, 300)
    assert cgle.reference_label(a) == label


def test_plane_wave_keeps_its_size():
    a = cgle.simulate(0.0, -0.5, 1, **SMALL)
    assert np.abs(a).std() < 1e-3 and np.abs(a).mean() == pytest.approx(1.0, abs=0.05)


def test_ensemble_is_written_and_described(tmp_path):
    runs = cgle.make_ensemble(tmp_path, b=[0.0, 3.0], c=[-0.5, -1.1], seeds=(0,), **SMALL)
    assert len(runs) == 4 and set(runs.reference) == {"plane waves", "phase turbulence", "defect turbulence"}
    ens = load_ensemble1d(tmp_path / "phasewall.toml")
    table, errors = describe_ensemble(ens)
    assert errors == []
    assert len(table) == 4 * ens.windows
    assert list(table.columns[:4]) == ["run", "time", "b", "c"]
    assert table.time.max() == pytest.approx(300.0 + 0.5 * 299)
    slips = table.groupby("run")["A.defects.slips"].mean()
    assert slips[runs.reference == "defect turbulence"].min() > 0
    assert slips[runs.reference != "defect turbulence"].max() == 0
