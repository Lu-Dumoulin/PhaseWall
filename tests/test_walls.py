"""Tests of walls.py on synthetic descriptor tables with a known answer: one smooth phase (no
wall), two and three phases (walls on the right links), a bistable point (mixed), one phase in
two places joined across the phase between them, and the
building blocks (energy distance, halves, cuts on a chain)."""

import numpy as np
import pandas as pd
import pytest

from phasewall import walls as w

NA, NB = 10, 8          # grid of NA x NB parameter points
RUNS, SNAPS = 2, 5      # runs per point, snapshots per run
DIM = 6                 # descriptor columns


def table(mean_of, noise: float = 1.0, seed: int = 0, run_mean_of=None) -> pd.DataFrame:
    """One row per snapshot on the NA x NB grid (a, b in 0, 1, ...). The descriptors are
    Gaussian with mean mean_of(a, b) (a vector of length DIM) and standard deviation `noise`;
    run_mean_of(a, b, r), if given, sets the mean of run r instead."""
    rng = np.random.default_rng(seed)
    rows, run = [], 0
    for a in range(NA):
        for b in range(NB):
            for r in range(RUNS):
                m = run_mean_of(a, b, r) if run_mean_of else mean_of(a, b)
                for t in range(SNAPS):
                    x = m + noise * rng.standard_normal(DIM)
                    rows.append({"run": run, "time": float(t), "a": float(a), "b": float(b)}
                                | {f"f.blk.x{j}": x[j] for j in range(DIM)})
                run += 1
    return pd.DataFrame(rows)


def walls_between(result: w.Walls) -> set:
    """The walls as pairs of parameter points."""
    n, e = result.nodes, result.edges[result.edges.wall]
    return {(tuple(n.loc[a, ["a", "b"]]), tuple(n.loc[b, ["a", "b"]])) for a, b in zip(e.a, e.b)}


def test_energy_distance():
    rng = np.random.default_rng(0)
    A, B = rng.standard_normal((200, 3)), rng.standard_normal((200, 3))
    assert w.energy_distance(A, B) < 0.15
    far = w.energy_distance(A, B + [3, 0, 0])
    # for a large shift s, 2 <|x - y|> - <|x - x'|> - <|y - y'|> tends to 2 s - 2 * 1.6
    assert far == pytest.approx(np.sqrt(2 * 3 - 2 * 1.6), rel=0.15)
    assert w.energy_distance(A, B + [6, 0, 0]) > far


def test_halves_by_run_or_by_time():
    run = np.array([4, 4, 7, 7, 9, 9])
    time = np.array([0, 1, 0, 1, 0, 1.0])
    assert w.halves(run, time).tolist() == [True, True, False, False, True, True]
    one_run = np.zeros(4, dtype=int)
    assert w.halves(one_run, np.array([3, 0, 2, 1.0])).tolist() == [False, True, False, True]
    assert w.halves(one_run, np.zeros(4)).tolist() == [True, False, True, False]


def test_feature_matrix():
    t = pd.DataFrame({"run": 0, "time": 0.0, "f.one.x": [1.0, 2.0, np.nan, 3.0],
                      "f.four.a": [1.0, 2, 3, 4], "f.four.b": [4.0, 3, 2, 1],
                      "f.four.c": [0.0, 1, 0, 1], "f.four.d": [1.0, 1, 2, 2], "f.flat.x": 5.0})
    X, names = w.feature_matrix(t)
    assert names == ["f.one.x", "f.four.a", "f.four.b", "f.four.c", "f.four.d"]
    assert X[2, 0] == pytest.approx((2 - 2) / np.std([1, 2, 2, 3]))      # NaN -> median 2
    assert np.allclose(X.mean(axis=0), 0)
    # a block of 4 columns is divided by 2, so it weighs as much as a block of 1
    assert np.allclose(X.std(axis=0), [1, 0.5, 0.5, 0.5, 0.5])
    X, _ = w.feature_matrix(t, weight_blocks=False)
    assert np.allclose(X.std(axis=0), 1)


def test_cut_of_a_chain_with_one_large_jump():
    """Chain of 10 points; the link 5-6 jumps 10 times more than the others."""
    a, b = np.arange(9), np.arange(1, 10)
    log_jump = np.log(np.where(a == 5, 10.0, 1.0) * (1 + 0.05 * np.sin(a)))
    region, tried, wall = w.split_regions(10, a, b, log_jump, min_phase=2, alpha=0.2)
    assert wall.tolist() == [i == 5 for i in range(9)]
    assert region.tolist() == [0] * 6 + [1] * 4      # region 0 is the larger side
    assert tried[0]["accepted"] and tried[0]["crossing_links"] == 1


@pytest.mark.parametrize("seed", range(5))
def test_no_wall_in_pure_noise(seed):
    """The same distribution everywhere. Before the cuts were tested on other halves than
    those that placed them, 2 or 3 phases came out here for about half of the seeds."""
    res = w.find_walls(table(lambda a, b: np.zeros(DIM), seed=seed), ["a", "b"])
    assert res.n_phases == 1
    assert not res.edges.wall.any()


def test_no_wall_in_one_smooth_phase():
    """The mean drifts smoothly across the grid, by less than the noise between neighbours."""
    res = w.find_walls(table(lambda a, b: np.full(DIM, 0.3 * a + 0.2 * b)), ["a", "b"])
    assert res.n_phases == 1
    assert not res.edges.wall.any()
    assert not res.nodes.mixed.any()
    assert len(res.cuts) >= 1 and not res.cuts.accepted.any()


def test_two_phases():
    """The mean jumps by 3 noise widths between a = 5 and a = 6."""
    res = w.find_walls(table(lambda a, b: np.full(DIM, 3.0 * (a > 5))), ["a", "b"])
    assert res.n_phases == 2
    assert walls_between(res) == {((5.0, b), (6.0, b)) for b in range(NB)}
    left = res.nodes.a <= 5
    assert res.nodes.loc[left, "phase"].nunique() == 1
    assert res.nodes.loc[~left, "phase"].nunique() == 1
    assert res.nodes.loc[left, "phase"].iloc[0] != res.nodes.loc[~left, "phase"].iloc[0]
    assert (res.nodes.runs == RUNS).all() and (res.nodes.snapshots == RUNS * SNAPS).all()


def test_three_phases_on_a_smooth_background():
    """A smooth drift everywhere, plus a jump at a = 5 | 6 and, for a >= 6, another at b = 3 | 4
    along other descriptors."""
    def mean(a, b):
        m = np.full(DIM, 0.2 * a + 0.1 * b)
        m[:3] += 3.0 * (a > 5)
        m[3:] += 3.0 * ((a > 5) & (b > 3))
        return m

    res = w.find_walls(table(mean), ["a", "b"])
    assert res.n_phases == 3
    expected = ({((5.0, b), (6.0, b)) for b in range(NB)}
                | {((a, 3.0), (a, 4.0)) for a in np.arange(6.0, NA)})
    assert walls_between(res) == expected


def test_bistable_point_is_flagged_mixed():
    """At (2, 2) one run is in each of two very different states; everywhere else both runs
    are in the same state."""
    def run_mean(a, b, r):
        return np.full(DIM, 8.0 if (a, b, r) == (2, 2, 1) else 0.0)

    res = w.find_walls(table(None, run_mean_of=run_mean), ["a", "b"])
    mixed = res.nodes[res.nodes.mixed]
    assert list(zip(mixed.a, mixed.b)) == [(2.0, 2.0)]


def test_phase_count_against_ratio():
    t = table(lambda a, b: np.full(DIM, 3.0 * (a > 5)))
    curve = w.phases_against_ratio(t, ["a", "b"], ratios=(1.5, 2.0, 1e6))
    assert curve.phases.tolist() == [2, 2, 1]
    assert curve.walls.tolist()[-1] == 0


def bands(levels: tuple) -> callable:
    """A mean that is levels[k] (every descriptor) in the k-th third of the grid along a."""
    def mean_of(a, b):
        return np.full(DIM, float(levels[min(int(3 * a / NA), 2)]))
    return mean_of


def test_one_phase_in_two_places_is_joined():
    t = table(bands((0.0, 8.0, 0.0)), noise=0.5, seed=3)
    apart = w.find_walls(t, ["a", "b"], merge=False)
    joined = w.find_walls(t, ["a", "b"], merge=True)
    assert apart.n_phases == 3 and joined.n_phases == 2
    left, right = joined.nodes[joined.nodes.a == 0].phase, joined.nodes[joined.nodes.a == NA - 1].phase
    assert left.nunique() == 1 and set(left) == set(right)
    assert walls_between(joined) == walls_between(apart)            # the walls stay
    assert joined.merges.merged.tolist() == [True]


def test_different_phases_in_two_places_stay_apart():
    t = table(bands((0.0, 8.0, 16.0)), noise=0.5, seed=3)
    r = w.find_walls(t, ["a", "b"], merge=True)
    assert r.n_phases == 3 and r.merges.merged.tolist() == [False]


def test_grid_coordinates_count_steps_by_rank():
    pts = np.array([[0.01, 1.0], [0.1, 1.0], [1.0, 2.0], [0.1, 5.0]])
    assert w.grid_coordinates(pts).tolist() == [[0, 0], [1, 0], [2, 1], [1, 2]]
