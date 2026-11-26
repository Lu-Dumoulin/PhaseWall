"""Tests of graph.py: nodes of a scan, links of a full grid and of a refined line, regions, k-NN links of
scattered points (step scaling, log sampling, holes, the choice of graph)."""

import numpy as np
import pandas as pd

from phasewall.graph import (
    grid_edges,
    is_grid,
    knn_edges,
    links,
    neighbours,
    parameter_nodes,
    regions,
    sampled_in_log,
    scaled,
)


def edge_set(points: np.ndarray, edges: pd.DataFrame) -> set:
    return {(tuple(points[a]), tuple(points[b]), axis) for a, b, axis in edges.itertuples(index=False)}


def test_nodes_merge_rounding_errors():
    node, points = parameter_nodes(np.array([[0.1 + 0.2, 1.0], [0.3, 1.0], [0.3, 2.0]]))
    assert node.tolist() == [0, 0, 1]
    assert points.tolist() == [[0.3, 1.0], [0.3, 2.0]]


def test_full_grid():
    """3 x 2 grid: two lines of 3 points along a (4 links), three lines of 2 along b (3 links)."""
    points = np.array([(a, b) for a in (0, 1, 2) for b in (5, 6)], dtype=float)
    edges = grid_edges(points)
    assert len(edges) == 7
    assert edge_set(points, edges) == {
        ((0, 5), (1, 5), 0), ((1, 5), (2, 5), 0), ((0, 6), (1, 6), 0), ((1, 6), (2, 6), 0),
        ((0, 5), (0, 6), 1), ((1, 5), (1, 6), 1), ((2, 5), (2, 6), 1)}


def test_refined_line_links_to_the_coarse_points_around_it():
    """a in {0, 1, 2} at b = 0 and 1, refined with a = 0.5 and 1.5 at b = 0 only: the refined
    points sit between the coarse ones along a and have no neighbour along b."""
    coarse = [(a, b) for a in (0, 1, 2) for b in (0, 1)]
    points = np.array(sorted(coarse + [(0.5, 0), (1.5, 0)]), dtype=float)
    assert edge_set(points, grid_edges(points)) == {
        ((0, 0), (0.5, 0), 0), ((0.5, 0), (1, 0), 0), ((1, 0), (1.5, 0), 0), ((1.5, 0), (2, 0), 0),
        ((0, 1), (1, 1), 0), ((1, 1), (2, 1), 0),
        ((0, 0), (0, 1), 1), ((1, 0), (1, 1), 1), ((2, 0), (2, 1), 1)}


def test_one_parameter():
    points = np.array([[2.0], [0.0], [1.0]])
    assert edge_set(points, grid_edges(points)) == {((0.0,), (1.0,), 0), ((1.0,), (2.0,), 0)}


def test_regions_are_numbered_from_the_largest():
    """Chain 0-1-2-3-4-5 cut between 1 and 2: region 0 = {2, 3, 4, 5}, region 1 = {0, 1}."""
    edges = pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": [1, 2, 3, 4, 5], "axis": 0})
    keep = np.array([True, False, True, True, True])
    assert regions(6, edges, keep).tolist() == [1, 1, 0, 0, 0, 0]
    # equal sizes: the region holding the lowest node comes first
    keep = np.array([True, True, False, True, True])
    assert regions(6, edges, keep).tolist() == [0, 0, 0, 1, 1, 1]


# ---------------------------------------------------------------- scattered points: k-NN links

def test_a_full_grid_is_a_grid_and_random_points_are_not():
    rng = np.random.default_rng(0)
    grid = np.array([(a, b) for a in range(5) for b in range(4)], dtype=float)
    assert is_grid(grid)
    assert not is_grid(rng.uniform(0, 1, (40, 2)))
    assert links(grid).kind == "grid" and links(rng.uniform(0, 1, (40, 2))).kind == "knn"


def test_scaling_by_the_step_makes_the_four_nearest_the_grid_neighbours():
    """A 21 x 5 grid with steps 0.0025 and 0.1 (very different units): divided by their steps,
    the k = 4 nearest neighbours include every grid link."""
    a, b = np.meshgrid(np.arange(21) * 0.0025, 1.8 + np.arange(5) * 0.1, indexing="ij")
    points = np.c_[a.ravel(), b.ravel()]
    knn = knn_edges(points, k=4)
    assert set(zip(grid_edges(points).a, grid_edges(points).b)) <= set(zip(knn.a, knn.b))
    assert np.allclose(scaled(points)[:, 0], np.round(scaled(points)[:, 0]))


def test_knn_links_are_symmetrised_and_ordered():
    rng = np.random.default_rng(1)
    e = knn_edges(rng.uniform(0, 1, (50, 2)))
    assert (e.a < e.b).all() and not e.duplicated(["a", "b"]).any() and (e.axis == -1).all()
    degree = np.bincount(np.r_[e.a, e.b], minlength=50)
    assert degree.min() >= 1 and 6 <= degree.mean() <= 9


def test_no_knn_link_across_a_hole():
    """Thirty points on the left and four far away on the right: the four must take neighbours
    across the hole, and those links, longer than MAX_LENGTH times the median, are dropped."""
    rng = np.random.default_rng(1)
    points = np.r_[rng.uniform([0, 0], [2.5, 4], (30, 2)), rng.uniform([5.2, 1.2], [6.2, 2.8], (4, 2))] - 3.0
    # (shifted to cross 0: a positive parameter crowded at its low end could be taken as sampled in log)
    e = knn_edges(points)
    assert not ((e.a < 30) != (e.b < 30)).any()


def test_a_parameter_sampled_in_log_is_scaled_in_log():
    """A log grid and points drawn uniformly in log10 are scaled in log10; a linear grid and points
    drawn uniformly in linear scale, even from near 0, are not."""
    rng = np.random.default_rng(0)
    assert sampled_in_log(np.array([0.01, 0.1, 1.0, 10.0]))
    assert np.allclose(scaled(np.c_[[0.01, 0.1, 1.0, 10.0], np.zeros(4)])[:, 0], [0, 1, 2, 3])
    assert sampled_in_log(10 ** rng.uniform(-2, 0, 100))
    assert not sampled_in_log(np.arange(1.0, 11.0))
    assert not sampled_in_log(rng.uniform(0, 10, 100))
    assert not sampled_in_log(np.arange(0.0, 1.01, 0.1))           # starts at 0


def test_more_parameters_get_more_neighbours():
    """k = max(6, 2 d): 6 up to 3 parameters, then 2 per parameter; the links follow it."""
    assert [neighbours(d) for d in (1, 2, 3, 4, 5, 8, 16)] == [6, 6, 6, 8, 10, 16, 32]
    rng = np.random.default_rng(2)
    few, many = knn_edges(rng.uniform(0, 1, (300, 2))), knn_edges(rng.uniform(0, 1, (300, 8)))
    assert 2 * len(few) / 300 < 9 < 2 * len(many) / 300
