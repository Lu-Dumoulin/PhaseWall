"""Tests of graph.py: nodes of a scan, links of a full grid and of a refined line, regions."""

import numpy as np
import pandas as pd

from phasewall.graph import grid_edges, parameter_nodes, regions


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
