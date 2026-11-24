"""PhaseWall for one-dimensional systems: kymographs instead of snapshots.

A run of a 1D system is described by one kymograph per field, a(x, t). PhaseWall cuts the last
part of every run into K time windows; each window is described (hand-made observables in space
and time) and gives one row of the same table as in 2D: run, time, the parameters and
descriptor columns `field.block.name`. From that table on, the walls, the tree and the blind
grouping of the 2D package are used unchanged.

This subpackage only imports from the 2D package (a few helpers of describe.py, the parallel
runner); the 2D package never imports it, so the 2D code works whatever the state of the 1D one.

- kymo.py: fields of a 1D system, the ensemble and its TOML file, reading kymographs, windows.
- describe1d.py: the hand-made observables of a window.
- cgle.py: the example with a known answer, the 1D complex Ginzburg-Landau equation.
"""
