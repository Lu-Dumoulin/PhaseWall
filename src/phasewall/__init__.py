"""phasewall: phases of simulated hydrodynamic systems, found from the walls between them in
parameter space.

An ensemble of runs, each at one point of a parameter grid, is described snapshot by snapshot
(generic descriptors or a self-supervised network). Neighbouring parameter points are compared
with a two-sample test; the links where the description jumps more than its own spread are cut,
and the connected regions that remain are the candidate phases of the scanned range.
"""

__version__ = "0.1.0"
