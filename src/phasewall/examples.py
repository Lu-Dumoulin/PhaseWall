"""The demo ensembles, by name: each is a module with make_ensemble(folder, seeds=..., workers=...,
progress=..., **grid and settings), simulate, reference_label, GRID (default parameter grid,
{name: values}), SETTINGS (default numerical settings), N_FIELDS, SEEDS (default runs per point), QUICK
(a small grid and box for a fast try) and seconds_per_run(**settings) (cost estimate).

- "swift-hohenberg" (toy.py): uniform state, hexagons, stripes; one scalar field.
- "gray-scott" (gray_scott.py): two uniform states, spots, isolated spots, holes, stripes and
  labyrinths, moving states; two scalar fields.
"""

from types import ModuleType

from . import gray_scott, toy

EXAMPLES: dict[str, ModuleType] = {"swift-hohenberg": toy, "gray-scott": gray_scott}


def example(name: str) -> ModuleType:
    """The module of a demo; ValueError for an unknown name."""
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; known: {', '.join(EXAMPLES)}")
    return EXAMPLES[name]
