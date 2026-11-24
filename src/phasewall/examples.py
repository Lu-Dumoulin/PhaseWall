"""The demo ensembles, by name: each is a module with make_ensemble(folder, seeds=..., workers=...,
progress=..., **grid and settings), simulate, reference_label, GRID (default parameter grid,
{name: values}), SETTINGS (default numerical settings), N_FIELDS, SEEDS (default runs per point), QUICK
(a small grid and box for a fast try) and seconds_per_run(**settings) (cost estimate).
fields_of(name) gives the fields of its ensembles before it is simulated (u, and v when it has two).

- "swift-hohenberg" (toy.py): uniform state, hexagons, stripes; one scalar field.
- "gray-scott" (gray_scott.py): two uniform states, spots, isolated spots, holes, stripes and
  labyrinths, moving states; two scalar fields.
- "fitzhugh-nagumo" (fitzhugh_nagumo.py): fronts, moving domains, uniform states, Turing
  labyrinths and spots; two scalar fields.
- "sh-localized" (sh_localized.py): cubic-quintic Swift-Hohenberg from a localised stripe patch:
  localised states (dissipative solitons), extended stripes, uniform state.
- "barkley" (barkley.py): excitable medium: rest, plane waves, spirals; two scalar fields.
- "ginzburg-landau-1d" (oned/cgle.py): a 1D example (kymographs, DIMS = 1): plane waves, phase
  turbulence, defect turbulence, spatiotemporal intermittency; one complex field.

A module without DIMS is 2D; a 1D module also gives FIELDS (its oned.kymo.Field1D).
"""

from types import ModuleType

from . import barkley, fitzhugh_nagumo, gray_scott, sh_localized, toy
from .fields import Field
from .oned import cgle

EXAMPLES: dict[str, ModuleType] = {"swift-hohenberg": toy, "gray-scott": gray_scott,
                                   "fitzhugh-nagumo": fitzhugh_nagumo, "sh-localized": sh_localized,
                                   "barkley": barkley, "ginzburg-landau-1d": cgle}


def example(name: str) -> ModuleType:
    """The module of a demo; ValueError for an unknown name."""
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; known: {', '.join(EXAMPLES)}")
    return EXAMPLES[name]


def dims_of(name: str) -> int:
    """1 for a 1D example (kymographs), 2 otherwise."""
    return getattr(example(name), "DIMS", 2)


def fields_of(name: str) -> list:
    """The fields of a demo's ensembles (2D: scalars u, and v for a two-field model; 1D: the
    module's FIELDS), as its make_ensemble writes them into phasewall.toml."""
    module = example(name)
    if dims_of(name) == 1:
        return list(module.FIELDS)
    return [Field(n, (n,)) for n in ("u", "v")[: module.N_FIELDS]]
