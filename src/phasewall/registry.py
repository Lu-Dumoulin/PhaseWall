"""The registries of encoders and methods: how phasewall is extended.

An encoder turns the snapshots of an ensemble into tables of descriptions; a method turns one
table into groups of runs (the phases). Each is registered once, with a settings dataclass, and
then runs in the pipeline, the scripts and the interface without any other change:

    from dataclasses import dataclass, field
    from phasewall import registry

    @dataclass
    class MySettings:
        k: int = field(default=3, metadata={"help": "number of groups", "range": (2, 10)})

    @registry.method("mine", MySettings, help="my grouping of the runs", uses_grid=False)
    def mine(table, parameters, settings, context):
        ...                                  # one group per run (Series indexed by run, -1: noise)
        return registry.MethodResult(groups)

    pl.Settings(..., methods=pl.Methods(others={"mine": MySettings(k=4)}))

Another package registers its entries through Python entry points, groups
"phasewall.encoders" and "phasewall.methods" (an entry point names a module or a function;
loading it runs the decorators), read by load_plugins() when the registry is first used.

Contract. An encoder is run(ensemble, settings, context) -> EncoderResult: tables with one row
per snapshot (columns run, time, the parameters, then descriptors named name.block.column), all
finite. A method is run(table, parameters, settings, context) -> MethodResult: one integer group
per run of the table (-1: noise). tests/test_contracts.py checks every registered entry against
these rules and against synthetic answers (one phase in noise, two phases found, invariance of
the descriptors under the symmetries of the box), so a new entry is tested by adding it.

Settings dataclasses describe themselves for the interface: settings_spec(cls) lists every field
with its type, default and the optional metadata keys "help", "choices" and "range".
"""

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import entry_points

import pandas as pd


@dataclass
class Context:
    """What an encoder or a method may need besides its settings."""

    workers: int = 1
    cache: str | None = None            # folder for cached results (None: no cache)
    key: str = ""                       # identifies the ensemble, for cache names
    frame: object = None                # the frame settings (pipeline.FrameSettings), for cache names
    labels: pd.Series | None = None     # never given to methods; only to checks
    progress: object = None             # progress(step, done, total) or None
    device: str | None = None


@dataclass
class EncoderResult:
    tables: dict[str, pd.DataFrame]                     # name of the description -> table
    log: pd.DataFrame | None = None                     # training logs, if any (column name)


@dataclass
class MethodResult:
    groups: pd.Series                                   # group of every run (index: run; -1: noise)
    extra: object = None                                # what the method wants to keep (walls, rules, ...)


@dataclass(frozen=True)
class Entry:
    name: str
    settings: type                                      # a dataclass; its defaults are the method's defaults
    run: object                                         # the function
    help: str = ""
    needs: tuple[str, ...] = ()                         # optional packages (import names), for messages
    uses_grid: bool = False                             # methods: works on the parameter grid (contract tests)
    invariant: bool = False                             # encoders: descriptions invariant under the box symmetries
    test_settings: object = None                        # encoders: cheap settings for the contract tests


ENCODERS: dict[str, Entry] = {}
METHODS: dict[str, Entry] = {}
_LOADED = False


def _register(table: dict, name: str, settings: type, **kw):
    if not dataclasses.is_dataclass(settings):
        raise TypeError(f"{name}: settings must be a dataclass")

    def wrap(fn):
        if name in table and table[name].run is not fn:
            raise ValueError(f"{name!r} is already registered")
        table[name] = Entry(name, settings, fn, **kw)
        return fn
    return wrap


def encoder(name: str, settings: type, help: str = "", needs=(), invariant: bool = False, test_settings=None):
    """Decorator: register an encoder (see the module docstring)."""
    return _register(ENCODERS, name, settings, help=help, needs=tuple(needs), invariant=invariant,
                     test_settings=test_settings)


def method(name: str, settings: type, help: str = "", needs=(), uses_grid: bool = False):
    """Decorator: register a method (see the module docstring)."""
    return _register(METHODS, name, settings, help=help, needs=tuple(needs), uses_grid=uses_grid)


def load() -> None:
    """The built-in entries, then those of other packages (entry points), once."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from . import builtins  # noqa: F401  (registers the built-in encoders and methods)
    for group in ("phasewall.encoders", "phasewall.methods"):
        for ep in entry_points(group=group):
            ep.load()


def encoders() -> dict[str, Entry]:
    load()
    return ENCODERS


def methods() -> dict[str, Entry]:
    load()
    return METHODS


def available(entry: Entry) -> bool:
    """Whether the optional packages of an entry can be imported."""
    from importlib.util import find_spec
    return all(find_spec(n) is not None for n in entry.needs)


def cache_key(*parts) -> str:
    """A short hash of settings (dataclasses, dicts, values), for names of cached results."""
    def plain(x):
        if not dataclasses.is_dataclass(x):
            return x
        d = asdict(x)
        for k in getattr(x, "OPTIONAL_KEYS", ()):       # settings added later: left out while unset, so
            if d.get(k) is None:                        # that results cached before them are still found
                d.pop(k, None)
        return d
    text = json.dumps([plain(p) for p in parts], sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def settings_spec(cls) -> list[dict]:
    """The fields of a settings dataclass for an interface: name, type, default, help,
    choices, range (None when not given)."""
    out = []
    for f in dataclasses.fields(cls):
        default = f.default if f.default is not dataclasses.MISSING else (
            f.default_factory() if f.default_factory is not dataclasses.MISSING else None)
        out.append({"name": f.name, "type": f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type)),
                    "default": default, "help": f.metadata.get("help", ""),
                    "choices": f.metadata.get("choices"), "range": f.metadata.get("range")})
    return out


def check_settings(settings) -> None:
    """ValueError for a value outside the choices or the range given in the metadata."""
    for f in dataclasses.fields(settings):
        v = getattr(settings, f.name)
        choices, rng = f.metadata.get("choices"), f.metadata.get("range")
        if choices is not None and v not in choices:
            raise ValueError(f"{type(settings).__name__}.{f.name} = {v!r}: choose one of {list(choices)}")
        if rng is not None and v is not None:
            values = v if isinstance(v, tuple | list) else [v]
            if any(x < rng[0] or x > rng[1] for x in values):
                raise ValueError(f"{type(settings).__name__}.{f.name} = {v!r}: outside {rng[0]} to {rng[1]}")

