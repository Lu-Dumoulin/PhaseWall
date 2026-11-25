"""PhaseWall explorer: a marimo notebook that is also a dashboard.

    marimo edit apps/explore.py    notebook: the code of every cell is visible and editable
    marimo run apps/explore.py     dashboard: only the controls and the results
    python apps/explore.py --run   no browser: runs the default settings once (a smoke test);
                                   add --quick for the quick demo, --cache DIR for another cache,
                                   --tree to add the threshold tree, --runs FILE for your own runs

The notebook only calls the package (phasewall.pipeline, phasewall.registry, phasewall.report);
it computes nothing itself. The controls of the tree, of the network and of any encoder or method
registered by another package are built from their settings (registry.settings_spec), so a new
entry appears here without changing this file. How marimo works, for a Pluto user: each cell is a
function whose arguments are the names it reads from other cells and whose return value lists the
names it defines. When a control changes, every cell that reads it runs again, as in Pluto. The
expensive steps run only when the Run button is pressed (mo.stop holds the cell otherwise), and the
pipeline caches them on disk, so a second run with another method or threshold takes seconds.
"""
# ruff: noqa: PLR1711   (marimo writes a bare `return` at the end of every cell)

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="PhaseWall explorer", css_file="explore.css")

with app.setup:
    import dataclasses
    import io
    import os
    import time
    from html import escape
    from importlib.util import find_spec
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    from phasewall import pipeline as pl
    from phasewall import registry, report
    from phasewall.builtins import JepaSettings, TreeSettings
    from phasewall.describe import explain
    from phasewall.examples import EXAMPLES, dims_of, example, fields_of
    from phasewall.fields import BLOCK_HELP, BLOCKS, KINDS, Field
    from phasewall.io import inspect_snapshot, list_snapshots, load_any, suggest_fields, write_config
    from phasewall.nn.architecture import Network
    from phasewall.nn.architecture import check as check_network
    from phasewall.nn.architecture import summary as network_summary
    from phasewall.oned.describe1d import ALLOWED as ALLOWED_1D
    from phasewall.oned.describe1d import BLOCK_HELP as BLOCK_HELP_1D
    from phasewall.oned.describe1d import BLOCKS as BLOCKS_1D
    from phasewall.oned.kymo import KINDS as KINDS_1D
    from phasewall.oned.kymo import SHARE as KYMO_SHARE
    from phasewall.oned.kymo import WINDOWS as KYMO_WINDOWS
    from phasewall.oned.kymo import Field1D, suggest_fields1d, suggest_order, write_config1d
    from phasewall.trees import rules_text

    HAS_TORCH = find_spec("torch") is not None          # learned description (extra [nn])
    HAS_SKLEARN = find_spec("sklearn") is not None      # blind grouping and probe (extra [baselines])
    HDBSCAN_SIZES = (5, 10, 20, 40)                     # offered sizes, as phasewall.baselines.HDBSCAN_SIZES
    BUILTIN_ENCODERS = ("hand", "learned", "jepa")       # have their own controls below
    BUILTIN_METHODS = ("walls", "tree", "mixture", "kmeans", "hdbscan")
    OP_SHOWN = 3                                         # order parameters shown per comparison (a choice;
                                                         # order_parameters.csv keeps orderparams.TOP)
    APPS = mo.notebook_dir() or Path.cwd()               # the folder of this notebook
    DEFAULT_CACHE = APPS.parent / "data" / "cache_app"   # its own cache (data/ is git-ignored), apart from
                                                         # the command line and scripts, which may run at the same time
    GEOMETRIC_TOLERANCE = 1e-3   # geometric range: last value set to stop within this relative distance (a choice)
    GEOMETRIC_SPREAD = 2e-2      # shown as a geometric range when ratios agree within this (a choice)
    SETTING_LABELS = {                                   # labels of the numerical settings of the demos
        "n": "grid points per side N",
        "waves": "wavelengths across the box",
        "dx": "grid spacing dx",
        "dt": "time step",
        "t_relax": "time before the first snapshot",
        "every": "time between snapshots",
        "n_save": "snapshots per run",
        "length": "length of the line",
        "rows": "rows saved per run (kymograph)",
    }


@app.function
def parse_values(text: str) -> list[float]:
    """Parameter values typed as "a, b, c", as a range "start:step:stop" (stop included, as in
    Julia and CommonUI's parse_values) or as a geometric range "start:*factor:stop" (start,
    start factor, start factor^2, ...; the last value is set to stop when it is within
    GEOMETRIC_TOLERANCE of it, so that a factor written with a few digits still ends on stop).
    ValueError with a readable message otherwise."""
    text = text.strip()
    if not text:
        raise ValueError("no values given")
    try:
        if ":*" in text:
            start, factor, stop = (float(p.lstrip("*")) for p in text.split(":"))
            if start <= 0 or factor <= 1 or stop < start:
                raise ValueError
            n = int(np.floor(np.log(stop / start) / np.log(factor) + GEOMETRIC_TOLERANCE)) + 1
            values = [float(f"{start * factor**i:.6g}") for i in range(n)]
            if abs(values[-1] / stop - 1) < GEOMETRIC_TOLERANCE:
                values[-1] = stop
            return values
        if ":" in text:
            start, step, stop = (float(p) for p in text.split(":"))
            if step <= 0 or stop < start:
                raise ValueError
            n = int(np.floor((stop - start) / step + 1e-9)) + 1
            return [round(start + i * step, 10) for i in range(n)]
        return [float(p) for p in text.split(",") if p.strip()]
    except ValueError:
        raise ValueError(f"{text!r}: write values as 'a, b, c', 'start:step:stop' (step > 0) "
                         "or 'start:*factor:stop' (start > 0, factor > 1)") from None


@app.function
def format_values(values) -> str:
    """The inverse of parse_values: a range "start:step:stop" when the values are evenly spaced,
    "start:*factor:stop" when their ratios are equal within GEOMETRIC_SPREAD (values rounded in
    the demo, as np.round(np.geomspace(...), 4), are not exactly geometric), else a list."""
    v = [float(x) for x in values]
    if len(v) >= 3:
        steps = np.diff(v)
        if steps[0] > 0 and np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
            return f"{v[0]:.10g}:{round(steps[0], 10):.10g}:{v[-1]:.10g}"
        if v[0] > 0 and steps[0] > 0:
            ratios = np.array(v[1:]) / np.array(v[:-1])
            factor = (v[-1] / v[0]) ** (1 / (len(v) - 1))
            if np.allclose(ratios, factor, rtol=GEOMETRIC_SPREAD):
                return f"{v[0]:.10g}:*{factor:.5g}:{v[-1]:.10g}"
    return ", ".join(f"{x:.10g}" for x in v)


@app.function
def grid_values(text: str, default) -> list[float]:
    """The values of one parameter of a demo: its default values exactly when the field still
    shows them (its text is unchanged), else the values typed."""
    if text.strip() == format_values(default):
        return [float(x) for x in default]
    return parse_values(text)


@app.function
def parse_ints(text: str) -> tuple[int, ...]:
    """Whole numbers typed as "1, 2, 4"; empty text gives ()."""
    try:
        return tuple(int(p) for p in text.split(",") if p.strip())
    except ValueError:
        raise ValueError(f"{text!r}: write whole numbers separated by commas") from None


@app.function
def optional_float(text: str, what: str) -> float | None:
    """A number, or None for empty text (the setting of the ensemble is then kept)."""
    if not text.strip():
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{what}: {text!r} is not a number") from None


@app.function
def png(fig) -> bytes:
    """A matplotlib Figure as PNG bytes, for display and download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=report.DPI, facecolor=fig.get_facecolor())
    return buf.getvalue()


@app.function
def setting_input(name: str, value, choices=None):
    """The control of one numerical setting of a demo, after the type of its default: a
    checkbox for a yes/no setting, a dropdown when the demo lists the choices (in a dict
    CHOICES = {setting: [values]}), a text field for other words, else a number."""
    label = SETTING_LABELS.get(name, name)
    if isinstance(value, (bool, np.bool_)):
        return mo.ui.checkbox(value=bool(value), label=label)
    if isinstance(value, str):
        if choices:
            return mo.ui.dropdown([str(c) for c in choices], value=value, label=label)
        return mo.ui.text(value=value, label=label)
    return mo.ui.number(value=value, label=label)


@app.function
def spec_controls(cls, skip=()):
    """One control per field of a settings dataclass, from its description
    (registry.settings_spec): a checkbox for yes/no, a dropdown when the field lists its choices,
    a number (within its range) for a number, a text for a tuple ("a, b") or words. Fields that
    hold a dict or another dataclass are left out (they have their own controls)."""
    items = {}
    for f in registry.settings_spec(cls):
        name, default = f["name"], f["default"]
        if name in skip or isinstance(default, dict) or dataclasses.is_dataclass(default):
            continue
        label = name.replace("_", " ") + (f": {f['help']}" if f["help"] else "")
        if isinstance(default, bool):
            items[name] = mo.ui.checkbox(value=default, label=label)
        elif f["choices"] is not None:
            items[name] = mo.ui.dropdown([str(c) for c in f["choices"]], value=str(default), label=label)
        elif isinstance(default, int | float):
            lo, hi = f["range"] or (None, None)
            items[name] = mo.ui.number(start=lo, stop=hi, step=1 if isinstance(default, int) else None,
                                       value=default, label=label)
        elif isinstance(default, tuple):
            items[name] = mo.ui.text(value=", ".join(str(x) for x in default), label=label)
        else:
            items[name] = mo.ui.text(value=str(default), label=label)
    return mo.ui.dictionary(items)


@app.function
def settings_from(cls, values: dict):
    """The settings dataclass from the values of spec_controls, each converted to the type of
    its default (a dropdown gives text)."""
    kw = {}
    defaults = cls()
    for f in dataclasses.fields(cls):
        if f.name not in values:
            continue
        default, v = getattr(defaults, f.name), values[f.name]
        if isinstance(default, bool):
            kw[f.name] = bool(v)
        elif isinstance(default, int):
            kw[f.name] = int(float(v))
        elif isinstance(default, float):
            kw[f.name] = float(v)
        elif isinstance(default, tuple):
            kw[f.name] = parse_ints(v) if all(isinstance(x, int) for x in default) else tuple(parse_values(v))
        else:
            kw[f.name] = v
    return cls(**kw)


@app.function
def parse_field_lines(text: str, dims: int = 2) -> list:
    """Fields typed one per line as "name kind dataset[,dataset] [relative] [single]": one dataset
    with all components, or two datasets that are the components of a vector or tensor;
    "relative" (scalars) describes (f - <f>) / <f>; "single" confirms that a name such as vx is a
    scalar of its own. ValueError with the line at fault."""
    fields = []
    for line in text.splitlines():
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        if len(words) < 3:
            raise ValueError(f"{line!r}: write 'name kind dataset' (kinds: {', '.join(KINDS if dims == 2 else KINDS_1D)})")
        name, kind, data, flags = words[0], words[1], words[2].split(","), set(words[3:])
        unknown = flags - {"relative", "single"}
        if unknown:
            raise ValueError(f"{line!r}: unknown words {sorted(unknown)} (relative, single)")
        if dims == 1:                                   # 1D: oned.kymo.Field1D (no single_component)
            fields.append(Field1D(name, () if len(data) == 2 else (data[0],), kind, "relative" in flags,
                                  tuple(data) if len(data) == 2 else ()))
        elif len(data) == 2:
            fields.append(Field(name, (name,), kind, components=tuple(data)))
        else:
            fields.append(Field(name, (data[0],), kind, "relative" in flags, single_component="single" in flags))
    return fields


@app.function
def kymo_preview(inspection: dict, order: str, julia_order: bool):
    """The first 2D dataset of a 1D run file read with the chosen axes, x across and t upward, so
    that the order can be checked by eye; a note if it cannot be drawn."""
    import h5py
    import matplotlib.pyplot as plt

    from phasewall.oned.kymo import _read
    name = next((r.name for r in inspection["datasets"].itertuples() if len(r.shape) == 2 and min(r.shape) > 1), None)
    if name is None:
        return mo.md("No 2D dataset to draw.")
    with h5py.File(inspection["file"], "r") as _f:
        a = _read(_f, name, julia_order, order)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.imshow(a.T, origin="lower", aspect="auto", cmap="RdBu_r")
    ax.set_xlabel(f"x ({a.shape[0]} points)"); ax.set_ylabel(f"t ({a.shape[1]} rows)")
    ax.set_title(f"{name}, first run", fontsize=9)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return mo.image(buf.getvalue())


@app.function
def card(title: str, body):
    """A section of the page: a title and its content in a light frame."""
    return mo.vstack([mo.md(f"### {title}"), body], gap=0.6).style({
        "border": "1px solid var(--pw-border, #dcdad5)", "border-left": "4px solid var(--pw-accent, #2a78d6)",
        "background": "var(--pw-card-bg, transparent)", "border-radius": "12px", "padding": "1.0rem 1.4rem",
        "margin": "0.5rem 0"})


@app.function
def columns(*items):
    """Content side by side, in columns of equal width, aligned at the top."""
    return mo.hstack(list(items), widths="equal", gap=2, align="start")


@app.function
def block_grid(block_inputs, fields):
    """The checkboxes of the blocks as a grid: one row per field, one column per block (a box only
    where the block fits the kind of the field), the definition of each block on hover and below."""
    one_d = any(not hasattr(f, "symmetry") for f in fields)          # fields of a 1D system
    order, help_ = (BLOCKS_1D, BLOCK_HELP_1D) if one_d else (list(BLOCKS), BLOCK_HELP)
    used = [b for b in order if any(b in block_inputs[f.name] for f in fields)]
    widths = [2] + [1] * len(used)

    def head(b):
        return mo.Html(f'<span title="{escape(help_[b])}" style="cursor: help; border-bottom: 1px dotted; '
                       f'font-family: var(--monospace-font, monospace); font-size: 0.85em">{b}</span>')
    rows = [mo.hstack([mo.md(" "), *[head(b) for b in used]], widths=widths, justify="start")]
    for f in fields:
        boxes = block_inputs[f.name]
        rows.append(mo.hstack([mo.md(f"**{f.name}** <span style='opacity: 0.6'>{f.kind}</span>"),
                               *[boxes[b] if b in boxes else mo.md("") for b in used]],
                              widths=widths, justify="start"))
    legend = mo.md("  \n".join(f"`{b}`: {escape(help_[b])}" for b in used)).style(
        {"font-size": "0.82em", "opacity": "0.75", "margin-top": "0.4rem"})
    return mo.vstack([*rows, legend], gap=0.3)


@app.function
def description_text(name: str, table) -> str:
    """One line saying what a description is (shown under its tab in the results)."""
    from phasewall.describe import descriptor_columns

    n = len(descriptor_columns(table))
    if name.startswith("hand"):
        return f"Hand-made observables of every field: **{n} descriptors** per sample (snapshot, or window in 1D)."
    if name.startswith("learned"):
        return f"h of the learned encoder (CNN, contrastive): **{n} dimensions** per sample."
    if name.startswith("jepa"):
        return f"h of the predictive encoder (CNN, JEPA): **{n} dimensions** per sample."
    if name.startswith("combined"):
        return f"Hand-made observables and learned h side by side: **{n} columns** per sample."
    return f"**{n} columns** per sample."


@app.function
def seconds_text(s: float) -> str:
    if s < 1:
        return "under 1 s"
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.1f} min" if s < 5400 else f"{s / 3600:.1f} h"


@app.class_definition
class Bars:
    """The progress(step, done, total) callback of the pipeline, shown as one marimo progress
    bar per step; it also records how long each step took."""

    def __init__(self):
        self.bar, self.key, self.done = None, None, 0
        self.seconds: dict[str, float] = {}
        self.started = time.time()

    def __call__(self, step: str, done: int, total: int) -> None:
        if (step, total) != self.key or done < self.done:      # a new step (or a new stage of one)
            self.close()
            self.bar = mo.status.progress_bar(total=total, title=step, remove_on_exit=True)
            self.key, self.done = (step, total), 0
        self.bar.progress.update(increment=done - self.done)   # the time since the last report belongs to this step
        self.done = done
        self.seconds[step] = self.seconds.get(step, 0.0) + time.time() - self.started
        self.started = time.time()

    def close(self) -> None:
        if self.bar is not None:
            self.bar.__exit__(None, None, None)
            self.bar = None


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # PhaseWall explorer

    PhaseWall finds the phases of a simulated system from runs on a grid of parameters: it
    describes every snapshot (2D) or every time window of a kymograph (1D) with encoders, then finds
    the phases from the descriptions with methods, without labels. Choose each step below, then press **Run**; steps already computed with the
    same settings are read from the cache.

    <div class="pw-steps"><span class="pw-pill">1 System</span><span class="pw-arrow">→</span>
    <span class="pw-pill">2 Fields</span><span class="pw-arrow">→</span>
    <span class="pw-pill">3 Encoders</span><span class="pw-arrow">→</span>
    <span class="pw-pill">4 Methods</span><span class="pw-arrow">→</span>
    <span class="pw-pill">Run</span><span class="pw-arrow">→</span><span class="pw-pill">Results</span></div>
    """)
    return


@app.cell
def _():
    # The dimension of the system, first: it decides the sources, fields, observables and figures.
    dimension = mo.ui.radio(
        {"2D: snapshots of fields on a plane": 2, "1D: kymographs of fields on a line": 1},
        value="1D: kymographs of fields on a line" if "1d" in mo.cli_args() else "2D: snapshots of fields on a plane",
        label="Dimension of the system", inline=True)
    card("Dimension", dimension)
    return (dimension,)


@app.cell
def _(dimension):
    # Source of the runs. mo.ui.radio is PlutoUI's Radio; its .value is read in other cells.
    # 1D runs come from an example, an ensemble ([kymograph] TOML file) or a table of descriptors.
    _own = mo.cli_args().get("runs")
    _runs_choice = ("your runs (CSV of runs and their snapshot files)" if dimension.value == 2
                    else "your runs (CSV of runs, one HDF5 or JLD2 file each)")
    _choices = {"an example": "example", _runs_choice: "runs",
                "an ensemble (TOML file)": "config", "a table of descriptors (CSV)": "table"}
    source_kind = mo.ui.radio(_choices, value=_runs_choice if _own else "an example", label="Runs from")
    _examples = [n for n in EXAMPLES if dims_of(n) == dimension.value]
    example_choice = mo.ui.dropdown(_examples, value="swift-hohenberg" if dimension.value == 2 else _examples[0],
                                    label="example")
    quick = mo.ui.checkbox(value="quick" in mo.cli_args(),
                           label="quick: small grid and box, seconds (too coarse for all states)")
    runs_path = mo.ui.text(value=str(_own or ""), placeholder="path/to/runs.csv", full_width=True,
                           label="table of runs: a column `path` (run folder) and one per parameter" if dimension.value == 2
                           else "table of runs: a column `path` (the HDF5 or JLD2 file of the run) and one per parameter")
    dt_input = mo.ui.text(value=str(mo.cli_args().get("dt") or ""), placeholder="e.g. 0.5",
                          label="time between two rows of the kymograph (dt)")
    snapshots_input = mo.ui.text(value=str(mo.cli_args().get("snapshots") or "Data/*.h5"), label="snapshot files in a run folder")
    time_regex_input = mo.ui.text(value=r"(\d+)\.h5$", label="time in the file name (regular expression)")
    time_unit_input = mo.ui.number(value=1.0, label="time unit (number in the name x unit)")
    julia_input = mo.ui.checkbox(value=True, label="arrays written by Julia (column-major)")
    dx_input = mo.ui.text(value=str(mo.cli_args().get("dx") or ""), placeholder="or a column dx", label="grid spacing dx")
    periodic_own = mo.ui.checkbox(value=True, label="periodic boundaries")
    last_input = mo.ui.number(start=1, stop=10_000, step=1, value=10, label="last snapshots described per run")
    pair_dt_input = mo.ui.text(value=str(mo.cli_args().get("pair-dt") or ""), placeholder="none",
                               label="time between the two snapshots of a pair (dynamics, learned encoder)")
    runs_label = mo.ui.text(placeholder="optional", label="label column (only for the scores)")
    config_path = mo.ui.text(placeholder="path/to/ensemble.toml", label="TOML file", full_width=True)
    config_label = mo.ui.text(placeholder="optional", label="label column of runs.csv (only for the scores)")
    table_path = mo.ui.text(placeholder="path/to/descriptors.csv", label="CSV file", full_width=True)
    table_parameters = mo.ui.text(placeholder="rho0, zeta", label="parameter columns", full_width=True)
    table_label = mo.ui.text(placeholder="optional", label="label column (only for the scores)")
    table_prefix = mo.ui.text(placeholder="optional", label="descriptor columns start with")
    return (
        config_label,
        config_path,
        dt_input,
        dx_input,
        example_choice,
        julia_input,
        last_input,
        pair_dt_input,
        periodic_own,
        quick,
        runs_label,
        runs_path,
        snapshots_input,
        source_kind,
        table_label,
        table_parameters,
        table_path,
        table_prefix,
        time_regex_input,
        time_unit_input,
    )


@app.cell
def _(example_choice, quick):
    # Grid and numerical settings of the chosen example, pre-filled with its defaults (or its quick
    # version). This cell runs again when the example or `quick` changes, so the fields are reset.
    demo = example(example_choice.value)
    _quick = dict(demo.QUICK) if quick.value else {}
    grid_defaults = {p: _quick.get(p, v) for p, v in demo.GRID.items()}
    grid_inputs = mo.ui.dictionary(
        {p: mo.ui.text(value=format_values(v), full_width=True) for p, v in grid_defaults.items()}
    )
    demo_defaults = demo.SETTINGS | {k: v for k, v in _quick.items() if k not in demo.GRID}
    setting_inputs = mo.ui.dictionary(
        {k: setting_input(k, v, getattr(demo, "CHOICES", {}).get(k)) for k, v in demo_defaults.items()}
    )
    seeds_input = mo.ui.number(
        start=1, stop=64, step=1, value=pl.QUICK_SEEDS if quick.value else len(demo.SEEDS),
        label="runs per point (different random initial conditions)",
    )
    return demo, demo_defaults, grid_defaults, grid_inputs, seeds_input, setting_inputs


@app.cell
def _(dimension, julia_input, runs_path, snapshots_input, source_kind, time_regex_input, time_unit_input):
    # Your runs: the first snapshot of the first run (2D), or the file of the first run (1D: one HDF5
    # or JLD2 file per run, its path in the table), is inspected; its datasets are listed and fields
    # are suggested (phasewall.io.suggest_fields, phasewall.oned.kymo.suggest_fields1d). Nothing is
    # used until you confirm the kind of each field below; a component alone (vx) is refused as a 2D
    # scalar. In 1D you also say which axis of the arrays is x and which is t (a guess is preselected).
    inspection, inspection_problem = None, None
    _one_d = dimension.value == 1
    if source_kind.value == "runs" and runs_path.value.strip():
        try:
            _runs = pd.read_csv(runs_path.value)
            if "path" not in _runs.columns:
                raise ValueError("the table of runs needs a column `path`")
            _first = Path(runs_path.value).resolve().parent / _runs["path"].iloc[0]
            if _one_d:
                if not _first.is_file():
                    raise ValueError(f"{_first} is not a file: in 1D the column `path` gives the HDF5 or JLD2 file "
                                     "of each run")
                _file = _first
            else:
                _snaps = list_snapshots(_first, snapshots_input.value, time_regex_input.value,
                                        float(time_unit_input.value))
                if not _snaps:
                    raise ValueError(f"no file {snapshots_input.value!r} with a time matching "
                                     f"{time_regex_input.value!r} in {_first}")
                _file = _snaps[-1][1]
            _datasets = inspect_snapshot(_file, julia_input.value)
            inspection = {"runs": _runs, "file": _file, "datasets": _datasets,
                          "suggestions": suggest_fields1d(_datasets) if _one_d else suggest_fields(_datasets)}
        except (OSError, ValueError, KeyError) as _e:
            inspection_problem = str(_e)
    _usable = [s for s in (inspection["suggestions"] if inspection else []) if s["kind"] is not None]
    _kinds = list(KINDS_1D if _one_d else KINDS)
    field_rows = mo.ui.array([
        mo.ui.dictionary({
            "use": mo.ui.checkbox(value=True, label=f"**{s['name']}** from {', '.join(s.get('components') or s['keys'])}"),
            "kind": mo.ui.dropdown(_kinds, value=s["kind"], label="kind"),
            "relative": mo.ui.checkbox(value=bool(s.get("relative")), label="relative (scalars)"),
        }) for s in _usable
    ])
    extra_fields = mo.ui.text_area(placeholder="name kind dataset[,dataset] [relative]" + ("" if _one_d else " [single]"),
                                   rows=2, label="other fields, one per line", full_width=True)
    _numeric = [c for c in (inspection["runs"].columns if inspection else [])
                if c not in ("run", "path", "dx") and pd.api.types.is_numeric_dtype(inspection["runs"][c])]
    parameter_choice = mo.ui.multiselect(_numeric, value=[c for c in _numeric if c != "seed"],
                                         label="parameter columns")
    _shape = next((tuple(r.shape) for r in inspection["datasets"].itertuples() if len(r.shape) == 2
                   and min(r.shape) > 1), None) if inspection else None
    _order, _why = suggest_order(_shape) if _shape else ("xt", "")
    order_input = mo.ui.radio({"[x, t]: the first axis is x": "xt", "[t, x]: the first axis is t": "tx"},
                              value="[x, t]: the first axis is x" if _order == "xt" else "[t, x]: the first axis is t",
                              label="axes of the arrays" + (f" (guess: {_why})" if _why else ""))
    usable_suggestions = _usable
    return (extra_fields, field_rows, inspection, inspection_problem, order_input, parameter_choice,
            usable_suggestions)


@app.cell
def _(config_path, dimension, example_choice, extra_fields, field_rows, source_kind, usable_suggestions):
    # The fields of the chosen system, known before any run: an example's, those of a TOML file,
    # or those you confirmed for your runs. Their kinds decide which blocks of observables are offered.
    known_fields, field_problems = [], []
    try:
        if source_kind.value == "example":
            known_fields = fields_of(example_choice.value)
        elif source_kind.value == "config" and Path(config_path.value).is_file():
            known_fields = list(load_any(config_path.value).fields)
        elif source_kind.value == "runs" and dimension.value == 1:
            for _s, _row in zip(usable_suggestions, field_rows.value):
                if _row["use"]:
                    _kind = _row["kind"]
                    known_fields.append(Field1D(_s["name"], tuple(_s.get("keys", ())), _kind,
                                                bool(_row["relative"]) and _kind == "scalar",
                                                tuple(_s.get("components", ()))))
            known_fields += parse_field_lines(extra_fields.value, dims=1)
        elif source_kind.value == "runs":
            for _s, _row in zip(usable_suggestions, field_rows.value):
                if not _row["use"]:
                    continue
                _kind = _row["kind"]
                if _s.get("components"):
                    known_fields.append(Field(_s["name"], (_s["name"],), _kind, components=tuple(_s["components"])))
                else:
                    known_fields.append(Field(_s["name"], tuple(_s["keys"]), _kind,
                                              bool(_row["relative"]) and _kind == "scalar"))
            known_fields += parse_field_lines(extra_fields.value)
            from phasewall.io import check_not_component
            for _f in known_fields:
                check_not_component(_f, "your runs")
    except ValueError as _e:
        field_problems.append(str(_e))
    return field_problems, known_fields


@app.cell
def _(config_path, source_kind):
    # 2D frame: which part of each box is described, at which resolution, as sliders in shares of the box
    # side and a coarsening factor (pipeline.frame_from_shares turns them into physical units; a value equal
    # to what the ensemble does already is passed as None, so the cache of the default frame is kept). They
    # start at the frame of the TOML file when one is given, else at the whole box, as simulated.
    from phasewall.frames import MAX_SHIFT_FRACTION
    _crop, _factor, _shift = 1.0, 1, MAX_SHIFT_FRACTION
    if source_kind.value == "config" and Path(config_path.value).is_file():
        _geo = pl.frame_geometry(pl.Source(example=None, config=config_path.value))
        if _geo is not None:
            _L = _geo.side * _geo.dx
            _crop = 1.0 if _geo.crop is None else _geo.crop / _L
            _factor = 1 if _geo.dx_target is None else max(1, round(_geo.dx_target / _geo.dx))
            _shift = MAX_SHIFT_FRACTION if _geo.max_shift is None else _geo.max_shift / _L
    crop_input = mo.ui.slider(start=0.1, stop=1.0, step=0.05, value=min(max(round(_crop, 2), 0.1), 1.0),
                              show_value=True,
                              label="part of the box described: a centred square of this share of the side "
                                    "(1: the whole box, 0.5: a square of half the side)")
    dx_target_input = mo.ui.slider(start=1, stop=max(8, _factor), step=1, value=_factor, show_value=True,
                                   label="coarsening: grid points merged per side (1: as simulated, "
                                         "2: each 2 x 2 block averaged into one point)")
    max_shift_input = mo.ui.slider(start=0.0, stop=0.5, step=0.05, value=min(max(round(_shift, 2), 0.0), 0.5),
                                   show_value=True,
                                   label="moving correlation, non-periodic frames: largest shift tried, as a share "
                                         "of the box side (0.2: a fifth)")
    min_wavelengths_input = mo.ui.slider(
        start=1, stop=10, step=0.5, value=pl.FrameSettings().min_wavelengths, show_value=True,
        label="non-periodic frames: S(k) kept for wavelengths up to the frame side divided by this (3: a third)",
    )
    periodic_input = mo.ui.dropdown(
        {"as the ensemble": None, "periodic": True, "not periodic": False},
        value="as the ensemble", label="boundaries of the box (periodic: the pattern wraps around)",
    )
    return crop_input, dx_target_input, max_shift_input, min_wavelengths_input, periodic_input


@app.cell
def _(config_path, source_kind):
    # 1D: which part of each run is described, and in how many time windows (each window is one row of the
    # table, like a snapshot in 2D). They start at the values of the TOML file when one is given, else at the
    # defaults of oned.kymo (SHARE, WINDOWS), which the examples use; this cell runs again when the file changes.
    _share, _windows = KYMO_SHARE, KYMO_WINDOWS
    if source_kind.value == "config" and Path(config_path.value).is_file():
        try:
            _ens = load_any(config_path.value)
            if hasattr(_ens, "windows"):
                _share, _windows = float(_ens.share), int(_ens.windows)
        except (ValueError, KeyError, OSError):
            pass
    share_input = mo.ui.slider(start=0.1, stop=1.0, step=0.05, value=min(max(_share, 0.1), 1.0), show_value=True,
                               label="part of each run described, counted from its end (1: the whole run, "
                                     "0.1: its last 10%)")
    windows_input = mo.ui.slider(start=1, stop=max(16, _windows), step=1, value=_windows, show_value=True,
                                 label="time windows that part is cut into (each window is one sample, "
                                       "a row of the table)")
    return share_input, windows_input


@app.cell
def _():
    # Encoders. The learned one needs PyTorch (extra [nn]); its devices are listed if present.
    # The network's controls come from the settings of nn.architecture.Network; encoders
    # registered by other packages get controls from their own settings.
    _devices = {"fastest available": None, "CPU": "cpu"}
    if HAS_TORCH:
        import torch

        if torch.backends.mps.is_available():
            _devices["Apple GPU (mps)"] = "mps"
        if torch.cuda.is_available():
            _devices["CUDA GPU"] = "cuda"
    hand_input = mo.ui.checkbox(value=True, label="hand-made descriptors")
    learned_input = mo.ui.checkbox(label="learned (CNN, one per seed)", disabled=not HAS_TORCH)
    combined_input = mo.ui.checkbox(label="combined (hand-made + learned)", disabled=not HAS_TORCH)
    jepa_input = mo.ui.checkbox(label="predictive (JEPA, one per seed; being tested)", disabled=not HAS_TORCH)
    epochs_input = mo.ui.number(start=1, stop=1000, step=1, value=pl.Describers().epochs, label="epochs")
    nn_seeds_input = mo.ui.text(value="0", label="training seeds")
    _fastest = next((k for k in ("CUDA GPU", "Apple GPU (mps)") if k in _devices), "CPU")
    _devices = {f"fastest available ({_fastest})": None, **{k: v for k, v in _devices.items() if v is not None}}
    device_input = mo.ui.dropdown(_devices, value=f"fastest available ({_fastest})",
                                  label="device that trains the learned encoders (CNN, JEPA)")
    network_inputs = spec_controls(Network)
    plugin_encoder_inputs = mo.ui.dictionary({
        _name: mo.ui.dictionary({"use": mo.ui.checkbox(label=f"{_name}: {_e.help}",
                                                       disabled=not registry.available(_e)),
                                 "settings": spec_controls(_e.settings)})
        for _name, _e in registry.encoders().items() if _name not in BUILTIN_ENCODERS
    })
    return (
        combined_input,
        device_input,
        epochs_input,
        hand_input,
        jepa_input,
        learned_input,
        network_inputs,
        nn_seeds_input,
        plugin_encoder_inputs,
    )


@app.cell
def _(known_fields):
    # Blocks of observables of the hand-made encoder: one checkbox per field and block, only
    # where the block fits the kind of the field; the defaults of the kind are ticked.
    def _offered(f):
        if not hasattr(f, "symmetry"):                     # a field of a 1D system (oned.kymo.Field1D)
            return [b for b in BLOCKS_1D if b in ALLOWED_1D[f.kind] and not (f.kind == "complex" and b == "kinks")]
        return [b for b, kinds in BLOCKS.items() if f.kind in kinds]

    block_inputs = mo.ui.dictionary({
        _f.name: mo.ui.dictionary({b: mo.ui.checkbox(value=b in _f.selected) for b in _offered(_f)})
        for _f in known_fields
    })
    return (block_inputs,)


@app.cell
def _():
    # Methods, run on every description. Thresholds start at the package defaults; the tree and
    # methods registered by other packages get controls from their settings.
    _m = pl.Methods()
    walls_input = mo.ui.checkbox(value=True, label="walls")
    alpha_input = mo.ui.number(start=1e-6, stop=0.5, value=_m.alpha, label="ALPHA (Mann-Whitney p below)")
    ratio_input = mo.ui.number(start=1.0, stop=10.0, step=0.05, value=_m.ratio, label="RATIO (crossing / inner jump)")
    min_phase_input = mo.ui.number(start=1, stop=100, step=1, value=_m.min_phase, label="MIN_PHASE (points)")
    merge_input = mo.ui.checkbox(value=_m.merge, label="join separated parts of one phase")
    robustness_input = mo.ui.checkbox(label="robustness curve (phases against RATIO)")
    tree_input = mo.ui.checkbox(value="tree" in mo.cli_args(), label="unsupervised threshold tree")
    tree_inputs = spec_controls(TreeSettings)
    mixture_input = mo.ui.checkbox(label="Gaussian mixture (k by BIC)", disabled=not HAS_SKLEARN)
    kmeans_input = mo.ui.checkbox(label="k-means (k by silhouette)", disabled=not HAS_SKLEARN)
    hdbscan_input = mo.ui.dictionary({str(s): mo.ui.checkbox(label=str(s), disabled=not HAS_SKLEARN)
                                      for s in HDBSCAN_SIZES})
    probe_input = mo.ui.checkbox(label="logistic probe (needs labels; a check, never a method)",
                                 disabled=not HAS_SKLEARN)
    curve_input = mo.ui.text(placeholder="e.g. 1, 2, 4", label="learning curve: training runs per point")
    plugin_method_inputs = mo.ui.dictionary({
        _name: mo.ui.dictionary({"use": mo.ui.checkbox(label=f"{_name}: {_e.help}",
                                                       disabled=not registry.available(_e)),
                                 "settings": spec_controls(_e.settings)})
        for _name, _e in registry.methods().items() if _name not in BUILTIN_METHODS
    })
    return (
        alpha_input,
        curve_input,
        hdbscan_input,
        kmeans_input,
        merge_input,
        min_phase_input,
        mixture_input,
        plugin_method_inputs,
        probe_input,
        ratio_input,
        robustness_input,
        tree_input,
        tree_inputs,
        walls_input,
    )


@app.cell
def _():
    workers_input = mo.ui.number(start=1, stop=os.cpu_count() or 1, step=1, value=min(8, os.cpu_count() or 1),
                                 label=f"CPU cores used in parallel to simulate and describe the runs "
                                       f"(this computer has {os.cpu_count()})")
    cache_input = mo.ui.text(value=str(mo.cli_args().get("cache") or DEFAULT_CACHE),
                             label="cache folder (simulations, descriptions and trainings are kept here)",
                             full_width=True)
    run_button = mo.ui.run_button(label="Run", kind="success", full_width=True)
    return cache_input, run_button, workers_input


@app.cell
def _(
    alpha_input,
    block_inputs,
    cache_input,
    combined_input,
    config_label,
    config_path,
    crop_input,
    curve_input,
    demo_defaults,
    device_input,
    dimension,
    dt_input,
    dx_input,
    dx_target_input,
    epochs_input,
    example_choice,
    field_problems,
    grid_defaults,
    grid_inputs,
    hand_input,
    hdbscan_input,
    inspection,
    inspection_problem,
    jepa_input,
    julia_input,
    kmeans_input,
    known_fields,
    last_input,
    learned_input,
    max_shift_input,
    merge_input,
    min_phase_input,
    min_wavelengths_input,
    mixture_input,
    network_inputs,
    nn_seeds_input,
    order_input,
    pair_dt_input,
    parameter_choice,
    periodic_input,
    periodic_own,
    plugin_encoder_inputs,
    plugin_method_inputs,
    probe_input,
    quick,
    ratio_input,
    robustness_input,
    runs_label,
    runs_path,
    seeds_input,
    setting_inputs,
    share_input,
    snapshots_input,
    source_kind,
    table_label,
    table_parameters,
    table_path,
    table_prefix,
    time_regex_input,
    time_unit_input,
    tree_input,
    tree_inputs,
    walls_input,
    windows_input,
    workers_input,
):
    # The settings of the pipeline, from the controls. Mistakes are collected as messages, shown
    # in the panel; settings is None until they are fixed.
    def _source() -> pl.Source:
        kind = source_kind.value
        if kind == "example":
            grid = {p: grid_values(t, grid_defaults[p]) for p, t in grid_inputs.value.items()}
            changed = {k: type(demo_defaults[k])(v) for k, v in setting_inputs.value.items() if v != demo_defaults[k]}
            return pl.Source(example=example_choice.value, grid=grid, seeds=int(seeds_input.value),
                             quick=quick.value, settings=changed)
        if kind == "runs":
            if inspection_problem:
                raise ValueError(inspection_problem)
            if inspection is None:
                raise ValueError("give the table of runs")
            if field_problems:
                raise ValueError(field_problems[0])
            if not known_fields:
                raise ValueError("tick at least one field")
            if not parameter_choice.value:
                raise ValueError("choose the parameter columns")
            _dx = optional_float(dx_input.value, "grid spacing")
            if _dx is None and "dx" not in inspection["runs"].columns:
                raise ValueError("give the grid spacing dx (or a column dx in the table of runs)")
            if dimension.value == 1:
                # 1D: one file per run (file = ""), the axes as chosen, dt between two rows
                _dt = optional_float(dt_input.value, "time between two rows")
                if _dt is None:
                    raise ValueError("give the time between two rows of the kymograph (dt)")
                _ens1 = {"file": "", "julia_order": julia_input.value, "order": order_input.value, "dx": _dx,
                         "dt": _dt, "periodic": periodic_own.value}
                _cfg = Path(cache_input.value) / "configs" / (
                    pl.key(str(Path(runs_path.value).resolve()), parameter_choice.value,
                           [dataclasses.asdict(f) for f in known_fields], _ens1) + ".toml")
                write_config1d(_cfg, runs_path.value, list(parameter_choice.value), known_fields, **_ens1)
                return pl.Source(example=None, config=str(_cfg), label=runs_label.value or None)
            _ens = {"snapshots": snapshots_input.value, "time_regex": time_regex_input.value,
                    "time_unit": float(time_unit_input.value), "julia_order": julia_input.value, "dx": _dx,
                    "periodic": periodic_own.value, "last": int(last_input.value),
                    "pair_dt": optional_float(pair_dt_input.value, "time between the snapshots of a pair")}
            # written into the cache folder; an unchanged configuration is not rewritten
            _cfg = Path(cache_input.value) / "configs" / (
                pl.key(str(Path(runs_path.value).resolve()), parameter_choice.value,
                       [dataclasses.asdict(f) for f in known_fields], _ens) + ".toml")
            write_config(_cfg, runs_path.value, list(parameter_choice.value), known_fields, **_ens)
            return pl.Source(example=None, config=str(_cfg), label=runs_label.value or None)
        if kind == "config":
            if not Path(config_path.value).is_file():
                raise ValueError(f"no TOML file at {config_path.value!r}")
            return pl.Source(example=None, config=config_path.value, label=config_label.value or None)
        if not Path(table_path.value).is_file():
            raise ValueError(f"no CSV file at {table_path.value!r}")
        params = tuple(p.strip() for p in table_parameters.value.split(",") if p.strip())
        if not params:
            raise ValueError("give the parameter columns of the table")
        return pl.Source(example=None, table=table_path.value, parameters=params,
                         label=table_label.value or None, prefix=table_prefix.value or None)

    settings, settings_problems, settings_notes = None, [], []
    frame_note = ""
    try:
        _learned = learned_input.value or combined_input.value
        _blocks = {}
        for _f in known_fields:
            _boxes = block_inputs.value.get(_f.name)
            _chosen = tuple(b for b, on in _boxes.items() if on) if _boxes is not None else _f.selected
            if not _chosen:
                raise ValueError(f"choose at least one block of observables for {_f.name}")
            if set(_chosen) != set(_f.selected):
                _blocks[_f.name] = _chosen
        _others_m = {n: settings_from(registry.methods()[n].settings, v["settings"])
                     for n, v in plugin_method_inputs.value.items() if v["use"]}
        if tree_input.value:
            _others_m["tree"] = settings_from(TreeSettings, tree_inputs.value)
        _others_e = {n: settings_from(registry.encoders()[n].settings, v["settings"])
                     for n, v in plugin_encoder_inputs.value.items() if v["use"]}
        if jepa_input.value:
            _others_e["jepa"] = JepaSettings(int(epochs_input.value), parse_ints(nn_seeds_input.value) or (0,))
        _one_d = pl.dims_for(_src := _source()) == 1
        _network = settings_from(Network, network_inputs.value)
        _geo = None if _one_d else pl.frame_geometry(_src)
        _frame = (None, None, None) if _geo is None else pl.frame_from_shares(
            _geo, float(crop_input.value), int(dx_target_input.value), float(max_shift_input.value),
            periodic_input.value)
        settings = pl.Settings(
            source=_src,
            frame=pl.FrameSettings(
                crop=_frame[0], dx_target=_frame[1], max_shift=_frame[2],
                min_wavelengths=float(min_wavelengths_input.value),
                periodic=periodic_input.value,
                share=float(share_input.value) if _one_d else None,
                windows=int(windows_input.value) if _one_d else None,
            ),
            describers=pl.Describers(
                hand=hand_input.value, learned=learned_input.value, combined=combined_input.value,
                epochs=int(epochs_input.value), seeds=parse_ints(nn_seeds_input.value) or (0,),
                device=device_input.value, blocks=_blocks, others=_others_e, network=_network,
            ),
            methods=pl.Methods(
                walls=walls_input.value, alpha=float(alpha_input.value), ratio=float(ratio_input.value),
                min_phase=int(min_phase_input.value), merge=merge_input.value, robustness=robustness_input.value,
                mixture=mixture_input.value, kmeans=kmeans_input.value,
                hdbscan=tuple(sorted(int(k) for k, on in hdbscan_input.value.items() if on)), probe=probe_input.value,
                learning_curve=parse_ints(curve_input.value) if _learned and pl.dims_for(_src) == 2 else (),
                others=_others_m,
            ),
            execution=pl.Execution(workers=int(workers_input.value), cache=cache_input.value),
        )
        if _geo is not None:
            frame_note = pl.frame_text(_geo, settings.frame)
        for _st in [*_others_m.values(), *_others_e.values()]:
            registry.check_settings(_st)
        if _learned:
            _side = pl.frame_side(settings)
            if _side is not None:
                check_network(_network, _side)
                _channels = 2 * sum(1 if f.symmetry == "scalar" else 2 for f in known_fields) or 2   # a pair of frames
                _info = network_summary(_network, _side, in_channels=_channels)
                settings_notes.append(
                    f"views {_network.views}{' normalised' if _network.normalise else ''}, input {_info['input']} "
                    f"of {_side} points; network: channels {_info['channels']}, last map {_info['last_map']} points, "
                    f"receptive field {_info['receptive_field']} points, {_info['weights_f']:,} weights; memory of one "
                    f"training step about {_info['memory_gb_rough']:.1f} GB on the device that trains (a rough "
                    f"estimate, not measured)")
    except ValueError as _e:
        settings_problems.append(str(_e))
        settings = None
    if settings is not None:
        _kind = settings.source.kind()
        if _kind == "table" and (learned_input.value or combined_input.value):
            settings_problems.append("a table of descriptors cannot be described again: untick learned and combined")
        if _kind != "table" and not (hand_input.value or learned_input.value or combined_input.value
                                     or settings.describers.others):
            settings_problems.append("choose at least one encoder")
        if not pl.methods_list(settings.methods) and not settings.methods.probe:
            settings_problems.append("choose at least one method")
        if settings_problems:
            settings = None
    _labelled = source_kind.value == "example" or config_label.value or table_label.value or runs_label.value
    if probe_input.value and not _labelled:
        settings_notes.append("the probe needs a label column; without one it is skipped")
    if curve_input.value.strip() and not (learned_input.value or combined_input.value):
        settings_notes.append("the learning curve needs the learned description; it is skipped")
    if curve_input.value.strip() and settings is not None and pl.dims_for(settings.source) == 1:
        settings_notes.append("the learning curve is not available for 1D systems yet; it is skipped")
    return frame_note, settings, settings_notes, settings_problems


@app.cell
def _(settings, settings_notes, settings_problems):
    # Messages and the cost estimate shown above the Run button.
    if settings_problems:
        status = mo.callout(mo.md("\n".join(f"- {p}" for p in settings_problems)), kind="danger")
    else:
        _rows = pl.estimate_details(settings)
        _lines = [f"- **{r['step']}**: {seconds_text(r['seconds']) if r['seconds'] is not None else 'not estimated'}"
                  f" · on {r['where']}  \n  <span style='color: #6b6a66'>{r['alternative']}</span>"
                  for r in _rows]
        _est = {r["step"]: r["seconds"] for r in _rows if r["seconds"] is not None}
        _total = sum(_est.values())
        _text = ("**Estimated cost** (steps in the cache take seconds):\n\n" + "\n".join(_lines)
                 + f"\n\n**total**: {seconds_text(_total)}") if _est else "No simulation: the cost depends on your runs."
        _notes = "\n".join(f"- {n}" for n in settings_notes)
        status = mo.vstack([mo.md(_text)] + ([mo.callout(mo.md(_notes), kind="info")] if _notes else []))
    return (status,)


@app.cell
def _(
    config_label,
    config_path,
    demo,
    dimension,
    dt_input,
    dx_input,
    example_choice,
    grid_inputs,
    julia_input,
    last_input,
    pair_dt_input,
    periodic_own,
    quick,
    runs_label,
    runs_path,
    seeds_input,
    setting_inputs,
    snapshots_input,
    source_kind,
    table_label,
    table_parameters,
    table_path,
    table_prefix,
    time_regex_input,
    time_unit_input,
):
    # 1. System: where the runs come from. Only the part of the chosen source is shown.
    _kind = source_kind.value
    if _kind == "example":
        _body = columns(
            mo.vstack([example_choice, mo.md(f"*{demo.__doc__.split(chr(10) * 2)[0]}*"), quick, seeds_input]),
            mo.vstack([mo.md("**Parameter grid**: `a, b, c`, `start:step:stop` or `start:*factor:stop`"),
                       *[mo.hstack([mo.md(f"`{p}`"), el], widths=[1, 6]) for p, el in grid_inputs.items()]]),
            mo.vstack([mo.md("**Numerical settings**"), *setting_inputs.values()]),
        )
    elif _kind == "runs" and dimension.value == 1:
        _body = columns(
            mo.vstack([runs_path, mo.md("All the files in one folder is fine: the column `path` gives each run's file "
                                        "(relative to the table). The axes are chosen in 2 · Fields, after a look at "
                                        "the first file."), julia_input]),
            mo.vstack([periodic_own, dx_input, dt_input, runs_label]),
        )
    elif _kind == "runs":
        _body = columns(
            mo.vstack([runs_path, snapshots_input, time_regex_input, time_unit_input]),
            mo.vstack([julia_input, periodic_own, dx_input, last_input, pair_dt_input, runs_label]),
        )
    elif _kind == "config":
        _body = columns(config_path, config_label)
    else:
        _body = columns(mo.vstack([table_path, table_parameters]), mo.vstack([table_label, table_prefix]))
    card("1 · System", mo.vstack([source_kind, _body], gap=1))
    return


@app.cell
def _(
    dimension,
    extra_fields,
    field_rows,
    inspection,
    inspection_problem,
    julia_input,
    known_fields,
    order_input,
    parameter_choice,
    source_kind,
):
    # 2. Fields: for your runs, the datasets of the first snapshot and the fields you confirm;
    # otherwise the fields of the example or of the TOML file.
    _kind = source_kind.value
    if _kind == "runs":
        if inspection_problem:
            _body = mo.callout(inspection_problem, kind="warn")
        elif inspection is None:
            _body = mo.md("Give the table of runs above: the first snapshot of the first run is inspected.")
        else:
            _unused = [s for s in inspection["suggestions"] if s["kind"] is None]
            _axes = []
            if dimension.value == 1:
                _axes = [mo.md("**Axes.** Choose which axis of the arrays is space; the kymograph of the first "
                               "field is drawn with x across and t upward. If the stripes or waves look "
                               "transposed, choose the other order."), order_input,
                         kymo_preview(inspection, order_input.value, julia_input.value)]
            _body = columns(
                mo.vstack([
                    *_axes,
                    mo.md(f"Datasets of `{inspection['file'].name}` (shape as read"
                          + (")" if dimension.value == 1 else ", [x, y, component])") + ":"),
                    mo.ui.table(inspection["datasets"][["name", "shape", "dtype"]].astype(str), selection=None,
                                pagination=False),
                    *([mo.md("Not used: " + "; ".join(f"`{s['name']}` ({s['reason']})" for s in _unused))]
                      if _unused else []),
                ]),
                mo.vstack([
                    mo.md("Suggested from the names; **check every kind**. "
                          + ("Pairs saved apart are grouped: Re and Im parts as a complex amplitude, p_x and p_y "
                             "as a polarity (kinds on a line: scalar, velocity, polarity, complex)."
                             if dimension.value == 1 else
                             "Components saved apart are grouped: a component alone is not invariant under rotations.")),
                    *[mo.vstack([r["use"], mo.hstack([r["kind"], r["relative"]], justify="start")])
                      for r in field_rows],
                    extra_fields, parameter_choice,
                ]),
            )
    elif _kind == "table":
        _body = mo.md("A table of descriptors has no fields: its columns are the description.")
    elif known_fields:
        _body = mo.md("  \n".join(f"**{f.name}**: {f.kind}" for f in known_fields))
    else:
        _body = mo.md("Give the TOML file above.")
    card("2 · Fields", _body)
    return


@app.cell
def _(
    block_inputs,
    combined_input,
    device_input,
    dimension,
    epochs_input,
    hand_input,
    jepa_input,
    learned_input,
    network_inputs,
    nn_seeds_input,
    known_fields,
    plugin_encoder_inputs,
    source_kind,
):
    # 3. Encoders: from a snapshot (2D) or a window of a kymograph (1D) to a vector h.
    if source_kind.value == "table":
        _body = mo.md("The table already holds the description.")
    else:
        _hand = mo.vstack([mo.md("**Hand-made**"), hand_input,
                           *([mo.md("Blocks of observables per field (a box only where the block fits the kind "
                                    "of the field; hover a name for its definition):"),
                              block_grid(block_inputs, known_fields)] if len(block_inputs) else [])])
        _nn = [] if HAS_TORCH else [mo.md("*PyTorch is not installed: `pip install -e \".[nn]\"`*")]
        _learned = mo.vstack([mo.md("**Learned** (convolutional network)"), learned_input, combined_input,
                              jepa_input, *_nn,
                              mo.accordion({"training and network shape": mo.vstack(
                                  [epochs_input, nn_seeds_input, *network_inputs.values()])})])
        _plugins = [mo.vstack([v["use"], v["settings"]]) for v in plugin_encoder_inputs.values()]
        _body = mo.vstack([_hand, mo.md("---"), columns(_learned, *(
            [mo.vstack([mo.md("**Registered by other packages**"), *_plugins])] if _plugins else []))], gap=1)
        if dimension.value == 1:
            _body = mo.vstack([_body, mo.md("*1D: the network uses its channels, head and temperature; the views "
                                            "are crops of the window and the mirror (rotation and zoom do not "
                                            "apply).*").style({"font-size": "0.85em", "opacity": "0.75"})])
    card("3 · Encoders", _body)
    return


@app.cell
def _(
    alpha_input,
    curve_input,
    hdbscan_input,
    kmeans_input,
    merge_input,
    min_phase_input,
    mixture_input,
    plugin_method_inputs,
    probe_input,
    ratio_input,
    robustness_input,
    tree_input,
    tree_inputs,
    walls_input,
):
    # 4. Methods: from h to phases; and the checks, which may use labels only to score.
    _sk = [] if HAS_SKLEARN else [mo.md("*scikit-learn is not installed: `pip install -e \".[baselines]\"`*")]
    _plugins = [mo.vstack([v["use"], v["settings"]]) for v in plugin_method_inputs.values()]
    card("4 · Methods", columns(
        mo.vstack([mo.md("**On the parameter grid**"), walls_input, alpha_input, ratio_input, min_phase_input,
                   merge_input, robustness_input]),
        mo.vstack([mo.md("**Threshold tree**"), tree_input,
                   mo.accordion({"tree settings": mo.vstack(list(tree_inputs.values()))})]),
        mo.vstack([mo.md("**Blind grouping**"), *_sk, mixture_input, kmeans_input,
                   mo.vstack([mo.md("HDBSCAN, minimum sizes"),
                              mo.hstack(list(hdbscan_input.values()), justify="start", wrap=True)]),
                   *([mo.md("**Registered by other packages**"), *_plugins] if _plugins else [])]),
        mo.vstack([mo.md("**Checks** (labels only to score)"), probe_input, curve_input]),
    ))
    return


@app.cell
def _(
    cache_input,
    crop_input,
    device_input,
    dimension,
    dx_target_input,
    frame_note,
    max_shift_input,
    min_wavelengths_input,
    periodic_input,
    run_button,
    share_input,
    source_kind,
    status,
    windows_input,
    workers_input,
):
    # 5. Frame (2D) or windows (1D), folded; then the run bar: execution on the left, the estimate (or the
    # problems) and the Run button on the right. While anything runs, explore.css greys the button out and
    # makes it unclickable (marimo shows an hourglass, .running-app-icon, while a cell runs).
    _parts = []
    if source_kind.value != "table" and dimension.value == 1:
        _parts = [card("5 · Windows of each kymograph", mo.accordion({"show": mo.vstack([
            share_input, windows_input, periodic_input])}))]
    elif source_kind.value != "table":
        _parts = [card("5 · Frame of each snapshot", mo.accordion({"show": mo.vstack([
            mo.md("Which part of each box the encoders see, at which resolution. The sliders start at the "
                  "frame of the ensemble (the whole box, as simulated, unless its TOML file says otherwise)."),
            crop_input, dx_target_input, max_shift_input, min_wavelengths_input, periodic_input,
            *([mo.callout(mo.md(frame_note), kind="neutral")] if frame_note else [])])}))]
    _button = mo.Html(f'<div class="pw-run-button">{run_button.text}</div>')
    mo.vstack([
        *_parts,
        card("Run", columns(mo.vstack([mo.md("**Execution**"), workers_input, device_input,
                                       mo.md("Simulations and hand-made descriptions use the CPU cores; only "
                                             "the learned encoders (CNN, JEPA) use the GPU."), cache_input]),
                            mo.vstack([status, _button], gap=1))),
    ], gap=1.2)
    return


@app.cell
def _():
    # The last results are kept in a marimo state, so that they stay on screen while the
    # settings are being changed; the run cell below replaces them.
    get_results, set_results = mo.state(None)
    return get_results, set_results


@app.cell
def _(run_button, set_results, settings):
    # The run. It goes on only when Run was pressed (or with `python apps/explore.py --run`).
    _auto = "run" in mo.cli_args()
    mo.stop(settings is None or not (run_button.value or _auto))
    _bars = Bars()
    _t0 = time.time()
    try:
        _res = pl.run(settings, progress=_bars)
    except (ValueError, RuntimeError, ImportError, FileNotFoundError, KeyError) as _e:
        _bars.close()
        mo.stop(True, mo.callout(mo.md(f"**The run stopped:** {_e}"), kind="danger"))
    _bars.close()
    _seconds = time.time() - _t0
    set_results({"res": _res, "settings": settings, "seconds": _seconds, "steps": dict(_bars.seconds)})
    _steps = ", ".join(f"{k} {seconds_text(v)}" for k, v in _bars.seconds.items())
    mo.md(f"Done in {seconds_text(_seconds)}" + (f" ({_steps})." if _steps else " (all steps from the cache)."))
    return


@app.cell
def _(get_results, settings):
    results = get_results()
    # the hint is hidden by explore.css while a cell runs (.pw-idle-hint)
    mo.stop(results is None, mo.Html('<div class="pw-idle-hint">'
                                     + mo.callout(mo.md("Choose the settings above and press **Run**."),
                                                  kind="info").text + "</div>"))
    res = results["res"]
    _stale = settings is not None and results["settings"] != settings
    _n_runs = len(res.runs)
    _n_points = len(res.runs[res.parameters].drop_duplicates())
    _head = mo.md(
        "## Results\n\n"
        f"**{_n_runs} runs** at **{_n_points} parameter points** ({', '.join(res.parameters)}); "
        f"descriptions: {', '.join(res.tables)}; computed in {seconds_text(results['seconds'])}."
    )
    mo.vstack([_head] + ([mo.callout(
        "The settings changed since these results were computed: press Run to update them.", kind="warn")]
        if _stale else []))
    return (res,)


@app.cell
def _(res):
    # Which description the results show: a tab bar like the results tabs (its value is the
    # name of the open tab), each tab saying what the description is. Which method the phase
    # map and the snapshots show (the walls or the tree); which field the snapshots show.
    # open_tab keeps the results tab open when the description changes: a plain dict, updated
    # by on_change, read when the results cell runs again (mo.state would rerun it on every click).
    description = mo.ui.tabs({name: mo.md(description_text(name, table)) for name, table in res.tables.items()})
    open_tab = {"name": "Phase map"}
    _maps = [m for m in ("walls", "tree") if any(k[1] == m for k in res.extras)]
    map_method = mo.ui.dropdown(_maps or ["walls"], value=(_maps or ["walls"])[0], label="phases from")
    _fields = [f.name for f in res.ensemble.fields] if res.ensemble is not None else []
    snapshot_field = mo.ui.dropdown(_fields, value=_fields[0] if _fields else None, label="field")
    return description, map_method, open_tab, snapshot_field


@app.cell
def _(description, map_method, open_tab, res, snapshot_field):
    # The results, in tabs. Each tab is a function, drawn only when the tab is opened (lazy).
    _name = description.value
    _one_d = res.ensemble is not None and hasattr(res.ensemble, "windows")
    _walls = res.walls.get(_name)
    _tree = res.extras.get((_name, "tree"))
    _shown = _tree.as_walls() if (map_method.value == "tree" and _tree is not None) else _walls

    def _figure(make):
        try:
            return mo.image(png(make()))
        except ValueError as e:       # for example a phase map of a scan of three parameters
            return mo.callout(str(e), kind="warn")

    def _reference(w):
        if res.labels is None:
            return None
        return report.node_labels(w, res.runs.assign(reference=res.labels.loc[res.runs.run].to_numpy()))

    def _comparison():
        if res.scores is None or not len(res.scores):
            return mo.md("No grouping method was run.")
        parts = [
            mo.md("One row per description and method. **groups**: groups found; **pieces**: connected "
                  "pieces of the groups on the parameter grid; **agree**: points whose label is the most "
                  "common label of their group; **ari**: adjusted Rand index with the labels (1: same "
                  "partition, 0: chance), **ari_all** with noise counted as a group; **noise**: points and "
                  "share of runs left out by HDBSCAN. Labels are used only here, never by a method."),
            mo.ui.table(res.scores.round(3), selection=None, pagination=False),
        ]
        if res.probe:
            parts.append(mo.md("**Logistic probe** (balanced accuracy on held-out parameter points; it sees "
                               "the labels, so it is a ceiling, not a method): "
                               + ", ".join(f"{k} {v:.3f}" for k, v in res.probe.items())))
        if len(res.parameters) == 2:
            parts.append(_figure(res.methods_map))
        return mo.vstack(parts)

    def _phase_map():
        if _shown is None:
            return mo.md("Neither the walls nor the tree were run.")
        what = "tree" if _shown is not _walls else "walls"
        if what == "walls":
            panels = ("Left: jump across each link. Middle: phases and walls; a ring marks a mixed point (its "
                      "runs end in different states), a square a point reassigned to the next phase (its snapshots "
                      "look more like that phase than like its own). Right: labels, if known (never seen by "
                      "PhaseWall).")
        else:
            panels = ("Left: phases and the walls between them (the tree places its splits by thresholds "
                      "on descriptors, see the Rules tab, so there is no jump to show). Right: labels, if "
                      "known (never seen by PhaseWall).")
        parts = [
            map_method,
            mo.md(f"**{_shown.n_phases} phase{'' if _shown.n_phases == 1 else 's'}** found by the **{what}** "
                  f"on **{_name}**. {panels}"),
            _figure(lambda: report.make_phase_map(_shown, _reference(_shown))),
        ]
        if what == "walls" and _shown.n_phases > 1:
            _fields = res.ensemble.fields if res.ensemble is not None else None
            _obs = report.wall_observables_table(_shown, res.tables[_name], fields=_fields)
            parts += [mo.md("**What changes across each wall**: for each pair of neighbouring phases, the "
                            "descriptors whose jump across their wall is largest compared with their jumps inside "
                            "the two phases (**ratio**), and their median value on each side of the wall. They "
                            "describe the walls found; they play no part in finding them."),
                      mo.ui.table(_obs.round(3), selection=None, pagination=False)]
        if _shown.n_phases > 1:
            # written on the hand-made descriptors when the phases were found on a learned description
            from phasewall.describe import descriptor_columns
            _table = res.tables[_name]
            if all(c.startswith("cnn.") for c in descriptor_columns(_table)) and "hand" in res.tables:
                _table = res.tables["hand"]
            _fields = res.ensemble.fields if res.ensemble is not None else None
            _ops = report.order_parameters_table(_shown, _table, top=OP_SHOWN, fields=_fields)
            if not _ops.empty:
                _pairs = report.order_parameters_shown(_ops[_ops.a >= 0])
                _rest = report.order_parameters_shown(_ops[_ops.a < 0]).drop(columns="shared neighbour")
                parts += [mo.md("**Candidate order parameters**. " + report.ORDER_PARAMETERS_NOTE),
                          mo.md("*Between pairs of phases* (a | b: the rule names b)"),
                          mo.ui.table(_pairs, selection=None, pagination=False)]
                if len(_rest):
                    parts += [mo.md("*Each phase against all the others*"),
                              mo.ui.table(_rest, selection=None, pagination=False)]
        return mo.vstack(parts)

    def _snapshots():
        if _shown is None or res.ensemble is None:
            return mo.md("Snapshots need the walls or the tree, and runs (not a table).")
        _field = next(f for f in res.ensemble.fields if f.name == snapshot_field.value)
        if _one_d:
            from phasewall.oned.report1d import make_kymographs
            text, make = ("The described part of one run at the central point of each phase (x across, t up), "
                          "on a common colour scale."), lambda: make_kymographs(_shown, res.ensemble, _field)
        else:
            text, make = ("The last snapshot of one run at the central point of each phase, on a common colour "
                          "scale."), lambda: report.make_snapshots(_shown, res.ensemble, _field)
        return mo.vstack([mo.hstack([map_method, snapshot_field], justify="start"), mo.md(text), _figure(make)])

    def _rules():
        if _tree is None:
            return mo.md("The tree was not run (tick it under Methods).")
        _fields = res.ensemble.fields if res.ensemble is not None else None
        _splits = _tree.splits.round(4)
        _used = list(dict.fromkeys(_splits.loc[_splits.accepted, "column"])) if len(_splits) else []
        if len(_splits):
            _splits.insert(2, "meaning", [explain(c, _fields) for c in _splits.column])
        parts = [mo.md(f"**{_tree.n_phases} phase{'' if _tree.n_phases == 1 else 's'}** ({_tree.halves_by} halves). Each phase is the set of "
                       "points that satisfy its rule; thresholds are in the units of the descriptor."),
                 mo.md("```\n" + rules_text(_tree) + "\n```"),
                 *([mo.md("**The descriptors in the rules:**\n\n" + "\n".join(
                     f"- `{c}`: {explain(c, _fields)}" for c in _used))] if _used else []),
                 mo.md("Splits tried (**both_halves**: share of points the two halves put on the same sides):"),
                 mo.ui.table(_splits, selection=None, pagination=False)]
        if len(_tree.merges):
            parts += [mo.md("Separated leaves compared (joined when the gap between their cores is below RATIO "
                            "times their spread):"), mo.ui.table(_tree.merges.round(4), selection=None)]
        return mo.vstack(parts)

    def _walls_tab():
        if _walls is None:
            return mo.md("The walls were not run.")
        curve = res.robustness.get(_name)
        parts = [mo.md(report.summary(_walls, curve))]
        if curve is not None:
            parts.append(_figure(lambda: res.robustness_plot(_name)))
        tables = {"points (nodes)": mo.ui.table(_walls.nodes.round(4), selection=None),
                  "links (edges)": mo.ui.table(_walls.edges.round(4), selection=None),
                  "cuts tried": mo.ui.table(_walls.cuts.round(4), selection=None)}
        if len(_walls.merges):
            tables["separated phases compared"] = mo.ui.table(_walls.merges.round(4), selection=None)
        parts.append(mo.ui.tabs(tables))
        return mo.vstack(parts)

    def _learned_tab():
        parts = []
        net = res.settings.describers.network
        if res.training is not None and "alignment" in res.training.columns:
            parts += [mo.md(f"Network: channels {net.channels}, head {net.hidden} and {net.projection}, "
                            f"temperature {net.temperature}. Training: InfoNCE loss (dotted: its value at collapse) "
                            "and effective rank of h. Choose a network by these label-free diagnostics, not by "
                            "the scores."),
                      _figure(res.training_plot)]
        elif res.training is not None:
            parts += [mo.md("Predictive training (JEPA): loss and spread of h per epoch."),
                      mo.ui.table(res.training.round(4), selection=None)]
        if res.learning_curve is not None:
            parts += [mo.md("Learning curve: phases found against training runs per parameter point."),
                      _figure(res.learning_curve_plot)]
        return mo.vstack(parts) if parts else mo.md("No learned description in these results.")

    def _export():
        def csv(df):
            return lambda: df.to_csv(index=False).encode()

        items = [mo.download(csv(res.tables[_name]), filename=f"{_name}_descriptors.csv",
                             label=f"descriptors of {_name}")]
        if res.scores is not None and len(res.scores):
            items.append(mo.download(csv(res.scores), filename="scores.csv", label="scores"))
        if _name in res.groups:
            items.append(mo.download(csv(res.groups[_name]), filename=f"{_name}_groups.csv", label="groups of runs"))
        if _walls is not None:
            items += [
                mo.download(csv(_walls.nodes), filename=f"{_name}_nodes.csv", label="points (nodes)"),
                mo.download(csv(_walls.edges), filename=f"{_name}_edges.csv", label="links (edges)"),
                mo.download(csv(_walls.cuts), filename=f"{_name}_cuts.csv", label="cuts"),
            ]
        if _tree is not None:
            items.append(mo.download(lambda: rules_text(_tree).encode(), filename=f"{_name}_tree_rules.txt",
                                     label="rules of the tree"))
        if _shown is not None and len(res.parameters) == 2:
            items.append(mo.download(lambda: png(report.make_phase_map(_shown, _reference(_shown))),
                                     filename=f"{_name}_phase_map.png", label="phase map (PNG)"))
        return mo.vstack([mo.md(f"Files of **{_name}** and of the comparison:"), mo.hstack(items, wrap=True,
                                                                                           justify="start")])

    _tabs = {
        "Phase map": _phase_map,
        "Comparison": _comparison,
        "Kymographs" if _one_d else "Snapshots": _snapshots,
        "Rules (tree)": _rules,
        "Walls": _walls_tab,
        "Learned description": _learned_tab,
        "Export": _export,
    }
    # Without a browser (--run, e.g. `marimo export html apps/explore.py -- --run`) every tab is
    # drawn at once, so that the export holds all of them. results_tabs is a global name (no
    # leading underscore): marimo sends the clicks of an element to Python only then, so that
    # on_change runs. No other cell reads it, so a click reruns nothing.
    if "run" in mo.cli_args():
        results_tabs = mo.ui.tabs({k: f() for k, f in _tabs.items()})
    else:
        results_tabs = mo.ui.tabs(_tabs, value=open_tab["name"] if open_tab["name"] in _tabs else None,
                                  lazy=True, on_change=lambda name: open_tab.update(name=name))
    mo.vstack([mo.md("**Description shown**"), description, results_tabs])
    return (results_tabs,)


if __name__ == "__main__":
    app.run()
