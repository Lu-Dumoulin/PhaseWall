"""phasewall explorer: a marimo notebook that is also a dashboard.

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
app = marimo.App(width="full", app_title="phasewall explorer")

with app.setup:
    import dataclasses
    import io
    import os
    import time
    from importlib.util import find_spec
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    from phasewall import pipeline as pl
    from phasewall import registry, report
    from phasewall.builtins import TreeSettings
    from phasewall.examples import EXAMPLES, example, fields_of
    from phasewall.fields import BLOCKS, KINDS, Field
    from phasewall.io import inspect_snapshot, list_snapshots, load_ensemble, suggest_fields, write_config
    from phasewall.nn.architecture import Network
    from phasewall.nn.architecture import check as check_network
    from phasewall.nn.architecture import summary as network_summary
    from phasewall.trees import rules_text

    HAS_TORCH = find_spec("torch") is not None          # learned description (extra [nn])
    HAS_SKLEARN = find_spec("sklearn") is not None      # blind grouping and probe (extra [baselines])
    HDBSCAN_SIZES = (5, 10, 20, 40)                     # offered sizes, as phasewall.baselines.HDBSCAN_SIZES
    BUILTIN_ENCODERS = ("hand", "learned")               # have their own controls below
    BUILTIN_METHODS = ("walls", "tree", "mixture", "kmeans", "hdbscan")
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
def parse_field_lines(text: str) -> list:
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
            raise ValueError(f"{line!r}: write 'name kind dataset' (kinds: {', '.join(KINDS)})")
        name, kind, data, flags = words[0], words[1], words[2].split(","), set(words[3:])
        unknown = flags - {"relative", "single"}
        if unknown:
            raise ValueError(f"{line!r}: unknown words {sorted(unknown)} (relative, single)")
        if len(data) == 2:
            fields.append(Field(name, (name,), kind, components=tuple(data)))
        else:
            fields.append(Field(name, (data[0],), kind, "relative" in flags, single_component="single" in flags))
    return fields


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
    # phasewall explorer

    phasewall finds the phases of a simulated system from runs on a grid of parameters: it
    describes every snapshot (encoders), then finds the phases from the descriptions (methods).
    Choose the system, its fields, the encoders and the methods in the panel on the left, then
    press **Run**. Steps already computed with the same settings are read from the cache.
    """)
    return


@app.cell
def _():
    # Source of the runs. mo.ui.radio is PlutoUI's Radio; its .value is read in other cells.
    _own = mo.cli_args().get("runs")
    source_kind = mo.ui.radio(
        {"an example": "example", "your runs (CSV of runs and their snapshot files)": "runs",
         "an ensemble (TOML file)": "config", "a table of descriptors (CSV)": "table"},
        value="your runs (CSV of runs and their snapshot files)" if _own else "an example", label="System",
    )
    example_choice = mo.ui.dropdown(list(EXAMPLES), value="swift-hohenberg", label="example")
    quick = mo.ui.checkbox(value="quick" in mo.cli_args(),
                           label="quick: small grid and box, seconds (too coarse for all states)")
    runs_path = mo.ui.text(value=str(_own or ""), placeholder="path/to/runs.csv", full_width=True,
                           label="table of runs: a column `path` (run folder) and one per parameter")
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
def _(julia_input, runs_path, snapshots_input, source_kind, time_regex_input, time_unit_input):
    # Your runs: the first snapshot of the first run is inspected; its datasets are listed and
    # fields are suggested (phasewall.io.suggest_fields). Nothing is used until you confirm the
    # kind of each field below; a component alone (vx) is refused as a scalar.
    inspection, inspection_problem = None, None
    if source_kind.value == "runs" and runs_path.value.strip():
        try:
            _runs = pd.read_csv(runs_path.value)
            if "path" not in _runs.columns:
                raise ValueError("the table of runs needs a column `path`")
            _folder = (Path(runs_path.value).resolve().parent / _runs["path"].iloc[0])
            _snaps = list_snapshots(_folder, snapshots_input.value, time_regex_input.value, float(time_unit_input.value))
            if not _snaps:
                raise ValueError(f"no file {snapshots_input.value!r} with a time matching {time_regex_input.value!r} "
                                 f"in {_folder}")
            _datasets = inspect_snapshot(_snaps[-1][1], julia_input.value)
            inspection = {"runs": _runs, "file": _snaps[-1][1], "datasets": _datasets,
                          "suggestions": suggest_fields(_datasets)}
        except (OSError, ValueError, KeyError) as _e:
            inspection_problem = str(_e)
    _usable = [s for s in (inspection["suggestions"] if inspection else []) if s["kind"] is not None]
    field_rows = mo.ui.array([
        mo.ui.dictionary({
            "use": mo.ui.checkbox(value=True, label=f"**{s['name']}** from {', '.join(s.get('components') or s['keys'])}"),
            "kind": mo.ui.dropdown(list(KINDS), value=s["kind"], label="kind"),
            "relative": mo.ui.checkbox(value=bool(s.get("relative")), label="relative (scalars)"),
        }) for s in _usable
    ])
    extra_fields = mo.ui.text_area(placeholder="name kind dataset[,dataset] [relative] [single]", rows=2,
                                   label="other fields, one per line", full_width=True)
    _numeric = [c for c in (inspection["runs"].columns if inspection else [])
                if c not in ("run", "path", "dx") and pd.api.types.is_numeric_dtype(inspection["runs"][c])]
    parameter_choice = mo.ui.multiselect(_numeric, value=[c for c in _numeric if c != "seed"],
                                         label="parameter columns")
    usable_suggestions = _usable
    return extra_fields, field_rows, inspection, inspection_problem, parameter_choice, usable_suggestions


@app.cell
def _(config_path, example_choice, extra_fields, field_rows, source_kind, usable_suggestions):
    # The fields of the chosen system, known before any run: an example's, those of a TOML file,
    # or those you confirmed for your runs. Their kinds decide which blocks of observables are offered.
    known_fields, field_problems = [], []
    try:
        if source_kind.value == "example":
            known_fields = fields_of(example_choice.value)
        elif source_kind.value == "config" and Path(config_path.value).is_file():
            known_fields = list(load_ensemble(config_path.value).fields)
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
def _():
    # Frame: which part of each box is described, at which resolution. Empty: as the ensemble.
    crop_input = mo.ui.text(placeholder="whole box", label="crop side (physical units)")
    dx_target_input = mo.ui.text(placeholder="as simulated", label="coarsen to dx")
    max_shift_input = mo.ui.text(placeholder="box / 5", label="max shift of the moving correlation")
    min_wavelengths_input = mo.ui.number(
        start=1, stop=20, step=0.5, value=pl.FrameSettings().min_wavelengths,
        label="non-periodic S(k): frame / longest wavelength",
    )
    periodic_input = mo.ui.dropdown(
        {"as the ensemble": None, "periodic": True, "not periodic": False},
        value="as the ensemble", label="boundaries",
    )
    return crop_input, dx_target_input, max_shift_input, min_wavelengths_input, periodic_input


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
    epochs_input = mo.ui.number(start=1, stop=1000, step=1, value=pl.Describers().epochs, label="epochs")
    nn_seeds_input = mo.ui.text(value="0", label="training seeds")
    device_input = mo.ui.dropdown(_devices, value="fastest available", label="device")
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
        learned_input,
        network_inputs,
        nn_seeds_input,
        plugin_encoder_inputs,
    )


@app.cell
def _(known_fields):
    # Blocks of observables of the hand-made encoder, per field: those that fit its kind are
    # offered, the defaults of the kind are ticked.
    block_inputs = mo.ui.dictionary({
        _f.name: mo.ui.multiselect([b for b, kinds in BLOCKS.items() if _f.kind in kinds], value=list(_f.selected),
                                   label=f"{_f.name} ({_f.kind})")
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
    hdbscan_input = mo.ui.multiselect(
        [str(s) for s in HDBSCAN_SIZES], label="HDBSCAN minimum sizes", disabled=not HAS_SKLEARN
    )
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
                                 label="processes")
    cache_input = mo.ui.text(value=str(mo.cli_args().get("cache") or DEFAULT_CACHE), label="cache folder",
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
    try:
        _learned = learned_input.value or combined_input.value
        _blocks = {}
        for _f in known_fields:
            _chosen = tuple(block_inputs.value.get(_f.name, _f.selected))
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
        _network = settings_from(Network, network_inputs.value)
        settings = pl.Settings(
            source=_source(),
            frame=pl.FrameSettings(
                crop=optional_float(crop_input.value, "crop"),
                dx_target=optional_float(dx_target_input.value, "coarsen to dx"),
                max_shift=optional_float(max_shift_input.value, "max shift"),
                min_wavelengths=float(min_wavelengths_input.value),
                periodic=periodic_input.value,
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
                hdbscan=tuple(sorted(int(s) for s in hdbscan_input.value)), probe=probe_input.value,
                learning_curve=parse_ints(curve_input.value) if _learned else (), others=_others_m,
            ),
            execution=pl.Execution(workers=int(workers_input.value), cache=cache_input.value),
        )
        for _st in [*_others_m.values(), *_others_e.values()]:
            registry.check_settings(_st)
        if _learned:
            _side = pl.frame_side(settings)
            if _side is not None:
                check_network(_network, _side)
                _channels = 2 * sum(1 if f.symmetry == "scalar" else 2 for f in known_fields) or 2   # a pair of frames
                _info = network_summary(_network, _side, in_channels=_channels)
                settings_notes.append(
                    f"network: channels {_info['channels']}, last map {_info['last_map']} points, receptive field "
                    f"{_info['receptive_field']} points for a frame of {_side}, {_info['weights_f']:,} weights")
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
    return settings, settings_notes, settings_problems


@app.cell
def _(settings, settings_notes, settings_problems):
    # Messages and the cost estimate shown above the Run button.
    if settings_problems:
        status = mo.callout(mo.md("\n".join(f"- {p}" for p in settings_problems)), kind="danger")
    else:
        _est = pl.estimate(settings)
        _lines = [f"- {step}: {seconds_text(s)}" for step, s in _est.items()]
        _total = sum(_est.values())
        _text = ("**Estimated cost** (steps in the cache take seconds):\n\n" + "\n".join(_lines)
                 + f"\n\n**total**: {seconds_text(_total)}") if _est else "No simulation: the cost depends on your runs."
        _notes = "\n".join(f"- {n}" for n in settings_notes)
        status = mo.vstack([mo.md(_text)] + ([mo.callout(mo.md(_notes), kind="info")] if _notes else []))
    return (status,)


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
    demo,
    device_input,
    dx_input,
    dx_target_input,
    epochs_input,
    example_choice,
    extra_fields,
    field_rows,
    grid_inputs,
    hand_input,
    hdbscan_input,
    inspection,
    inspection_problem,
    julia_input,
    kmeans_input,
    last_input,
    learned_input,
    max_shift_input,
    merge_input,
    min_phase_input,
    min_wavelengths_input,
    mixture_input,
    network_inputs,
    nn_seeds_input,
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
    run_button,
    runs_label,
    runs_path,
    seeds_input,
    setting_inputs,
    snapshots_input,
    source_kind,
    status,
    table_label,
    table_parameters,
    table_path,
    table_prefix,
    time_regex_input,
    time_unit_input,
    tree_input,
    tree_inputs,
    usable_suggestions,
    walls_input,
    workers_input,
):
    # The control panel, in the sidebar: System, Fields, Frame, Encoders, Methods, Execution.
    _kind = source_kind.value
    _sections = {}
    if _kind == "example":
        _sections["System"] = mo.vstack([
            source_kind, example_choice, mo.md(f"*{demo.__doc__.split(chr(10) * 2)[0]}*"), quick,
            mo.md("Parameter values: `a, b, c`, `start:step:stop` or `start:*factor:stop` (geometric)."),
            *[mo.hstack([mo.md(f"`{p}`"), el], widths=[1, 5]) for p, el in grid_inputs.items()],
            seeds_input, mo.md("Numerical settings:"), *setting_inputs.values(),
        ])
    elif _kind == "runs":
        _sections["System"] = mo.vstack([source_kind, runs_path, snapshots_input, time_regex_input, time_unit_input,
                                         julia_input, dx_input, periodic_own, last_input, pair_dt_input, runs_label])
        if inspection_problem:
            _fields_view = mo.callout(inspection_problem, kind="warn")
        elif inspection is None:
            _fields_view = mo.md("Give the table of runs: the first snapshot is inspected.")
        else:
            _unused = [s for s in inspection["suggestions"] if s["kind"] is None]
            _fields_view = mo.vstack([
                mo.md(f"Datasets of `{inspection['file'].name}` (shape as read, [x, y, component]):"),
                mo.ui.table(inspection["datasets"][["name", "shape", "dtype"]].astype(str), selection=None,
                            pagination=False),
                mo.md("**Fields** (suggested from the names: check every kind; components saved apart are "
                      "grouped, since a component alone is not invariant under rotations):"),
                *[mo.vstack([r["use"], mo.hstack([r["kind"], r["relative"]], justify="start")])
                  for r in field_rows],
                *([mo.md("Not used: " + "; ".join(f"`{s['name']}` ({s['reason']})" for s in _unused))]
                  if _unused else []),
                extra_fields, parameter_choice,
            ])
        _sections["Fields"] = _fields_view
    elif _kind == "config":
        _sections["System"] = mo.vstack([source_kind, config_path, config_label])
    else:
        _sections["System"] = mo.vstack([source_kind, table_path, table_parameters, table_label, table_prefix])
    if _kind != "table":
        _sections["Frame"] = mo.vstack([crop_input, dx_target_input, max_shift_input, min_wavelengths_input,
                                        periodic_input])
        _nn = [] if HAS_TORCH else [mo.md("*PyTorch is not installed: `pip install -e \".[nn]\"`*")]
        _blocks = [mo.md("Blocks of observables per field:"), *block_inputs.values()] if len(block_inputs) else []
        _plugins = [mo.vstack([v["use"], v["settings"]]) for v in plugin_encoder_inputs.values()]
        _sections["Encoders"] = mo.vstack([
            hand_input, *_blocks, mo.md("---"), learned_input, combined_input, *_nn,
            mo.accordion({"training and network": mo.vstack([epochs_input, nn_seeds_input, device_input,
                                                             *network_inputs.values()])}),
            *([mo.md("---"), *_plugins] if _plugins else []),
        ])
    _sk = [] if HAS_SKLEARN else [mo.md("*scikit-learn is not installed: `pip install -e \".[baselines]\"`*")]
    _plugin_m = [mo.vstack([v["use"], v["settings"]]) for v in plugin_method_inputs.values()]
    _sections["Methods"] = mo.vstack([
        walls_input, alpha_input, ratio_input, min_phase_input, merge_input, robustness_input, mo.md("---"),
        tree_input, mo.accordion({"tree settings": mo.vstack(list(tree_inputs.values()))}), mo.md("---"),
        *_sk, mixture_input, kmeans_input, hdbscan_input, *([mo.md("---"), *_plugin_m] if _plugin_m else []),
        mo.md("---"), mo.md("**Checks**"), probe_input, curve_input,
    ])
    _sections["Execution"] = mo.vstack([workers_input, cache_input])
    mo.sidebar(
        mo.vstack([mo.md("### Settings"), mo.accordion(_sections, multiple=True, expanded=["System"]),
                   status, run_button]),
        width="28rem",
    )
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
    mo.stop(results is None, mo.callout(mo.md("Choose the settings on the left and press **Run**."), kind="info"))
    res = results["res"]
    _stale = settings is not None and results["settings"] != settings
    _n_runs = len(res.runs)
    _n_points = len(res.runs[res.parameters].drop_duplicates())
    _head = mo.md(
        f"**{_n_runs} runs** at **{_n_points} parameter points** ({', '.join(res.parameters)}); "
        f"descriptions: {', '.join(res.tables)}; computed in {seconds_text(results['seconds'])}."
    )
    mo.vstack([_head] + ([mo.callout(
        "The settings changed since these results were computed: press Run to update them.", kind="warn")]
        if _stale else []))
    return (res,)


@app.cell
def _(res):
    # Which description the per-description tabs show; which method the phase map and the
    # snapshots show (the walls or the tree); which field the snapshots show.
    description = mo.ui.dropdown(list(res.tables), value=next(iter(res.tables)), label="description")
    _maps = [m for m in ("walls", "tree") if any(k[1] == m for k in res.extras)]
    map_method = mo.ui.dropdown(_maps or ["walls"], value=(_maps or ["walls"])[0], label="phases from")
    _fields = [f.name for f in res.ensemble.fields] if res.ensemble is not None else []
    snapshot_field = mo.ui.dropdown(_fields, value=_fields[0] if _fields else None, label="field")
    return description, map_method, snapshot_field


@app.cell
def _(description, map_method, res, snapshot_field):
    # The results, in tabs. Each tab is a function, drawn only when the tab is opened (lazy).
    _name = description.value
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
        return mo.vstack([
            map_method,
            mo.md(f"**{_shown.n_phases} phase{'' if _shown.n_phases == 1 else 's'}** found by the **{what}** on **{_name}**. Left: jump across each "
                  "link. Middle: phases and walls; a ring marks a mixed point. Right: labels, if known (never "
                  "seen by phasewall)."),
            _figure(lambda: report.make_phase_map(_shown, _reference(_shown))),
        ])

    def _snapshots():
        if _shown is None or res.ensemble is None:
            return mo.md("Snapshots need the walls or the tree, and runs (not a table).")
        _field = next(f for f in res.ensemble.fields if f.name == snapshot_field.value)
        return mo.vstack([
            mo.hstack([map_method, snapshot_field], justify="start"),
            mo.md("The last snapshot of one run at the central point of each phase, on a common colour scale."),
            _figure(lambda: report.make_snapshots(_shown, res.ensemble, _field)),
        ])

    def _rules():
        if _tree is None:
            return mo.md("The tree was not run (tick it under Methods).")
        parts = [mo.md(f"**{_tree.n_phases} phase{'' if _tree.n_phases == 1 else 's'}** ({_tree.halves_by} halves). Each phase is the set of "
                       "points that satisfy its rule; thresholds are in the units of the descriptor."),
                 mo.md("```\n" + rules_text(_tree) + "\n```"),
                 mo.md("Splits tried (**both_halves**: share of points the two halves put on the same sides):"),
                 mo.ui.table(_tree.splits.round(4), selection=None, pagination=False)]
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
        if res.training is not None:
            parts += [mo.md(f"Network: channels {net.channels}, head {net.hidden} and {net.projection}, "
                            f"temperature {net.temperature}. Training: InfoNCE loss (dotted: its value at collapse) "
                            "and effective rank of h. Choose a network by these label-free diagnostics, not by "
                            "the scores."),
                      _figure(res.training_plot)]
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
        "Snapshots": _snapshots,
        "Rules (tree)": _rules,
        "Walls": _walls_tab,
        "Learned description": _learned_tab,
        "Export": _export,
    }
    # Without a browser (--run, e.g. `marimo export html apps/explore.py -- --run`) every tab is
    # drawn at once, so that the export holds all of them.
    if "run" in mo.cli_args():
        _view = mo.ui.tabs({k: f() for k, f in _tabs.items()})
    else:
        _view = mo.ui.tabs(_tabs, lazy=True)
    mo.vstack([description, _view])
    return


if __name__ == "__main__":
    app.run()
