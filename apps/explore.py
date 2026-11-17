"""phasewall explorer: a marimo notebook that is also a dashboard.

    marimo edit apps/explore.py    notebook: the code of every cell is visible and editable
    marimo run apps/explore.py     dashboard: only the controls and the results
    python apps/explore.py --run   no browser: runs the default settings once (a smoke test);
                                   add --quick for the quick demo, --cache DIR for another cache

The notebook only calls the package (phasewall.pipeline, phasewall.report); it computes
nothing itself. How marimo works, for a Pluto user: each cell is a function whose arguments are
the names it reads from other cells and whose return value lists the names it defines. When a
control changes, every cell that reads it runs again, as in Pluto. The expensive steps run only
when the Run button is pressed (mo.stop holds the cell otherwise), and the pipeline caches them
on disk, so a second run with another method or threshold takes seconds.
"""
# ruff: noqa: PLR1711   (marimo writes a bare `return` at the end of every cell)

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="phasewall explorer")

with app.setup:
    import io
    import os
    import time
    from importlib.util import find_spec
    from pathlib import Path

    import marimo as mo
    import numpy as np

    from phasewall import pipeline as pl
    from phasewall import report
    from phasewall.examples import EXAMPLES, example

    HAS_TORCH = find_spec("torch") is not None          # learned description (extra [nn])
    HAS_SKLEARN = find_spec("sklearn") is not None      # blind grouping and probe (extra [baselines])
    HDBSCAN_SIZES = (5, 10, 20, 40)                     # offered sizes, as phasewall.baselines.HDBSCAN_SIZES
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
    describes every snapshot, compares neighbouring parameter points and places a wall wherever
    the description jumps more than it varies within one point. Choose the runs, the frame, the
    descriptions and the methods in the panel on the left, then press **Run**. Steps already
    computed with the same settings are read from the cache.
    """)
    return


@app.cell
def _():
    # Source of the runs. mo.ui.radio is PlutoUI's Radio; its .value is read in other cells.
    source_kind = mo.ui.radio(
        {"a demo": "example", "an ensemble (TOML file)": "config", "a table of descriptors (CSV)": "table"},
        value="a demo", label="Runs from",
    )
    example_choice = mo.ui.dropdown(list(EXAMPLES), value="swift-hohenberg", label="demo")
    quick = mo.ui.checkbox(value="quick" in mo.cli_args(),
                           label="quick: small grid and box, seconds (too coarse for all states)")
    config_path = mo.ui.text(placeholder="path/to/ensemble.toml", label="TOML file", full_width=True)
    config_label = mo.ui.text(placeholder="optional", label="label column of runs.csv (for the probe)")
    table_path = mo.ui.text(placeholder="path/to/descriptors.csv", label="CSV file", full_width=True)
    table_parameters = mo.ui.text(placeholder="rho0, zeta", label="parameter columns", full_width=True)
    table_label = mo.ui.text(placeholder="optional", label="label column (for the probe)")
    table_prefix = mo.ui.text(placeholder="optional", label="descriptor columns start with")
    return (
        config_label,
        config_path,
        example_choice,
        quick,
        source_kind,
        table_label,
        table_parameters,
        table_path,
        table_prefix,
    )


@app.cell
def _(example_choice, quick):
    # Grid and numerical settings of the chosen demo, pre-filled with its defaults (or its quick
    # version). This cell runs again when the demo or `quick` changes, so the fields are reset.
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
    # Descriptions. The learned one needs PyTorch (extra [nn]); its devices are listed if present.
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
    return combined_input, device_input, epochs_input, hand_input, learned_input, nn_seeds_input


@app.cell
def _():
    # Methods, run on every description. Thresholds start at the package defaults.
    _m = pl.Methods()
    walls_input = mo.ui.checkbox(value=True, label="walls")
    alpha_input = mo.ui.number(start=1e-6, stop=0.5, value=_m.alpha, label="ALPHA (Mann-Whitney p below)")
    ratio_input = mo.ui.number(start=1.0, stop=10.0, step=0.05, value=_m.ratio, label="RATIO (crossing / inner jump)")
    min_phase_input = mo.ui.number(start=1, stop=100, step=1, value=_m.min_phase, label="MIN_PHASE (points)")
    robustness_input = mo.ui.checkbox(label="robustness curve (phases against RATIO)")
    mixture_input = mo.ui.checkbox(label="Gaussian mixture (k by BIC)", disabled=not HAS_SKLEARN)
    kmeans_input = mo.ui.checkbox(label="k-means (k by silhouette)", disabled=not HAS_SKLEARN)
    hdbscan_input = mo.ui.multiselect(
        [str(s) for s in HDBSCAN_SIZES], label="HDBSCAN minimum sizes", disabled=not HAS_SKLEARN
    )
    probe_input = mo.ui.checkbox(label="logistic probe (needs labels)", disabled=not HAS_SKLEARN)
    curve_input = mo.ui.text(placeholder="e.g. 1, 2, 4", label="learning curve: training runs per point")
    return (
        alpha_input,
        curve_input,
        hdbscan_input,
        kmeans_input,
        min_phase_input,
        mixture_input,
        probe_input,
        ratio_input,
        robustness_input,
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
    cache_input,
    combined_input,
    config_label,
    config_path,
    crop_input,
    curve_input,
    demo_defaults,
    device_input,
    dx_target_input,
    epochs_input,
    example_choice,
    grid_defaults,
    grid_inputs,
    hand_input,
    hdbscan_input,
    kmeans_input,
    learned_input,
    max_shift_input,
    min_phase_input,
    min_wavelengths_input,
    mixture_input,
    nn_seeds_input,
    periodic_input,
    probe_input,
    quick,
    ratio_input,
    robustness_input,
    seeds_input,
    setting_inputs,
    source_kind,
    table_label,
    table_parameters,
    table_path,
    table_prefix,
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

    settings, settings_problems = None, []
    try:
        _learned = learned_input.value or combined_input.value
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
                device=device_input.value,
            ),
            methods=pl.Methods(
                walls=walls_input.value, alpha=float(alpha_input.value), ratio=float(ratio_input.value),
                min_phase=int(min_phase_input.value), robustness=robustness_input.value,
                mixture=mixture_input.value, kmeans=kmeans_input.value,
                hdbscan=tuple(sorted(int(s) for s in hdbscan_input.value)), probe=probe_input.value,
                learning_curve=parse_ints(curve_input.value) if _learned else (),
            ),
            execution=pl.Execution(workers=int(workers_input.value), cache=cache_input.value),
        )
    except ValueError as _e:
        settings_problems.append(str(_e))
    if settings is not None:
        _kind = settings.source.kind()
        if _kind == "table" and (learned_input.value or combined_input.value):
            settings_problems.append("a table of descriptors cannot be described again: untick learned and combined")
        if _kind != "table" and not (hand_input.value or learned_input.value or combined_input.value):
            settings_problems.append("choose at least one description")
        if not pl.methods_list(settings.methods) and not settings.methods.probe:
            settings_problems.append("choose at least one method")
        if settings_problems:
            settings = None
    settings_notes = []
    if probe_input.value and source_kind.value != "example" and not (config_label.value or table_label.value):
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
                 + f"\n\n**total**: {seconds_text(_total)}") if _est else "Methods only: seconds."
        _notes = "\n".join(f"- {n}" for n in settings_notes)
        status = mo.vstack([mo.md(_text)] + ([mo.callout(mo.md(_notes), kind="warn")] if _notes else []))
    return (status,)


@app.cell
def _(
    alpha_input,
    cache_input,
    combined_input,
    config_label,
    config_path,
    crop_input,
    curve_input,
    demo,
    device_input,
    dx_target_input,
    epochs_input,
    example_choice,
    grid_inputs,
    hand_input,
    hdbscan_input,
    kmeans_input,
    learned_input,
    max_shift_input,
    min_phase_input,
    min_wavelengths_input,
    mixture_input,
    nn_seeds_input,
    periodic_input,
    probe_input,
    quick,
    ratio_input,
    robustness_input,
    run_button,
    seeds_input,
    setting_inputs,
    source_kind,
    status,
    table_label,
    table_parameters,
    table_path,
    table_prefix,
    walls_input,
    workers_input,
):
    # The control panel, in the sidebar. Only the part of the chosen source is shown.
    _kind = source_kind.value
    if _kind == "example":
        _source = mo.vstack([
            source_kind, example_choice, mo.md(f"*{demo.__doc__.split(chr(10) * 2)[0]}*"), quick,
        ])
        _grid = mo.vstack(
            [mo.md("Parameter values: `a, b, c`, `start:step:stop` or `start:*factor:stop` (geometric).")]
            + [mo.hstack([mo.md(f"`{p}`"), el], widths=[1, 5]) for p, el in grid_inputs.items()]
            + [seeds_input, mo.md("Numerical settings:"), *setting_inputs.values()]
        )
    elif _kind == "config":
        _source, _grid = mo.vstack([source_kind, config_path, config_label]), None
    else:
        _source = mo.vstack([source_kind, table_path, table_parameters, table_label, table_prefix])
        _grid = None
    _sections = {"Source": _source}
    if _grid is not None:
        _sections["Simulation"] = _grid
    if _kind != "table":
        _sections["Frame"] = mo.vstack([crop_input, dx_target_input, max_shift_input, min_wavelengths_input,
                                        periodic_input])
        _nn = [] if HAS_TORCH else [mo.md("*PyTorch is not installed: `pip install -e \".[nn]\"`*")]
        _sections["Description"] = mo.vstack([hand_input, learned_input, combined_input, *_nn,
                                              epochs_input, nn_seeds_input, device_input])
    _sk = [] if HAS_SKLEARN else [mo.md("*scikit-learn is not installed: `pip install -e \".[baselines]\"`*")]
    _sections["Methods"] = mo.vstack([
        walls_input, alpha_input, ratio_input, min_phase_input, robustness_input, mo.md("---"),
        *_sk, mixture_input, kmeans_input, hdbscan_input, probe_input, mo.md("---"), curve_input,
    ])
    _sections["Execution"] = mo.vstack([workers_input, cache_input])
    mo.sidebar(
        mo.vstack([mo.md("### Settings"), mo.accordion(_sections, multiple=True, expanded=["Source"]),
                   status, run_button]),
        width="26rem",
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
    # Which description the per-description tabs show; which field the snapshots show.
    description = mo.ui.dropdown(list(res.tables), value=next(iter(res.tables)), label="description")
    _fields = [f.name for f in res.ensemble.fields] if res.ensemble is not None else []
    snapshot_field = mo.ui.dropdown(_fields, value=_fields[0] if _fields else None, label="field")
    return description, snapshot_field


@app.cell
def _(description, res, snapshot_field):
    # The results, in tabs. Each tab is a function, drawn only when the tab is opened (lazy).
    _name = description.value
    _walls = res.walls.get(_name)

    def _figure(make):
        try:
            return mo.image(png(make()))
        except ValueError as e:       # for example a phase map of a scan of three parameters
            return mo.callout(str(e), kind="warn")

    def _comparison():
        if res.scores is None or not len(res.scores):
            return mo.md("No grouping method was run.")
        parts = [
            mo.md("One row per description and method. **groups**: groups found; **pieces**: connected "
                  "pieces of the groups on the parameter grid; **agree**: points whose label is the most "
                  "common label of their group; **ari**: adjusted Rand index with the labels (1: same "
                  "partition, 0: chance), **ari_all** with noise counted as a group; **noise**: points and "
                  "share of runs left out by HDBSCAN."),
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
        if _walls is None:
            return mo.md("The walls were not run.")
        return mo.vstack([
            mo.md(f"**{_walls.n_phases} phases** found on **{_name}**. Left: jump across each link. Middle: "
                  "phases and walls; a ring marks a mixed point. Right: labels, if known (never seen by "
                  "phasewall)."),
            _figure(lambda: res.phase_map(_name)),
        ])

    def _snapshots():
        if _walls is None or res.ensemble is None:
            return mo.md("Snapshots need the walls and an ensemble (not a table).")
        _field = next(f for f in res.ensemble.fields if f.name == snapshot_field.value)
        return mo.vstack([
            snapshot_field,
            mo.md("The last snapshot of one run at the central point of each phase, on a common colour scale."),
            _figure(lambda: report.make_snapshots(_walls, res.ensemble, _field)),
        ])

    def _walls_tab():
        if _walls is None:
            return mo.md("The walls were not run.")
        curve = res.robustness.get(_name)
        parts = [mo.md(report.summary(_walls, curve))]
        if curve is not None:
            parts.append(_figure(lambda: res.robustness_plot(_name)))
        parts.append(mo.ui.tabs({
            "points (nodes)": mo.ui.table(_walls.nodes.round(4), selection=None),
            "links (edges)": mo.ui.table(_walls.edges.round(4), selection=None),
            "cuts tried": mo.ui.table(_walls.cuts.round(4), selection=None),
        }))
        return mo.vstack(parts)

    def _learned_tab():
        parts = []
        if res.training is not None:
            parts += [mo.md("Training: InfoNCE loss (dotted: its value at collapse) and effective rank of h."),
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
            if len(res.parameters) == 2:
                items.append(mo.download(lambda: png(res.phase_map(_name)), filename=f"{_name}_phase_map.png",
                                         label="phase map (PNG)"))
        return mo.vstack([mo.md(f"Files of **{_name}** and of the comparison:"), mo.hstack(items, wrap=True,
                                                                                           justify="start")])

    _tabs = {
        "Phase map": _phase_map,
        "Comparison": _comparison,
        "Snapshots": _snapshots,
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
