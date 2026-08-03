"""The public surface (PRD 6.3) — deliberately tiny.

Eight new public callables, plus SymPy re-exports that MathSlate does not wrap
at all. Everything else in this package exists to make ``plot()`` correct.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any, Mapping, Sequence

import sympy as sp

from ._text import safe_print
from .core import binding
from .core._failure import SYMBOLIC_FAILURE
from .core.analysis import Analysis, analyze_expression
from .core.dispatch import PlotPlan, build_plan
from .core.data import Dataset, FitResult, load_dataset
from .core.tables import DEFAULT_ROWS, Table, tabulate
from .core.sampling import DEFAULT_CONFIG, SamplingConfig
from .errors import NotYetImplementedError, UnsupportedInputError
from .render import axes as _axes
from .render import plotly_backend
from .render.options import RenderOptions
from .result import PlotResult, get_range_controls, set_range_controls
from .ui import interact
from .ui.adapters import detect_frontend, frontend_report
from .ui.interact import Slider, slider

__all__ = [
    "plot",
    "polar",
    "slider",
    "animate",
    "table",
    "analyze",
    "show_python",
    "dataset",
    "set_verbose",
    "get_verbose",
    "set_range_controls",
    "get_range_controls",
    "detect_frontend",
    "frontend_report",
]

_VERBOSE: bool = True


def set_verbose(value: bool) -> None:
    """Turn the one-line inference report on or off."""
    global _VERBOSE
    _VERBOSE = bool(value)


def get_verbose() -> bool:
    return _VERBOSE


def plot(
    obj: object,
    *ranges: object,
    polar: bool = False,
    kind: str | None = None,
    label: str | None = None,
    title: str | None = None,
    yscale: str | None = None,
    show_legend: bool | None = None,
    points: int | None = None,
    exclusions: Sequence[float] | bool | None = None,
    mesh: bool | int = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    verbose: bool | None = None,
    parameters: Mapping[sp.Symbol, float] | Sequence[sp.Symbol] = (),
) -> PlotResult:
    """Plot almost anything mathematical.

    The mapping from input to picture is the documented dispatch contract in
    :mod:`mathslate.core.dispatch`. One rule covers it: a ``list`` means
    "several things together", a ``tuple`` means "one vector-valued object".

    Examples
    --------
    >>> from mathslate import plot, x, sin
    >>> _ = plot(sin(x)/x)                      # a curve
    >>> _ = plot([sin(x), cos(x)])              # two curves overlaid
    >>> _ = plot((sin(x), cos(x)))              # one parametric curve
    >>> _ = plot(sin(x), (x, 0, 6.28))          # explicit range
    """
    options = _render_options(
        title=title, yscale=yscale, show_legend=show_legend, kind=kind,
        mesh=mesh, xlim=xlim, ylim=ylim, zlim=zlim,
    )
    config = _sampling_config(points, exclusions)

    plan = build_plan(
        obj,
        *ranges,
        polar=polar,
        kind=kind,
        label=label,
        config=config,
        parameters=parameters,
    )

    y_data = plotly_backend.figure_y_data(plan)
    hint = _axes.log_scale_hint(plan.exprs, y_data)
    if hint is not None:
        plan.notes.append(hint)

    warning = _axes.log_scale_warning(options.log_y, y_data)
    if warning is not None:
        plan.notes.append(warning)

    driving = _driving_slider(obj, plan)
    if driving is not None:
        driver, note = driving
        return _interactive(
            obj, ranges, driver, options, verbose, note, play=False,
            polar=polar, kind=kind, label=label, points=points,
            exclusions=exclusions, parameters=parameters,
        )

    figure = plotly_backend.figure_from_plan(plan, options)
    result = PlotResult(plan, figure, options)

    if _VERBOSE if verbose is None else verbose:
        safe_print(result.summary())
        for note in plan.notes:
            safe_print(f"  · {note}")
    return result


def _render_options(
    *,
    title: str | None = None,
    yscale: str | None = None,
    show_legend: bool | None = None,
    kind: str | None = None,
    mesh: bool | int = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
) -> RenderOptions:
    """Validate the drawing keywords and build the options both renderers read.

    One builder for `plot()` and `animate()`. They used to construct
    `RenderOptions` separately, and `animate()` listed the fields it knew about
    by hand — so `mesh`, `xlim`, `ylim` and `zlim` were silently dropped from
    every animation, and an invalid window was accepted there because the
    validation lived in `plot()` rather than here. An option that ignores the
    value it was given is a bug, not a convenience; so is one that only works on
    half the entry points.
    """
    if yscale not in (None, "linear", "log"):
        raise UnsupportedInputError(f"yscale must be 'linear' or 'log'; got {yscale!r}.")
    return RenderOptions(
        title=title,
        yscale=yscale,
        show_legend=show_legend,
        kind=kind,
        mesh=_mesh_option(mesh),
        xlim=_view_limit(xlim, "xlim"),
        ylim=_view_limit(ylim, "ylim"),
        zlim=_view_limit(zlim, "zlim"),
    )


def _mesh_option(mesh: bool | int | None) -> bool | int:
    """Validate ``mesh``: off, on at the default density, or a line count.

    ``bool(mesh)`` used to be all this did, which silently flattened a
    density (PRD §23.3's whole point) down to plain on/off — ``bool`` is a
    subclass of ``int`` in Python, so the bool case has to be checked first
    or ``True``/``False`` would themselves be read as a line count of 1/0.
    """
    if mesh is None:
        return True
    if isinstance(mesh, bool):
        return mesh
    if isinstance(mesh, int):
        if mesh < 2:
            raise UnsupportedInputError(
                f"mesh={mesh!r} draws too few lines to read as a grid; use "
                "at least 2, or True for the default density."
            )
        return mesh
    raise UnsupportedInputError(
        f"mesh must be True, False, or a positive number of lines; got {mesh!r}."
    )


def _view_limit(
    limit: tuple[float, float] | None, name: str
) -> tuple[float, float] | None:
    """Validate one view-window override into an ordered ``(lo, hi)`` pair."""
    if limit is None:
        return None
    try:
        lo, hi = (float(v) for v in limit)
    except (TypeError, ValueError) as error:
        raise UnsupportedInputError(
            f"{name} must be a pair of numbers, e.g. {name}=(-2, 5); got {limit!r}."
        ) from error
    if not hi > lo:
        raise UnsupportedInputError(
            f"{name} must be (low, high) with low < high; got ({lo}, {hi})."
        )
    return (lo, hi)


def _driving_slider(obj: object, plan: PlotPlan) -> tuple[Slider, str] | None:
    """The slider frames are drawn over, and what to say about the others.

    v0.5 animates one parameter. Any others stay frozen at their own values —
    Plotly's frames are a one-dimensional sequence, and a grid over several
    sliders would multiply out into thousands of pre-computed curves.

    The note is returned rather than appended here: this plan is about to be
    thrown away and rebuilt once per slider position, so a note left on it
    would never reach the reader.
    """
    if not plan.exprs:
        return None
    found = interact.sliders_in(*_original_exprs(obj))
    if not found:
        return None
    note = ""
    if len(found) > 1:
        note = (
            f"{len(found)} sliders here; the plot varies {found[0].label} and holds "
            + ", ".join(s.label for s in found[1:])
            + " at their current values."
        )
    return found[0], note


def _original_exprs(obj: object) -> tuple[sp.Expr, ...]:
    """The expressions as written — before parameters were substituted away."""
    candidates = obj if isinstance(obj, (list, tuple)) else [obj]
    out: list[sp.Expr] = []
    for item in candidates:
        converted = item._sympy_() if hasattr(item, "_sympy_") else item
        if isinstance(converted, sp.Basic):
            out.append(converted)
    return tuple(out)


def _interactive(
    obj: object,
    ranges: tuple[object, ...],
    driver: Slider,
    options: RenderOptions,
    verbose: bool | None,
    note: str = "",
    *,
    play: bool,
    **passed: Any,
) -> PlotResult:
    """Draw one plan per slider position and hand Plotly the frames.

    ``passed`` is the `plot()` arguments that must be identical in every
    frame — everything `build_plan` reads, plus the two sampling keywords it
    reads through a config. Only the axis being animated may differ between
    frames; anything else varying would make the slider change two things at
    once.
    """
    steps = driver.values()
    held = driver.value
    plans: list[PlotPlan] = []
    try:
        for value in steps:
            driver.value = value
            plans.append(
                build_plan(obj, *ranges, config=_config_for(passed), **_plan_args(passed))
            )
    finally:
        driver.value = held

    figure = plotly_backend.figure_with_frames(
        plans, steps, driver.label, options, play=play
    )
    current = plans[list(steps).index(driver.nearest(held))]
    if note:
        current.notes.append(note)
    result = PlotResult(
        current,
        figure,
        options,
        frames=(plans, steps, driver.label),
        frame_play=play,
    )

    if _VERBOSE if verbose is None else verbose:
        safe_print(f"{result.summary()} | {len(steps)} frames over {driver.label}")
        for note in current.notes:
            safe_print(f"  · {note}")
    return result


def _plan_args(passed: dict[str, Any]) -> dict[str, Any]:
    """Everything `build_plan` takes — the sampling keywords go via the config."""
    return {k: v for k, v in passed.items() if k not in _SAMPLING_KEYWORDS}


def _config_for(passed: dict[str, Any]) -> SamplingConfig:
    return _sampling_config(passed.get("points"), passed.get("exclusions"))


def _sampling_config(
    points: int | None, exclusions: Sequence[float] | bool | None
) -> SamplingConfig:
    """Turn the two sampling keywords into one config.

    ``exclusions`` is the escape hatch for PRD 5.3: automatic discontinuity
    detection is the package's differentiator, and a differentiator with no
    override is a black box the moment it is wrong.

    * ``None`` — detect automatically. The default, and what almost everyone wants.
    * a sequence — cut the line at these places *as well*.
    * ``False`` — do not go looking. Symbolic singularities still count, because
      those are SymPy's answer rather than a guess; only the numeric probe stops.
    """
    changes: dict[str, Any] = {}
    if points is not None:
        # An option that silently ignores the value it was given is a bug, not
        # a convenience: points=0 used to sample 200 anyway.
        if int(points) < 2:
            raise UnsupportedInputError(f"points must be at least 2; got {points!r}.")
        changes["initial_points"] = int(points)
        changes["max_points"] = max(int(points), DEFAULT_CONFIG.max_points)
    if exclusions is not None:
        if exclusions is False:
            changes["probe_jumps"] = False
        elif exclusions is True:
            raise UnsupportedInputError(
                "exclusions=True has no meaning: automatic detection is already on. "
                "Pass a list of places to cut, or False to stop probing."
            )
        else:
            changes["exclusions"] = _exclusion_points(exclusions)
    return replace(DEFAULT_CONFIG, **changes) if changes else DEFAULT_CONFIG


def _exclusion_points(given: Sequence[float]) -> tuple[float, ...]:
    """Coerce the caller's break points, refusing anything that is not a number.

    ``sympify``-able values are welcome — ``exclusions=[pi/2]`` is the natural
    thing to write next to ``plot(tan(x))`` — so this goes through SymPy rather
    than ``float()`` and keeps the message specific when it cannot.
    """
    if isinstance(given, str):
        # A string iterates into characters, so without this the complaint is
        # about the letter 'n' rather than about the string.
        raise UnsupportedInputError(
            f"exclusions must be a list of places to cut, not the string {given!r}. "
            "For one place, write exclusions=[0]."
        )
    try:
        iterated = list(given)
    except TypeError as error:
        raise UnsupportedInputError(
            f"exclusions must be a list of places to cut, not {type(given).__name__}."
        ) from error
    places: list[float] = []
    for value in iterated:
        try:
            places.append(float(sp.sympify(value)))
        except SYMBOLIC_FAILURE as error:
            raise UnsupportedInputError(
                f"exclusions must be numbers on the axis; {value!r} is not one."
            ) from error
    return tuple(places)


# --------------------------------------------------------------------------
# what `animate()` accepts, named by the functions that consume it
# --------------------------------------------------------------------------
#
# `animate()` takes `**kwargs` because it forwards `plot()`'s whole keyword
# surface, and the two used to be kept in step by hand: one list of names for
# the drawing options, another for the planning ones. That is not a style
# preference, it is the bug's shape — `mesh`, `xlim`, `ylim` and `zlim` were
# dropped from every animation until the review that introduced
# `_render_options`, because they were added to `plot()` and to nothing else.
# Asking the consuming functions what they take means the next keyword `plot()`
# gains reaches `animate()` without anyone remembering that it has to.

#: The drawing keywords — whatever `_render_options` validates.
_RENDER_KEYWORDS: frozenset[str] = frozenset(
    inspect.signature(_render_options).parameters
)
#: The two keywords that reach `build_plan` as a `SamplingConfig` rather than
#: as themselves, so they are excluded from the call and passed via `config=`.
_SAMPLING_KEYWORDS: frozenset[str] = frozenset(
    inspect.signature(_sampling_config).parameters
)
#: Everything that decides *what is sampled*, as opposed to how it is drawn.
#: ``obj``/``ranges`` are positional and ``config`` is built from the sampling
#: keywords above, so none of the three is a keyword a caller passes.
_PLAN_KEYWORDS: frozenset[str] = (
    frozenset(inspect.signature(build_plan).parameters)
    - {"obj", "ranges", "config"}
) | _SAMPLING_KEYWORDS


def _animate_keywords(
    kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split `animate()`'s keywords into the drawing half and the sampling half.

    ``kind`` is deliberately in both: it picks the trace mode *and* steers
    dispatch, which is exactly what `plot()` does with it too.

    Only the keywords actually given are forwarded, so each consumer's own
    default applies to the rest — a default repeated here is one more thing
    that can drift out of step with `plot()`.

    An unknown keyword is refused rather than ignored. `plot()` raises
    `TypeError` for one, while `animate(a*sin(x), tittle="Moving")` used to
    accept it in silence and draw an untitled plot — the same "option that
    quietly ignores the value it was given" this module refuses everywhere
    else.
    """
    unknown = set(kwargs) - _RENDER_KEYWORDS - _PLAN_KEYWORDS
    if unknown:
        accepted = sorted(_RENDER_KEYWORDS | _PLAN_KEYWORDS | {"over", "verbose"})
        raise UnsupportedInputError(
            f"animate() got {', '.join(sorted(unknown))}, which plot() does not "
            f"take either. animate() accepts: {', '.join(accepted)}."
        )
    return (
        {name: value for name, value in kwargs.items() if name in _RENDER_KEYWORDS},
        {name: value for name, value in kwargs.items() if name in _PLAN_KEYWORDS},
    )


def animate(
    obj: object,
    *ranges: object,
    over: Slider | None = None,
    **kwargs: Any,
) -> PlotResult:
    """The same picture as :func:`plot`, with a play button (PRD 5.7).

    ``over`` names which slider to run through when the expression has more
    than one. With none given, the first slider in the expression is used.

    Examples
    --------
    >>> from mathslate import animate, slider, sin, x
    >>> a = slider(1, 3, name="a")
    >>> _ = animate(a*sin(x), verbose=False)
    """
    found = interact.sliders_in(*_original_exprs(obj))
    if over is not None and all(candidate is not over for candidate in found):
        raise UnsupportedInputError(
            f"animate(..., over={over.name}) cannot vary that slider because its "
            "symbol does not appear in the expression. Choose one of the "
            f"expression's sliders: {', '.join(s.name for s in found) or 'none'}."
        )
    driver = over or (found[0] if found else None)
    if driver is None:
        raise UnsupportedInputError(
            "animate() needs something to vary. Bind a parameter with slider() "
            "first, e.g. a = slider(1, 3); animate(a*sin(x))."
        )
    verbose = kwargs.pop("verbose", None)
    render, passed = _animate_keywords(kwargs)
    options = _render_options(**render)
    note = ""
    if len(found) > 1:
        others = [s for s in found if s is not driver]
        note = (
            f"{len(found)} sliders here; the animation varies {driver.label} and holds "
            + ", ".join(s.label for s in others)
            + " at their current values."
        )
    return _interactive(
        obj,
        ranges,
        driver,
        options,
        verbose,
        note,
        play=True,
        **passed,
    )


def polar(
    radial: object,
    *ranges: object,
    **kwargs: Any,
) -> PlotResult:
    """``r = f(θ)``. Inference cannot distinguish this from ``y = f(x)``,
    so the PRD requires it to be asked for explicitly (PRD 5.1)."""
    kwargs.pop("polar", None)
    return plot(radial, *ranges, polar=True, **kwargs)


def show_python(result: PlotResult) -> str:
    """Module-level twin of :meth:`PlotResult.show_python`."""
    return result.show_python()


def analyze(
    obj: object,
    *ranges: object,
    parameters: Mapping[sp.Symbol, float] | Sequence[sp.Symbol] = (),
) -> Analysis:
    """Report the properties of an expression (PRD 5.6).

    Roots, extrema, inflection points, symmetry, periodicity, asymptotes,
    discontinuities and monotonic intervals — computed exactly wherever SymPy
    can, and by sampling where it cannot, with the approximate lines labelled
    as such.

    Never called automatically: property detection is slow enough and noisy
    enough that running it on every plot would spoil both.

    Examples
    --------
    >>> from mathslate import analyze, x
    >>> analyze(x**3 - 3*x).roots.values
    (-1.7320508075688772, 0.0, 1.7320508075688772)

    Notes
    -----
    This reports what the expression *is*. It does not explain how any of it
    was derived, and it never will — see PRD 2.2.
    """
    if isinstance(obj, PlotResult):
        return obj.analyze()

    expr = _analysable_expression(obj)
    specs = [binding.parse_range(r) for r in ranges]
    values = {s: float(v) for s, v in parameters.items()} if isinstance(
        parameters, Mapping
    ) else {}
    if values:
        expr = expr.subs({s: sp.Float(v) for s, v in values.items()})

    symbol, window = _analysis_axis(expr, specs, tuple(parameters))
    return analyze_expression(expr, symbol, window)


def _analysable_expression(obj: object) -> sp.Expr:
    """``analyze()`` describes one ``y = f(x)``, so anything else is refused."""
    if isinstance(obj, sp.Expr):
        return obj
    if isinstance(obj, str):
        try:
            return sp.sympify(obj)
        except SYMBOLIC_FAILURE as error:
            raise UnsupportedInputError(
                f"SymPy could not read {obj!r} as an expression."
            ) from error
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return sp.Float(obj) if isinstance(obj, float) else sp.Integer(obj)
    raise UnsupportedInputError(
        f"analyze() describes a single expression y = f(x); it does not know what "
        f"to do with {type(obj).__name__}. For raw data or a plain Python "
        "function there is nothing to solve symbolically."
    )


def _analysis_axis(
    expr: sp.Expr,
    specs: Sequence[binding.RangeSpec],
    parameters: Sequence[sp.Symbol],
) -> tuple[sp.Symbol, tuple[float, float]]:
    symbol = binding.choose_symbols([expr], 1, explicit=specs, parameters=parameters)[0]
    leftover = sorted(s.name for s in expr.free_symbols if s != symbol)
    if leftover:
        raise UnsupportedInputError(
            f"{symbol.name} is the variable, but {', '.join(leftover)} "
            f"{'is' if len(leftover) == 1 else 'are'} still free. Give "
            f"{'it' if len(leftover) == 1 else 'them'} a value: "
            f"parameters={{{', '.join(f'{n}: 1' for n in leftover)}}}."
        )
    for spec_symbol, lo, hi in specs:
        if spec_symbol == symbol:
            return symbol, (lo, hi)
    return symbol, binding.DEFAULT_SPAN


# --------------------------------------------------------------------------
# documented API scheduled for later milestones (PRD 7)
# --------------------------------------------------------------------------


def _later(name: str, milestone: str, what: str) -> NotYetImplementedError:
    return NotYetImplementedError(
        f"{name} arrives in MathSlate {milestone} ({what}). "
        "v0.1 is scoped to 'one graph, done properly': 2D curves, overlaid "
        "curves, parametric curves, adaptive sampling and show_python()."
    )


def table(
    obj: object,
    *ranges: object,
    rows: int = DEFAULT_ROWS,
    label: str | None = None,
    parameters: Mapping[sp.Symbol, float] | Sequence[sp.Symbol] = (),
) -> Table:
    """The same function read as numbers rather than as a picture.

    A graph shows shape; a table shows value. A learner checking a hand-worked
    answer needs the second, at exactly the moment the plot stops being enough.

    Examples
    --------
    >>> from mathslate import table, sin, x
    >>> print(table(sin(x), (x, 0, 1), rows=3).headers())
    ('x', 'sin(x)')
    """
    if isinstance(obj, PlotResult):
        return obj.table(rows=rows)

    items = obj if isinstance(obj, list) else [obj]
    exprs = [_analysable_expression(item) for item in items]
    specs = [binding.parse_range(r) for r in ranges]
    values = (
        {s: float(v) for s, v in parameters.items()}
        if isinstance(parameters, Mapping)
        else {}
    )
    if values:
        exprs = [e.subs({s: sp.Float(v) for s, v in values.items()}) for e in exprs]

    symbol = binding.choose_symbols(exprs, 1, explicit=specs, parameters=tuple(parameters))[0]
    leftover = sorted(
        s.name for e in exprs for s in e.free_symbols if s != symbol
    )
    if leftover:
        raise UnsupportedInputError(
            f"{symbol.name} is the column, but {', '.join(sorted(set(leftover)))} "
            "is still free. Give it a value with parameters=."
        )
    span = next(
        ((lo, hi) for s, lo, hi in specs if s == symbol),
        binding.default_range(exprs),
    )
    labels = [label] if label and len(exprs) == 1 else None
    return tabulate(exprs, symbol, span, rows=rows, labels=labels)


def dataset(
    source: object,
    columns: Sequence[str] | None = None,
    encoding: str | None = None,
) -> Dataset:
    """Named columns of numbers — the bridge from symbolic to data (PRD 7).

    Built from a mapping of columns, a CSV path, or arrays. A CSV is decoded
    as UTF-8 and then, failing that, as the encodings spreadsheets on a
    non-English system write; ``encoding=`` names one directly. The bridge itself
    is :meth:`Dataset.fit`: write the model as you would on paper and get the
    same expression back with its parameters filled in, still SymPy, so
    everything MathSlate already does works on the result.

    Examples
    --------
    >>> from mathslate import dataset, x
    >>> d = dataset({"x": [0, 1, 2, 3], "y": [1.0, 3.0, 5.0, 7.0]})
    >>> a, b = sp.symbols("a b", real=True)
    >>> round(float(d.fit(a*x + b).parameters[a]), 6)
    2.0
    """
    return load_dataset(source, columns, encoding)
