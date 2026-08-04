"""``plot()`` input → plot kind inference (PRD 5.1).

The dispatch table below is a **documented contract**, not an implementation
detail. There is exactly one rule to memorise:

* a ``list`` means "several things together"
* a ``tuple`` means "one vector-valued object"

============================  ============  ==========================
Input form                    Free symbols  Result
============================  ============  ==========================
``Expr``                      1             2D curve
``Expr``                      2             surface (v0.5)
``Expr``                      0             horizontal line + message
``Eq(lhs, rhs)``              2             implicit curve (v0.5)
inequality                    2             filled region (v1.5)
inequality                    1             shaded bands on the axis (v1.5)
``list[Expr]``                1 shared      curves overlaid
``tuple[Expr, Expr]``         1 shared      2D parametric curve
``tuple[Expr x 3]``           1 shared      3D space curve (v0.5)
``tuple[Expr x 3]``           2 shared      parametric surface (v0.5)
``callable``                  --            numeric sampling
array-like                    --            data series
``(xdata, ydata)``            --            scatter / line
============================  ============  ==========================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

import numpy as np
import sympy as sp

from ..errors import NotYetImplementedError, UnsupportedInputError
from . import binding
from ._budget import SymbolicTimeout, within_budget
from ._failure import EVALUATION_FAILURE, SYMBOLIC_FAILURE
from ._sets import set_to_intervals
from .sampling import (
    DEFAULT_CONFIG,
    SampleResult,
    SamplingConfig,
    sample_callable,
    sample_expression,
    sample_parametric,
)
from .data import Dataset
from .surfaces import (
    SurfaceSample,
    sample_parametric_surface,
    sample_region,
    sample_surface,
)

__all__ = ["PlotKind", "Series", "PlotPlan", "build_plan", "KINDS", "TWO_VARIABLE"]

PlotKind = str  # one of the values documented in KINDS
KINDS: tuple[str, ...] = (
    "curve",
    "curves",
    "constant",
    "parametric",
    "polar",
    "callable",
    "data",
    "xy",
    # v0.5 — two variables
    "surface",
    "contour",
    "implicit",
    "space",
    "psurface",
    # v1.5 — inequalities
    "region",
    "band",
    # v1.0 — data, statistics, linear algebra
    "hist",
    "box",
    "linalg",
)

#: Grid used to find and check the shaded intervals of a `band` plot.
_BAND_SAMPLES: Final[int] = 4001
#: Bisection steps refining one band edge. 60 takes a double past its last bit.
_BAND_BISECTIONS: Final[int] = 60

#: Kinds drawn over two variables rather than along one.
TWO_VARIABLE: tuple[str, ...] = (
    "surface", "contour", "implicit", "psurface", "region",
)

#: The ``kind=`` values :func:`build_plan` accepts *from a caller*. Every other
#: member of ``PlotKind`` — ``parametric``, ``polar``, ``space``, ``psurface``,
#: ``implicit``, ``band``, ``region``, ``data``, ``callable``, ``linalg`` — is
#: inferred from the object's shape and cannot be asked for by name (PRD 5.1:
#: the two cases inference cannot decide are ``polar()`` and ``kind="contour"``,
#: and those are the only ones spelled out).
#:
#: Exported rather than written inline below because anything that *rebuilds* a
#: plan has to feed its kind back through here — :meth:`PlotResult._replotted`
#: does, on every keystroke in the range-control sidebar — and reading a plan's
#: own ``kind`` back is only valid for these. The set drifting from the check
#: was exactly the defect: an inferred kind handed back raised ``unknown
#: kind=...``, which the sidebar swallowed as a mid-edit value, leaving every
#: control dead on six of the plot families.
REQUESTABLE_KINDS: frozenset[str] = frozenset(
    {"line", "scatter", "curve", "surface", "contour", "hist", "box"}
)


@dataclass(frozen=True)
class Series:
    """One drawn line and the object it came from."""

    name: str
    sample: SampleResult | SurfaceSample
    expr: sp.Expr | None = None
    components: tuple[sp.Expr, ...] = ()


@dataclass
class PlotPlan:
    """A fully resolved plot: everything render and codegen need, nothing more."""

    kind: PlotKind
    series: list[Series]
    symbol: sp.Symbol | None = None
    param_range: tuple[float, float] | None = None
    exprs: tuple[sp.Expr, ...] = ()
    polar: bool = False
    notes: list[str] = field(default_factory=list)
    config: SamplingConfig = DEFAULT_CONFIG
    #: Both axis symbols, for the two-variable kinds. ``symbol`` is axes[0].
    axes: tuple[sp.Symbol, sp.Symbol] | None = None
    #: The second axis's window, for the two-variable kinds.
    second_range: tuple[float, float] | None = None
    #: ``(x, y)`` axis titles when they come from data rather than a symbol.
    axis_labels: tuple[str, str] | None = None
    #: The 2x2 matrix behind a ``linalg`` plot.
    matrix: np.ndarray | None = None
    #: ``(eigenvalue, unit eigenvector)`` pairs, real ones only.
    eigen: list[tuple[float, np.ndarray]] = field(default_factory=list)
    #: The relation an ``implicit``, ``band`` or ``region`` plan came from.
    #:
    #: ``exprs`` cannot stand in for it on the first two: both store the
    #: *difference* ``lhs - rhs``, which is what gets sampled but not what
    #: identifies the picture. ``Eq(x**2 + y**2, 1)`` and ``x**2 + y**2 - 1``
    #: reduce to the same expression and draw a curve and a surface; ``sin(x) >
    #: 0`` and ``sin(x) < 0`` reduce to the same one and shade opposite
    #: intervals. Anything rebuilding the plan (:meth:`PlotResult._replotted`)
    #: therefore has to start from this rather than from ``exprs``.
    relation: sp.Rel | None = None
    #: ``(start, end)`` intervals to shade, for a ``band`` plot.
    bands: tuple[tuple[float, float], ...] = ()
    #: Whether `solveset` solved the inequality or the intervals were sampled.
    bands_exact: bool = False

    @property
    def two_variable(self) -> bool:
        return self.kind in TWO_VARIABLE

    @property
    def y_range(self) -> tuple[float, float] | None:
        return _widest(s.sample.y_range for s in self.series)

    @property
    def x_range(self) -> tuple[float, float] | None:
        """The clipped horizontal window, where x is not the parameter."""
        return _widest(s.sample.x_range for s in self.series)

    @property
    def total_points(self) -> int:
        return sum(s.sample.n_points for s in self.series)


def _grid(config: SamplingConfig) -> dict[str, int]:
    """``resolution=`` for a two-variable sampler, or nothing at all.

    Empty when the config names no budget, so each sampler keeps its own
    default — a surface's 60 and a region's 200 differ deliberately (a region
    is judged by its boundary, a surface by its interior) and passing one
    number for both would quietly halve or triple one of them.
    """
    return {} if config.grid_points is None else {"resolution": config.grid_points}


def _widest(
    ranges: Iterable[tuple[float, float] | None],
) -> tuple[float, float] | None:
    """The union of every window that was chosen, or ``None`` if none were."""
    present = [r for r in ranges if r is not None]
    if not present:
        return None
    return (min(r[0] for r in present), max(r[1] for r in present))


# --------------------------------------------------------------------------
# input classification helpers
# --------------------------------------------------------------------------


def _is_expr(obj: object) -> bool:
    return isinstance(obj, sp.Basic) and not isinstance(obj, sp.Set)


def _to_expr(obj: object) -> sp.Expr | None:
    if hasattr(obj, "_sympy_") and not isinstance(obj, sp.Basic):
        converted = obj._sympy_()
        return converted if isinstance(converted, sp.Expr) else None
    if isinstance(obj, sp.Expr):
        return obj
    if isinstance(obj, str):
        try:
            return sp.sympify(obj)
        except (sp.SympifyError, SyntaxError, TypeError):
            return None
    if isinstance(obj, bool):
        return None
    if isinstance(obj, int):
        return sp.Integer(obj)  # so the label reads "3", not "3.00000000000000"
    if isinstance(obj, float):
        return sp.Float(obj)
    return None


def _is_arraylike(obj: object) -> bool:
    if isinstance(obj, np.ndarray):
        return obj.ndim == 1
    if isinstance(obj, (list, tuple)) and obj:
        return all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in obj)
    return False


# --------------------------------------------------------------------------
# plan construction
# --------------------------------------------------------------------------


def build_plan(
    obj: object,
    *ranges: object,
    polar: bool = False,
    kind: str | None = None,
    label: str | None = None,
    config: SamplingConfig = DEFAULT_CONFIG,
    parameters: Mapping[sp.Symbol, float] | Sequence[sp.Symbol] = (),
) -> PlotPlan:
    """Classify ``obj`` and sample it. The single entry point of dispatch."""
    specs = [binding.parse_range(r) for r in ranges]
    parameter_values = _normalise_parameters(parameters)

    if kind is not None and kind not in REQUESTABLE_KINDS:
        raise UnsupportedInputError(f"unknown kind={kind!r}")

    if isinstance(obj, Dataset):
        return _plan_dataset(obj, kind, label)

    if isinstance(obj, sp.MatrixBase):
        return _plan_matrix(obj, label)

    if hasattr(obj, "_sympy_") and not isinstance(obj, sp.Basic):
        obj = obj._sympy_()  # a Slider, a FitResult — anything that is really an Expr

    if kind in {"hist", "box"} and _is_arraylike(obj):
        return _plan_distribution(
            {label or "values": np.asarray(obj, dtype=np.float64)}, kind, label
        )

    if isinstance(obj, sp.Eq):
        return _plan_implicit(obj, specs, parameter_values, config, label)

    if isinstance(obj, (sp.Rel, sp.logic.boolalg.BooleanFunction)):
        return _plan_relation(obj, specs, parameter_values, config, label)

    if isinstance(obj, sp.logic.boolalg.BooleanAtom):
        # SymPy settles a comparison it can decide from assumptions alone, and
        # hands back a bare True or False with the expressions gone — so there is
        # nothing left to sample even though the inequality was well formed.
        verdict = (
            "holds for every real number" if bool(obj) else "holds nowhere at all"
        )
        raise UnsupportedInputError(
            f"SymPy decided this inequality before plot() saw it: it {verdict}, so "
            f"it arrived as the bare value {obj} with the expressions already gone. "
            "There is nothing to sample. If a picture is still what you want, plot "
            "the two sides and compare them: plot([lhs, rhs])."
        )

    # (xdata, ydata)
    if isinstance(obj, tuple) and len(obj) == 2 and all(_is_arraylike(p) for p in obj):
        return _plan_xy(obj[0], obj[1], label)

    # tuple of expressions → one vector-valued object
    if isinstance(obj, tuple):
        parts = [_to_expr(p) for p in obj]
        if all(p is not None for p in parts):
            exprs = tuple(p for p in parts if p is not None)
            if len(exprs) == 2:
                return _plan_parametric(exprs, specs, parameter_values, config, label)
            if len(exprs) == 3:
                return _plan_triple(exprs, specs, parameter_values, config, label)
            raise UnsupportedInputError(
                f"a tuple of {len(exprs)} expressions has no meaning in the dispatch "
                "contract; use a list to overlay several curves."
            )

    # bare array-like → data series
    if _is_arraylike(obj):
        values = np.asarray(obj, dtype=np.float64)
        return _plan_xy(np.arange(values.size, dtype=np.float64), values, label)

    # list → several things overlaid
    if isinstance(obj, list):
        if not obj:
            raise UnsupportedInputError(
                "an empty list has nothing to draw; a list holds the curves to "
                "overlay, e.g. plot([sin(x), cos(x)]), or the numbers of a data series."
            )
        parts = [_to_expr(p) for p in obj]
        if all(p is not None for p in parts):
            exprs = tuple(p for p in parts if p is not None)
            return _plan_curves(exprs, specs, parameter_values, polar, config, label)
        raise UnsupportedInputError(
            "a list must contain expressions (to overlay curves) or numbers (data)."
        )

    # single expression
    expr = _to_expr(obj)
    if expr is not None:
        if _wants_surface(expr, specs, parameter_values, polar, kind):
            return _plan_surface(expr, specs, parameter_values, kind, config, label)
        return _plan_curves((expr,), specs, parameter_values, polar, config, label)

    if isinstance(obj, str):
        # A string *is* a documented input — it just did not parse. Saying
        # "does not know what to do with str" would send the reader looking for
        # the wrong problem.
        raise UnsupportedInputError(
            f"SymPy could not read {obj!r} as an expression. Check the syntax, or "
            "build it from symbols instead, e.g. plot(sin(x)/x)."
        )

    if callable(obj):
        return _plan_callable(obj, specs, config, label)

    raise UnsupportedInputError(
        f"plot() does not know what to do with {type(obj).__name__}. "
        "See the dispatch contract in the documentation."
    )


def _wants_surface(
    expr: sp.Expr,
    specs: Sequence[binding.RangeSpec],
    parameters: Mapping[sp.Symbol, float | None],
    polar: bool,
    kind: str | None,
) -> bool:
    """PRD 5.1: one expression with two free symbols is a surface.

    Bound parameters and a single explicit range both take a symbol out of the
    running first, so `plot(a*sin(x))` with a slider stays a curve and
    `plot(x*y, (x, -5, 5), parameters={y: 1})` stays one too.

    The sliders have to be counted here and not only in ``_apply_parameters``:
    they are bound in a registry rather than passed in, so an expression that
    reads as two free symbols at this point may have only one once the slider's
    value is substituted. Missing that turned `plot(a*sin(x))` — the headline
    example of PRD 5.7 — into a surface over `x` and `a`.
    """
    if polar:
        return False
    if kind in {"surface", "contour"}:
        return True
    # By name, not by symbol: a bound `a` fixes this expression's `a` however
    # either was spelled, and a set difference over symbol objects misses that
    # whenever the assumptions differ (see `binding._BOUND`).
    bound = binding.bound_names()
    fixed = set(parameters) | {s for s in expr.free_symbols if s.name in bound}
    free = expr.free_symbols - fixed - {s[0] for s in specs}
    # Axis candidates are the still-free symbols plus every symbol given a
    # range. Two of them is a surface, however they were arrived at: two ranges
    # says so as plainly as two free symbols does.
    return len(free) + len(specs) >= 2


def _normalise_parameters(
    parameters: Mapping[sp.Symbol, float] | Sequence[sp.Symbol],
) -> dict[sp.Symbol, float | None]:
    """Accept ``{a: 2}`` or ``[a]``; a value of ``None`` means "no value yet"."""
    if isinstance(parameters, Mapping):
        return {symbol: float(value) for symbol, value in parameters.items()}
    return {symbol: None for symbol in parameters}


def _apply_parameters(
    exprs: Sequence[sp.Expr],
    values: Mapping[sp.Symbol, float | None],
    specs: Sequence[binding.RangeSpec] = (),
    *,
    axes_needed: int = 1,
) -> tuple[sp.Expr, ...]:
    """Freeze parameters at their current value (PRD 5.2 rule 2).

    Symbols a UI control has bound are folded in automatically — that is what
    makes rule 2 real rather than something the user has to restate. An
    explicit ``parameters=`` entry outranks the control, and an explicit range
    outranks both: asking for a symbol to be the axis means it is the axis.
    """
    axes = {spec[0] for spec in specs}
    free = {s for expr in exprs for s in expr.free_symbols}
    # Keyed by the *expression's* symbols, so `expr.subs` below matches them.
    named_axes = {s.name for s in axes}
    named_values = {s.name for s in values}
    automatic = {
        symbol: value
        for symbol, value in binding.bindings_for(free).items()
        if symbol.name not in named_axes and symbol.name not in named_values
    }
    _reject_axis_eating_sliders(automatic, values, free, axes, axes_needed)
    values = {**automatic, **values}

    both = sorted(s.name for s in values if s in axes)
    if both:
        raise UnsupportedInputError(
            f"{', '.join(both)} is given both an explicit range and a value in "
            "parameters=. A symbol is either an axis or a parameter, not both."
        )

    bindings = {s: sp.Float(v) for s, v in values.items() if v is not None}
    frozen = tuple(expr.subs(bindings) for expr in exprs) if bindings else tuple(exprs)

    # A parameter with no value is excluded from the axis candidates but is
    # still in the expression, so there is nothing to evaluate. Saying so here
    # beats the "produced no finite values" that used to come out of the
    # sampler several steps later.
    unset = sorted(
        s.name
        for s, value in values.items()
        if value is None and any(s in expr.free_symbols for expr in frozen)
    )
    if unset:
        names = ", ".join(unset)
        raise UnsupportedInputError(
            f"{names} is listed in parameters= but has no value, so the curve is "
            f"not a number anywhere. Give one, e.g. parameters={{{unset[0]}: 1}}."
        )
    return frozen


def _reject_axis_eating_sliders(
    automatic: Mapping[sp.Symbol, float | None],
    explicit: Mapping[sp.Symbol, float | None],
    free: set[sp.Symbol],
    axes: set[sp.Symbol],
    axes_needed: int,
) -> None:
    """Refuse to let a slider silently consume the last axis.

    A slider binds its symbol for the whole session, so ``slider(name="t")``
    turns every later ``t`` into a parameter — including the ``t`` of
    ``plot((cos(t), sin(t)))``, which then froze to a single point and drew it
    without a word. Rule 2 removes symbols from the axis candidates; it was
    never meant to empty the pool.

    Only automatic (control-bound) symbols are checked. An explicit
    ``parameters=`` entry is the user saying so in this very call, and is left
    to the "no axis left" errors further down, which can be more specific.
    """
    if not automatic:
        return
    remaining = free - set(automatic) - set(explicit) - axes
    if len(remaining) + len(axes & free) >= axes_needed:
        return

    names = ", ".join(sorted(s.name for s in automatic))
    first = sorted(s.name for s in automatic)[0]
    raise UnsupportedInputError(
        f"{names} is bound to a slider, but it is also the only symbol left to "
        f"put on an axis here — freezing it would draw a single point rather "
        f"than a curve.\n"
        f"  · to plot over it:      plot(expr, ({first}, -10, 10))\n"
        f"  · to free the slider:   from mathslate.ui import release_all; release_all()\n"
        f"  · to keep both:         give the slider another name, "
        f"slider(..., name='a')"
    )


def _resolve_range(
    exprs: Sequence[sp.Expr],
    specs: Sequence[binding.RangeSpec],
    parameters: Sequence[sp.Symbol],
    *,
    periodic_default: bool,
) -> tuple[sp.Symbol, tuple[float, float]]:
    symbol = binding.choose_symbols(exprs, 1, explicit=specs, parameters=parameters)[0]
    _reject_leftover_symbols(exprs, symbol)
    for spec_symbol, lo, hi in specs:
        if spec_symbol == symbol:
            return symbol, (lo, hi)
    return symbol, binding.default_range(exprs, periodic_default=periodic_default)


def _reject_leftover_symbols(
    exprs: Sequence[sp.Expr], *symbols: sp.Symbol
) -> None:
    """Nothing but the axes may still be free once the axes are chosen.

    Naming an axis does not give the *other* symbols values, and a curve with
    an unbound symbol in it is not a number anywhere. This used to fall through
    to the sampler and come back as "produced no finite values", which reads as
    a statement about the mathematics rather than about the call. A surface has
    two axes rather than one, and the same holds for a third symbol.
    """
    axes = set(symbols)
    leftover = sorted(
        s.name for s in binding.free_symbols_of(exprs) if s not in axes
    )
    if not leftover:
        return
    names = ", ".join(leftover)
    freeze = ", ".join(f"{name}: 1" for name in leftover)
    axis_text = " and ".join(s.name for s in symbols)
    raise UnsupportedInputError(
        f"{axis_text} {'is the axis' if len(symbols) == 1 else 'are the axes'}, but {names} "
        f"{'is' if len(leftover) == 1 else 'are'} still free, so the curve has no "
        f"value anywhere. Give {'it' if len(leftover) == 1 else 'them'} a value: "
        f"parameters={{{freeze}}}."
    )


def _plan_curves(
    exprs: tuple[sp.Expr, ...],
    specs: Sequence[binding.RangeSpec],
    parameters: Mapping[sp.Symbol, float | None],
    polar: bool,
    config: SamplingConfig,
    label: str | None,
) -> PlotPlan:
    exprs = _apply_parameters(exprs, parameters, specs)
    free = binding.free_symbols_of(exprs) - set(parameters) - {s[0] for s in specs}

    symbol, span = _resolve_range(exprs, specs, tuple(parameters), periodic_default=polar)
    notes: list[str] = []
    series: list[Series] = []

    for expr in exprs:
        if polar:
            radial = expr
            components = (radial * sp.cos(symbol), radial * sp.sin(symbol))
            sample = sample_parametric(components, symbol, span, config)
            series.append(
                Series(name=_label_of(expr, label), sample=sample, expr=expr, components=components)
            )
            continue
        if not expr.free_symbols:
            sample = _constant_sample(expr, span)
            notes.append(
                f"{expr} has no free symbols, so it draws as the horizontal line y = {sp.N(expr, 6)}."
            )
        else:
            sample = sample_expression(expr, symbol, span, config)
        series.append(Series(name=_label_of(expr, label), sample=sample, expr=expr))
        notes.extend(sample.notes)

    kind: PlotKind
    if polar:
        kind = "polar"
    elif len(series) > 1:
        kind = "curves"
    elif not exprs[0].free_symbols:
        kind = "constant"
    else:
        kind = "curve"

    return PlotPlan(
        kind=kind,
        series=series,
        symbol=symbol,
        param_range=span,
        exprs=exprs,
        polar=polar,
        notes=_dedupe(notes),
        config=config,
    )


# --------------------------------------------------------------------------
# two variables: surfaces, contours, implicit curves, space curves (PRD 5.1)
# --------------------------------------------------------------------------

#: A surface's default window. Narrower than a curve's [-10, 10], because a
#: 60x60 grid over that range resolves far less than 200 points along a line.
SURFACE_SPAN: tuple[float, float] = (-5.0, 5.0)


def _two_ranges(
    exprs: Sequence[sp.Expr],
    specs: Sequence[binding.RangeSpec],
    parameters: Mapping[sp.Symbol, float | None],
) -> tuple[tuple[sp.Symbol, sp.Symbol], tuple[float, float], tuple[float, float]]:
    """Pick both axes and their windows, by the rules of PRD 5.2."""
    chosen = binding.choose_symbols(exprs, 2, explicit=specs, parameters=tuple(parameters))
    _reject_leftover_symbols(exprs, *chosen)
    given = {spec[0]: (spec[1], spec[2]) for spec in specs}
    first, second = chosen[0], chosen[1]
    return (
        (first, second),
        given.get(first, SURFACE_SPAN),
        given.get(second, SURFACE_SPAN),
    )


def _plan_surface(
    expr: sp.Expr,
    specs: Sequence[binding.RangeSpec],
    parameters: Mapping[sp.Symbol, float | None],
    kind: str | None,
    config: SamplingConfig,
    label: str | None,
) -> PlotPlan:
    """``Expr`` with 2 free symbols — a surface, or a contour on request."""
    (frozen,) = _apply_parameters([expr], parameters, specs, axes_needed=2)
    symbols, xrange, yrange = _two_ranges([frozen], specs, parameters)
    sample = sample_surface(frozen, symbols, xrange, yrange, **_grid(config))
    notes = list(sample.notes)
    if kind != "contour":
        notes.append("view it flat with kind='contour'.")
    return PlotPlan(
        kind="contour" if kind == "contour" else "surface",
        series=[Series(name=label or sp.sstr(frozen), sample=sample, expr=frozen)],
        symbol=symbols[0],
        param_range=xrange,
        exprs=(frozen,),
        notes=_dedupe(notes),
        config=config,
        axes=symbols,
        second_range=yrange,
    )


def _plan_relation(
    relation: sp.Basic,
    specs: Sequence[binding.RangeSpec],
    parameters: Mapping[sp.Symbol, float | None],
    config: SamplingConfig,
    label: str | None,
) -> PlotPlan:
    """An inequality: the *set where it holds*, not a curve through it.

    ``Eq`` asks where two expressions are equal, which is a curve. ``<`` asks
    where one exceeds the other, which is an area — so the two cannot share a
    planner even though they arrive looking alike.

    Which picture depends on how many symbols are free, exactly as it does for
    an ordinary expression (PRD 5.1): two symbols make a filled region in the
    plane, one makes shaded intervals on the axis. The second is the Korean
    curriculum's *부등식의 해* and the first its *부등식의 영역*; both are the
    same question asked in different dimensions.
    """
    free = binding.free_symbols_of([relation]) - set(parameters)
    free |= {spec[0] for spec in specs}
    if len(free) >= 2:
        return _plan_region(relation, specs, parameters, config, label)
    return _plan_band(relation, specs, parameters, config, label)


def _plan_region(
    relation: sp.Basic,
    specs: Sequence[binding.RangeSpec],
    parameters: Mapping[sp.Symbol, float | None],
    config: SamplingConfig,
    label: str | None,
) -> PlotPlan:
    """Shade the part of the plane where ``relation`` is true."""
    (frozen,) = _apply_parameters([relation], parameters, specs, axes_needed=2)
    symbols, xrange, yrange = _two_ranges([frozen], specs, parameters)
    sample = sample_region(frozen, symbols, xrange, yrange, **_grid(config))
    return PlotPlan(
        kind="region",
        series=[Series(name=label or sp.sstr(relation), sample=sample, expr=frozen)],
        symbol=symbols[0],
        param_range=xrange,
        exprs=(frozen,),
        relation=frozen if isinstance(frozen, sp.Rel) else None,
        notes=_dedupe(list(sample.notes)),
        config=config,
        axes=symbols,
        second_range=yrange,
    )


def _plan_band(
    relation: sp.Basic,
    specs: Sequence[binding.RangeSpec],
    parameters: Mapping[sp.Symbol, float | None],
    config: SamplingConfig,
    label: str | None,
) -> PlotPlan:
    """One free symbol: draw ``lhs - rhs`` and shade where the relation holds.

    The intervals are asked of ``solveset`` first, so ``sin(x) > 0`` on a window
    reports its solution exactly rather than to grid resolution. When solveset
    restates the question instead of answering it — a ``ConditionSet`` — the
    boolean is sampled and contiguous runs are taken, which is the same
    exact-then-approximate ladder the rest of the package climbs.

    The curve is drawn as well as the bands. A shaded strip on a bare axis says
    *where* the answer is; the curve crossing zero at the same places says *why*,
    and for the audience of PRD 2.1 the second is the point of the exercise.
    """
    if not isinstance(relation, sp.Rel):
        raise UnsupportedInputError(
            f"plot() can shade a single inequality over one variable, but "
            f"{sp.sstr(relation)} combines several. Give it two free symbols to "
            "draw as a region, or plot the parts one at a time."
        )
    difference = relation.lhs - relation.rhs
    (frozen,) = _apply_parameters([difference], parameters, specs, axes_needed=1)
    symbol, span = _resolve_range(
        [frozen], specs, tuple(parameters), periodic_default=False
    )
    sample = sample_expression(frozen, symbol, span, config)
    frozen_relation = relation.func(frozen, sp.Integer(0))
    bands, exact = _solution_bands(frozen_relation, symbol, span)
    notes = list(sample.notes)
    if not bands:
        notes.append(
            f"{sp.sstr(relation)} holds nowhere on this window — the curve is drawn "
            "with nothing shaded."
        )
    elif not exact:
        notes.append(
            "solveset could not solve the inequality, so the shaded intervals were "
            "found by sampling and their edges are approximate."
        )
    return PlotPlan(
        kind="band",
        series=[Series(name=label or sp.sstr(relation), sample=sample, expr=frozen)],
        symbol=symbol,
        param_range=span,
        exprs=(frozen,),
        relation=frozen_relation,
        notes=_dedupe(notes),
        config=config,
        bands=tuple(bands),
        bands_exact=exact,
    )


def _solution_bands(
    relation: sp.Rel, symbol: sp.Symbol, span: tuple[float, float]
) -> tuple[list[tuple[float, float]], bool]:
    """Where the relation holds, exactly if solveset can and by sampling if not.

    ``solveset`` is asked first and then **checked**, because for a periodic
    inequality it answers with the principal period only and does not say so:
    ``solveset(sin(x) > 0, x, Interval(-6, 6))`` is ``Interval.open(0, pi)``,
    silently missing ``(-6, -pi)``. Reporting that as exact would be the one
    failure mode this package cannot have — a confident wrong answer — and the
    range argument does not prevent it.

    So the closed form only wins if sampling cannot find a solution outside it.
    That is the same trust-but-verify shape as the ``ConditionSet`` rule in
    `analysis.py`, and it holds regardless of *why* solveset came back short.
    """
    lo, hi = float(span[0]), float(span[1])
    sampled = _sampled_bands(relation, symbol, lo, hi)
    window = sp.Interval(sp.Float(lo), sp.Float(hi))
    try:
        solution = within_budget(sp.solveset, relation, symbol, window)
    except (SymbolicTimeout, *SYMBOLIC_FAILURE):
        return sampled, False
    if solution.has(sp.ConditionSet):
        return sampled, False
    try:
        claimed = set_to_intervals(solution, lo, hi)
    except SYMBOLIC_FAILURE:
        return sampled, False
    if _covers(claimed, sampled, lo, hi):
        return claimed, True
    return sampled, False


def _covers(
    claimed: Sequence[tuple[float, float]],
    sampled: Sequence[tuple[float, float]],
    lo: float,
    hi: float,
) -> bool:
    """True when every sampled solution sits inside a claimed interval.

    Compared at the *midpoints* of the sampled bands, which sidesteps the
    boundary disagreement that a grid and a closed form will always have. Bands
    narrower than a couple of grid steps are ignored: those are sampling noise
    around a tangency, and letting them veto a correct closed form would give up
    the exact answer for `x**2 >= 0`.
    """
    noise = 3.0 * (hi - lo) / (_BAND_SAMPLES - 1)
    for start, end in sampled:
        if end - start < noise:
            continue
        middle = 0.5 * (start + end)
        if not any(a <= middle <= b for a, b in claimed):
            return False
    return True


def _sampled_bands(
    relation: sp.Rel, symbol: sp.Symbol, lo: float, hi: float
) -> list[tuple[float, float]]:
    """Contiguous runs where a sampled boolean is true, with refined edges."""
    grid = np.linspace(lo, hi, _BAND_SAMPLES, dtype=np.float64)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            test = sp.lambdify(symbol, relation, modules="numpy")
            truth = np.asarray(
                np.broadcast_to(np.asarray(test(grid)), grid.shape), dtype=bool
            )
        except (*SYMBOLIC_FAILURE, *EVALUATION_FAILURE):
            return []

        bands: list[tuple[float, float]] = []
        edges = np.flatnonzero(np.diff(truth.astype(np.int8)))
        start = lo if truth[0] else None
        for index in edges:
            # The flip is somewhere in (grid[index], grid[index+1]); bisect for it
            # so a shaded edge lands on the root rather than on a grid line.
            left, right = float(grid[index]), float(grid[index + 1])
            if truth[index]:
                place = _flip_point(test, left, right)
                bands.append((start if start is not None else lo, place))
                start = None
            else:
                start = _flip_point(test, right, left)
        if start is not None or truth[-1]:
            bands.append((start if start is not None else lo, hi))
    return bands


def _flip_point(
    test: Callable[[float], object], holds_at: float, fails_at: float
) -> float:
    """Bisect for the place where the relation stops holding.

    ``holds_at`` is known true and ``fails_at`` known false, so no re-evaluation
    of the endpoints is needed — only the midpoint each round.
    """
    for _ in range(_BAND_BISECTIONS):
        middle = 0.5 * (holds_at + fails_at)
        try:
            holds = bool(test(middle))
        except (*SYMBOLIC_FAILURE, *EVALUATION_FAILURE):
            return middle
        if holds:
            holds_at = middle
        else:
            fails_at = middle
    return 0.5 * (holds_at + fails_at)


def _plan_implicit(
    relation: sp.Rel,
    specs: Sequence[binding.RangeSpec],
    parameters: Mapping[sp.Symbol, float | None],
    config: SamplingConfig,
    label: str | None,
) -> PlotPlan:
    """``Eq(lhs, rhs)`` — the curve where ``lhs - rhs`` is zero.

    Drawn as the single zero level of a contour, which is the only honest way
    to render an implicit curve on a grid: there is no parametrisation to
    sample and no guarantee the solution set is even a curve.
    """
    assert isinstance(relation, sp.Eq)
    difference = sp.simplify(relation.lhs - relation.rhs)
    (frozen,) = _apply_parameters([difference], parameters, specs, axes_needed=2)
    symbols, xrange, yrange = _two_ranges([frozen], specs, parameters)
    sample = sample_surface(frozen, symbols, xrange, yrange, **_grid(config))
    return PlotPlan(
        kind="implicit",
        series=[
            Series(name=label or sp.sstr(relation), sample=sample, expr=frozen)
        ],
        symbol=symbols[0],
        param_range=xrange,
        exprs=(frozen,),
        # Not the caller's `relation`: any bound parameter has been substituted
        # into `frozen`, and resampling must not put the free symbol back.
        relation=sp.Eq(frozen, sp.Integer(0)),
        notes=_dedupe(list(sample.notes)),
        config=config,
        axes=symbols,
        second_range=yrange,
    )


def _plan_triple(
    exprs: tuple[sp.Expr, ...],
    specs: Sequence[binding.RangeSpec],
    parameters: Mapping[sp.Symbol, float | None],
    config: SamplingConfig,
    label: str | None,
) -> PlotPlan:
    """``tuple[Expr x 3]`` — one shared symbol is a space curve, two a surface."""
    frozen = _apply_parameters(exprs, parameters, specs)
    free = binding.free_symbols_of(frozen) - set(parameters)
    wanted = 2 if len(free - {s[0] for s in specs}) >= 2 or len(specs) >= 2 else 1

    if wanted == 1:
        symbol, span = _resolve_range(frozen, specs, tuple(parameters), periodic_default=True)
        sample = sample_parametric(
            (frozen[0], frozen[1]), symbol, span, config, third=frozen[2]
        )
        name = label or f"({', '.join(sp.sstr(e) for e in frozen)})"
        return PlotPlan(
            kind="space",
            series=[Series(name=name, sample=sample, components=tuple(frozen))],
            symbol=symbol,
            param_range=span,
            exprs=tuple(frozen),
            notes=_dedupe(list(sample.notes)),
            config=config,
        )

    symbols, urange, vrange = _two_ranges(frozen, specs, parameters)
    surface = sample_parametric_surface(
        (frozen[0], frozen[1], frozen[2]), symbols, urange, vrange, **_grid(config)
    )
    name = label or f"({', '.join(sp.sstr(e) for e in frozen)})"
    return PlotPlan(
        kind="psurface",
        series=[Series(name=name, sample=surface, components=tuple(frozen))],
        symbol=symbols[0],
        param_range=urange,
        exprs=tuple(frozen),
        notes=_dedupe(list(surface.notes)),
        config=config,
        axes=symbols,
        second_range=vrange,
    )


def _constant_sample(expr: sp.Expr, span: tuple[float, float]) -> SampleResult:
    try:
        value = float(sp.N(expr))
    except TypeError as error:
        # A constant with no free symbols that is not a real number — `I`,
        # `2 + 3*I`, `zoo` — has no height to draw on a real y-axis. `float()`
        # says "Cannot convert complex to float" from two frames down; say what
        # it means for a plot instead.
        raise UnsupportedInputError(
            f"{sp.sstr(expr)} is not a real number, so it has no value to plot "
            "on a real axis. Plot its real or imaginary part, e.g. re(...) or "
            "im(...)."
        ) from error
    x = np.linspace(span[0], span[1], 2, dtype=np.float64)
    return SampleResult(x=x, y=np.full(2, value, dtype=np.float64), t=x)


def _plan_parametric(
    exprs: tuple[sp.Expr, ...],
    specs: Sequence[binding.RangeSpec],
    parameters: Mapping[sp.Symbol, float | None],
    config: SamplingConfig,
    label: str | None,
) -> PlotPlan:
    exprs = _apply_parameters(exprs, parameters, specs)
    symbol, span = _resolve_range(exprs, specs, tuple(parameters), periodic_default=True)
    components = (exprs[0], exprs[1])
    sample = sample_parametric(components, symbol, span, config)
    name = label or f"({sp.sstr(exprs[0])}, {sp.sstr(exprs[1])})"
    return PlotPlan(
        kind="parametric",
        series=[Series(name=name, sample=sample, components=components)],
        symbol=symbol,
        param_range=span,
        exprs=components,
        notes=_dedupe(list(sample.notes)),
        config=config,
    )


def _plan_callable(
    function: Callable[..., Any],
    specs: Sequence[binding.RangeSpec],
    config: SamplingConfig,
    label: str | None,
) -> PlotPlan:
    span = (specs[0][1], specs[0][2]) if specs else binding.DEFAULT_SPAN
    sample = sample_callable(function, span, config)
    name = label or _callable_name(function)
    return PlotPlan(
        kind="callable",
        series=[Series(name=name, sample=sample)],
        param_range=span,
        notes=_dedupe(list(sample.notes)),
        config=config,
    )


def _plan_xy(xdata: object, ydata: object, label: str | None) -> PlotPlan:
    x = np.asarray(xdata, dtype=np.float64)
    y = np.asarray(ydata, dtype=np.float64)
    if x.shape != y.shape:
        raise UnsupportedInputError(
            f"x and y must have the same length; got {x.size} and {y.size}."
        )
    sample = SampleResult(x=x, y=y, domain_intervals=((float(x.min()), float(x.max())),))
    return PlotPlan(
        kind="data",
        series=[Series(name=label or "data", sample=sample)],
        param_range=(float(x.min()), float(x.max())),
    )


# --------------------------------------------------------------------------
# v1.0: data, distributions, linear algebra
# --------------------------------------------------------------------------


def _plan_dataset(data: Dataset, kind: str | None, label: str | None) -> PlotPlan:
    """A ``Dataset`` reads like a spreadsheet: first column across, rest up.

    ``kind="hist"`` and ``kind="box"`` ask about the shape of each column on
    its own instead, which is a different question and gets a different plan.
    """
    if kind in {"hist", "box"}:
        return _plan_distribution(dict(data.columns), kind, label)

    names = data.names
    if len(names) == 1:
        values = data[names[0]]
        plan = _plan_xy(np.arange(values.size, dtype=np.float64), values, names[0])
        plan.axis_labels = ("index", names[0])
        return plan

    inputs = data[names[0]]
    series = [
        Series(
            name=name,
            sample=SampleResult(x=inputs, y=data[name]),
        )
        for name in names[1:]
    ]
    plan = PlotPlan(
        kind="data",
        series=series,
        param_range=(float(np.nanmin(inputs)), float(np.nanmax(inputs))),
    )
    plan.axis_labels = (names[0], names[1] if len(names) == 2 else "value")
    return plan


def _plan_distribution(
    columns: Mapping[str, np.ndarray], kind: str, label: str | None
) -> PlotPlan:
    """A histogram or box plot per column — the shape of the numbers."""
    series = [
        Series(name=name, sample=SampleResult(x=np.asarray(values, dtype=np.float64),
                                              y=np.asarray(values, dtype=np.float64)))
        for name, values in columns.items()
    ]
    plan = PlotPlan(kind=kind, series=series)
    plan.axis_labels = ("value", "count") if kind == "hist" else ("", "value")
    return plan


def _plan_matrix(matrix: sp.MatrixBase, label: str | None) -> PlotPlan:
    """A 2x2 matrix drawn as what it *does*: the unit square, transformed.

    A matrix of numbers on a page is not a picture of anything. What makes it
    mean something is seeing where it sends things — so the plan carries the
    image of the unit square, the image of each basis vector, and the
    eigenvectors, which are exactly the directions the transformation leaves
    alone.
    """
    if matrix.shape != (2, 2):
        raise UnsupportedInputError(
            f"plot() draws a 2x2 matrix as a transformation of the plane; "
            f"this one is {matrix.shape[0]}x{matrix.shape[1]}. Its columns are "
            "yours to plot directly if you want them."
        )
    numeric = np.array(
        [[float(matrix[r, c]) for c in range(2)] for r in range(2)], dtype=np.float64
    )
    square = np.array([[0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]])
    image = numeric @ square

    series = [
        Series(name="unit square", sample=SampleResult(x=square[0], y=square[1])),
        Series(name=f"{label or 'M'}·(unit square)",
               sample=SampleResult(x=image[0], y=image[1])),
    ]
    plan = PlotPlan(kind="linalg", series=series)
    plan.matrix = numeric
    plan.eigen = _eigen_directions(matrix)
    plan.axis_labels = ("x", "y")
    if plan.eigen:
        listed = ", ".join(f"{value:.4g}" for value, _ in plan.eigen)
        plan.notes.append(
            f"eigenvalues {listed} — the drawn eigenvectors are the directions "
            "this matrix only stretches, never turns."
        )
    else:
        plan.notes.append(
            "no real eigenvectors: this matrix turns every direction."
        )
    determinant = float(np.linalg.det(numeric))
    plan.notes.append(
        f"determinant {determinant:.4g} — areas are scaled by "
        f"{abs(determinant):.4g}"
        + (", and orientation is flipped." if determinant < 0 else ".")
    )
    return plan


def _eigen_directions(matrix: sp.MatrixBase) -> list[tuple[float, np.ndarray]]:
    """Real eigenvalues with unit eigenvectors, exactly where SymPy can."""
    found: list[tuple[float, np.ndarray]] = []
    try:
        pairs = matrix.eigenvects()
    except SYMBOLIC_FAILURE:
        return found
    for value, _multiplicity, vectors in pairs:
        number = complex(sp.N(value))
        if abs(number.imag) > 1e-9 * max(abs(number.real), 1.0):
            continue
        for vector in vectors:
            try:
                pair = np.array([float(sp.N(vector[0])), float(sp.N(vector[1]))])
            except (TypeError, ValueError):
                continue
            norm = float(np.hypot(*pair))
            if norm > 0:
                found.append((float(number.real), pair / norm))
    return found


def _callable_name(function: Callable[..., Any]) -> str:
    """A legend label for a plain function.

    ``<lambda>`` is not a name a reader wants in a legend, and it is not valid
    Python either — which matters because the label is written into the code
    ``show_python()`` emits.
    """
    name = getattr(function, "__name__", "")
    return name if name.isidentifier() else "f"


def _label_of(expr: sp.Expr, label: str | None) -> str:
    return label or sp.sstr(expr)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
