"""``analyze()`` — the properties of an expression (PRD 5.6).

Eight properties, in the order the PRD names them: roots, extrema, inflection
points, symmetry, periodicity, asymptotes, discontinuities and monotonic
intervals.

Two rules govern the whole module.

**Exact where SymPy can, approximate where it cannot — and say which.** Every
property is attempted symbolically first. ``solveset`` answers with a
``FiniteSet`` when it has really solved the equation and a ``ConditionSet``
when it has merely restated it, which is the signal to fall back to a numeric
search. A numeric answer is never presented as an exact one: it carries
``approximate=True`` all the way out to the printed panel, because a learner
who cannot tell a proof from a sample is worse off than one who was told
nothing.

**Properties, never derivations** (PRD 2.2). This module reports *what an
expression is*. It does not narrate how any of it was obtained, and it must
never grow a function that does. That is the scope defence line, not a
stylistic preference.

The module is pure Python and SymPy: no Plotly, no frontend (PRD 6.1).
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from typing import Callable, Final, Iterable, Iterator, Sequence

import numpy as np
import sympy as sp

from ._budget import SymbolicTimeout, get_budget, within_budget
from ._failure import SYMBOLIC_FAILURE
from ._source import expr_source, import_block
from ._sets import set_to_points
from .sampling import (
    DEFAULT_CONFIG,
    DomainInfo,
    NumericFunction,
    describe_domain,
    detect_jumps,
)

__all__ = [
    "Points",
    "Asymptote",
    "Symmetry",
    "Periodicity",
    "Analysis",
    "analyze_expression",
]

Array = np.ndarray

#: Samples per continuous piece used by every numeric fallback.
_PROBE_POINTS: Final[int] = 2001
#: Bisection steps for a numeric root. 80 takes a double to its last bit.
_BISECTIONS: Final[int] = 80
#: Ternary-search steps when refining a numeric extremum.
_TERNARY_STEPS: Final[int] = 60
#: A found root must be this small relative to the local scale of the curve.
_ROOT_TOLERANCE: Final[float] = 1e-6


# --------------------------------------------------------------------------
# running a symbolic attempt under a deadline
# --------------------------------------------------------------------------

#: Symbolic steps that ran out of time during the current `analyze_expression`.
#:
#: A timeout and a refusal both send a property down the numeric path, and the
#: existing `approximate=True` already says the answer was sampled. What it
#: cannot say is *why*, and the two reasons are not interchangeable to a reader:
#: SymPy declining is a fact about the mathematics, running out of five seconds
#: is a fact about this machine — re-run it with a larger budget and the answer
#: may become exact.
#:
#: Thread-local and reset at the top of every `analyze_expression`, so this is
#: append-only diagnostics scoped to one call and never read across calls. That
#: is the whole of its contract; it is state, and it is kept this small on
#: purpose rather than threaded through nine signatures that do not otherwise
#: need it.
_expired: threading.local = threading.local()


def _expired_steps() -> list[str]:
    log: list[str] | None = getattr(_expired, "steps", None)
    if log is None:
        log = []
        _expired.steps = log
    return log


def _attempt(operation: Callable[..., object], *args: object) -> object:
    """One symbolic attempt, under the module budget.

    Returns what ``operation`` returned, or :data:`_NO_RESULT` when SymPy
    declined or ran out of time. A timeout is additionally recorded, because
    only the caller knows which property was being computed.

    Once *anything* has expired in this call, the rest are not attempted. Eight
    properties over one expression means up to a dozen symbolic steps, and a
    budget applied to each independently turns a 5-second limit into a
    minute-long wait for the same answer — the first timeout is already evidence
    about this expression, not about that one step, so it is used as such.
    """
    if _expired_steps():
        return _NO_RESULT
    try:
        return within_budget(operation, *args)
    except SymbolicTimeout:
        _expired_steps().append(getattr(operation, "__name__", str(operation)))
        return _NO_RESULT
    except SYMBOLIC_FAILURE:
        return _NO_RESULT


class _NoResult:
    """The absence of a symbolic answer.

    A distinct sentinel rather than ``None``, because ``None`` is a perfectly
    good answer from ``periodicity()``: it means "not periodic", which is a
    result, not a failure to produce one.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<no symbolic result>"


_NO_RESULT: Final[_NoResult] = _NoResult()


def _budget_note(steps: Sequence[str]) -> str:
    """One line telling the reader what a larger budget might buy."""
    named = ", ".join(sorted(set(steps)))
    budget = get_budget()
    return (
        f"{named} exceeded the {budget:g}s symbolic budget, so the affected "
        "properties were found numerically and are marked approximate. "
        "mathslate.core.set_symbolic_budget(30) — or None for no limit — "
        "will let SymPy keep going."
    )


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Points:
    """Locations on the axis, and how confident we are about them.

    ``approximate`` describes the *locations*: ``False`` means SymPy solved the
    equation and ``symbolic`` holds the closed forms, ``True`` means they came
    from sampling and bisection and there may be more that the sampling missed.
    """

    values: tuple[float, ...] = ()
    symbolic: tuple[sp.Expr, ...] = ()
    approximate: bool = False
    #: What produced these — a receipt, not a derivation. See `Analysis.provenance`.
    method: str = ""

    def __bool__(self) -> bool:
        return bool(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[float]:
        return iter(self.values)

    def __getitem__(self, index: int) -> float:
        return self.values[index]

    def describe(self) -> str:
        if not self.values:
            return "none"
        if self.symbolic and not self.approximate:
            listed = ", ".join(sp.sstr(v) for v in self.symbolic)
        else:
            listed = ", ".join(_number(v) for v in self.values)
        return listed + (" (approximate)" if self.approximate else "")


@dataclass(frozen=True)
class Asymptote:
    """One asymptote. ``kind`` is ``"vertical"``, ``"horizontal"`` or ``"oblique"``."""

    kind: str
    #: Where it sits: the x of a vertical one, the line ``y = ...`` otherwise.
    line: sp.Expr
    #: ``"-oo"`` / ``"+oo"`` for the direction approached; ``""`` for vertical.
    direction: str = ""
    approximate: bool = False
    method: str = ""

    def describe(self) -> str:
        if self.kind == "vertical":
            text = f"x = {sp.sstr(self.line)}"
        else:
            arrow = "x -> -oo" if self.direction == "-oo" else "x -> +oo"
            text = f"y = {sp.sstr(self.line)} as {arrow}"
        return text + (" (approximate)" if self.approximate else "")


@dataclass(frozen=True)
class Symmetry:
    """``"even"``, ``"odd"``, ``"neither"`` or ``"unknown"`` about the origin."""

    kind: str
    approximate: bool = False
    method: str = ""

    def describe(self) -> str:
        return self.kind + (" (approximate)" if self.approximate else "")


@dataclass(frozen=True)
class Periodicity:
    """The period, or ``None`` when the expression is not periodic."""

    period: float | None = None
    symbolic: sp.Expr | None = None
    approximate: bool = False
    method: str = ""

    def describe(self) -> str:
        if self.period is None:
            return "not periodic"
        text = sp.sstr(self.symbolic) if self.symbolic is not None else _number(self.period)
        return text + (" (approximate)" if self.approximate else "")


@dataclass(frozen=True)
class Analysis:
    """Everything :func:`analyze_expression` found, plus how it found it.

    ``.sympy`` returns the expression, so the escape hatch of PRD 4 holds here
    as it does on ``PlotResult``.
    """

    expr: sp.Expr
    symbol: sp.Symbol
    window: tuple[float, float]
    roots: Points = field(default_factory=Points)
    maxima: Points = field(default_factory=Points)
    minima: Points = field(default_factory=Points)
    inflections: Points = field(default_factory=Points)
    discontinuities: Points = field(default_factory=Points)
    asymptotes: tuple[Asymptote, ...] = ()
    symmetry: Symmetry = field(default_factory=lambda: Symmetry("unknown"))
    periodicity: Periodicity = field(default_factory=Periodicity)
    increasing: tuple[tuple[float, float], ...] = ()
    decreasing: tuple[tuple[float, float], ...] = ()
    notes: tuple[str, ...] = ()

    # -- escape hatch ------------------------------------------------------

    @property
    def sympy(self) -> sp.Expr:
        """The expression these properties are about."""
        return self.expr

    @property
    def approximate(self) -> bool:
        """True when *any* line of the report came from sampling."""
        return any(
            part.approximate
            for part in (
                self.roots,
                self.maxima,
                self.minima,
                self.inflections,
                self.discontinuities,
                self.symmetry,
                self.periodicity,
                *self.asymptotes,
            )
        )

    # -- presentation ------------------------------------------------------

    def rows(self) -> tuple[tuple[str, str], ...]:
        """The report as ``(label, value)`` pairs — the panel's contents."""
        return (
            ("expression", sp.sstr(self.expr)),
            ("window", f"{self.symbol.name} in [{_number(self.window[0])}, "
                       f"{_number(self.window[1])}]"),
            ("roots", self.roots.describe()),
            ("maxima", self.maxima.describe()),
            ("minima", self.minima.describe()),
            ("inflections", self.inflections.describe()),
            ("symmetry", self.symmetry.describe()),
            ("period", self.periodicity.describe()),
            ("asymptotes", _join(a.describe() for a in self.asymptotes)),
            ("discontinuities", self.discontinuities.describe()),
            ("increasing on", _join(_interval(i) for i in self.increasing)),
            ("decreasing on", _join(_interval(i) for i in self.decreasing)),
        )

    def provenance(self) -> tuple[tuple[str, str], ...]:
        """``(label, what produced it)`` for every property that has an answer.

        This is a **receipt, not a derivation** (PRD 2.2). It says which call
        was made and whether it succeeded — facts recorded as they happened,
        never a reconstruction of how a human would reach the same answer.
        SymPy exposes no derivation trace for solving, differentiating or
        taking limits, so anything shaped like a worked solution here would
        have to be invented, which is exactly what 2.2 forbids.
        """
        parts = (
            ("roots", self.roots),
            ("maxima", self.maxima),
            ("minima", self.minima),
            ("inflections", self.inflections),
            ("discontinuities", self.discontinuities),
            ("symmetry", self.symmetry),
            ("period", self.periodicity),
        )
        out = [(label, part.method) for label, part in parts if part.method]
        seen: set[str] = set()
        for asymptote in self.asymptotes:
            if asymptote.method and asymptote.method not in seen:
                seen.add(asymptote.method)
                out.append((f"{asymptote.kind} asymptote", asymptote.method))
        return tuple(out)

    def python(self) -> str:
        """The SymPy that reproduces this report, as runnable source.

        The same promise ``show_python()`` makes for a plot (PRD 5.5), applied
        to the properties: it runs verbatim, and it is the honest answer to
        "how did you get this" — the calls that were actually made.
        """
        return _analysis_source(self)

    def show_python(self) -> str:
        """Print :meth:`python` and return it."""
        code = self.python()
        print(code)
        return code

    def text(self) -> str:
        """The plain-text panel."""
        rows = self.rows()
        width = max(len(label) for label, _ in rows)
        lines = [f"{label.rjust(width)} : {value}" for label, value in rows]
        if self.notes:
            lines += [f"{'note'.rjust(width)} : {note}" for note in self.notes]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """PRD 5.6 asks for a *collapsed* panel; `<details>` is exactly that."""
        cells = "".join(
            f"<tr><th style='text-align:right;padding:2px 8px;"
            f"font-weight:500;opacity:0.7'>{_escape(label)}</th>"
            f"<td style='padding:2px 8px'><code>{_escape(value)}</code></td></tr>"
            for label, value in self.rows()
        )
        notes = "".join(
            f"<div style='padding:2px 8px;opacity:0.7'>{_escape(note)}</div>"
            for note in self.notes
        )
        # The receipt, collapsed inside the panel: available to whoever wants
        # it, in nobody's way otherwise.
        receipts = "".join(
            f"<tr><th style='text-align:right;padding:1px 8px;font-weight:400;"
            f"opacity:0.6'>{_escape(label)}</th>"
            f"<td style='padding:1px 8px;opacity:0.6'>{_escape(how)}</td></tr>"
            for label, how in self.provenance()
        )
        if receipts:
            receipts = (
                "<details style='padding:2px 8px'><summary style='cursor:pointer;"
                "opacity:0.6'>how each was obtained</summary>"
                f"<table>{receipts}</table></details>"
            )
        summary = sp.sstr(self.expr)
        if self.approximate:
            summary += "  (contains approximate results)"
        return (
            "<details><summary style='cursor:pointer'>"
            f"analyze({_escape(summary)})</summary>"
            f"<table>{cells}</table>{notes}{receipts}</details>"
        )

    def __str__(self) -> str:
        return self.text()

    def __repr__(self) -> str:
        return f"<Analysis {sp.sstr(self.expr)} on {self.window}>"


# --------------------------------------------------------------------------
# reproducing the report as source (PRD 5.5, applied to analyze)
# --------------------------------------------------------------------------


def _analysis_source(report: Analysis) -> str:
    """Emit the SymPy that produces ``report``. It runs verbatim.

    Exactly solved properties emit the call that solved them. Sampled ones
    emit their values as literals with a comment naming the search — the same
    choice ``show_python()`` makes for a callable, and for the same reason:
    emitting a call that cannot reproduce the answer would be a lie that
    happens to run.
    """
    name = report.symbol.name
    source, needed = expr_source(report.expr, {name})
    lo, hi = report.window

    body: list[str] = [
        f"{name} = sp.symbols({name!r}, real=True)",
        f"expr = {source}",
        f"window = sp.Interval({_literal(lo)}, {_literal(hi)})",
        "",
    ]

    # Every template is built against the report's own symbol name. Patching
    # a hardcoded "x" afterwards does not work — `sp.diff(expr, x)` has no
    # trailing comma to match on, and the emitted code fails with a NameError
    # for any expression in t, u or theta.
    body += _emit(
        report.roots,
        "roots",
        exact=[f"roots = sp.solveset(expr, {name}, window)"],
    )
    if report.maxima.method or report.minima.method or report.inflections.method:
        body += [
            "",
            f"first = sp.diff(expr, {name})",
            f"second = sp.diff(expr, {name}, 2)",
        ]

    body += _emit(
        report.maxima,
        "maxima",
        exact=[
            f"stationary = sp.solveset(first, {name}, window)",
            f"maxima = {{p for p in stationary if second.subs({name}, p) < 0}}",
            f"minima = {{p for p in stationary if second.subs({name}, p) > 0}}",
        ],
        also=report.minima,
    )
    body += _emit(
        report.inflections,
        "inflections",
        exact=[
            f"inflections = {{p for p in sp.solveset(second, {name}, window)",
            f"               if second.subs({name}, p - 1e-6)"
            f" * second.subs({name}, p + 1e-6) < 0}}",
        ],
    )
    body += _emit(
        report.discontinuities,
        "discontinuities",
        exact=[
            "from sympy.calculus.singularities import singularities",
            f"discontinuities = singularities(expr, {name}, sp.S.Reals)",
        ],
    )

    if report.symmetry.method and not report.symmetry.approximate:
        body += [
            "",
            f"# symmetry: {report.symmetry.kind}",
            f"mirrored = expr.subs({name}, -{name})",
            "even = sp.simplify(mirrored - expr) == 0",
            "odd = sp.simplify(mirrored + expr) == 0",
        ]
    if report.periodicity.method:
        body += ["", "# period", f"period = sp.periodicity(expr, {name})"]
    if report.asymptotes:
        body += [
            "",
            "# asymptotes: the tail is y = m*x + b when both limits settle.",
            f"m = sp.limit(expr / {name}, {name}, sp.oo)",
            f"b = sp.limit(expr - m * {name}, {name}, sp.oo)",
        ]

    body += [
        "",
        "print(" + ", ".join(_printable(report)) + ")",
    ]
    header = [
        "# The calls that produced this report — this runs exactly as printed.",
        import_block(needed, numpy=False, plotly=False),
        "",
    ]
    return "\n".join(header + body).rstrip() + "\n"


def _emit(
    points: Points, label: str, exact: list[str], also: Points | None = None
) -> list[str]:
    """One property's worth of source: the real call, or the literals."""
    if not points.method and not (also and also.method):
        return []
    if not points.approximate:
        return ["", f"# {label}: {points.method}", *exact]
    lines = ["", f"# {label}: {points.method}.", "# Sampled, so the values go in as literals."]
    lines.append(f"{label} = {list(points.values)!r}")
    if also is not None:
        lines.append(f"minima = {list(also.values)!r}")
    return lines


def _printable(report: Analysis) -> list[str]:
    names = []
    for label, part in (
        ("roots", report.roots),
        ("maxima", report.maxima),
        ("inflections", report.inflections),
    ):
        if part.method:
            names.append(label)
    return names or ["expr"]


def _literal(value: float) -> str:
    return repr(round(float(value), 12))


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------


def analyze_expression(
    expr: sp.Expr, symbol: sp.Symbol, window: tuple[float, float]
) -> Analysis:
    """Detect every property PRD 5.6 lists, exactly where possible."""
    lo, hi = float(window[0]), float(window[1])
    if not hi > lo:
        raise ValueError(f"empty window: ({lo}, {hi})")

    notes: list[str] = []
    expired = _expired_steps()
    expired.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        domain = describe_domain(expr, symbol, lo, hi)
        function = NumericFunction(expr, symbol)
        derivative = _derivative(expr, symbol, 1)
        second = _derivative(expr, symbol, 2)

        roots = _roots(expr, symbol, domain, function, lo, hi, notes)
        maxima, minima = _extrema(expr, symbol, domain, function, derivative, second, lo, hi)
        inflections = _inflections(expr, symbol, domain, second, lo, hi)
        discontinuities = _discontinuities(domain, expr, symbol, function, lo, hi)
        asymptotes = _asymptotes(expr, symbol, discontinuities)
        symmetry = _symmetry(expr, symbol, function, lo, hi)
        period = _periodicity(expr, symbol)
        increasing, decreasing = _monotonic(
            derivative, symbol, domain, maxima, minima, discontinuities, lo, hi
        )

    if derivative is None:
        notes.append(
            "SymPy could not differentiate this expression, so extrema, "
            "inflection points and monotonic intervals were not computed."
        )
    if not domain.domain_known:
        notes.append(
            "SymPy could not determine the real domain; the window was searched whole."
        )
    if expired:
        notes.append(_budget_note(expired))

    return Analysis(
        expr=expr,
        symbol=symbol,
        window=(lo, hi),
        roots=roots,
        maxima=maxima,
        minima=minima,
        inflections=inflections,
        discontinuities=discontinuities,
        asymptotes=asymptotes,
        symmetry=symmetry,
        periodicity=period,
        increasing=increasing,
        decreasing=decreasing,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# symbolic attempts
# --------------------------------------------------------------------------


def _derivative(expr: sp.Expr, symbol: sp.Symbol, order: int) -> sp.Expr | None:
    """The derivative, or ``None`` when SymPy could not really take it.

    ``diff`` does not raise on ``floor(x)``: it returns an *unevaluated*
    ``Derivative`` node, which is a restatement rather than an answer and which
    ``lambdify`` cannot print at all. Treating it as a derivative crashes one
    step later, so it is rejected here.
    """
    result = _attempt(sp.diff, expr, symbol, order)
    if isinstance(result, _NoResult):
        return None
    if result.has(sp.Derivative):
        return None
    return result


def _numeric(expr: sp.Expr | None, symbol: sp.Symbol) -> NumericFunction | None:
    """A callable for ``expr``, or ``None`` if it cannot be made into one."""
    if expr is None:
        return None
    try:
        return NumericFunction(expr, symbol)
    except SYMBOLIC_FAILURE:
        return None


def _solve_exactly(
    equation: sp.Expr, symbol: sp.Symbol, lo: float, hi: float
) -> tuple[tuple[float, ...], tuple[sp.Expr, ...]] | None:
    """Solve ``equation = 0`` on the window, or ``None`` if SymPy could not.

    ``solveset`` answers a question it has not solved with a ``ConditionSet``,
    which is a restatement rather than a solution. Treating that as an answer
    would report "no roots" for ``x - cos(x)``, which has one.
    """
    window = sp.Interval(sp.Float(lo), sp.Float(hi))
    solution = _attempt(sp.solveset, equation, symbol, window)
    if isinstance(solution, _NoResult):
        return None
    if solution.has(sp.ConditionSet) or isinstance(solution, sp.ConditionSet):
        return None
    if solution is sp.S.Reals or solution.has(sp.Interval):
        # A whole interval of solutions is not a set of isolated points.
        return None
    try:
        values = tuple(set_to_points(solution, lo, hi))
    except SYMBOLIC_FAILURE:
        return None
    symbolic: tuple[sp.Expr, ...] = ()
    if isinstance(solution, sp.FiniteSet):
        ordered = sorted(solution.args, key=lambda e: _to_float(e) or 0.0)
        inside = [e for e in ordered if (f := _to_float(e)) is not None and lo <= f <= hi]
        if len(inside) == len(values):
            symbolic = tuple(sp.nsimplify(e) for e in inside)
    return values, symbolic


def _symmetry(
    expr: sp.Expr, symbol: sp.Symbol, function: NumericFunction, lo: float, hi: float
) -> Symmetry:
    """Even, odd, or neither — proven if `simplify` can, sampled if not."""
    mirrored = _attempt(lambda: expr.subs(symbol, -symbol))
    if not isinstance(mirrored, _NoResult):
        how = "simplify(f(-x) -+ f(x)) == 0"
        # One `simplify` at a time. It is the most expensive call in the module —
        # the reason this module needs a budget at all — so an even function must
        # not pay for the odd test it does not need.
        even = _attempt(sp.simplify, mirrored - expr)
        if not isinstance(even, _NoResult):
            if even == 0:
                return Symmetry("even", method=how)
            odd = _attempt(sp.simplify, mirrored + expr)
            if not isinstance(odd, _NoResult):
                return Symmetry("odd" if odd == 0 else "neither", method=how)
    reach = min(abs(lo), abs(hi)) or 1.0
    grid = np.linspace(-reach, reach, 401, dtype=np.float64)
    values = function(grid)
    flipped = values[::-1]
    usable = np.isfinite(values) & np.isfinite(flipped)
    if np.count_nonzero(usable) < 8:
        return Symmetry("unknown")
    scale = max(float(np.nanmax(np.abs(values[usable]))), 1.0)
    how = "simplify() could not decide, so f(-x) was compared with f(x) at 401 points"
    if np.allclose(values[usable], flipped[usable], atol=1e-9 * scale):
        return Symmetry("even", approximate=True, method=how)
    if np.allclose(values[usable], -flipped[usable], atol=1e-9 * scale):
        return Symmetry("odd", approximate=True, method=how)
    return Symmetry("neither", approximate=True, method=how)


def _periodicity(expr: sp.Expr, symbol: sp.Symbol) -> Periodicity:
    found = _attempt(sp.periodicity, expr, symbol)
    if isinstance(found, _NoResult):
        return Periodicity()
    if found is None or found is sp.S.Zero:
        return Periodicity()
    value = _to_float(found)
    if value is None or value <= 0.0:
        return Periodicity()
    return Periodicity(period=value, symbolic=found, method="sympy.periodicity()")


def _asymptotes(
    expr: sp.Expr, symbol: sp.Symbol, discontinuities: Points
) -> tuple[Asymptote, ...]:
    """Vertical ones from the poles; horizontal and oblique from the limits."""
    found: list[Asymptote] = []
    for index, place in enumerate(discontinuities.values):
        exact = (
            discontinuities.symbolic[index]
            if index < len(discontinuities.symbolic)
            else sp.Float(place)
        )
        # Probe the limit at the *exact* location. `limit(tan(x), x, 1.5707963)`
        # is a large finite number; only `limit(tan(x), x, pi/2)` is infinite.
        if _blows_up(expr, symbol, exact):
            found.append(
                Asymptote(
                    "vertical",
                    exact,
                    approximate=discontinuities.approximate,
                    method=f"limit(expr, {symbol.name}, place, '+'/'-') is infinite",
                )
            )

    for direction, limit_point in (("-oo", -sp.oo), ("+oo", sp.oo)):
        line = _end_behaviour(expr, symbol, limit_point)
        if line is not None:
            kind = "horizontal" if line.is_number else "oblique"
            found.append(
                Asymptote(
                    kind,
                    line,
                    direction=direction,
                    method=(
                        f"m = limit(expr/{symbol.name}, {symbol.name}, {direction}), "
                        f"b = limit(expr - m*{symbol.name}, {symbol.name}, {direction})"
                    ),
                )
            )
    return tuple(found)


def _blows_up(expr: sp.Expr, symbol: sp.Symbol, place: sp.Expr) -> bool:
    """True when either one-sided limit at ``place`` is infinite."""
    for side in ("+", "-"):
        value = _attempt(sp.limit, expr, symbol, place, side)
        if isinstance(value, _NoResult):
            continue
        if value in (sp.oo, -sp.oo, sp.zoo) or value.has(sp.oo, sp.zoo):
            return True
    return False


def _end_behaviour(expr: sp.Expr, symbol: sp.Symbol, at: sp.Expr) -> sp.Expr | None:
    """``y = mx + b`` approached at ``at``, or ``None`` if there is none."""
    found = _attempt(sp.limit, expr / symbol, symbol, at)
    if isinstance(found, _NoResult):
        return None
    slope = _settled(found)
    if slope is None:
        return None
    found = _attempt(sp.limit, expr - slope * symbol, symbol, at)
    if isinstance(found, _NoResult):
        return None
    intercept = _settled(found)
    if intercept is None:
        return None
    line = sp.simplify(slope * symbol + intercept)
    if line.free_symbols and slope == 0:
        return None
    return line


def _settled(limit: sp.Expr) -> sp.Expr | None:
    """A limit that is a single finite number, or ``None``.

    SymPy reports an oscillating end behaviour as ``AccumBounds``: the limit of
    ``sin(x)`` at infinity comes back as ``AccumBounds(-1, 1)``, which is
    finite in the sense of being bounded but is emphatically not a value the
    curve approaches. Reading it as one would give ``sin(x)`` a horizontal
    asymptote.
    """
    if isinstance(limit, sp.AccumBounds) or limit.has(sp.AccumBounds):
        return None
    if not (limit.is_number and limit.is_finite and limit.is_real):
        return None
    return limit


# --------------------------------------------------------------------------
# roots, extrema, inflections — symbolic first, then sampled
# --------------------------------------------------------------------------


def _roots(
    expr: sp.Expr,
    symbol: sp.Symbol,
    domain: DomainInfo,
    function: NumericFunction,
    lo: float,
    hi: float,
    notes: list[str],
) -> Points:
    exact = _solve_exactly(expr, symbol, lo, hi)
    if exact is not None:
        values, symbolic = exact
        inside = _within_domain(values, domain)
        return Points(
            values=inside,
            symbolic=symbolic if len(inside) == len(values) else (),
            method=f"solveset(expr, {symbol.name}, window) returned a FiniteSet",
        )
    values_n, flat = _numeric_roots(function, domain)
    if flat:
        notes.append(
            "the expression is zero across whole stretches, not at isolated "
            "points; each root listed is where one such stretch begins."
        )
    return Points(
        values=values_n,
        approximate=True,
        method=(
            "solveset returned a ConditionSet, so each continuous piece was "
            f"scanned at {_PROBE_POINTS} points for sign changes and each "
            "bracket bisected"
        ),
    )


def _numeric_roots(
    function: NumericFunction, domain: DomainInfo
) -> tuple[tuple[float, ...], bool]:
    """Sign changes on each continuous piece, bisected.

    Searching per piece is what keeps a pole out of the answer: ``1/x`` changes
    sign across the origin without ever being zero, and the origin is not in
    any piece.

    Returns ``(roots, flat)``. ``flat`` says the expression was zero across a
    whole stretch rather than at isolated points — ``floor(x)`` is zero on all
    of ``[0, 1)`` — in which case only the start of each stretch is listed and
    the caller says so. Listing every sample would report a hundred roots where
    the honest answer is "an interval".
    """
    found: list[float] = []
    flat = False
    for start, end in domain.intervals:
        grid = np.linspace(start, end, _PROBE_POINTS, dtype=np.float64)
        values = function(grid)
        finite = np.isfinite(values)
        scale = _scale_of(values)
        zero = finite & (values == 0.0)
        # The first sample of each run of zeros; a run longer than one sample
        # is a stretch, not a point.
        starts_run = zero & ~np.concatenate(([False], zero[:-1]))
        for index in np.flatnonzero(starts_run):
            found.append(float(grid[index]))
            if index + 1 < zero.size and zero[index + 1]:
                flat = True
        for index in np.flatnonzero(finite[:-1] & finite[1:]):
            left, right = float(values[index]), float(values[index + 1])
            if left == 0.0 or right == 0.0:
                continue
            if left * right < 0.0:
                place = _bisect(function, float(grid[index]), float(grid[index + 1]))
                if place is not None and abs(_at(function, place)) <= _ROOT_TOLERANCE * scale:
                    found.append(place)
    return _tidy(found), flat


def _bisect(function: NumericFunction, left: float, right: float) -> float | None:
    left_value = _at(function, left)
    right_value = _at(function, right)
    if not (np.isfinite(left_value) and np.isfinite(right_value)):
        return None
    if left_value * right_value > 0.0:
        return None
    for _ in range(_BISECTIONS):
        middle = 0.5 * (left + right)
        if middle in (left, right):
            break
        value = _at(function, middle)
        if not np.isfinite(value):
            return None
        if value == 0.0:
            return middle
        if left_value * value < 0.0:
            right, right_value = middle, value
        else:
            left, left_value = middle, value
    return 0.5 * (left + right)


def _extrema(
    expr: sp.Expr,
    symbol: sp.Symbol,
    domain: DomainInfo,
    function: NumericFunction,
    derivative: sp.Expr | None,
    second: sp.Expr | None,
    lo: float,
    hi: float,
) -> tuple[Points, Points]:
    """Stationary points, split into maxima and minima."""
    if derivative is None:
        return Points(), Points()

    exact = _solve_exactly(derivative, symbol, lo, hi)
    if exact is not None:
        values, symbolic = exact
        inside = _within_domain(values, domain)
        highs: list[float] = []
        lows: list[float] = []
        high_exact: list[sp.Expr] = []
        low_exact: list[sp.Expr] = []
        aligned = symbolic if len(symbolic) == len(values) else ()
        for place in inside:
            index = values.index(place)
            verdict = _classify(
                second, derivative, symbol, place, aligned[index] if aligned else None
            )
            if verdict > 0:
                lows.append(place)
                if aligned:
                    low_exact.append(aligned[index])
            elif verdict < 0:
                highs.append(place)
                if aligned:
                    high_exact.append(aligned[index])
        how = f"solveset(f', {symbol.name}, window), split by the sign of f'' at each"
        return (
            Points(tuple(highs), tuple(high_exact), method=how),
            Points(tuple(lows), tuple(low_exact), method=how),
        )

    highs_n, lows_n = _numeric_extrema(function, domain)
    how = (
        f"solveset could not solve f' = 0, so {_PROBE_POINTS} samples per piece "
        "were scanned for local turns and each refined by ternary search"
    )
    return (
        Points(values=highs_n, approximate=True, method=how),
        Points(values=lows_n, approximate=True, method=how),
    )


def _classify(
    second: sp.Expr | None,
    derivative: sp.Expr,
    symbol: sp.Symbol,
    place: float,
    exact: sp.Expr | None = None,
) -> int:
    """``+1`` minimum, ``-1`` maximum, ``0`` neither (a saddle or a corner).

    The second-derivative test is used where it is decisive. Two traps:
    substituting the *float* 1.5707963 into ``cos(x)`` gives 6e-17 rather than
    the 0 that ``cos(pi/2)`` gives, so the exact location is preferred and a
    negligible curvature counts as zero. Otherwise ``x - cos(x)``, whose slope
    ``1 + sin(x)`` merely touches zero, would be reported as having extrema
    where it only has saddle points.
    """
    if second is not None:
        for candidate in (exact, sp.Float(place)):
            if candidate is None:
                continue
            try:
                curvature = sp.N(second.subs(symbol, candidate))
            except SYMBOLIC_FAILURE:
                continue
            value = _to_float(curvature)
            if value is None:
                continue
            if abs(value) > 1e-12:
                return 1 if value > 0 else -1
            break

    # The second derivative vanished or could not be read: watch the slope
    # change sign instead, which also covers a corner such as |x|.
    slope = _numeric(derivative, symbol)
    if slope is None:
        return 0
    step = max(abs(place), 1.0) * 1e-6
    before = _at(slope, place - step)
    after = _at(slope, place + step)
    if not (np.isfinite(before) and np.isfinite(after)):
        return 0
    if before < 0.0 < after:
        return 1
    if after < 0.0 < before:
        return -1
    return 0


def _numeric_extrema(
    function: NumericFunction, domain: DomainInfo
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Local extrema read off a fine sample, then refined by ternary search."""
    highs: list[float] = []
    lows: list[float] = []
    for start, end in domain.intervals:
        grid = np.linspace(start, end, _PROBE_POINTS, dtype=np.float64)
        values = function(grid)
        usable = np.isfinite(values[:-2]) & np.isfinite(values[1:-1]) & np.isfinite(values[2:])
        middle = values[1:-1]
        is_high = usable & (middle > values[:-2]) & (middle > values[2:])
        is_low = usable & (middle < values[:-2]) & (middle < values[2:])
        for index in np.flatnonzero(is_high):
            highs.append(_ternary(function, grid[index], grid[index + 2], maximise=True))
        for index in np.flatnonzero(is_low):
            lows.append(_ternary(function, grid[index], grid[index + 2], maximise=False))
    return _tidy(highs), _tidy(lows)


def _ternary(
    function: NumericFunction, left: float, right: float, *, maximise: bool
) -> float:
    """Narrow a bracketing triple onto the extremum inside it."""
    for _ in range(_TERNARY_STEPS):
        span = right - left
        if span <= max(abs(left), 1.0) * 1e-15:
            break
        a = left + span / 3.0
        b = right - span / 3.0
        fa, fb = _at(function, a), _at(function, b)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            break
        if (fa < fb) if maximise else (fa > fb):
            left = a
        else:
            right = b
    return 0.5 * (left + right)


def _inflections(
    expr: sp.Expr,
    symbol: sp.Symbol,
    domain: DomainInfo,
    second: sp.Expr | None,
    lo: float,
    hi: float,
) -> Points:
    """Where concavity really changes — a zero of ``f''`` is not enough.

    ``x**4`` has ``f'' = 12x**2``, which vanishes at the origin without the
    curve ever changing concavity. Reporting it would be wrong.
    """
    curvature = _numeric(second, symbol)
    if second is None or curvature is None:
        return Points()
    exact = _solve_exactly(second, symbol, lo, hi)
    if exact is not None:
        values, symbolic = exact
        inside = [p for p in _within_domain(values, domain) if _changes_sign(curvature, p)]
        aligned = symbolic if len(symbolic) == len(values) else ()
        keep = tuple(aligned[values.index(p)] for p in inside) if aligned else ()
        return Points(
            tuple(inside),
            keep,
            method=f"solveset(f'', {symbol.name}, window), keeping only where concavity changes",
        )

    found: list[float] = []
    for start, end in domain.intervals:
        grid = np.linspace(start, end, _PROBE_POINTS, dtype=np.float64)
        values = curvature(grid)
        finite = np.isfinite(values)
        for index in np.flatnonzero(finite[:-1] & finite[1:]):
            if float(values[index]) * float(values[index + 1]) < 0.0:
                place = _bisect(curvature, float(grid[index]), float(grid[index + 1]))
                if place is not None:
                    found.append(place)
    return Points(
        values=_tidy(found),
        approximate=True,
        method=f"sign changes of f'' over {_PROBE_POINTS} samples per piece, bisected",
    )


def _changes_sign(curvature: NumericFunction, place: float) -> bool:
    step = max(abs(place), 1.0) * 1e-6
    before = _at(curvature, place - step)
    after = _at(curvature, place + step)
    if not (np.isfinite(before) and np.isfinite(after)):
        return False
    return before * after < 0.0


# --------------------------------------------------------------------------
# discontinuities and monotonic intervals
# --------------------------------------------------------------------------


def _discontinuities(
    domain: DomainInfo,
    expr: sp.Expr,
    symbol: sp.Symbol,
    function: NumericFunction,
    lo: float,
    hi: float,
) -> Points:
    """Symbolic poles, plus the jumps SymPy cannot see.

    The numeric half reuses PRD 5.3's bisection probe: a genuine jump keeps its
    size as the bracketing interval shrinks. That is what finds every step of
    ``floor`` without naming ``floor``.
    """
    if domain.singular_points:
        symbolic = tuple(sp.nsimplify(sp.Float(p), [sp.pi]) for p in domain.singular_points)
        return Points(
            values=tuple(domain.singular_points),
            symbolic=symbolic,
            method="sympy.calculus.singularities()",
        )

    grid = np.linspace(lo, hi, _PROBE_POINTS, dtype=np.float64)

    def evaluate(values: Array) -> tuple[Array, Array]:
        with np.errstate(all="ignore"):
            return values, function(values)

    _, ys = evaluate(grid)
    # Strictly inside: a jump exactly at the window's edge is not visible in it,
    # and reporting one edge but not the other reads as an asymmetry in the
    # function rather than in where we happened to look.
    margin = (hi - lo) * 1e-9
    jumps = [
        j
        for j in detect_jumps(evaluate, grid, grid, ys, (), DEFAULT_CONFIG)
        if lo + margin < j < hi - margin
    ]
    if not jumps:
        return Points()
    return Points(
        values=_tidy(jumps),
        approximate=True,
        method="the bisection jump probe of PRD 5.3 step 4 (a real jump keeps its size)",
    )


def _monotonic(
    derivative: sp.Expr | None,
    symbol: sp.Symbol,
    domain: DomainInfo,
    maxima: Points,
    minima: Points,
    discontinuities: Points,
    lo: float,
    hi: float,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """Split the window at every turning point, then read the slope's sign."""
    slope = _numeric(derivative, symbol)
    if derivative is None or slope is None:
        return (), ()

    # A turning point may be joined across when the slope keeps its sign — that
    # is how `x**3` reads as one rise. A discontinuity or a domain edge never
    # may: `tan(x)` rises on each branch separately, not once across every pole.
    barriers = {p for p in discontinuities.values if lo < p < hi} | {
        edge for piece in domain.intervals for edge in piece if lo < edge < hi
    }
    cuts = sorted(
        {lo, hi}
        | {p for p in (*maxima.values, *minima.values) if lo < p < hi}
        | barriers
    )
    up: list[tuple[float, float]] = []
    down: list[tuple[float, float]] = []
    for start, end in zip(cuts, cuts[1:]):
        if end - start <= (hi - lo) * 1e-12:
            continue
        if not _inside_domain(0.5 * (start + end), domain):
            continue
        value = _slope_sign(slope, start, end)
        if value == 0:
            continue
        (up if value > 0 else down).append((start, end))
    return _merge(up, barriers), _merge(down, barriers)


def _slope_sign(slope: NumericFunction, start: float, end: float) -> int:
    """The sign of the slope somewhere strictly inside ``(start, end)``.

    Several probe points, because one is not enough: the midpoint of
    ``x**3``'s window ``(-10, 10)`` is exactly its stationary point, where the
    slope is 0 and a single probe would conclude "neither increasing nor
    decreasing" about a curve that rises throughout.
    """
    for fraction in (0.5, 0.25, 0.75, 0.1, 0.9):
        value = _at(slope, start + (end - start) * fraction)
        if np.isfinite(value) and value != 0.0:
            return 1 if value > 0.0 else -1
    return 0


def _merge(
    intervals: list[tuple[float, float]], barriers: set[float]
) -> tuple[tuple[float, float], ...]:
    """Join intervals that touch, unless a barrier sits between them."""
    if not intervals:
        return ()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        touching = abs(start - last_end) <= max(abs(start), 1.0) * 1e-12
        blocked = any(abs(start - b) <= max(abs(start), 1.0) * 1e-12 for b in barriers)
        if touching and not blocked:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return tuple(merged)


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------


def _at(function: NumericFunction, place: float) -> float:
    return float(function(np.array([place], dtype=np.float64))[0])


def _scale_of(values: Array) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(float(np.abs(finite).max()), 1.0)


def _to_float(value: sp.Expr) -> float | None:
    try:
        result = float(sp.N(value))
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _within_domain(values: Sequence[float], domain: DomainInfo) -> tuple[float, ...]:
    return tuple(v for v in values if _inside_domain(v, domain))


def _inside_domain(place: float, domain: DomainInfo) -> bool:
    return any(start <= place <= end for start, end in domain.intervals)


def _tidy(values: Sequence[float], tolerance: float = 1e-9) -> tuple[float, ...]:
    """Sort and drop duplicates that differ only by search noise."""
    out: list[float] = []
    for value in sorted(values):
        if not np.isfinite(value):
            continue
        if out and abs(value - out[-1]) <= tolerance * max(abs(value), 1.0):
            continue
        out.append(float(value))
    return tuple(out)


def _number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def _interval(span: tuple[float, float]) -> str:
    return f"({_number(span[0])}, {_number(span[1])})"


def _join(parts: Iterable[str]) -> str:
    listed = ", ".join(parts)
    return listed or "none"


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
