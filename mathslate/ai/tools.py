"""MathSlate as a tool an outside agent can call.

Every other part of this package points outward: MathSlate asks a model for
code. This one points inward. An agent — Claude, ChatGPT, anything speaking the
usual tool-calling protocols — hands MathSlate an expression and gets back
*computed* roots, extrema, samples and asymptotes, rather than a plausible
recollection of them.

That direction is the useful one for the thing models are worst at. A model
asked for the roots of ``x**3 - 3*x`` will often produce them; asked for the
roots of something ugly it will produce something that looks like roots. Here
SymPy solves it and the answer comes back with :attr:`approximate` and the
``method`` that produced it, so the agent can tell "solveset returned a
FiniteSet" from "sign changes over 2000 samples" instead of treating both as
fact. PRD goal 5 asks that the API be easy for an AI to generate correctly;
this is the same goal approached from the other end — easy for an AI to *use*
correctly, including knowing how far to trust the reply.

Three tools, because three is what the escape hatches already are: draw it,
describe it, tabulate it.

**Arguments are untrusted.** They arrive from a model and reach SymPy, where a
string is evaluated as code — the hole this package had until string literals
were refused in restricted validation. Nothing here interpolates a caller's text
into an expression and hopes: every request is assembled into MathSlate source,
put through the same :func:`~mathslate.ai.suggest._validate_code` allowlist as a
generated suggestion, and run in the same isolated process under the same
wall-clock budget. A tool call is not a more trusted path than a suggestion, and
does not get one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import sympy as sp

from ..errors import MathSlateError, UnsupportedInputError
from .suggest import _run_restricted, _validate_code

__all__ = ["Tool", "TOOLS", "call", "tool_schemas", "tool_names"]

#: How many sampled rows a table tool may return. A model pays for every one of
#: them in context, and a hundred rows of a smooth curve says nothing the first
#: ten do not.
_MAX_ROWS: int = 50
#: Points listed back per property. A window holding hundreds of roots is a fact
#: about the window; the agent is told the count and shown the first few.
_MAX_POINTS: int = 24


@dataclass(frozen=True)
class Tool:
    """One callable an agent can invoke, and the schema describing it."""

    name: str
    description: str
    #: JSON Schema for the arguments object, in the form every provider accepts.
    parameters: dict[str, Any]
    run: Callable[[Mapping[str, Any]], dict[str, Any]]


# --------------------------------------------------------------------------
# turning a request into MathSlate source, safely
# --------------------------------------------------------------------------


def _expression_source(arguments: Mapping[str, Any]) -> str:
    """The ``expr`` argument as source text, refusing anything but a string."""
    expression = arguments.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise UnsupportedInputError(
            "expression must be a non-empty string of MathSlate/SymPy source, "
            'e.g. "sin(x)/x".'
        )
    return expression.strip()


def _range_source(arguments: Mapping[str, Any]) -> str:
    """``(symbol, lo, hi)`` source, or ``""`` when the caller named no window.

    The bounds are re-emitted from ``float()`` rather than pasted through, so a
    model that sends ``"0; import os"`` as a bound produces a number or an
    error, never source.
    """
    window = arguments.get("range")
    if window is None:
        return ""
    if not isinstance(window, Sequence) or isinstance(window, str) or len(window) != 3:
        raise UnsupportedInputError(
            'range must be [symbol, lo, hi], e.g. ["x", -10, 10].'
        )
    symbol, low, high = window
    if not isinstance(symbol, str) or not symbol.isidentifier():
        raise UnsupportedInputError(
            f"the first item of range must be a symbol name; got {symbol!r}."
        )
    try:
        lo, hi = float(low), float(high)
    except (TypeError, ValueError) as error:
        raise UnsupportedInputError(
            f"range bounds must be numbers; got ({low!r}, {high!r})."
        ) from error
    if not (math.isfinite(lo) and math.isfinite(hi)) or not hi > lo:
        raise UnsupportedInputError(
            f"range must be finite with lo < hi; got ({lo}, {hi})."
        )
    return f", ({symbol}, {lo!r}, {hi!r})"


def _evaluate(source: str) -> Any:
    """Validate ``source`` and run it in the isolated process, returning ``result``.

    The same allowlist and the same subprocess as a model-written suggestion.
    An agent's tool call has exactly the trust of a suggestion — which is none —
    so it takes exactly the same road.
    """
    code = f"result = {source}"
    _validate_code(code)
    assigned, _output = _run_restricted(code)
    if "result" not in assigned:
        raise UnsupportedInputError(f"{source} produced no result.")
    return assigned["result"]


# --------------------------------------------------------------------------
# serialising results into something a model can read
# --------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """A JSON-safe float. ``NaN``/``inf`` become ``None`` — JSON has no word."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _points(points: Any) -> dict[str, Any]:
    """One :class:`~mathslate.core.analysis.Points` as fact plus confidence.

    ``approximate`` and ``method`` are carried rather than dropped: they are the
    difference between an answer an agent may state and one it should hedge, and
    a serialiser that keeps only the numbers throws away exactly that.
    """
    values = [v for v in (_number(p) for p in points.values) if v is not None]
    payload: dict[str, Any] = {
        "count": len(points.values),
        "values": values[:_MAX_POINTS],
        "approximate": bool(points.approximate),
        "method": points.method,
    }
    if points.symbolic:
        payload["exact"] = [sp.sstr(e) for e in points.symbolic[:_MAX_POINTS]]
    if len(points.values) > _MAX_POINTS:
        payload["truncated"] = True
    return payload


def _analysis(analysis: Any) -> dict[str, Any]:
    return {
        "expression": sp.sstr(analysis.expr),
        "variable": analysis.symbol.name,
        "window": [_number(analysis.window[0]), _number(analysis.window[1])],
        "roots": _points(analysis.roots),
        "maxima": _points(analysis.maxima),
        "minima": _points(analysis.minima),
        "inflections": _points(analysis.inflections),
        "discontinuities": _points(analysis.discontinuities),
        "asymptotes": [
            {
                "kind": a.kind,
                "at": sp.sstr(a.line),
                "direction": a.direction,
                "approximate": bool(a.approximate),
            }
            for a in analysis.asymptotes
        ],
        "symmetry": analysis.symmetry.kind,
        "period": (
            None
            if analysis.periodicity.period is None
            else _number(analysis.periodicity.period)
        ),
        "increasing": [[_number(lo), _number(hi)] for lo, hi in analysis.increasing],
        "decreasing": [[_number(lo), _number(hi)] for lo, hi in analysis.decreasing],
        "notes": list(analysis.notes),
        # One flag the agent can branch on without reading every `method`.
        "any_approximate": bool(analysis.approximate),
    }


def _plot(result: Any) -> dict[str, Any]:
    """A plot as what can be *said* about it — no image crosses this boundary."""
    plan = result.plan
    drawn = result.sympy
    payload: dict[str, Any] = {
        "summary": result.summary(),
        "kind": plan.kind,
        "notes": list(plan.notes),
        "python": result.python(),
    }
    # What was drawn, not merely how it turned out. The summary reports the
    # sampling ("411 samples"); without the expression itself a reader of these
    # facts cannot say anything about the *function*, and a follow-up request —
    # "the same thing on a log scale" — has nothing to name. Raw data has no
    # expression, and says so by leaving the key out.
    if drawn is not None:
        payload["expression"] = (
            sp.sstr(drawn)
            if isinstance(drawn, sp.Basic)
            else [sp.sstr(part) for part in drawn]
        )
    return payload


def _table(table: Any) -> dict[str, Any]:
    rows = [
        [v for v in (_number(cell) for cell in row)]
        for row in table.rows()[:_MAX_ROWS]
    ]
    return {
        "headers": list(table.headers()),
        "rows": rows,
        "row_count": len(table.rows()),
        "notes": list(getattr(table, "notes", ()) or ()),
    }


# --------------------------------------------------------------------------
# the tools
# --------------------------------------------------------------------------


def _run_analyze(arguments: Mapping[str, Any]) -> dict[str, Any]:
    source = f"analyze({_expression_source(arguments)}{_range_source(arguments)})"
    return _analysis(_evaluate(source))


def _run_plot(arguments: Mapping[str, Any]) -> dict[str, Any]:
    extra = ""
    kind = arguments.get("kind")
    if kind is not None:
        if not isinstance(kind, str) or not kind.isidentifier():
            raise UnsupportedInputError(f"kind must be a plain name; got {kind!r}.")
        extra = f", kind={kind!r}"
    source = (
        f"plot({_expression_source(arguments)}{_range_source(arguments)}"
        f"{extra}, verbose=False)"
    )
    return _plot(_evaluate(source))


def _run_table(arguments: Mapping[str, Any]) -> dict[str, Any]:
    rows = arguments.get("rows", 10)
    try:
        count = int(rows)
    except (TypeError, ValueError) as error:
        raise UnsupportedInputError(f"rows must be a number; got {rows!r}.") from error
    count = max(2, min(count, _MAX_ROWS))
    source = (
        f"table({_expression_source(arguments)}{_range_source(arguments)}"
        f", rows={count})"
    )
    return _table(_evaluate(source))


_EXPRESSION_PROPERTY: dict[str, Any] = {
    "type": "string",
    "description": (
        "The expression, written as MathSlate/SymPy source — 'sin(x)/x', "
        "'x**3 - 3*x', 'exp(-x**2)'. The symbols x, y, z, t, n, k and theta "
        "already exist. Do not quote it twice and do not write '=' for equality; "
        "use Eq(lhs, rhs)."
    ),
}
_RANGE_PROPERTY: dict[str, Any] = {
    "type": "array",
    "description": (
        "Optional window as [symbol, lo, hi], e.g. ['x', -10, 10]. Omit to let "
        "MathSlate choose one from the expression."
    ),
    "minItems": 3,
    "maxItems": 3,
}


TOOLS: tuple[Tool, ...] = (
    Tool(
        name="mathslate_analyze",
        description=(
            "Compute the properties of a single real function y = f(x): roots, "
            "maxima, minima, inflection points, discontinuities, asymptotes, "
            "symmetry, period and monotonic intervals. Solved symbolically with "
            "SymPy where possible and by sampling where not; every property "
            "reports 'approximate' and the 'method' that produced it, so state "
            "exact results plainly and hedge approximate ones. Use this instead "
            "of working the answer out yourself."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": _EXPRESSION_PROPERTY,
                "range": _RANGE_PROPERTY,
            },
            "required": ["expression"],
        },
        run=_run_analyze,
    ),
    Tool(
        name="mathslate_table",
        description=(
            "Evaluate an expression at evenly spaced points and return the "
            "numbers. Use this to check a value, to see how a function behaves "
            "over a window, or to get data you can quote. A point where the "
            "function is undefined comes back as null rather than being skipped."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": _EXPRESSION_PROPERTY,
                "range": _RANGE_PROPERTY,
                "rows": {
                    "type": "integer",
                    "description": f"How many rows, 2 to {_MAX_ROWS}. Default 10.",
                },
            },
            "required": ["expression"],
        },
        run=_run_table,
    ),
    Tool(
        name="mathslate_plot",
        description=(
            "Build a plot and return what can be said about it: a one-line "
            "summary of what was inferred and sampled, any notes MathSlate "
            "attached (handled discontinuities, clipped views, log-scale "
            "hints), and the equivalent plain NumPy/SymPy/Plotly program. "
            "Returns no image — use it to describe a plot accurately, or to "
            "hand the caller runnable code."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": _EXPRESSION_PROPERTY,
                "range": _RANGE_PROPERTY,
                "kind": {
                    "type": "string",
                    "description": (
                        "Optional plot kind when inference would pick another, "
                        "e.g. 'contour', 'scatter', 'hist'."
                    ),
                },
            },
            "required": ["expression"],
        },
        run=_run_plot,
    ),
)


def tool_names() -> tuple[str, ...]:
    return tuple(tool.name for tool in TOOLS)


def call(name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run one tool call and return a JSON-safe result.

    A failure comes back as ``{"error": ...}`` rather than raising, because the
    caller is an agent mid-conversation: a message it can read and correct is
    worth more than an exception it has to be shielded from. Only MathSlate's own
    errors are turned into that message — anything else is a bug here and is left
    to propagate rather than being disguised as a rejected argument.
    """
    for tool in TOOLS:
        if tool.name == name:
            try:
                return tool.run(arguments or {})
            except MathSlateError as error:
                return {"error": str(error)}
            except (TypeError, ValueError) as error:
                # SymPy and NumPy refuse malformed input this way, and a range
                # of the wrong shape is documented to (manual §4.1).
                return {"error": f"{type(error).__name__}: {error}"}
    raise UnsupportedInputError(
        f"unknown tool {name!r}; MathSlate offers {', '.join(tool_names())}."
    )


def tool_schemas(dialect: str = "anthropic") -> list[dict[str, Any]]:
    """The tool definitions in one provider's shape.

    The two shapes differ only in spelling, so they are generated from one
    description rather than maintained twice: ``anthropic`` names the schema
    ``input_schema``, ``openai`` wraps the same thing in ``{"type": "function"}``.
    """
    if dialect == "anthropic":
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in TOOLS
        ]
    if dialect == "openai":
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in TOOLS
        ]
    raise UnsupportedInputError(
        f"unknown dialect {dialect!r}; tool_schemas() speaks 'anthropic' and 'openai'."
    )
