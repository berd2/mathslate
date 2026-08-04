"""Context-aware axes (PRD 5.4).

* trigonometric functions or ``pi`` present → ticks at multiples of π
* ``exp`` / ``log`` dominant → *suggest* a log scale, never apply it silently
  (a learner reading a log-scaled plot without knowing it is worse off)
* otherwise → plain numeric ticks

All three answer the same question — *what should this axis be labelled with* —
and :func:`window_ticks` is that question asked again for a window the reader
has zoomed to rather than the one the plot was built over. Ticks chosen once
for the initial domain and never revisited are wrong the moment the reader
scrolls: a π array holds still while the window moves out from under it, and
numeric labels grow a digit per decade of zoom while the space between them
shrinks. Both are decided here, from the visible window and nothing else, so
the renderer and the live sidebar cannot answer them differently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import sympy as sp

__all__ = [
    "PiTicks",
    "uses_pi_ticks",
    "pi_ticks",
    "numeric_tick_limit",
    "window_ticks",
    "log_scale_hint",
    "log_scale_warning",
]

_TRIG_FUNCS: Final[tuple[type[sp.Function], ...]] = (
    sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc,
    sp.asin, sp.acos, sp.atan, sp.acot, sp.asec, sp.acsc,
)
#: Candidate tick spacings, coarse last.
_STEPS: Final[tuple[sp.Expr, ...]] = (
    sp.pi / 4, sp.pi / 2, sp.pi, 2 * sp.pi, 4 * sp.pi, 8 * sp.pi,
)
_MIN_TICKS: Final[int] = 5
_MAX_TICKS: Final[int] = 17

#: Most labels a numeric axis is given before anything narrower applies. Plotly
#: left to itself scales the count with the axis's pixel length and *then*
#: rounds each label to the tick spacing, so a deep zoom asks for the same
#: dozen ticks it always does and hands them nine-digit labels to wear.
_DEFAULT_MAX_TICKS: Final[int] = 11
#: Fewest a window is ever cut down to. Three labelled ticks still say where
#: the window is and how wide; two would be the endpoints and no scale.
_MIN_NUMERIC_TICKS: Final[int] = 3
#: The drawable width of one axis, in pixels, when nothing has measured it.
#: Plotly's own default figure is 700px wide and MathSlate takes 80px of that
#: in margins; the sidebar's figure is wider still, so this errs toward fewer
#: labels than would strictly fit — the failure it exists to prevent is
#: overlap, and being one label short of full is not a failure.
_ASSUMED_AXIS_PIXELS: Final[int] = 620
#: Width of one label character at Plotly's default 12px tick font. Digits are
#: tabular in the default sans face, so this is close to exact for the numerals
#: and generous for ``.`` and ``-``.
_CHAR_PIXELS: Final[float] = 7.5
#: Clear space each label needs beside it before it reads as touching its
#: neighbour rather than merely being near it.
_LABEL_GUTTER_PIXELS: Final[float] = 16.0


@dataclass(frozen=True)
class PiTicks:
    """Tick positions and their π-flavoured labels."""

    values: tuple[float, ...]
    text: tuple[str, ...]


def uses_pi_ticks(exprs: Sequence[sp.Expr]) -> bool:
    """True when the expression is 'about angles' in the ordinary sense."""
    for expr in exprs:
        if not isinstance(expr, sp.Basic):
            continue
        if expr.has(*_TRIG_FUNCS) or expr.has(sp.pi):
            return True
    return False


def pi_ticks(lo: float, hi: float) -> PiTicks | None:
    """Ticks at multiples of π covering ``[lo, hi]``, or ``None`` if no fit."""
    if not hi > lo:
        return None
    for step in _STEPS:
        size = float(step)
        count = (hi - lo) / size
        if _MIN_TICKS <= count <= _MAX_TICKS:
            first = int(np.ceil(lo / size))
            last = int(np.floor(hi / size))
            multiples = list(range(first, last + 1))
            if len(multiples) < 2:
                continue
            values = tuple(m * size for m in multiples)
            text = tuple(_label(sp.Rational(m) * step) for m in multiples)
            return PiTicks(values=values, text=text)
    return None


def _label_width(lo: float, hi: float, ticks: int) -> int:
    """Characters in the widest label ``ticks`` of them across ``[lo, hi]`` need.

    Plotly rounds every label on a linear axis to the tick spacing and offers no
    offset line the way matplotlib does, so a narrow window a long way from
    zero is spelled out in full: ``[4.99999999, 5.00000001]`` is labelled
    ``4.999999995``, not ``5`` plus ``+1e-8`` in the corner. The label is
    therefore as long as the *position* is precise, which is what makes zooming
    in — not out — the direction that overlaps.
    """
    spacing = (hi - lo) / max(ticks, 1)
    decimals = 0 if spacing <= 0 else max(0, math.ceil(-math.log10(spacing)) + 1)
    magnitude = max(abs(lo), abs(hi))
    integer_digits = 1 if magnitude < 1 else int(math.floor(math.log10(magnitude))) + 1
    width = integer_digits + decimals
    if decimals:
        width += 1  # the decimal point
    if lo < 0:
        width += 1  # the minus sign
    return width


def numeric_tick_limit(
    lo: float,
    hi: float,
    *,
    axis_pixels: int = _ASSUMED_AXIS_PIXELS,
    ceiling: int | None = None,
) -> int:
    """How many labels fit across ``[lo, hi]`` without touching each other.

    The count and the label width determine each other — more ticks mean finer
    spacing means longer labels — so this walks down from the ceiling and takes
    the first count whose own labels fit. Terminating and cheap, where solving
    it in closed form would be neither.

    ``ceiling`` is the reader's own ``ticks=`` when they have set one, and it is
    an upper bound rather than a target: asking for 20 labels does not make 20
    of them legible at a zoom where 4 fit.
    """
    top = _DEFAULT_MAX_TICKS if ceiling is None else max(ceiling, _MIN_NUMERIC_TICKS)
    if not hi > lo:
        return top
    for count in range(top, _MIN_NUMERIC_TICKS, -1):
        width = _label_width(lo, hi, count) * _CHAR_PIXELS + _LABEL_GUTTER_PIXELS
        if count * width <= axis_pixels:
            return count
    return _MIN_NUMERIC_TICKS


def window_ticks(
    lo: float,
    hi: float,
    *,
    pi: bool,
    axis_pixels: int = _ASSUMED_AXIS_PIXELS,
    ceiling: int | None = None,
) -> dict[str, object]:
    """A Plotly axis update labelling ``[lo, hi]`` — the window actually shown.

    ``pi`` says whether this axis is one π ticks were chosen for in the first
    place; they are re-fitted to the new window rather than carried over, and
    dropped for plain numbers when no multiple of π lands often enough in it.
    That fall-back is not a failure: below about 4 units there is no π spacing
    a reader wants, and numbers are the better answer.

    Returned as a payload rather than applied, because the two callers apply it
    differently — one to a ``FigureWidget``, one to a plain ``Figure`` — and
    neither should be deciding this for itself. ``tickvals``/``ticktext`` are
    cleared explicitly on the numeric branch: a stale array otherwise outranks
    ``tickmode="auto"`` and the axis keeps the labels it had before the zoom.
    """
    if pi:
        ticks = pi_ticks(lo, hi)
        if ticks is not None:
            return {
                "tickmode": "array",
                "tickvals": list(ticks.values),
                "ticktext": list(ticks.text),
            }
    return {
        "tickmode": "auto",
        "tickvals": None,
        "ticktext": None,
        "nticks": numeric_tick_limit(lo, hi, axis_pixels=axis_pixels, ceiling=ceiling),
    }


def _label(value: sp.Expr) -> str:
    """Render ``k·π`` as ``-3π/2``, ``π``, ``0`` … the way a textbook does."""
    value = sp.nsimplify(value, [sp.pi])
    if value == 0:
        return "0"
    coefficient = sp.simplify(value / sp.pi)
    if not coefficient.is_Rational:
        return f"{float(value):.3g}"
    numerator, denominator = coefficient.as_numer_denom()
    sign = "-" if numerator < 0 else ""
    numerator = abs(numerator)
    head = "π" if numerator == 1 else f"{numerator}π"
    if denominator == 1:
        return f"{sign}{head}"
    return f"{sign}{head}/{denominator}"


def log_scale_hint(exprs: Sequence[sp.Expr], y: np.ndarray) -> str | None:
    """Suggest — never impose — a log y-scale."""
    dominant = any(
        isinstance(e, sp.Basic) and e.has(sp.exp, sp.log) for e in exprs
    )
    if not dominant:
        return None
    finite = y[np.isfinite(y) & (y > 0)]
    if finite.size < 2:
        return None
    # Subtract logarithms instead of dividing first: max/min can overflow even
    # though each endpoint is a perfectly valid finite float.
    decades = float(np.log10(finite.max()) - np.log10(finite.min()))
    if decades < 3.0:
        return None
    return (
        f"y spans about {decades:.0f} orders of magnitude — "
        "a log scale may read better: plot(..., yscale='log')"
    )


def log_scale_warning(log_y: bool, y: np.ndarray) -> str | None:
    """Say when ``yscale="log"`` is about to hide most of the curve.

    A log axis cannot show zero or negative values; Plotly drops them without
    comment, so ``plot(sin(x), yscale="log")`` draws half a curve and explains
    nothing. The option is still honoured — the user asked for it — but the
    consequence is stated.
    """
    if not log_y:
        return None
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return None
    hidden = int(np.count_nonzero(finite <= 0.0))
    if hidden == 0:
        return None
    share = 100.0 * hidden / finite.size
    return (
        f"yscale='log' cannot show zero or negative values, so {share:.0f}% of "
        "this curve is not drawn. Use yscale='linear' to see all of it."
    )
