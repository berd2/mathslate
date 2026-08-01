"""Context-aware axes (PRD 5.4).

* trigonometric functions or ``pi`` present → ticks at multiples of π
* ``exp`` / ``log`` dominant → *suggest* a log scale, never apply it silently
  (a learner reading a log-scaled plot without knowing it is worse off)
* otherwise → plain numeric ticks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import sympy as sp

__all__ = ["PiTicks", "uses_pi_ticks", "pi_ticks", "log_scale_hint", "log_scale_warning"]

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
