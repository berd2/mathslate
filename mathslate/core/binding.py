"""Symbol-to-axis binding (PRD 5.2).

Resolution order, in full:

1. an explicit range wins:                ``plot(expr, (x, -10, 10))``
2. symbols bound by ``slider()`` and friends are parameters, never axes
3. remaining symbols sort by convention:  ``x, y, z`` → ``t, u, v`` → ``r, θ``
   → then alphabetically
4. if more remain than are needed, do not guess — *ask*
"""

from __future__ import annotations

from typing import Final, Iterable, Sequence

import sympy as sp

from ..errors import AmbiguousAxisError, UnsupportedInputError

__all__ = [
    "CONVENTIONAL_ORDER",
    "RangeSpec",
    "free_symbols_of",
    "sort_by_convention",
    "choose_symbols",
    "check_ranges",
    "bind_parameter",
    "unbind_parameter",
    "bound_parameters",
    "default_range",
    "parse_range",
]

#: The convention from PRD 5.2 rule 3, most-axis-like first.
CONVENTIONAL_ORDER: Final[tuple[str, ...]] = (
    "x", "y", "z", "t", "u", "v", "r", "theta", "θ", "phi", "φ",
)

#: ``(symbol, lo, hi)`` as the user writes it.
RangeSpec = tuple[sp.Symbol, float, float]

DEFAULT_SPAN: Final[tuple[float, float]] = (-10.0, 10.0)
_TRIG = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)


def free_symbols_of(exprs: Iterable[sp.Expr]) -> set[sp.Symbol]:
    """Union of the free symbols of every expression."""
    found: set[sp.Symbol] = set()
    for expr in exprs:
        found |= {s for s in expr.free_symbols if isinstance(s, sp.Symbol)}
    return found


def sort_by_convention(symbols: Iterable[sp.Symbol]) -> list[sp.Symbol]:
    """Order symbols the way a mathematician would read them."""

    def key(symbol: sp.Symbol) -> tuple[int, str]:
        name = symbol.name
        if name in CONVENTIONAL_ORDER:
            return (CONVENTIONAL_ORDER.index(name), name)
        return (len(CONVENTIONAL_ORDER), name)

    return sorted(symbols, key=key)


def choose_symbols(
    exprs: Sequence[sp.Expr],
    count: int,
    *,
    explicit: Sequence[RangeSpec] = (),
    parameters: Iterable[sp.Symbol] = (),
) -> list[sp.Symbol]:
    """Pick ``count`` axis symbols, or raise :class:`AmbiguousAxisError`."""
    check_ranges(exprs, explicit)
    if len(explicit) >= count:
        return [spec[0] for spec in explicit[:count]]

    bound = {spec[0] for spec in explicit}
    parameter_set = set(parameters)
    candidates = sort_by_convention(free_symbols_of(exprs) - bound - parameter_set)
    chosen = [spec[0] for spec in explicit]
    needed = count - len(chosen)

    if len(candidates) < needed:
        # A constant expression is legal: invent the conventional axis symbols.
        filler = [sp.Symbol(name, real=True) for name in ("x", "y", "z")]
        pool = [s for s in filler if s not in chosen and s not in candidates]
        candidates = candidates + pool[: needed - len(candidates)]
    if len(candidates) > needed:
        names = tuple(s.name for s in candidates)
        axis_word = "axis" if needed == 1 else "axes"
        example = ", ".join(
            f"({name}, -10, 10)" for name in names[:needed]
        )
        question = (
            f"{len(names)} free symbols found ({', '.join(names)}). "
            f"Which {needed} should be the {axis_word}? "
            f"Give an explicit range, e.g. plot(expr, {example})."
        )
        raise AmbiguousAxisError(question, names)
    return chosen + candidates[:needed]


# --------------------------------------------------------------------------
# rule 2: symbols bound by slider() and friends
# --------------------------------------------------------------------------

#: ``symbol -> current value`` for every symbol some UI control has bound.
#: The registry lives here, in ``core``, because this *is* binding (PRD 5.2
#: rule 2). It holds plain floats and knows nothing about widgets, so ``core``
#: stays independent of the frontend as PRD 6.1 requires; ``ui/interact.py``
#: is what puts entries in it.
_BOUND: dict[sp.Symbol, float] = {}


def bind_parameter(symbol: sp.Symbol, value: float) -> None:
    """Record ``symbol`` as a parameter sitting at ``value``."""
    _BOUND[symbol] = float(value)


def unbind_parameter(symbol: sp.Symbol) -> None:
    _BOUND.pop(symbol, None)


def bound_parameters() -> dict[sp.Symbol, float]:
    """Every currently bound symbol. A copy: callers must not mutate ours."""
    return dict(_BOUND)


def check_ranges(exprs: Sequence[sp.Expr], explicit: Sequence[RangeSpec]) -> None:
    """Reject ranges that cannot mean what the user wrote.

    Rule 1 is "an explicit range wins", but it can only win among symbols the
    expression actually has. ``plot(sin(x), (a, -1, 1))`` used to make ``a``
    the axis and then evaluate ``sin(x)`` with ``x`` still free, which yields
    nothing finite and reports itself as an empty domain — an error message
    about the wrong thing entirely.
    """
    seen: set[sp.Symbol] = set()
    for symbol, _lo, _hi in explicit:
        if symbol in seen:
            raise UnsupportedInputError(
                f"two ranges were given for {symbol.name}; give exactly one."
            )
        seen.add(symbol)

    free = free_symbols_of(exprs)
    if not free:
        # A constant draws as a horizontal line, and any axis will do for it.
        return
    for symbol, _lo, _hi in explicit:
        if symbol not in free:
            available = ", ".join(sorted(s.name for s in free))
            raise UnsupportedInputError(
                f"the range names {symbol.name}, which does not appear in what you "
                f"are plotting (its symbols are: {available}). "
                f"Did you mean plot(expr, ({available.split(', ')[0]}, ...))?"
            )


def parse_range(spec: object) -> RangeSpec:
    """Accept ``(symbol, lo, hi)`` and normalise it."""
    if not (isinstance(spec, (tuple, list)) and len(spec) == 3):
        raise TypeError(f"a range must be written (symbol, lo, hi); got {spec!r}")
    symbol, lo, hi = spec
    if not isinstance(symbol, sp.Symbol):
        raise TypeError(f"the first item of a range must be a symbol; got {symbol!r}")
    low, high = float(lo), float(hi)
    if not high > low:
        raise ValueError(f"range for {symbol} is empty: ({low}, {high})")
    return (symbol, low, high)


def default_range(exprs: Sequence[sp.Expr], *, periodic_default: bool = False) -> tuple[float, float]:
    """The range used when the user gives none.

    ``periodic_default`` is set for parametric and polar curves, where one full
    turn is far more useful than a symmetric numeric window.
    """
    if periodic_default and any(expr.has(*_TRIG) for expr in exprs):
        return (0.0, float(2 * sp.pi))
    return DEFAULT_SPAN
