"""Helpers that turn SymPy ``Set`` objects into plain float intervals/points.

``sympy.calculus`` answers questions about domains and singularities with
set algebra (``Union``, ``Complement``, ``ImageSet``). The sampler needs
plain ``float`` intervals, so the translation lives here and nowhere else.
"""

from __future__ import annotations

import sympy as sp

__all__ = ["set_to_intervals", "set_to_points"]

# An ImageSet over Integers is enumerated by walking n; this caps the walk so a
# pathological expression cannot hang the plot.
_MAX_IMAGESET_POINTS: int = 2000


def _finite(value: sp.Expr) -> float | None:
    """Best-effort conversion of a SymPy number to a finite float."""
    try:
        result = float(value.evalf())
    except (TypeError, ValueError, AttributeError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def set_to_points(source: sp.Set, lo: float, hi: float) -> list[float]:
    """Return the isolated real points of ``source`` inside ``[lo, hi]``.

    Intervals of positive measure are ignored: they are not isolated points.
    """
    window = sp.Interval(sp.Float(lo), sp.Float(hi))
    points: list[float] = []
    _collect_points(source, window, points)
    return sorted({p for p in points if lo <= p <= hi})


def _collect_points(source: sp.Set, window: sp.Interval, out: list[float]) -> None:
    if source is sp.S.EmptySet or source is sp.S.Reals:
        return
    if isinstance(source, sp.FiniteSet):
        for element in source.args:
            value = _finite(element)
            if value is not None:
                out.append(value)
        return
    if isinstance(source, sp.Union):
        for part in source.args:
            _collect_points(part, window, out)
        return
    if isinstance(source, sp.Complement):
        # Complement(A, B): only B contributes removed points.
        _collect_points(source.args[1], window, out)
        return
    if isinstance(source, sp.Intersection):
        for part in source.args:
            _collect_points(part, window, out)
        return
    if isinstance(source, sp.ImageSet):
        out.extend(_enumerate_imageset(source, window))
        return
    # Unknown set flavour: try the generic intersection, but only once.
    try:
        narrowed = source.intersect(window)
    except (TypeError, NotImplementedError):
        return
    if isinstance(narrowed, sp.FiniteSet):
        _collect_points(narrowed, window, out)


def _enumerate_imageset(source: sp.ImageSet, window: sp.Interval) -> list[float]:
    """Enumerate ``{f(n) : n in Integers}`` restricted to ``window``."""
    try:
        narrowed = source.intersect(window)
    except (TypeError, NotImplementedError):
        narrowed = source
    if isinstance(narrowed, sp.FiniteSet):
        return [v for v in (_finite(e) for e in narrowed.args) if v is not None]

    # Fall back to solving f(n) in [lo, hi] for a linear lambda, the shape
    # produced by every trigonometric singularity set.
    lam = source.lamda
    if len(lam.variables) != 1 or source.base_sets != (sp.S.Integers,):
        return []
    n = lam.variables[0]
    body = sp.expand(lam.expr)
    slope = sp.simplify(body.diff(n))
    if slope.free_symbols or slope == 0:
        return []
    intercept = sp.simplify(body - slope * n)
    lo_f, hi_f = _finite(window.start), _finite(window.end)
    slope_f, intercept_f = _finite(slope), _finite(intercept)
    if None in (lo_f, hi_f, slope_f, intercept_f):
        return []
    assert lo_f is not None and hi_f is not None
    assert slope_f is not None and intercept_f is not None
    edges = sorted(((lo_f - intercept_f) / slope_f, (hi_f - intercept_f) / slope_f))
    first, last = int(edges[0]) - 1, int(edges[1]) + 1
    if last - first > _MAX_IMAGESET_POINTS:
        return []
    values = [intercept_f + slope_f * k for k in range(first, last + 1)]
    return [v for v in values if lo_f <= v <= hi_f]


def set_to_intervals(
    source: sp.Set, lo: float, hi: float
) -> list[tuple[float, float]] | None:
    """Return ``source ∩ [lo, hi]`` as float intervals, or ``None`` if unknown.

    ``None`` means "could not decide" and the caller should fall back to the
    full window; it never means "empty".
    """
    if source is sp.S.Reals:
        return [(lo, hi)]
    try:
        narrowed = source.intersect(sp.Interval(sp.Float(lo), sp.Float(hi)))
    except (TypeError, NotImplementedError):
        return None
    intervals: list[tuple[float, float]] = []
    if not _collect_intervals(narrowed, intervals):
        return None
    return _merge(intervals)


def _collect_intervals(source: sp.Set, out: list[tuple[float, float]]) -> bool:
    if source is sp.S.EmptySet:
        return True
    if isinstance(source, sp.FiniteSet):
        return True  # measure zero: nothing to sample
    if isinstance(source, sp.Interval):
        start, end = _finite(source.start), _finite(source.end)
        if start is None or end is None or end <= start:
            return end is not None and start is not None
        out.append((start, end))
        return True
    if isinstance(source, sp.Union):
        return all(_collect_intervals(part, out) for part in source.args)
    if isinstance(source, sp.Complement):
        base, removed = source.args
        # Isolated removed points are handled as breakpoints elsewhere, so the
        # base interval is a faithful answer here.
        if _is_discrete(removed):
            return _collect_intervals(base, out)
        return False
    return False


def _is_discrete(source: sp.Set) -> bool:
    if isinstance(source, (sp.FiniteSet, sp.ImageSet)):
        return True
    if isinstance(source, sp.Union):
        return all(_is_discrete(part) for part in source.args)
    return False


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
