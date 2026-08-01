"""Adaptive sampling, singularity detection and domain handling.

This module is the technical heart of MathSlate (PRD 5.3). The rest of the
package is convenience; correctness here is what separates the project from a
thin plotting wrapper.

The algorithm, in the order the PRD specifies it:

1. symbolic singularity detection  (``sympy.calculus.singularities``)
2. real domain computation         (``sympy.calculus.util.continuous_domain``)
3. adaptive subdivision            (angle criterion, bounded depth/points)
4. line breaking                   (NaN inserted at every discontinuity)
5. y-axis clipping                 (2nd-98th percentile, poles excluded)
6. vectorised evaluation           (with a *loud* element-wise fallback)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Final, Sequence

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from sympy.calculus.singularities import singularities
from sympy.calculus.util import continuous_domain

from ..errors import SamplingError
from ._budget import SymbolicTimeout, get_budget, within_budget
from ._failure import EVALUATION_FAILURE, SYMBOLIC_FAILURE
from ._sets import set_to_intervals, set_to_points

__all__ = [
    "SamplingConfig",
    "DomainInfo",
    "SampleResult",
    "NumericFunction",
    "describe_domain",
    "describe_parametric_domain",
    "sample_expression",
    "sample_parametric",
    "sample_callable",
    "detect_jumps",
]

Array = NDArray[np.float64]
#: ``evaluate(t) -> (x, y)`` — the only thing the refiner needs to know.
CurveEvaluator = Callable[[Array], tuple[Array, Array]]

_TINY: Final[float] = 1e-300


@dataclass(frozen=True)
class SamplingConfig:
    """Tunables for the adaptive sampler. Defaults follow PRD 5.3."""

    initial_points: int = 200
    max_depth: int = 8
    max_points: int = 5000
    #: Interior angle (degrees) below which a triple is considered "curved".
    angle_threshold: float = 177.5
    #: Percentiles used for y-axis clipping.
    clip_percentiles: tuple[float, float] = (2.0, 98.0)
    #: Largest number of numeric jump probes per curve.
    max_jump_probes: int = 200
    #: Break points the caller named, in axis units, added to the detected ones.
    #:
    #: PRD 5.3 detects discontinuities automatically and that is the package's
    #: differentiator — but automatic detection with no override is a black box
    #: the moment it is wrong, and PRD 4 asks for escape-hatch completeness. This
    #: is the hatch. `Exclusions` in Mathematica is the same idea.
    exclusions: tuple[float, ...] = ()
    #: Whether the numeric jump probe runs at all.
    #:
    #: False leaves symbolic singularities in place — those come from SymPy and
    #: are not guesses — and switches off only the bisection search of PRD 5.3
    #: step 4. For a curve that is genuinely continuous but steep enough to look
    #: otherwise, that is the difference between one line and several.
    probe_jumps: bool = True

    @property
    def deviation_radians(self) -> float:
        return float(np.deg2rad(180.0 - self.angle_threshold))


DEFAULT_CONFIG: Final[SamplingConfig] = SamplingConfig()


@dataclass(frozen=True)
class DomainInfo:
    """What SymPy could tell us about an expression before any sampling."""

    intervals: tuple[tuple[float, float], ...]
    singular_points: tuple[float, ...]
    domain_known: bool
    notes: tuple[str, ...] = ()


@dataclass
class SampleResult:
    """Sampled curve data plus everything the renderer and codegen need."""

    x: Array
    y: Array
    #: The third coordinate of a 3D space curve, or ``None``.
    z: Array | None = None
    t: Array | None = None
    breakpoints: tuple[float, ...] = ()
    domain_intervals: tuple[tuple[float, float], ...] = ()
    y_range: tuple[float, float] | None = None
    #: Set only where x is not the parameter — a parametric or polar curve can
    #: run away horizontally as well as vertically.
    x_range: tuple[float, float] | None = None
    vectorized: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_points(self) -> int:
        return int(self.x.size)

    @property
    def finite_count(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.x) & np.isfinite(self.y)))


# --------------------------------------------------------------------------
# 6. vectorised evaluation
# --------------------------------------------------------------------------


class NumericFunction:
    """A lambdified expression that degrades loudly, never silently."""

    def __init__(self, expr: sp.Expr, symbol: sp.Symbol) -> None:
        self.expr: sp.Expr = expr
        self.symbol: sp.Symbol = symbol
        self._vector = sp.lambdify(symbol, expr, modules="numpy")
        self._scalar = sp.lambdify(symbol, expr, modules="math")
        #: The mpmath callable, built on first need. ``False`` means "asked for
        #: and unavailable", which is distinct from ``None`` = "not asked yet".
        self._mp: Callable[[float], object] | None | bool = None
        self.vectorized: bool = True
        self.notes: tuple[str, ...] = ()

    def __call__(self, values: Array) -> Array:
        if self.vectorized:
            try:
                return _as_real(self._vector(values), values.shape)
            except EVALUATION_FAILURE as exc:  # any failure here means fall back to element-wise
                self.vectorized = False
                self.notes = (
                    f"vectorised evaluation failed ({type(exc).__name__}: {exc}); "
                    "falling back to element-wise evaluation, which is slower.",
                )
                warnings.warn(self.notes[0], RuntimeWarning, stacklevel=2)
        return self._elementwise(values)

    def _elementwise(self, values: Array) -> Array:
        out = np.full(values.shape, np.nan, dtype=np.float64)
        flat_in = values.ravel()
        flat_out = out.ravel()
        for index, value in enumerate(flat_in):
            for candidate in (self._scalar, self._vector, self._mpmath, self._exact):
                try:
                    flat_out[index] = _scalar_real(candidate(value))
                    break
                except EVALUATION_FAILURE:  # noqa: S112 - the point is simply undefined
                    continue
        return flat_out.reshape(values.shape)

    def _mpmath(self, value: float) -> object:
        """mpmath, which knows the special functions and is not slow about it.

        NumPy and ``math`` do not carry ``zeta``, ``Si`` or ``besselj``, so an
        expression built from them lands in this element-wise loop with only
        :meth:`_exact` left — and ``subs().evalf()`` rebuilds and re-evaluates
        the whole expression tree per point. `analyze(besselj(2, x) - 1/x)` spent
        **66 of its 68 seconds** there, across 12942 calls, which reads to the
        user as a hung notebook.

        mpmath is what ``evalf`` calls underneath, so going to it directly costs
        nothing in accuracy and skips the tree walk. Measured per 400 points:
        ``besselj`` 0.270 s → 0.006 s, ``Si`` 0.350 s → 0.004 s, ``gamma``
        0.108 s → 0.004 s, ``zeta`` 0.310 s → 0.159 s. It is already a hard
        dependency of SymPy, so this adds nothing to install.
        """
        if self._mp is None:
            # Once per function, not once per point: `lambdify` raises for a head
            # mpmath cannot print, and that verdict is cached too.
            try:
                self._mp = sp.lambdify(self.symbol, self.expr, modules="mpmath")
            except EVALUATION_FAILURE:
                self._mp = False
        if self._mp is False:
            raise TypeError("no mpmath callable for this expression")
        return self._mp(value)  # type: ignore[operator]

    def _exact(self, value: float) -> object:
        """Last resort: ask SymPy itself.

        Reached when even mpmath has no printer for a head. Slow, and correct,
        which is the right order of priorities for something this rare.
        """
        return self.expr.subs(self.symbol, sp.Float(value)).evalf()


def _announce(*functions: "NumericFunction") -> None:
    """Re-raise the fallback warning where the caller can actually hear it.

    Sampling runs inside ``simplefilter("ignore", RuntimeWarning)`` — it has
    to, or NumPy's overflow and divide-by-zero chatter would bury every plot of
    ``tan(x)``. That block also swallows the one warning that matters, so the
    degraded-to-element-wise notice is emitted again out here. PRD 5.3 step 6:
    loudly, never silently.
    """
    for function in functions:
        if not function.vectorized and function.notes:
            warnings.warn(function.notes[0], RuntimeWarning, stacklevel=3)


def _as_real(raw: object, shape: tuple[int, ...]) -> Array:
    """Coerce a lambdify result to a real float array, NaN where non-real."""
    array = np.asarray(raw)
    if array.shape != shape:
        array = np.broadcast_to(array, shape)
    if np.iscomplexobj(array):
        real = np.real(array).astype(np.float64)
        imaginary = np.abs(np.imag(array))
        scale = np.maximum(np.abs(real), 1.0)
        real = np.where(imaginary > 1e-9 * scale, np.nan, real)
    else:
        real = array.astype(np.float64)
    # Plotly serializes non-finite numbers as null eventually, but normalizing
    # them here prevents overflow warnings and keeps every downstream numeric
    # algorithm on the same explicit "not drawable" representation.
    return np.where(np.isfinite(real), real, np.nan)


def _scalar_real(raw: object) -> float:
    value = complex(raw)  # type: ignore[arg-type]
    if abs(value.imag) > 1e-9 * max(abs(value.real), 1.0):
        return float("nan")
    real = float(value.real)
    return real if np.isfinite(real) else float("nan")


# --------------------------------------------------------------------------
# 1 + 2. singularities and domain
# --------------------------------------------------------------------------


def describe_domain(expr: sp.Expr, symbol: sp.Symbol, lo: float, hi: float) -> DomainInfo:
    """Ask SymPy where the expression is real and where it blows up."""
    notes: list[str] = []
    points: list[float] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Both questions get a wall-clock budget. Neither is slow on anything in
        # the corpus — the worst measured `plot()` is 1.7 s end to end — but
        # `singularities` is one `solveset` under the skin, and the point of a
        # budget is the input nobody thought of.
        try:
            singular = within_budget(singularities, expr, symbol, sp.S.Reals)
        except SymbolicTimeout:
            singular = sp.S.EmptySet
            notes.append(
                f"symbolic singularity detection exceeded its {get_budget():g}s "
                "budget; using numeric probes only."
            )
        except SYMBOLIC_FAILURE:
            singular = sp.S.EmptySet
            notes.append("symbolic singularity detection failed; using numeric probes only.")
        else:
            points = set_to_points(singular, lo, hi)

        intervals: list[tuple[float, float]] | None
        try:
            domain = within_budget(continuous_domain, expr, symbol, sp.S.Reals)
        except SymbolicTimeout:
            intervals = None
            notes.append(
                f"real-domain detection exceeded its {get_budget():g}s budget; "
                "the window was sampled whole."
            )
        except SYMBOLIC_FAILURE:
            intervals = None
        else:
            intervals = set_to_intervals(domain, lo, hi)

    domain_known = intervals is not None
    if not domain_known:
        intervals = [(lo, hi)]
    assert intervals is not None

    intervals = _split_at(intervals, points)
    return DomainInfo(
        intervals=tuple(intervals),
        singular_points=tuple(points),
        domain_known=domain_known,
        notes=tuple(notes),
    )


def describe_parametric_domain(
    components: Sequence[sp.Expr], symbol: sp.Symbol, lo: float, hi: float
) -> DomainInfo:
    """The same question as :func:`describe_domain`, for ``(x(t), y(t))``.

    A point of the curve is drawable only where *both* components are, so the
    continuous pieces intersect and the singular points union. Without this a
    parametric or polar curve gets none of PRD 5.3 — which is how
    ``plot((tan(t), t))`` came to draw a line out to 10^16.
    """
    parts = [describe_domain(expr, symbol, lo, hi) for expr in components]
    intervals: list[tuple[float, float]] = [(lo, hi)]
    for part in parts:
        intervals = _intersect(intervals, list(part.intervals))
    points = sorted({p for part in parts for p in part.singular_points})
    return DomainInfo(
        intervals=tuple(_split_at(intervals, points)),
        singular_points=tuple(points),
        domain_known=all(part.domain_known for part in parts),
        notes=tuple(dict.fromkeys(note for part in parts for note in part.notes)),
    )


def _intersect(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Overlapping portions of two sorted interval lists."""
    out: list[tuple[float, float]] = []
    for a_lo, a_hi in left:
        for b_lo, b_hi in right:
            start, end = max(a_lo, b_lo), min(a_hi, b_hi)
            if end > start:
                out.append((start, end))
    return sorted(out)


def _split_at(
    intervals: list[tuple[float, float]], points: list[float]
) -> list[tuple[float, float]]:
    """Cut every interval open at each singular point inside it."""
    result: list[tuple[float, float]] = []
    for start, end in intervals:
        inside = sorted(p for p in points if start < p < end)
        cursor = start
        for point in inside:
            if point - cursor > (end - start) * 1e-12:
                result.append((cursor, point))
            cursor = point
        if end - cursor > (end - start) * 1e-12:
            result.append((cursor, end))
    return result


# --------------------------------------------------------------------------
# 3. adaptive subdivision
# --------------------------------------------------------------------------


def _robust_span(values: Array, percentiles: tuple[float, float]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    low = float(np.percentile(finite, percentiles[0]))
    high = float(np.percentile(finite, percentiles[1]))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return (low - 0.5, low + 0.5)
    return (low, high)


def refine(
    evaluate: CurveEvaluator,
    params: Array,
    config: SamplingConfig = DEFAULT_CONFIG,
) -> tuple[Array, Array, Array]:
    """Adaptively subdivide ``params`` until the curve looks smooth.

    Returns ``(params, x, y)`` sorted by ``params``.
    """
    x, y = evaluate(params)
    for _ in range(config.max_depth):
        if params.size >= config.max_points or params.size < 3:
            break
        targets = _segments_to_split(x, y, config)
        if targets.size == 0:
            break
        budget = config.max_points - params.size
        if targets.size > budget:
            targets = targets[:budget]
        midpoints = 0.5 * (params[targets] + params[targets + 1])
        midpoints = midpoints[np.isfinite(midpoints)]
        if midpoints.size == 0:
            break
        mid_x, mid_y = evaluate(midpoints)
        params = np.concatenate([params, midpoints])
        x = np.concatenate([x, mid_x])
        y = np.concatenate([y, mid_y])
        order = np.argsort(params, kind="stable")
        params, x, y = params[order], x[order], y[order]
    return params, x, y


def _segments_to_split(x: Array, y: Array, config: SamplingConfig) -> NDArray[np.int64]:
    """Indices ``i`` whose segment ``[i, i+1]`` deserves a midpoint."""
    finite = np.isfinite(x) & np.isfinite(y)
    x_lo, x_hi = _robust_span(x, config.clip_percentiles)
    y_lo, y_hi = _robust_span(y, config.clip_percentiles)
    x_span = max(x_hi - x_lo, _TINY)
    y_span = max(y_hi - y_lo, _TINY)
    nx = x / x_span
    ny = np.clip(y, y_lo - 4.0 * y_span, y_hi + 4.0 * y_span) / y_span

    flagged = np.zeros(x.size - 1, dtype=bool)

    # (a) a segment straddling the edge of the domain: resolve the boundary.
    edge = finite[:-1] != finite[1:]
    flagged |= edge

    # (b) curvature: the interior angle of three adjacent points.
    if x.size >= 3:
        usable = finite[:-2] & finite[1:-1] & finite[2:]
        ax = nx[1:-1] - nx[:-2]
        ay = ny[1:-1] - ny[:-2]
        bx = nx[2:] - nx[1:-1]
        by = ny[2:] - ny[1:-1]
        len_a = np.hypot(ax, ay)
        len_b = np.hypot(bx, by)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = (ax * bx + ay * by) / (len_a * len_b)
        deviation = np.arccos(np.clip(cosine, -1.0, 1.0))
        curved = usable & np.isfinite(deviation) & (deviation > config.deviation_radians)
        # A curved triple splits both of its segments.
        flagged[:-1] |= curved
        flagged[1:] |= curved

    return np.flatnonzero(flagged).astype(np.int64)


# --------------------------------------------------------------------------
# 4. line breaking (numeric jump detection + NaN insertion)
# --------------------------------------------------------------------------


def detect_jumps(
    evaluate: CurveEvaluator,
    params: Array,
    x: Array,
    y: Array,
    known: tuple[float, ...],
    config: SamplingConfig,
    axes: tuple[int, ...] = (1,),
) -> list[float]:
    """Find true discontinuities that SymPy did not report (PRD 5.3 step 4).

    A steep-but-continuous segment shrinks its own jump as the interval
    shrinks; a genuine discontinuity does not. Bisection tells them apart, so
    ``floor``, ``sign`` and ``Piecewise`` are handled without special cases.

    Public because :mod:`mathslate.core.analysis` needs the same probe to
    report discontinuities that ``singularities()`` cannot see. Two core
    modules sharing one algorithm is fine; reaching across the boundary for a
    name spelled as private was not.

    ``axes`` says which components to watch: ``(1,)`` — y alone — for an
    explicit curve, where x is the parameter and cannot jump, and ``(0, 1)``
    for a parametric curve, where a jump in either coordinate breaks the line.

    Returns nothing when ``config.probe_jumps`` is off. The switch lives here so
    that callers need not restate it — and so a new caller cannot forget to.
    """
    if not config.probe_jumps:
        return []
    found: list[float] = []
    for axis in axes:
        seen = known + tuple(found)
        found.extend(_jumps_along(evaluate, params, x, y, axis, seen, config))
    return sorted(found)


def _jumps_along(
    evaluate: CurveEvaluator,
    params: Array,
    x: Array,
    y: Array,
    axis: int,
    known: tuple[float, ...],
    config: SamplingConfig,
) -> list[float]:
    """Jumps visible in one coordinate. A point is drawn only if both are finite."""
    values = (x, y)[axis]
    finite = np.isfinite(x) & np.isfinite(y)
    pair = finite[:-1] & finite[1:]
    if not pair.any():
        return []
    lo, hi = _robust_span(values, config.clip_percentiles)
    threshold = max((hi - lo) * 0.01, _TINY)
    with np.errstate(invalid="ignore"):
        # inf - inf is nan here, which simply means "not a candidate".
        gaps = np.where(pair, np.abs(np.diff(values)), 0.0)
    candidates = np.flatnonzero(gaps > threshold)
    if candidates.size == 0:
        return []
    candidates = candidates[np.argsort(gaps[candidates])[::-1]][: config.max_jump_probes]

    tolerance = float(np.ptp(params)) * 1e-12 if params.size else 0.0
    found: list[float] = []
    for index in candidates:
        left, right = float(params[index]), float(params[index + 1])
        if any(left <= k <= right for k in known):
            continue
        location = _probe_jump(evaluate, left, right, tolerance, axis)
        if location is not None:
            found.append(location)
    return sorted(found)


def _probe_jump(
    evaluate: CurveEvaluator, left: float, right: float, tolerance: float, axis: int = 1
) -> float | None:
    """Bisect ``[left, right]`` and report where the curve really jumps.

    ``axis`` picks the coordinate being watched — 1 for y, 0 for the x of a
    parametric curve — so the values below are that coordinate, not always y.
    """

    def value_at(param: float) -> float:
        pair = evaluate(np.array([param], dtype=np.float64))
        return float(pair[axis][0])

    at_left, at_right = value_at(left), value_at(right)
    if not (np.isfinite(at_left) and np.isfinite(at_right)):
        return 0.5 * (left + right)
    initial_gap = abs(at_right - at_left)
    if initial_gap == 0.0:
        return None

    for _ in range(60):
        if right - left <= tolerance:
            break
        middle = 0.5 * (left + right)
        at_middle = value_at(middle)
        if not np.isfinite(at_middle):
            return middle
        if abs(at_middle - at_left) >= abs(at_right - at_middle):
            right, at_right = middle, at_middle
        else:
            left, at_left = middle, at_middle

    remaining = abs(at_right - at_left)
    if remaining >= 0.5 * initial_gap:
        return 0.5 * (left + right)
    return None


def _insert_breaks(
    evaluate: CurveEvaluator,
    params: Array,
    x: Array,
    y: Array,
    breakpoints: list[float],
    span: float,
    *,
    cut_x: bool = False,
) -> tuple[Array, Array, Array]:
    """Insert ``x = b-δ, NaN, b+δ`` at every breakpoint so lines are cut.

    ``cut_x`` blanks the horizontal coordinate too. For an explicit curve x is
    the parameter and is perfectly well defined at the break, but on a
    parametric curve the break may be a jump *in x*, and leaving a real value
    there would let the line be drawn straight across it.
    """
    if not breakpoints:
        return params, x, y
    delta = max(span, _TINY) * 1e-9
    extra = np.array(
        [p + sign * delta for p in breakpoints for sign in (-1.0, 1.0)], dtype=np.float64
    )
    ex, ey = evaluate(extra)
    gap_params = np.array(breakpoints, dtype=np.float64)
    if cut_x:
        gap_x = np.full(gap_params.shape, np.nan, dtype=np.float64)
    else:
        gap_x, _ = evaluate(gap_params)
    gap_y = np.full(gap_params.shape, np.nan, dtype=np.float64)

    params = np.concatenate([params, extra, gap_params])
    x = np.concatenate([x, ex, gap_x])
    y = np.concatenate([y, ey, gap_y])
    order = np.argsort(params, kind="stable")
    return params[order], x[order], y[order]


# --------------------------------------------------------------------------
# 5. y-axis clipping
# --------------------------------------------------------------------------


def _uniform_subset(params: Array, values: Array, count: int = 400) -> Array:
    """Values read off an evenly spaced grid, ignoring sampling density."""
    if params.size <= count:
        return values
    order = np.argsort(params, kind="stable")
    ordered_params, ordered_values = params[order], values[order]
    targets = np.linspace(float(ordered_params[0]), float(ordered_params[-1]), count)
    index = np.unique(
        np.clip(np.searchsorted(ordered_params, targets), 0, ordered_params.size - 1)
    )
    return ordered_values[index]


def _clip_range(
    params: Array,
    values: Array,
    breakpoints: Sequence[float],
    span: float,
    config: SamplingConfig,
    force: bool,
) -> tuple[float, float] | None:
    """Pick the visible window from the 2nd-98th percentile of the samples.

    Samples in the immediate neighbourhood of a pole are excluded first: they
    are exactly the values that would otherwise flatten the whole curve.

    Proximity to a pole and the uniform re-read are both measured in
    ``params``, the sampling parameter. For an explicit curve that *is* x; for
    a parametric curve it is t, and using x there would be wrong twice over —
    x is not monotone, and the breakpoints are values of t.
    """
    usable = np.isfinite(values) & np.isfinite(params)
    keep = usable
    if breakpoints:
        guard = max(abs(span), _TINY) * 0.02
        for point in breakpoints:
            keep = keep & (np.abs(params - point) > guard)
        if np.count_nonzero(keep) < 4:
            keep = usable
    if np.count_nonzero(keep) < 4:
        return None
    # Adaptive refinement piles points up near poles, which would drag the
    # percentiles with it; re-read the curve on a uniform grid first.
    sampled = _uniform_subset(params[keep], values[keep])
    sampled = sampled[np.isfinite(sampled)]
    if sampled.size < 4:
        return None
    low, high = _robust_span(sampled, config.clip_percentiles)
    if high <= low:
        return None

    all_finite = values[np.isfinite(values)]
    data_low, data_high = float(all_finite.min()), float(all_finite.max())
    # Only clip when the tails really run away, or when poles are known.
    if not (force or (data_high - data_low) > 8.0 * (high - low)):
        return None

    # The bare percentile window is too tight to read: widen it threefold, then
    # clamp back to data the curve actually reaches so bounded functions such
    # as sin(x)/x are not padded with empty space.
    centre = 0.5 * (low + high)
    reach = 1.5 * (high - low)
    window_low = max(centre - reach, data_low)
    window_high = min(centre + reach, data_high)
    if window_high <= window_low:
        return None
    pad = (window_high - window_low) * 0.05
    return (window_low - pad, window_high + pad)


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------


def sample_expression(
    expr: sp.Expr,
    symbol: sp.Symbol,
    xrange: tuple[float, float],
    config: SamplingConfig = DEFAULT_CONFIG,
) -> SampleResult:
    """Sample ``y = expr(symbol)`` over ``xrange`` following PRD 5.3."""
    lo, hi = float(xrange[0]), float(xrange[1])
    if not hi > lo:
        raise SamplingError(f"empty range: ({lo}, {hi})")

    info = describe_domain(expr, symbol, lo, hi)
    function = NumericFunction(expr, symbol)

    def evaluate(values: Array) -> tuple[Array, Array]:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            return values, function(values)

    if not info.intervals:
        raise SamplingError(
            f"{expr} is not real anywhere on [{lo}, {hi}] — nothing to draw."
        )

    total_span = hi - lo
    all_params: list[Array] = []
    all_x: list[Array] = []
    all_y: list[Array] = []
    breakpoints: list[float] = [p for p in info.singular_points]
    named = _named_exclusions(config, lo, hi)
    breakpoints.extend(named)

    for start, end in info.intervals:
        share = max((end - start) / total_span, 0.02)
        count = max(int(config.initial_points * share), 25)
        inset = (end - start) * 1e-9
        grid = np.linspace(start + inset, end - inset, count, dtype=np.float64)
        params, xs, ys = refine(evaluate, grid, config)
        found = detect_jumps(evaluate, params, xs, ys, tuple(breakpoints), config)
        breakpoints.extend(found)
        cuts = found + [v for v in named if start < v < end]
        params, xs, ys = _insert_breaks(evaluate, params, xs, ys, cuts, end - start)
        all_params.append(params)
        all_x.append(xs)
        all_y.append(ys)

    params, xs, ys = _join_with_gaps(all_params, all_x, all_y)

    notes = list(info.notes) + list(function.notes)
    if not info.domain_known:
        notes.append("SymPy could not determine the real domain; sampled the full range.")
    if info.singular_points:
        listed = ", ".join(_format_number(p) for p in info.singular_points[:6])
        notes.append(f"singularities at x = {listed}")

    result = SampleResult(
        x=xs,
        y=ys,
        t=params,
        breakpoints=tuple(sorted(set(breakpoints))),
        domain_intervals=info.intervals,
        y_range=_clip_range(
            xs,
            ys,
            tuple(sorted(set(breakpoints))),
            hi - lo,
            config,
            force=bool(info.singular_points),
        ),
        vectorized=function.vectorized,
        notes=tuple(notes),
    )
    if result.finite_count == 0:
        raise SamplingError(
            f"{expr} produced no finite values on [{lo}, {hi}] — nothing to draw."
        )
    _announce(function)
    return result


def sample_parametric(
    components: tuple[sp.Expr, sp.Expr],
    symbol: sp.Symbol,
    trange: tuple[float, float],
    config: SamplingConfig = DEFAULT_CONFIG,
    third: sp.Expr | None = None,
) -> SampleResult:
    """Sample a 2D parametric curve ``(x(t), y(t))``, per PRD 5.3.

    ``third`` makes it a 3D space curve. The extra component rides along on the
    parameter grid the first two chose: it shares their domain and their cuts,
    so a pole in any component breaks all three together, which is the only
    way a broken space curve stays one curve.

    Everything an explicit curve gets, a parametric curve gets too: the real
    domain of both components, symbolic poles, the bisection jump probe, NaN
    cuts and a readable window. The one addition is that the window is clipped
    horizontally as well — for ``y = f(x)`` the x extent is the range the user
    asked for, but ``x(t)`` can run away just as ``y(t)`` can.
    """
    lo, hi = float(trange[0]), float(trange[1])
    if not hi > lo:
        raise SamplingError(f"empty range: ({lo}, {hi})")

    parts = (*components, third) if third is not None else components
    info = describe_parametric_domain(parts, symbol, lo, hi)
    fx = NumericFunction(components[0], symbol)
    fy = NumericFunction(components[1], symbol)

    def evaluate(values: Array) -> tuple[Array, Array]:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            return fx(values), fy(values)

    if not info.intervals:
        raise SamplingError(
            f"({components[0]}, {components[1]}) is not real anywhere on "
            f"[{lo}, {hi}] — nothing to draw."
        )

    total_span = hi - lo
    all_params: list[Array] = []
    all_x: list[Array] = []
    all_y: list[Array] = []
    breakpoints: list[float] = list(info.singular_points)
    named = _named_exclusions(config, lo, hi)
    breakpoints.extend(named)

    for start, end in info.intervals:
        share = max((end - start) / total_span, 0.02)
        count = max(int(config.initial_points * share), 25)
        inset = (end - start) * 1e-9
        grid = np.linspace(start + inset, end - inset, count, dtype=np.float64)
        params, xs, ys = refine(evaluate, grid, config)
        # Either coordinate jumping breaks the line, so both are watched.
        found = detect_jumps(
            evaluate, params, xs, ys, tuple(breakpoints), config, axes=(0, 1)
        )
        breakpoints.extend(found)
        cuts = found + [v for v in named if start < v < end]
        params, xs, ys = _insert_breaks(
            evaluate, params, xs, ys, cuts, end - start, cut_x=True
        )
        all_params.append(params)
        all_x.append(xs)
        all_y.append(ys)

    params, xs, ys = _join_with_gaps(all_params, all_x, all_y)
    cuts = tuple(sorted(set(breakpoints)))
    force = bool(info.singular_points)

    notes = list(info.notes) + list(fx.notes) + list(fy.notes)
    if not info.domain_known:
        notes.append("SymPy could not determine the real domain; sampled the full range.")
    if info.singular_points:
        listed = ", ".join(_format_number(p) for p in info.singular_points[:6])
        notes.append(f"singularities at {symbol.name} = {listed}")

    z: Array | None = None
    if third is not None:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            z = NumericFunction(third, symbol)(params)
        z = np.where(np.isfinite(xs) & np.isfinite(ys), z, np.nan)

    result = SampleResult(
        x=xs,
        y=ys,
        z=z,
        t=params,
        breakpoints=cuts,
        domain_intervals=info.intervals,
        y_range=_clip_range(params, ys, cuts, total_span, config, force=force),
        x_range=_clip_range(params, xs, cuts, total_span, config, force=force),
        vectorized=fx.vectorized and fy.vectorized,
        notes=tuple(_unique(notes)),
    )
    if result.finite_count == 0:
        raise SamplingError("the parametric curve produced no finite points.")
    _announce(fx, fy)
    return result


def sample_callable(
    function: Callable[[float], float],
    xrange: tuple[float, float],
    config: SamplingConfig = DEFAULT_CONFIG,
) -> SampleResult:
    """Sample a plain Python callable numerically."""
    lo, hi = float(xrange[0]), float(xrange[1])

    def evaluate(values: Array) -> tuple[Array, Array]:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                out = np.asarray(function(values), dtype=np.float64)  # type: ignore[arg-type]
                if out.shape != values.shape:
                    raise ValueError("not vectorised")
            except EVALUATION_FAILURE:
                out = np.array(
                    [_safe_scalar(function, float(v)) for v in values], dtype=np.float64
                )
            return values, out

    grid = np.linspace(lo, hi, config.initial_points, dtype=np.float64)
    params, xs, ys = refine(evaluate, grid, config)
    named = _named_exclusions(config, lo, hi)
    found = named + detect_jumps(evaluate, params, xs, ys, tuple(named), config)
    params, xs, ys = _insert_breaks(evaluate, params, xs, ys, found, hi - lo)
    return SampleResult(
        x=xs,
        y=ys,
        t=params,
        breakpoints=tuple(found),
        domain_intervals=((lo, hi),),
        y_range=_clip_range(xs, ys, tuple(found), hi - lo, config, force=False),
    )


def _named_exclusions(config: SamplingConfig, lo: float, hi: float) -> list[float]:
    """The caller's break points, keeping only the ones on this window.

    A value outside the window is not an error — a reader who says
    ``exclusions=[0]`` and then narrows the range has not made a mistake, and
    refusing the plot over it would be pedantry.
    """
    return [float(v) for v in config.exclusions if lo < float(v) < hi]


def _safe_scalar(function: Callable[[float], float], value: float) -> float:
    try:
        return float(function(value))
    except EVALUATION_FAILURE:  # an undefined point is not an error
        return float("nan")


def _join_with_gaps(
    params: list[Array], xs: list[Array], ys: list[Array]
) -> tuple[Array, Array, Array]:
    """Concatenate per-interval samples with a NaN separator between them."""
    if len(params) == 1:
        return params[0], xs[0], ys[0]
    gap = np.array([np.nan], dtype=np.float64)
    joined_p: list[Array] = []
    joined_x: list[Array] = []
    joined_y: list[Array] = []
    for index, (p, x, y) in enumerate(zip(params, xs, ys)):
        if index:
            joined_p.append(gap)
            joined_x.append(gap)
            joined_y.append(gap)
        joined_p.append(p)
        joined_x.append(x)
        joined_y.append(y)
    return (
        np.concatenate(joined_p),
        np.concatenate(joined_x),
        np.concatenate(joined_y),
    )


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4g}"
