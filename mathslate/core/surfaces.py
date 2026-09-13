"""Sampling over two variables — surfaces, contours and implicit curves.

The one-variable sampler (:mod:`mathslate.core.sampling`) can ask SymPy exactly
where a curve is real and exactly where it blows up, and it does. Neither
question has a usable answer in two variables: ``continuous_domain`` takes a
single symbol, and there is no two-variable ``singularities``. So this module
does what can be done honestly instead — evaluate on a grid, turn everything
non-real or non-finite into ``NaN`` so Plotly leaves a hole rather than drawing
a wrong surface, and clip the colour range so one pole cannot flatten the rest.

That is a real reduction in guarantees compared with 2D, and it is stated in
the notes rather than glossed over.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Final

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from ..errors import SamplingError
from ._failure import EVALUATION_FAILURE

__all__ = [
    "SurfaceSample",
    "sample_surface",
    "sample_parametric_surface",
    "sample_region",
    "comparison_operands",
    "GRID",
]

Array = NDArray[np.float64]

#: Samples per axis. PRD 5.1's own example reports "60x60 samples".
GRID: Final[int] = 60

#: Samples per axis for a filled region.
#:
#: Higher than :data:`GRID` because the eye judges a region by its *boundary*,
#: and a boundary rendered on a 60-wide grid is visibly a staircase. A surface
#: is judged by its interior, where 60 is plenty. Booleans are also cheaper per
#: point than floats — no clipping, no percentiles — so the extra resolution
#: costs little.
REGION_GRID: Final[int] = 200


@dataclass
class SurfaceSample:
    """A sampled surface. Shaped to stand in for a ``SampleResult``.

    ``x`` and ``y`` are the grid lines and ``z`` is ``(len(y), len(x))`` — the
    orientation Plotly's ``Surface`` and ``Contour`` both expect. For a
    parametric surface all three are ``(m, n)`` and the grid lines are the
    parameters instead.
    """

    x: Array
    y: Array
    z: Array
    #: Present when the surface is parametric, where x and y are 2-D too.
    parametric: bool = False
    vectorized: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)
    z_range: tuple[float, float] | None = None

    # -- the parts of SampleResult the plan and renderer reach for ---------

    #: Two-variable sampling finds no cut locations to report (see module docs).
    breakpoints: tuple[float, ...] = ()
    domain_intervals: tuple[tuple[float, float], ...] = ()
    y_range: tuple[float, float] | None = None
    x_range: tuple[float, float] | None = None
    t: Array | None = None

    @property
    def n_points(self) -> int:
        return int(self.z.size)

    @property
    def finite_count(self) -> int:
        finite = np.isfinite(self.z)
        if self.parametric:
            finite &= np.isfinite(self.x) & np.isfinite(self.y)
        return int(np.count_nonzero(finite))


def _grid_values(
    expr: sp.Expr, symbols: tuple[sp.Symbol, sp.Symbol], xs: Array, ys: Array
) -> tuple[Array, bool, tuple[str, ...]]:
    """Evaluate ``expr`` over the mesh, NaN wherever it is not a real number."""
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    vectorized = True
    notes: tuple[str, ...] = ()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        function = sp.lambdify(symbols, expr, modules="numpy")
        try:
            raw = np.asarray(function(mesh_x, mesh_y))
            if raw.shape != mesh_x.shape:
                raw = np.broadcast_to(raw, mesh_x.shape)
        except EVALUATION_FAILURE:  # any failure here means element-wise
            vectorized = False
            notes = (
                "vectorised evaluation failed; the surface was sampled point by "
                "point, which is slower.",
            )
            point = _PointEvaluator(expr, symbols, function)
            scalar = np.vectorize(
                lambda a, b: _safe(point, a, b), otypes=[np.complex128]
            )
            raw = scalar(mesh_x, mesh_y)

    values = np.asarray(raw)
    if np.iscomplexobj(values):
        real = np.real(values).astype(np.float64)
        scale = np.maximum(np.abs(real), 1.0)
        real = np.where(np.abs(np.imag(values)) > 1e-9 * scale, np.nan, real)
        values = real
    return values.astype(np.float64), vectorized, notes


class _PointEvaluator:
    """One grid point at a time, for an expression NumPy could not take whole.

    The fallback used to call the *same* NumPy-lambdified function point by
    point. That rescues an expression whose only problem was broadcasting, and
    nothing else: when vectorised evaluation failed because NumPy has no
    ``besselj``, ``Si`` or ``zeta``, it failed identically at every one of the
    3600 points, every point became NaN, and ``plot(besselj(0, x*y))`` was
    refused as having "no real values anywhere" — a false statement about a
    function that is real everywhere on that grid. The one-variable sampler
    solved exactly this with an mpmath tier (see
    :meth:`mathslate.core.sampling.NumericFunction._mpmath`); this is the same
    ladder in two variables: NumPy, then mpmath, then SymPy's own ``evalf``.

    A tier that raises ``NameError`` or ``KeyError`` is dropped for the rest of
    the grid. Those mean "this backend has no printer or no function of that
    name", which is true at every point alike, so retrying it 40 000 times on a
    200×200 region grid would only cost time. Anything else — ``log`` of a
    negative number, a division by zero — is about *this* point, and the next
    point still gets every tier.
    """

    def __init__(
        self,
        expr: sp.Basic,
        symbols: tuple[sp.Symbol, sp.Symbol],
        vector: Callable[[float, float], object],
    ) -> None:
        self._expr: sp.Basic = expr
        self._symbols: tuple[sp.Symbol, sp.Symbol] = symbols
        #: The mpmath callable, built on first need. ``False`` means "asked for
        #: and unavailable", distinct from ``None`` = "not asked yet".
        self._mp: Callable[[float, float], object] | None | bool = None
        self._tiers: list[Callable[[float, float], object]] = [
            vector,
            self._mpmath,
            self._exact,
        ]

    def __call__(self, a: float, b: float) -> object:
        for tier in list(self._tiers):
            try:
                return tier(a, b)
            except (NameError, KeyError):
                self._tiers.remove(tier)  # missing at every point, not just this one
            except EVALUATION_FAILURE:  # noqa: S112 - undefined here; try the next tier
                continue
        raise ValueError("no evaluation tier produced a value at this point")

    def _mpmath(self, a: float, b: float) -> object:
        if self._mp is None:
            # Once per grid, not once per point, and the verdict is cached too.
            try:
                self._mp = sp.lambdify(self._symbols, self._expr, modules="mpmath")
            except EVALUATION_FAILURE:
                self._mp = False
        if self._mp is False:
            raise NameError("mpmath has no callable for this expression")
        return self._mp(a, b)  # type: ignore[operator]

    def _exact(self, a: float, b: float) -> object:
        """Last resort, for a head mpmath cannot print. Slow and correct."""
        first, second = self._symbols
        return self._expr.subs({first: sp.Float(a), second: sp.Float(b)}).evalf()


def _safe(point: Callable[[float, float], object], a: float, b: float) -> complex:
    try:
        return complex(point(a, b))  # type: ignore[arg-type]
    except EVALUATION_FAILURE:  # an undefined point is not an error
        return complex("nan")


def _safe_truth(point: Callable[[float, float], object], a: float, b: float) -> bool:
    """A relation's truth value at one point, ``False`` where there is none.

    ``bool(_safe(...))`` was used here before, and a NaN is truthy, so every
    point the relation could not be evaluated at counted as *inside* the region.
    The operand check in :func:`_grid_truth` then turned most of those into
    holes, but only where an operand was itself non-finite — a failure in the
    comparison with finite operands was shaded in.
    """
    try:
        return bool(point(a, b))
    except EVALUATION_FAILURE:  # no truth value here, so not shown as inside
        return False


def _clip(values: Array, percentiles: tuple[float, float] = (2.0, 98.0)) -> tuple[float, float] | None:
    """The readable z-window, on the same principle as PRD 5.3 step 5.

    A surface with a pole is worse off than a curve with one: the colour scale
    collapses as well as the axis, so everything but the pole becomes one flat
    shade.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return None
    low = float(np.percentile(finite, percentiles[0]))
    high = float(np.percentile(finite, percentiles[1]))
    if not high > low:
        return None
    full_low, full_high = float(finite.min()), float(finite.max())
    if (full_high - full_low) <= 8.0 * (high - low):
        return None
    centre = 0.5 * (low + high)
    reach = 1.5 * (high - low)
    return (max(centre - reach, full_low), min(centre + reach, full_high))


def sample_surface(
    expr: sp.Expr,
    symbols: tuple[sp.Symbol, sp.Symbol],
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    resolution: int = GRID,
) -> SurfaceSample:
    """Sample ``z = expr(x, y)`` on a rectangular grid."""
    xs = np.linspace(float(xrange[0]), float(xrange[1]), resolution, dtype=np.float64)
    ys = np.linspace(float(yrange[0]), float(yrange[1]), resolution, dtype=np.float64)
    z, vectorized, notes = _grid_values(expr, symbols, xs, ys)

    sample = SurfaceSample(
        x=xs, y=ys, z=z, vectorized=vectorized, notes=notes, z_range=_clip(z)
    )
    if sample.finite_count == 0:
        raise SamplingError(
            f"{expr} produced no real values anywhere on the grid — nothing to draw."
        )
    holes = sample.n_points - sample.finite_count
    if holes:
        share = 100.0 * holes / sample.n_points
        sample.notes = sample.notes + (
            f"{share:.0f}% of the grid is not a real number and is left as a hole. "
            "Two-variable domains are not computed symbolically — see the manual.",
        )
    return sample


def sample_region(
    relation: sp.Basic,
    symbols: tuple[sp.Symbol, sp.Symbol],
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    resolution: int = REGION_GRID,
) -> SurfaceSample:
    """Sample where an inequality holds, as 1 inside and 0 outside.

    ``z`` carries 1.0 where the relation is satisfied, 0.0 where it is not, and
    ``NaN`` where the expressions in it are not real — so a region like
    ``sqrt(x*y) > 1`` leaves the two quadrants it cannot speak about as holes
    rather than claiming they are outside.

    ``lambdify`` does the vectorising: ``And`` and ``Or`` print as
    ``logical_and.reduce`` and ``logical_or.reduce`` under the NumPy printer, and
    a bare comparison as ``less``/``greater``, so a whole compound region
    evaluates in one pass over the mesh.
    """
    xs = np.linspace(float(xrange[0]), float(xrange[1]), resolution, dtype=np.float64)
    ys = np.linspace(float(yrange[0]), float(yrange[1]), resolution, dtype=np.float64)
    inside, vectorized, notes = _grid_truth(relation, symbols, xs, ys)

    sample = SurfaceSample(x=xs, y=ys, z=inside, vectorized=vectorized, notes=notes)
    covered = int(np.count_nonzero(inside == 1.0))
    if covered == 0:
        raise SamplingError(
            f"{relation} is satisfied nowhere on this window — there is no region "
            "to shade. Widen the ranges, or check the direction of the inequality."
        )
    undefined = int(np.count_nonzero(~np.isfinite(inside)))
    if undefined:
        share = 100.0 * undefined / inside.size
        sample.notes = sample.notes + (
            f"{share:.0f}% of the grid is not a real number, so the relation has "
            "no truth value there and it is left blank rather than shaded out.",
        )
    return sample


def _grid_truth(
    relation: sp.Basic, symbols: tuple[sp.Symbol, sp.Symbol], xs: Array, ys: Array
) -> tuple[Array, bool, tuple[str, ...]]:
    """Evaluate a relation over the mesh as 1.0 / 0.0 / NaN.

    "Not real" has to be found separately from "false", because a comparison
    against a NaN is *false* in both NumPy and Python — indistinguishable from
    being genuinely outside the region. So the operands are evaluated too, and
    every point where any of them is not finite becomes a hole.
    """
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    vectorized = True
    notes: tuple[str, ...] = ()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        test = sp.lambdify(symbols, relation, modules="numpy")
        try:
            raw = np.asarray(test(mesh_x, mesh_y))
            if raw.shape != mesh_x.shape:
                raw = np.broadcast_to(raw, mesh_x.shape)
        except EVALUATION_FAILURE:
            vectorized = False
            notes = (
                "vectorised evaluation failed; the region was tested point by "
                "point, which is slower.",
            )
            point = _PointEvaluator(relation, symbols, test)
            scalar = np.vectorize(
                lambda a, b: _safe_truth(point, a, b), otypes=[bool]
            )
            raw = scalar(mesh_x, mesh_y)

        inside = np.where(np.asarray(raw, dtype=bool), 1.0, 0.0)
        for operand in comparison_operands(relation):
            values, _, _ = _grid_values(operand, symbols, xs, ys)
            inside = np.where(np.isfinite(values), inside, np.nan)
    return inside, vectorized, notes


def comparison_operands(relation: sp.Basic) -> tuple[sp.Expr, ...]:
    """Every expression compared anywhere inside ``relation``.

    Public because ``codegen`` needs the same list: it decides where a region has
    no truth value, and the figure and the program ``show_python()`` prints must
    agree about that. Two copies of this rule would drift silently.
    """
    found: list[sp.Expr] = []
    for node in sp.preorder_traversal(relation):
        if isinstance(node, sp.Rel):
            found.extend(side for side in node.args if not side.is_number)
    # Deduplicated: a compound like `(y > x**2) & (y < x + 2)` names `y` twice,
    # and each operand costs its own `lambdify` plus a grid evaluation.
    return tuple(dict.fromkeys(found))


def sample_parametric_surface(
    components: tuple[sp.Expr, sp.Expr, sp.Expr],
    symbols: tuple[sp.Symbol, sp.Symbol],
    urange: tuple[float, float],
    vrange: tuple[float, float],
    resolution: int = GRID,
) -> SurfaceSample:
    """Sample ``(x(u, v), y(u, v), z(u, v))`` on a grid in the parameters."""
    us = np.linspace(float(urange[0]), float(urange[1]), resolution, dtype=np.float64)
    vs = np.linspace(float(vrange[0]), float(vrange[1]), resolution, dtype=np.float64)

    grids: list[Array] = []
    vectorized = True
    notes: tuple[str, ...] = ()
    for component in components:
        values, ok, extra = _grid_values(component, symbols, us, vs)
        grids.append(values)
        vectorized = vectorized and ok
        notes = notes + extra

    sample = SurfaceSample(
        x=grids[0],
        y=grids[1],
        z=grids[2],
        parametric=True,
        vectorized=vectorized,
        notes=tuple(dict.fromkeys(notes)),
    )
    if sample.finite_count == 0:
        raise SamplingError("the parametric surface produced no finite points.")
    return sample
