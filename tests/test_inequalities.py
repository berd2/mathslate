"""Inequalities as pictures — PRD 5.1's two v1.5 rows.

`Eq(lhs, rhs)` asks where two expressions are *equal*, which is a curve, and
that row has been there since v0.5. `lhs < rhs` asks where one *exceeds* the
other, which is an area — so it cannot share a planner even though it arrives
looking almost identical, and until now it was refused outright.

Which picture you get follows the same rule as an ordinary expression: two free
symbols make a region in the plane, one makes shaded intervals on the axis.
Those are the Korean curriculum's 부등식의 영역 and 부등식의 해 respectively.

`TestSolvesetIsCheckedNotTrusted` is the class to read. `solveset` answers a
periodic inequality with its principal period and does not mention that it has
done so, which would make an exact-looking answer that is simply wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import sympy as sp

from mathslate import cos, exp, plot, sin, sqrt, x, y
from mathslate.core.surfaces import REGION_GRID, sample_region
from mathslate.errors import SamplingError, UnsupportedInputError


class TestTheDispatchRows:
    @pytest.mark.parametrize(
        "relation",
        [x**2 + y**2 < 1, sp.Le(y, sin(x)), y > x, (x**2 + y**2 < 4) & (y > x),
         (x < 0) | (y < 0)],
        ids=["disc", "under-a-curve", "half-plane", "and", "or"],
    )
    def test_two_free_symbols_make_a_region(self, relation: object) -> None:
        assert plot(relation, verbose=False).plan.kind == "region"

    @pytest.mark.parametrize(
        "relation",
        [sin(x) > 0, x**2 - 4 < 0, exp(x) > 3, sp.Ge(x**3 - x, 0)],
        ids=["periodic", "quadratic", "exponential", "cubic"],
    )
    def test_one_free_symbol_makes_bands(self, relation: object) -> None:
        assert plot(relation, (x, -4, 4), verbose=False).plan.kind == "band"

    def test_an_equation_is_still_a_curve(self) -> None:
        """The new rows must not swallow the row that was already there."""
        assert plot(sp.Eq(x**2 + y**2, 4), verbose=False).plan.kind == "implicit"

    def test_a_compound_over_one_variable_says_what_to_do(self) -> None:
        with pytest.raises(UnsupportedInputError, match="combines several"):
            plot((x > 0) & (x < 1), (x, -2, 2), verbose=False)

    @pytest.mark.parametrize(
        "settled", [x**2 + 1 < 0, sp.Ge(x**2, 0)], ids=["never", "always"]
    )
    def test_an_inequality_sympy_already_decided_is_explained(
        self, settled: object
    ) -> None:
        """`x**2 >= 0` reaches plot() as the bare value True, expressions gone."""
        with pytest.raises(UnsupportedInputError, match="decided this inequality"):
            plot(settled, (x, -2, 2), verbose=False)


class TestTheRegionIsTheRightSet:
    def _grid(self, relation: object, span: float = 3.0) -> tuple[Any, Any, Any]:
        sample = plot(relation, (x, -span, span), (y, -span, span), verbose=False)
        s = sample.plan.series[0].sample
        mesh_x, mesh_y = np.meshgrid(s.x, s.y)
        return mesh_x, mesh_y, s.z

    def test_the_unit_disc_is_the_unit_disc(self) -> None:
        mesh_x, mesh_y, inside = self._grid(x**2 + y**2 < 1)
        radius = np.hypot(mesh_x, mesh_y)
        # Every shaded point is inside, every unshaded one outside. Points within
        # one grid step of the boundary are nobody's business.
        step = 2.0 * 3.0 / (REGION_GRID - 1)
        clear = np.abs(radius - 1.0) > step
        assert np.all(inside[clear & (radius < 1)] == 1.0)
        assert np.all(inside[clear & (radius > 1)] == 0.0)

    def test_a_half_plane_is_shaded_on_the_right_side(self) -> None:
        mesh_x, mesh_y, inside = self._grid(y > x)
        above = mesh_y > mesh_x + 0.1
        assert np.all(inside[above] == 1.0)
        assert np.all(inside[mesh_y < mesh_x - 0.1] == 0.0)

    def test_an_intersection_is_the_intersection(self) -> None:
        mesh_x, mesh_y, inside = self._grid((x**2 + y**2 < 4) & (y > x))
        both = (mesh_x**2 + mesh_y**2 < 3.8) & (mesh_y > mesh_x + 0.1)
        assert np.all(inside[both] == 1.0)
        # One condition alone is not enough.
        only_one = (mesh_x**2 + mesh_y**2 < 3.8) & (mesh_y < mesh_x - 0.1)
        assert np.all(inside[only_one] == 0.0)


class TestNotRealIsNotTheSameAsOutside:
    """The subtle one: a comparison against NaN is False in NumPy and in Python.

    Without separating the two, `sqrt(x*y) > 1` would report the quadrants where
    `x*y < 0` as *outside the region* — a definite claim about a place where the
    inequality has no truth value at all.
    """

    def test_a_non_real_quadrant_becomes_a_hole_not_an_exclusion(self) -> None:
        sample = plot(sqrt(x * y) > 1, (x, -3, 3), (y, -3, 3), verbose=False)
        mesh_x, mesh_y = np.meshgrid(*(sample.plan.series[0].sample.x,
                                       sample.plan.series[0].sample.y))
        inside = sample.plan.series[0].sample.z
        negative = mesh_x * mesh_y < -0.05
        assert np.all(~np.isfinite(inside[negative])), (
            "where sqrt has no real value the relation has no truth value either"
        )
        assert np.isfinite(inside[mesh_x * mesh_y > 0.05]).all()

    def test_and_the_share_is_reported(self) -> None:
        notes = plot(sqrt(x * y) > 1, verbose=False).notes
        assert any("no truth value" in n for n in notes)

    def test_a_region_that_is_empty_is_an_error_not_a_blank_picture(self) -> None:
        with pytest.raises(SamplingError, match="satisfied nowhere"):
            plot(x**2 + y**2 < 1, (x, 5, 9), (y, 5, 9), verbose=False)


class TestSolvesetIsCheckedNotTrusted:
    """`solveset` reports the principal period of a periodic inequality only.

    `solveset(sin(x) > 0, x, Interval(-6, 6))` is `Interval.open(0, pi)`. The
    window argument does not help and nothing in the answer marks it as partial,
    so taking it at face value would put a confident wrong answer on the screen —
    the one failure mode this package must not have. The closed form is therefore
    only used when sampling cannot find a solution outside it.
    """

    def test_a_periodic_inequality_gets_every_band(self) -> None:
        plan = plot(sin(x) > 0, (x, -6, 6), verbose=False).plan
        assert len(plan.bands) == 2, f"solveset's single band was trusted: {plan.bands}"
        (first_lo, first_hi), (second_lo, second_hi) = plan.bands
        assert first_lo == pytest.approx(-6.0)
        assert first_hi == pytest.approx(-np.pi, abs=1e-9)
        assert second_lo == pytest.approx(0.0, abs=1e-9)
        assert second_hi == pytest.approx(np.pi, abs=1e-9)

    def test_and_says_it_is_approximate(self) -> None:
        result = plot(sin(x) > 0, (x, -6, 6), verbose=False)
        assert result.plan.bands_exact is False
        assert any("sampling" in n for n in result.notes)

    def test_solveset_still_wins_when_it_is_complete(self) -> None:
        """Rejecting a correct closed form would be the opposite mistake."""
        result = plot(x**2 - 4 < 0, (x, -4, 4), verbose=False)
        assert result.plan.bands_exact is True
        assert result.plan.bands == ((-2.0, 2.0),)
        assert not any("sampling" in n for n in result.notes)

    def test_a_tangency_does_not_veto_the_exact_answer(self) -> None:
        """`x**3 - x >= 0` touches zero; sampling noise there must not disqualify."""
        result = plot(sp.Ge(x**3 - x, 0), (x, -2, 2), verbose=False)
        assert result.plan.bands_exact is True
        assert [tuple(b) for b in result.plan.bands] == [
            pytest.approx((-1.0, 0.0)), pytest.approx((1.0, 2.0))
        ]

    @pytest.mark.parametrize(
        "relation,expected",
        [
            (cos(x) < 0, [(-4.712389, -1.5707963), (1.5707963, 4.712389)]),
            (sin(x) ** 2 > 0.5, [(-5.4977871, -3.9269908),
                                 (-2.3561945, -0.7853982),
                                 (0.7853982, 2.3561945),
                                 (3.9269908, 5.4977871)]),
        ],
        ids=["cos", "sin-squared"],
    )
    def test_every_period_in_the_window_is_shaded(
        self, relation: object, expected: list[tuple[float, float]]
    ) -> None:
        bands = plot(relation, (x, -6, 6), verbose=False).plan.bands
        assert len(bands) == len(expected)
        for (lo, hi), (want_lo, want_hi) in zip(bands, expected):
            assert lo == pytest.approx(want_lo, abs=1e-4)
            assert hi == pytest.approx(want_hi, abs=1e-4)

    def test_the_sampled_edges_are_refined_not_left_on_grid_lines(self) -> None:
        """Bisection puts an edge on the root, not on the nearest sample."""
        bands = plot(sin(x) > 0, (x, -6, 6), verbose=False).plan.bands
        assert abs(bands[1][1] - np.pi) < 1e-12


class TestTheBandsAreDrawnOverTheCurve:
    def test_the_curve_is_drawn_as_well_as_the_shading(self) -> None:
        """A strip on a bare axis says where; the curve says why."""
        figure = plot(x**2 - 4 < 0, (x, -4, 4), verbose=False).plotly
        assert len(figure.data) == 1 and figure.data[0].type == "scatter"
        assert len(figure.layout.shapes) == 1

    def test_the_shading_is_behind_the_curve(self) -> None:
        figure = plot(x**2 - 4 < 0, (x, -4, 4), verbose=False).plotly
        assert figure.layout.shapes[0].layer == "below"

    def test_one_shape_per_band(self) -> None:
        figure = plot(sin(x) > 0, (x, -6, 6), verbose=False).plotly
        assert len(figure.layout.shapes) == 2

    def test_the_shapes_sit_where_the_bands_do(self) -> None:
        result = plot(sin(x) > 0, (x, -6, 6), verbose=False)
        drawn = [(s.x0, s.x1) for s in result.plotly.layout.shapes]
        assert drawn == [pytest.approx(b) for b in result.plan.bands]

    def test_the_curve_is_of_lhs_minus_rhs(self) -> None:
        """So that it crosses zero exactly where the shading starts."""
        result = plot(x**2 - 4 < 0, (x, -4, 4), verbose=False)
        assert sp.simplify(result.sympy - (x**2 - 4)) == 0


class TestTheRegionTrace:
    def test_it_is_a_heatmap_not_a_contour(self) -> None:
        """A contour interpolates, inventing a soft edge for a hard boundary."""
        assert plot(x**2 + y**2 < 1, verbose=False).plotly.data[0].type == "heatmap"

    def test_outside_is_transparent_so_the_page_shows_through(self) -> None:
        trace = plot(x**2 + y**2 < 1, verbose=False).plotly.data[0]
        assert trace.colorscale[0][1] == "rgba(0,0,0,0)"
        assert trace.zmin == 0.0 and trace.zmax == 1.0

    def test_the_grid_is_finer_than_a_surfaces(self) -> None:
        """A region is judged by its boundary, a surface by its interior."""
        from mathslate.core.surfaces import GRID

        assert REGION_GRID > GRID
        assert plot(x**2 + y**2 < 1, verbose=False).plan.series[0].sample.z.shape == (
            REGION_GRID,
            REGION_GRID,
        )

    def test_the_report_names_the_kind_and_both_windows(self) -> None:
        summary = plot(x**2 + y**2 < 1, verbose=False).summary()
        assert summary.startswith("region")
        assert "x ∈" in summary and "y ∈" in summary


class TestSampleRegionDirectly:
    def test_it_refuses_a_window_with_no_solution(self) -> None:
        with pytest.raises(SamplingError, match="satisfied nowhere"):
            sample_region(x**2 + y**2 < 1, (x, y), (5, 9), (5, 9), resolution=21)

    def test_the_message_suggests_both_likely_causes(self) -> None:
        with pytest.raises(SamplingError) as caught:
            sample_region(x**2 + y**2 < 1, (x, y), (5, 9), (5, 9), resolution=21)
        assert "Widen the ranges" in str(caught.value)
        assert "direction of the inequality" in str(caught.value)

    def test_truth_values_are_only_ever_one_zero_or_nan(self) -> None:
        z = sample_region(sqrt(x * y) > 1, (x, y), (-2, 2), (-2, 2), resolution=31).z
        finite = z[np.isfinite(z)]
        assert set(np.unique(finite)) <= {0.0, 1.0}
