"""Adaptive sampling internals (PRD 5.3) — the six numbered steps."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from mathslate import Abs, cos, floor, log, sin, sqrt, tan, x
from mathslate.core import sampling
from mathslate.errors import SamplingError


class TestStep1SymbolicSingularities:
    def test_poles_come_from_sympy_not_from_guessing(self) -> None:
        info = sampling.describe_domain(tan(x), x, -10.0, 10.0)
        expected = [k * np.pi / 2 for k in (-5, -3, -1, 1, 3, 5)]
        assert info.singular_points == pytest.approx(tuple(expected))

    def test_a_simple_pole_is_exact(self) -> None:
        info = sampling.describe_domain(1 / (x - 3), x, -10.0, 10.0)
        assert info.singular_points == (3.0,)

    def test_singular_points_split_the_domain(self) -> None:
        info = sampling.describe_domain(1 / x, x, -10.0, 10.0)
        assert info.intervals == ((-10.0, 0.0), (0.0, 10.0))


class TestStep2Domain:
    def test_sqrt_is_restricted(self) -> None:
        info = sampling.describe_domain(sqrt(x), x, -10.0, 10.0)
        assert info.intervals == ((0.0, 10.0))[:] or info.intervals == ((0.0, 10.0),)

    def test_log_is_restricted(self) -> None:
        info = sampling.describe_domain(log(x), x, -10.0, 10.0)
        assert info.intervals[0][0] >= 0.0

    def test_a_two_sided_restriction(self) -> None:
        info = sampling.describe_domain(sqrt(4 - x**2), x, -10.0, 10.0)
        assert info.intervals == ((-2.0, 2.0),)

    def test_an_undecidable_domain_falls_back_loudly(self) -> None:
        info = sampling.describe_domain(floor(x), x, -5.0, 5.0)
        assert info.domain_known is False
        assert info.intervals == ((-5.0, 5.0),)


class TestStep3AdaptiveSubdivision:
    def test_a_wiggly_curve_gets_more_points_than_a_straight_one(self) -> None:
        straight = sampling.sample_expression(2 * x + 1, x, (-10.0, 10.0))
        wiggly = sampling.sample_expression(sin(5 * x), x, (-10.0, 10.0))
        assert wiggly.n_points > straight.n_points

    def test_the_point_cap_holds(self) -> None:
        result = sampling.sample_expression(sin(x**3), x, (-10.0, 10.0))
        assert result.n_points <= sampling.DEFAULT_CONFIG.max_points * 1.5

    def test_refinement_is_configurable(self) -> None:
        config = sampling.SamplingConfig(initial_points=50, max_depth=0)
        result = sampling.sample_expression(sin(x), x, (-10.0, 10.0), config)
        assert result.n_points == 50


class TestStep4LineBreaking:
    def test_a_pole_gets_a_nan(self) -> None:
        result = sampling.sample_expression(1 / x, x, (-10.0, 10.0))
        assert np.isnan(result.y).any()

    def test_floor_is_broken_at_every_integer(self) -> None:
        result = sampling.sample_expression(floor(x), x, (-5.0, 5.0))
        found = np.array(result.breakpoints)
        assert found.size == 9
        assert np.allclose(found, np.round(found), atol=1e-6)

    def test_a_steep_but_continuous_curve_is_not_broken(self) -> None:
        """The bisection probe is what tells these two apart."""
        result = sampling.sample_expression(sp.atan(1000 * x), x, (-1.0, 1.0))
        assert result.breakpoints == ()

    def test_a_jump_survives_bisection_but_a_slope_does_not(self) -> None:
        jump = sampling.sample_expression(sp.sign(x), x, (-2.0, 2.0))
        assert len(jump.breakpoints) == 1
        assert jump.breakpoints[0] == pytest.approx(0.0, abs=1e-6)


class TestStep5YClipping:
    def test_a_pole_triggers_clipping(self) -> None:
        result = sampling.sample_expression(1 / x, x, (-10.0, 10.0))
        assert result.y_range is not None

    def test_a_bounded_curve_is_left_alone(self) -> None:
        result = sampling.sample_expression(sin(x), x, (-10.0, 10.0))
        assert result.y_range is None

    def test_a_removable_singularity_does_not_squash_the_curve(self) -> None:
        result = sampling.sample_expression(sin(x) / x, x, (-10.0, 10.0))
        # A hole with a finite limit is not a pole.  Leave the view automatic
        # so it is based on every finite sample, including the peak tending to
        # one at zero, instead of forcing a percentile clip around the hole.
        assert result.y_range is None
        finite = result.y[np.isfinite(result.y)]
        assert finite.max() >= 0.99
        assert finite.min() <= -0.2


class TestStep6Vectorisation:
    def test_the_fast_path_is_the_default(self) -> None:
        function = sampling.NumericFunction(sin(x), x)
        values = function(np.linspace(-1.0, 1.0, 5))
        assert function.vectorized is True
        assert np.allclose(values, np.sin(np.linspace(-1.0, 1.0, 5)))

    def test_complex_results_become_nan_not_garbage(self) -> None:
        function = sampling.NumericFunction(sqrt(x), x)
        values = function(np.array([-4.0, 4.0]))
        assert np.isnan(values[0])
        assert values[1] == pytest.approx(2.0)

    def test_the_fallback_is_loud(self) -> None:
        function = sampling.NumericFunction(sin(x), x)

        def broken(_: object) -> object:
            raise ValueError("no numpy for you")

        function._vector = broken  # type: ignore[assignment]
        with pytest.warns(RuntimeWarning, match="element-wise"):
            values = function(np.linspace(0.0, 1.0, 3))
        assert function.vectorized is False
        assert np.allclose(values, np.sin(np.linspace(0.0, 1.0, 3)))
        assert function.notes and "element-wise" in function.notes[0]


class TestFailureModes:
    def test_an_empty_range_is_rejected(self) -> None:
        with pytest.raises(SamplingError, match="empty range"):
            sampling.sample_expression(x, x, (1.0, 1.0))

    def test_a_nowhere_real_expression_says_so(self) -> None:
        with pytest.raises(SamplingError):
            sampling.sample_expression(sqrt(-1 - x**2), x, (-1.0, 1.0))


class TestParametric:
    def test_the_unit_circle_stays_on_the_unit_circle(self) -> None:
        result = sampling.sample_parametric((cos(x), sin(x)), x, (0.0, 2 * np.pi))
        radius = np.hypot(result.x, result.y)
        assert np.allclose(radius[np.isfinite(radius)], 1.0, atol=1e-9)

    def test_absolute_value_corner_is_refined(self) -> None:
        result = sampling.sample_expression(Abs(x), x, (-1.0, 1.0))
        near_corner = np.count_nonzero(np.abs(result.x) < 0.05)
        assert near_corner > 5
