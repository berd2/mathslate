"""The dispatch contract (PRD 5.1) is a promise, so it gets tested like one."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from mathslate import Eq, cos, plot, polar, sin, t, theta, x, y
from mathslate.errors import UnsupportedInputError


class TestTheOneRule:
    """A list means "several things"; a tuple means "one vector-valued thing"."""

    def test_list_of_expressions_overlays_curves(self) -> None:
        plan = plot([sin(x), cos(x)], verbose=False).plan
        assert plan.kind == "curves"
        assert len(plan.series) == 2

    def test_tuple_of_two_expressions_is_one_parametric_curve(self) -> None:
        plan = plot((sin(x), cos(x)), verbose=False).plan
        assert plan.kind == "parametric"
        assert len(plan.series) == 1

    def test_the_two_forms_disagree_on_purpose(self) -> None:
        overlay = plot([sin(t), cos(t)], verbose=False).plan
        vector = plot((sin(t), cos(t)), verbose=False).plan
        assert overlay.kind != vector.kind


class TestSingleExpressions:
    def test_one_free_symbol_is_a_curve(self) -> None:
        assert plot(x**2, verbose=False).plan.kind == "curve"

    def test_zero_free_symbols_is_a_horizontal_line_with_a_message(self) -> None:
        result = plot(sp.Integer(3), verbose=False)
        assert result.plan.kind == "constant"
        assert np.allclose(result.numpy[1], 3.0)
        assert any("horizontal line" in note for note in result.notes)

    def test_two_free_symbols_is_a_surface(self) -> None:
        assert plot(x * y, verbose=False).plan.kind == "surface"

    @pytest.mark.parametrize("value", [sp.I, 2 + 3 * sp.I, sp.zoo])
    def test_a_non_real_constant_is_explained_not_crashed(self, value: sp.Expr) -> None:
        """`float(sp.N(I))` raised a raw TypeError two frames down; a constant
        with no real value gets a plot-shaped message instead."""
        with pytest.raises(UnsupportedInputError, match="not a real number"):
            plot(value, verbose=False)

    def test_a_string_is_sympified(self) -> None:
        assert plot("sin(x)/x", verbose=False).plan.kind == "curve"


class TestDataInputs:
    def test_bare_arraylike_is_a_data_series(self) -> None:
        result = plot([1.0, 4.0, 9.0, 16.0], verbose=False)
        assert result.plan.kind == "data"
        assert np.allclose(result.numpy[0], [0, 1, 2, 3])

    def test_xy_pair_is_a_scatter(self) -> None:
        result = plot(([0.0, 1.0, 2.0], [0.0, 1.0, 4.0]), verbose=False)
        assert result.plan.kind == "data"
        assert np.allclose(result.numpy[1], [0.0, 1.0, 4.0])

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(UnsupportedInputError, match="same length"):
            plot(([0.0, 1.0], [0.0, 1.0, 2.0]), verbose=False)

    def test_a_callable_is_sampled_numerically(self) -> None:
        result = plot(np.sin, (x, -3.0, 3.0), verbose=False)
        assert result.plan.kind == "callable"
        assert result.plan.series[0].sample.finite_count > 100


class TestWhatInferenceCannotDecide:
    """PRD 5.1: acknowledge honestly, require an explicit argument."""

    def test_polar_must_be_asked_for(self) -> None:
        cartesian = plot(1 + cos(t), verbose=False).plan
        polar_plan = polar(1 + cos(t), verbose=False).plan
        assert cartesian.kind == "curve"
        assert polar_plan.kind == "polar"

    def test_polar_closes_the_cardioid(self) -> None:
        result = polar(1 + cos(t), verbose=False)
        xs, ys = result.numpy
        assert np.isclose(xs[0], xs[-1], atol=1e-6)
        assert np.isclose(ys[0], ys[-1], atol=1e-6)

    def test_contour_is_the_documented_toggle(self) -> None:
        """PRD 5.1: surface by default, contour on request — the same object."""
        assert plot(x * y, verbose=False).plan.kind == "surface"
        assert plot(x * y, kind="contour", verbose=False).plan.kind == "contour"


class TestDrawingOptions:
    """Options that are documented must actually do what the manual says."""

    def test_kind_selects_the_trace_mode(self) -> None:
        assert plot(sin(x), kind="scatter", verbose=False).plotly.data[0].mode == "markers"
        assert plot(sin(x), kind="line", verbose=False).plotly.data[0].mode == "lines"

    def test_kind_defaults_to_markers_for_small_data_and_lines_otherwise(self) -> None:
        assert plot([1.0, 2.0, 3.0], verbose=False).plotly.data[0].mode == "markers"
        assert plot(sin(x), verbose=False).plotly.data[0].mode == "lines"

    def test_an_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(UnsupportedInputError, match="bar"):
            plot(sin(x), kind="bar", verbose=False)

    def test_yscale_log_is_applied(self) -> None:
        figure = plot(sp.exp(x), yscale="log", verbose=False).plotly
        assert figure.layout.yaxis.type == "log"

    def test_yscale_log_suppresses_clipping(self) -> None:
        """A log axis and the percentile window solve the same problem."""
        assert plot(1 / x, yscale="log", verbose=False).plotly.layout.yaxis.range is None

    def test_an_unknown_yscale_is_rejected_not_ignored(self) -> None:
        with pytest.raises(UnsupportedInputError, match="yscale"):
            plot(sin(x), yscale="semilog", verbose=False)


class TestDeferredRows:
    def test_an_equation_is_an_implicit_curve(self) -> None:
        assert plot(Eq(x**2 + y**2, 1), verbose=False).plan.kind == "implicit"

    def test_a_three_tuple_over_one_symbol_is_a_space_curve(self) -> None:
        assert plot((sin(t), cos(t), t), verbose=False).plan.kind == "space"

    def test_a_three_tuple_over_two_symbols_is_a_parametric_surface(self) -> None:
        plan = plot((t, theta, t * theta), (t, 0, 1), (theta, 0, 1), verbose=False).plan
        assert plan.kind == "psurface"

    def test_unknown_input_says_so_plainly(self) -> None:
        with pytest.raises(UnsupportedInputError):
            plot({"not": "plottable"}, verbose=False)


class TestTheInferenceReport:
    """PRD 5.1: every call reports what it inferred, in one line."""

    def test_the_line_names_kind_range_and_samples(self) -> None:
        summary = plot(sin(x), verbose=False).summary()
        assert summary.startswith("curve")
        assert "x ∈ [-10, 10]" in summary
        assert "samples" in summary

    def test_it_reports_handled_discontinuities(self) -> None:
        from mathslate import tan

        assert "discontinuities handled" in plot(tan(x), verbose=False).summary()

    def test_verbose_prints_it(self, capsys: pytest.CaptureFixture[str]) -> None:
        plot(sin(x), verbose=True)
        assert "curve" in capsys.readouterr().out
