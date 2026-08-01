"""Options and arguments that were accepted but could not mean anything.

PRD §11.4 item 8 names the standard: an option that cannot be documented
truthfully is a bug. Each case below was previously accepted and then either
ignored, or carried far enough downstream to fail with a message about the
wrong thing — `plot(sin(x), (a, -1, 1))` reported an empty domain, which is a
true statement about a question nobody asked.

The learner in PRD §3 is the reason this matters more here than in most
libraries: P1 has no way to tell "I typed something meaningless" apart from
"mathematics is hard" unless the error says which.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from mathslate import cos, exp, plot, sin, t, x
from mathslate.errors import UnsupportedInputError

a = sp.Symbol("a", real=True)
b = sp.Symbol("b", real=True)


class TestRangesMustBeAboutTheExpression:
    def test_a_range_naming_an_absent_symbol_is_refused(self) -> None:
        """`a` used to become the axis, leaving `x` free and the curve all NaN."""
        with pytest.raises(UnsupportedInputError, match="does not appear"):
            plot(sin(x), (a, -1.0, 1.0), verbose=False)

    def test_the_message_names_the_symbols_that_would_work(self) -> None:
        with pytest.raises(UnsupportedInputError, match=r"its symbols are: x"):
            plot(sin(x), (a, -1.0, 1.0), verbose=False)

    def test_a_second_range_for_the_same_symbol_is_refused(self) -> None:
        """One of the two was silently discarded, and it was not obvious which."""
        with pytest.raises(UnsupportedInputError, match="two ranges"):
            plot(sin(x), (x, -1.0, 1.0), (x, -2.0, 2.0), verbose=False)

    def test_a_spare_range_is_not_quietly_dropped(self) -> None:
        with pytest.raises(UnsupportedInputError, match="does not appear"):
            plot(sin(x), (x, -1.0, 1.0), (a, -5.0, 5.0), verbose=False)

    def test_a_constant_still_accepts_any_axis(self) -> None:
        """`plot(3, (x, ...))` has no symbols to match, and draws a line anyway."""
        plan = plot(3, (x, -1.0, 1.0), verbose=False).plan
        assert plan.kind == "constant"
        assert plan.param_range == (-1.0, 1.0)

    def test_the_documented_binding_example_still_works(self) -> None:
        """Manual §5 rule 1: an explicit range wins over the conventional order."""
        assert plot(b * x, (b, -2.0, 2.0), parameters={x: 1.0}, verbose=False).plan.symbol == b


class TestParametersMustBeUsable:
    def test_a_parameter_without_a_value_says_what_is_missing(self) -> None:
        """The bare-sequence form excludes `a` from the axes but leaves it free."""
        with pytest.raises(UnsupportedInputError, match="has no value"):
            plot(a * sin(x), parameters=[a], verbose=False)

    def test_the_message_shows_the_call_that_fixes_it(self) -> None:
        with pytest.raises(UnsupportedInputError, match=r"parameters=\{a: 1\}"):
            plot(a * sin(x), parameters=[a], verbose=False)

    def test_a_symbol_cannot_be_both_axis_and_parameter(self) -> None:
        """It was substituted away and then used as the axis of what remained."""
        with pytest.raises(UnsupportedInputError, match="not both"):
            plot(a * sin(x), (a, -1.0, 1.0), parameters={a: 2.0}, verbose=False)

    def test_a_valued_parameter_is_still_frozen(self) -> None:
        result = plot(a * sin(x), parameters={a: 3.0}, verbose=False)
        assert result.plan.symbol == x
        assert np.nanmax(result.numpy[1]) == pytest.approx(3.0, rel=1e-3)

    def test_an_unused_valueless_parameter_is_harmless(self) -> None:
        """`a` is not in the expression, so there is nothing to complain about."""
        assert plot(sin(x), parameters=[a], verbose=False).plan.symbol == x


class TestTheAdviceInTheErrorsActuallyWorks:
    """An error that recommends a broken call is worse than one that does not.

    `plot(x*y)` used to be a v0.5 deferral whose message told the reader to
    give an explicit range; doing exactly that left `y` free and died in the
    sampler. Two free symbols now draw a surface, so the advice has no reader
    left — but the underlying rule it was about still holds for three.
    """

    def test_two_free_symbols_draw_a_surface_rather_than_advising(self) -> None:
        y = sp.Symbol("y", real=True)
        assert plot(x * y, verbose=False).plan.kind == "surface"

    def test_freezing_one_of_them_gives_a_curve_again(self) -> None:
        y = sp.Symbol("y", real=True)
        assert plot(x * y, parameters={y: 1}, verbose=False).plan.kind == "curve"

    def test_a_third_free_symbol_is_still_named(self) -> None:
        """Two symbols are a surface; a third has nowhere left to go."""
        y = sp.Symbol("y", real=True)
        with pytest.raises(UnsupportedInputError, match="a is still free"):
            plot(a * x * y, (x, -5.0, 5.0), (y, -5.0, 5.0), verbose=False)


class TestPointsIsHonouredOrRefused:
    @pytest.mark.parametrize("points", [0, 1, -5])
    def test_a_count_that_cannot_draw_a_line_is_refused(self, points: int) -> None:
        """These used to fall through to the default 200 without a word."""
        with pytest.raises(UnsupportedInputError, match="at least 2"):
            plot(sin(x), points=points, verbose=False)

    def test_a_real_count_is_honoured(self) -> None:
        assert plot(sin(x), points=1000, verbose=False).plan.total_points >= 1000


class TestEmptyAndUnreadableInput:
    def test_an_empty_list_explains_what_a_list_is_for(self) -> None:
        """This was an IndexError out of the middle of dispatch."""
        with pytest.raises(UnsupportedInputError, match="empty list"):
            plot([], verbose=False)

    def test_a_string_that_does_not_parse_says_so(self) -> None:
        """A string is a documented input, so "unknown type str" misdirects."""
        with pytest.raises(UnsupportedInputError, match="could not read"):
            plot("not an expr $$", verbose=False)

    def test_a_string_that_does_parse_still_plots(self) -> None:
        assert plot("sin(x)/x", verbose=False).plan.kind == "curve"


class TestLogScaleTellsTheTruth:
    def test_hiding_half_a_curve_is_reported(self) -> None:
        """Plotly drops non-positive values from a log axis without comment."""
        notes = plot(sin(x), yscale="log", verbose=False).notes
        assert any("cannot show zero or negative" in note for note in notes)

    def test_the_option_is_still_honoured(self) -> None:
        """A warning, not a refusal — the user asked for it."""
        assert plot(sin(x), yscale="log", verbose=False).plotly.layout.yaxis.type == "log"

    def test_a_positive_curve_draws_no_warning(self) -> None:
        notes = plot(exp(x), yscale="log", verbose=False).notes
        assert not any("cannot show zero" in note for note in notes)


class TestSpecialFunctionsAreReachable:
    """PRD §5.3 step 6: fall back to element-wise evaluation, loudly.

    The element-wise path went through `lambdify(modules="math")`, which knows
    no more special functions than NumPy does, so the fallback could not
    actually rescue anything NumPy had just failed on. SymPy can evaluate all
    of them.
    """

    @pytest.mark.parametrize(
        "expr",
        [sp.zeta(x), sp.Si(x), sp.besselj(0, x), sp.gamma(x)],
        ids=["zeta", "Si", "besselj", "gamma"],
    )
    def test_a_sympy_only_function_plots(self, expr: sp.Expr) -> None:
        result = plot(expr, (x, 1.0, 5.0), verbose=False)
        values = result.numpy[1]
        assert np.isfinite(values).all()

    def test_the_values_are_right_not_merely_finite(self) -> None:
        result = plot(sp.zeta(x), (x, 2.0, 4.0), verbose=False)
        xs, ys = result.numpy
        expected = [float(sp.zeta(sp.Float(v)).evalf()) for v in xs[:20]]
        assert np.allclose(ys[:20], expected, rtol=1e-6)

    def test_the_fallback_is_audible_through_plot(self) -> None:
        """The sampler suppresses RuntimeWarning wholesale to quiet NumPy.

        That also swallowed this one, so the manual's promise of "a
        RuntimeWarning, a note on the result, and an extra field in the summary
        line" delivered only the last two.
        """
        with pytest.warns(RuntimeWarning, match="element-wise"):
            result = plot(sp.zeta(x), (x, 2.0, 4.0), verbose=False)
        assert "element-wise evaluation" in result.summary()
        assert any("element-wise" in note for note in result.notes)

    def test_a_vectorisable_expression_stays_on_the_fast_path(self) -> None:
        result = plot(sin(x), verbose=False)
        assert result.plan.series[0].sample.vectorized is True
        assert "element-wise" not in result.summary()

    def test_the_parametric_path_is_audible_too(self) -> None:
        with pytest.warns(RuntimeWarning, match="element-wise"):
            plot((sp.zeta(t), t), (t, 2.0, 4.0), verbose=False)


class TestNothingAboveBrokeTheOrdinaryCases:
    @pytest.mark.parametrize(
        "build",
        [
            lambda: plot(sin(x), verbose=False),
            lambda: plot([sin(x), cos(x)], verbose=False),
            lambda: plot((cos(t), sin(t)), verbose=False),
            lambda: plot(sin(x), (x, 0.0, 6.28), verbose=False),
            lambda: plot([1.0, 2.0, 3.0], verbose=False),
            lambda: plot(np.sin, (x, -3.0, 3.0), verbose=False),
        ],
    )
    def test_it_still_draws(self, build: object) -> None:
        assert build().plan.series  # type: ignore[operator]
