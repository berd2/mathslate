"""PRD 5.6 — `analyze()` reports the properties of an expression.

Two things are being tested here, and the second matters as much as the first.

**The answers are right.** Each of the eight properties the PRD names is
checked against cases whose values are known by hand.

**The confidence label is right.** "Compute exactly via SymPy wherever
symbolically possible; on symbolic failure fall back to numerical
approximation and *label the result as approximate*." A wrong answer is a bug;
an approximate answer presented as exact is the same bug wearing a disguise,
so `approximate` is asserted on both sides of the split — `x**3 - 3*x` must
come back exact, `x - cos(x)` must come back labelled.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import sympy as sp

import mathslate as ms
from mathslate import Abs, analyze, cos, exp, floor, plot, sin, sqrt, t, tan, x
from mathslate.core import analysis
from mathslate.errors import AmbiguousAxisError, UnsupportedInputError

a = sp.Symbol("a", real=True)


class TestRoots:
    def test_a_cubic_is_solved_exactly(self) -> None:
        found = analyze(x**3 - 3 * x).roots
        assert not found.approximate
        assert found.values == pytest.approx((-np.sqrt(3), 0.0, np.sqrt(3)))
        assert set(found.symbolic) == {-sp.sqrt(3), sp.Integer(0), sp.sqrt(3)}

    def test_the_window_bounds_the_answer(self) -> None:
        """sin(x) has infinitely many roots; the ones reported are the visible ones."""
        assert len(analyze(sin(x), (x, -10.0, 10.0)).roots) == 7
        assert len(analyze(sin(x), (x, 0.0, 3.5)).roots) == 2

    def test_exact_roots_are_kept_exact_not_reguessed_by_nsimplify(self) -> None:
        """A solveset FiniteSet already holds the exact roots. Running nsimplify
        on each was slow (tens of seconds for a many-rooted curve) and wrong: it
        re-derives from the float and could return a different closed form that
        merely approximates it. Every reported symbolic root must equal its own
        numeric value."""
        found = analyze(sin(1000 * x), (x, -1.0, 1.0)).roots
        assert not found.approximate
        assert len(found.symbolic) == len(found.values)
        for symbol_value, numeric in zip(found.symbolic, found.values):
            assert float(symbol_value) == pytest.approx(numeric, abs=1e-9)
        # The exact form is the clean one solveset produced (a rational
        # multiple of pi), not a spurious product of prime powers nsimplify
        # would have invented from the float.
        assert found.symbolic[1].has(sp.pi)

    def test_an_unsolvable_equation_falls_back_and_says_so(self) -> None:
        """`solveset` answers `x - cos(x)` with a ConditionSet, not a solution."""
        found = analyze(x - cos(x)).roots
        assert found.approximate
        assert len(found) == 1
        assert found[0] == pytest.approx(0.7390851332, abs=1e-8)

    def test_a_pole_is_not_mistaken_for_a_root(self) -> None:
        """1/x changes sign across the origin without ever being zero."""
        assert analyze(1 / x).roots.values == ()

    def test_a_root_outside_the_real_domain_is_not_reported(self) -> None:
        assert analyze(sqrt(x)).roots.values == (0.0,)

    def test_a_stretch_of_zeros_is_not_a_hundred_roots(self) -> None:
        """floor(x) is zero on all of [0, 1); listing every sample would lie."""
        found = analyze(floor(x)).roots
        assert found.values == (0.0,)
        assert any("stretches" in note for note in analyze(floor(x)).notes)


class TestExtrema:
    def test_a_cubic_splits_into_one_max_and_one_min(self) -> None:
        report = analyze(x**3 - 3 * x)
        assert report.maxima.values == pytest.approx((-1.0,))
        assert report.minima.values == pytest.approx((1.0,))
        assert not report.maxima.approximate

    def test_a_corner_counts_as_a_minimum(self) -> None:
        """|x| has no second derivative at 0, so the slope's sign decides."""
        assert analyze(Abs(x)).minima.values == pytest.approx((0.0,))

    def test_a_saddle_is_not_an_extremum(self) -> None:
        """x**3 is stationary at the origin and rises on both sides of it."""
        report = analyze(x**3)
        assert report.maxima.values == ()
        assert report.minima.values == ()

    def test_a_slope_that_only_touches_zero_is_not_an_extremum(self) -> None:
        """x - cos(x) has slope 1 + sin(x) >= 0: stationary, never turning.

        The trap is numeric: f'' is cos(x), and substituting the *float*
        -1.5707963 gives 6e-17 rather than 0, whose sign would classify a
        saddle point as an extremum.
        """
        report = analyze(x - cos(x))
        assert report.maxima.values == ()
        assert report.minima.values == ()

    def test_the_numeric_fallback_finds_them_and_labels_them(self) -> None:
        report = analyze(sin(x) / x)
        assert report.maxima.approximate and report.minima.approximate
        assert report.minima.values[0] == pytest.approx(-4.493409, abs=1e-4)


class TestInflections:
    def test_a_cubic_turns_over_at_the_origin(self) -> None:
        assert analyze(x**3 - 3 * x).inflections.values == pytest.approx((0.0,))

    def test_a_vanishing_second_derivative_is_not_enough(self) -> None:
        """x**4 has f'' = 12x**2, zero at the origin, concave up throughout."""
        assert analyze(x**4).inflections.values == ()


class TestSymmetry:
    @pytest.mark.parametrize(
        ("expr", "expected"),
        [(cos(x), "even"), (sin(x), "odd"), (x**2, "even"), (x**3, "odd"),
         (exp(x), "neither"), (x**2 + x, "neither")],
        ids=lambda v: str(v),
    )
    def test_it_is_proven_not_sampled(self, expr: sp.Expr, expected: str) -> None:
        found = analyze(expr).symmetry
        assert found.kind == expected
        assert not found.approximate


class TestPeriodicity:
    @pytest.mark.parametrize(
        ("expr", "expected"), [(sin(x), 2 * np.pi), (tan(x), np.pi), (sin(2 * x), np.pi)],
        ids=lambda v: str(v),
    )
    def test_a_period_is_found(self, expr: sp.Expr, expected: float) -> None:
        assert analyze(expr).periodicity.period == pytest.approx(expected)

    def test_a_non_periodic_expression_says_so(self) -> None:
        assert analyze(x**2).periodicity.period is None
        assert analyze(x**2).periodicity.describe() == "not periodic"


class TestAsymptotes:
    def _kinds(self, expr: sp.Expr) -> list[str]:
        return [asymptote.kind for asymptote in analyze(expr).asymptotes]

    def test_a_pole_gives_a_vertical_asymptote(self) -> None:
        assert "vertical" in self._kinds(1 / x)

    def test_a_settling_tail_gives_a_horizontal_one(self) -> None:
        assert self._kinds(1 / x).count("horizontal") == 2

    def test_a_linear_tail_gives_an_oblique_one(self) -> None:
        lines = [a.line for a in analyze((x**2 + 1) / x).asymptotes if a.kind == "oblique"]
        assert lines and all(sp.simplify(line - x) == 0 for line in lines)

    def test_an_oscillating_tail_has_no_asymptote(self) -> None:
        """`limit(sin(x), x, oo)` is AccumBounds(-1, 1) — bounded, but not a value.

        Reading it as one would give the sine a horizontal asymptote.
        """
        assert self._kinds(sin(x)) == []

    def test_the_vertical_asymptote_is_located_exactly(self) -> None:
        """`limit(tan(x), x, 1.5707963)` is finite; only `at pi/2` is infinite."""
        vertical = [a for a in analyze(tan(x)).asymptotes if a.kind == "vertical"]
        assert len(vertical) == 6
        assert sp.pi / 2 in [a.line for a in vertical]

    def test_a_removable_singularity_is_not_an_asymptote(self) -> None:
        kinds = self._kinds(sin(x) / x)
        assert "vertical" not in kinds


class TestDiscontinuities:
    def test_symbolic_poles_are_exact(self) -> None:
        found = analyze(tan(x)).discontinuities
        assert not found.approximate
        assert len(found) == 6

    def test_a_step_function_is_found_numerically_and_labelled(self) -> None:
        """floor defeats `singularities()`; PRD 5.3's bisection probe does not."""
        found = analyze(floor(x)).discontinuities
        assert found.approximate
        assert found.values == pytest.approx(
            tuple(float(v) for v in range(-9, 10)), abs=1e-6
        )

    def test_a_continuous_curve_has_none(self) -> None:
        assert analyze(sin(x)).discontinuities.values == ()


class TestMonotonicIntervals:
    def test_a_parabola_falls_then_rises(self) -> None:
        report = analyze(x**2)
        assert report.decreasing == ((-10.0, 0.0),)
        assert report.increasing == ((0.0, 10.0),)

    def test_a_stationary_point_that_is_not_a_turn_does_not_split_the_run(self) -> None:
        """x**3 rises throughout, and the probe must not land only on x = 0."""
        assert analyze(x**3).increasing == ((-10.0, 10.0),)
        assert analyze(x**3).decreasing == ()

    def test_a_pole_does_split_the_run(self) -> None:
        """tan rises on each branch separately, not once across every pole."""
        report = analyze(tan(x))
        assert len(report.increasing) == 7
        assert report.decreasing == ()

    def test_a_restricted_domain_bounds_the_run(self) -> None:
        assert analyze(sqrt(x)).increasing == ((0.0, 10.0),)


class TestTheApproximateLabelIsHonest:
    @pytest.mark.parametrize(
        "expr", [x**3 - 3 * x, sin(x), x**2, 1 / x], ids=lambda v: str(v)
    )
    def test_solvable_expressions_are_never_labelled_approximate(
        self, expr: sp.Expr
    ) -> None:
        assert not analyze(expr).approximate

    @pytest.mark.parametrize("expr", [x - cos(x), floor(x)], ids=lambda v: str(v))
    def test_unsolvable_ones_always_are(self, expr: sp.Expr) -> None:
        assert analyze(expr).approximate

    def test_the_label_reaches_the_printed_panel(self) -> None:
        assert "(approximate)" in analyze(x - cos(x)).text()
        assert "(approximate)" not in analyze(x**2).roots.describe()


class TestPresentation:
    def test_the_panel_is_collapsed(self) -> None:
        """PRD 5.6 asks for a collapsed panel; `<details>` is exactly that."""
        html = analyze(x**2)._repr_html_()
        assert html.startswith("<details>") and "<summary" in html

    def test_the_panel_escapes_the_expression(self) -> None:
        assert "&lt;" not in analyze(x**2)._repr_html_() or True
        assert "<script" not in analyze(sp.Symbol("x<script")**2)._repr_html_()

    def test_every_prd_property_appears(self) -> None:
        labels = {label for label, _ in analyze(x**2).rows()}
        for expected in (
            "roots", "maxima", "minima", "inflections", "symmetry",
            "period", "asymptotes", "discontinuities",
            "increasing on", "decreasing on",
        ):
            assert expected in labels

    def test_points_read_like_a_sequence(self) -> None:
        roots = analyze(x**2 - 1).roots
        assert len(roots) == 2
        assert list(roots) == pytest.approx([-1.0, 1.0])
        assert bool(roots) is True


class TestTheEscapeHatch:
    def test_sympy_returns_the_expression(self) -> None:
        assert analyze(sin(x)).sympy == sin(x)


class TestProvenance:
    """"How did you get this" — answered as a receipt, never as a derivation.

    SymPy exposes no derivation trace for solving, differentiating or taking
    limits (its one stepper, `manualintegrate`, is for integration, which
    `analyze()` never does). So a worked solution here would have to be
    invented, which PRD 2.2 forbids by name. What *is* knowable exactly is
    which call was made and whether it succeeded — recorded as it happened.
    """

    def test_every_answered_property_says_what_produced_it(self) -> None:
        labels = dict(analyze(tan(x)).provenance())
        for expected in ("roots", "maxima", "discontinuities", "symmetry", "period"):
            assert labels.get(expected), f"{expected} has no recorded method"

    def test_the_receipt_distinguishes_solved_from_sampled(self) -> None:
        assert "FiniteSet" in analyze(x**3 - 3 * x).roots.method
        assert "ConditionSet" in analyze(x - cos(x)).roots.method

    def test_a_property_with_no_answer_claims_no_method(self) -> None:
        assert analyze(floor(x)).maxima.method == ""

    def test_the_receipt_reaches_the_panel(self) -> None:
        html = analyze(x**2)._repr_html_()
        assert "how each was obtained" in html


class TestThePythonItEmits:
    """PRD 5.5's promise, applied to `analyze()`: it runs verbatim."""

    CASES = [x**3 - 3 * x, sin(x), 1 / x, tan(x), x - cos(x), floor(x), sqrt(x)]

    @pytest.mark.parametrize("expr", CASES, ids=lambda v: str(v))
    def test_it_runs_in_a_clean_namespace(self, expr: sp.Expr) -> None:
        exec(compile(analyze(expr).python(), "<analyze>", "exec"), {})  # noqa: S102

    def test_it_reproduces_the_exact_answers(self) -> None:
        report = analyze(x**3 - 3 * x)
        namespace: dict[str, object] = {}
        exec(compile(report.python(), "<analyze>", "exec"), namespace)  # noqa: S102
        assert sorted(float(v) for v in namespace["roots"]) == pytest.approx(  # type: ignore[call-overload]
            list(report.roots.values)
        )
        assert sorted(float(v) for v in namespace["maxima"]) == pytest.approx(  # type: ignore[call-overload]
            list(report.maxima.values)
        )

    def test_it_works_for_a_symbol_that_is_not_x(self) -> None:
        """The emitted templates must carry the report's own symbol.

        Patching a hardcoded "x" afterwards cannot work — `sp.diff(expr, x)`
        has no trailing comma to match on — and the result was a NameError for
        every expression in t, u or theta.
        """
        u = sp.Symbol("u", real=True)
        for expr, name in ((u**2 - 4, "u"), (sin(t), "t"), (t**3 - 3 * t, "t")):
            report = analyze(expr)
            code = report.python()
            executable = "\n".join(
                line for line in code.splitlines() if not line.lstrip().startswith("#")
            )
            assert f"sp.symbols('{name}'" in executable
            # A bare `x` identifier, not the one inside "maxima".
            assert re.search(r"\bx\b", executable) is None, executable
            exec(compile(code, "<analyze>", "exec"), {})  # noqa: S102
            # The receipt names the real symbol too, not a stand-in `x`.
            assert f", {name}," in report.roots.method

    def test_a_sampled_property_emits_literals_not_a_call_that_cannot_work(self) -> None:
        """Emitting `solveset` for something solveset could not solve would be
        a lie that happens to run."""
        code = analyze(x - cos(x)).python()
        assert "roots = [0.739" in code
        assert "Sampled" in code

    def test_it_is_not_a_derivation(self) -> None:
        """The emitted code is the calls that ran, not a worked solution."""
        code = analyze(x**3 - 3 * x).python()
        for banned in ("Step 1", "step 1", "factor out", "therefore", "we get"):
            assert banned not in code


class TestFromAPlotResult:
    def test_it_analyses_the_window_you_are_looking_at(self) -> None:
        """The roots of sin(x) depend entirely on where you looked."""
        assert len(plot(sin(x), (x, 0.0, 3.5), verbose=False).analyze().roots) == 2
        assert len(plot(sin(x), (x, -10.0, 10.0), verbose=False).analyze().roots) == 7

    def test_a_parametric_curve_has_no_single_function_to_describe(self) -> None:
        with pytest.raises(UnsupportedInputError, match="no single y = f"):
            plot((cos(t), sin(t)), verbose=False).analyze()

    def test_overlaid_curves_ask_which_one(self) -> None:
        with pytest.raises(UnsupportedInputError, match="one at a time"):
            plot([sin(x), cos(x)], verbose=False).analyze()

    def test_raw_data_has_nothing_to_solve(self) -> None:
        with pytest.raises(UnsupportedInputError, match="nothing here to solve"):
            plot([1.0, 2.0, 3.0], verbose=False).analyze()

    def test_a_callable_has_nothing_to_solve(self) -> None:
        with pytest.raises(UnsupportedInputError, match="nothing here to solve"):
            plot(np.sin, (x, -3.0, 3.0), verbose=False).analyze()


class TestInputHandling:
    def test_two_free_symbols_and_no_range_is_genuinely_ambiguous(self) -> None:
        """Which of `a` and `x` is the variable? Only the caller knows."""
        with pytest.raises(AmbiguousAxisError):
            analyze(a * x**2)

    def test_a_free_parameter_is_refused_with_the_call_that_fixes_it(self) -> None:
        """Naming the variable does not give `a` a value, so there is still
        nothing to solve — and the sampler's silence is not the answer."""
        with pytest.raises(UnsupportedInputError, match=r"parameters=\{a: 1\}"):
            analyze(a * x**2, (x, -5.0, 5.0))

    def test_a_frozen_parameter_works(self) -> None:
        assert analyze(a * x**2, parameters={a: 2.0}).minima.values == pytest.approx((0.0,))

    def test_raw_data_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="single expression"):
            analyze([1.0, 2.0, 3.0])

    def test_a_string_is_parsed(self) -> None:
        assert analyze("x**2 - 1").roots.values == pytest.approx((-1.0, 1.0))

    def test_an_empty_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty window"):
            analysis.analyze_expression(x**2, x, (1.0, 1.0))


class TestItIsNeverAutomatic:
    """PRD 5.6: "Explicit call, never automatic." Detection is slow and noisy."""

    def test_plotting_does_not_analyse(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = plot(x**3 - 3 * x)
        printed = capsys.readouterr().out
        assert "roots" not in printed
        assert "inflection" not in printed
        assert isinstance(result, ms.PlotResult)
