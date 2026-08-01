"""PRD 5.7 and 6.2 — `slider()` and `animate()`.

    a = slider(-5, 5, default=1)
    plot(a*sin(x))

The requirement that shapes everything here is "**the user never writes a
callback binding**", in all three environments of 6.2. Two things have to hold
for those two lines to work, and each has its own class below: a slider has to
survive being put inside a SymPy expression, and `plot()` has to recognise its
symbol as a parameter rather than an axis (5.2 rule 2).

The control is Plotly's own, built from frames carried inside the figure. 6.2
lists that as the fallback for "other / static export"; it is used as the
*default* because it behaves identically in all three hosts, needs no frontend
package at all (acceptance criterion 6), and survives export to one HTML file.
`Slider.widget()` returns the native control for readers who want live
recomputation instead.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import mathslate as ms
from mathslate import animate, cos, plot, sin, slider, t, x
from mathslate.core import binding
from mathslate.errors import UnsupportedInputError
from mathslate.ui.interact import DEFAULT_STEPS, Slider


@pytest.fixture(autouse=True)
def _clean_registry():
    """Sliders bind globally, so a test must not leak into the next."""
    before = binding.bound_parameters()
    yield
    for symbol in binding.bound_parameters():
        if symbol not in before:
            binding.unbind_parameter(symbol)


class TestItLivesInsideAnExpression:
    """`a*sin(x)` has to build an ordinary SymPy expression."""

    def test_multiplying_by_a_function_works_both_ways(self) -> None:
        a = slider(0, 2, name="a")
        assert a * sin(x) == sp.Symbol("a", real=True) * sin(x)
        assert sin(x) * a == sp.Symbol("a", real=True) * sin(x)

    def test_plain_numbers_work_too(self) -> None:
        """SymPy can sympify us; `int` cannot, so the dunders are explicit."""
        a = slider(0, 2, name="a")
        for built in (a * 2, 2 * a, a + 1, 1 + a, a - 1, 1 - a, a / 2, 2 / a, a**2, -a):
            assert isinstance(built, sp.Expr)

    def test_sympify_finds_the_symbol(self) -> None:
        a = slider(0, 2, name="a")
        assert sp.sympify(a) is a.symbol

    def test_the_expression_carries_no_trace_of_the_widget(self) -> None:
        a = slider(0, 2, name="a")
        assert (a * sin(x)).free_symbols == {a.symbol, x}

    def test_a_bare_slider_inside_a_tuple_is_sympified(self) -> None:
        a = slider(0, 2, name="tuple_slider")
        result = plot((a, sin(t)), verbose=False)
        assert result.plan.kind == "parametric"
        assert result.plan.symbol == t


class TestRule2IsNowAutomatic:
    """5.2 rule 2: a bound symbol is a parameter, never an axis."""

    def test_the_axis_is_the_other_symbol(self) -> None:
        a = slider(1, 3, default=2, name="a")
        assert plot(a * sin(x), verbose=False).plan.symbol == x

    def test_the_curve_is_drawn_at_the_sliders_value(self) -> None:
        a = slider(1, 3, default=3, name="a")
        result = plot(a * sin(x), verbose=False)
        drawn = result.frames[0][result.frames[1].index(a.nearest(3.0))]
        assert np.nanmax(drawn.series[0].sample.y) == pytest.approx(3.0, rel=1e-3)

    def test_an_explicit_parameters_entry_outranks_the_slider(self) -> None:
        a = slider(1, 3, default=2, name="a")
        result = plot(a * sin(x), parameters={a.symbol: 1.0}, verbose=False)
        assert np.nanmax(result.numpy[1]) == pytest.approx(1.0, rel=1e-3)

    def test_an_explicit_range_outranks_both(self) -> None:
        """Asking for a symbol to be the axis means it is the axis."""
        a = slider(1, 3, name="a")
        plan = plot(sin(a), (a.symbol, -1.0, 1.0), verbose=False).plan
        assert plan.symbol == a.symbol
        assert plan.param_range == (-1.0, 1.0)

    def test_a_slider_not_in_the_expression_changes_nothing(self) -> None:
        slider(1, 3, name="unused_one")
        assert plot(sin(x), verbose=False).plan.symbol == x


class TestTheFrames:
    def test_one_frame_per_position(self) -> None:
        a = slider(0, 1, name="a")
        assert len(plot(a * sin(x), verbose=False).plotly.frames) == DEFAULT_STEPS

    def test_a_step_sets_the_positions(self) -> None:
        a = slider(0, 1, 0.25, name="a")
        assert a.values() == pytest.approx((0.0, 0.25, 0.5, 0.75, 1.0))
        assert len(plot(a * sin(x), verbose=False).plotly.frames) == 5

    def test_the_figure_carries_a_slider_control(self) -> None:
        a = slider(0, 1, name="a")
        figure = plot(a * sin(x), verbose=False).plotly
        assert figure.layout.sliders
        assert figure.layout.sliders[0].currentvalue.prefix == "a = "

    def test_no_callback_is_needed_anywhere(self) -> None:
        """The frames travel inside the figure, so the HTML stands alone."""
        a = slider(0, 1, name="a")
        html = plot(a * sin(x), verbose=False).plotly.to_html(include_plotlyjs=False)
        # Plotly ships frames into the page as an addFrames call.
        assert "addFrames" in html

    def test_the_axes_are_locked_across_frames(self) -> None:
        """Autoscaling per frame makes the curve look still while the axis moves."""
        a = slider(1, 5, name="a")
        figure = plot(a * sin(x), verbose=False).plotly
        low, high = figure.layout.yaxis.range
        assert high >= 5.0 and low <= -5.0

    def test_a_still_plot_stays_still(self) -> None:
        assert not plot(sin(x), verbose=False).interactive
        assert not plot(sin(x), verbose=False).plotly.frames


class TestAnimate:
    def test_it_adds_a_play_button(self) -> None:
        a = slider(1, 3, name="a")
        figure = animate(a * sin(x), verbose=False).plotly
        labels = [b.label for b in figure.layout.updatemenus[0].buttons]
        assert labels == ["play", "pause"]

    def test_plot_does_not_add_one(self) -> None:
        a = slider(1, 3, name="a")
        assert not plot(a * sin(x), verbose=False).plotly.layout.updatemenus

    def test_it_needs_something_to_vary(self) -> None:
        with pytest.raises(UnsupportedInputError, match="needs something to vary"):
            animate(sin(x), verbose=False)

    def test_over_picks_which_slider_runs(self) -> None:
        a = slider(1, 2, name="a")
        b = slider(1, 2, 0.5, name="b")
        figure = animate(a * b * sin(x), over=b, verbose=False).plotly
        assert len(figure.frames) == len(b.values())

    def test_over_refuses_a_slider_absent_from_the_expression(self) -> None:
        a = slider(1, 2, name="present")
        other = slider(1, 2, name="absent")
        with pytest.raises(UnsupportedInputError, match="absent.*does not appear"):
            animate(a * sin(x), over=other, verbose=False)


class TestSeveralSliders:
    def test_the_first_varies_and_the_rest_are_held(self) -> None:
        a = slider(1, 2, name="a")
        b = slider(1, 2, name="b")
        result = plot(a * b * sin(x), verbose=False)
        assert len(result.plotly.frames) == len(a.values())

    def test_it_says_which_is_which(self) -> None:
        a = slider(1, 2, name="a")
        b = slider(1, 2, name="b")
        notes = plot(a * b * sin(x), verbose=False).notes
        assert any("varies a" in note and "holds b" in note for note in notes)


class TestShowPythonKeepsTheMotion:
    """A still picture is not the same result when the figure moves."""

    def test_the_emitted_code_runs(self) -> None:
        a = slider(1, 3, name="a")
        exec(compile(plot(a * sin(x), verbose=False).python(), "<c>", "exec"), {})  # noqa: S102

    def test_it_emits_every_frame(self) -> None:
        a = slider(1, 3, name="a")
        result = plot(a * sin(x), verbose=False)
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<c>", "exec"), namespace)  # noqa: S102
        assert len(namespace["fig"].frames) == len(result.plotly.frames)  # type: ignore[union-attr]

    def test_the_window_matches(self) -> None:
        a = slider(1, 3, name="a")
        result = plot(a * sin(x), verbose=False)
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<c>", "exec"), namespace)  # noqa: S102
        emitted = namespace["fig"]
        for axis in ("xaxis", "yaxis"):
            ours = getattr(result.plotly.layout, axis).range
            theirs = getattr(emitted.layout, axis).range  # type: ignore[union-attr]
            assert np.allclose(
                np.asarray(ours, float), np.asarray(theirs, float), rtol=1e-6
            )

    def test_animate_keeps_its_title_and_play_controls(self) -> None:
        a = slider(1, 3, name="a")
        result = animate(a * sin(x), title="Moving", verbose=False)
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<c>", "exec"), namespace)  # noqa: S102
        emitted = namespace["fig"]
        assert emitted.layout.title.text == "Moving"  # type: ignore[union-attr]
        labels = [
            button.label for button in emitted.layout.updatemenus[0].buttons  # type: ignore[union-attr]
        ]
        assert labels == ["play", "pause"]


class TestTheSliderItself:
    def test_a_backwards_range_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="stop greater than start"):
            slider(5, 1)

    def test_a_default_outside_the_range_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="outside the slider"):
            slider(0, 1, default=5)

    def test_a_negative_step_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="step must be positive"):
            slider(0, 1, -0.1)

    def test_it_defaults_to_the_start(self) -> None:
        assert slider(2, 6, name="defaulting").value == 2.0

    def test_setting_the_value_rebinds_the_symbol(self) -> None:
        a = slider(0, 10, name="rebinding")
        a.value = 7.0
        assert binding.bound_parameters()[a.symbol] == 7.0

    def test_an_unnamed_slider_still_gets_a_name(self) -> None:
        one, two = Slider(0, 1), Slider(0, 1)
        assert one.name != two.name
        assert one.symbol.name.isidentifier()

    def test_the_repr_says_where_it_sits(self) -> None:
        assert "at 2" in repr(slider(0, 5, default=2, name="reprtest"))


class TestNativeWidgets:
    """6.2's table. Neither package is a dependency, so this is best-effort."""

    def test_a_plain_script_has_no_native_widget_and_says_so(self) -> None:
        a = slider(0, 1, name="plainhost")
        with pytest.raises(UnsupportedInputError, match="no notebook frontend"):
            a.widget()

    def test_the_plotly_control_needs_no_frontend_at_all(self) -> None:
        """Which is exactly why it is the default rather than the fallback."""
        a = slider(0, 1, name="nofrontend")
        assert plot(a * sin(x), verbose=False).plotly.layout.sliders
