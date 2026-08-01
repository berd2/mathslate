"""A slider binds its symbol for the whole session. That must not go unnoticed.

PRD 5.2 rule 2 takes slider-bound symbols out of the axis candidates, which is
what makes ``plot(a*sin(x))`` work without the user restating anything. The
registry that implements it is process-wide and outlives the cell that created
it — deliberately — so a slider named after an axis symbol used to reach into
every later call and silently freeze it.

``slider(name="t")`` followed by ``plot((cos(t), sin(t)))`` drew two hundred
samples of a single point and said nothing.
"""

from __future__ import annotations

import pytest

from mathslate import Matrix, cos, plot, sin, slider, t, x, y
from mathslate.errors import UnsupportedInputError
from mathslate.ui import Slider, release, release_all, sliders_in


@pytest.fixture()
def clean_session() -> object:
    release_all()
    yield
    release_all()


class TestASliderMayNotEatTheLastAxis:
    def test_a_parametric_curve_over_the_slider_symbol_is_refused(
        self, clean_session: object
    ) -> None:
        slider(0, 10, default=3, name="t")
        with pytest.raises(UnsupportedInputError, match="only symbol left"):
            plot((cos(t), sin(t)), verbose=False)

    def test_a_plain_curve_over_the_slider_symbol_is_refused(
        self, clean_session: object
    ) -> None:
        slider(0, 3, default=2, name="x")
        with pytest.raises(UnsupportedInputError, match="only symbol left"):
            plot(sin(x), verbose=False)

    def test_a_surface_losing_both_axes_is_refused(self, clean_session: object) -> None:
        slider(0, 3, default=2, name="x")
        slider(0, 3, default=2, name="y")
        with pytest.raises(UnsupportedInputError, match="only symbol left"):
            plot(x * y, verbose=False)

    def test_a_surface_losing_one_of_two_axes_becomes_a_curve(
        self, clean_session: object
    ) -> None:
        """Intended, not a casualty: one bound symbol leaves one axis, so a
        surface legitimately collapses to a curve with a slider on it."""
        slider(0, 3, default=2, name="y")
        plan = plot(x * y, verbose=False).plan
        assert plan.kind == "curve" and plan.symbol == x

    def test_the_message_offers_all_three_ways_out(self, clean_session: object) -> None:
        slider(0, 10, default=3, name="t")
        with pytest.raises(UnsupportedInputError) as caught:
            plot((cos(t), sin(t)), verbose=False)
        message = str(caught.value)
        assert "plot(expr, (t, -10, 10))" in message
        assert "release_all()" in message
        assert "name='a'" in message


class TestTheWaysOutActuallyWork:
    def test_an_explicit_range_reclaims_the_symbol(self, clean_session: object) -> None:
        slider(0, 10, default=3, name="t")
        result = plot((cos(t), sin(t)), (t, 0, 6.283185307), verbose=False)
        xs, ys = result.numpy
        assert xs.min() == pytest.approx(-1.0, abs=1e-3)
        assert ys.max() == pytest.approx(1.0, abs=1e-3)

    def test_release_all_reclaims_it(self, clean_session: object) -> None:
        slider(0, 10, default=3, name="t")
        release_all()
        assert plot((cos(t), sin(t)), verbose=False).plan.total_points > 100

    def test_releasing_one_slider_leaves_the_others(self, clean_session: object) -> None:
        first = slider(-1, 1, default=0, name="t")
        second = slider(-5, 5, default=2, name="g")
        release(first)
        assert plot((cos(t), sin(t)), verbose=False).plan.kind == "parametric"
        assert sliders_in(second.symbol * x) == (second,)

    def test_another_name_keeps_both(self, clean_session: object) -> None:
        control = slider(-2, 2, default=1, name="a")
        assert plot(control * sin(x), verbose=False).plan.symbol == x
        assert plot((cos(t), sin(t)), verbose=False).plan.kind == "parametric"


class TestTheRuleStillDoesItsJob:
    """The guard must not undo PRD 5.2 rule 2, only stop it going too far."""

    def test_a_slider_is_still_a_parameter_not_an_axis(
        self, clean_session: object
    ) -> None:
        control = slider(-2, 2, default=1, name="a")
        plan = plot(control * sin(x), verbose=False).plan
        assert plan.symbol == x
        assert plan.kind == "curve"

    def test_a_slider_still_animates(self, clean_session: object) -> None:
        control = slider(-2, 2, default=1, name="a")
        figure = plot(control * sin(x), verbose=False).plotly
        assert len(figure.frames) == len(control.values()) > 1

    def test_two_sliders_and_one_axis_still_work(self, clean_session: object) -> None:
        first = slider(-2, 2, default=1, name="a")
        second = slider(0, 3, default=1, name="b")
        plan = plot(first * sin(second * x), verbose=False).plan
        assert plan.symbol == x

    def test_explicit_parameters_are_unaffected(self, clean_session: object) -> None:
        from mathslate import symbols

        a = symbols("a", real=True)
        assert plot(a * sin(x), parameters={a: 3.0}, verbose=False).plan.symbol == x

    def test_a_constant_plot_is_untouched(self, clean_session: object) -> None:
        slider(0, 3, default=2, name="g")
        assert plot(3, verbose=False).plan.kind == "constant"

    def test_a_matrix_plot_is_untouched(self, clean_session: object) -> None:
        slider(0, 3, default=2, name="x")
        assert plot(Matrix([[2, 1], [1, 3]]), verbose=False).plan.kind == "linalg"


class TestReleaseIsReachable:
    """The remedy the error message names has to be importable from where it says."""

    def test_from_the_ui_package(self) -> None:
        from mathslate.ui import release, release_all  # noqa: F401

    def test_the_registry_empties(self, clean_session: object) -> None:
        control = slider(-1, 1, name="a")
        assert sliders_in(control.symbol) == (control,)
        release_all()
        assert sliders_in(control.symbol) == ()

    def test_release_is_idempotent(self, clean_session: object) -> None:
        control = slider(-1, 1, name="a")
        release(control)
        release(control)
        release_all()


class TestSliderConstruction:
    def test_a_slider_is_still_a_slider(self, clean_session: object) -> None:
        control = slider(-5, 5, default=1, name="a", label="amplitude")
        assert isinstance(control, Slider)
        assert control.value == 1.0 and control.label == "amplitude"

    def test_a_recreated_slider_owns_its_binding(self, clean_session: object) -> None:
        old = slider(0, 1, default=0.2, name="a")
        current = slider(0, 1, default=0.8, name="a")

        release(old)
        assert sliders_in(current.symbol) == (current,)

        with pytest.raises(UnsupportedInputError, match="no longer active"):
            old.value = 0.1
        current.value = 0.7
        assert current.value == 0.7

    def test_value_cannot_leave_the_declared_range(
        self, clean_session: object
    ) -> None:
        control = slider(0, 1, default=0.5, name="a")
        with pytest.raises(UnsupportedInputError, match="outside the slider"):
            control.value = 2

    def test_non_divisible_step_stays_inside_the_range(
        self, clean_session: object
    ) -> None:
        control = slider(0, 1, step=0.26, name="a")
        assert control.values() == pytest.approx((0.0, 0.26, 0.52, 0.78, 1.0))
