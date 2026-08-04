"""Axis tick labelling — chosen for the window on screen, not the one built.

Ticks used to be decided once, from the plot's initial domain, and never
revisited. That is wrong in both directions the moment a reader zooms:

* the π array holds still while the window moves out from under it, so
  zooming `plot(sin(x))` into a third of a period leaves the one label that
  happens to fall inside and an otherwise bare axis;
* numeric labels grow a digit per decade of zoom — Plotly rounds each one to
  the tick spacing and has no offset line to put the common part in — while it
  goes on asking for the same dozen ticks, until they overlap.

Three things answer that here: `axes.window_ticks` decides, the live sidebar
re-asks it on every zoom, and `ticks=` lets the reader override the count on
any figure including the 3D ones, where Plotly recomputes nothing at all
because a scene's labels are positioned in the projection rather than against
an axis's pixel length.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import plotly.graph_objects as go
import pytest

from mathslate import cos, exp, plot, sin, t, x, y
from mathslate.errors import UnsupportedInputError
from mathslate.render import axes


def _emitted(result: Any) -> go.Figure:
    namespace: dict[str, object] = {}
    exec(compile(result.python(), "<emitted>", "exec"), namespace)  # noqa: S102
    return namespace["fig"]  # type: ignore[return-value]


class TestLabelsShrinkInCountAsTheyGrowInLength:
    """PRD 5.4 — the count and the label width decide each other."""

    def test_a_wide_window_gets_the_full_complement(self) -> None:
        assert axes.numeric_tick_limit(-10.0, 10.0) == 11

    def test_a_deep_zoom_far_from_zero_gets_fewer(self) -> None:
        """`[4.99999999, 5.00000001]` is labelled in full, 12 characters at a time."""
        near = axes.numeric_tick_limit(5.0 - 1e-8, 5.0 + 1e-8)
        assert near < axes.numeric_tick_limit(-10.0, 10.0)
        assert near >= 3

    def test_the_same_span_at_the_origin_needs_no_thinning_of_its_own(self) -> None:
        """The digits come from the *offset*, not from the span alone."""
        assert axes.numeric_tick_limit(-1e-8, 1e-8) >= axes.numeric_tick_limit(
            5.0 - 1e-8, 5.0 + 1e-8
        )

    def test_three_is_the_floor(self) -> None:
        """Two labels are the endpoints and no scale between them.

        A millimetre-wide window a trillion units out needs 19 characters to
        say where it is; nothing thins far enough to make more than three of
        those fit, and the floor is what stops it trying.
        """
        assert axes.numeric_tick_limit(1e12, 1e12 + 1e-3) == 3

    def test_a_window_of_no_width_is_left_to_plotly(self) -> None:
        """Nothing to label yet — an axis mid-autorange, or a constant."""
        assert axes.numeric_tick_limit(5.0, 5.0) == 11

    def test_a_narrow_figure_fits_fewer_than_a_wide_one(self) -> None:
        assert axes.numeric_tick_limit(
            -10.0, 10.0, axis_pixels=200
        ) < axes.numeric_tick_limit(-10.0, 10.0, axis_pixels=1200)

    def test_the_readers_ceiling_is_a_maximum_and_not_a_target(self) -> None:
        """20 labels do not become legible by being asked for."""
        assert axes.numeric_tick_limit(5.0 - 1e-8, 5.0 + 1e-8, ceiling=20) < 20
        assert axes.numeric_tick_limit(-10.0, 10.0, ceiling=4) == 4


class TestPiTicksAreRefittedToTheWindow:
    def test_a_zoom_re_fits_the_spacing_instead_of_keeping_the_old_labels(self) -> None:
        wide = axes.window_ticks(-10.0, 10.0, pi=True)
        near = axes.window_ticks(-4.0, 4.0, pi=True)
        assert wide["tickmode"] == near["tickmode"] == "array"
        assert near["ticktext"] != wide["ticktext"]
        # π/2 over 20 units, π/4 over 8 — the same 5..17 labels either way.
        assert "π/4" in "".join(near["ticktext"])  # type: ignore[arg-type]

    def test_below_a_quarter_period_numbers_are_the_better_answer(self) -> None:
        payload = axes.window_ticks(0.0, 0.5, pi=True)
        assert payload["tickmode"] == "auto"
        assert payload["nticks"]

    def test_the_stale_array_is_cleared_rather_than_left_to_outrank_auto(self) -> None:
        """`tickmode="auto"` alone does not undo `tickvals`; Plotly keeps them."""
        payload = axes.window_ticks(0.0, 0.5, pi=True)
        assert payload["tickvals"] is None
        assert payload["ticktext"] is None

    def test_an_axis_that_was_never_a_pi_axis_is_left_numeric(self) -> None:
        assert axes.window_ticks(-10.0, 10.0, pi=False)["tickmode"] == "auto"


class TestTicksOption:
    """PRD §23.3's shape, applied to labels: off, automatic, or a ceiling."""

    def test_a_count_caps_both_flat_axes(self) -> None:
        figure = plot(exp(x), ticks=5, verbose=False).plotly
        assert figure.layout.xaxis.nticks == 5
        assert figure.layout.yaxis.nticks == 5

    def test_false_removes_the_labels(self) -> None:
        figure = plot(exp(x), ticks=False, verbose=False).plotly
        assert figure.layout.xaxis.showticklabels is False
        assert figure.layout.yaxis.showticklabels is False

    def test_none_leaves_plotly_to_it(self) -> None:
        figure = plot(exp(x), verbose=False).plotly
        assert figure.layout.xaxis.nticks is None
        assert figure.layout.xaxis.showticklabels is None

    def test_true_is_the_automatic_count_and_not_a_cap_of_one(self) -> None:
        """`bool` is an `int`; reading it as a count is the trap `mesh` fell into."""
        figure = plot(exp(x), ticks=True, verbose=False).plotly
        assert figure.layout.xaxis.nticks is None
        assert figure.layout.xaxis.showticklabels is not False

    def test_a_surface_scene_takes_it_on_all_three_axes(self) -> None:
        """The only tick lever 3D has: nothing recomputes a scene's labels."""
        scene = plot(x * y, ticks=4, verbose=False).plotly.layout.scene
        assert scene.xaxis.nticks == scene.yaxis.nticks == scene.zaxis.nticks == 4

    def test_a_space_curve_scene_takes_it_too(self) -> None:
        scene = plot((cos(t), sin(t), t), ticks=6, verbose=False).plotly.layout.scene
        assert scene.xaxis.nticks == scene.yaxis.nticks == scene.zaxis.nticks == 6

    def test_a_contour_is_flat_and_takes_the_flat_form(self) -> None:
        figure = plot(x * y, kind="contour", ticks=4, verbose=False).plotly
        assert figure.layout.xaxis.nticks == 4

    @pytest.mark.parametrize("bad", [0, 1, 2, -3])
    def test_too_few_to_read_is_refused_rather_than_drawn(self, bad: int) -> None:
        with pytest.raises(UnsupportedInputError, match="readable scale"):
            plot(sin(x), ticks=bad, verbose=False)

    def test_a_nonsense_value_says_what_it_takes(self) -> None:
        with pytest.raises(UnsupportedInputError, match="ticks must be"):
            plot(sin(x), ticks="lots", verbose=False)

    def test_an_animation_gets_it_too(self) -> None:
        """`animate()` forwards whatever `_render_options` validates."""
        from mathslate import animate, slider
        from mathslate.ui import interact

        a = slider(1, 3, default=2, name="ticks_a")
        try:
            moving = animate(a * sin(x), (x, -6, 6), ticks=5, verbose=False)
        finally:
            interact.release_all()
        assert moving._options.ticks == 5


class TestShowPythonReproducesTheTicks:
    """`ticks` is in `REPRODUCED`, so the emitted program has to set it."""

    def test_a_capped_flat_axis_round_trips(self) -> None:
        result = plot(exp(x), ticks=5, verbose=False)
        assert _emitted(result).layout.xaxis.nticks == result.plotly.layout.xaxis.nticks

    def test_hidden_labels_round_trip(self) -> None:
        result = plot(exp(x), ticks=False, verbose=False)
        assert _emitted(result).layout.xaxis.showticklabels is False

    def test_a_surface_scene_round_trips(self) -> None:
        result = plot(x * y, ticks=4, verbose=False)
        assert _emitted(result).layout.scene.xaxis.nticks == 4

    def test_a_space_curve_scene_round_trips(self) -> None:
        result = plot((cos(t), sin(t), t), ticks=6, verbose=False)
        assert _emitted(result).layout.scene.zaxis.nticks == 6

    def test_the_default_emits_nothing_about_ticks(self) -> None:
        """Silence is the promise here: an untouched figure gains no line."""
        assert "nticks" not in plot(exp(x), verbose=False).python()


class TestLiveReticking:
    """The sidebar's half — Plotly syncs the range back, so this can follow it."""

    @pytest.fixture
    def _colab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = types.ModuleType("google.colab")
        monkeypatch.setitem(sys.modules, "google.colab", stub)

    @staticmethod
    def _figure(result: Any) -> Any:
        container, _ = result.range_controls().children
        return next(
            child for child in container.children if isinstance(child, go.FigureWidget)
        )

    @staticmethod
    def _sidebar(result: Any) -> Any:
        _, controls = result.range_controls().children
        return controls

    @staticmethod
    def _sliders(controls: Any) -> list[Any]:
        found: list[Any] = []

        def visit(widget: Any) -> None:
            if widget.__class__.__name__ == "IntSlider":
                found.append(widget)
            for child in getattr(widget, "children", ()):
                visit(child)

        visit(controls)
        return found

    def test_zooming_past_the_pi_ticks_falls_back_to_numbers(
        self, _colab: None
    ) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        figure = self._figure(plot(sin(x), verbose=False))
        assert "π" in "".join(figure.layout.xaxis.ticktext)

        figure.layout.xaxis.range = (0.0, 0.5)
        assert figure.layout.xaxis.tickmode == "auto"
        assert not figure.layout.xaxis.ticktext

    def test_zooming_within_them_re_fits_the_spacing(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        figure = self._figure(plot(sin(x), verbose=False))
        before = figure.layout.xaxis.ticktext

        figure.layout.xaxis.range = (-4.0, 4.0)
        after = figure.layout.xaxis.ticktext
        assert after != before
        assert "π" in "".join(after)  # still a π axis, at a finer step

    def test_a_deep_zoom_thins_a_numeric_axis(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        figure = self._figure(plot(exp(x), verbose=False))
        wide = figure.layout.xaxis.nticks

        figure.layout.xaxis.range = (5.0 - 1e-8, 5.0 + 1e-8)
        assert figure.layout.xaxis.nticks < wide

    def test_the_ticks_slider_caps_a_flat_axis_in_place(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(exp(x), verbose=False)
        bundle = result.range_controls()
        container, controls = bundle.children
        figure = next(
            child for child in container.children if isinstance(child, go.FigureWidget)
        )
        ticks = self._sliders(controls)[0]

        ticks.value = 5
        assert figure.layout.xaxis.nticks == 5
        ticks.value = ticks.min  # the "leave it to Plotly" end of the track
        assert figure.layout.xaxis.nticks != 5

    def test_the_ticks_slider_reaches_a_3d_scene(self, _colab: None) -> None:
        """Flat axes re-label in place; a scene's counts need the full redraw."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        bundle = plot(x * y, verbose=False).range_controls()
        container, controls = bundle.children
        figure = next(
            child for child in container.children if isinstance(child, go.FigureWidget)
        )

        self._sliders(controls)[0].value = 4
        assert figure.layout.scene.xaxis.nticks == 4
        assert figure.layout.scene.zaxis.nticks == 4
