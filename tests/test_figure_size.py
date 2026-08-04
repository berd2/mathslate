"""How big a plot is, and whether the sidebar takes a fifth of it.

Plotly's default figure is 450px tall — a dashboard tile's height. A notebook
cell is the full width of the page, the graph is the thing being read rather
than one panel among several, and once the range-control sidebar takes its
fifth of the width a 450px curve reads as a letterbox. So the default is
`DEFAULT_HEIGHT`, and `width=`/`height=`/`set_plot_size()` move it.

Width stays unset by default and that is deliberate: Plotly reads an unset
width as "measure the container", which is what makes a figure fill its cell.
A pixel width would leave a gap beside it on a wide screen and clip it on a
narrow one — the trap §23.1 already fell into with the sidebar split.

`controls=` is the other half: the sidebar is worth its width on a curve you
are exploring and not on one you are only looking at, and that is a per-plot
judgement rather than a notebook-wide one.
"""

from __future__ import annotations

import pickle
import sys
import types
from typing import Any, Iterator

import plotly.graph_objects as go
import pytest

from mathslate import cos, exp, plot, set_plot_size, get_plot_size, sin, t, x, y
from mathslate.errors import UnsupportedInputError
from mathslate.render.options import DEFAULT_HEIGHT


@pytest.fixture(autouse=True)
def _restore_default_size() -> Iterator[None]:
    """`set_plot_size` is process state; no test may leak it to the next."""
    yield
    set_plot_size(reset=True)


def _emitted(result: Any) -> go.Figure:
    namespace: dict[str, object] = {}
    exec(compile(result.python(), "<emitted>", "exec"), namespace)  # noqa: S102
    return namespace["fig"]  # type: ignore[return-value]


class TestTheDefaultHeight:
    def test_it_is_taller_than_plotlys_own(self) -> None:
        """450 is Plotly's; the point of a default here is that it differs."""
        assert DEFAULT_HEIGHT > 450
        assert plot(sin(x), verbose=False).plotly.layout.height == DEFAULT_HEIGHT

    def test_width_stays_unset_so_the_figure_fills_its_cell(self) -> None:
        assert plot(sin(x), verbose=False).plotly.layout.width is None
        assert get_plot_size() == (None, DEFAULT_HEIGHT)

    @pytest.mark.parametrize(
        "build",
        [
            lambda: plot(sin(x), verbose=False),                       # curve
            lambda: plot((cos(t), sin(t)), verbose=False),             # parametric
            lambda: plot(x * y, verbose=False),                        # surface
            lambda: plot(x * y, kind="contour", verbose=False),        # contour
            lambda: plot((cos(t), sin(t), t), verbose=False),          # space
            lambda: plot([1.0, 2.0, 2.0, 3.0] * 5, kind="hist", verbose=False),
        ],
        ids=["curve", "parametric", "surface", "contour", "space", "hist"],
    )
    def test_every_kind_leaves_through_the_same_gate(self, build: Any) -> None:
        """Six builders, one answer — the reason it is applied where it is."""
        assert build().plotly.layout.height == DEFAULT_HEIGHT


class TestSizeOnOneCall:
    def test_height_and_width_are_honoured(self) -> None:
        figure = plot(sin(x), width=1000, height=800, verbose=False).plotly
        assert (figure.layout.width, figure.layout.height) == (1000, 800)

    def test_height_alone_leaves_the_width_responsive(self) -> None:
        figure = plot(sin(x), height=800, verbose=False).plotly
        assert figure.layout.height == 800
        assert figure.layout.width is None

    @pytest.mark.parametrize("bad", [0, -100])
    def test_a_size_that_cannot_be_drawn_is_refused(self, bad: int) -> None:
        with pytest.raises(UnsupportedInputError, match="not a size"):
            plot(sin(x), height=bad, verbose=False)

    def test_a_non_numeric_size_says_what_it_takes(self) -> None:
        with pytest.raises(UnsupportedInputError, match="number of pixels"):
            plot(sin(x), height="tall", verbose=False)

    def test_true_is_not_a_number_of_pixels(self) -> None:
        """`bool` is an `int`; `height=True` is a mistake, not a 1px figure."""
        with pytest.raises(UnsupportedInputError, match="number of pixels"):
            plot(sin(x), height=True, verbose=False)

    def test_an_animation_gets_it_too(self) -> None:
        from mathslate import animate, slider
        from mathslate.ui import interact

        a = slider(1, 3, default=2, name="size_a")
        try:
            moving = animate(a * sin(x), (x, -6, 6), height=700, verbose=False)
        finally:
            interact.release_all()
        assert moving.plotly.layout.height == 700


class TestSetPlotSize:
    def test_it_moves_the_default_for_later_plots(self) -> None:
        set_plot_size(height=720)
        assert plot(sin(x), verbose=False).plotly.layout.height == 720

    def test_one_call_still_wins_over_it(self) -> None:
        set_plot_size(height=720)
        assert plot(sin(x), height=300, verbose=False).plotly.layout.height == 300

    def test_setting_one_dimension_leaves_the_other_alone(self) -> None:
        """Raising the height must not quietly drop a width set a moment ago."""
        set_plot_size(width=900)
        set_plot_size(height=700)
        assert get_plot_size() == (900, 700)

    def test_reset_restores_what_was_shipped(self) -> None:
        set_plot_size(width=900, height=300)
        set_plot_size(reset=True)
        assert get_plot_size() == (None, DEFAULT_HEIGHT)

    def test_a_size_that_cannot_be_drawn_is_refused_here_too(self) -> None:
        with pytest.raises(UnsupportedInputError, match="not a size"):
            set_plot_size(height=0)


class TestShowPythonReproducesTheSize:
    """A figure and a program said to build it must not differ in size."""

    def test_the_default_height_is_emitted(self) -> None:
        result = plot(sin(x), verbose=False)
        assert _emitted(result).layout.height == result.plotly.layout.height

    def test_an_explicit_size_is_emitted(self) -> None:
        result = plot(sin(x), width=1000, height=800, verbose=False)
        emitted = _emitted(result)
        assert (emitted.layout.width, emitted.layout.height) == (1000, 800)

    @pytest.mark.parametrize(
        "build",
        [
            lambda: plot(x * y, verbose=False),
            lambda: plot((cos(t), sin(t), t), verbose=False),
            lambda: plot(exp(x), verbose=False),
            lambda: plot([1.0, 2.0, 2.0, 3.0] * 5, kind="hist", verbose=False),
        ],
        ids=["surface", "space", "curve", "hist"],
    )
    def test_every_emitter_carries_it(self, build: Any) -> None:
        """One insertion point, so a new emitter cannot forget it."""
        result = build()
        assert _emitted(result).layout.height == result.plotly.layout.height


class TestControlsOnOnePlot:
    """`set_range_controls()` is the notebook-wide switch; this is per plot."""

    @pytest.fixture
    def _colab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = types.ModuleType("google.colab")
        monkeypatch.setitem(sys.modules, "google.colab", stub)

    def test_a_curve_shows_the_sidebar_by_default(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        assert plot(sin(x), verbose=False)._wants_live_range_controls()

    def test_controls_false_gives_the_plain_figure_back(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        quiet = plot(sin(x), controls=False, verbose=False)
        assert not quiet._wants_live_range_controls()

    def test_controls_true_overrides_the_notebook_wide_switch(
        self, _colab: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two settings are a preference and an instruction, in that order."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        from mathslate import set_range_controls

        set_range_controls(False)
        try:
            assert not plot(sin(x), verbose=False)._wants_live_range_controls()
            assert plot(
                sin(x), controls=True, verbose=False
            )._wants_live_range_controls()
        finally:
            set_range_controls(True)

    def test_calling_range_controls_by_name_still_works(self, _colab: None) -> None:
        """`controls=False` declines the default display, not the method."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        quiet = plot(sin(x), controls=False, verbose=False)
        assert quiet.range_controls() is not None

    def test_it_survives_a_resample(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        quiet = plot(sin(x), controls=False, verbose=False)
        assert not quiet._replotted(x_range=(0.0, 1.0))._wants_live_range_controls()

    def test_it_survives_the_pickle_the_ai_sandbox_sends_it_through(
        self, _colab: None
    ) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        quiet = pickle.loads(pickle.dumps(plot(sin(x), controls=False, verbose=False)))
        assert not quiet._wants_live_range_controls()

    def test_the_figure_is_unchanged_by_it(self) -> None:
        """A display choice must not reach the figure or the emitted program."""
        plain = plot(sin(x), verbose=False)
        quiet = plot(sin(x), controls=False, verbose=False)
        assert quiet.plotly.layout.height == plain.plotly.layout.height
        assert quiet.python() == plain.python()
