"""3D surface mesh, and view-window overrides — PRD §18.

Two look-and-feel controls that a commercial tool has and a bare Plotly wrapper
does not.

**`mesh`** draws grid lines on a 3D surface. A colour gradient alone reads as
flatter than the surface is — Mathematica's Plot3D rules its surfaces for
exactly this reason — so the lines are on by default and `mesh=False` returns
the smooth look.

**`xlim` / `ylim` / `zlim`** set the *view window*, which is not the *domain*.
The domain — where the function is sampled — is the range argument `(x, -10,
10)`. The window is what the axis shows, and the two differ whenever they should:
to undo the automatic y-clip and look at a pole, to zoom, or to put two figures
on one scale.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest
import sympy as sp

from mathslate import cos, exp, plot, sin, slider, t, tan, theta, x, y, z
from mathslate.errors import UnsupportedInputError
from mathslate.ui import release_all

SADDLE = x * y
POLE_SURFACE = 1 / (x * y)
TORUS = (cos(t) * cos(theta), cos(t) * sin(theta), sin(t))


def _emitted(result: Any) -> go.Figure:
    namespace: dict[str, object] = {}
    exec(compile(result.python(), "<emitted>", "exec"), namespace)  # noqa: S102
    return next(v for v in namespace.values() if isinstance(v, go.Figure))


def _unwrap(bundle: Any) -> tuple[Any, Any]:
    """``(figure_widget, controls)`` from a `range_controls()` bundle.

    The figure sits inside a plain `Box` (PRD §23.1) so its wrapper — not
    `FigureWidget` itself, which shadows the ipywidgets CSS layout trait —
    can carry the percentage width the sidebar's own width is measured
    against; `bundle.children[0]` is that `Box`, not the `FigureWidget`.
    """
    figure_container, controls = bundle.children
    figure_widget = next(
        child for child in figure_container.children if isinstance(child, go.FigureWidget)
    )
    return figure_widget, controls


def _number_controls(controls: Any) -> tuple[Any, ...]:
    """FloatText fields, including the compact two-column range grid."""
    found: list[Any] = []

    def visit(widget: Any) -> None:
        if widget.__class__.__name__ == "FloatText":
            found.append(widget)
        for child in getattr(widget, "children", ()):
            visit(child)

    visit(controls)
    return tuple(found)


def _buttons(controls: Any) -> dict[str, Any]:
    """Every Button in a nested controller layout, keyed by its caption."""
    found: dict[str, Any] = {}

    def visit(widget: Any) -> None:
        if widget.__class__.__name__ == "Button":
            found[widget.description] = widget
        for child in getattr(widget, "children", ()):
            visit(child)

    visit(controls)
    return found


def _widgets(controls: Any, class_name: str) -> tuple[Any, ...]:
    """Widgets of one class anywhere inside the compact controller."""
    found: list[Any] = []

    def visit(widget: Any) -> None:
        if widget.__class__.__name__ == class_name:
            found.append(widget)
        for child in getattr(widget, "children", ()):
            visit(child)

    visit(controls)
    return tuple(found)


class TestSurfaceMesh:
    def test_a_surface_is_ruled_by_default(self) -> None:
        contours = plot(SADDLE, verbose=False).plotly.data[0].contours
        assert contours.x.show is True and contours.y.show is True

    def test_a_parametric_surface_is_ruled_too(self) -> None:
        contours = plot(TORUS, (t, -1.5, 1.5), (theta, 0, 6.28),
                        verbose=False).plotly.data[0].contours
        assert contours.x.show is True and contours.y.show is True

    def test_mesh_false_returns_the_smooth_look(self) -> None:
        trace = plot(SADDLE, mesh=False, verbose=False).plotly.data[0]
        # No grid drawn: `contours.x.show` was never set to True.
        assert not trace.contours.x.show

    def test_both_families_are_drawn_so_it_is_a_grid(self) -> None:
        """One family alone is parallel ribbons, not a mesh."""
        contours = plot(SADDLE, verbose=False).plotly.data[0].contours
        assert contours.x.show and contours.y.show

    def test_the_lines_do_not_recolour_the_surface(self) -> None:
        """`usecolormap` off: the mesh sits over the fill, it does not replace it."""
        contours = plot(SADDLE, verbose=False).plotly.data[0].contours
        assert not contours.x.usecolormap

    def test_a_flat_kind_ignores_mesh_without_error(self) -> None:
        """Contour, region and the 2D curve have no surface to rule."""
        assert plot(sin(x), mesh=True, verbose=False).plan.kind == "curve"
        assert plot(x * y, kind="contour", mesh=True, verbose=False).plan.kind == "contour"

    def test_show_python_reproduces_the_mesh(self, headless_show: list[Any]) -> None:
        ruled = plot(SADDLE, verbose=False)
        assert _emitted(ruled).data[0].contours.x.show is True

    def test_show_python_reproduces_its_absence(self, headless_show: list[Any]) -> None:
        smooth = plot(SADDLE, mesh=False, verbose=False)
        assert not _emitted(smooth).data[0].contours.x.show


class TestMeshDensity:
    """PRD §23.3 — Plotly's own automatic line spacing read as sparse."""

    def test_the_default_is_denser_than_no_spacing_at_all(self) -> None:
        """Leaving `size` unset is what produced the reported sparse mesh —
        this pins that a `size` is always computed now."""
        contours = plot(SADDLE, verbose=False).plotly.data[0].contours
        assert contours.x.size is not None
        assert contours.y.size is not None

    def test_an_explicit_count_changes_the_spacing(self) -> None:
        span = 10.0  # SADDLE's default domain is (-5, 5) on both axes
        sparse = plot(SADDLE, mesh=5, verbose=False).plotly.data[0].contours
        dense = plot(SADDLE, mesh=50, verbose=False).plotly.data[0].contours
        assert sparse.x.size == pytest.approx(span / 5)
        assert dense.x.size == pytest.approx(span / 50)
        assert dense.x.size < sparse.x.size
        assert (dense.x.start, dense.x.end) == pytest.approx((-5.0, 5.0))
        assert (dense.y.start, dense.y.end) == pytest.approx((-5.0, 5.0))

    def test_the_count_scales_with_the_actual_domain(self) -> None:
        """The same `mesh=` line count means the same number of lines
        whether the surface spans 2 units or 2000, not the same spacing."""
        narrow = plot(x * y, (x, -1, 1), (y, -1, 1), mesh=20, verbose=False)
        wide = plot(x * y, (x, -1000, 1000), (y, -1000, 1000), mesh=20, verbose=False)
        narrow_lines = 2.0 / narrow.plotly.data[0].contours.x.size
        wide_lines = 2000.0 / wide.plotly.data[0].contours.x.size
        assert narrow_lines == pytest.approx(wide_lines)

    @pytest.mark.parametrize("bad", [1, 0, -3])
    def test_too_few_lines_is_refused(self, bad: int) -> None:
        with pytest.raises(UnsupportedInputError, match="too few lines"):
            plot(SADDLE, mesh=bad, verbose=False)

    def test_a_non_bool_non_int_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="True, False, or a positive"):
            plot(SADDLE, mesh="dense", verbose=False)  # type: ignore[arg-type]

    def test_show_python_reproduces_the_density(self, headless_show: list[Any]) -> None:
        """codegen.py used to hand-write the mesh kwarg and always omitted
        `size`, so show_python() drew Plotly's sparser default regardless of
        what the real figure showed — this is that fidelity gap."""
        ruled = plot(SADDLE, mesh=15, verbose=False)
        assert _emitted(ruled).data[0].contours.x.size == pytest.approx(
            ruled.plotly.data[0].contours.x.size
        )

    def test_show_python_reproduces_the_density_on_a_parametric_surface(
        self, headless_show: list[Any]
    ) -> None:
        ruled = plot(TORUS, (t, -1.5, 1.5), (theta, 0, 6.28), mesh=15, verbose=False)
        assert _emitted(ruled).data[0].contours.x.size == pytest.approx(
            ruled.plotly.data[0].contours.x.size
        )


class TestViewLimitsOnACurve:
    def test_xlim_sets_the_window(self) -> None:
        result = plot(sin(x), (x, -10, 10), xlim=(-3, 3), verbose=False)
        assert tuple(result.plotly.layout.xaxis.range) == (-3.0, 3.0)

    def test_xlim_is_the_window_not_the_domain(self) -> None:
        """The curve is still sampled over the whole range, only shown zoomed."""
        result = plot(sin(x), (x, -10, 10), xlim=(-3, 3), verbose=False)
        xs = result.plan.series[0].sample.x
        assert xs.min() < -9 and xs.max() > 9

    def test_ylim_overrides_the_automatic_clip(self) -> None:
        """Auto-clip hides a pole; ylim is how you ask to see it."""
        auto = plot(tan(x), (x, -4, 4), verbose=False)
        shown = plot(tan(x), (x, -4, 4), ylim=(-50, 50), verbose=False)
        assert auto.plotly.layout.yaxis.range != (-50.0, 50.0)
        assert tuple(shown.plotly.layout.yaxis.range) == (-50.0, 50.0)

    def test_show_python_carries_both(self, headless_show: list[Any]) -> None:
        result = plot(sin(x), (x, -10, 10), xlim=(-3, 3), ylim=(-2, 2), verbose=False)
        emitted = _emitted(result)
        assert tuple(emitted.layout.xaxis.range) == (-3.0, 3.0)
        assert tuple(emitted.layout.yaxis.range) == (-2.0, 2.0)


class TestViewLimitsOnASurface:
    def test_zlim_overrides_the_z_clip(self) -> None:
        auto = plot(POLE_SURFACE, verbose=False)
        shown = plot(POLE_SURFACE, zlim=(-10, 10), verbose=False)
        assert tuple(auto.plotly.layout.scene.zaxis.range) != (-10.0, 10.0)
        assert tuple(shown.plotly.layout.scene.zaxis.range) == (-10.0, 10.0)

    def test_show_python_carries_zlim(self, headless_show: list[Any]) -> None:
        result = plot(POLE_SURFACE, zlim=(-8, 8), verbose=False)
        assert tuple(_emitted(result).layout.scene.zaxis.range) == (-8.0, 8.0)

    def test_a_smooth_surface_still_gets_no_axis_unless_asked(self) -> None:
        assert plot(x + y, verbose=False).plotly.layout.scene.zaxis.range is None
        bounded = plot(x + y, zlim=(-3, 3), verbose=False)
        assert tuple(bounded.plotly.layout.scene.zaxis.range) == (-3.0, 3.0)


class TestViewLimitsOnAFlatRegion:
    def test_xlim_and_ylim_zoom_a_region(self) -> None:
        result = plot(x**2 + y**2 < 4, xlim=(-1, 1), ylim=(-1, 1), verbose=False)
        assert tuple(result.plotly.layout.xaxis.range) == (-1.0, 1.0)
        assert tuple(result.plotly.layout.yaxis.range) == (-1.0, 1.0)


class TestTheRefusals:
    @pytest.mark.parametrize("name", ["xlim", "ylim", "zlim"])
    def test_a_non_pair_is_refused(self, name: str) -> None:
        with pytest.raises(UnsupportedInputError, match="pair of numbers"):
            plot(sin(x), **{name: (1, 2, 3)}, verbose=False)

    @pytest.mark.parametrize("name", ["xlim", "ylim"])
    def test_a_backwards_window_is_refused(self, name: str) -> None:
        with pytest.raises(UnsupportedInputError, match="low < high"):
            plot(sin(x), **{name: (5, 1)}, verbose=False)

    def test_an_equal_window_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="low < high"):
            plot(sin(x), xlim=(2, 2), verbose=False)

    def test_a_non_numeric_window_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="pair of numbers"):
            plot(sin(x), xlim=("a", "b"), verbose=False)


class TestEveryEntryPointAndKindHonoursThem:
    """The review found three ways an accepted option quietly did nothing."""

    def test_animate_honours_a_view_limit(self) -> None:
        """`animate()` built its own RenderOptions and listed the fields by hand,
        so mesh/xlim/ylim/zlim were dropped from every animation."""
        from mathslate import animate, slider
        from mathslate.ui import interact

        a = slider(1, 3, default=2, name="anim_view_a")
        try:
            moving = animate(a * sin(x), (x, -6, 6), ylim=(-9, 9), verbose=False)
        finally:
            interact.release_all()
        assert tuple(moving.plotly.layout.yaxis.range) == (-9.0, 9.0)

    def test_animate_validates_a_view_limit(self) -> None:
        """Validation lived in plot(), so animate() accepted nonsense in silence."""
        from mathslate import animate, slider
        from mathslate.ui import interact

        a = slider(1, 3, default=2, name="anim_view_b")
        try:
            with pytest.raises(UnsupportedInputError, match="pair of numbers"):
                animate(a * sin(x), (x, -6, 6), ylim="junk", verbose=False)
        finally:
            interact.release_all()

    @pytest.mark.parametrize(
        "build",
        [
            lambda: plot([1.0, 2.0, 2.0, 3.0] * 5, kind="hist", xlim=(0, 4), verbose=False),
            lambda: plot([1.0, 2.0, 2.0, 9.0] * 5, kind="box", ylim=(0, 4), verbose=False),
            lambda: plot(sp.Matrix([[2, 1], [1, 3]]), xlim=(0, 4), verbose=False),
            lambda: plot(x**2 + y**2 < 4, xlim=(0, 4), verbose=False),
            lambda: plot(sin(x), (x, -10, 10), xlim=(0, 4), verbose=False),
        ],
        ids=["hist", "box", "linalg", "region", "curve"],
    )
    def test_a_view_limit_reaches_every_flat_kind(self, build: Any) -> None:
        """hist, box and linalg ignored it entirely: their builders never asked."""
        layout = build().plotly.layout
        window = tuple(layout.xaxis.range or ()) or tuple(layout.yaxis.range or ())
        assert window == (0.0, 4.0), "the option was accepted and then ignored"

    def test_a_region_cannot_be_analysed_but_says_so(self) -> None:
        """It used to raise a raw SymPy TypeError from inside `limit()`."""
        with pytest.raises(UnsupportedInputError, match="set rather than a function"):
            plot(x * y < 1, verbose=False).analyze()

    def test_a_band_can_still_be_analysed(self) -> None:
        """The guard must not catch the kind that does have a y = f(x)."""
        report = plot(sin(x) > 0, (x, -6, 6), verbose=False).analyze()
        assert "pi" in report.roots.describe()


class TestItSurvivesTheSliderPath:
    def test_view_limits_reach_an_interactive_plot(self) -> None:
        """The slider path rebuilds the plan but must keep the render options."""
        from mathslate import slider
        from mathslate.ui import interact

        a = slider(1, 3, default=2, name="view_mesh_a")
        try:
            result = plot(a * sin(x), (x, -6, 6), ylim=(-4, 4), verbose=False)
        finally:
            interact.release_all()
        assert result.interactive
        assert tuple(result.plotly.layout.yaxis.range) == (-4.0, 4.0)


class TestLiveRangeResampling:
    """PRD §19 — the gap between `xlim` and Plotly's own drag-to-zoom.

    Both of those only crop what was already sampled; narrowing the view into
    a tenth of the domain still shows a tenth of the original points.
    ``_replotted`` is the pure rebuild `range_controls()` calls on every edit,
    and is tested directly here because it needs neither a notebook kernel
    nor ipywidgets to be worth getting right.
    """

    def test_narrowing_x_resamples_at_full_resolution(self) -> None:
        wide = plot(sin(x) / x, (x, -10, 10), verbose=False)
        narrow = wide._replotted(x_range=(-1.0, 1.0))
        assert narrow.plan.param_range == (-1.0, 1.0)
        xs = narrow.numpy[0]
        finite = xs[np.isfinite(xs)]
        assert finite.min() > -1.0001 and finite.max() < 1.0001
        # A mere view-crop of the wide sample would land ~20 of its points in
        # this span (200-ish points spread over 20 units); a real resample
        # gives it the full initial-point budget instead.
        assert narrow.plan.total_points > 100

    def test_y_alone_is_a_view_clip_not_a_resample_on_a_curve(self) -> None:
        """Y is derived from X on an ordinary curve; there is nothing to resample."""
        base = plot(sin(x) / x, (x, -10, 10), verbose=False)
        clipped = base._replotted(y_range=(-0.3, 0.9))
        assert clipped.plan.total_points == base.plan.total_points
        assert clipped.plan.param_range == base.plan.param_range
        assert tuple(clipped.plotly.layout.yaxis.range) == (-0.3, 0.9)

    def test_both_axes_resample_on_a_surface(self) -> None:
        """A surface's Y is a domain too, unlike an ordinary curve's."""
        wide = plot(x * y, verbose=False)
        narrow = wide._replotted(x_range=(-1.0, 1.0), y_range=(-2.0, 2.0))
        assert narrow.plan.param_range == (-1.0, 1.0)
        assert narrow.plan.second_range == (-2.0, 2.0)

    def test_raw_data_has_nothing_to_resample(self) -> None:
        raw = plot([1.0, 2.0, 2.0, 3.0] * 5, kind="hist", verbose=False)
        with pytest.raises(UnsupportedInputError, match="no symbolic domain"):
            raw.range_controls()

    def test_a_plain_script_is_told_to_use_marimo_or_xlim(self) -> None:
        result = plot(sin(x), verbose=False)
        with pytest.raises(UnsupportedInputError, match="marimo"):
            result.range_controls()


class TestLiveRangeResamplingWidget:
    """The ipywidgets path — best-effort, like `Slider.widget()`'s (PRD 6.2)."""

    @pytest.fixture
    def _colab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = types.ModuleType("google.colab")
        monkeypatch.setitem(sys.modules, "google.colab", stub)

    def test_moving_only_x_resamples_and_leaves_y_alone(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        bundle = result.range_controls()
        figure_widget, controls = _unwrap(bundle)
        x_min, x_max, y_min, y_max = _number_controls(controls)

        before = len(figure_widget.data[0].x)
        x_min.value, x_max.value = -1.0, 1.0
        assert len(figure_widget.data[0].x) != before
        xs = np.asarray(figure_widget.data[0].x, dtype=np.float64)
        finite = xs[np.isfinite(xs)]
        assert finite.min() > -1.0001 and finite.max() < 1.0001

    def test_moving_only_y_leaves_the_point_count_alone(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        bundle = result.range_controls()
        figure_widget, controls = _unwrap(bundle)
        x_min, x_max, y_min, y_max = _number_controls(controls)

        before = len(figure_widget.data[0].x)
        y_min.value, y_max.value = -0.3, 0.9
        assert len(figure_widget.data[0].x) == before
        assert tuple(figure_widget.layout.yaxis.range) == (-0.3, 0.9)

    def test_a_mid_edit_inconsistent_pair_is_left_alone(self, _colab: None) -> None:
        """min temporarily exceeding max, while the reader is still typing."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        bundle = result.range_controls()
        figure_widget, controls = _unwrap(bundle)
        x_min, x_max, _, _ = _number_controls(controls)

        before = tuple(figure_widget.layout.xaxis.range or ())
        x_min.value = 50.0  # now greater than x_max — an inconsistent mid-edit
        assert tuple(figure_widget.layout.xaxis.range or ()) == before

    def test_a_non_finite_edit_is_ignored_until_it_becomes_a_real_range(
        self, _colab: None
    ) -> None:
        """FloatText accepts NaN/Infinity, but they must not escape a callback."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        figure_widget, controls = _unwrap(result.range_controls())
        x_min, x_max, _, _ = _number_controls(controls)

        before = tuple(figure_widget.layout.xaxis.range or ())
        x_min.value = float("nan")
        assert tuple(figure_widget.layout.xaxis.range or ()) == before

        x_min.value, x_max.value = -1.0, 1.0
        assert tuple(figure_widget.layout.xaxis.range) == (-1.0, 1.0)

    def test_the_replot_boundary_refuses_non_finite_ranges(self) -> None:
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        with pytest.raises(UnsupportedInputError, match="finite numbers"):
            result._replotted(x_range=(float("nan"), 1.0))


class TestAutoYFitsTheCurrentXDomain:
    """Auto Y must answer "what Y does this X window reach", on any curve.

    The auto-*clip* (`plan.y_range`) is deliberately unset wherever a curve is
    well behaved enough not to need clipping, so reading it alone answered
    "nothing" for `x**2`, `sin(x)` and most of the corpus.
    """

    @pytest.fixture
    def _colab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = types.ModuleType("google.colab")
        monkeypatch.setitem(sys.modules, "google.colab", stub)

    @staticmethod
    def _buttons(controls: Any) -> dict[str, Any]:
        return _buttons(controls)

    def test_a_curve_with_no_clip_still_gets_a_fitted_window(
        self, _colab: None
    ) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(x**2, (x, -10, 10), verbose=False)
        assert result.plan.y_range is None  # the precondition: nothing to clip
        figure_widget, controls = _unwrap(result.range_controls())
        x_min, x_max, y_min, y_max = _number_controls(controls)

        x_min.value, x_max.value = 0.0, 2.0
        self._buttons(controls)["Fit Y"].click()

        assert (y_min.value, y_max.value) == pytest.approx((0.0, 4.0), abs=0.05)
        assert tuple(figure_widget.layout.yaxis.range) == pytest.approx(
            (y_min.value, y_max.value)
        )

    def test_the_boxes_start_at_the_window_actually_drawn(self, _colab: None) -> None:
        """They are pushed back as `ylim` on the next edit, so a placeholder
        unrelated to the plot does not just look wrong — it crops it."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(x**2, (x, -10, 10), verbose=False)
        figure_widget, controls = _unwrap(result.range_controls())
        x_min, x_max, y_min, y_max = _number_controls(controls)
        assert (y_min.value, y_max.value) == pytest.approx((0.0, 100.0), abs=0.5)

        # The crop is a *view*, so the samples are no evidence either way —
        # `data[0].y` still reaches 100 with the axis pinned to -1..1.
        x_min.value = -5.0
        assert tuple(figure_widget.layout.yaxis.range) == pytest.approx(
            (y_min.value, y_max.value)
        )
        assert figure_widget.layout.yaxis.range[1] > 20.0

    def test_a_pole_still_keeps_its_clip_rather_than_the_raw_extent(
        self, _colab: None
    ) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(tan(x), (x, -10, 10), verbose=False)
        clip = result.plan.y_range
        assert clip is not None  # the precondition: a clip is active
        _, controls = _unwrap(result.range_controls())
        _, _, y_min, y_max = _number_controls(controls)

        y_min.value, y_max.value = -500.0, 500.0
        self._buttons(controls)["Fit Y"].click()
        assert (y_min.value, y_max.value) == pytest.approx(clip)

    def test_auto_y_escapes_an_explicit_ylim(self, _colab: None) -> None:
        """`ylim` is the window the reader asked for; Auto Y is them asking
        for a different one, so it must clear it rather than fit to it."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(x**2, (x, -10, 10), ylim=(-1.0, 1.0), verbose=False)
        _, controls = _unwrap(result.range_controls())
        _, _, y_min, y_max = _number_controls(controls)
        assert (y_min.value, y_max.value) == (-1.0, 1.0)

        self._buttons(controls)["Fit Y"].click()
        assert y_max.value == pytest.approx(100.0, abs=0.5)

    def test_auto_y_keeps_a_removable_singularity_peak(self, _colab: None) -> None:
        """A removable hole is bounded data, not a pole to clip away.

        Over a wide interval the ``sin(x)/x`` peak at zero is easy for a
        percentile-based pole guard to mistake for an outlier.  Fit Y must
        retain it, otherwise the displayed upper bound wrongly lands near
        0.5 instead of the function's limiting value of one.
        """
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -40, 40), verbose=False)
        figure_widget, controls = _unwrap(result.range_controls())
        _, _, y_min, y_max = _number_controls(controls)

        self._buttons(controls)["Fit Y"].click()

        assert y_min.value < -0.2
        assert y_max.value > 0.99
        assert tuple(figure_widget.layout.yaxis.range) == pytest.approx(
            (y_min.value, y_max.value)
        )

    def test_on_a_log_axis_the_boxes_speak_powers_of_ten(self, _colab: None) -> None:
        """Plotly reads a log axis's range as exponents, and `ylim` already
        follows that there — so seeding the boxes with raw sample values would
        ask for a window of 10**22026 the moment anything else was edited."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(exp(x), (x, -10, 10), yscale="log", verbose=False)
        assert result.plotly.layout.yaxis.type == "log"  # the precondition
        figure_widget, controls = _unwrap(result.range_controls())
        x_min, x_max, y_min, y_max = _number_controls(controls)

        # exp(-10) .. exp(10) is 1e-4.34 .. 1e4.34, not 4.5e-05 .. 22026.
        assert (y_min.value, y_max.value) == pytest.approx((-4.343, 4.343), abs=0.01)

        x_min.value, x_max.value = 0.0, 2.0
        self._buttons(controls)["Fit Y"].click()
        assert (y_min.value, y_max.value) == pytest.approx((0.0, 0.869), abs=0.01)
        assert tuple(figure_widget.layout.yaxis.range) == pytest.approx(
            (y_min.value, y_max.value)
        )

    def test_auto_z_escapes_an_explicit_zlim(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(x**2 - y**2, (x, -5, 5), (y, -5, 5), zlim=(-1.0, 1.0), verbose=False)
        _, controls = _unwrap(result.range_controls())
        *_, z_min, z_max = _number_controls(controls)
        assert (z_min.value, z_max.value) == (-1.0, 1.0)

        self._buttons(controls)["Fit Z"].click()
        assert z_max.value == pytest.approx(25.0, abs=0.5)


class TestRangeControlsOnA3DSurface:
    """PRD §23.2 — a real 3D surface also gets z min/max, view-only."""

    @pytest.fixture
    def _colab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = types.ModuleType("google.colab")
        monkeypatch.setitem(sys.modules, "google.colab", stub)

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            (lambda: plot(SADDLE, verbose=False), 6),
            (lambda: plot(x * y, kind="contour", verbose=False), 4),
            (lambda: plot(x**2 + y**2 < 4, verbose=False), 4),
        ],
        ids=["surface", "contour", "region"],
    )
    def test_only_a_true_surface_gets_z_boxes(
        self, build: Any, expected: int, _colab: None
    ) -> None:
        """`contour`/`region` are two-variable too but flat — no Z to window."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        _, controls = _unwrap(build().range_controls())
        assert len(_number_controls(controls)) == expected

    def test_z_seeds_from_the_actual_data_when_no_clip_is_active(
        self, _colab: None
    ) -> None:
        """SADDLE has no pole, so `sample.z_range` (the auto-*clip*) is
        `None` — the box must not seed from that absence, or it shows a
        number with no relationship to what is on screen."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(SADDLE, verbose=False)
        assert result.plan.series[0].sample.z_range is None  # the precondition
        _, controls = _unwrap(result.range_controls())
        *_, z_min, z_max = _number_controls(controls)
        z_data = result.plan.series[0].sample.z
        assert z_min.value == pytest.approx(float(z_data.min()))
        assert z_max.value == pytest.approx(float(z_data.max()))

    def test_z_seeds_from_the_clip_when_one_is_active(self, _colab: None) -> None:
        result = plot(POLE_SURFACE, verbose=False)
        clip = result.plan.series[0].sample.z_range
        assert clip is not None  # the precondition: a pole forces a clip
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        _, controls = _unwrap(result.range_controls())
        *_, z_min, z_max = _number_controls(controls)
        assert (z_min.value, z_max.value) == pytest.approx(clip)

    def test_moving_z_clips_the_view_without_resampling(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(SADDLE, verbose=False)
        figure_widget, controls = _unwrap(result.range_controls())
        *_, z_min, z_max = _number_controls(controls)

        before = len(figure_widget.data[0].x)
        z_min.value, z_max.value = -5.0, 5.0
        assert len(figure_widget.data[0].x) == before  # no resample
        assert tuple(figure_widget.layout.scene.zaxis.range) == (-5.0, 5.0)

    def test_a_later_x_change_keeps_the_z_clip(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(SADDLE, verbose=False)
        figure_widget, controls = _unwrap(result.range_controls())
        x_min, x_max, _, _, z_min, z_max = _number_controls(controls)

        z_min.value, z_max.value = -5.0, 5.0
        x_min.value, x_max.value = -2.0, 2.0
        assert tuple(figure_widget.layout.scene.zaxis.range) == (-5.0, 5.0)

    def test_a_z_mid_edit_is_left_alone(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(SADDLE, verbose=False)
        figure_widget, controls = _unwrap(result.range_controls())
        *_, z_min, z_max = _number_controls(controls)

        before = figure_widget.layout.scene.zaxis.range
        z_min.value = 999.0  # now greater than z_max
        assert figure_widget.layout.scene.zaxis.range == before

    def test_z_range_buttons_zoom_the_surface_view(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(SADDLE, verbose=False)
        figure_widget, controls = _unwrap(result.range_controls())
        *_, z_min, z_max = _number_controls(controls)
        buttons = _buttons(controls)
        action_rows = [
            row for row in _widgets(controls, "HBox")
            if row.children and all(child.__class__.__name__ == "Button" for child in row.children)
        ]
        assert "Fit Z" in buttons
        assert "Fit Y" not in buttons
        assert [len(row.children) for row in action_rows] == [4, 4]
        assert [button.description for row in action_rows for button in row.children] == [
            "Fit Z", "[X]+", "[Y]+", "[Z]+", "Reset", "[X]−", "[Y]−", "[Z]−",
        ]
        before = z_max.value - z_min.value
        buttons["[Z]−"].click()
        assert z_max.value - z_min.value == pytest.approx(before / 2)
        assert tuple(figure_widget.layout.scene.zaxis.range) == pytest.approx(
            (z_min.value, z_max.value)
        )

        z_min.value, z_max.value = -999.0, 999.0
        buttons["Fit Z"].click()
        assert tuple(figure_widget.layout.scene.zaxis.range) != (-999.0, 999.0)

    def test_a_space_curve_gets_parameter_and_z_controls(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot((cos(t), sin(t), t), (t, 0, 12), verbose=False)
        figure_widget, controls = _unwrap(result.range_controls())
        t_min, t_max, _, _, z_min, z_max = _number_controls(controls)
        assert (t_min.description, t_max.description) == ("", "")
        buttons = _buttons(controls)
        assert {"[t]+", "[t]−", "Fit Z", "[Z]+", "[Z]−"} <= set(buttons)

        z_min.value, z_max.value = 1.0, 3.0
        assert tuple(figure_widget.layout.scene.zaxis.range) == (1.0, 3.0)

    def test_a_space_curve_gets_thickness_but_not_surface_mesh_controls(
        self, _colab: None
    ) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot((cos(t), sin(t), t), (t, 0, 13.5), verbose=False)
        figure_widget, controls = _unwrap(result.range_controls())
        thickness, = _widgets(controls, "IntSlider")

        assert not any(
            toggle.description in {"On", "Off"}
            for toggle in _widgets(controls, "ToggleButton")
        )
        thickness.value = 150
        assert figure_widget.data[0].line.width == pytest.approx(6.0)

    def test_auto_z_on_a_space_curve_keeps_its_actual_y_extent(self, _colab: None) -> None:
        """Auto Z must not redraw a wide space curve through a -1..1 Y window."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(
            (
                cos(t) * (3 + cos(5 * t)),
                sin(t) * (3 + cos(5 * t)),
                sin(5 * t),
            ),
            (t, 0, 2 * sp.pi),
            verbose=False,
        )
        figure_widget, controls = _unwrap(result.range_controls())
        _, _, y_min, y_max, z_min, z_max = _number_controls(controls)
        sample = result.plan.series[0].sample
        assert (y_min.value, y_max.value) == pytest.approx(
            (float(sample.y.min()), float(sample.y.max()))
        )

        z_min.value, z_max.value = -20.0, 20.0
        buttons = _buttons(controls)
        buttons["Fit Z"].click()

        assert (z_min.value, z_max.value) == pytest.approx(
            (float(sample.z.min()), float(sample.z.max()))
        )
        assert tuple(figure_widget.layout.scene.yaxis.range) == pytest.approx(
            (float(sample.y.min()), float(sample.y.max()))
        )


class TestRangeControlsDisplayController:
    @pytest.fixture
    def _colab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = types.ModuleType("google.colab")
        monkeypatch.setitem(sys.modules, "google.colab", stub)

    def test_a_positive_2d_curve_gets_scale_and_mode_switches(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        import ipywidgets as widgets

        figure_widget, controls = _unwrap(plot(exp(x), verbose=False).range_controls())
        switches = _widgets(controls, "ToggleButtons")
        scale_switch, = switches
        trace_switch = next(
            switch for switch in _widgets(controls, "ToggleButton")
            if switch.description == "Points"
        )
        thickness, = _widgets(controls, "IntSlider")
        rows = [
            row for row in _widgets(controls, "HBox")
            if len(row.children) == 2 and isinstance(row.children[0], widgets.Label)
        ]
        action_rows = [
            row for row in _widgets(controls, "HBox")
            if row.children and all(child.__class__.__name__ == "Button" for child in row.children)
        ]

        assert scale_switch.description == ""
        assert "flex-flow: row nowrap" in next(
            widget for widget in controls.children
            if isinstance(widget, widgets.HTML)
        ).value
        assert [row.children[0].value for row in rows] == ["Scale", "Mode"]
        assert [len(row.children) for row in action_rows] == [3, 3]
        assert [button.description for row in action_rows for button in row.children] == [
            "Fit Y", "[X]+", "[Y]+", "Reset", "[X]−", "[Y]−",
        ]
        scale_switch.value = "log"
        trace_switch.value = True

        assert figure_widget.layout.yaxis.type == "log"
        assert figure_widget.data[0].mode == "markers"
        thickness.value = 150
        assert figure_widget.data[0].line.width == pytest.approx(3.0)
        assert figure_widget.data[0].marker.size == pytest.approx(9.0)

    def test_a_2d_controller_keeps_display_choices_after_resampling(
        self, _colab: None
    ) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        figure_widget, controls = _unwrap(plot(exp(x), verbose=False).range_controls())
        switches = _widgets(controls, "ToggleButtons")
        switches[0].value = "log"
        next(
            switch for switch in _widgets(controls, "ToggleButton")
            if switch.description == "Points"
        ).value = True
        x_min, x_max, _, _ = _number_controls(controls)
        x_min.value, x_max.value = -2, 2

        assert figure_widget.layout.yaxis.type == "log"
        assert figure_widget.data[0].mode == "markers"

    def test_surface_gets_z_scale_and_mesh_controls_only(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        figure_widget, controls = _unwrap(plot(exp(x + y), verbose=False).range_controls())
        z_scale_switch = next(
            switch for switch in _widgets(controls, "HBox")
            if all(hasattr(button, "description") for button in switch.children)
            and [button.description for button in switch.children] == ["Linear", "Log"]
        )
        z_log = z_scale_switch.children[1]
        mesh_switch = next(
            switch for switch in _widgets(controls, "ToggleButton")
            if switch.description == "Off"
        )
        density, = _widgets(controls, "IntSlider")

        assert z_scale_switch.layout.width == "148px"
        assert [button.layout.width for button in z_scale_switch.children] == ["72px", "72px"]
        assert mesh_switch.value is False
        z_log.value = True
        mesh_switch.value = True
        assert figure_widget.layout.scene.zaxis.type == "log"
        assert density.disabled is True


class TestRangeControlsLayoutAndSize:
    """PRD §22 — the controls sit beside the figure, and its size is settable."""

    @pytest.fixture
    def _colab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = types.ModuleType("google.colab")
        monkeypatch.setitem(sys.modules, "google.colab", stub)

    def test_the_figure_comes_first_and_controls_sit_beside_it(
        self, _colab: None
    ) -> None:
        pytest.importorskip("ipywidgets")
        import ipywidgets as widgets

        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        bundle = result.range_controls()
        assert isinstance(bundle, widgets.HBox)
        figure_widget, controls = _unwrap(bundle)
        assert isinstance(figure_widget, go.FigureWidget)
        assert isinstance(controls, widgets.VBox)
        assert len(_number_controls(controls)) == 4

    def test_width_and_height_size_the_figure(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        figure_widget, _ = _unwrap(result.range_controls(width=800, height=400))
        assert figure_widget.layout.width == 800
        assert figure_widget.layout.height == 400

    def test_height_only_keeps_the_figure_width_responsive(self, _colab: None) -> None:
        """A height override must not bring back Plotly's 700px default width."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        bundle = result.range_controls(height=400)
        figure_container, _ = bundle.children
        figure_widget, _ = _unwrap(bundle)

        assert figure_widget.layout.height == 400
        assert figure_widget.layout.width is None
        assert figure_widget._config.get("responsive") is True
        assert figure_container.layout.flex == "1 1 0"
        assert figure_container.layout.min_width == "0"

    def test_left_unset_the_split_is_a_percentage_close_to_80_20(
        self, _colab: None
    ) -> None:
        """The first regression: the sidebar had grown to ~40% of the
        widget's width, not the ~20% asked for. The second: a fixed-pixel
        split (§22.4's first fix) left blank space once the notebook was
        wider than the pixel total (§23.1) — a percentage on both sides,
        summing to 100%, is what this pins instead of a pixel count."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        import ipywidgets as widgets

        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        bundle = result.range_controls()
        figure_container, controls = bundle.children
        assert isinstance(figure_container, widgets.Box)

        figure_percent = int(str(figure_container.layout.width).removesuffix("%"))
        sidebar_percent = int(str(controls.layout.width).removesuffix("%"))
        assert bundle.layout.width == "100%"
        assert bundle.layout.display == "flex"
        assert bundle.layout.flex_flow == "row nowrap"
        assert bundle.layout.align_items == "stretch"
        assert figure_container.layout.flex == "1 1 0"
        assert figure_container.layout.min_width == "0"
        assert controls.layout.flex == f"0 0 {sidebar_percent}%"
        style, figure_widget = figure_container.children
        assert isinstance(style, widgets.HTML)
        assert "width: 100% !important" in style.value
        assert "mathslate-range-controls-figure" in figure_widget._dom_classes
        figure_widget._view_count = 1
        assert figure_widget.layout.autosize is True
        assert figure_percent + sidebar_percent == 100
        assert 15 < sidebar_percent < 25, (
            f"sidebar is {sidebar_percent}% of the total width, not close to 20%"
        )
        # The figure itself stays unsized in `layout` — a pixel size there
        # would fight the percentage wrapper instead of matching it.
        figure_widget, _ = _unwrap(bundle)
        assert figure_widget.layout.width is None

    def test_left_unset_the_figure_actually_fills_its_wrapper(
        self, _colab: None
    ) -> None:
        """The third regression: an unsized `layout.width` alone was *not*
        enough to make the figure grow into the 80% wrapper next to it — it
        left the graph at whatever size it already had, unchanged, with the
        rest of the 80% going unused. Plotly.js only measures its container
        and fills it when `config.responsive` says so; `layout.autosize`
        alone (`layout.width`/`height` left `None`) only stops it from
        fighting a container size, it does not make one up on its own."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        figure_widget, _ = _unwrap(result.range_controls())
        assert figure_widget._config.get("responsive") is True

    def test_an_explicit_size_does_not_force_responsive_on(
        self, _colab: None
    ) -> None:
        """`responsive` would otherwise resize the chart to its container
        regardless of an explicit width=/height=, defeating the point of
        asking for a specific pixel size."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        figure_widget, _ = _unwrap(result.range_controls(width=800, height=500))
        assert not figure_widget._config.get("responsive")

    def test_the_number_inside_each_box_has_room_to_be_read(self, _colab: None) -> None:
        """The reported symptom: the label ate the whole box and the
        editable number was not visible at all. Box width is now a
        percentage of the sidebar rather than a pixel count, so what is
        checked instead is the worst case: even at the sidebar's minimum
        width, the label's small fixed reservation leaves the number room."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        from mathslate.result import _RANGE_CONTROLS_MIN_SIDEBAR_WIDTH

        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        _, controls = _unwrap(result.range_controls())
        floor = int(_RANGE_CONTROLS_MIN_SIDEBAR_WIDTH.removesuffix("px"))
        for box in _number_controls(controls):
            assert box.layout.width == "calc(50% - 12px)"
            assert box.style.description_width == "0px"
            assert floor / 2 - 12 >= 80

    def test_sidebar_actions_zoom_autoscale_and_reset(self, _colab: None) -> None:
        """The sidebar is also a compact axis toolbar, not inputs only."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        figure_widget, controls = _unwrap(result.range_controls())
        x_min, x_max, y_min, y_max = _number_controls(controls)
        buttons = _buttons(controls)
        assert set(buttons) == {"Fit Y", "Reset", "[X]+", "[X]−", "[Y]+", "[Y]−"}

        x_span = x_max.value - x_min.value
        y_span = y_max.value - y_min.value
        buttons["[X]−"].click()
        buttons["[Y]+"].click()
        assert x_max.value - x_min.value == pytest.approx(x_span / 2)
        assert y_max.value - y_min.value == pytest.approx(y_span * 2)

        y_min.value, y_max.value = -10.0, 10.0
        buttons["Fit Y"].click()
        assert tuple(figure_widget.layout.yaxis.range) != (-10.0, 10.0)

        buttons["Reset"].click()
        assert (x_min.value, x_max.value) == (-10.0, 10.0)

    def test_the_size_survives_a_resample(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        bundle = result.range_controls(width=800, height=400)
        figure_widget, controls = _unwrap(bundle)
        x_min, x_max, _, _ = _number_controls(controls)
        x_min.value, x_max.value = -1.0, 1.0
        assert figure_widget.layout.width == 800
        assert figure_widget.layout.height == 400

    def test_left_unset_the_figure_keeps_whatever_size_it_already_had(
        self, _colab: None
    ) -> None:
        """`.plotly.update_layout(...)` before calling range_controls() already
        worked; this only adds a shortcut, not a replacement for it."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        result.plotly.update_layout(width=777, height=333)
        figure_widget, _ = _unwrap(result.range_controls())
        assert figure_widget.layout.width == 777
        assert figure_widget.layout.height == 333

    @pytest.mark.parametrize("name", ["width", "height"])
    def test_a_non_positive_size_is_refused(self, name: str, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        with pytest.raises(UnsupportedInputError, match="positive number of pixels"):
            result.range_controls(**{name: 0})


class TestRangeControlsAreTheDefaultDisplay:
    """PRD §19.4 — a bare `plot(...)` shows range_controls() where it can.

    `_ipython_display_` only takes over inside a real, initialized
    `InteractiveShell` — that is IPython's own rule, not this codebase's, and
    it means the plain-script/pytest-collection path is provably inert rather
    than merely untriggered in whatever is tested here. `IPython.core.
    interactiveshell.InteractiveShell.instance()` builds that real shell so
    the hook is exercised for real, the same way nbclient's kernel would.
    """

    @pytest.fixture
    def _colab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = types.ModuleType("google.colab")
        monkeypatch.setitem(sys.modules, "google.colab", stub)

    @pytest.fixture
    def _shell(self) -> Any:
        """A real, initialized `InteractiveShell` — the hook is dormant without one.

        Returns its formatter's own ``format()``, not the top-level
        ``IPython.display.display()``: a real ``marimo`` import elsewhere in
        the *same test process* replaces that top-level function with its own
        compatibility shim (verified — ``IPython.display.display`` is a
        different object after `test_notebook_display.py`'s marimo formatters
        register), which is an artifact of two frontends sharing one Python
        process in the suite and not something a real Jupyter kernel — which
        never imports marimo — would ever do. Calling the formatter directly
        is what ``display()`` itself calls, so it exercises the same protocol
        without depending on that top-level name staying what it started as.
        """
        pytest.importorskip("IPython")
        from IPython.core.interactiveshell import InteractiveShell

        shell = InteractiveShell.instance()
        return shell.display_formatter.format

    def test_wants_it_when_everything_lines_up(self, _colab: None, _shell: Any) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        assert result._wants_live_range_controls() is True

    def test_declines_for_raw_data_even_when_the_host_qualifies(
        self, _colab: None, _shell: Any
    ) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        raw = plot([1.0, 2.0, 2.0, 3.0] * 5, kind="hist", verbose=False)
        assert raw._wants_live_range_controls() is False

    def test_declines_on_a_plain_host(self, _shell: Any) -> None:
        """No marimo/colab stub here — detect_frontend() must say PLAIN."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x), verbose=False)
        assert result._wants_live_range_controls() is False

    def test_a_bare_display_call_shows_the_widget(
        self, _colab: None, _shell: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        shown: list[Any] = []
        monkeypatch.setattr(result, "range_controls", lambda: shown.append("live") or "widget")
        format_dict, _ = _shell(result)
        assert format_dict == {}  # empty means "_ipython_display_ took over"
        assert shown == ["live"]

    def test_a_bare_display_call_falls_back_for_raw_data(
        self, _colab: None, _shell: Any
    ) -> None:
        """The fallback branch must not raise just because deps are present."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        raw = plot([1.0, 2.0, 2.0, 3.0] * 5, kind="hist", verbose=False)
        raw._ipython_display_()  # must show the plain figure, not raise

    def test_a_slider_driven_plot_is_shown_as_its_figure(
        self, _colab: None, _shell: Any
    ) -> None:
        """`go.FigureWidget` rejects frames, and a slider's positions *are*
        frames (§11) — so routing one into range_controls() turned merely
        evaluating `plot(a*sin(x))` in a cell into Plotly's raw ValueError.
        Both tour cells of the interactive section died this way."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        amplitude = slider(-3, 3, default=1, name="review_amp")
        try:
            result = plot(amplitude * sin(x), verbose=False)
            assert result.interactive is True  # the precondition: it has frames
            assert result._wants_live_range_controls() is False
            result._ipython_display_()  # must show the figure, not raise
        finally:
            release_all()

    def test_range_controls_on_a_slider_plot_explains_itself(
        self, _colab: None
    ) -> None:
        """Asked for by name it must still refuse in MathSlate's own words,
        not leak `ValueError: Figure Widgets do not support frames`."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        amplitude = slider(-3, 3, default=1, name="review_amp2")
        try:
            result = plot(amplitude * sin(x), verbose=False)
            with pytest.raises(UnsupportedInputError, match="frames"):
                result.range_controls()
        finally:
            release_all()

    def test_marimo_is_unaffected_because_it_never_looks_for_the_hook(
        self, monkeypatch: pytest.MonkeyPatch, _shell: Any
    ) -> None:
        """`_repr_html_` is what marimo actually renders through (PRD §11);
        `_ipython_display_` existing on the class must not change that."""
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        stub = types.ModuleType("marimo")
        stub.running_in_notebook = lambda: True  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "marimo", stub)

        result = plot(sin(x) / x, (x, -10, 10), verbose=False)
        assert result._wants_live_range_controls() is False
        # Delegation, not string equality: Plotly mints a fresh div id per call.
        monkeypatch.setattr(
            result.plotly, "_repr_html_", lambda: "<div>the figure spoke</div>"
        )
        assert result._repr_html_() == "<div>the figure spoke</div>"


class TestFrontendReportExplainsWhy:
    """PRD §21 — the diagnostic line `set_range_controls()`'s docstring points to.

    `_wants_live_range_controls()` returning ``False`` gives no reason; the
    report exists because "why did nothing show?" is the actual question a
    reader asks, not "is this boolean true".
    """

    @pytest.fixture
    def _colab(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = types.ModuleType("google.colab")
        monkeypatch.setitem(sys.modules, "google.colab", stub)

    def test_off_says_how_to_turn_it_back_on(self) -> None:
        from mathslate import frontend_report, get_range_controls, set_range_controls

        set_range_controls(False)
        try:
            assert "set_range_controls(True)" in frontend_report()
        finally:
            set_range_controls(True)
        assert get_range_controls() is True

    def test_a_plain_host_names_what_it_needs(self) -> None:
        from mathslate import frontend_report

        report = frontend_report()
        assert "Jupyter or Colab" in report

    def test_ready_once_everything_lines_up(self, _colab: None) -> None:
        pytest.importorskip("ipywidgets")
        pytest.importorskip("anywidget")
        from mathslate import frontend_report

        assert "range_controls(): ready" in frontend_report()
