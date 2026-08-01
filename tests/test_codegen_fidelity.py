"""`show_python()` must reproduce the figure, not merely produce *a* figure.

PRD §4 requires the emitted code to "run verbatim **and reproduce the same
result**". `tests/test_codegen.py` checks the first half. This module checks
the second: it builds every plot twice — once through MathSlate and once by
executing the emitted source in a clean namespace — and compares the figures
property by property.

The properties compared are exactly `render.options.REPRODUCED`; the ones
deliberately left out of the emitted code are `render.options.NOT_REPRODUCED`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import plotly.graph_objects as go
import pytest

from mathslate import (
    Abs, Eq, Matrix, cos, exp, floor, log, plot, polar, sin, slider, sqrt, t, tan, x, y, z,
)
from mathslate.render.options import NOT_REPRODUCED, REPRODUCED

#: (label, callable building the PlotResult). Covers every dispatch kind
#: crossed with every render option that reaches the figure.
CASES: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("curve", lambda: plot(sin(x), verbose=False)),
    ("curve-scatter", lambda: plot(sin(x), kind="scatter", verbose=False)),
    ("curve-line", lambda: plot(sin(x), kind="line", verbose=False)),
    ("curve-poles", lambda: plot(tan(x), verbose=False)),
    ("curve-domain", lambda: plot(sqrt(x), verbose=False)),
    ("curve-steps", lambda: plot(floor(x), verbose=False)),
    ("curve-jump", lambda: plot(x / Abs(x), verbose=False)),
    ("curve-log", lambda: plot(exp(x), yscale="log", verbose=False)),
    ("curve-title", lambda: plot(sin(x), title="A wave", verbose=False)),
    ("curve-legend-on", lambda: plot(sin(x), show_legend=True, verbose=False)),
    ("band", lambda: plot(sin(x) > 0, (x, -6, 6), verbose=False)),
    ("band-exact", lambda: plot(x**2 - 4 < 0, (x, -4, 4), verbose=False)),
    ("band-excluded", lambda: plot(sin(x), (x, -6, 6), exclusions=[0], verbose=False)),
    ("curve-xlim", lambda: plot(sin(x), (x, -10, 10), xlim=(-3, 3), verbose=False)),
    ("curve-ylim", lambda: plot(tan(x), (x, -4, 4), ylim=(-20, 20), verbose=False)),
    ("curves", lambda: plot([sin(x), cos(x)], verbose=False)),
    ("curves-legend-off", lambda: plot([sin(x), cos(x)], show_legend=False, verbose=False)),
    ("curves-scatter", lambda: plot([sin(x), log(x)], kind="scatter", verbose=False)),
    ("constant", lambda: plot(3, verbose=False)),
    ("parametric", lambda: plot((cos(t), sin(t)), verbose=False)),
    ("parametric-scatter", lambda: plot((cos(t), sin(t)), kind="scatter", verbose=False)),
    # A parametric curve can break and run away exactly as an explicit one can;
    # until these cases existed the matrix only ever asked the easy question.
    ("parametric-poles", lambda: plot((tan(t), t), verbose=False)),
    ("parametric-domain", lambda: plot((sqrt(t), t), (t, -5.0, 5.0), verbose=False)),
    ("parametric-steps", lambda: plot((floor(t), t), (t, -5.0, 5.0), verbose=False)),
    ("polar", lambda: polar(1 + cos(t), verbose=False)),
    ("polar-poles", lambda: polar(tan(t), verbose=False)),
    ("data", lambda: plot([1.0, 2.0, 3.0, 4.0], verbose=False)),
    ("data-line", lambda: plot([1.0, 2.0, 3.0, 4.0], kind="line", verbose=False)),
    ("data-log", lambda: plot([1.0, 2.0, 3.0, 4.0], yscale="log", verbose=False)),
    ("data-xy", lambda: plot(([0.0, 1.0, 2.0], [0.0, 1.0, 4.0]), verbose=False)),
    ("callable-numpy", lambda: plot(np.sin, (x, -3.0, 3.0), verbose=False)),
    ("callable-lambda", lambda: plot(lambda v: v**2, (x, -3.0, 3.0), verbose=False)),
    ("callable-builtin", lambda: plot(np.tanh, (x, -5.0, 5.0), verbose=False)),
    ("callable-scatter", lambda: plot(np.sin, (x, -3.0, 3.0), kind="scatter", verbose=False)),
    ("callable-log", lambda: plot(np.exp, (x, -3.0, 3.0), yscale="log", verbose=False)),
    ("callable-title", lambda: plot(np.sin, (x, -3.0, 3.0), title="mine", verbose=False)),
)
IDS = [label for label, _ in CASES]


@pytest.fixture(params=CASES, ids=IDS)
def pair(request: pytest.FixtureRequest, headless_show: list[Any]) -> tuple[go.Figure, go.Figure]:
    """(the figure MathSlate drew, the figure its emitted code draws)."""
    result = request.param[1]()
    namespace: dict[str, Any] = {}
    exec(compile(result.python(), "<show_python>", "exec"), namespace)  # noqa: S102
    emitted = namespace.get("fig")
    assert isinstance(emitted, go.Figure), "the emitted code did not build a figure"
    return result.plotly, emitted


class TestTheEmittedCodeRuns:
    def test_in_a_namespace_holding_nothing_but_builtins(self, pair: tuple[go.Figure, go.Figure]) -> None:
        """The fixture already proved it; this names the guarantee."""
        assert pair[1] is not None

    def test_it_calls_show_exactly_once(self, pair: tuple[go.Figure, go.Figure], headless_show: list[Any]) -> None:
        assert len(headless_show) == 1


class TestTheFiguresMatch:
    def test_trace_count(self, pair: tuple[go.Figure, go.Figure]) -> None:
        assert len(pair[0].data) == len(pair[1].data)

    def test_trace_mode(self, pair: tuple[go.Figure, go.Figure]) -> None:
        assert [tr.mode for tr in pair[0].data] == [tr.mode for tr in pair[1].data]

    def test_trace_names(self, pair: tuple[go.Figure, go.Figure]) -> None:
        assert [tr.name for tr in pair[0].data] == [tr.name for tr in pair[1].data]

    def test_gaps_are_never_connected(self, pair: tuple[go.Figure, go.Figure]) -> None:
        for figure in pair:
            assert all(tr.connectgaps is False for tr in figure.data)

    def test_title(self, pair: tuple[go.Figure, go.Figure]) -> None:
        assert pair[0].layout.title.text == pair[1].layout.title.text

    def test_legend_visibility(self, pair: tuple[go.Figure, go.Figure]) -> None:
        assert bool(pair[0].layout.showlegend) == bool(pair[1].layout.showlegend)

    def test_y_axis_type(self, pair: tuple[go.Figure, go.Figure]) -> None:
        assert _axis_type(pair[0].layout.yaxis) == _axis_type(pair[1].layout.yaxis)

    def test_y_axis_range(self, pair: tuple[go.Figure, go.Figure]) -> None:
        _assert_range_equal(pair[0].layout.yaxis.range, pair[1].layout.yaxis.range)

    def test_x_axis_range(self, pair: tuple[go.Figure, go.Figure]) -> None:
        _assert_range_equal(pair[0].layout.xaxis.range, pair[1].layout.xaxis.range)

    def test_pi_ticks(self, pair: tuple[go.Figure, go.Figure]) -> None:
        assert pair[0].layout.xaxis.ticktext == pair[1].layout.xaxis.ticktext
        _assert_range_equal(pair[0].layout.xaxis.tickvals, pair[1].layout.xaxis.tickvals)

    def test_template_is_the_same_theme(self, pair: tuple[go.Figure, go.Figure]) -> None:
        assert pair[0].layout.template.layout.colorway == pair[1].layout.template.layout.colorway


class TestTheCurvesMatch:
    def test_the_drawn_points_agree(self, pair: tuple[go.Figure, go.Figure]) -> None:
        """The two pictures coincide inside the window the reader actually sees.

        Compared as point *sets*, not element-wise: the two are sampled
        differently by design (§9 of the manual), and interpolating against x
        would be meaningless for parametric and polar curves, where x is not
        monotone, and across the NaN cuts of a broken curve.
        """
        window = _window(pair[0])
        for ours, theirs in zip(pair[0].data, pair[1].data):
            mine = _visible_points(ours, window)
            yours = _visible_points(theirs, window)
            if mine.shape[0] < 8 or yours.shape[0] < 8:
                continue
            diagonal = _diagonal(np.vstack([mine, yours]))
            # Measured point-to-*segment*: the two are sampled at different
            # densities, so a point-to-point metric would flag a steep stretch
            # where our vertices are simply further apart than theirs.
            assert _off_curve(yours, ours) <= 0.02 * diagonal, "emitted strays off ours"
            assert _off_curve(mine, theirs) <= 0.02 * diagonal, "ours strays off emitted"

    def test_breaks_are_preserved(self, pair: tuple[go.Figure, go.Figure]) -> None:
        for ours, theirs in zip(pair[0].data, pair[1].data):
            if np.isnan(np.asarray(ours.y, dtype=float)).any():
                assert np.isnan(np.asarray(theirs.y, dtype=float)).any(), (
                    "the emitted code lost a discontinuity"
                )


class TestCallablesInParticular:
    """The reported failure: a function's name is not a way to reproduce it."""

    def test_a_numpy_ufunc_does_not_leak_an_undefined_name(self, headless_show: list[Any]) -> None:
        code = plot(np.sin, (x, -3.0, 3.0), verbose=False).python()
        assert "sin(v)" not in code
        exec(compile(code, "<c>", "exec"), {})  # noqa: S102

    def test_a_lambda_does_not_emit_invalid_syntax(self, headless_show: list[Any]) -> None:
        code = plot(lambda v: v**2, (x, -3.0, 3.0), verbose=False).python()
        assert "<lambda>" not in code
        exec(compile(code, "<c>", "exec"), {})  # noqa: S102

    def test_the_samples_are_embedded_and_explained(self) -> None:
        code = plot(np.sin, (x, -3.0, 3.0), verbose=False).python()
        assert "cannot be written" in code
        assert "xs = np.array([" in code

    def test_no_sympy_import_is_emitted_when_there_is_no_expression(self) -> None:
        code = plot(np.sin, (x, -3.0, 3.0), verbose=False).python()
        assert "import sympy" not in code


class TestLargeArraysAreReducedNotTruncated:
    def test_an_oversized_data_series_keeps_its_full_extent(self, headless_show: list[Any]) -> None:
        """Truncating to the first N points would emit a different picture."""
        xs = np.linspace(0.0, 100.0, 5000)
        result = plot((xs, np.sin(xs)), verbose=False)
        namespace: dict[str, Any] = {}
        exec(compile(result.python(), "<c>", "exec"), namespace)  # noqa: S102
        emitted = np.asarray(namespace["fig"].data[0].x, dtype=float)
        assert emitted.max() == pytest.approx(100.0, abs=0.05)
        assert emitted.size <= 2000


#: The kinds v0.5 and v1.0 added. They are kept apart from ``CASES`` because a
#: surface has no ``mode`` and a histogram has no ``connectgaps``, so the 2D
#: assertions above do not apply — but the guarantee does, and until these were
#: listed nothing checked it for two whole milestones.
RICH_CASES: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("surface", lambda: plot(x * y, verbose=False)),
    ("surface-titled", lambda: plot(x * y, title="A saddle", verbose=False)),
    ("contour", lambda: plot(x * y, kind="contour", verbose=False)),
    ("surface-ranges", lambda: plot(x * y, (x, -2, 2), (y, -3, 3), verbose=False)),
    ("surface-poles", lambda: plot(1 / (x * y), verbose=False)),
    ("surface-nomesh", lambda: plot(x * y, mesh=False, verbose=False)),
    ("surface-zlim", lambda: plot(1 / (x * y), zlim=(-8, 8), verbose=False)),
    ("implicit", lambda: plot(Eq(x**2 + y**2, 4), verbose=False)),
    ("implicit-hyperbola", lambda: plot(Eq(x**2 - y**2, 1), verbose=False)),
    ("space", lambda: plot((cos(t), sin(t), t), verbose=False)),
    ("space-ranged", lambda: plot((cos(t), sin(t), t), (t, 0, 12), verbose=False)),
    ("psurface", lambda: plot((cos(t) * sin(z), sin(t) * sin(z), cos(z)), verbose=False)),
    ("linalg", lambda: plot(Matrix([[2, 1], [1, 3]]), verbose=False)),
    ("linalg-rotation", lambda: plot(Matrix([[0, -1], [1, 0]]), verbose=False)),
    ("hist", lambda: plot([1.0, 2.0, 2.0, 3.0, 3.0, 3.0] * 5, kind="hist", verbose=False)),
    ("box", lambda: plot([1.0, 2.0, 2.0, 3.0, 9.0] * 5, kind="box", verbose=False)),
    ("region", lambda: plot(x**2 + y**2 < 1, verbose=False)),
    ("region-compound", lambda: plot((x**2 + y**2 < 4) & (y > x), verbose=False)),
    ("region-holes", lambda: plot(sqrt(x * y) > 1, verbose=False)),
)
RICH_IDS = [label for label, _ in RICH_CASES]


@pytest.fixture(params=RICH_CASES, ids=RICH_IDS)
def rich_pair(
    request: pytest.FixtureRequest, headless_show: list[Any]
) -> tuple[go.Figure, go.Figure]:
    result = request.param[1]()
    namespace: dict[str, Any] = {}
    exec(compile(result.python(), "<show_python>", "exec"), namespace)  # noqa: S102
    emitted = namespace.get("fig")
    assert isinstance(emitted, go.Figure), "the emitted code did not build a figure"
    return result.plotly, emitted


class TestTheRicherKindsMatch:
    """Surfaces, contours, implicit curves, 3D, matrices and distributions."""

    def test_trace_count_and_type(self, rich_pair: tuple[go.Figure, go.Figure]) -> None:
        ours, theirs = rich_pair
        assert [tr.type for tr in ours.data] == [tr.type for tr in theirs.data]

    def test_trace_names(self, rich_pair: tuple[go.Figure, go.Figure]) -> None:
        ours, theirs = rich_pair
        assert [tr.name for tr in ours.data] == [tr.name for tr in theirs.data]

    @pytest.mark.parametrize("field", ["x", "y", "z"])
    def test_the_sampled_grids_are_identical(
        self, rich_pair: tuple[go.Figure, go.Figure], field: str
    ) -> None:
        """Grid-based kinds emit their samples as literals, so they match exactly.

        A space curve does not: like every other curve it is sampled adaptively
        here and on a uniform grid in the emitted code (manual §9), so it is
        checked by shape instead — see below.
        """
        ours, theirs = rich_pair
        for mine, yours in zip(ours.data, theirs.data):
            if mine.type == "scatter3d":
                continue
            _assert_array_equal(getattr(mine, field, None), getattr(yours, field, None))

    def test_space_curves_trace_the_same_path(
        self, rich_pair: tuple[go.Figure, go.Figure]
    ) -> None:
        ours, theirs = rich_pair
        curves = [
            (mine, yours)
            for mine, yours in zip(ours.data, theirs.data)
            if mine.type == "scatter3d"
        ]
        if not curves:
            pytest.skip("not a space curve")
        for mine, yours in curves:
            us = _points_3d(mine)
            them = _points_3d(yours)
            diagonal = _diagonal(np.vstack([us, them]))
            assert _off_polyline(them, us) <= 0.02 * diagonal, "emitted strays off ours"
            assert _off_polyline(us, them) <= 0.02 * diagonal, "ours strays off emitted"

    def test_title(self, rich_pair: tuple[go.Figure, go.Figure]) -> None:
        assert rich_pair[0].layout.title.text == rich_pair[1].layout.title.text

    def test_legend_visibility(self, rich_pair: tuple[go.Figure, go.Figure]) -> None:
        assert bool(rich_pair[0].layout.showlegend) == bool(rich_pair[1].layout.showlegend)

    def test_the_3d_scene_survives(self, rich_pair: tuple[go.Figure, go.Figure]) -> None:
        ours, theirs = rich_pair
        if not any(tr.type in {"surface", "scatter3d"} for tr in ours.data):
            pytest.skip("not a 3D plot")
        for axis in ("xaxis", "yaxis", "zaxis"):
            mine = getattr(ours.layout.scene, axis).title.text
            yours = getattr(theirs.layout.scene, axis).title.text
            assert mine == yours, f"scene.{axis} title differs"


class TestAnimationFramesSurvive:
    """A slider is carried in the figure's frames, so the code must carry them too."""

    def test_frame_count_and_data(self, headless_show: list[Any]) -> None:
        from mathslate.ui import release_all

        try:
            control = slider(-2, 2, default=1, name="q")
            result = plot(control * sin(x), verbose=False)
        finally:
            release_all()
        namespace: dict[str, Any] = {}
        exec(compile(result.python(), "<show_python>", "exec"), namespace)  # noqa: S102
        emitted = namespace["fig"]

        assert len(result.plotly.frames) == len(emitted.frames) > 1
        for mine, yours in zip(result.plotly.frames, emitted.frames):
            _assert_array_equal(mine.data[0].y, yours.data[0].y)

    def test_the_slider_control_itself_is_emitted(self, headless_show: list[Any]) -> None:
        from mathslate.ui import release_all

        try:
            control = slider(-2, 2, default=1, name="q")
            result = plot(control * sin(x), verbose=False)
        finally:
            release_all()
        namespace: dict[str, Any] = {}
        exec(compile(result.python(), "<show_python>", "exec"), namespace)  # noqa: S102
        assert len(namespace["fig"].layout.sliders) == len(result.plotly.layout.sliders) == 1

    def test_every_overlay_trace_is_emitted_in_every_frame(
        self, headless_show: list[Any]
    ) -> None:
        from mathslate.ui import release_all

        try:
            control = slider(-2, 2, 4.0, default=1, name="overlay_q")
            result = plot(
                [control * sin(x), control * cos(x)],
                verbose=False,
            )
        finally:
            release_all()
        namespace: dict[str, Any] = {}
        exec(compile(result.python(), "<show_python>", "exec"), namespace)  # noqa: S102
        emitted = namespace["fig"]

        assert len(result.plotly.data) == len(emitted.data) == 2
        for mine, yours in zip(result.plotly.frames, emitted.frames):
            assert len(mine.data) == len(yours.data) == 2
            for mine_trace, your_trace in zip(mine.data, yours.data):
                _assert_array_equal(mine_trace.y, your_trace.y)

    @pytest.mark.parametrize(
        "build",
        [
            lambda control: plot(control * x * y, verbose=False),
            lambda control: plot(
                (control * cos(t), sin(t), t),
                verbose=False,
            ),
            lambda control: plot(x**2 + y**2 < control.symbol, verbose=False),
        ],
        ids=["surface", "space", "region"],
    )
    def test_rich_frames_keep_the_base_trace_types(
        self, build: Callable[[Any], Any], headless_show: list[Any]
    ) -> None:
        from mathslate.ui import release_all

        try:
            # Two frames are sufficient to prove trace compatibility and keep
            # this regression test focused rather than resampling 3-D grids 21
            # times per case.
            control = slider(1, 2, 1.0, name="rich_q")
            result = build(control)
        finally:
            release_all()
        namespace: dict[str, Any] = {}
        exec(compile(result.python(), "<show_python>", "exec"), namespace)  # noqa: S102
        emitted = namespace["fig"]

        expected = [trace.type for trace in result.plotly.data]
        assert expected
        for figure in (result.plotly, emitted):
            assert [trace.type for trace in figure.data] == expected
            assert len(figure.frames) > 1
            for frame in figure.frames:
                assert [trace.type for trace in frame.data] == expected


class TestEveryDispatchKindIsCovered:
    """The gap this file is for: a new kind that nobody compares."""

    def test_no_kind_is_left_unchecked(self, headless_show: list[Any]) -> None:
        covered = set()
        for _, build in CASES + RICH_CASES:
            covered.add(build().plan.kind)
        from mathslate.core.dispatch import KINDS

        missing = sorted(set(KINDS) - covered - {"xy"})  # "xy" reports as "data"
        assert missing == [], f"these dispatch kinds have no fidelity case: {missing}"


class TestTheContractIsDocumented:
    def test_the_two_lists_do_not_overlap(self) -> None:
        assert not set(REPRODUCED) & set(NOT_REPRODUCED)

    def test_both_lists_are_non_empty(self) -> None:
        assert REPRODUCED and NOT_REPRODUCED


def _axis_type(axis: Any) -> str:
    return "log" if axis.type == "log" else "linear"


def _assert_array_equal(ours: Any, theirs: Any) -> None:
    """Two trace arrays, either of which may legitimately be absent."""
    if ours is None or theirs is None:
        assert ours is None and theirs is None, "one figure has data the other lacks"
        return
    mine = np.asarray(ours, dtype=float)
    yours = np.asarray(theirs, dtype=float)
    assert mine.shape == yours.shape, f"{mine.shape} vs {yours.shape}"
    assert np.allclose(mine, yours, rtol=1e-9, atol=1e-12, equal_nan=True)


def _assert_range_equal(ours: Any, theirs: Any) -> None:
    if ours is None or theirs is None:
        assert ours is None and theirs is None, f"{ours!r} vs {theirs!r}"
        return
    assert np.allclose(np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float), rtol=1e-6)


def _window(figure: go.Figure) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """The visible extent, which is what "the same picture" means."""

    def span(axis: Any) -> tuple[float, float] | None:
        if axis.range is None or axis.type == "log":
            return None
        lo, hi = (float(v) for v in axis.range)
        return (lo, hi)

    return span(figure.layout.xaxis), span(figure.layout.yaxis)


def _visible_points(
    trace: Any, window: tuple[tuple[float, float] | None, tuple[float, float] | None]
) -> np.ndarray:
    xs = np.asarray(trace.x, dtype=float)
    ys = np.asarray(trace.y, dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    x_span, y_span = window
    if x_span is not None:
        keep &= (xs >= x_span[0]) & (xs <= x_span[1])
    if y_span is not None:
        keep &= (ys >= y_span[0]) & (ys <= y_span[1])
    return np.column_stack([xs[keep], ys[keep]])


def _diagonal(points: np.ndarray) -> float:
    """The bounding box's diagonal, in however many dimensions there are."""
    extent = points.max(axis=0) - points.min(axis=0)
    return max(float(np.sqrt(np.sum(extent**2))), 1e-9)


def _segments(trace: Any) -> tuple[np.ndarray, np.ndarray]:
    """The drawn segments of a trace: consecutive pairs, never spanning a NaN."""
    xs = np.asarray(trace.x, dtype=float)
    ys = np.asarray(trace.y, dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    drawn = finite[:-1] & finite[1:]
    starts = np.column_stack([xs[:-1][drawn], ys[:-1][drawn]])
    ends = np.column_stack([xs[1:][drawn], ys[1:][drawn]])
    return starts, ends


def _points_3d(trace: Any) -> np.ndarray:
    """The finite ``(x, y, z)`` vertices of a 3D trace."""
    columns = [np.asarray(getattr(trace, axis), dtype=float) for axis in ("x", "y", "z")]
    stacked = np.column_stack(columns)
    return stacked[np.isfinite(stacked).all(axis=1)]


def _off_polyline(points: np.ndarray, vertices: np.ndarray) -> float:
    """Largest distance from ``points`` to the polyline through ``vertices``.

    Dimension-agnostic, so the same measure serves a 2D curve and a 3D one.
    """
    if vertices.shape[0] < 2:
        return float("inf")
    return _point_to_segments(points, vertices[:-1], vertices[1:])


def _off_curve(points: np.ndarray, trace: Any) -> float:
    """Largest distance from ``points`` to the polyline drawn by ``trace``."""
    starts, ends = _segments(trace)
    if starts.shape[0] == 0:
        return float("inf")
    return _point_to_segments(points, starts, ends)


def _point_to_segments(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> float:
    direction = ends - starts
    length_squared = np.maximum((direction**2).sum(axis=1), 1e-30)

    worst = 0.0
    for begin in range(0, points.shape[0], 128):
        chunk = points[begin : begin + 128]
        offset = chunk[:, None, :] - starts[None, :, :]
        # Projection parameter of each point onto each segment, clamped to it.
        position = np.clip(
            (offset * direction[None, :, :]).sum(axis=2) / length_squared[None, :], 0.0, 1.0
        )
        nearest = starts[None, :, :] + position[:, :, None] * direction[None, :, :]
        gaps = np.linalg.norm(chunk[:, None, :] - nearest, axis=2)
        worst = max(worst, float(gaps.min(axis=1).max()))
    return worst
