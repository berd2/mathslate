"""PRD 5.1's remaining dispatch rows — surfaces, contours, implicit curves,
space curves and parametric surfaces.

These were the four rows that raised `NotYetImplementedError` naming v0.5.
Filling them in is mostly plan construction, because `dispatch.build_plan()`
already routed each one to its own error.

What is genuinely different is the sampling. The one-variable sampler asks
SymPy exactly where a curve is real and exactly where it blows up. Neither
question has a usable answer in two variables — `continuous_domain` takes a
single symbol, and there is no two-variable `singularities` — so
`core/surfaces.py` evaluates on a grid and turns everything non-real into NaN.
That is a real reduction in guarantees, and `TestTheHonestLimits` is where it
is written down rather than glossed over.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import plotly.graph_objects as go

from mathslate import Eq, cos, exp, plot, sin, sqrt, t, theta, x, y
from mathslate.core import surfaces
from mathslate.errors import (
    AmbiguousAxisError,
    SamplingError,
    UnsupportedInputError,
)


class TestTheDispatchRows:
    def test_one_expression_two_symbols_is_a_surface(self) -> None:
        assert plot(x * y, verbose=False).plan.kind == "surface"

    def test_contour_is_the_toggle_not_a_different_object(self) -> None:
        """PRD 5.1: "default to surface; kind='contour' to switch"."""
        flat = plot(x * y, kind="contour", verbose=False)
        solid = plot(x * y, verbose=False)
        assert flat.plan.kind == "contour" and solid.plan.kind == "surface"
        assert np.allclose(
            flat.plan.series[0].sample.z, solid.plan.series[0].sample.z, equal_nan=True
        )

    def test_the_surface_report_offers_the_toggle(self) -> None:
        assert any("contour" in n for n in plot(x * y, verbose=False).notes)

    def test_an_equation_is_an_implicit_curve(self) -> None:
        assert plot(Eq(x**2 + y**2, 4), verbose=False).plan.kind == "implicit"

    def test_a_three_tuple_over_one_symbol_is_a_space_curve(self) -> None:
        assert plot((cos(t), sin(t), t), (t, 0.0, 12.0), verbose=False).plan.kind == "space"

    def test_a_three_tuple_over_two_symbols_is_a_parametric_surface(self) -> None:
        plan = plot(
            (cos(t) * cos(theta), cos(t) * sin(theta), sin(t)),
            (t, -1.5, 1.5),
            (theta, 0.0, 6.28),
            verbose=False,
        ).plan
        assert plan.kind == "psurface"

    def test_an_inequality_over_two_symbols_is_a_region(self) -> None:
        """It used to be a refusal. `Eq` asks where two things are equal, which
        is a curve; `<` asks where one exceeds the other, which is an area."""
        assert plot(sp.Lt(x, y), verbose=False).plan.kind == "region"


class TestTheRightTraceIsDrawn:
    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            (lambda: plot(x * y, verbose=False), go.Surface),
            (lambda: plot(x * y, kind="contour", verbose=False), go.Contour),
            (lambda: plot(Eq(x**2 + y**2, 4), verbose=False), go.Contour),
            (lambda: plot((cos(t), sin(t), t), (t, 0.0, 9.0), verbose=False), go.Scatter3d),
        ],
        ids=["surface", "contour", "implicit", "space"],
    )
    def test_it(self, build: object, expected: type) -> None:
        assert isinstance(build().plotly.data[0], expected)  # type: ignore[operator]

    def test_the_implicit_curve_draws_only_the_zero_level(self) -> None:
        """Any other level is a different equation."""
        contours = plot(Eq(x**2 + y**2, 4), verbose=False).plotly.data[0].contours
        assert contours.start == 0.0 and contours.end == 0.0

    def test_the_implicit_curve_passes_through_its_solutions(self) -> None:
        """x**2 + y**2 = 4 is zero on the circle of radius 2."""
        sample = plot(Eq(x**2 + y**2, 4), verbose=False).plan.series[0].sample
        mesh_x, mesh_y = np.meshgrid(sample.x, sample.y)
        on_curve = np.abs(sample.z) < 0.2
        radii = np.hypot(mesh_x[on_curve], mesh_y[on_curve])
        assert np.allclose(radii, 2.0, atol=0.1)


class TestAxesAndWindows:
    def test_both_axes_are_named_in_the_report(self) -> None:
        summary = plot(x * y, verbose=False).summary()
        assert "x ∈" in summary and "y ∈" in summary

    def test_the_grid_size_is_reported_as_a_grid(self) -> None:
        assert "60×60 samples" in plot(x * y, verbose=False).summary()

    def test_explicit_ranges_are_honoured_on_both_axes(self) -> None:
        plan = plot(x * y, (x, -1.0, 1.0), (y, -2.0, 2.0), verbose=False).plan
        assert plan.param_range == (-1.0, 1.0)
        assert plan.second_range == (-2.0, 2.0)

    def test_the_default_window_is_tighter_than_a_curves(self) -> None:
        """A 60x60 grid over [-10, 10] resolves far less than 200 points along a line."""
        assert plot(x * y, verbose=False).plan.param_range == (-5.0, 5.0)

    def test_a_third_free_symbol_is_ambiguous_not_guessed(self) -> None:
        """PRD 5.2 rule 4: three candidates for two axes — ask, do not guess."""
        a = sp.Symbol("a", real=True)
        with pytest.raises(AmbiguousAxisError):
            plot(a * x * y, verbose=False)

    def test_a_third_symbol_left_over_after_the_axes_is_named(self) -> None:
        a = sp.Symbol("a", real=True)
        with pytest.raises(UnsupportedInputError, match="are the axes"):
            plot(a * x * y, (x, -5.0, 5.0), (y, -5.0, 5.0), verbose=False)


class TestParametersStillWin:
    def test_freezing_one_symbol_turns_a_surface_back_into_a_curve(self) -> None:
        assert plot(x * y, parameters={y: 2.0}, verbose=False).plan.kind == "curve"

    def test_a_slider_bound_symbol_does_not_make_a_surface(self) -> None:
        """The headline example of PRD 5.7 must not become a surface over x and a."""
        from mathslate import slider
        from mathslate.ui import interact

        a = slider(1, 3, default=3, name="surface_guard_a")
        try:
            assert plot(a * sin(x), verbose=False).plan.kind == "curve"
        finally:
            interact.release_all()


class TestSpaceCurves:
    def test_the_helix_is_a_helix(self) -> None:
        sample = plot((cos(t), sin(t), t), (t, 0.0, 12.0), verbose=False).plan.series[0].sample
        radius = np.hypot(sample.x, sample.y)
        assert np.allclose(radius[np.isfinite(radius)], 1.0, atol=1e-6)
        assert sample.z is not None

    def test_a_pole_breaks_all_three_coordinates_together(self) -> None:
        """Otherwise the break is not a break — the line rejoins in 3D."""
        from mathslate import tan

        sample = plot((cos(t), sin(t), tan(t)), verbose=False).plan.series[0].sample
        assert sample.breakpoints
        cut = np.isnan(sample.x) & np.isnan(sample.y) & np.isnan(sample.z)
        assert cut.any()


class TestTheHonestLimits:
    """Two-variable sampling cannot promise what one-variable sampling does."""

    def test_a_hole_is_left_where_the_expression_is_not_real(self) -> None:
        sample = plot(sqrt(x * y), verbose=False).plan.series[0].sample
        assert not np.isfinite(sample.z).all()

    def test_and_it_says_how_much_of_the_grid_that_was(self) -> None:
        notes = plot(sqrt(x * y), verbose=False).notes
        assert any("not a real number" in n for n in notes)

    def test_nowhere_real_is_an_error_not_an_empty_picture(self) -> None:
        with pytest.raises(SamplingError, match="no real values"):
            plot(sqrt(-1 - x**2 - y**2), verbose=False)

    def test_a_pole_does_not_flatten_the_colour_scale(self) -> None:
        """1/(x*y) reaches 10^6 near the axes; without clipping everything else
        becomes one shade."""
        sample = plot(1 / (x * y), verbose=False).plan.series[0].sample
        assert sample.z_range is not None
        assert max(abs(v) for v in sample.z_range) < 1e4

    def test_a_well_behaved_surface_is_not_clipped(self) -> None:
        assert plot(x + y, verbose=False).plan.series[0].sample.z_range is None


class TestThePoleIsClippedInTheGeometryToo:
    """Clipping the colour scale alone leaves the *shape* wrong.

    `z_range` was computed and then used only for `cmin`/`cmax`, so a pole
    3481 tall on a surface whose features live within 14 was still drawn as a
    3481-unit wall: the camera zoomed out to contain it and everything the
    reader came to see became a flat sheet at the bottom. The range has to
    reach the z axis as well.
    """

    def test_the_figure_bounds_the_z_axis(self) -> None:
        result = plot(1 / (x * y), verbose=False)
        limits = result.plan.series[0].sample.z_range
        assert limits is not None
        assert tuple(result.plotly.layout.scene.zaxis.range) == pytest.approx(limits)

    def test_a_smooth_surface_keeps_an_automatic_z_axis(self) -> None:
        """Bounding a surface that needs no bounding would crop it."""
        assert plot(x + y, verbose=False).plotly.layout.scene.zaxis.range is None

    def test_a_parametric_surface_is_bounded_the_same_way(self) -> None:
        result = plot(
            (t * cos(theta), t * sin(theta), 1 / t),
            (t, 0.1, 2.0),
            (theta, 0.0, 6.28),
            verbose=False,
        )
        limits = result.plan.series[0].sample.z_range
        drawn = result.plotly.layout.scene.zaxis.range
        assert (drawn is None) == (limits is None)
        if limits is not None:
            assert tuple(drawn) == pytest.approx(limits)

    def test_the_emitted_code_reproduces_the_same_bound(
        self, headless_show: list[object]
    ) -> None:
        """PRD 5.5: the code must draw the figure you are looking at."""
        result = plot(1 / (x * y), verbose=False)
        limits = result.plan.series[0].sample.z_range
        assert limits is not None
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<emitted>", "exec"), namespace)  # noqa: S102
        figure = next(
            value for value in namespace.values() if isinstance(value, go.Figure)
        )
        assert tuple(figure.layout.scene.zaxis.range) == pytest.approx(limits)


class TestShowPython:
    CASES = (
        ("surface", lambda: plot(x * y, verbose=False)),
        ("contour", lambda: plot(x * y, kind="contour", verbose=False)),
        ("implicit", lambda: plot(Eq(x**2 + y**2, 4), verbose=False)),
        ("space", lambda: plot((cos(t), sin(t), t), (t, 0.0, 9.0), verbose=False)),
        (
            "psurface",
            lambda: plot(
                (cos(t) * cos(theta), cos(t) * sin(theta), sin(t)),
                (t, -1.5, 1.5),
                (theta, 0.0, 6.28),
                verbose=False,
            ),
        ),
        ("surface-domain", lambda: plot(sqrt(x * y), verbose=False)),
        ("surface-title", lambda: plot(exp(-(x**2) - y**2), title="bell", verbose=False)),
    )

    @pytest.mark.parametrize(("label", "build"), CASES, ids=[c[0] for c in CASES])
    def test_it_runs(self, label: str, build: object, headless_show: list[object]) -> None:
        exec(compile(build().python(), "<c>", "exec"), {})  # type: ignore[operator]  # noqa: S102

    @pytest.mark.parametrize(
        ("label", "build"),
        [c for c in CASES if c[0] != "space"],
        ids=[c[0] for c in CASES if c[0] != "space"],
    )
    def test_the_grid_is_reproduced_exactly(
        self, label: str, build: object, headless_show: list[object]
    ) -> None:
        """A uniform grid has nothing adaptive about it, so §9's caveat does not
        apply here: these must match to the last bit."""
        result = build()  # type: ignore[operator]
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<c>", "exec"), namespace)  # noqa: S102
        emitted = namespace["fig"]
        assert type(emitted.data[0]).__name__ == type(result.plotly.data[0]).__name__  # type: ignore[union-attr]
        assert np.allclose(
            np.asarray(emitted.data[0].z, dtype=float),  # type: ignore[union-attr]
            np.asarray(result.plotly.data[0].z, dtype=float),
            equal_nan=True,
        )

    def test_the_space_curve_traces_the_same_path(
        self, headless_show: list[object]
    ) -> None:
        """Sampled differently by design (§9), so compared as nearest points."""
        result = plot((cos(t), sin(t), t), (t, 0.0, 9.0), verbose=False)
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<c>", "exec"), namespace)  # noqa: S102
        theirs = _points(namespace["fig"].data[0])  # type: ignore[union-attr]
        ours = _points(result.plotly.data[0])
        gaps = np.sqrt(((theirs[:, None, :] - ours[None, :, :]) ** 2).sum(-1)).min(axis=1)
        assert gaps.max() < 0.05

    def test_a_space_curve_title_survives_show_python(
        self, headless_show: list[object]
    ) -> None:
        result = plot(
            (cos(t), sin(t), t),
            (t, 0.0, 9.0),
            kind="scatter",
            title="Helix",
            verbose=False,
        )
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<c>", "exec"), namespace)  # noqa: S102
        assert namespace["fig"].layout.title.text == "Helix"  # type: ignore[union-attr]
        assert namespace["fig"].data[0].mode == "markers"  # type: ignore[union-attr]


def _points(trace: object) -> np.ndarray:
    columns = [np.asarray(getattr(trace, axis), dtype=float) for axis in "xyz"]
    stacked = np.column_stack(columns)
    return stacked[np.isfinite(stacked).all(axis=1)]
