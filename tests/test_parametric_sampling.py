"""Parametric and polar curves get PRD §5.3 too — the gap this module closes.

`sample_parametric` used to be a bare `linspace` + `refine`: no real domain, no
symbolic poles, no jump probe, no NaN cuts, no readable window. Every existing
parametric test used a bounded, everywhere-continuous curve (`(cos t, sin t)`,
the cardioid), so nothing noticed.

What that cost: `plot((tan(t), t))` drew a line out to 10^16 with zero
breakpoints — the exact failure acceptance criterion 1 forbids for `tan(x)`,
one dispatch row over. `polar()` shipped in v0.1 (PRD §11.4 item 11) and runs
entirely through this path, so it was affected too.

The other half of the gap is horizontal. For `y = f(x)` the x extent is the
range the user asked for and cannot run away; for `(x(t), y(t))` it can, so the
window is clipped on both axes here.
"""

from __future__ import annotations

import numpy as np
import pytest

from mathslate import cos, floor, plot, polar, sin, sqrt, t, tan, theta, x
from mathslate.core import sampling
from tests.helpers import crossing_points


class TestTheDomainIsRespected:
    def test_a_component_restricts_the_shared_domain(self) -> None:
        """`sqrt(t)` is real only for t >= 0, so no sample may come from below."""
        sample = plot((sqrt(t), t), (t, -5.0, 5.0), verbose=False).plan.series[0].sample
        assert sample.domain_intervals == ((0.0, 5.0),)
        drawn = sample.t[np.isfinite(sample.x) & np.isfinite(sample.y)]
        assert drawn.min() >= 0.0

    def test_the_pieces_are_the_intersection_of_both_components(self) -> None:
        info = sampling.describe_parametric_domain((sqrt(t), 1 / t), t, -5.0, 5.0)
        # sqrt kills t < 0; 1/t cuts the rest open at the origin.
        assert info.intervals == ((0.0, 5.0),)
        assert 0.0 in info.singular_points

    def test_a_curve_that_is_real_nowhere_says_so(self) -> None:
        with pytest.raises(sampling.SamplingError):
            sampling.sample_parametric((sqrt(-1 - t**2), t), t, (-1.0, 1.0))


class TestPolesAreFound:
    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            (lambda: plot((tan(t), t), verbose=False), 2),
            (lambda: polar(tan(theta), verbose=False), 2),
            (lambda: polar(1 / cos(theta), verbose=False), 2),
        ],
        ids=["parametric-tan", "polar-tan", "polar-secant"],
    )
    def test_symbolic_poles_inside_one_turn(self, build: object, expected: int) -> None:
        """pi/2 and 3pi/2 are inside the default [0, 2pi] window."""
        sample = build().plan.series[0].sample  # type: ignore[operator]
        assert len(sample.breakpoints) == expected

    def test_no_segment_is_drawn_across_a_pole(self) -> None:
        sample = plot((tan(t), t), verbose=False).plan.series[0].sample
        assert crossing_points(sample, sample.breakpoints, axis="t") == []

    def test_the_break_blanks_both_coordinates(self) -> None:
        """The jump is in x here, so a real x at the cut would still join up."""
        sample = plot((tan(t), t), verbose=False).plan.series[0].sample
        cut = np.isnan(sample.y) & np.isnan(sample.x)
        assert cut.any()


class TestJumpsWithoutSymbolicHelp:
    def test_a_step_in_the_x_component_is_found_numerically(self) -> None:
        """`floor` defeats `singularities()`; the bisection probe does not.

        Watching y alone would have missed every one of these — the jump is in
        the horizontal coordinate.
        """
        sample = plot((floor(t), t), (t, -5.0, 5.0), verbose=False).plan.series[0].sample
        assert len(sample.breakpoints) == 9  # every integer inside (-5, 5)
        assert crossing_points(sample, sample.breakpoints, axis="t") == []

    def test_a_smooth_closed_curve_is_left_alone(self) -> None:
        """The probe must not invent breaks in the curve everyone plots first."""
        sample = plot((cos(t), sin(t)), verbose=False).plan.series[0].sample
        assert sample.breakpoints == ()
        assert sample.x_range is None and sample.y_range is None


class TestTheWindowIsReadable:
    def test_a_pole_does_not_flatten_the_curve_horizontally(self) -> None:
        """Unclipped, x(t) = tan(t) reaches 10^16 and the curve becomes a dot."""
        result = plot((tan(t), t), verbose=False)
        assert result.plan.x_range is not None
        low, high = result.plan.x_range
        assert 2.0 < high < 500.0
        assert -500.0 < low < -2.0
        assert list(result.plotly.layout.xaxis.range) == [low, high]

    def test_a_pole_in_the_radius_clips_the_vertical_window(self) -> None:
        result = polar(tan(theta), verbose=False)
        assert result.plan.y_range is not None
        assert max(abs(v) for v in result.plan.y_range) < 500.0

    def test_a_bounded_curve_keeps_its_natural_extent(self) -> None:
        """Clipping a cardioid would pad or crop a picture that was already right."""
        result = polar(1 + cos(t), verbose=False)
        assert result.plan.x_range is None
        assert result.plotly.layout.xaxis.range is None


class TestShowPythonKeepsUp:
    """A cut the emitted code does not make is a picture that does not match."""

    @pytest.mark.parametrize(
        ("label", "build"),
        [
            ("parametric-pole", lambda: plot((tan(t), t), verbose=False)),
            ("parametric-step", lambda: plot((floor(t), t), (t, -5.0, 5.0), verbose=False)),
            ("polar-pole", lambda: polar(tan(theta), verbose=False)),
            ("parametric-domain", lambda: plot((sqrt(t), t), (t, -5.0, 5.0), verbose=False)),
        ],
    )
    def test_the_emitted_figure_matches_ours(
        self, label: str, build: object, headless_show: list[object]
    ) -> None:
        result = build()  # type: ignore[operator]
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<show_python>", "exec"), namespace)  # noqa: S102
        emitted = namespace["fig"]

        ours = result.plotly
        assert _range(emitted.layout.xaxis) == pytest.approx(  # type: ignore[union-attr]
            _range(ours.layout.xaxis), rel=1e-6
        )
        assert _range(emitted.layout.yaxis) == pytest.approx(  # type: ignore[union-attr]
            _range(ours.layout.yaxis), rel=1e-6
        )
        theirs = np.asarray(emitted.data[0].y, dtype=float)  # type: ignore[union-attr]
        if np.isnan(np.asarray(ours.data[0].y, dtype=float)).any():
            assert np.isnan(theirs).any(), "the emitted code lost a discontinuity"


def _range(axis: object) -> list[float]:
    values = getattr(axis, "range", None)
    return [0.0] if values is None else [float(v) for v in values]


class TestTheReportStillReads:
    def test_discontinuities_are_counted_in_the_summary(self) -> None:
        assert "2 discontinuities handled" in plot((tan(t), t), verbose=False).summary()

    def test_the_singularity_note_names_the_parameter_not_x(self) -> None:
        notes = plot((tan(t), t), verbose=False).notes
        assert any(note.startswith("singularities at t =") for note in notes)
