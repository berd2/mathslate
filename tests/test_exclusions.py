"""`exclusions=` — the override PRD 5.3 was missing.

Automatic discontinuity detection is the package's differentiator (PRD 5.3, and
the reason `plot(tan(x))` is right where a Matplotlib wrapper is wrong). But a
differentiator with no override is a black box the moment it is wrong, and PRD 4
asks for escape-hatch completeness. `points=` was the only sampling knob there
was; there was no way to say "cut it here" or "stop guessing".

Mathematica calls this `Exclusions`. Two shapes:

* ``exclusions=[a, b]`` — cut here as well as wherever detection says.
* ``exclusions=False`` — stop the numeric probe. Symbolic singularities stay,
  because those are SymPy's answer rather than a guess.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest
import sympy as sp

from mathslate import cos, floor, pi, plot, sin, sqrt, tan, x
from mathslate.core.sampling import DEFAULT_CONFIG
from mathslate.errors import UnsupportedInputError


def _breaks(result: Any) -> list[float]:
    return sorted(result.plan.series[0].sample.breakpoints)


class TestNamingAPlace:
    def test_it_becomes_a_breakpoint(self) -> None:
        assert _breaks(plot(sin(x), (x, -6, 6), exclusions=[0], verbose=False)) == [0.0]

    def test_several_places_all_count(self) -> None:
        breaks = _breaks(plot(sin(x), (x, -6, 6), exclusions=[-3, 0, 3], verbose=False))
        assert breaks == [-3.0, 0.0, 3.0]

    def test_the_line_is_actually_cut_there(self) -> None:
        """A breakpoint nobody draws is a breakpoint that does nothing."""
        plain = plot(sin(x), (x, -6, 6), verbose=False)
        cut = plot(sin(x), (x, -6, 6), exclusions=[0], verbose=False)
        assert not np.isnan(np.asarray(plain.plotly.data[0].y, dtype=float)).any()
        ys = np.asarray(cut.plotly.data[0].y, dtype=float)
        xs = np.asarray(cut.plotly.data[0].x, dtype=float)
        gap = np.flatnonzero(np.isnan(ys))
        assert gap.size == 1
        assert xs[gap[0]] == pytest.approx(0.0)

    def test_a_symbolic_place_is_welcome(self) -> None:
        """`exclusions=[pi/2]` is the natural thing to write next to tan(x)."""
        breaks = _breaks(plot(sin(x), (x, -6, 6), exclusions=[pi / 2], verbose=False))
        assert breaks == [pytest.approx(float(sp.pi / 2))]

    def test_a_place_outside_the_window_is_ignored_not_refused(self) -> None:
        """Narrowing the range after naming a cut is not a mistake."""
        assert _breaks(plot(sin(x), (x, -1, 1), exclusions=[99], verbose=False)) == []

    def test_detected_and_named_places_both_survive(self) -> None:
        breaks = _breaks(plot(tan(x), (x, -4, 4), exclusions=[0], verbose=False))
        assert 0.0 in breaks
        assert any(abs(b - float(sp.pi / 2)) < 1e-6 for b in breaks)

    def test_it_reaches_a_parametric_curve_as_a_parameter_value(self) -> None:
        from mathslate import t

        result = plot((cos(t), sin(t)), (t, 0, 6), exclusions=[3], verbose=False)
        assert 3.0 in _breaks(result)


class TestTurningTheProbeOff:
    def test_the_numeric_probe_stops(self) -> None:
        steep = sp.atan(1000 * x)
        with_probe = plot(steep, (x, -1, 1), verbose=False)
        without = plot(steep, (x, -1, 1), exclusions=False, verbose=False)
        assert len(_breaks(without)) <= len(_breaks(with_probe))

    def test_a_genuine_jump_stops_being_cut(self) -> None:
        """`floor(x)` is all jumps, and all of them come from the probe."""
        found = plot(floor(x), (x, -3, 3), verbose=False)
        silent = plot(floor(x), (x, -3, 3), exclusions=False, verbose=False)
        assert len(_breaks(found)) >= 5
        assert _breaks(silent) == []

    def test_symbolic_singularities_are_not_switched_off(self) -> None:
        """Those are SymPy's answer, not a guess, so False must not touch them."""
        poles = _breaks(plot(tan(x), (x, -4, 4), exclusions=False, verbose=False))
        assert len(poles) == 2
        assert all(abs(abs(p) - float(sp.pi / 2)) < 1e-9 for p in poles)

    def test_the_real_domain_is_not_switched_off_either(self) -> None:
        sample = plot(sqrt(x), (x, -4, 4), exclusions=False, verbose=False)
        assert sample.plan.series[0].sample.domain_intervals == ((0.0, 4.0),)

    def test_naming_places_while_the_probe_is_off_is_still_possible(self) -> None:
        """`exclusions=False` and `exclusions=[...]` are one argument, so this
        needs the config directly — the test pins that the config supports it."""
        from dataclasses import replace

        config = replace(DEFAULT_CONFIG, probe_jumps=False, exclusions=(1.0,))
        from mathslate.core.sampling import sample_expression

        sample = sample_expression(floor(x), sp.Symbol("x", real=True), (-3, 3), config)
        assert sample.breakpoints == (1.0,)


class TestTheRefusals:
    def test_true_has_no_meaning_and_says_so(self) -> None:
        with pytest.raises(UnsupportedInputError, match="already on"):
            plot(sin(x), exclusions=True, verbose=False)

    def test_a_string_complains_about_the_string(self) -> None:
        """It iterates into characters, so the naive message blames a letter."""
        with pytest.raises(UnsupportedInputError, match="not the string"):
            plot(sin(x), exclusions="nope", verbose=False)

    def test_a_free_symbol_is_not_a_place(self) -> None:
        with pytest.raises(UnsupportedInputError, match="numbers on the axis"):
            plot(sin(x), exclusions=[sp.Symbol("q")], verbose=False)

    def test_a_non_sequence_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="list of places"):
            plot(sin(x), exclusions=object(), verbose=False)


class TestItSurvivesShowPython:
    def test_the_emitted_code_carries_the_named_places(
        self, headless_show: list[Any]
    ) -> None:
        result = plot(sin(x), (x, -6, 6), exclusions=[0, pi / 2], verbose=False)
        code = result.python()
        namespace: dict[str, object] = {}
        exec(compile(code, "<emitted>", "exec"), namespace)  # noqa: S102
        figure = next(v for v in namespace.values() if isinstance(v, go.Figure))
        ys = np.asarray(figure.data[0].y, dtype=float)
        # Each emitted segment ends with a NaN separator, so the last one is
        # structure rather than a cut; only the interior gaps are the exclusions.
        interior = np.flatnonzero(np.isnan(ys[:-1]))
        assert interior.size == 2, "the emitted code drew a continuous line"

    def test_the_emitted_code_omits_a_probe_that_was_switched_off(
        self, headless_show: list[Any]
    ) -> None:
        result = plot(floor(x), (x, -3, 3), exclusions=False, verbose=False)
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<emitted>", "exec"), namespace)  # noqa: S102
        figure = next(v for v in namespace.values() if isinstance(v, go.Figure))
        ys = np.asarray(figure.data[0].y, dtype=float)
        assert not np.isnan(ys[:-1]).any(), "a switched-off probe still cut the line"


class TestTheConfigCarriesIt:
    def test_the_defaults_keep_todays_behaviour(self) -> None:
        assert DEFAULT_CONFIG.exclusions == ()
        assert DEFAULT_CONFIG.probe_jumps is True

    def test_points_and_exclusions_compose(self) -> None:
        """They used to be built by two copies of the same code."""
        result = plot(sin(x), (x, -6, 6), points=500, exclusions=[0], verbose=False)
        assert result.plan.config.initial_points == 500
        assert result.plan.config.exclusions == (0.0,)

    def test_points_is_still_validated(self) -> None:
        with pytest.raises(UnsupportedInputError, match="at least 2"):
            plot(sin(x), points=1, verbose=False)
