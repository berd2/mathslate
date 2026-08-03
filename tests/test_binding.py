"""Symbol-to-axis binding (PRD 5.2). Without this the slider feature is unstable."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from mathslate import plot, sin, t, x, y, z
from mathslate.core import binding
from mathslate.errors import AmbiguousAxisError


class TestResolutionOrder:
    def test_rule_1_explicit_range_wins(self) -> None:
        a = sp.Symbol("a", real=True)
        plan = plot(a * sin(a), (a, -1.0, 1.0), verbose=False).plan
        assert plan.symbol == a
        assert plan.param_range == (-1.0, 1.0)

    def test_rule_2_parameters_are_never_axes(self) -> None:
        a = sp.Symbol("a", real=True)
        assert binding.choose_symbols([a * sin(x)], 1, parameters=[a]) == [x]

    def test_rule_2_a_bound_parameter_is_frozen_at_its_value(self) -> None:
        """This is the hook ``slider()`` plugs into in v0.5."""
        a = sp.Symbol("a", real=True)
        plan = plot(a * sin(x), parameters={a: 3.0}, verbose=False).plan
        assert plan.symbol == x
        assert max(plan.series[0].sample.y[np.isfinite(plan.series[0].sample.y)]) == pytest.approx(
            3.0, rel=1e-3
        )

    def test_rule_3_convention_puts_x_before_a(self) -> None:
        a = sp.Symbol("a", real=True)
        assert binding.sort_by_convention([a, x]) == [x, a]

    def test_rule_3_full_order(self) -> None:
        names = ["theta", "r", "v", "u", "t", "z", "y", "x", "b", "a"]
        symbols = [sp.Symbol(n, real=True) for n in names]
        ordered = [s.name for s in binding.sort_by_convention(symbols)]
        assert ordered == ["x", "y", "z", "t", "u", "v", "r", "theta", "a", "b"]

    def test_rule_4_asks_rather_than_guessing(self) -> None:
        a = sp.Symbol("a", real=True)
        b = sp.Symbol("b", real=True)
        with pytest.raises(AmbiguousAxisError) as caught:
            binding.choose_symbols([a * b], 1)
        assert "Which 1 should be the axis?" in caught.value.question
        assert set(caught.value.candidates) == {"a", "b"}

    def test_the_question_shows_the_way_out(self) -> None:
        a, b = sp.symbols("a b", real=True)
        with pytest.raises(AmbiguousAxisError, match=r"plot\(expr, \(a, -10, 10\)\)"):
            binding.choose_symbols([a + b], 1)


class TestRanges:
    def test_default_range_is_symmetric(self) -> None:
        assert binding.default_range([x**2]) == (-10.0, 10.0)

    def test_periodic_default_covers_one_turn(self) -> None:
        lo, hi = binding.default_range([sin(t)], periodic_default=True)
        assert (lo, hi) == pytest.approx((0.0, 6.283185307179586))

    def test_parametric_curves_use_the_periodic_default(self) -> None:
        plan = plot((sp.cos(t), sp.sin(t)), verbose=False).plan
        assert plan.param_range == pytest.approx((0.0, 6.283185307179586))

    def test_malformed_ranges_are_explained(self) -> None:
        # The exception *types* are the documented contract (manual §4.1): a
        # mis-shaped range is a programming mistake, not a MathSlateError. The
        # messages, though, must say what was actually wrong.
        with pytest.raises(TypeError, match=r"\(symbol, lo, hi\)"):
            binding.parse_range((x, 1.0))
        with pytest.raises(ValueError, match="empty"):
            binding.parse_range((x, 1.0, 1.0))
        with pytest.raises(ValueError, match="finite"):
            binding.parse_range((x, float("nan"), 1.0))
        with pytest.raises(ValueError, match="finite"):
            binding.parse_range((x, 0.0, float("inf")))
        with pytest.raises(ValueError, match="must be numbers"):
            binding.parse_range((x, "a", "b"))

    def test_a_constant_still_gets_an_axis(self) -> None:
        chosen = binding.choose_symbols([sp.Integer(4)], 1)
        assert chosen[0].name == "x"


class TestPredefinedSymbols:
    def test_the_prd_symbols_exist_and_are_real(self) -> None:
        for symbol in (x, y, z, t):
            assert symbol.is_real
