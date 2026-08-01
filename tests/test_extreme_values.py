"""Extreme numeric values stay explicit without destabilising rendering."""

from __future__ import annotations

import warnings

import numpy as np

from mathslate import exp, plot, x


class TestNonFiniteValues:
    def test_overflow_becomes_a_gap_before_it_reaches_plotly(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = plot(exp(1000 * x), (x, -2, 2), verbose=False)

        values = np.asarray(result.plotly.data[0].y, dtype=float)
        assert np.isnan(values).any()
        assert not np.isinf(values).any()
        assert "Infinity" not in result.plotly.to_html(include_plotlyjs=False)

    def test_a_pole_is_cut_and_gets_a_finite_view_window(self) -> None:
        result = plot(1 / (x - 1), (x, -2, 2), verbose=False)
        values = np.asarray(result.plotly.data[0].y, dtype=float)
        assert np.isnan(values).any()
        assert np.isfinite(np.asarray(result.plotly.layout.yaxis.range, dtype=float)).all()


class TestLargeButFiniteValues:
    def test_valid_large_values_are_not_deleted_by_the_renderer(self) -> None:
        result = plot(1e300 * x, (x, -1, 1), verbose=False)
        values = np.asarray(result.plotly.data[0].y, dtype=float)
        assert np.isfinite(values).all()
        assert np.nanmax(np.abs(values)) > 9e299
