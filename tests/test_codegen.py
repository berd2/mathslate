"""show_python() (PRD 5.5): the emitted code must actually run, not decorate."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest

from mathslate import cos, exp, floor, log, plot, sin, sqrt, t, tan, x


def _run(code: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(code, "<show_python>", "exec"), namespace)  # noqa: S102
    return namespace


class TestItRuns:
    @pytest.mark.parametrize(
        "obj",
        [sin(x) / x, tan(x), 1 / x, floor(x), sqrt(x), log(x), exp(x),
         [sin(x), cos(x)], (cos(t), sin(t)), [1.0, 2.0, 3.0]],
        ids=lambda o: str(o)[:20],
    )
    def test_every_kind_of_plot(self, obj: Any, headless_show: list[Any]) -> None:
        namespace = _run(plot(obj, verbose=False).python())
        assert isinstance(namespace["fig"], go.Figure)

    @pytest.mark.parametrize(
        "function",
        [np.sin, np.tanh, lambda v: v**2],
        ids=["ufunc", "tanh", "lambda"],
    )
    def test_a_callable_plot_runs_without_the_function(
        self, function: Any, headless_show: list[Any]
    ) -> None:
        """The function cannot be serialised, so the samples must be embedded.

        Naming it would emit `sin(v)` for `np.sin` (NameError in a clean
        namespace) or `<lambda>(v)` for a lambda (SyntaxError).
        """
        namespace = _run(plot(function, (x, -1.0, 1.0), verbose=False).python())
        assert isinstance(namespace["fig"], go.Figure)


class TestItTeaches:
    """This is the user's first lambdify and first linspace."""

    def test_it_shows_lambdify_and_linspace(self) -> None:
        code = plot(sin(x) / x, verbose=False).python()
        assert "sp.lambdify" in code
        assert "np.linspace" in code
        assert "go.Figure" in code

    def test_it_is_not_pseudocode(self) -> None:
        code = plot(tan(x), verbose=False).python()
        assert "..." not in code
        assert "TODO" not in code

    def test_it_explains_the_parts_mathslate_did_silently(self) -> None:
        code = plot(tan(x), verbose=False).python()
        assert "NaN" in code or "np.nan" in code
        assert "percentile" in code

    def test_imports_cover_every_name_used(self, headless_show: list[Any]) -> None:
        namespace = _run(plot(sqrt(x) + floor(x), verbose=False).python())
        assert isinstance(namespace["fig"], go.Figure)


class TestItReproduces:
    def test_domain_restriction_survives(self, headless_show: list[Any]) -> None:
        namespace = _run(plot(sqrt(x), verbose=False).python())
        xs = np.asarray(namespace["fig"].data[0].x, dtype=float)
        ys = np.asarray(namespace["fig"].data[0].y, dtype=float)
        assert np.nanmin(xs[np.isfinite(ys)]) >= 0.0

    def test_pi_ticks_survive(self, headless_show: list[Any]) -> None:
        namespace = _run(plot(sin(x), verbose=False).python())
        ticktext = namespace["fig"].layout.xaxis.ticktext
        assert ticktext is not None and "π" in "".join(ticktext)

    def test_the_y_window_survives(self, headless_show: list[Any]) -> None:
        result = plot(1 / x, verbose=False)
        namespace = _run(result.python())
        emitted = tuple(namespace["fig"].layout.yaxis.range)
        assert result.plan.y_range is not None
        assert emitted == pytest.approx(result.plan.y_range, rel=1e-6)

    def test_overlaid_curves_stay_overlaid(self, headless_show: list[Any]) -> None:
        namespace = _run(plot([sin(x), cos(x)], verbose=False).python())
        assert len(namespace["fig"].data) == 2


class TestShowPythonPrints:
    def test_it_prints_and_returns(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = plot(sin(x), verbose=False).show_python()
        assert code in capsys.readouterr().out

    def test_the_module_level_twin_agrees(self) -> None:
        from mathslate import show_python

        result = plot(sin(x), verbose=False)
        assert show_python(result) == result.python()
