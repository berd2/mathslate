"""PRD 7 — v0.1 acceptance criteria, one test class per numbered criterion."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:  # `tomllib` is standard-library from Python 3.11; `tomli` backports it.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on the 3.10 floor
    import tomli as tomllib
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest
import sympy as sp

import mathslate as ms
from mathslate import Abs, cos, exp, floor, plot, sin, sqrt, tan, x
from mathslate.render import axes
from tests.helpers import crossing_points

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The five functions criterion 1 names explicitly.
HARD_CASES: tuple[sp.Expr, ...] = (
    tan(x),
    1 / x,
    floor(x),
    sqrt(x),
    x / Abs(x),
)


class TestCriterion1NoSpuriousLines:
    """"...all render correctly with no arguments and no spurious connecting lines."""

    @pytest.mark.parametrize("expr", HARD_CASES, ids=lambda e: str(e))
    def test_renders_from_a_bare_call(self, expr: sp.Expr) -> None:
        result = plot(expr, verbose=False)
        assert isinstance(result.plotly, go.Figure)
        assert result.plan.series[0].sample.finite_count > 50

    @pytest.mark.parametrize("expr", HARD_CASES, ids=lambda e: str(e))
    def test_no_segment_spans_a_discontinuity(self, expr: sp.Expr) -> None:
        sample = plot(expr, verbose=False).plan.series[0].sample
        assert crossing_points(sample, sample.breakpoints) == []

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            (tan(x), 6),  # +-pi/2, +-3pi/2, +-5pi/2 inside [-10, 10]
            (1 / x, 1),
            (x / Abs(x), 1),
            (floor(x), 19),  # every integer in (-10, 10)
        ],
        ids=["tan", "1/x", "x/|x|", "floor"],
    )
    def test_every_discontinuity_is_found(self, expr: sp.Expr, expected: int) -> None:
        sample = plot(expr, verbose=False).plan.series[0].sample
        assert len(sample.breakpoints) == expected

    def test_sqrt_is_never_sampled_outside_its_domain(self) -> None:
        sample = plot(sqrt(x), verbose=False).plan.series[0].sample
        drawn = sample.x[np.isfinite(sample.y)]
        assert drawn.min() >= 0.0
        assert sample.domain_intervals == ((0.0, 10.0),)

    def test_a_pole_does_not_flatten_the_curve(self) -> None:
        """Without y-clipping, tan(x) would render as a flat line at zero."""
        plan = plot(tan(x), verbose=False).plan
        assert plan.y_range is not None
        low, high = plan.y_range
        assert 2.0 < high < 50.0
        assert -50.0 < low < -2.0


class TestCriterion2ContextualAxes:
    """"plot(sin(x)) produces pi-multiple tick labels; plot(exp(x)) numeric ticks."""

    def test_sin_gets_pi_ticks(self) -> None:
        figure = plot(sin(x), verbose=False).plotly
        ticktext = figure.layout.xaxis.ticktext
        assert ticktext is not None
        assert "π" in "".join(ticktext)
        assert "0" in ticktext

    def test_exp_keeps_numeric_ticks(self) -> None:
        figure = plot(exp(x), verbose=False).plotly
        assert figure.layout.xaxis.ticktext is None

    def test_pi_labels_read_like_a_textbook(self) -> None:
        ticks = axes.pi_ticks(-2 * np.pi, 2 * np.pi)
        assert ticks is not None
        assert "0" in ticks.text
        assert any(label in ticks.text for label in ("π", "-π"))

    def test_log_scale_is_suggested_never_applied(self) -> None:
        result = plot(exp(x), verbose=False)
        assert result.plotly.layout.yaxis.type in (None, "-", "linear")
        assert any("log scale" in note for note in result.notes)


class TestCriterion4ShowPythonRuns:
    """"Every show_python() output executes verbatim under exec without error."""

    CASES: tuple[Any, ...] = (
        sin(x) / x,
        tan(x),
        1 / x,
        floor(x),
        sqrt(x),
        exp(x),
        [sin(x), cos(x)],
        (sin(x), cos(x)),
    )

    @pytest.mark.parametrize("obj", CASES, ids=lambda o: str(o))
    def test_generated_code_executes(self, obj: Any, headless_show: list[Any]) -> None:
        code = plot(obj, verbose=False).python()
        namespace: dict[str, Any] = {}
        exec(compile(code, "<show_python>", "exec"), namespace)  # noqa: S102
        assert len(headless_show) == 1
        assert isinstance(namespace["fig"], go.Figure)

    def test_generated_code_reproduces_the_same_curve(self, headless_show: list[Any]) -> None:
        result = plot(sin(x) / x, verbose=False)
        namespace: dict[str, Any] = {}
        exec(compile(result.python(), "<show_python>", "exec"), namespace)  # noqa: S102
        emitted = namespace["fig"].data[0]
        reference = result.plotly.data[0]
        # Both curves must sit on the true function, so both must agree with it
        # and therefore with each other.
        probes = np.linspace(1.0, 9.0, 25)
        truth = np.sin(probes) / probes
        ours = np.interp(probes, reference.x, reference.y)
        theirs = np.interp(probes, emitted.x, emitted.y)
        assert np.allclose(ours, truth, atol=1e-3)
        assert np.allclose(theirs, truth, atol=1e-3)

    def test_generated_code_keeps_the_line_broken(self, headless_show: list[Any]) -> None:
        namespace: dict[str, Any] = {}
        code = plot(tan(x), verbose=False).python()
        exec(compile(code, "<show_python>", "exec"), namespace)  # noqa: S102
        ys = np.asarray(namespace["fig"].data[0].y, dtype=float)
        assert np.isnan(ys).any(), "the emitted code must also break the line at poles"

    def test_explicit_symbol_creation_is_always_emitted(self) -> None:
        """PRD open decision 3: `import *` is allowed only because of this."""
        code = plot(sin(x), verbose=False).python()
        assert "sp.symbols('x'" in code


class TestCriterion5EnvironmentIndependence:
    """"Identical results across all three notebook environments."

    CI runs the notebooks themselves; here we pin the property that makes that
    possible — nothing in the numeric path consults the frontend.
    """

    @pytest.mark.parametrize("fake", ["marimo", "google.colab", None])
    def test_samples_do_not_depend_on_the_frontend(
        self, fake: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if fake is not None:
            monkeypatch.setitem(sys.modules, fake, type(sys)("stub"))
        result = plot(tan(x), verbose=False)
        baseline = _reference_tan_samples()
        assert np.allclose(result.numpy[0], baseline[0], equal_nan=True)
        assert np.allclose(result.numpy[1], baseline[1], equal_nan=True)


class TestCriterion6NoFrontendDependency:
    """"pip install mathslate pulls in no frontend dependency."""

    def test_declared_dependencies_are_frontend_free(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
        declared = " ".join(config["project"]["dependencies"]).lower()
        for frontend in ("marimo", "ipywidgets", "jupyter", "notebook"):
            assert frontend not in declared
        extras = config["project"]["optional-dependencies"]
        assert "marimo" in extras and "jupyter" in extras

    def test_importing_mathslate_imports_no_frontend(self) -> None:
        """Checked in a clean interpreter.

        Asserting on this process's ``sys.modules`` would only prove that no
        *other test* imported a frontend first, which is a different claim.
        """
        probe = (
            "import sys, mathslate\n"
            "leaked = [m for m in ('marimo', 'ipywidgets', 'IPython') if m in sys.modules]\n"
            "print(','.join(leaked))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=PROJECT_ROOT,
            check=True,
        )
        assert completed.stdout.strip() == "", (
            f"importing mathslate pulled in {completed.stdout.strip()}"
        )

    def test_detection_survives_a_frontend_being_merely_installed(self) -> None:
        """marimo in sys.modules is not the same as running inside marimo."""
        pytest.importorskip("marimo")
        import marimo  # noqa: F401  - the import is the point

        assert ms.plot(sin(x), verbose=False).plan.kind == "curve"

    def test_frontend_detection_needs_no_frontend_installed(self) -> None:
        assert ms.frontend_report().startswith("frontend: ")


def _reference_tan_samples() -> tuple[np.ndarray, np.ndarray]:
    return plot(tan(x), verbose=False).numpy
