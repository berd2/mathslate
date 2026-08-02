"""PRD 4 and 6.3 — the API-surface budget and escape-hatch completeness."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest
import sympy as sp

import mathslate as ms
from mathslate import cos, plot, sin, x, y
from mathslate.ui import Frontend, detect_frontend


class TestApiBudget:
    """New API surface: <=20 public symbols AND <=20 plot() keywords (PRD §17.8, §20).

    The keyword half was added after v1.5: the metric counted only top-level
    names, but `plot()` had grown eleven keyword arguments, and keyword growth is
    surface growth the old count could not see. Both halves are guarded now.

    The cap itself was raised from 15 to 20 in §20, after two rounds in a row
    hit it exactly (§18.5's plot() keywords, §19.4's top-level symbols) and it
    started deciding scope by accident rather than by the design pressure it
    was meant to apply. Nothing was added to reach the new number — it is
    still 15 symbols and 14 keywords, same as §19.5 left it.
    """

    def test_the_symbol_budget_holds(self) -> None:
        assert len(ms.NEW_API) <= 20

    def test_the_plot_keyword_budget_holds(self) -> None:
        import inspect

        keywords = [
            name
            for name, param in inspect.signature(ms.plot).parameters.items()
            if name != "obj"
            and param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        assert len(keywords) <= 20, f"plot() has {len(keywords)} keywords: {keywords}"

    def test_the_eight_new_callables_all_exist(self) -> None:
        for name in (
            "plot", "polar", "slider", "animate",
            "table", "analyze", "show_python", "dataset",
        ):
            assert callable(getattr(ms, name))

    def test_sympy_functions_are_re_exported_not_wrapped(self) -> None:
        for name in ("solve", "simplify", "expand", "factor", "diff",
                     "integrate", "limit", "series", "Matrix"):
            assert getattr(ms, name) is getattr(sp, name)

    def test_import_star_is_usable(self) -> None:
        namespace: dict[str, object] = {}
        exec("from mathslate import *", namespace)  # noqa: S102
        for name in ("plot", "x", "sin", "solve"):
            assert name in namespace


class TestEscapeHatches:
    """"Every returned object exposes .sympy / .plotly / .numpy where applicable."""

    def test_plotly(self) -> None:
        assert isinstance(plot(sin(x), verbose=False).plotly, go.Figure)

    def test_sympy(self) -> None:
        assert plot(sin(x), verbose=False).sympy == sin(x)

    def test_sympy_is_a_tuple_when_several_curves_are_drawn(self) -> None:
        assert plot([sin(x), cos(x)], verbose=False).sympy == (sin(x), cos(x))

    def test_sympy_is_none_for_raw_data(self) -> None:
        assert plot([1.0, 2.0, 3.0], verbose=False).sympy is None

    def test_numpy(self) -> None:
        xs, ys = plot(sin(x), verbose=False).numpy
        assert isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray)
        assert xs.shape == ys.shape

    def test_the_figure_is_ours_to_mutate(self) -> None:
        result = plot(sin(x), verbose=False)
        result.plotly.update_layout(title="mine now")
        assert result.plotly.layout.title.text == "mine now"

    def test_the_result_renders_like_the_figure_it_wraps(self) -> None:
        result = plot(sin(x), verbose=False)
        assert result._repr_mimebundle_() == result.plotly._repr_mimebundle_()


class TestTheEscapeHatchesSurvivePickling:
    """PRD §25.1 — a result that crossed a process boundary is still a result.

    Plotly's own ``__reduce__`` runs the figure through ``to_dict()``, which
    encodes numeric arrays into the ``{"dtype", "bdata"}`` spec plotly.js
    reads. A figure rebuilt from that renders correctly and hands back a
    *dict* where `.plotly.data[0].x` used to be an array, so `np.asarray` on
    it raises. Nothing pickled a result until restricted AI execution moved
    out of process, which is when this became reachable.
    """

    @staticmethod
    def _roundtrip(result: Any) -> Any:
        import pickle

        return pickle.loads(pickle.dumps(result))

    def test_trace_arrays_come_back_as_arrays(self) -> None:
        original = plot(sin(x) / x, verbose=False)
        clone = self._roundtrip(original)
        assert isinstance(clone.plotly.data[0].x, np.ndarray)
        assert np.array_equal(
            np.asarray(original.plotly.data[0].y, dtype=float),
            np.asarray(clone.plotly.data[0].y, dtype=float),
            equal_nan=True,  # 1/x and friends break their line with NaN
        )

    def test_a_two_dimensional_grid_keeps_its_shape(self) -> None:
        """A surface's z is the case with a `shape` key in the spec."""
        original = plot(x * y, verbose=False)
        clone = self._roundtrip(original)
        assert np.asarray(clone.plotly.data[0].z).shape == np.asarray(
            original.plotly.data[0].z
        ).shape

    def test_slider_frames_survive(self) -> None:
        from mathslate import slider
        from mathslate.ui import release_all

        amplitude = slider(-3, 3, default=1, name="pickle_amp")
        try:
            original = plot(amplitude * sin(x), verbose=False)
            clone = self._roundtrip(original)
            assert len(clone.plotly.frames) == len(original.plotly.frames) > 0
            assert clone.interactive is True
        finally:
            release_all()

    def test_a_mutation_the_caller_made_is_not_thrown_away(self) -> None:
        """Which is why the figure is decoded rather than regenerated from
        the plan: `.plotly` is documented as theirs to mutate."""
        original = plot(sin(x), verbose=False)
        original.plotly.update_layout(title="mine now")
        assert self._roundtrip(original).plotly.layout.title.text == "mine now"

    def test_the_other_hatches_still_work_on_the_clone(self) -> None:
        clone = self._roundtrip(plot(sin(x) / x, verbose=False))
        assert clone.sympy == sin(x) / x
        assert all(isinstance(array, np.ndarray) for array in clone.numpy)
        assert clone.plan.kind == "curve"
        assert clone.python()


class TestDeferredApi:
    """Every milestone-deferred name has now landed."""

    def test_analyze_has_landed(self) -> None:
        """First of the v0.5 list to ship (PRD 11.6 step 1)."""
        assert ms.analyze(sin(x)).roots
        assert plot(sin(x), verbose=False).analyze().roots

    def test_slider_and_animate_have_landed(self) -> None:
        """Second of the v0.5 list (PRD 11.6 step 2)."""
        a = ms.slider(1, 3, name="budget_a")
        assert plot(a * sin(x), verbose=False).interactive
        assert ms.animate(a * sin(x), verbose=False).plotly.frames

    def test_table_has_landed(self) -> None:
        """Fourth of the v0.5 list (PRD 11.6 step 4) — v0.5 is complete."""
        assert ms.table(sin(x), rows=3).headers() == ("x", "sin(x)")

    def test_dataset_has_landed(self) -> None:
        """The last deferred name (PRD 7, v1.0). Nothing is deferred now."""
        assert len(ms.dataset({"x": [0.0, 1.0]})) == 2


class TestScopeDefence:
    """PRD 2.2 — explain() must not exist, in any spelling."""

    @pytest.mark.parametrize("name", ["explain", "Explain", "steps", "show_steps", "derive"])
    def test_no_step_by_step_api(self, name: str) -> None:
        assert not hasattr(ms, name)
        assert not hasattr(plot(sin(x), verbose=False), name)

    @pytest.mark.parametrize("name", ["explain", "Explain", "steps", "show_steps", "derive"])
    def test_analyze_reports_properties_and_nothing_more(self, name: str) -> None:
        """`analyze()` is where a derivation would be tempting to bolt on.

        PRD 5.6 is explicit that it reports properties *of the object* and
        never how they were obtained, so the guard belongs on `Analysis` too.
        """
        assert not hasattr(ms.analyze(x**2 - 1), name)


class TestTheInternalModuleBoundary:
    """One algorithm shared between two core modules, spelled as shared.

    `analysis.py` needs the same bisection jump probe that `sampling.py` uses —
    `singularities()` cannot see a discontinuity that is not a pole, so both
    modules have to run it. It was doing that by importing `_detect_jumps`, and
    a leading underscore that four call sites reach across is not privacy, it is
    a comment that has stopped being true. The name is public; this pins it, so
    a later rename has to notice the second caller.
    """

    def test_the_jump_probe_is_a_published_name(self) -> None:
        from mathslate.core import sampling

        assert "detect_jumps" in sampling.__all__
        assert callable(sampling.detect_jumps)

    def test_no_module_imports_it_under_its_old_private_name(self) -> None:
        from pathlib import Path

        package = Path(ms.__file__).parent
        offenders = [
            str(source.relative_to(package))
            for source in package.rglob("*.py")
            if "_detect_jumps" in source.read_text(encoding="utf-8")
        ]
        assert offenders == [], f"still reaching for the private spelling: {offenders}"

    def test_both_callers_agree_on_what_it_returns(self) -> None:
        """A shared helper is only shared if both sides get the same answer."""
        from mathslate import floor

        analysis = ms.analyze(floor(x), (x, -3, 3))
        plan = plot(floor(x), (x, -3, 3), verbose=False).plan
        assert analysis.discontinuities, "floor() has jumps at every integer"
        found = plan.series[0].sample.breakpoints
        for place in analysis.discontinuities.values:
            assert any(abs(place - b) < 0.05 for b in found), (
                f"analyze() reports a jump at {place} that the plot does not break at"
            )


class TestTheTypesAreVisibleToCallers:
    """PEP 561 — 422 annotated functions are worth nothing without the marker.

    A package without `py.typed` is treated as untyped by mypy and pyright no
    matter how thoroughly it is annotated: every name a caller imports from it
    silently becomes `Any`. The annotations were all already here; only the
    one empty file saying "mean them" was missing.
    """

    def test_the_marker_ships_inside_the_package(self) -> None:
        from pathlib import Path

        assert (Path(ms.__file__).parent / "py.typed").is_file()

    def test_public_functions_are_annotated(self) -> None:
        import ast
        from pathlib import Path

        unannotated: list[str] = []
        for source in sorted(Path(ms.__file__).parent.rglob("*.py")):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # `vararg`/`kwarg` too: `**kwargs` with no annotation is the
                # easiest hole to leave in an otherwise-typed signature, and
                # `args + kwonlyargs` alone does not look at it.
                arguments = [
                    argument
                    for argument in (
                        *node.args.args,
                        *node.args.kwonlyargs,
                        node.args.vararg,
                        node.args.kwarg,
                    )
                    if argument is not None
                ]
                if node.returns is None or any(
                    argument.annotation is None and argument.arg not in ("self", "cls")
                    for argument in arguments
                ):
                    unannotated.append(f"{source.name}:{node.lineno} {node.name}")
        assert unannotated == [], f"unannotated, but shipped as typed: {unannotated}"


class TestFrontendAdapters:
    def test_detection_returns_a_known_environment(self) -> None:
        assert detect_frontend() in set(Frontend)

    def test_the_report_mentions_widgets(self) -> None:
        assert "interactive widgets" in ms.frontend_report()

    def test_the_report_says_why_range_controls_would_not_show(self) -> None:
        """PRD §21: the diagnostic a reader asking 'why no widget?' needs."""
        assert "range_controls()" in ms.frontend_report()


class TestVerbosity:
    def test_it_can_be_switched_off_globally(self, capsys: pytest.CaptureFixture[str]) -> None:
        ms.set_verbose(False)
        try:
            plot(sin(x))
            assert capsys.readouterr().out == ""
        finally:
            ms.set_verbose(True)
        assert ms.get_verbose() is True


class TestRangeControlsToggle:
    """PRD §21 — set_range_controls(), the escape hatch §20's cap raise made room for."""

    def test_it_is_on_by_default(self) -> None:
        assert ms.get_range_controls() is True

    def test_it_can_be_switched_off_globally(self) -> None:
        import sys
        import types

        stub = types.ModuleType("google.colab")
        sys.modules["google.colab"] = stub
        try:
            ms.set_range_controls(False)
            try:
                result = plot(sin(x) / x, (x, -10, 10), verbose=False)
                assert result._wants_live_range_controls() is False
                assert ms.get_range_controls() is False
            finally:
                ms.set_range_controls(True)
            assert ms.get_range_controls() is True
        finally:
            del sys.modules["google.colab"]
