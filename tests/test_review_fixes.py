"""Five defects found by review, and the promises that now hold instead.

Each class below pins one of them. They have nothing in common as code — a
symbolic guess, a number formatter, a credential mask, a registry key and an
exception handler — but they share a shape worth naming, because it is the
shape a test suite is least likely to catch on its own: every one of them was
*silent*. Nothing raised, nothing printed, no test failed. A wrong root came
back labelled exact, a window collapsed to a point in emitted code the figure
never ran, a key rode out in a traceback nobody reads until it is too late, a
slider stopped binding, and six plot families lost their sidebar.

So these tests assert on the quiet part: not that the good path works, which
was never in doubt, but that the failure now has somewhere to show up.
"""

from __future__ import annotations

import sys
import types
import warnings
from typing import Any

import numpy as np
import pytest
import sympy as sp

from mathslate import cos, plot, sin, sqrt, t, x
from mathslate.codegen import _fmt
from mathslate.core import binding
from mathslate.core.analysis import analyze_expression
from mathslate.errors import SamplingError, UnsupportedInputError
from mathslate.ui import range_controls as rc


# --------------------------------------------------------------------------
# #8 — a guessed closed form is not an exact answer
# --------------------------------------------------------------------------


class TestAClosedFormIsReportedOnlyWhenItIsReallyTheAnswer:
    """`nsimplify` guesses, and a guess must not be printed as a proof.

    Given `1.4142135623` it answers `sqrt(2)` — a *different* number, 7.3e-11
    away, disagreeing in digits the Float actually carried. That came out of
    `analyze()` as `roots: sqrt(2)` with `approximate=False`, which is the one
    thing this module exists not to do: not a sampled answer honestly labelled,
    but a wrong answer labelled proven.
    """

    @pytest.mark.parametrize(
        "expr, root",
        [
            (x - sp.Float("1.4142135623"), 1.4142135623),
            (x**2 - sp.Float("2.718281828"), 1.64872127056092),
            (x**3 - sp.Float("7.3890560989"), 1.94773404105198),
        ],
    )
    def test_a_float_root_is_not_dressed_up_as_an_irrational(
        self, expr: sp.Expr, root: float
    ) -> None:
        found = analyze_expression(expr, x, (-5.0, 5.0))
        assert found.roots, f"no roots found for {expr}"
        assert min(abs(v - root) for v in found.roots.values) < 1e-9
        for form in found.roots.symbolic:
            assert float(sp.N(form)) == pytest.approx(
                min(found.roots.values, key=lambda v: abs(v - float(sp.N(form)))),
                abs=1e-12,
            ), f"{form} is not equal to the root it is offered as"

    @pytest.mark.parametrize(
        "expr, expected",
        [(x**2 - 2, "sqrt(2)"), (x**3 - 2, "2**(1/3)"), (sin(x), "pi")],
    )
    def test_a_genuinely_exact_root_still_prints_in_closed_form(
        self, expr: sp.Expr, expected: str
    ) -> None:
        """The fix must not cost the exact answers, which are the point."""
        described = analyze_expression(expr, x, (-5.0, 5.0)).roots.describe()
        assert expected in described

    def test_the_reported_case_end_to_end(self) -> None:
        found = analyze_expression(x - sp.Float("1.4142135623"), x, (-5.0, 5.0))
        assert "sqrt(2)" not in found.roots.describe()


# --------------------------------------------------------------------------
# #15 — emitted source must be the same numbers the figure used
# --------------------------------------------------------------------------


class TestEmittedNumbersSurviveAtEveryScale:
    """`show_python()` promises "this runs exactly as printed", at any zoom.

    `_fmt` broke that with two *absolute* tolerances — `abs(v - round(v)) <
    1e-12` and `round(v, 12)`, twelve decimal places rather than twelve digits.
    Together they turned any magnitude below about 5e-13 into `0.0`, so a real
    window came out of the emitted program as `range=[0.0, 0.0]`: a figure that
    drew fine beside a program that drew nothing.
    """

    @pytest.mark.parametrize(
        "value",
        [1e-13, 5e-13, -2.7e-14, 1e-300, 3.14159e-9, 0.5, 2.5, -10.0,
         1e20, 1234567.0000001, 0.30000000000000004],
    )
    def test_a_formatted_value_reads_back_as_itself(self, value: float) -> None:
        read_back = eval(_fmt(value), {"np": np})  # noqa: S307
        assert read_back == pytest.approx(value, rel=1e-11, abs=0.0)

    @pytest.mark.parametrize("value", [0.0, -0.0, 1.0, -3.0])
    def test_whole_numbers_still_print_as_whole_numbers(self, value: float) -> None:
        assert _fmt(value) == f"{value:.1f}"

    def test_a_tiny_window_is_not_flattened_to_a_point(self) -> None:
        source = plot(sin(x / 1e-13), (x, 0, 5e-13), verbose=False).show_python()
        assert "(0.0, 0.0)" not in source
        assert "range=[0.0, 0.0]" not in source
        assert "5e-13" in source

    def test_the_emitted_program_draws_the_same_window(self) -> None:
        result = plot(sin(x / 1e-13), (x, 0, 5e-13), verbose=False)
        namespace: dict[str, Any] = {}
        exec(compile(result.python(), "<emitted>", "exec"), namespace)  # noqa: S102
        emitted = namespace["fig"].layout.xaxis.range
        assert emitted[1] > emitted[0], "emitted window collapsed to a point"
        assert emitted[1] == pytest.approx(5e-13, rel=1e-11)


# --------------------------------------------------------------------------
# #10 — a credential must not ride out inside an error
# --------------------------------------------------------------------------


_KEY = "sk-ant-api03-NOTAREALKEY1234567890abcdef"


class TestAProviderErrorCarriesNoCredential:
    """Masking the new message is not enough while the old one is attached.

    A provider rejecting a key rarely quotes it whole — the usual shape is a
    head and a tail around an ellipsis, which `str.replace(key, ...)` walks
    straight past. And `raise ... from exc` re-prints the original exception,
    the one that actually quoted the key, under "The above exception was the
    direct cause of the following exception".
    """

    @pytest.mark.parametrize(
        "text",
        [
            f"401: invalid key {_KEY}",
            f"401 Unauthorized: invalid x-api-key {_KEY[:12]}...{_KEY[-4:]}",
            f"bad credential starting {_KEY[:16]}",
        ],
    )
    def test_no_recognisable_run_of_the_key_survives(self, text: str) -> None:
        from mathslate.ai.suggest import _redacted

        cleaned = _redacted(text, _KEY)
        for size in range(8, len(_KEY) + 1):
            assert _KEY[:size] not in cleaned
            assert _KEY[-size:] not in cleaned

    def test_a_key_we_never_held_is_masked_by_shape(self) -> None:
        """The SDK may read its own key from the environment; we never see it."""
        from mathslate.ai.suggest import _redacted

        assert "AIzaSyD-1234567890abcdefghij" not in _redacted(
            "auth failed for AIzaSyD-1234567890abcdefghij", None
        )

    def test_ordinary_error_prose_is_left_alone(self) -> None:
        from mathslate.ai.suggest import _redacted

        text = "503 Service Unavailable: upstream timeout after 30s"
        assert _redacted(text, _KEY) == text

    def test_the_traceback_chain_does_not_reprint_the_original(self) -> None:
        import traceback

        from mathslate.ai.suggest import _provider_failed

        try:
            try:
                raise RuntimeError(f"401 invalid x-api-key {_KEY}")
            except Exception as exc:  # noqa: BLE001
                raise _provider_failed(exc, "anthropic", _KEY, "the request") from None
        except UnsupportedInputError:
            printed = traceback.format_exc()
        assert _KEY not in printed
        assert _KEY[:12] not in printed
        assert "RuntimeError" in printed, "the cause's type is still worth saying"


# --------------------------------------------------------------------------
# #5 — binding is by name, because symbol identity includes assumptions
# --------------------------------------------------------------------------


class TestASliderBindsTheSymbolTheReaderWrote:
    """`Symbol('a')` and `Symbol('a', real=True)` print alike and are unequal.

    `slider(name="a")` binds the second. A reader who wrote `a = sp.Symbol('a')`
    holds the first, and a registry keyed by symbol object answered "not bound"
    for it — so `plot(sin(a*x))` quietly made the slider's own parameter a
    second axis instead of substituting its value.
    """

    @pytest.fixture(autouse=True)
    def _released(self) -> Any:
        from mathslate.ui.interact import release_all

        release_all()
        yield
        release_all()

    @pytest.mark.parametrize(
        "spelling",
        [
            sp.Symbol("a"),
            sp.Symbol("a", real=True),
            sp.Symbol("a", positive=True),
        ],
    )
    def test_every_spelling_of_the_name_resolves(self, spelling: sp.Symbol) -> None:
        from mathslate.ui.interact import slider

        slider(0, 5, default=2, name="a")
        found = binding.bindings_for([spelling])
        assert found == {spelling: 2.0}

    def test_the_answer_is_keyed_by_the_callers_own_symbol(self) -> None:
        """Anything else and `expr.subs()` matches nothing and silently no-ops."""
        from mathslate.ui.interact import slider

        slider(0, 5, default=2, name="a")
        plain = sp.Symbol("a")
        (key,) = binding.bindings_for([plain])
        assert key is plain

    def test_a_plain_symbol_is_substituted_rather_than_made_an_axis(self) -> None:
        from mathslate.ui.interact import slider

        slider(0, 5, default=2, name="a")
        drawn = plot(sin(sp.Symbol("a") * x), verbose=False)
        assert drawn.plan.kind == "curve"
        assert drawn.plan.exprs == (sin(2.0 * x),)

    def test_the_registry_holds_one_entry_per_visible_name(self) -> None:
        for spelling in (
            sp.Symbol("a"),
            sp.Symbol("a", real=True),
            sp.Symbol("a", positive=True),
        ):
            binding.bind_parameter(spelling, 1.0)
        assert len(binding.bound_parameters()) == 1, (
            "three entries all printing as 'a' is a registry a reader cannot read"
        )


# --------------------------------------------------------------------------
# #14 — a swallowed redraw failure must still leave a trace
# --------------------------------------------------------------------------


@pytest.fixture()
def _colab(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = types.ModuleType("google.colab")
    monkeypatch.setitem(sys.modules, "google.colab", stub)


def _boxes(node: Any, out: list[Any] | None = None) -> list[Any]:
    out = [] if out is None else out
    if type(node).__name__ == "FloatText":
        out.append(node)
    for child in getattr(node, "children", ()):
        _boxes(child, out)
    return out


class TestADeadSidebarAnnouncesItself:
    """The handler that hid the six-family bug now reports it.

    `_replotted()` raised `UnsupportedInputError("unknown kind='parametric'")`
    for six of the ten plot families; a bare `except MathSlateError` swallowed
    it; the sidebar shipped inert on all six with nothing raised and nothing
    printed. It was found by a reader trying the controls in the tour.

    The swallow is still right — a traceback out of a widget callback helps
    nobody, and a half-working sidebar beats a broken cell. Being *silent* was
    the defect, and the split is between a value no keystroke fixes and one the
    next keystroke will.
    """

    @pytest.fixture(autouse=True)
    def _forget_reports(self) -> Any:
        rc._REPORTED_REDRAW_FAILURES.clear()
        yield
        rc._REPORTED_REDRAW_FAILURES.clear()

    def test_a_plan_that_cannot_be_rebuilt_is_reported(
        self, _colab: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = plot([cos(t), sin(t)], verbose=False)
        boxes = _boxes(drawn.range_controls())

        def refuse(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise UnsupportedInputError("unknown kind='parametric'")

        monkeypatch.setattr(type(drawn), "_replotted", refuse)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            boxes[0].value = boxes[0].value - 1.0
        assert len(caught) == 1
        assert "not tracking the figure" in str(caught[0].message)
        assert "unknown kind='parametric'" in str(caught[0].message)

    def test_the_same_failure_is_reported_once_not_per_keystroke(
        self, _colab: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = plot([cos(t), sin(t)], verbose=False)
        boxes = _boxes(drawn.range_controls())

        def refuse(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise UnsupportedInputError("unknown kind='parametric'")

        monkeypatch.setattr(type(drawn), "_replotted", refuse)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for step in range(1, 5):
                boxes[0].value = boxes[0].value - step
        assert len(caught) == 1, "a held-down arrow key must not become a wall of text"

    def test_a_window_this_expression_cannot_be_sampled_over_stays_quiet(
        self, _colab: None
    ) -> None:
        """The reader is mid-edit. The next keystroke fixes it; saying so is noise."""
        drawn = plot(sqrt(x), verbose=False)
        boxes = _boxes(drawn.range_controls())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            boxes[1].value = -49.0
            boxes[0].value = -50.0
        assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []

    def test_a_sampling_failure_is_classed_as_transient(
        self, _colab: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drawn = plot(sin(x), verbose=False)
        boxes = _boxes(drawn.range_controls())

        def refuse(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise SamplingError("no finite values on this window")

        monkeypatch.setattr(type(drawn), "_replotted", refuse)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            boxes[0].value = boxes[0].value - 1.0
        assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []

    def test_the_sidebar_survives_either_way(
        self, _colab: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reporting must not become raising: the cell has to stay usable."""
        drawn = plot([cos(t), sin(t)], verbose=False)
        boxes = _boxes(drawn.range_controls())

        def refuse(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise UnsupportedInputError("unknown kind='parametric'")

        monkeypatch.setattr(type(drawn), "_replotted", refuse)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boxes[0].value = boxes[0].value - 1.0  # must not propagate
