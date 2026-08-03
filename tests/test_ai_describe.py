"""`describe()` — the model narrates facts MathSlate already computed."""

from __future__ import annotations

import json
from typing import Any

import pytest
import sympy as sp

from mathslate import analyze, dataset, plot, sin, table, x
from mathslate.ai import describe
from mathslate.ai.describe import facts
from mathslate.ai.providers import Provider
from mathslate.errors import UnsupportedInputError

a, b = sp.symbols("a b")


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A provider that records the prompt and answers with fixed prose."""
    seen: dict[str, Any] = {}

    class Backend:
        def complete(self, system: str, question: str, model: str) -> str:
            seen["system"] = system
            seen["question"] = question
            return "  An odd cubic with three roots at -sqrt(3), 0 and sqrt(3).  "

    provider = Provider(
        name="claude", package="sys", pip_name="anthropic",
        env_var="ANTHROPIC_API_KEY", default_model="model",
        build=lambda key: Backend(),
    )
    import mathslate.ai.suggest as suggest

    monkeypatch.setattr(
        suggest, "resolve_provider", lambda name, *, api_key=None: provider
    )
    return seen


class TestTheFactsAreTheWholeWorld:
    def test_an_analysis_carries_its_confidence_and_exact_forms(self) -> None:
        payload = facts(analyze(x**3 - 3 * x))
        assert payload["result"] == "analysis"
        assert payload["roots"]["exact"] == ["-sqrt(3)", "0", "sqrt(3)"]
        assert payload["roots"]["approximate"] is False
        assert payload["any_approximate"] is False

    def test_a_plot_keeps_its_own_kind_alongside_the_discriminator(self) -> None:
        """Both were once spelled `kind`, so the plan's silently overwrote the
        marker saying what sort of result this was."""
        payload = facts(plot(sin(x) / x, verbose=False))
        assert payload["result"] == "plot"
        assert payload["kind"] == "curve"

    def test_a_plots_notes_survive_because_they_are_the_explanation(self) -> None:
        payload = facts(plot(sin(x) / x, verbose=False))
        assert any("singularities" in note for note in payload["notes"])

    def test_a_plots_generated_program_is_left_out(self) -> None:
        """Useful to an agent that wants to run it; here it is a page of NumPy
        that crowds out the numbers."""
        assert "python" not in facts(plot(sin(x), verbose=False))

    def test_a_table_carries_its_numbers(self) -> None:
        payload = facts(table(sin(x), (x, 0, 1), rows=3))
        assert payload["result"] == "table"
        assert payload["headers"] == ["x", "sin(x)"]
        assert len(payload["rows"]) == 3

    def test_a_fit_carries_its_parameters_and_quality(self) -> None:
        payload = facts(dataset({"x": [0, 1, 2, 3], "y": [1.0, 3.0, 5.0, 7.0]}).fit(a * x + b))
        assert payload["result"] == "fit"
        assert payload["parameters"]["a"] == pytest.approx(2.0)
        assert payload["r_squared"] == pytest.approx(1.0)

    def test_a_dataset_carries_its_shape(self) -> None:
        payload = facts(dataset({"x": [0, 1, 2], "y": [1.0, 3.0, 5.0]}))
        assert payload == {"result": "dataset", "columns": ["x", "y"], "rows": 3}

    def test_everything_sent_is_json_serialisable(self) -> None:
        for result in (
            analyze(x**3 - 3 * x),
            plot(sin(x) / x, verbose=False),
            table(sin(x), (x, 0, 1), rows=3),
            dataset({"x": [0, 1], "y": [1.0, 3.0]}),
        ):
            json.dumps(facts(result))

    def test_something_mathslate_did_not_compute_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="does not know what to do"):
            facts(42)


class TestDescribe:
    def test_it_returns_and_prints_the_explanation(
        self, spy: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        text = describe(analyze(x**3 - 3 * x))
        assert text == "An odd cubic with three roots at -sqrt(3), 0 and sqrt(3)."
        assert text in capsys.readouterr().out

    def test_the_model_is_sent_the_computed_facts(self, spy: dict[str, Any]) -> None:
        describe(analyze(x**3 - 3 * x), verbose=False)
        assert '"exact"' in spy["question"]
        assert "-sqrt(3)" in spy["question"]

    def test_the_model_is_told_not_to_compute(self, spy: dict[str, Any]) -> None:
        """The one failure that would matter: a number the reader believes and
        MathSlate never computed."""
        describe(analyze(x**2 - 2), verbose=False)
        assert "Never compute anything yourself" in spy["system"]
        assert "not in the facts" in spy["system"]

    def test_the_model_is_told_to_hedge_a_sampled_answer(
        self, spy: dict[str, Any]
    ) -> None:
        describe(analyze(x - sp.cos(x)), verbose=False)
        assert "approximate" in spy["system"]
        assert '"approximate": true' in spy["question"]

    def test_a_question_is_carried_and_answered_from_the_same_facts(
        self, spy: dict[str, Any]
    ) -> None:
        describe(plot(sin(x) / x, verbose=False), "why is there a gap at zero?", verbose=False)
        assert "why is there a gap at zero?" in spy["question"]
        assert "singularities" in spy["question"]

    def test_an_empty_question_is_refused(self, spy: dict[str, Any]) -> None:
        with pytest.raises(UnsupportedInputError, match="needs a question"):
            describe(analyze(x**2), "   ")
