"""`ask(..., about=result)` — a follow-up question that has an "it"."""

from __future__ import annotations

from typing import Any

import pytest

from mathslate import analyze, cos, dataset, plot, sin, t, table, x
from mathslate.ai import ask
from mathslate.ai.providers import Provider
from mathslate.errors import UnsupportedInputError


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    class Backend:
        def complete(self, system: str, question: str, model: str) -> str:
            seen["system"] = system
            seen["question"] = question
            return "```python\nplot(sin(x)/x, yscale='log')\n```"

    provider = Provider(
        name="claude", package="sys", pip_name="anthropic",
        env_var="ANTHROPIC_API_KEY", default_model="model",
        build=lambda key: Backend(),
    )
    import mathslate.ai.suggest as suggest
    from mathslate.ai import credentials as credentials_module

    monkeypatch.setattr(
        suggest, "resolve_provider", lambda name, *, api_key=None: provider
    )
    # Mocking resolve_provider does not stop _resolve_request from reaching
    # load_credential first — every ask() below omits provider/model, so
    # without this a developer's real saved credential decides model_name
    # (via saved.model) ahead of the fake provider's own default, on a
    # machine with one stored through assistant()'s "remember" option.
    monkeypatch.setattr(credentials_module, "load_credential", lambda *_: None)
    return seen


class TestAFollowUpKnowsWhatItMeans:
    def test_the_expression_travels_so_it_has_something_to_name(
        self, spy: dict[str, Any]
    ) -> None:
        """The summary reports the sampling; without the expression a follow-up
        like "the same thing on a log scale" has nothing to refer to."""
        drawn = plot(sin(x) / x, verbose=False)
        ask("show the same thing on a log scale", about=drawn)
        assert '"expression": "sin(x)/x"' in spy["question"]
        assert "show the same thing on a log scale" in spy["question"]

    def test_the_plots_own_notes_travel_too(self, spy: dict[str, Any]) -> None:
        ask("why the gap?", about=plot(sin(x) / x, verbose=False))
        assert "singularities" in spy["question"]

    def test_the_model_is_told_the_facts_are_computed(self, spy: dict[str, Any]) -> None:
        ask("and its roots?", about=plot(sin(x), verbose=False))
        assert "computed by MathSlate and are correct" in spy["system"]
        assert "do not contradict them" in spy["system"]

    def test_the_model_is_told_to_write_standalone_code(
        self, spy: dict[str, Any]
    ) -> None:
        """It was never told what the reader called the variable."""
        ask("now on a log scale", about=plot(sin(x), verbose=False))
        assert "make it standalone" in spy["system"]

    @pytest.mark.parametrize(
        "result",
        [
            plot(sin(x) / x, verbose=False),
            plot([sin(x), cos(x)], verbose=False),
            plot((cos(t), sin(t)), verbose=False),
            analyze(x**3 - 3 * x),
            table(sin(x), (x, 0, 1), rows=3),
            dataset({"x": [0.0, 1.0, 2.0], "y": [1.0, 3.0, 5.0]}),
        ],
    )
    def test_every_kind_of_result_can_be_followed_up(
        self, spy: dict[str, Any], result: object
    ) -> None:
        ask("what next?", about=result)
        assert "The reader is looking at this result" in spy["question"]

    def test_something_mathslate_did_not_compute_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="does not know what to do"):
            ask("what next?", about=42)


class TestAskWithoutContextIsUnchanged:
    def test_the_bare_question_is_all_that_is_sent(self, spy: dict[str, Any]) -> None:
        """Context is named, never collected: nothing about the session leaves
        the machine because a question was asked near it."""
        ask("plot the tangent")
        assert spy["question"] == "plot the tangent"

    def test_the_follow_up_rules_are_not_added(self, spy: dict[str, Any]) -> None:
        ask("plot the tangent")
        assert "looking at this result" not in spy["system"]
        assert "make it standalone" not in spy["system"]

    def test_an_empty_question_is_still_refused(self, spy: dict[str, Any]) -> None:
        with pytest.raises(UnsupportedInputError, match="needs a question"):
            ask("   ", about=plot(sin(x), verbose=False))
