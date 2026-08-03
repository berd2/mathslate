"""`explain()` — the error is the evidence, not the model's memory."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from mathslate import plot, sin, x
from mathslate.ai import explain
from mathslate.ai.explain import _raised_by_mathslate, _report
from mathslate.ai.providers import Provider
from mathslate.errors import UnsupportedInputError


def _raised(call: Any) -> BaseException:
    """The exception ``call`` raises, with its traceback attached."""
    try:
        call()
    except BaseException as error:  # noqa: BLE001 - the point of the helper
        return error
    raise AssertionError("expected the call to raise")


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A provider that records what it was asked and answers a fixed fix."""
    seen: dict[str, Any] = {}

    class Backend:
        def complete(self, system: str, question: str, model: str) -> str:
            seen["system"] = system
            seen["question"] = question
            return (
                "A range is one tuple naming its symbol.\n"
                "```python\nplot(sin(x), (x, 0, 6.28))\n```"
            )

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


class TestWhatCountsAsMathSlatesOwnMessage:
    """Which half of the prompt is used turns on this, so it is tested directly."""

    def test_a_mis_shaped_range_is_mathslates_message(self) -> None:
        """The commonest beginner mistake raises a plain TypeError on purpose
        (manual §4.1), so `isinstance(error, MathSlateError)` would have called
        MathSlate's own curated message untrusted."""
        error = _raised(lambda: plot(sin(x), 0, 6.28))
        assert isinstance(error, TypeError)
        assert _raised_by_mathslate(error)

    def test_a_mathslate_error_is_too(self) -> None:
        error = _raised(lambda: plot("!!! not maths !!!", verbose=False))
        assert isinstance(error, UnsupportedInputError)
        assert _raised_by_mathslate(error)

    def test_an_unrelated_python_error_is_not(self) -> None:
        assert not _raised_by_mathslate(_raised(lambda: 1 / 0))

    def test_nothing_is_not(self) -> None:
        assert not _raised_by_mathslate(None)


class TestTheEvidenceHandedToTheModel:
    def test_it_carries_the_message_and_the_traceback(self) -> None:
        report = _report(_raised(lambda: plot(sin(x), 0, 6.28)), None)
        assert "a range must be written (symbol, lo, hi)" in report
        assert "authoritative" in report
        assert "Traceback" in report

    def test_an_unrelated_error_is_labelled_as_not_mathslates(self) -> None:
        report = _report(_raised(lambda: 1 / 0), None)
        assert "not from MathSlate's own checks" in report

    def test_code_alone_is_reviewed_rather_than_diagnosed(self) -> None:
        report = _report(None, "plot(sin(x), 0, 6.28)")
        assert "has not been run" in report
        assert "plot(sin(x), 0, 6.28)" in report


class TestExplain:
    def test_it_returns_the_corrected_code_and_the_reason(self, spy: dict[str, Any]) -> None:
        suggestion = explain(_raised(lambda: plot(sin(x), 0, 6.28)), verbose=False)
        assert suggestion.code == "plot(sin(x), (x, 0, 6.28))"
        assert "one tuple naming its symbol" in suggestion.commentary
        assert suggestion.provider == "claude"

    def test_it_tells_the_model_to_trust_a_mathslate_message(
        self, spy: dict[str, Any]
    ) -> None:
        explain(_raised(lambda: plot(sin(x), 0, 6.28)), verbose=False)
        assert "is correct" in spy["system"]
        assert "do not guess at a different cause" in spy["system"]

    def test_it_admits_uncertainty_for_an_error_mathslate_did_not_raise(
        self, spy: dict[str, Any]
    ) -> None:
        explain(_raised(lambda: 1 / 0), verbose=False)
        assert "general Python and SymPy knowledge" in spy["system"]

    def test_the_prompt_carries_the_real_dispatch_contract(
        self, spy: dict[str, Any]
    ) -> None:
        """Built from KINDS, so a new kind cannot be added without the
        diagnosing prompt learning about it in the same commit."""
        from mathslate.core.dispatch import KINDS

        explain(_raised(lambda: plot(sin(x), 0, 6.28)), verbose=False)
        for kind in KINDS:
            assert kind in spy["system"]

    def test_it_reads_the_last_error_when_given_nothing(
        self, spy: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`sys.last_value` is what a failed cell leaves behind, which is why
        this needs no exception hook of its own."""
        monkeypatch.setattr(
            sys, "last_value", _raised(lambda: plot(sin(x), 0, 6.28)), raising=False
        )
        assert explain(verbose=False).code == "plot(sin(x), (x, 0, 6.28))"

    def test_it_reviews_a_snippet_without_running_it(self, spy: dict[str, Any]) -> None:
        explain(code="plot(sin(x), 0, 6.28)", verbose=False)
        assert "has not been run" in spy["question"]

    def test_it_says_so_when_there_is_no_error_to_explain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delattr(sys, "last_value", raising=False)
        with pytest.raises(UnsupportedInputError, match="no error to explain"):
            explain()

    def test_a_non_exception_is_refused_with_the_keyword_spelled_out(self) -> None:
        with pytest.raises(UnsupportedInputError, match='explain\\(code='):
            explain("plot(sin(x), 0, 6.28)")  # type: ignore[arg-type]

    def test_it_prints_the_explanation_by_default(
        self, spy: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        explain(_raised(lambda: plot(sin(x), 0, 6.28)))
        printed = capsys.readouterr().out
        assert "one tuple naming its symbol" in printed
        assert "plot(sin(x), (x, 0, 6.28))" in printed
