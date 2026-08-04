"""`explain()` — the error is the evidence, not the model's memory."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from mathslate import plot, sin, x
from mathslate.ai import explain
from mathslate.ai.explain import _raised_by_mathslate, _redact, _report
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
    from mathslate.ai import credentials as credentials_module

    monkeypatch.setattr(
        suggest, "resolve_provider", lambda name, *, api_key=None: provider
    )
    # Mocking resolve_provider does not stop _resolve_request from reaching
    # load_credential first — explain() below omits provider/model, so
    # without this a developer's real saved credential decides model_name
    # (via saved.model) ahead of the fake provider's own default, on a
    # machine with one stored through assistant()'s "remember" option.
    monkeypatch.setattr(credentials_module, "load_credential", lambda *_: None)
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
        assert str(__file__) not in report
        assert "test_ai_explain.py" in report

    def test_it_redacts_credentials_from_messages_and_source_lines(self) -> None:
        def fail() -> None:
            api_key = "sk-ant-this-must-not-leave-the-machine"
            raise RuntimeError(f"request used api_key={api_key}")

        report = _report(_raised(fail), None)

        assert "this-must-not-leave-the-machine" not in report
        assert "api_key=[hidden]" in report

    def test_it_redacts_a_credential_named_like_an_environment_variable(self) -> None:
        """`OPENAI_API_KEY`, `db_password`: a real credential's name is almost
        never the bare word alone, it is that word joined to a prefix by `_` —
        and `_` is a word character, so a naive `\\b` in front of the keyword
        never reaches it. Every one of these leaked in full before this was
        fixed to look for the keyword as a substring instead."""
        assert "hunter2CorrectHorse" not in _redact(
            'DATABASE_PASSWORD = "hunter2CorrectHorse"'
        )
        assert "my-signing-secret-value" not in _redact(
            'JWT_SECRET = "my-signing-secret-value"'
        )
        assert "sk-abcdef1234567890" not in _redact(
            "OPENAI_API_KEY=sk-abcdef1234567890"
        )

    def test_it_does_not_flag_an_unrelated_word_sharing_the_suffix(self) -> None:
        """`password_hash`, `passwordless_login`: dropping the boundary check
        to reach the cases above must not make every word starting with one
        of the four keywords a redaction target — only one immediately
        followed by an assignment is."""
        assert _redact("passwordless_login = True") == "passwordless_login = True"
        assert _redact("password_hash = compute(x)") == "password_hash = compute(x)"

    def test_it_does_not_swallow_the_code_around_the_credential(self) -> None:
        """`Anthropic(api_key=os.environ['X'])`: the value used to be matched
        by "everything that isn't a comma or space", which does not stop at a
        closing bracket either — so the redaction ate the call's own closing
        parenthesis along with the value, corrupting the line."""
        redacted = _redact("results.append(Anthropic(api_key=api_key)); log('done')")
        assert redacted.count("(") == redacted.count(")")
        assert "log('done')" in redacted

    def test_it_does_not_leak_the_remainder_of_a_value_containing_a_comma(
        self,
    ) -> None:
        """The value was matched up to the first comma even inside a quoted
        string, so a credential containing one leaked everything after it."""
        redacted = _redact('api_key="abc,def-secret-value"')
        assert "def-secret-value" not in redacted

    def test_it_redacts_a_credential_whose_name_ends_in_key_or_token(self) -> None:
        """`STRIPE_SECRET_KEY`, `AWS_SECRET_ACCESS_KEY`: the keyword `secret`
        is in there, but it is followed by `_KEY`, not `=` — the check right
        after the keyword that keeps `password_hash` from matching also kept
        these from matching, for a reason that does not apply to them.
        `GITHUB_TOKEN` has no `access_` prefix at all. Bare `key`/`token`
        catch the case these are actually written."""
        assert "sk_live_51AbCdEfGhIjKlMnOpQr" not in _redact(
            'STRIPE_SECRET_KEY = "sk_live_51AbCdEfGhIjKlMnOpQr"'
        )
        assert "wJalrXUtnFEMI" not in _redact(
            'AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/bPxRfiCYEXAMPLEKEY"'
        )
        assert "ghp_1234567890" not in _redact(
            'GITHUB_TOKEN = "ghp_1234567890abcdefghijklmnop"'
        )

    def test_it_does_not_flag_a_bare_loop_or_iteration_variable_named_key(
        self,
    ) -> None:
        """Widening to bare `key`/`token` must not turn `for key in
        d.items():` — one of the most common idioms in the language — into a
        redaction target. It doesn't, because nothing here is immediately
        followed by an assignment operator, which every match still requires."""
        assert _redact("for key in d.items():") == "for key in d.items():"
        assert (
            _redact("for key, value in d.items():")
            == "for key, value in d.items():"
        )

    def test_an_escaped_quote_inside_the_value_does_not_end_the_match_early(
        self,
    ) -> None:
        """The value was matched up to the first literal quote character,
        which does not distinguish one escaped by a backslash from the
        string's real end — so a value containing `\\"` leaked everything
        from there to its actual closing quote."""
        redacted = _redact('api_key="first\\"still-secret"')
        assert "still-secret" not in redacted

    def test_an_unrelated_error_is_labelled_as_not_mathslates(self) -> None:
        report = _report(_raised(lambda: 1 / 0), None)
        assert "not from MathSlate's own checks" in report

    def test_code_alone_is_reviewed_rather_than_diagnosed(self) -> None:
        report = _report(None, "plot(sin(x), 0, 6.28)")
        assert "has not been run" in report
        assert "plot(sin(x), 0, 6.28)" in report

    def test_explicit_code_is_redacted_too(self) -> None:
        report = _report(None, 'api_key = "AIza-not-for-a-provider"')
        assert "not-for-a-provider" not in report
        assert "api_key = [hidden]" in report


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
