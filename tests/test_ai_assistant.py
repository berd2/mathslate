from __future__ import annotations

from typing import Any

import pytest

from mathslate.ai import credentials
from mathslate.ai.providers import PROVIDERS, Provider
from mathslate.errors import UnsupportedInputError
from mathslate.ai.suggest import (
    Suggestion,
    ask,
    check_connection,
    configure,
    configured,
    forget,
)


class MemoryKeyring:
    priority = 1

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_keyring(self) -> "MemoryKeyring":
        return self

    def set_password(self, service: str, account: str, value: str) -> None:
        self.values[(service, account)] = value

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def delete_password(self, service: str, account: str) -> None:
        self.values.pop((service, account), None)


@pytest.fixture(autouse=True)
def _clean_session() -> Any:
    forget()
    yield
    forget()


@pytest.fixture
def memory_store(monkeypatch: pytest.MonkeyPatch) -> MemoryKeyring:
    store = MemoryKeyring()
    monkeypatch.setattr(credentials, "_keyring", lambda *, required: store)
    return store


def test_remembered_credentials_live_in_the_secure_store(
    monkeypatch: pytest.MonkeyPatch, memory_store: MemoryKeyring
) -> None:
    import mathslate.ai.suggest as suggest

    monkeypatch.setattr(
        suggest,
        "resolve_provider",
        lambda name, *, api_key=None: Provider(
            name=name or "gemini", package="sys", pip_name="google-genai",
            env_var="GOOGLE_API_KEY", default_model="model", build=lambda key: None,
        ),
    )

    configure(
        provider="gemini", model="gemini-test", api_key="private-key", remember=True
    )

    assert configured() == {
        "provider": "gemini",
        "model": "gemini-test",
        "api_key": "saved securely",
    }
    assert "private-key" not in repr(configured())
    assert credentials.load_credential("gemini") == credentials.SavedCredential(
        "gemini", "private-key", "gemini-test"
    )


def test_a_new_session_automatically_reuses_the_saved_key(
    monkeypatch: pytest.MonkeyPatch, memory_store: MemoryKeyring
) -> None:
    import mathslate.ai.suggest as suggest

    credentials.save_credential("gemini", "remembered-key", "remembered-model")
    seen: dict[str, Any] = {}

    class Backend:
        def complete(self, system: str, question: str, model: str) -> str:
            seen["model"] = model
            return "```python\nplot(sin(x))\n```"

    provider = Provider(
        name="gemini", package="sys", pip_name="google-genai",
        env_var="GOOGLE_API_KEY", default_model="fallback", build=lambda key: (
            seen.update(key=key) or Backend()
        ),
    )
    monkeypatch.setattr(suggest, "resolve_provider", lambda name, *, api_key=None: provider)

    result = ask("plot sine")

    assert result.provider == "gemini"
    assert seen == {"key": "remembered-key", "model": "remembered-model"}


def test_connection_check_uses_a_remembered_credential(
    monkeypatch: pytest.MonkeyPatch, memory_store: MemoryKeyring
) -> None:
    import mathslate.ai.suggest as suggest

    credentials.save_credential("gemini", "remembered-key", "remembered-model")
    seen: dict[str, Any] = {}

    class Backend:
        def complete(self, system: str, question: str, model: str) -> str:
            seen.update(system=system, question=question, model=model)
            return "OK"

    provider = Provider(
        name="gemini", package="sys", pip_name="google-genai",
        env_var="GOOGLE_API_KEY", default_model="fallback", build=lambda key: (
            seen.update(key=key) or Backend()
        ),
    )
    monkeypatch.setattr(suggest, "resolve_provider", lambda name, *, api_key=None: provider)

    assert check_connection() == "Gemini connection is working (remembered-model)."
    assert seen["key"] == "remembered-key"
    assert seen["question"] == "Reply exactly OK."


def test_forget_persistent_removes_the_saved_key(memory_store: MemoryKeyring) -> None:
    credentials.save_credential("gemini", "private-key")
    forget(persistent=True, provider="gemini")
    assert credentials.load_credential("gemini") is None


def test_an_empty_provider_reply_is_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import mathslate.ai.suggest as suggest

    class Backend:
        def complete(self, system: str, question: str, model: str) -> str:
            return "  "

    provider = Provider(
        name="gemini", package="sys", pip_name="google-genai",
        env_var="GOOGLE_API_KEY", default_model="model", build=lambda key: Backend(),
    )
    monkeypatch.setattr(suggest, "resolve_provider", lambda name, *, api_key=None: provider)

    with pytest.raises(Exception, match="returned no text"):
        ask("plot sine", provider="gemini", api_key="key")


def test_gemini_accepts_either_documented_environment_key_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini = next(provider for provider in PROVIDERS if provider.name == "gemini")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "key")

    assert gemini.configured() is True
    assert gemini.credential_label == "GOOGLE_API_KEY or GEMINI_API_KEY"
    assert gemini.default_model == "gemini-3.5-flash"


def test_provider_errors_name_the_recovery_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mathslate.ai.suggest as suggest

    class Backend:
        def complete(self, system: str, question: str, model: str) -> str:
            raise RuntimeError("403 credential rejected")

    provider = Provider(
        name="gemini", package="sys", pip_name="google-genai",
        env_var="GOOGLE_API_KEY", default_model="model", build=lambda key: Backend(),
    )
    monkeypatch.setattr(suggest, "resolve_provider", lambda name, *, api_key=None: provider)

    with pytest.raises(UnsupportedInputError, match="Check the API key"):
        ask("plot sine", provider="gemini", api_key="private-key")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "ServerError: 503 UNAVAILABLE. This model is currently experiencing high demand. "
            "Check the API key, model access, quota, and network, then retry.",
            "The provider is temporarily busy (503). Wait a moment and retry.",
        ),
        (
            "429 rate limit exceeded",
            "The provider rejected the request because its quota or rate limit was reached.",
        ),
        (
            "403 API key not valid",
            "Authentication failed. Replace the API key in Settings and try again.",
        ),
    ],
)
def test_panel_error_messages_classify_provider_failures(
    message: str, expected: str
) -> None:
    from mathslate.ai.ui import _safe_error

    assert _safe_error(UnsupportedInputError(message)) == expected


def test_panel_shows_progress_and_the_generated_code(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("ipywidgets")
    import mathslate.ai.ui as ui

    monkeypatch.setattr(Provider, "installed", lambda self: True)
    monkeypatch.setattr(Provider, "configured", lambda self: False)
    monkeypatch.setattr(ui, "credential_store_available", lambda: True)
    monkeypatch.setattr(ui, "load_credential", lambda provider=None: None)
    monkeypatch.setattr(ui, "configure", lambda **kwargs: None)
    monkeypatch.setattr(
        ui,
        "ask",
        lambda *args, **kwargs: Suggestion(
            "plot(sin(x))", "plot sine", "gemini", "test-model"
        ),
    )
    panel = ui.assistant("plot sine")
    assert panel.remember.value is True
    panel.api_key.value = "private-key"
    panel.remember.value = True

    panel.ask_button.click()

    assert panel.suggestion is not None
    assert panel.code.value == "plot(sin(x))"
    assert "Received code" in panel.status.value
    assert panel.api_key.value == ""
    assert panel.run_button.disabled is False

    ran: list[str] = []
    monkeypatch.setattr(
        Suggestion,
        "run",
        lambda self, **kwargs: ran.append(self.code) or {},
    )
    panel.code.value = "plot(cos(x))"
    panel.run_button.click()

    assert ran == ["plot(cos(x))"]
    assert panel.suggestion.code == "plot(cos(x))"


def test_jupyterlite_disables_persistent_credentials(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("ipywidgets")
    import mathslate.ai.ui as ui

    monkeypatch.setattr(ui.sys, "platform", "emscripten")
    monkeypatch.setattr(Provider, "installed", lambda self: True)
    monkeypatch.setattr(Provider, "configured", lambda self: False)
    panel = ui.assistant()

    assert panel.remember.disabled is True
    assert panel.remember.value is False


def test_local_panel_keeps_remember_selectable_when_keyring_is_initially_missing(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("ipywidgets")
    import mathslate.ai.ui as ui

    monkeypatch.setattr(ui.sys, "platform", "win32")
    monkeypatch.setattr(ui, "credential_store_available", lambda: False)
    monkeypatch.setattr(ui, "load_credential", lambda provider=None: None)
    panel = ui.assistant()

    assert panel.remember.disabled is False
    assert panel.remember.value is False
    panel.remember.value = True
    assert panel.remember.value is True


def test_storage_failure_does_not_cancel_ask_or_connection_check(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("ipywidgets")
    import mathslate.ai.ui as ui

    calls: list[bool] = []

    def configure_for_session_then_fail_to_save(**kwargs: Any) -> None:
        remember = bool(kwargs["remember"])
        calls.append(remember)
        if remember:
            raise UnsupportedInputError("secure credential storage is unavailable")

    monkeypatch.setattr(Provider, "installed", lambda self: True)
    monkeypatch.setattr(Provider, "configured", lambda self: False)
    monkeypatch.setattr(ui, "credential_store_available", lambda: False)
    monkeypatch.setattr(ui, "load_credential", lambda provider=None: None)
    monkeypatch.setattr(ui, "configure", configure_for_session_then_fail_to_save)
    monkeypatch.setattr(
        ui,
        "ask",
        lambda *args, **kwargs: Suggestion(
            "plot(sin(x))", "plot sine", "gemini", "test-model"
        ),
    )
    monkeypatch.setattr(
        ui,
        "check_connection",
        lambda **kwargs: "Gemini connection is working (test-model).",
    )
    panel = ui.assistant("plot sine")
    panel.remember.value = True

    panel.api_key.value = "private-key"
    panel.ask_button.click()

    assert panel.suggestion is not None
    assert "Received code" in panel.status.value
    assert "active for this session but was not saved" in panel.status.value
    assert panel.api_key.value == ""

    panel.api_key.value = "private-key"
    panel.check_button.click()

    assert "connection is working" in panel.status.value
    assert "active for this session but was not saved" in panel.status.value
    assert panel.api_key.value == ""
    assert calls == [False, True, False, True]
