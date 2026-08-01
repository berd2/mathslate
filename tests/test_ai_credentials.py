"""An API key belongs to exactly one company. Guessing which is not acceptable.

``resolve_provider`` used to consult ``api_key`` only after it had already
picked whichever provider happened to be configured, so an explicit OpenAI key
passed to ``ask()`` on a machine with ``ANTHROPIC_API_KEY`` in the environment
was handed to Anthropic's client and sent over the wire. Nothing in a key
string reliably identifies its service, so where there is more than one
candidate the answer is to ask, not to pick.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from mathslate.ai import providers
from mathslate.ai.suggest import ask, configure, configured, forget
from mathslate.errors import UnsupportedInputError

FAKE_KEY = "sk-this-belongs-to-exactly-one-service"


@pytest.fixture()
def installed(monkeypatch: pytest.MonkeyPatch):
    """Pretend a chosen set of provider SDKs is installed."""

    def choose(names: set[str]) -> None:
        monkeypatch.setattr(
            providers.Provider, "installed", lambda self: self.name in names
        )

    return choose


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for provider in providers.PROVIDERS:
        monkeypatch.delenv(provider.env_var, raising=False)
    forget()
    yield
    forget()


class TestAnExplicitKeyIsNeverGuessedAt:
    def test_several_sdks_installed_means_ask(self, installed) -> None:
        installed({"claude", "openai"})
        with pytest.raises(UnsupportedInputError, match="explicit provider"):
            providers.resolve_provider(None, api_key=FAKE_KEY)

    def test_a_configured_provider_does_not_capture_someone_elses_key(
        self, installed, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reported bug, exactly: env key for one, explicit key for another."""
        installed({"claude", "openai"})
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-the-users-real-key")
        with pytest.raises(UnsupportedInputError, match="explicit provider"):
            providers.resolve_provider(None, api_key=FAKE_KEY)

    def test_the_message_names_the_candidates_and_the_remedy(self, installed) -> None:
        installed({"claude", "openai"})
        with pytest.raises(UnsupportedInputError) as caught:
            providers.resolve_provider(None, api_key=FAKE_KEY)
        message = str(caught.value)
        assert "claude" in message and "openai" in message
        assert "provider=" in message

    def test_one_sdk_installed_is_unambiguous(self, installed) -> None:
        installed({"openai"})
        assert providers.resolve_provider(None, api_key=FAKE_KEY).name == "openai"

    def test_naming_the_provider_always_works(self, installed) -> None:
        installed({"claude", "openai", "gemini"})
        for name in ("claude", "openai", "gemini"):
            assert providers.resolve_provider(name, api_key=FAKE_KEY).name == name

    def test_a_key_with_nowhere_to_go_says_so(self, installed) -> None:
        installed(set())
        with pytest.raises(UnsupportedInputError, match="no provider SDK is installed"):
            providers.resolve_provider(None, api_key=FAKE_KEY)


class TestResolutionWithoutAKey:
    def test_the_configured_provider_is_used(
        self, installed, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        installed({"claude", "openai"})
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        assert providers.resolve_provider(None).name == "openai"

    def test_nothing_ready_explains_every_provider(self, installed) -> None:
        installed({"claude"})
        with pytest.raises(UnsupportedInputError) as caught:
            providers.resolve_provider(None)
        message = str(caught.value)
        assert "works offline" in message
        assert "ANTHROPIC_API_KEY" in message

    def test_an_unknown_name_lists_the_known_ones(self) -> None:
        with pytest.raises(UnsupportedInputError, match="unknown provider"):
            providers.resolve_provider("gpt4all")


class TestTheKeyCanBeTakenBack:
    def test_configure_then_forget(self) -> None:
        configure(model="some-model", api_key=FAKE_KEY)
        assert configured()["api_key"] == "set (hidden)"
        forget()
        assert configured() == {"provider": None, "model": None, "api_key": None}

    def test_the_key_is_never_returned_in_the_clear(self) -> None:
        configure(api_key=FAKE_KEY)
        assert FAKE_KEY not in repr(configured())


class TestConfiguredCredentialsStayWithTheirProvider:
    def test_switching_provider_never_carries_the_old_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        received: list[tuple[str, str | None]] = []

        class FakeBackend:
            def __init__(self, name: str, key: str | None) -> None:
                received.append((name, key))

            def complete(self, system: str, question: str, model: str) -> str:
                return "```python\nplot(x)\n```"

        def fake(name: str, env_var: str) -> providers.Provider:
            return providers.Provider(
                name=name,
                package="sys",
                pip_name=f"{name}-sdk",
                env_var=env_var,
                default_model="model",
                build=lambda key: FakeBackend(name, key),
            )

        monkeypatch.setattr(
            providers,
            "PROVIDERS",
            (fake("claude", "CLAUDE_TEST_KEY"), fake("openai", "OPENAI_TEST_KEY")),
        )
        monkeypatch.setenv("OPENAI_TEST_KEY", "openai-from-environment")

        configure(provider="claude", api_key="anthropic-secret")
        configure(provider="openai")
        answer = ask("draw it")

        assert answer.provider == "openai"
        assert received[-1] == ("openai", None)

    def test_switch_without_a_new_credential_is_atomic(
        self, installed
    ) -> None:
        installed({"claude", "openai"})
        configure(provider="claude", api_key="anthropic-secret")

        with pytest.raises(UnsupportedInputError, match="OPENAI_API_KEY"):
            configure(provider="openai")

        assert configured()["provider"] == "claude"
        assert configured()["api_key"] == "set (hidden)"

    def test_a_per_call_key_does_not_inherit_a_different_default(
        self, installed
    ) -> None:
        installed({"claude", "openai"})
        configure(provider="claude", api_key="anthropic-secret")

        with pytest.raises(UnsupportedInputError, match="explicit provider"):
            ask("draw it", api_key="possibly-an-openai-key")


class TestTheCoreStaysOffline:
    def test_no_sdk_is_imported_by_importing_the_package(self) -> None:
        import sys

        import mathslate.ai  # noqa: F401

        for module in ("anthropic", "openai", "google.genai"):
            assert module not in sys.modules
