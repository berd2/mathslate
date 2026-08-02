"""The three backends, behind one small protocol.

Each adapter is the least code that turns "a system prompt and a question" into
"some text". Provider SDKs are imported **inside** the call, never at module
scope, so importing :mod:`mathslate.ai` costs nothing and a missing SDK is a
sentence rather than an ImportError from three frames down.

Model names are defaults, not decisions: every provider moves faster than a
release cycle, so ``ask(..., model=...)`` overrides and the default is only
what a first-time caller gets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Protocol

from ..errors import UnsupportedInputError

__all__ = [
    "Provider",
    "Backend",
    "PROVIDERS",
    "available_providers",
    "resolve_provider",
]

#: How many tokens a suggestion may take. MathSlate answers are short.
MAX_TOKENS: int = 1500
#: A notebook must not look frozen forever when a provider or network stalls.
REQUEST_TIMEOUT_SECONDS: float = 30.0


class Backend(Protocol):
    """What a provider adapter has to be able to do."""

    def complete(self, system: str, question: str, model: str) -> str:
        """Return the model's reply to ``question`` under ``system``."""


@dataclass(frozen=True)
class Provider:
    """One named backend: what to install, what to set, and what it defaults to."""

    name: str
    #: The import path, for detection.
    package: str
    #: What to type after `pip install` — not always the import path.
    pip_name: str
    env_var: str
    default_model: str
    build: Callable[[str | None], Backend]

    def installed(self) -> bool:
        import importlib.util

        try:
            return importlib.util.find_spec(self.package) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            # `find_spec("google.genai")` raises rather than returning None when
            # the parent package is absent, which is the ordinary case here.
            return False

    def configured(self) -> bool:
        if self.name == "gemini":
            return bool(
                os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            )
        return bool(os.environ.get(self.env_var))

    @property
    def credential_label(self) -> str:
        """The environment variable name(s) a user can configure."""
        if self.name == "gemini":
            return "GOOGLE_API_KEY or GEMINI_API_KEY"
        return self.env_var

    def ready(self) -> bool:
        return self.installed() and self.configured()

    def describe(self) -> str:
        if not self.installed():
            return f"{self.name}: not installed (pip install {self.pip_name})"
        if not self.configured():
            return f"{self.name}: installed, but {self.credential_label} is not set"
        return f"{self.name}: ready"


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------


class _Anthropic:
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def complete(self, system: str, question: str, model: str) -> str:
        import anthropic

        client = anthropic.Anthropic(
            api_key=self._api_key or None, timeout=REQUEST_TIMEOUT_SECONDS
        )
        reply = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": question}],
        )
        return "".join(
            block.text for block in reply.content if getattr(block, "type", "") == "text"
        )


class _OpenAI:
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def complete(self, system: str, question: str, model: str) -> str:
        import openai

        client = openai.OpenAI(
            api_key=self._api_key or None, timeout=REQUEST_TIMEOUT_SECONDS
        )
        reply = client.chat.completions.create(
            model=model,
            max_completion_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        )
        return reply.choices[0].message.content or ""


class _Gemini:
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def complete(self, system: str, question: str, model: str) -> str:
        from google import genai
        from google.genai import types

        # The SDK emits a warning and chooses for itself when both names are
        # present. Pass the documented GOOGLE_API_KEY precedence explicitly so
        # `ask()` is quiet and deterministic even in an inherited shell.
        key = (
            self._api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        client = genai.Client(
            api_key=key or None,
            http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT_SECONDS * 1000)),
        )
        reply = client.models.generate_content(
            model=model,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=MAX_TOKENS
            ),
        )
        return reply.text or ""


#: In preference order. Claude first because MathSlate's own development is
#: done against it, so its behaviour on this prompt is the best understood.
PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="claude",
        package="anthropic",
        pip_name="anthropic",
        env_var="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-5",
        build=_Anthropic,
    ),
    Provider(
        name="openai",
        package="openai",
        pip_name="openai",
        env_var="OPENAI_API_KEY",
        default_model="gpt-5",
        build=_OpenAI,
    ),
    Provider(
        name="gemini",
        package="google.genai",
        pip_name="google-genai",
        env_var="GOOGLE_API_KEY",
        default_model="gemini-3.5-flash",
        build=_Gemini,
    ),
)


def available_providers() -> tuple[Provider, ...]:
    """Every provider that is both installed and configured."""
    return tuple(provider for provider in PROVIDERS if provider.ready())


def resolve_provider(
    name: str | None = None, *, api_key: str | None = None
) -> Provider:
    """Pick a provider, or explain precisely what is missing.

    An explicitly supplied key is a complete credential. Requiring the same
    key to also exist in an environment variable would make ``ask(api_key=...)``
    and ``configure(api_key=...)`` unusable in notebooks.
    """
    if name is not None:
        for provider in PROVIDERS:
            if provider.name == name:
                if not provider.installed():
                    raise UnsupportedInputError(
                        f"provider {name!r} needs its SDK: "
                        f"pip install {provider.pip_name}."
                    )
                if not api_key and not provider.configured():
                    raise UnsupportedInputError(
                        f"provider {name!r} needs {provider.credential_label} in the "
                        "environment or an explicit api_key=."
                    )
                return provider
        known = ", ".join(p.name for p in PROVIDERS)
        raise UnsupportedInputError(f"unknown provider {name!r}; known: {known}.")

    if api_key:
        # A key is a credential for exactly one service, and nothing in the
        # string reliably says which. Guessing would send the user's secret to
        # a company it does not belong to, so an explicit key requires an
        # explicit provider unless there is only one candidate.
        candidates = tuple(p for p in PROVIDERS if p.installed())
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise UnsupportedInputError(
                "an api_key was given but no provider SDK is installed, so there "
                "is nothing to send it to. Install one with "
                "`pip install 'mathslate[ai]'`."
            )
        names = ", ".join(p.name for p in candidates)
        raise UnsupportedInputError(
            f"an explicit api_key needs an explicit provider: {names} are all "
            "installed, and sending a key to the wrong one would hand your "
            f"credential to the wrong company. Say which, e.g. "
            f"provider='{candidates[0].name}'."
        )

    ready = available_providers()
    if ready:
        return ready[0]
    status = "\n  ".join(provider.describe() for provider in PROVIDERS)
    raise UnsupportedInputError(
        "no AI provider is ready. MathSlate itself needs none of this and works "
        "offline; the assistant is an optional extra.\n"
        f"  {status}\n"
        "Install one with `pip install 'mathslate[ai]'` and set its API key."
    )
