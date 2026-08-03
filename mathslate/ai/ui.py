"""A small notebook control panel for the optional AI assistant."""

from __future__ import annotations

import html
import sys
from dataclasses import dataclass, field
from typing import Any

from ..errors import MathSlateError, UnsupportedInputError
from .credentials import credential_store_available, load_credential
from .providers import PROVIDERS, Provider
from .suggest import Suggestion, ask, check_connection, configure, configured, forget

__all__ = ["AssistantPanel", "assistant"]

_KEY_URLS = {
    "gemini": "https://aistudio.google.com/app/apikey",
    "openai": "https://platform.openai.com/api-keys",
    "claude": "https://console.anthropic.com/settings/keys",
}
_EXTRAS = {"gemini": "ai-gemini", "openai": "ai-openai", "claude": "ai"}


def _provider(name: str) -> Provider:
    return next(provider for provider in PROVIDERS if provider.name == name)


def _safe_error(error: Exception, secret: str = "") -> str:
    """Turn provider exceptions into an actionable sentence without a key."""
    message = str(error).strip() or type(error).__name__
    if secret:
        message = message.replace(secret, "[hidden]")
    lower = message.lower()
    if "401" in lower or "403" in lower or "api key" in lower or "authentication" in lower:
        return "Authentication failed. Replace the API key in Settings and try again."
    if "429" in lower or "quota" in lower or "rate limit" in lower:
        return "The provider rejected the request because its quota or rate limit was reached."
    if "timeout" in lower or "timed out" in lower:
        return "The provider did not answer in time. Check the network and retry."
    return message


@dataclass
class AssistantPanel:
    """Notebook widget plus the most recent :class:`Suggestion`."""

    widget: Any
    provider: Any
    api_key: Any
    remember: Any
    model: Any
    question: Any
    status: Any
    code: Any
    ask_button: Any
    check_button: Any
    run_button: Any
    forget_button: Any
    output: Any
    suggestion: Suggestion | None = field(default=None, init=False)

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.widget)


def assistant(question: str = "", *, about: object | None = None) -> AssistantPanel:
    """Return an interactive setup, question, status and result panel.

    ``about`` fixes a result every question in this panel follows on from — the
    plot being worked on — so "now on a log scale" means something in the box
    as well as in :func:`~mathslate.ai.ask`. It is the panel's context for as
    long as the panel lives; open another to ask about something else.

    API keys entered here are session-only unless the user explicitly selects
    ``Remember on this device``. Persistent storage is disabled in JupyterLite.
    """
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise UnsupportedInputError(
            "assistant() needs ipywidgets; install `mathslate[jupyter]`."
        ) from exc

    settings = configured()
    saved_credential = load_credential()
    saved_name = settings.get("provider") or (
        saved_credential.provider if saved_credential is not None else None
    )
    ready = [provider.name for provider in PROVIDERS if provider.ready()]
    initial = saved_name or (ready[0] if ready else "gemini")
    names = [(f"{p.name.title()}{' (recommended)' if p.name == 'gemini' else ''}", p.name)
             for p in PROVIDERS]

    provider_box = widgets.Dropdown(options=names, value=initial, description="Provider")
    key_box = widgets.Password(description="API key", placeholder="Paste the provider API key")
    local_persistence = sys.platform != "emscripten" and credential_store_available()
    remember_box = widgets.Checkbox(
        value=local_persistence,
        description="Remember on this device",
        disabled=not local_persistence,
        indent=False,
    )
    model_box = widgets.Text(
        value=_provider(initial).default_model,
        description="Model",
        placeholder="Provider default",
    )
    key_link = widgets.HTML()
    question_box = widgets.Textarea(
        value=question,
        description="Question",
        placeholder="Describe the MathSlate code you want",
        layout=widgets.Layout(width="100%", height="90px"),
    )
    status = widgets.HTML()
    code = widgets.Textarea(
        description="Code",
        placeholder="The generated code will appear here",
        layout=widgets.Layout(width="100%", height="140px"),
    )
    ask_button = widgets.Button(description="Ask", button_style="primary", icon="paper-plane")
    check_button = widgets.Button(description="Check connection", icon="check")
    run_button = widgets.Button(description="Validate & Run", disabled=True, icon="play")
    forget_button = widgets.Button(description="Forget saved key", icon="trash")
    output = widgets.Output(layout=widgets.Layout(width="100%"))

    panel = AssistantPanel(
        widget=None,
        provider=provider_box,
        api_key=key_box,
        remember=remember_box,
        model=model_box,
        question=question_box,
        status=status,
        code=code,
        ask_button=ask_button,
        check_button=check_button,
        run_button=run_button,
        forget_button=forget_button,
        output=output,
    )

    def set_status(message: str, kind: str = "info") -> None:
        colors = {"info": "#3b82f6", "ok": "#16803c", "error": "#b42318", "wait": "#8a5a00"}
        status.value = (
            f"<div style='border-left:4px solid {colors[kind]};padding:6px 9px'>"
            f"{html.escape(message)}</div>"
        )

    def refresh_provider(*_change: Any) -> None:
        chosen = _provider(provider_box.value)
        saved = load_credential(chosen.name)
        model_box.value = (saved.model if saved is not None and saved.model else chosen.default_model)
        url = _KEY_URLS[chosen.name]
        key_link.value = (
            f"<a href='{url}' target='_blank' rel='noopener noreferrer'>"
            f"Create or manage a {html.escape(chosen.name.title())} API key</a>"
        )
        if not chosen.installed():
            set_status(
                f"{chosen.name.title()} needs its SDK. Run: pip install "
                f"\"mathslate[{_EXTRAS[chosen.name]}]\"",
                "error",
            )
        elif saved is not None:
            set_status(f"{chosen.name.title()} is ready with a securely saved key.", "ok")
        elif chosen.configured():
            set_status(f"{chosen.name.title()} is ready from {chosen.env_var}.", "ok")
        else:
            set_status("Paste an API key, then Ask or Check connection.")

    def submit(_button: Any) -> None:
        chosen = _provider(provider_box.value)
        secret = key_box.value.strip()
        prompt = question_box.value.strip()
        if not prompt:
            set_status("Enter a question before pressing Ask.", "error")
            return
        if not chosen.installed():
            refresh_provider()
            return
        ask_button.disabled = True
        run_button.disabled = True
        set_status(f"Waiting for {chosen.name.title()}…", "wait")
        try:
            if secret:
                configure(
                    provider=chosen.name,
                    model=model_box.value.strip() or None,
                    api_key=secret,
                    remember=remember_box.value,
                )
                key_box.value = ""
            result = ask(
                prompt,
                provider=chosen.name,
                model=model_box.value.strip() or None,
                about=about,
            )
            panel.suggestion = result
            code.value = result.code
            run_button.disabled = False
            set_status(f"Received code from {result.provider} · {result.model}.", "ok")
        except Exception as exc:
            panel.suggestion = None
            code.value = ""
            set_status(_safe_error(exc, secret), "error")
        finally:
            key_box.value = ""
            ask_button.disabled = False

    def check_provider(_button: Any) -> None:
        chosen = _provider(provider_box.value)
        secret = key_box.value.strip()
        if not chosen.installed():
            refresh_provider()
            return
        check_button.disabled = True
        set_status(f"Checking {chosen.name.title()} connection...", "wait")
        try:
            if secret:
                configure(
                    provider=chosen.name,
                    model=model_box.value.strip() or None,
                    api_key=secret,
                    remember=remember_box.value,
                )
            message = check_connection(
                provider=chosen.name,
                model=model_box.value.strip() or None,
            )
            key_box.value = ""
            set_status(message, "ok")
        except Exception as exc:
            set_status(_safe_error(exc, secret), "error")
        finally:
            key_box.value = ""
            check_button.disabled = False

    def run_suggestion(_button: Any) -> None:
        if panel.suggestion is None:
            return
        output.clear_output()
        try:
            panel.suggestion.validate()
            scope = panel.suggestion.run(show_code=False)
            with output:
                from IPython.display import display

                visible = {
                    name: value
                    for name, value in scope.items()
                    if not name.startswith("_")
                }
                for value in visible.values():
                    display(value)
            set_status("Validated and ran the visible code.", "ok")
        except Exception as exc:
            set_status(_safe_error(exc), "error")

    def forget_key(_button: Any) -> None:
        chosen = provider_box.value
        try:
            saved = load_credential(chosen)
            forget(persistent=saved is not None, provider=chosen)
            key_box.value = ""
            set_status(f"Forgot the {chosen.title()} key.", "ok")
        except MathSlateError as exc:
            set_status(str(exc), "error")

    provider_box.observe(refresh_provider, names="value")
    ask_button.on_click(submit)
    check_button.on_click(check_provider)
    run_button.on_click(run_suggestion)
    forget_button.on_click(forget_key)

    if sys.platform == "emscripten":
        persistence_message = (
            "JupyterLite cannot securely persist API keys. Use a local Jupyter "
            "kernel for remembered credentials."
        )
    elif not local_persistence:
        persistence_message = (
            "Secure storage is unavailable, so Remember is disabled. Run "
            "`%pip install keyring` in this notebook, then restart its kernel."
        )
    else:
        persistence_message = (
            "Remembered keys are stored by the operating-system credential manager."
        )
    persistence_note = widgets.HTML(value=f"<small>{persistence_message}</small>")
    panel.widget = widgets.VBox(
        [
            widgets.HTML("<h3>MathSlate AI Assistant</h3>"),
            provider_box,
            key_link,
            key_box,
            remember_box,
            persistence_note,
            model_box,
            question_box,
            widgets.HBox([ask_button, check_button, run_button, forget_button]),
            status,
            code,
            output,
        ],
        layout=widgets.Layout(width="100%", max_width="760px"),
    )
    refresh_provider()
    return panel
