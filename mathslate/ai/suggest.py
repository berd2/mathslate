"""``ask()`` — a question in, MathSlate code out.

The system prompt is built from the dispatch contract itself rather than being
prose written alongside it, so the two cannot drift: if a row is added to
:data:`mathslate.core.dispatch.KINDS`, the assistant is told about it in the
same commit.

PRD goal 5 is "the API is easy for an AI to generate correctly". This module is
where that stops being an aspiration and becomes a measurement — the prompt is
short because the API is small, and that is the whole argument for keeping the
API small.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

from .._text import safe_print
from ..errors import MathSlateError, UnsupportedInputError
from .providers import Backend, Provider, resolve_provider
from .sandbox import _run_restricted, _validate_code

__all__ = [
    "check_connection",
    "RunResult",
    "Suggestion",
    "ask",
    "configure",
    "configured",
    "forget",
    "system_prompt",
]

_FENCE = re.compile(r"```(?:python)?\n(?P<body>.*?)```", re.DOTALL)

#: What replaces a credential on its way into an error message.
_HIDDEN: str = "[hidden]"

#: The shortest run of a key still worth masking. Below this a "prefix" is
#: `sk-a` — shared by every key the vendor ever issued, so hiding it protects
#: nothing and only mangles ordinary error prose.
_MIN_SECRET_RUN: int = 8

#: Key shapes a provider may echo even when we hold no copy to match against —
#: a key read from the environment by the vendor SDK itself, or the second key
#: in a message about the first. Matched by shape rather than by value, because
#: the point is to fail closed on the credential we were never given.
_SECRET_SHAPES = re.compile(
    r"sk-[A-Za-z0-9_\-]{4,}"       # OpenAI, Anthropic
    r"|AIza[A-Za-z0-9_\-]{10,}"    # Google
    r"|gsk_[A-Za-z0-9_\-]{10,}"    # Groq
    r"|\b[A-Fa-f0-9]{32,}\b"       # a bare hex secret
)


def _redacted(text: str, key: str | None) -> str:
    """Provider text with anything credential-shaped taken out of it.

    A plain ``text.replace(key, ...)`` is not enough, because a provider
    rejecting a key rarely quotes it whole: the usual shape is a head and a
    tail around an ellipsis — ``invalid x-api-key sk-ant-api03...7890`` — which
    the whole-key replacement walks straight past. So known-key *runs* are
    masked from the longest down, and then anything merely key-shaped, which
    also covers a key the SDK read from the environment and we never held.
    """
    if key:
        for size in range(len(key), _MIN_SECRET_RUN - 1, -1):
            text = text.replace(key[:size], _HIDDEN).replace(key[-size:], _HIDDEN)
    return _SECRET_SHAPES.sub(_HIDDEN, text)


def _provider_failed(
    exc: Exception, provider_name: str, key: str | None, doing: str
) -> UnsupportedInputError:
    """The error raised when a provider call fails, with no credential in it.

    Deliberately *not* chained with ``raise ... from exc``. Masking the text of
    the new message is pointless while the old one is still attached: Python
    prints the whole chain, so the original — the exception that actually
    quoted the key — is re-printed verbatim under "The above exception was the
    direct cause of the following exception". The type name and the redacted
    detail below carry everything the chain would have said that is safe to
    say, and both call sites go through here so neither can regain the leak.
    """
    detail = _redacted(str(exc).strip(), key)
    return UnsupportedInputError(
        f"{provider_name.title()} did not complete {doing} "
        f"({type(exc).__name__}: {detail or 'no detail'}). "
        "Check the API key, model access, quota, and network, then retry."
    )


#: Set by :func:`configure`; ``ask()`` falls back to auto-detection.
_DEFAULTS: dict[str, Any] = {
    "provider": None,
    "model": None,
    "api_key": None,
    "remembered": False,
}

class RunResult(dict[str, Any]):
    """Values produced by :meth:`Suggestion.run`.

    The isolated executor deliberately keeps its MathSlate/SymPy helper names
    private.  This mapping therefore contains only names written by the
    suggestion.  When the visible code ends with an expression (as generated
    plot requests normally do), its value is available as ``result`` and is
    displayed directly in a notebook.
    """

    def _ipython_display_(self) -> None:
        """Render a final plot/value instead of the implementation mapping."""
        if "result" not in self:
            return
        try:
            from IPython.display import display
        except ImportError:
            return
        display(self["result"])


def configure(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    *,
    remember: bool = False,
) -> None:
    """Fix the provider, model or key for the rest of the session.

    Every argument is optional and ``None`` means "leave as it was", so a
    notebook can set the provider once in its first cell. ``remember=True``
    stores the key in the operating-system credential manager, never in the
    project or notebook.
    """
    current_provider = _DEFAULTS["provider"]
    current_key = _DEFAULTS["api_key"]
    switching_provider = (
        provider is not None
        and current_provider is not None
        and provider != current_provider
    )
    # A session key is bound to its provider. On a switch, only an explicitly
    # supplied key or one saved for the new provider may be considered.
    next_key = (
        None if switching_provider and api_key is None
        else current_key if api_key is None
        else api_key
    )
    loaded_saved = False
    if provider is not None and next_key is None:
        from .credentials import load_credential

        saved = load_credential(provider)
        next_key = saved.api_key if saved is not None else None
        loaded_saved = saved is not None
    if provider is not None:
        if switching_provider and api_key is None:
            # An explicit key belongs to the provider it was configured with.
            # Never carry it across a provider switch. If the new provider has
            # an environment credential it can be selected safely; otherwise
            # the caller must supply the matching key alongside the provider.
            resolve_provider(provider, api_key=next_key)
        else:
            resolve_provider(
                provider, api_key=next_key
            )  # fail now, not at the first question
        _DEFAULTS["provider"] = provider
    if model is not None:
        _DEFAULTS["model"] = model
    if api_key is not None or loaded_saved or (provider is not None and next_key is None):
        _DEFAULTS["api_key"] = next_key
        _DEFAULTS["remembered"] = loaded_saved
    if remember:
        from .credentials import save_credential

        chosen = provider or _DEFAULTS["provider"]
        key = api_key or _DEFAULTS["api_key"]
        if not chosen or not key:
            raise UnsupportedInputError(
                "remember=True needs both a provider and an API key."
            )
        save_credential(chosen, key, model or _DEFAULTS["model"])
        _DEFAULTS["remembered"] = True


def forget(*, persistent: bool = False, provider: str | None = None) -> None:
    """Drop the configured provider, model and key.

    ``configure()`` treats ``None`` as "leave as it was", which is what makes
    it convenient to call twice — and which left no way to take a key back out
    of the session once it was in. This is that way.
    """
    chosen = provider or _DEFAULTS["provider"]
    if persistent:
        from .credentials import delete_credential

        delete_credential(chosen)
    _DEFAULTS.update(
        {"provider": None, "model": None, "api_key": None, "remembered": False}
    )


def configured() -> dict[str, Any]:
    """What :func:`configure` is currently holding. The key is never shown."""
    key = _DEFAULTS["api_key"]
    if key is None:
        from .credentials import load_credential

        saved = load_credential(_DEFAULTS["provider"])
        if saved is not None:
            return {
                "provider": saved.provider,
                "model": _DEFAULTS["model"] or saved.model,
                "api_key": "saved securely",
            }
    return {
        "provider": _DEFAULTS["provider"],
        "model": _DEFAULTS["model"],
        "api_key": (
            None
            if key is None
            else "saved securely" if _DEFAULTS["remembered"] else "set (hidden)"
        ),
    }


@dataclass(frozen=True)
class Suggestion:
    """Code a model wrote, which you read before it runs.

    Nothing here executes on its own. :meth:`run` exists and is one line, but
    it is a line *you* type — a model writing Python and the library running it
    unseen is the one thing a learner cannot check, and checking is the point.
    """

    code: str
    question: str
    provider: str
    model: str
    #: The reply with the code fence removed — the model's own commentary.
    commentary: str = ""
    raw: str = field(default="", repr=False)

    def show(self) -> str:
        """Print the code and return it, as ``show_python()`` does."""
        safe_print(self.code)
        return self.code

    def validate(self) -> None:
        """Refuse code outside MathSlate's small expression-oriented policy.

        This blocks prompt-injected imports, file/network access and Python
        introspection. It is a policy guard, not an operating-system sandbox;
        callers needing arbitrary Python must opt into :meth:`run`'s unsafe
        mode explicitly.
        """
        _validate_code(self.code)

    def repair(
        self,
        error: BaseException | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        verbose: bool = True,
    ) -> "Suggestion":
        """Ask for this code again, with the error it produced as evidence.

        A model writing against an API it half-remembers gets closer on the
        second try when it is shown what actually happened — the finding behind
        the SageMath-agent results, and the useful half of an agent loop. The
        loop is deliberately not closed here: this returns a *new suggestion*
        rather than running it, so the reader still sees the code before it
        executes. Retrying is the part worth automating; skipping the look is
        not.

        With no argument the code is validated and the refusal becomes the
        evidence. Pass the exception from :meth:`run` to repair a failure that
        only showed up once it ran.

        Examples
        --------
        >>> from mathslate.ai import ask                        # doctest: +SKIP
        >>> draft = ask("plot the tangent")                     # doctest: +SKIP
        >>> try:                                                # doctest: +SKIP
        ...     draft.run()
        ... except MathSlateError as failure:
        ...     better = draft.repair(failure)
        """
        from .explain import repair_suggestion

        if error is None:
            try:
                self.validate()
            except MathSlateError as refusal:
                error = refusal
            else:
                raise UnsupportedInputError(
                    "there is nothing to repair: this code passes validation. "
                    "If it failed when it ran, pass that error — "
                    "suggestion.repair(err)."
                )
        return repair_suggestion(
            self,
            error,
            provider=provider,
            model=model,
            api_key=api_key,
            verbose=verbose,
        )

    def run(
        self,
        namespace: dict[str, Any] | None = None,
        *,
        unsafe: bool = False,
        show_code: bool = True,
    ) -> RunResult | dict[str, Any]:
        """Execute the visible code after validating it.

        The default namespace contains only the documented MathSlate/SymPy
        names and a minimal builtin set. Restricted code runs in a separate
        process under a wall-clock budget (see :data:`_RUN_BUDGET`), so a
        legal-looking but runaway expression cannot hang or corrupt the caller.
        Its return value contains only values the suggestion created, never the
        internal MathSlate/SymPy execution namespace. A final bare expression
        such as ``plot(tan(x))`` is returned as ``result`` and displayed in a
        notebook. By default it also prints the visible code first, so
        ``ask("...").run()`` retains the review step; pass ``show_code=False``
        only when the code is already visible in another interface.
        A supplied namespace is deliberately available only in ``unsafe`` mode:
        its objects could carry capabilities that an allowed method name could
        invoke. Pass ``unsafe=True`` to recover normal Python execution,
        including imports, filesystem access, a supplied namespace and no time
        limit; never use that mode for untrusted model output.
        """
        if show_code:
            self.show()
        scope: dict[str, Any] = namespace if namespace is not None else {}
        if unsafe:
            if "plot" not in scope:
                exec("from mathslate import *", scope)  # noqa: S102
            exec(compile(self.code, "<mathslate.ai>", "exec"), scope)  # noqa: S102
            return scope

        self.validate()
        if namespace is not None:
            # `is not None`, not a truthiness test: an empty dict is still a
            # caller asking to be given the results in *their* object, and
            # restricted execution can no longer do that — the names come back
            # from another process, so nothing is filled in place. Accepting
            # `{}` silently would leave that caller reading an empty mapping
            # and finding a KeyError where the answer used to be.
            raise UnsupportedInputError(
                "restricted execution does not accept a namespace because its "
                "objects may carry capabilities, and its results come back from "
                "a separate process rather than being written into yours — read "
                "them from the returned mapping. Use run(namespace, unsafe=True) "
                "only if you trust both the code and the namespace."
            )

        assigned, output = _run_restricted(self.code)
        if output:
            safe_print(output, end="")
        return RunResult(assigned)

    def _repr_html_(self) -> str:
        escaped = (
            self.code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        return (
            f"<div style='opacity:0.6;font-size:0.9em'>{self.provider} · "
            f"{self.model} — read it before running it</div>"
            f"<pre style='padding:8px'><code>{escaped}</code></pre>"
        )

    def __str__(self) -> str:
        return self.code


def contract_reference() -> str:
    """The API description both the writing and the diagnosing prompts share.

    Built from :data:`mathslate.core.dispatch.KINDS` rather than written out
    beside it, so a kind cannot be added without every prompt learning about it
    in the same commit — which is the point §22 of the manual makes about this
    module, and applies as much to explaining a mistake as to writing code.
    """
    from ..core.dispatch import KINDS

    return textwrap.dedent(
        f"""\
        MathSlate wraps SymPy, NumPy and Plotly behind one function, `plot()`,
        which infers what was meant. Assume `from mathslate import *` has run:
        `plot`, `polar`, `analyze`, `table`, `slider`, `animate`, `dataset`,
        `show_python`, the SymPy functions (`sin`, `cos`, `exp`, `sqrt`, `log`,
        `floor`, `Abs`, `diff`, `integrate`, `solve`, `Eq`, `Matrix`, ...) and
        the symbols `x, y, z, t, n, k, theta` all already exist.

        The dispatch contract — one rule: a `list` means several things
        together, a `tuple` means one vector-valued object.

          plot(sin(x)/x)                      a 2D curve
          plot([sin(x), cos(x)])              two curves overlaid
          plot((cos(t), sin(t)))              one parametric curve
          plot((cos(t), sin(t), t))           a 3D space curve
          plot(x*y)                           a surface
          plot(x*y, kind="contour")           the same, flat
          plot(Eq(x**2 + y**2, 4))            an implicit curve
          polar(1 + cos(t))                   r = f(theta)
          plot(Matrix([[2, 1], [1, 3]]))      a matrix as a transformation
          plot(sin(x), (x, 0, 6.28))          an explicit range
          plot([1.0, 2.0, 3.0], kind="hist")  a histogram

        A range is always the triple `(symbol, lo, hi)` — `plot(sin(x), (x, 0,
        6.28))`, never `plot(sin(x), 0, 6.28)`.

        Other entry points:
          analyze(expr)      roots, extrema, asymptotes, symmetry, period
          table(expr)        the same function as a table of values
          a = slider(-5, 5)  then plot(a*sin(x)) — no callback needed
          dataset({{...}})     then .fit(a*x + b) to fit a symbolic model

        Plot kinds that can come back: {", ".join(KINDS)}.
        """
    )


def system_prompt() -> str:
    """What the model is told about MathSlate. Built from the real contract."""
    from ..core.dispatch import KINDS

    return textwrap.dedent(
        f"""\
        You write short Python using the MathSlate library, and nothing else.

        MathSlate wraps SymPy, NumPy and Plotly behind one function, `plot()`,
        which infers what was meant. Assume `from mathslate import *` has run:
        `plot`, `polar`, `analyze`, `table`, `slider`, `animate`, `dataset`,
        `show_python`, the SymPy functions (`sin`, `cos`, `exp`, `sqrt`, `log`,
        `floor`, `Abs`, `diff`, `integrate`, `solve`, `Eq`, `Matrix`, ...) and
        the symbols `x, y, z, t, n, k, theta` all already exist.

        The dispatch contract — one rule: a `list` means several things
        together, a `tuple` means one vector-valued object.

          plot(sin(x)/x)                      a 2D curve
          plot([sin(x), cos(x)])              two curves overlaid
          plot((cos(t), sin(t)))              one parametric curve
          plot((cos(t), sin(t), t))           a 3D space curve
          plot(x*y)                           a surface
          plot(x*y, kind="contour")           the same, flat
          plot(Eq(x**2 + y**2, 4))            an implicit curve
          polar(1 + cos(t))                   r = f(theta)
          plot(Matrix([[2, 1], [1, 3]]))      a matrix as a transformation
          plot(sin(x), (x, 0, 6.28))          an explicit range
          plot([1.0, 2.0, 3.0], kind="hist")  a histogram

        Other entry points:
          analyze(expr)      roots, extrema, asymptotes, symmetry, period
          table(expr)        the same function as a table of values
          a = slider(-5, 5)  then plot(a*sin(x)) — no callback needed
          dataset({{...}})     then .fit(a*x + b) to fit a symbolic model

        Plot kinds that can come back: {", ".join(KINDS)}.

        Rules for your answer:
        - Reply with ONE ```python fenced block and at most two short sentences
          outside it.
        - Do not import anything. Do not define symbols that already exist.
        - Do not invent MathSlate functions. If MathSlate cannot do it, say so
          in one sentence and give the plain SymPy or Plotly that can.
        - Prefer the shortest call that answers the question. MathSlate's
          defaults are good; do not pass arguments that only restate them.
        """
    )


def _complete(
    system: str,
    question: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
) -> tuple[str, Provider, str]:
    """Resolve a provider, send one message, and return ``(reply, provider, model)``.

    Every entry point that talks to a model goes through here, so credential
    resolution — a per-call key not inheriting a session provider, a saved key
    being found for the named one, the key never reaching an error message —
    is decided in one place rather than re-implemented per feature.
    """
    chosen, model_name, effective_key = _resolve_request(
        provider=provider, model=model, api_key=api_key
    )
    backend: Backend = chosen.build(effective_key)

    try:
        reply = backend.complete(system, question, model_name)
    # Blind on purpose: every provider SDK raises its own hierarchy, and a
    # network stack under it raises anything at all. `from None` is what
    # keeps the credential out of the chain — see `_provider_failed`.
    except Exception as exc:  # noqa: BLE001
        raise _provider_failed(
            exc, chosen.name, effective_key, "the request"
        ) from None
    if not reply or not reply.strip():
        raise UnsupportedInputError(
            f"provider {chosen.name!r} returned no text. Retry the request, "
            "check the provider status, or choose another model."
        )
    return reply, chosen, model_name


def _resolve_request(
    *, provider: str | None, model: str | None, api_key: str | None
) -> tuple[Provider, str, str | None]:
    """Resolve one request without carrying credentials across providers.

    Session credentials and model names belong to the session provider.  An
    explicit different provider must start with an empty credential context so
    its own saved key or environment variable is used.  Likewise, a per-call
    key with no provider must not inherit the session provider: only the caller
    knows which service issued that key.
    """
    default_provider = _DEFAULTS["provider"]
    use_session = api_key is None and (
        provider is None or provider == default_provider
    )
    provider_name = provider or (default_provider if use_session else None)
    effective_key = api_key or (_DEFAULTS["api_key"] if use_session else None)
    saved = None
    if effective_key is None and api_key is None:
        from .credentials import load_credential

        saved = load_credential(provider_name)
        if saved is not None:
            provider_name = saved.provider
            effective_key = saved.api_key
    chosen = resolve_provider(provider_name, api_key=effective_key)
    model_name = (
        model
        or (_DEFAULTS["model"] if use_session else None)
        or (saved.model if saved is not None else None)
        or chosen.default_model
    )
    return chosen, model_name, effective_key


#: Added to the prompt when the caller names a result the question is about.
_FOLLOW_UP_RULES: str = textwrap.dedent(
    """\

    The reader already has the result described below, and their question is
    about it. "It", "this", "the plot" and "the same thing" all mean that
    result. Those facts were computed by MathSlate and are correct — do not
    re-derive them, and do not contradict them.

    Write the code for what they are asking for *now*, and make it standalone:
    name the expression again rather than referring to a variable, because you
    were not told what the reader called it.
    """
)


def ask(
    question: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    verbose: bool = False,
    *,
    about: object | None = None,
) -> Suggestion:
    """Turn a question into MathSlate code. Nothing is executed.

    ``about`` names a result the question follows on from — a plot, an analysis,
    a table, a fit — so that "show it on a log scale" has an *it*. What is sent
    is the same computed summary :func:`mathslate.ai.facts` returns, so the
    context is a page of verified numbers rather than a transcript, and it is
    named rather than collected: nothing about the session leaves the machine
    because a question was asked near it.

    Examples
    --------
    >>> from mathslate.ai import ask                       # doctest: +SKIP
    >>> print(ask("plot the tangent over one period").code)  # doctest: +SKIP
    plot(tan(x), (x, -1.5, 1.5))
    >>> drawn = plot(sin(x)/x)                             # doctest: +SKIP
    >>> ask("show the same thing on a log scale", about=drawn)  # doctest: +SKIP
    """
    if not question.strip():
        raise UnsupportedInputError("ask() needs a question.")

    system = system_prompt()
    message = question
    if about is not None:
        from .describe import facts

        system += _FOLLOW_UP_RULES
        message = (
            "The reader is looking at this result:\n\n"
            f"{json.dumps(facts(about), indent=2, ensure_ascii=False)}\n\n"
            f"Their question: {question}"
        )

    reply, chosen, model_name = _complete(
        system, message, provider, model, api_key
    )
    code, commentary = _split(reply)
    suggestion = Suggestion(
        code=code,
        question=question,
        provider=chosen.name,
        model=model_name,
        commentary=commentary,
        raw=reply,
    )
    if verbose:
        safe_print(f"# {chosen.name} · {model_name}")
        safe_print(code)
    return suggestion


def check_connection(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Check a provider credential without asking it a MathSlate question.

    Saved credentials are resolved exactly as they are by :func:`ask`.  The
    probe sends only a fixed, non-mathematical message and returns a concise
    local status instead of treating the provider reply as generated code.
    """
    chosen, model_name, effective_key = _resolve_request(
        provider=provider, model=model, api_key=api_key
    )
    backend: Backend = chosen.build(effective_key)
    try:
        reply = backend.complete(
            "You are an API connection check. Reply with exactly OK.",
            "Reply exactly OK.",
            model_name,
        )
    # Blind on purpose: every provider SDK raises its own hierarchy, and a
    # network stack under it raises anything at all. `from None` is what
    # keeps the credential out of the chain — see `_provider_failed`.
    except Exception as exc:  # noqa: BLE001
        raise _provider_failed(
            exc, chosen.name, effective_key, "the connection check"
        ) from None
    if not reply or not reply.strip():
        raise UnsupportedInputError(
            f"provider {chosen.name!r} returned no text for the connection check. "
            "Check the API key, model access, quota, and network, then retry."
        )
    return f"{chosen.name.title()} connection is working ({model_name})."


def _split(reply: str) -> tuple[str, str]:
    """Separate the fenced code from the model's prose.

    A model that ignores the fence instruction still has to be usable, so an
    unfenced reply is treated as code rather than thrown away — but only after
    the fenced case, which is what the prompt asks for.
    """
    matches = _FENCE.findall(reply)
    if matches:
        code = "\n\n".join(block.strip() for block in matches)
        commentary = _FENCE.sub("", reply).strip()
        return code, commentary
    return reply.strip(), ""
