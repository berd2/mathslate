"""``explain()`` — what went wrong, and the line that fixes it.

A beginner's first mistakes are not mathematical, they are shape mistakes:
``plot(sin(x), 0, 6.28)`` instead of ``plot(sin(x), (x, 0, 6.28))``, a list
where a tuple was meant, ``=`` where ``Eq`` was meant. MathSlate already answers
those precisely — the whole point of writing its errors for a learner — but the
answer is a paragraph the reader still has to turn back into code.

This closes that last step, and it is deliberately the *smallest* thing that
does. The error message is the evidence, not the model's memory of MathSlate:

* A :class:`~mathslate.errors.MathSlateError` was raised by MathSlate itself.
  Its message already names the mistake and usually shows the right shape, so
  the model is told to treat it as authoritative and turn it into a corrected
  line rather than re-diagnose it. This is the same "explain the computation
  rather than perform it" split :func:`mathslate.ai.describe` makes.

* Anything else — a ``SyntaxError`` from ``solve(x**2 - 4 = 0)``, a ``TypeError``
  from inside SymPy — did not come from MathSlate and has no curated message.
  The model is working from general Python/SymPy knowledge there, so the reply
  is labelled as the less certain of the two rather than presented the same way.

Nothing is executed. Like :func:`~mathslate.ai.ask`, this returns the corrected
code for the reader to look at; ``.run()`` is still a line they type themselves.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

from .._text import safe_print
from ..errors import MathSlateError, UnsupportedInputError
from .suggest import Suggestion, _complete, _split, contract_reference

__all__ = ["explain"]

#: Frames of the traceback to show the model. The useful ones are the caller's
#: own line and the MathSlate frame that refused it; a deep SymPy stack is
#: noise that crowds out both.
_MAX_FRAMES: int = 12


def explain(
    error: BaseException | None = None,
    code: str | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    verbose: bool = True,
) -> Suggestion:
    """Explain the last error and suggest the corrected code.

    With no arguments this reads the exception Python just reported, so the
    call after a failed cell is simply ``explain()``. Pass ``error`` to explain
    a specific caught exception, or ``code`` alone to have a snippet reviewed
    without running it at all.

    Prints the explanation by default — reading it is the point — and returns
    the :class:`~mathslate.ai.Suggestion` holding the corrected code.

    Examples
    --------
    >>> from mathslate import plot, sin, x                      # doctest: +SKIP
    >>> from mathslate.ai import explain                        # doctest: +SKIP
    >>> plot(sin(x), 0, 6.28)                                   # doctest: +SKIP
    UnsupportedInputError: a range must be written (symbol, lo, hi) ...
    >>> explain()                                               # doctest: +SKIP
    A range is one tuple naming its symbol, not two bare numbers ...
    """
    if error is None and code is None:
        error = _last_exception()
        if error is None:
            raise UnsupportedInputError(
                "explain() found no error to explain. Call it straight after a "
                "failed cell, pass one you caught yourself — explain(err) — or "
                "pass code to review without running it: explain(code=\"...\")."
            )
    if error is not None and not isinstance(error, BaseException):
        raise UnsupportedInputError(
            f"explain() takes an exception, not {type(error).__name__}. To "
            'review a snippet instead, name the argument: explain(code="...").'
        )

    report = _report(error, code)
    reply, chosen, model_name = _complete(
        _explain_prompt(trusted=_raised_by_mathslate(error)),
        report,
        provider,
        model,
        api_key,
    )
    fixed, commentary = _split(reply)
    suggestion = Suggestion(
        code=fixed,
        question=report,
        provider=chosen.name,
        model=model_name,
        commentary=commentary,
        raw=reply,
    )
    if verbose:
        if commentary:
            safe_print(commentary)
        if fixed:
            safe_print("")
            safe_print(fixed)
    return suggestion


def repair_suggestion(
    suggestion: Suggestion,
    error: BaseException,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    verbose: bool = True,
) -> Suggestion:
    """A second draft of ``suggestion``, given the error the first one produced.

    Backs :meth:`mathslate.ai.Suggestion.repair`. The evidence is assembled the
    same way :func:`explain` assembles it, because it is the same evidence — but
    the prompt asks for working code rather than a lesson, and carries the
    original request so the repair still answers the question that was asked
    rather than merely compiling.
    """
    report = "\n\n".join(
        [
            f"This was written to answer: {suggestion.question.strip()}",
            "It did not work. The code:",
            suggestion.code.strip(),
            _report(error, None),
        ]
    )
    reply, chosen, model_name = _complete(
        _repair_prompt(trusted=_raised_by_mathslate(error)),
        report,
        provider,
        model,
        api_key,
    )
    code, commentary = _split(reply)
    repaired = Suggestion(
        code=code,
        question=suggestion.question,
        provider=chosen.name,
        model=model_name,
        commentary=commentary,
        raw=reply,
    )
    if verbose:
        if commentary:
            safe_print(commentary)
        safe_print(code)
    return repaired


def _repair_prompt(*, trusted: bool) -> str:
    """What the model is told when its own previous answer failed."""
    certainty = (
        "The error came from MathSlate and is accurate. Fix the cause it names."
        if trusted
        else "The error did not come from MathSlate's own checks, so work out "
        "the cause from ordinary Python and SymPy behaviour."
    )
    return (
        "Your previous answer to a MathSlate question did not work. Write the "
        "corrected version.\n\n"
        f"{contract_reference()}\n"
        f"{certainty}\n\n"
        "Rules for your answer:\n"
        "- ONE ```python fenced block containing the whole corrected snippet, "
        "not a diff and not only the changed line.\n"
        "- It must still answer the original question. Do not quietly solve an "
        "easier one, and do not drop part of the request to make the error go "
        "away.\n"
        "- One short sentence outside the block saying what was wrong.\n"
        "- If the request cannot be done in MathSlate at all, say so in that "
        "sentence and give the nearest thing it can do.\n"
        "- No imports. Do not invent MathSlate functions."
    )


def _last_exception() -> BaseException | None:
    """The exception Python or IPython just reported, if there was one.

    ``sys.last_value`` is set by the default excepthook and by IPython's, which
    is why ``explain()`` needs no hook of its own: the state it reads is
    already there whether the cell failed a moment ago or the script did.
    """
    return getattr(sys, "last_value", None)


def _report(error: BaseException | None, code: str | None) -> str:
    """The evidence handed to the model: the error, the call site, the stack."""
    lines: list[str] = []
    if code is not None:
        lines += ["The code:", "", code.strip(), ""]
    if error is None:
        lines.append(
            "This code has not been run. Say whether it is right, and if it is "
            "not, what it should be."
        )
        return "\n".join(lines)

    origin = (
        "MathSlate itself raised this, so the message below is authoritative."
        if _raised_by_mathslate(error)
        else "This came from Python or SymPy, not from MathSlate's own checks."
    )
    lines += [
        f"{type(error).__name__}: {error}",
        "",
        origin,
    ]
    call_site = _call_site(error)
    if call_site and code is None:
        lines += ["", "The line that failed:", "", call_site]
    stack = _stack(error)
    if stack:
        lines += ["", "Traceback (most recent call last):", stack]
    return "\n".join(lines)


def _raised_by_mathslate(error: BaseException | None) -> bool:
    """Whether MathSlate wrote this message, so it can be trusted as evidence.

    Not ``isinstance(error, MathSlateError)``. MathSlate deliberately raises the
    ordinary ``TypeError``/``ValueError`` for a mis-shaped range — a programming
    mistake rather than a mathematical one, documented in manual §4.1 — and that
    is *the* beginner error this function exists for: ``plot(sin(x), 0, 6.28)``.
    Those messages are as curated as any ``MathSlateError``. What actually
    distinguishes them is the frame that raised, so that is what is asked.
    """
    if error is None:
        return False
    if isinstance(error, MathSlateError):
        return True
    frames = traceback.extract_tb(error.__traceback__)
    return bool(frames) and _is_mathslate(frames[-1].filename)


def _call_site(error: BaseException) -> str | None:
    """The caller's own failing line — not the MathSlate frame that refused it.

    A ``SyntaxError`` never entered a frame at all and carries its own text,
    which is the only place the offending line exists.
    """
    if isinstance(error, SyntaxError) and error.text:
        return error.text.strip()
    frames = traceback.extract_tb(error.__traceback__)
    outside = [frame for frame in frames if not _is_mathslate(frame.filename)]
    for frame in reversed(outside):
        if frame.line:
            return frame.line.strip()
    return None


def _stack(error: BaseException) -> str:
    """A trimmed traceback. Deep SymPy frames crowd out the two that matter."""
    frames = traceback.extract_tb(error.__traceback__)
    if not frames:
        return ""
    return "".join(traceback.format_list(frames[-_MAX_FRAMES:])).rstrip()


#: This package's own directory. Matching on the *path* rather than on the
#: string "mathslate" matters: a checkout, a virtualenv and a notebook are all
#: routinely inside a folder of that name — the project's own repository is —
#: and a substring test calls the caller's file a MathSlate frame, which would
#: attribute the reader's mistake to the library and trust the wrong message.
_PACKAGE_ROOT: str = str(Path(__file__).resolve().parent.parent)


def _is_mathslate(filename: str) -> bool:
    """Whether ``filename`` is a module of this package.

    A pseudo-filename (``<stdin>``, ``<ipython-input-3>``) resolves against the
    working directory and so is never inside the package, which is the answer
    that is wanted: those frames are the caller's.
    """
    try:
        return Path(filename).resolve().is_relative_to(_PACKAGE_ROOT)
    except (OSError, ValueError):  # pragma: no cover - unresolvable path
        return False


def _explain_prompt(*, trusted: bool) -> str:
    """What the model is told when diagnosing rather than composing.

    ``trusted`` says whether the error came from MathSlate. When it did, the
    message is a curated sentence and re-deriving it can only make it worse; the
    model's job is to turn it into the corrected line. When it did not, the model
    really is reasoning from general knowledge, and the prompt asks it to say so
    rather than sound equally sure in both cases.
    """
    certainty = (
        "The error message came from MathSlate and is correct. Explain what it "
        "means in plainer words and give the corrected code. Do not contradict "
        "it and do not guess at a different cause."
        if trusted
        else "The error did not come from MathSlate's own checks, so you are "
        "reasoning from general Python and SymPy knowledge. If more than one "
        "cause is plausible, say which you think it is and why, in one sentence."
    )
    return (
        "You explain a failed piece of MathSlate code to someone who is still "
        "learning, then show the corrected line.\n\n"
        f"{contract_reference()}\n"
        f"{certainty}\n\n"
        "Rules for your answer:\n"
        "- Two or three short sentences of explanation, in plain language, "
        "outside the code block. Name what was written and what was expected.\n"
        "- Then ONE ```python fenced block: the smallest corrected version of "
        "the caller's own line. No imports, no invented MathSlate functions.\n"
        "- If the code cannot be repaired as written because it asks for "
        "something MathSlate does not do, say that in the explanation and give "
        "the nearest thing it does do.\n"
        "- Do not restate the traceback."
    )


def _ipython_hint(shell: Any) -> None:  # pragma: no cover - notebook glue
    """Add "run explain()" to MathSlate's own tracebacks in IPython.

    Registered by :func:`install_hint`, never on import. The hint is one line of
    text: no network call happens until the reader actually asks for one, which
    is the same bargain the rest of this package makes.
    """
    original = shell.showtraceback

    def showtraceback(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        error = getattr(sys, "last_value", None)
        if isinstance(error, MathSlateError):
            safe_print("  → mathslate.ai.explain() will suggest a fix.")
        return result

    shell.showtraceback = showtraceback


def install_hint() -> bool:
    """Offer ``explain()`` in the traceback of every MathSlate error.

    Opt-in and notebook-only: it appends one line of text to MathSlate's own
    error output so a reader who does not know the function exists can find it.
    Returns ``False`` when there is no IPython shell to attach to.
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    if getattr(shell, "_mathslate_explain_hint", False):
        return True
    _ipython_hint(shell)
    shell._mathslate_explain_hint = True
    return True
