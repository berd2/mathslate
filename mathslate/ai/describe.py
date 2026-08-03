"""``describe()`` — prose about an answer MathSlate has already computed.

:func:`~mathslate.ai.ask` asks a model to *write* something and hands the result
back for review, because code a learner cannot check is the one thing this
project refuses. That caution is about generation. Explaining is a different
job, and a safer one: the roots are already solved, the discontinuities already
found, the fit already least-squared. What is missing is the sentence that says
what they mean.

So the model here never computes. It is given the finished numbers — the same
serialised facts :mod:`mathslate.ai.tools` hands an outside agent, including
every ``approximate`` flag and the ``method`` behind it — and asked only to
narrate them. A number it invents is a number not in the facts, which is a thing
that can be looked for; a root it gets wrong is not a thing it was ever asked
for. That split is what makes this the one place in this package where a model's
reply is shown as an answer rather than as a draft to be inspected.

It is still an extra. Everything MathSlate computes is already legible without
it: ``Analysis.rows()``, ``Table.text()`` and ``PlotResult.summary()`` are the
same facts, written by hand and needing no network.
"""

from __future__ import annotations

import json
from typing import Any

from .._text import safe_print
from ..errors import UnsupportedInputError
from .suggest import _complete

__all__ = ["describe", "facts"]


def facts(result: object) -> dict[str, Any]:
    """The computed facts behind ``result``, as the model will be given them.

    Public because it is the honest answer to "what did you send?" — and
    because it is useful on its own, as the payload to hand any other assistant.
    """
    from ..core.analysis import Analysis
    from ..core.data import Dataset, FitResult
    from ..core.tables import Table
    from ..result import PlotResult
    from . import tools

    # The discriminator is `result`, not `kind`: a plot's serialised facts
    # already carry a `kind` of their own — the plan's, "curve" or "surface" —
    # and spelling both the same way let one silently overwrite the other.
    if isinstance(result, Analysis):
        return {"result": "analysis", **tools._analysis(result)}
    if isinstance(result, PlotResult):
        payload = {"result": "plot", **tools._plot(result)}
        # A plot's `python` is the whole program. It is what makes the tool
        # useful to an agent that wants to run something; here it is a page of
        # NumPy the reader can already see, and it crowds out the numbers.
        payload.pop("python", None)
        return payload
    if isinstance(result, Table):
        return {"result": "table", **tools._table(result)}
    if isinstance(result, FitResult):
        return {
            "result": "fit",
            "model": str(result.model),
            "fitted": str(result.expr),
            "parameters": {
                symbol.name: tools._number(value)
                for symbol, value in result.parameters.items()
            },
            "r_squared": tools._number(result.r_squared),
            "residual_count": int(result.residuals.size),
            "worst_residual": tools._number(
                max((abs(float(r)) for r in result.residuals), default=0.0)
            ),
        }
    if isinstance(result, Dataset):
        return {
            "result": "dataset",
            "columns": list(result.names),
            "rows": int(len(next(iter(result.columns.values())))),
        }
    raise UnsupportedInputError(
        "describe() explains something MathSlate computed — a plot, an "
        "analysis, a table, a fit or a dataset. It does not know what to do "
        f"with {type(result).__name__}. To ask a general question instead, use "
        "mathslate.ai.ask()."
    )


def describe(
    result: object,
    question: str | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    verbose: bool = True,
) -> str:
    """Explain a computed result in prose, and return the explanation.

    ``question`` narrows it — "why is there a gap at zero?", "is this fit any
    good?" — and is answered from the same facts rather than from a fresh
    computation.

    Examples
    --------
    >>> from mathslate import analyze, x                      # doctest: +SKIP
    >>> from mathslate.ai import describe                     # doctest: +SKIP
    >>> describe(analyze(x**3 - 3*x))                         # doctest: +SKIP
    'This is an odd cubic with three roots ...'
    """
    if question is not None and not question.strip():
        raise UnsupportedInputError(
            "describe(result, question=...) needs a question, or none at all."
        )
    payload = facts(result)
    message = (
        "Here is what MathSlate computed:\n\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )
    if question:
        message += f"\n\nThe reader asks: {question.strip()}"
    reply, _chosen, _model = _complete(
        _describe_prompt(), message, provider, model, api_key
    )
    text = reply.strip()
    if verbose:
        safe_print(text)
    return text


def _describe_prompt() -> str:
    """What the model is told when narrating rather than composing.

    The rules are all one rule — the facts are the whole world — because the
    single failure that would matter here is a number the reader believes and
    MathSlate never computed.
    """
    return (
        "You explain mathematics to someone who is still learning, from results "
        "that have already been computed for you.\n\n"
        "The JSON you are given is correct. SymPy solved it, or MathSlate "
        "sampled it and said so. Your job is to say what it means.\n\n"
        "Rules:\n"
        "- Never compute anything yourself, and never state a number that is "
        "not in the facts. If the reader's question cannot be answered from "
        "them, say which computation would answer it — for example "
        "analyze(expr) or table(expr) — rather than working it out.\n"
        "- Where a property is marked \"approximate\": true, say the value was "
        "found by sampling rather than solved, and that there may be more that "
        "the sampling missed. Where it is false, state it plainly.\n"
        "- \"exact\" holds closed forms such as -sqrt(3). Prefer them to the "
        "decimals when you name a value.\n"
        "- Carry any \"notes\" across: they are MathSlate telling the reader "
        "something it noticed, such as a handled discontinuity or a clipped "
        "view.\n"
        "- Three to six sentences of plain prose. No code block, no headings, "
        "no bullet list. Do not restate the JSON field by field."
    )
