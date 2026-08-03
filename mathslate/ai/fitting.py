"""``suggest_model()`` — which curve to fit, decided by the data.

:meth:`~mathslate.core.data.Dataset.fit` needs a model written on paper:
``a*x + b``, ``a*exp(b*x)``. Choosing it is the step before the mathematics and
the one a beginner has least to go on — the shape is in the numbers, but only if
you already know which numbers to look at.

Those numbers are computable, so MathSlate computes them. A model family is
whatever transform straightens the data: ``y`` against ``x`` for a line,
``log y`` against ``x`` for an exponential, ``log y`` against ``log x`` for a
power law, ``y`` against ``log x`` for a logarithm. The correlation after each
transform separates them cleanly — on exponential data ``log y ~ x`` is 1.000
while the others sit near 0.94 — and that is a measurement, not an opinion.

So the division is the same one :func:`mathslate.ai.describe` makes. MathSlate
measures; the model reads the measurements and names the family, which is a
naming job rather than an arithmetic one. What comes back is a
:class:`~mathslate.ai.Suggestion`: the fit is still a line the reader runs, and
``.r_squared`` still decides whether it was a good idea.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._text import safe_print
from ..errors import UnsupportedInputError
from .suggest import Suggestion, _complete, _split, contract_reference

__all__ = ["suggest_model", "fit_evidence"]

#: Rows shown to the model. Enough to see the shape; the transforms below are
#: what actually decide the family, and they are computed over every row.
_SAMPLE_ROWS: int = 12
#: Below this many usable points a correlation is noise dressed as evidence.
_MIN_POINTS: int = 4


def _straightness(
    horizontal: np.ndarray, vertical: np.ndarray, total: int
) -> dict[str, Any] | None:
    """|r| after one transform, with the number of rows it could actually use.

    The count is not decoration. ``log`` drops every non-positive value, so on
    data that crosses zero ``log(y) ~ x`` is measured over the positive tail
    alone — and a tail is easily straighter than the whole. Reporting the
    correlation by itself would offer a subset's shape as the dataset's, which
    is exactly the kind of true-but-misleading fact this module exists not to
    produce. ``None`` means the transform left too little to measure at all.

    A constant column correlates with nothing — ``corrcoef`` divides by a zero
    standard deviation — and that is a real answer ("this tells you nothing"),
    so it is reported as ``None`` rather than as a NaN.
    """
    usable = np.isfinite(horizontal) & np.isfinite(vertical)
    count = int(np.count_nonzero(usable))
    if count < _MIN_POINTS:
        return None
    a, b = horizontal[usable], vertical[usable]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        value = float(abs(np.corrcoef(a, b)[0, 1]))
    if not np.isfinite(value):
        return None
    return {"r": round(value, 4), "rows_used": count, "of": total}


def _log(values: np.ndarray) -> np.ndarray:
    """``log`` of the positive entries, ``nan`` elsewhere — no warning, no zero."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(values) & (values > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[positive] = np.log(values[positive])
    return out


def fit_evidence(
    data: Any, x: str | None = None, y: str | None = None
) -> dict[str, Any]:
    """What the data itself says about the shape to fit.

    Public because it is worth reading on its own: the straightness figures are
    the whole argument for one model family over another, and a reader who
    disagrees with the suggestion should be able to see what it was based on.
    """
    from ..core.data import Dataset

    if not isinstance(data, Dataset):
        raise UnsupportedInputError(
            "suggest_model() describes a dataset — build one with dataset(...) "
            f"first; got {type(data).__name__}."
        )
    names = data.names
    x_name = x or names[0]
    y_name = y or (names[1] if len(names) > 1 else names[0])
    for chosen in (x_name, y_name):
        if chosen not in names:
            raise UnsupportedInputError(
                f"{chosen!r} is not a column; this dataset has "
                f"{', '.join(names)}."
            )
    if x_name == y_name:
        raise UnsupportedInputError(
            f"the independent and dependent columns are both {x_name!r}. Name "
            f"them: suggest_model(data, x=..., y=...) from {', '.join(names)}."
        )

    inputs = np.asarray(data[x_name], dtype=np.float64)
    targets = np.asarray(data[y_name], dtype=np.float64)
    usable = np.isfinite(inputs) & np.isfinite(targets)
    if int(np.count_nonzero(usable)) < _MIN_POINTS:
        raise UnsupportedInputError(
            f"{int(np.count_nonzero(usable))} usable rows is too few to tell one "
            f"model from another; suggest_model() needs at least {_MIN_POINTS}."
        )
    xs, ys = inputs[usable], targets[usable]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    total = int(xs.size)

    differences = np.diff(ys)
    return {
        "columns": {"x": x_name, "y": y_name},
        "rows": int(xs.size),
        "x_range": [float(xs.min()), float(xs.max())],
        "y_range": [float(ys.min()), float(ys.max())],
        "x_all_positive": bool(np.all(xs > 0.0)),
        "y_all_positive": bool(np.all(ys > 0.0)),
        "y_monotonic": bool(np.all(differences >= 0) or np.all(differences <= 0)),
        # The discriminator. Each entry is |r| after the transform named, so the
        # largest names the family: 1.000 for `log(y) ~ x` is exponential data.
        # `rows_used` guards the comparison — a transform that dropped most of
        # the data is describing what it kept.
        "straightness": {
            "y ~ x  (linear/polynomial)": _straightness(xs, ys, total),
            "log(y) ~ x  (exponential)": _straightness(xs, _log(ys), total),
            "log(y) ~ log(x)  (power law)": _straightness(_log(xs), _log(ys), total),
            "y ~ log(x)  (logarithmic)": _straightness(_log(xs), ys, total),
            "y ~ 1/x  (reciprocal)": _straightness(
                np.where(xs != 0.0, 1.0 / np.where(xs == 0.0, np.nan, xs), np.nan),
                ys,
                total,
            ),
        },
        "sample": [
            [float(a), float(b)]
            for a, b in zip(xs[:_SAMPLE_ROWS], ys[:_SAMPLE_ROWS])
        ],
    }


def suggest_model(
    data: Any,
    x: str | None = None,
    y: str | None = None,
    *,
    name: str = "data",
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    verbose: bool = True,
) -> Suggestion:
    """Suggest a model to fit, from what the data measures.

    ``name`` is what the dataset is called in your session, because the code
    that comes back refers to it — ``data.fit(a*exp(b*x))`` — and runs in your
    own namespace rather than in restricted execution, which has no dataset of
    yours to reach.

    Examples
    --------
    >>> from mathslate import dataset                          # doctest: +SKIP
    >>> from mathslate.ai import suggest_model                 # doctest: +SKIP
    >>> data = dataset("readings.csv")                         # doctest: +SKIP
    >>> suggest_model(data).show()                             # doctest: +SKIP
    a, b = symbols('a b')
    data.fit(a*exp(b*x))
    """
    if not name.isidentifier():
        raise UnsupportedInputError(
            f"name must be the variable holding the dataset; got {name!r}."
        )
    evidence = fit_evidence(data, x, y)
    import json

    message = (
        f"A dataset held in the variable `{name}`. MathSlate measured it:\n\n"
        f"{json.dumps(evidence, indent=2)}"
    )
    reply, chosen, model_name = _complete(
        _model_prompt(name), message, provider, model, api_key
    )
    code, commentary = _split(reply)
    suggestion = Suggestion(
        code=code,
        question=f"which model fits {name}?",
        provider=chosen.name,
        model=model_name,
        commentary=commentary,
        raw=reply,
    )
    if verbose:
        if commentary:
            safe_print(commentary)
        safe_print(code)
    return suggestion


def _model_prompt(name: str) -> str:
    """What the model is told when choosing a family rather than computing one."""
    return (
        "You choose which curve to fit to a set of measurements, for someone "
        "who is still learning.\n\n"
        f"{contract_reference()}\n"
        "The numbers you are given were measured by MathSlate and are correct. "
        '"straightness" holds |r| after each transform: the one closest to '
        "1.000 names the family the data actually follows. Check its "
        '"rows_used" against "of" before believing it — a log transform drops '
        "every non-positive value, so a high correlation over a fraction of the "
        "rows describes that fraction, not the data, and the family is not "
        "supported. A null means the transform left too little to measure.\n\n"
        "Rules for your answer:\n"
        "- One or two short sentences naming the family and the straightness "
        "figure that says so.\n"
        f"- Then ONE ```python fenced block calling `{name}.fit(...)` with the "
        "model written as a SymPy expression. Declare its parameters first, "
        "e.g. `a, b = symbols('a b')`. `x` already exists.\n"
        "- The model must be linear in x only if the data is. Prefer the "
        "simplest form the figures support; do not add a parameter the data "
        "cannot determine.\n"
        "- If two families are close, fit the simpler one and say in your "
        "sentence what else to try.\n"
        "- Do not state an r-squared or a fitted parameter value: you have not "
        f"run the fit. `{name}.fit(...)` returns those."
    )
