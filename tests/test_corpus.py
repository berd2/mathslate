"""PRD acceptance criterion 3 — >=95% pass rate on the 200-function corpus."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import sympy as sp

from mathslate import plot, x
from tests.corpus import CORPUS
from tests.helpers import crossing_points

MIN_PASS_RATE: float = 0.95
MIN_FINITE_POINTS: int = 20


@dataclass(frozen=True)
class Outcome:
    source: str
    passed: bool
    reason: str = ""


def _check(source: str) -> Outcome:
    try:
        expr = sp.sympify(source, locals={"x": x})
        result = plot(expr, verbose=False)
    except Exception as error:  # noqa: BLE001 - any failure is a corpus failure
        return Outcome(source, False, f"{type(error).__name__}: {error}")

    sample = result.plan.series[0].sample
    if sample.finite_count < MIN_FINITE_POINTS:
        return Outcome(source, False, f"only {sample.finite_count} finite points")

    crossed = crossing_points(sample, sample.breakpoints)
    if crossed:
        return Outcome(source, False, f"line drawn across discontinuity at {crossed[:3]}")

    if _traverses_the_window(result):
        return Outcome(source, False, "a single segment crosses the whole visible window")

    return Outcome(source, True)


def _traverses_the_window(result: object) -> bool:
    """True if one drawn segment jumps from below the view to above it.

    That is the signature of a false connecting line: a real curve leaving the
    window through the top and re-entering from the bottom must be broken.
    """
    plan = result.plan  # type: ignore[attr-defined]
    sample = plan.series[0].sample
    span = plan.y_range
    if span is None:
        finite = sample.y[np.isfinite(sample.y)]
        if finite.size == 0:
            return False
        span = (float(finite.min()), float(finite.max()))
    low, high = span
    if high <= low:
        return False
    drawn = (
        np.isfinite(sample.x[:-1])
        & np.isfinite(sample.x[1:])
        & np.isfinite(sample.y[:-1])
        & np.isfinite(sample.y[1:])
    )
    left, right = sample.y[:-1], sample.y[1:]
    jumps = ((left < low) & (right > high)) | ((left > high) & (right < low))
    return bool((drawn & jumps).any())


@pytest.fixture(scope="module")
def outcomes() -> list[Outcome]:
    return [_check(source) for source in CORPUS]


def test_corpus_has_exactly_two_hundred_functions() -> None:
    assert len(CORPUS) == 200
    assert len(set(CORPUS)) == 200


def test_pass_rate_meets_the_target(outcomes: list[Outcome]) -> None:
    failures = [o for o in outcomes if not o.passed]
    rate = 1.0 - len(failures) / len(outcomes)
    detail = "\n".join(f"  {o.source}: {o.reason}" for o in failures)
    assert rate >= MIN_PASS_RATE, f"pass rate {rate:.1%} < {MIN_PASS_RATE:.0%}\n{detail}"


def test_no_curve_is_drawn_across_a_pole(outcomes: list[Outcome]) -> None:
    """The differentiating property, checked over the whole corpus."""
    offenders = [o.source for o in outcomes if not o.passed and "discontinuity" in o.reason]
    assert offenders == []
