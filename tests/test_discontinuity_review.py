"""PRD 4 — "Discontinuity review: all 30 hand-checked cases pass".

Each row was worked out on paper first and then pinned here, so a regression
in the sampler shows up as a wrong *number*, not merely a wrong picture.

``breaks`` is where the drawn line must be cut; ``pieces`` is how many
continuous stretches of the real domain fall inside the default [-10, 10]
window. A case passes only when both match and nothing is drawn across a cut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import sympy as sp

from mathslate import plot, x
from tests.helpers import crossing_points

PI = float(np.pi)


@dataclass(frozen=True)
class Case:
    source: str
    #: exact cut locations, or how many cuts there should be for a step function
    breaks: tuple[float, ...] | int
    pieces: int
    note: str
    #: for step functions: the spacing the cuts must land on
    spacing: float = 1.0


#: 30 hand-checked cases, grouped by the reason they are hard.
REVIEW: tuple[Case, ...] = (
    # --- trigonometric poles ---------------------------------------------
    Case("tan(x)", tuple(k * PI / 2 for k in (-5, -3, -1, 1, 3, 5)), 7, "poles at odd pi/2"),
    Case("cot(x)", tuple(k * PI for k in (-3, -2, -1, 0, 1, 2, 3)), 8, "poles at multiples of pi"),
    Case("sec(x)", tuple(k * PI / 2 for k in (-5, -3, -1, 1, 3, 5)), 7, "same poles as tan"),
    Case("csc(x)", tuple(k * PI for k in (-3, -2, -1, 0, 1, 2, 3)), 8, "same poles as cot"),
    Case("1/sin(x)", tuple(k * PI for k in (-3, -2, -1, 0, 1, 2, 3)), 8, "csc written out"),
    # --- rational poles ---------------------------------------------------
    Case("1/x", (0.0,), 2, "the textbook pole"),
    Case("1/x**2", (0.0,), 2, "a double pole, same sign both sides"),
    Case("1/(x - 3)", (3.0,), 2, "an off-centre pole"),
    Case("1/(x**2 - 1)", (-1.0, 1.0), 3, "two poles"),
    Case("1/(x**2 - 4)", (-2.0, 2.0), 3, "two poles, wider"),
    Case("1/((x - 1)*(x + 2))", (-2.0, 1.0), 3, "poles from a factored form"),
    # --- removable and mixed ---------------------------------------------
    Case("sin(x)/x", (0.0,), 2, "removable: the curve must still look whole"),
    Case("tan(x)/x", tuple(sorted((*(k * PI / 2 for k in (-5, -3, -1, 1, 3, 5)), 0.0))), 8,
         "poles plus a removable point"),
    Case("atan(1/x)", (0.0,), 2, "bounded, but genuinely jumps"),
    # --- sign-type jumps --------------------------------------------------
    Case("x/Abs(x)", (0.0,), 2, "the classic -1/+1 jump"),
    Case("Abs(x)/x", (0.0,), 2, "the same jump, written the other way"),
    Case("sign(x)", (0.0,), 1, "no pole, so SymPy sees a full domain"),
    Case("Piecewise((-1, x < 0), (1, True))", (0.0,), 1, "an explicit jump"),
    Case("Piecewise((x, x < 0), (x**2, True))", (), 1, "a piecewise that is continuous"),
    # --- step functions ---------------------------------------------------
    Case("floor(x)", 19, 1, "a jump at every integer"),
    Case("ceiling(x)", 19, 1, "same steps, shifted"),
    Case("x - floor(x)", 19, 1, "the sawtooth"),
    Case("floor(x/2)", 9, 1, "steps every 2", spacing=2.0),
    Case("floor(2*x)", 39, 1, "steps every 1/2", spacing=0.5),
    # --- restricted domains ------------------------------------------------
    Case("sqrt(x)", (), 1, "half the window is not real"),
    Case("sqrt(4 - x**2)", (), 1, "a bounded island"),
    Case("sqrt(x**2 - 1)", (), 2, "two islands with a gap between"),
    Case("log(x)", (0.0,), 1, "domain edge and a pole at the same point"),
    Case("log(Abs(x))", (0.0,), 2, "a pole with real values on both sides"),
    Case("1/sqrt(x)", (0.0,), 1, "domain edge that is also a pole"),
)


def test_the_review_has_thirty_cases() -> None:
    assert len(REVIEW) == 30


@pytest.mark.parametrize("case", REVIEW, ids=lambda c: c.source)
def test_the_cuts_are_where_they_belong(case: Case) -> None:
    sample = _sample(case)
    found = sample.breakpoints
    if isinstance(case.breaks, int):
        assert len(found) == case.breaks, case.note
        multiples = np.asarray(found) / case.spacing
        assert np.allclose(multiples, np.round(multiples), atol=1e-6), (
            f"steps must land on multiples of {case.spacing}"
        )
    else:
        assert len(found) == len(case.breaks), case.note
        assert np.allclose(found, case.breaks, atol=1e-6), case.note


@pytest.mark.parametrize("case", REVIEW, ids=lambda c: c.source)
def test_the_real_domain_is_split_correctly(case: Case) -> None:
    assert len(_sample(case).domain_intervals) == case.pieces, case.note


@pytest.mark.parametrize("case", REVIEW, ids=lambda c: c.source)
def test_nothing_is_drawn_across_a_cut(case: Case) -> None:
    sample = _sample(case)
    assert crossing_points(sample, sample.breakpoints) == [], case.note


@pytest.mark.parametrize("case", REVIEW, ids=lambda c: c.source)
def test_the_curve_is_still_worth_looking_at(case: Case) -> None:
    """Breaking a line is easy; breaking it and keeping a readable plot is not."""
    sample = _sample(case)
    assert sample.finite_count >= 20, case.note


def _sample(case: Case) -> object:
    expr = sp.sympify(case.source, locals={"x": x})
    return plot(expr, verbose=False).plan.series[0].sample
