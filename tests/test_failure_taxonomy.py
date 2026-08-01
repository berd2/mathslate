"""The exception tuples in ``core/_failure.py`` must match what SymPy raises.

Those tuples decide when MathSlate degrades gracefully and when it lets an
error through. They were derived by measurement rather than intuition, and this
module re-runs a representative slice of that measurement so a SymPy upgrade
that starts raising something new fails here — loudly, in one place — instead of
crashing a learner mid-plot.

It also pins the two ends of the line being drawn:

* narrower is wrong — the four types a reviewer naturally reaches for miss five
  of the nine that actually occur;
* broader is wrong — ``except Exception`` swallows ``MemoryError``, which is how
  "out of memory inside solveset" came to be reported as "no roots".
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest
import sympy as sp

from mathslate.core._failure import EVALUATION_FAILURE, SYMBOLIC_FAILURE

x = sp.Symbol("x", real=True)

#: One case per exception type observed, with the operation that raises it.
#: Kept small deliberately: this is a canary, not the original survey.
SYMBOLIC_CASES: tuple[tuple[str, object], ...] = (
    ("continuous_domain of floor", lambda: sp.calculus.util.continuous_domain(
        sp.floor(x), x, sp.S.Reals)),
    ("limit of nested gamma", lambda: sp.limit(sp.gamma(sp.gamma(x)), x, 0, "+")),
    ("solveset of a Matrix", lambda: sp.solveset(sp.Matrix([[x, 1], [1, x]]), x)),
    ("simplify of a Boolean", lambda: sp.simplify(sp.Eq(x, 2) - sp.Eq(x, -2))),
    ("limit of a relational", lambda: sp.limit(sp.Eq(x, 2), x, 0, "+")),
    ("lambdify of DiracDelta", lambda: sp.lambdify(x, sp.DiracDelta(x), "numpy")(0.5)),
    ("lambdify of zoo", lambda: sp.lambdify(x, sp.zoo * x, "numpy")(0.5)),
    ("lambdify of an Integral", lambda: sp.lambdify(
        x, sp.Integral(sp.exp(-x**2), x), "numpy")(0.5)),
    ("periodicity of an infinite Sum", lambda: sp.periodicity(
        sp.Sum(x**sp.Symbol("n"), (sp.Symbol("n"), 1, sp.oo)), x)),
    ("sympify of nonsense", lambda: sp.sympify("2 * ((( +")),
)

EVALUATION_CASES: tuple[tuple[str, object], ...] = (
    ("math.log of a negative", lambda: sp.lambdify(x, sp.log(x), "math")(-2.0)),
    ("reciprocal of exact zero", lambda: sp.lambdify(x, 1 / x, "math")(0.0)),
    ("overflow in exp", lambda: sp.lambdify(x, sp.exp(x**5), "math")(1e308)),
    ("numpy has no gamma", lambda: float(
        sp.lambdify(x, sp.gamma(x), "numpy")(np.array([-2.0]))[0])),
    ("numpy has no zeta", lambda: sp.lambdify(x, sp.zeta(x, 3), "numpy")(np.array([0.5]))),
)


#: Frames allowed above the current depth while a case is measured.
#:
#: One case above recurses until it runs out of stack, on purpose — that is how
#: `RecursionError` gets into the observed set. Python raises that error from a
#: *counter*, but the C stack underneath can overflow first, and a C stack
#: overflow is fatal: the interpreter dies with no traceback and no test report.
#: It did exactly that on Windows / Python 3.10 CI, where the crash was silent
#: (3.11 added C-level recursion guards, which is why 3.12 survived the same
#: test). Capping the counter close to the current depth makes Python's catchable
#: error arrive first on every platform, which is all this survey needs.
_HEADROOM: int = 120


def _current_depth() -> int:
    frame = sys._getframe()
    depth = 0
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


def _raised(operation) -> BaseException | None:  # type: ignore[no-untyped-def]
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(_current_depth() + _HEADROOM)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                operation()
            except BaseException as error:  # noqa: BLE001 - measuring, not handling
                return error
        return None
    finally:
        sys.setrecursionlimit(previous)


class TestTheSymbolicSetIsComplete:
    @pytest.mark.parametrize("label,operation", SYMBOLIC_CASES, ids=lambda v: v)
    def test_a_known_sympy_refusal_is_caught(self, label: str, operation) -> None:  # type: ignore[no-untyped-def]
        error = _raised(operation)
        assert error is not None, (
            f"{label} no longer raises — SymPy has grown a capability, and the "
            "comment in core/_failure.py citing this case is now stale"
        )
        assert isinstance(error, SYMBOLIC_FAILURE), (
            f"{label} raises {type(error).__name__}, which SYMBOLIC_FAILURE does "
            "not catch: analyze() would crash instead of falling back"
        )

    def test_every_documented_type_is_reachable(self) -> None:
        """Each entry in the tuple must be there for a reason still observable."""
        seen = {type(_raised(op)) for _, op in SYMBOLIC_CASES}
        for documented in (
            NotImplementedError, TypeError, ValueError,
            AttributeError, NameError, KeyError, RecursionError,
        ):
            assert any(issubclass(t, documented) for t in seen), (
                f"no case in this module still produces {documented.__name__}"
            )


class TestNarrowingWouldRegress:
    """The reviewer's shortlist is not enough, and this shows by how much."""

    SHORTLIST = (NotImplementedError, ValueError, TypeError, sp.SympifyError)

    def test_the_obvious_four_types_miss_real_failures(self) -> None:
        missed = [
            f"{label} ({type(error).__name__})"
            for label, op in SYMBOLIC_CASES
            if (error := _raised(op)) is not None
            and not isinstance(error, self.SHORTLIST)
        ]
        assert missed, (
            "if the shortlist now covers everything, SYMBOLIC_FAILURE can be "
            "narrowed to it — re-run the survey before doing so"
        )
        for case in missed:
            assert isinstance(_raised(dict(SYMBOLIC_CASES)[case.split(" (")[0]]), SYMBOLIC_FAILURE)


class TestBrokenIsNotTheSameAsUnsolvable:
    """What the named set buys over `except Exception`."""

    @pytest.mark.parametrize("escalation", [MemoryError, OSError, KeyboardInterrupt])
    def test_a_machine_problem_is_not_caught(self, escalation: type[BaseException]) -> None:
        assert not issubclass(escalation, SYMBOLIC_FAILURE)
        assert not issubclass(escalation, EVALUATION_FAILURE)

    def test_out_of_memory_reaches_the_caller(self) -> None:
        """It used to arrive at the learner as "this function has no roots"."""
        from mathslate.core import analysis

        def explode(*args: object, **kwargs: object) -> object:
            raise MemoryError("solveset needed more than this machine has")

        original = sp.solveset
        sp.solveset = explode  # type: ignore[assignment]
        try:
            with pytest.raises(MemoryError):
                analysis.analyze_expression(sp.sin(x), x, (-1.0, 1.0))
        finally:
            sp.solveset = original  # type: ignore[assignment]


class TestTheEvaluationSetIsComplete:
    @pytest.mark.parametrize("label,operation", EVALUATION_CASES, ids=lambda v: v)
    def test_an_undefined_point_is_caught(self, label: str, operation) -> None:  # type: ignore[no-untyped-def]
        error = _raised(operation)
        assert error is not None, f"{label} no longer raises"
        assert isinstance(error, EVALUATION_FAILURE), (
            f"{label} raises {type(error).__name__}, which EVALUATION_FAILURE "
            "does not catch: sampling would crash on an undefined point"
        )

    def test_it_stays_within_the_symbolic_set(self) -> None:
        """The last evaluation tier is SymPy's own evalf, so it must be a subset."""
        assert set(EVALUATION_FAILURE) <= set(SYMBOLIC_FAILURE)
