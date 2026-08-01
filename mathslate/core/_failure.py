"""Which third-party exceptions mean "could not", and which mean "broken".

MathSlate leans on SymPy and NumPy for everything hard, and both of them fail
routinely and legitimately: ``continuous_domain(floor(x))`` is not implemented,
``log(-2.0)`` has no real value. Degrading gracefully at those points is the
design, not a workaround — PRD 5.3 and 5.6 both depend on it.

The catch has to be broad, then, but ``except Exception`` is too broad in one
specific direction: it also swallows ``MemoryError`` and ``OSError``. A machine
running out of memory inside ``solveset`` was being reported to the learner as
"this function has no roots" — a lie about mathematics caused by a fact about
the computer. These two tuples draw that line, and they are *measured* rather
than guessed.

Method: every operation the package actually performs was run over three dozen
pathological inputs (``floor``, ``zeta``, ``DiracDelta``, ``Integral``,
``Matrix``, ``Eq``, ``zoo``, nested ``gamma``, ``Sum`` to infinity, …) and the
raised types collected. See ``tests/test_failure_taxonomy.py``, which re-runs a
representative slice of that survey so the tuples cannot silently drift out of
date as SymPy changes.

A tempting narrower list — ``NotImplementedError``, ``ValueError``,
``TypeError``, ``SympifyError`` — misses five of the nine types observed, so
narrowing to it would turn today's graceful fallback into a crash on, among
others, ``lambdify(DiracDelta(x))``.
"""

from __future__ import annotations

from typing import Final

__all__ = ["SYMBOLIC_FAILURE", "EVALUATION_FAILURE"]


#: SymPy declining to do a piece of algebra.
#:
#: Observed from ``diff``, ``solveset``, ``simplify``, ``periodicity``,
#: ``limit``, ``lambdify``, ``singularities``, ``continuous_domain``, ``N``,
#: ``float`` and ``sympify``:
#:
#: * ``NotImplementedError`` — much the most common: ``continuous_domain`` on
#:   ``floor(x)``, ``limit`` on nested ``gamma``. ``lambdify(Integral(...))``
#:   raises ``PrintMethodNotImplementedError``, a subclass.
#: * ``TypeError`` / ``ValueError`` — ``solveset`` on a ``Matrix``, ``simplify``
#:   on a ``Boolean``. ``SympifyError`` is a ``ValueError`` subclass.
#: * ``AttributeError`` — ``limit(Eq(x, 2), ...)`` reaches for expression
#:   attributes that a relational has not got.
#: * ``NameError`` — ``lambdify(DiracDelta(x))`` prints a name NumPy lacks.
#: * ``KeyError`` — ``lambdify(zoo*x)``: no printer entry for complex infinity.
#: * ``SyntaxError`` — ``lambdify`` emitting source Python cannot parse.
#: * ``RecursionError`` — ``periodicity(Sum(...))`` recurses until the stack ends.
#: * ``ArithmeticError`` — an exact substitution dividing by an exact zero.
SYMBOLIC_FAILURE: Final[tuple[type[Exception], ...]] = (
    ArithmeticError,
    AttributeError,
    KeyError,
    NameError,
    NotImplementedError,
    RecursionError,
    SyntaxError,
    TypeError,
    ValueError,
)


#: No number exists at this point, or none that this backend can produce.
#:
#: Observed from lambdified NumPy and ``math`` callables: ``ValueError``
#: (``math.log(-2)``), ``TypeError`` (NumPy handed a ``gamma`` it has not got),
#: ``ZeroDivisionError`` and ``OverflowError`` (both ``ArithmeticError``),
#: ``NameError`` and ``KeyError`` (a printed name or printer entry missing).
#:
#: This is currently a subset of :data:`SYMBOLIC_FAILURE` — unsurprising, since
#: the last-resort evaluation tier is SymPy's own ``evalf``. It stays a separate
#: name because the two say different things at the call site: one means "no
#: closed form", the other "not defined here", and a future divergence in
#: either should not have to argue with the other's call sites.
EVALUATION_FAILURE: Final[tuple[type[Exception], ...]] = (
    ArithmeticError,
    AttributeError,
    KeyError,
    NameError,
    TypeError,
    ValueError,
)
