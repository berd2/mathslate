"""A symbolic step that will not finish in time must not take the session with it.

Some `solveset` calls do not fail and do not hang forever — they simply take far
longer than a learner will wait, and to that learner a long silence is
indistinguishable from a broken library.

`core/_budget.py` gives each attempt a deadline and falls back to the numeric
path already used when SymPy *declines*. Three properties matter and each has a
class here:

* the budget never fires on ordinary work, so exact answers stay exact;
* when it does fire the result is still correct, marked approximate, and says why;
* an expired attempt is **stopped**, not merely abandoned, so it cannot degrade
  the calls that follow it.

**On test subjects.** Forcing a real timeout deterministically is the whole
difficulty. A genuinely slow SymPy call is slow *because* it does something
pathological — a high-degree polynomial recurses deep enough in `decompogen` to
overflow a worker thread's small stack on Windows, which is fatal, not a
catchable error. So this module does not feed SymPy stack bombs. The *mechanism*
is tested on a pure-Python busy loop (`_spin`), where an injected exception lands
in one bytecode on every platform; the *integration* — that `analyze()` degrades
— is tested with a tiny budget on ordinary expressions, which times out on the
first step without any deep recursion. Both are deterministic and
platform-independent.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator

import pytest
import sympy as sp

import mathslate as ms
from mathslate import x
from mathslate.core import _budget
from mathslate.core._budget import (
    DEFAULT_BUDGET,
    SymbolicTimeout,
    get_budget,
    set_budget,
    within_budget,
)

#: Small enough that the first symbolic step always exceeds it, ordinary enough
#: that reaching it needs no pathological input. Thread start-up alone outlasts
#: it, so the timeout is certain without depending on how fast SymPy runs.
_TINY: float = 0.0005

#: A real solve that exercises CRootOf — degree 5, so it recurses shallowly and
#: cannot overflow a stack — for the tests that must interrupt genuine SymPy.
_REAL_SOLVE = (x**5 - x - 1, x, sp.Reals)

# Interrupting a thread can deliver the exception inside a weakref-clearing
# callback, which runs during garbage collection with no frame of ours on the
# stack. CPython prints "Exception ignored in ..." and discards it; nothing is
# corrupted. Scoped to this module and to that one exception type on purpose —
# a blanket filter here would hide the genuine warnings the suite relies on.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Exception ignored in:pytest.PytestUnraisableExceptionWarning"
)


@pytest.fixture(autouse=True)
def _restore_budget() -> Iterator[None]:
    """The budget is module state, so a test that changes it must put it back."""
    before = get_budget()
    try:
        yield
    finally:
        set_budget(before)


def _settle() -> None:
    """Wait for any worker to finish, so one test cannot bleed into the next."""
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if not any(t.name == "mathslate-symbolic" for t in threading.enumerate()):
            return
        time.sleep(0.02)
    pytest.fail("a symbolic worker outlived its test")


def _spin() -> None:
    """A pure-Python busy loop — the deterministic worker for mechanism tests.

    Every iteration is a bytecode boundary, so an injected exception lands at
    once on every platform. `solveset` cannot promise that (it sits in C for
    long stretches, and recurses deep in Python), so tests of the *mechanism* —
    that an expired worker is stopped rather than abandoned — use this.
    """
    while True:
        pass


class TestTheMechanism:
    def test_a_quick_call_is_untouched(self) -> None:
        assert within_budget(lambda: 2 + 2) == 4

    def test_webassembly_runs_inline_when_worker_threads_are_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pyodide exposes ``threading`` but cannot start a Thread."""
        monkeypatch.setattr(_budget.sys, "platform", "emscripten")
        assert within_budget(lambda: "browser-safe", seconds=0.001) == "browser-safe"

    def test_a_slow_call_raises(self) -> None:
        with pytest.raises(SymbolicTimeout, match="budget"):
            within_budget(_spin, seconds=0.2)
        _settle()

    def test_the_message_names_the_operation(self) -> None:
        with pytest.raises(SymbolicTimeout, match="_spin"):
            within_budget(_spin, seconds=0.2)
        _settle()

    def test_an_exception_passes_through_unchanged(self) -> None:
        """Wrapping a call must not change what a failing call does."""
        with pytest.raises(ZeroDivisionError):
            within_budget(lambda: 1 / 0)

    def test_the_return_value_passes_through_unchanged(self) -> None:
        assert within_budget(sp.solveset, x**2 - 4, x, sp.Reals) == sp.FiniteSet(-2, 2)

    def test_none_disables_it(self) -> None:
        set_budget(None)
        assert get_budget() is None
        assert within_budget(lambda: "ran inline", seconds=-1.0) == "ran inline"

    @pytest.mark.parametrize("bad", [0, -1, -0.5])
    def test_a_non_positive_budget_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="positive or None"):
            set_budget(bad)

    def test_the_default_is_documented_and_used(self) -> None:
        assert get_budget() == DEFAULT_BUDGET == 5.0

    def test_a_nested_budget_uses_the_outer_worker(self) -> None:
        """A helper may be budgeted even when its caller already is."""
        assert within_budget(
            lambda: within_budget(lambda: "nested", seconds=0.2),
            seconds=1.0,
        ) == "nested"

    def test_concurrent_callers_do_not_cancel_each_other(self) -> None:
        """The second caller queues; it must not unwind the first caller."""
        entered = threading.Event()
        release = threading.Event()
        results: list[str] = []
        errors: list[BaseException] = []

        def first_operation() -> str:
            entered.set()
            assert release.wait(2.0)
            return "first"

        def first_call() -> None:
            try:
                results.append(within_budget(first_operation, seconds=1.0))
            except BaseException as error:  # noqa: BLE001 - asserted below
                errors.append(error)

        caller = threading.Thread(target=first_call)
        caller.start()
        assert entered.wait(1.0)
        timer = threading.Timer(0.1, release.set)
        timer.start()
        try:
            assert within_budget(lambda: "second", seconds=1.0) == "second"
        finally:
            release.set()
            caller.join(2.0)
            timer.cancel()

        assert caller.is_alive() is False
        assert errors == []
        assert results == ["first"]


class TestAnExpiredAttemptIsStopped:
    """Not just abandoned — `PyThreadState_SetAsyncExc` asks it to unwind."""

    def test_an_expired_worker_is_stopped_not_abandoned(self) -> None:
        """The mechanism, tested deterministically on a pure-Python worker.

        `_spin` catches the injected exception at its next bytecode, immediate on
        every platform — so this proves injection *stops* a running worker
        without depending on how long a `solveset` happens to sit in C.
        """
        with pytest.raises(SymbolicTimeout):
            within_budget(_spin, seconds=0.2)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if not any(t.name == "mathslate-symbolic" for t in threading.enumerate()):
                return
            time.sleep(0.02)
        pytest.fail("the expired worker was abandoned rather than stopped")

    def test_no_worker_thread_is_left_behind(self) -> None:
        for _ in range(3):
            with pytest.raises(SymbolicTimeout):
                within_budget(_spin, seconds=0.2)
            _settle()
        assert [t for t in threading.enumerate() if t.name == "mathslate-symbolic"] == []

    def test_the_next_call_is_still_given_its_full_budget(self) -> None:
        """The regression: a stuck worker used to make later calls decline."""
        with pytest.raises(SymbolicTimeout):
            within_budget(_spin, seconds=0.2)
        _settle()
        assert within_budget(sp.solveset, x**2 - 4, x, sp.Reals) == sp.FiniteSet(-2, 2)

    def test_an_overapplied_injection_is_immediately_rolled_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[type[BaseException] | None] = []

        def fake(ident: int, exception: type[BaseException] | None) -> int:
            assert ident == 123
            calls.append(exception)
            return 2 if exception is SymbolicTimeout else 0

        monkeypatch.setattr(_budget, "_set_async_exception", fake)
        assert _budget._inject_timeout(123) is False
        assert calls == [SymbolicTimeout, None]

    def test_exactly_one_injected_thread_is_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _budget,
            "_set_async_exception",
            lambda ident, exception: 1,
        )
        assert _budget._inject_timeout(123) is True


class TestDeepRecursionDoesNotCrashTheWorker:
    """A worker thread's default stack is small — 1 MB on Windows — and SymPy's
    solvers can recurse deep. There the C stack can overflow before Python's
    catchable recursion limit, which is fatal: it takes the whole interpreter
    down. The worker runs on a large stack now, so deep recursion raises a
    catchable error the budget handles like any other refusal.
    """

    def test_unbounded_recursion_raises_rather_than_aborts(self) -> None:
        """Pure-Python recursion hits Python's limit — catchable — not a segfault."""
        def bottomless(depth: int = 0) -> int:
            return bottomless(depth + 1)

        with pytest.raises((RecursionError, SymbolicTimeout)):
            within_budget(bottomless)
        _settle()

    def test_the_worker_stack_is_larger_than_a_default_thread(self) -> None:
        from mathslate.core._budget import _WORKER_STACK

        # 1 MB is the Windows default that overflowed; clear it with room to spare.
        assert _WORKER_STACK >= 16 * 1024 * 1024

    def test_the_global_stack_size_is_left_as_it_was(self) -> None:
        """Enlarging every later thread the user's program starts would be rude."""
        before = threading.stack_size()
        with pytest.raises(SymbolicTimeout):
            within_budget(_spin, seconds=0.2)
        _settle()
        assert threading.stack_size() == before


class TestOrdinaryWorkIsUnaffected:
    """If the budget changed any everyday answer it would be a bug, not a guard."""

    @pytest.mark.parametrize(
        "expr,roots",
        [
            (sp.sin(x), "pi"),
            (x**2 - 4, "-2, 2"),
            (x**3 - x, "-1, 0, 1"),
        ],
        ids=["sin", "quadratic", "cubic"],
    )
    def test_the_exact_answers_are_still_exact(self, expr: sp.Expr, roots: str) -> None:
        report = ms.analyze(expr)
        assert report.roots.approximate is False
        assert roots in report.roots.describe()

    def test_nothing_ordinary_trips_the_budget(self) -> None:
        for expr in (sp.sin(x), sp.tan(x), 1 / x, sp.exp(x), sp.floor(x), sp.sqrt(x)):
            notes = ms.analyze(expr).notes
            assert not any("budget" in n for n in notes), f"{expr} hit the budget"

    def test_a_plot_of_a_hard_function_stays_quick(self) -> None:
        """The mpmath tier, not the budget, is what makes this true."""
        start = time.perf_counter()
        ms.plot(sp.besselj(2, x) - 1 / x, verbose=False)
        assert time.perf_counter() - start < 15.0


class TestWhenItFires:
    """Forced with a tiny budget on an ordinary expression — no stack bomb."""

    def test_the_answer_is_still_right_just_approximate(self) -> None:
        set_budget(_TINY)
        report = ms.analyze(x**5 - x - 1)
        assert any("budget" in n for n in report.notes)
        assert report.approximate is True

    def test_the_note_says_which_step_and_how_to_change_it(self) -> None:
        set_budget(_TINY)
        note = next(n for n in ms.analyze(x**5 - x - 1).notes if "budget" in n)
        assert "budget" in note
        assert "set_symbolic_budget" in note

    def test_it_finishes_quickly_when_every_step_times_out(self) -> None:
        """A dozen symbolic steps must not each pay the budget separately — the
        first timeout disables the rest for the call."""
        set_budget(_TINY)
        start = time.perf_counter()
        ms.analyze(x**5 - x - 1)
        assert time.perf_counter() - start < 12.0

    def test_a_later_call_is_exact_again(self) -> None:
        """The session-poisoning regression, at the level a user would meet it."""
        set_budget(_TINY)
        assert ms.analyze(x**5 - x - 1).approximate is True
        set_budget(DEFAULT_BUDGET)
        assert ms.analyze(sp.sin(x)).roots.approximate is False

    def test_disabling_the_budget_brings_the_symbolic_answer_back(self) -> None:
        set_budget(None)
        assert get_budget() is None
        assert ms.analyze(x**2 - 4).roots.approximate is False


class TestTheKnobIsReachable:
    def test_it_is_exported_from_core_under_a_clear_name(self) -> None:
        from mathslate import core

        assert core.get_symbolic_budget() == get_budget()
        assert "set_symbolic_budget" in core.__all__

    def test_the_note_names_a_path_that_actually_works(self) -> None:
        """A suggestion the reader cannot follow is worse than none."""
        set_budget(_TINY)
        note = next(n for n in ms.analyze(x**5 - x - 1).notes if "budget" in n)
        assert "mathslate.core.set_symbolic_budget" in note
        set_budget(DEFAULT_BUDGET)
        import mathslate.core as core

        core.set_symbolic_budget(30)
        assert core.get_symbolic_budget() == 30.0

    def test_it_is_not_in_the_top_level_api(self) -> None:
        """PRD 4's surface budget is full; this is a knob, not a feature."""
        assert not hasattr(ms, "set_symbolic_budget")


class TestSympyIsLeftInAUsableState:
    """Injection lands at an arbitrary bytecode boundary — including mid-write.

    SymPy's memo caches are module-level state shared with the calling thread.
    Interrupting `solveset` inside `rootoftools` was observed to leave
    `_complexes_cache` holding a key whose value never arrived, and the next
    unrelated lookup raised `KeyError` from deep inside SymPy. So the caches are
    cleared after any interruption that actually landed.
    """

    def _flush_deferred_repair(self) -> None:
        """A repair owed by a not-yet-stopped worker runs on the next call."""
        within_budget(lambda: None)

    def test_root_finding_still_works_after_an_interruption(self) -> None:
        with pytest.raises(SymbolicTimeout):
            within_budget(sp.solveset, *_REAL_SOLVE, seconds=_TINY)
        _settle()
        self._flush_deferred_repair()
        # The exact operation whose cache may have been mid-write when stopped.
        assert sp.CRootOf(x**5 - x - 1, 0).is_real
        assert sp.solveset(x**2 - 2, x, sp.Reals) == sp.FiniteSet(-sp.sqrt(2), sp.sqrt(2))

    def test_repeated_interruptions_leave_it_usable(self) -> None:
        for _ in range(3):
            with pytest.raises(SymbolicTimeout):
                within_budget(sp.solveset, *_REAL_SOLVE, seconds=_TINY)
            _settle()
        self._flush_deferred_repair()
        assert ms.analyze(sp.sin(x)).roots.approximate is False

    def test_the_repair_runs_when_the_injection_lands(self) -> None:
        """When a worker is stopped, the caches are repaired — checked on a
        deterministic worker so the injection is certain to land."""
        from mathslate.core import _budget as budget_module

        calls: list[int] = []
        original = budget_module._repair_sympy_caches
        budget_module._repair_sympy_caches = lambda: calls.append(1)
        try:
            with pytest.raises(SymbolicTimeout):
                within_budget(_spin, seconds=0.2)
            _settle()
            within_budget(lambda: None)  # collect a deferred repair, if any
        finally:
            budget_module._repair_sympy_caches = original
        assert calls, "a stopped worker left the caches unrepaired"

    def test_a_failing_repair_does_not_replace_the_timeout(self) -> None:
        """The timeout is the news; a problem tidying up after it is not."""
        from mathslate.core import _budget as budget_module

        original = budget_module._repair_sympy_caches

        def explode() -> None:
            raise RuntimeError("cache repair itself failed")

        budget_module._repair_sympy_caches = explode
        try:
            with pytest.raises(SymbolicTimeout):
                within_budget(_spin, seconds=0.2)
        finally:
            budget_module._repair_sympy_caches = original
        _settle()


class TestSymbolicTimeoutIsItsOwnKind:
    def test_it_is_not_caught_by_the_failure_tuples(self) -> None:
        """Otherwise "declined" and "ran out of time" could not be told apart."""
        from mathslate.core._failure import EVALUATION_FAILURE, SYMBOLIC_FAILURE

        assert not issubclass(SymbolicTimeout, SYMBOLIC_FAILURE)
        assert not issubclass(SymbolicTimeout, EVALUATION_FAILURE)

    def test_it_is_not_an_oserror(self) -> None:
        """`TimeoutError` is one, and OSError is what the tuples let through."""
        assert not issubclass(SymbolicTimeout, OSError)
        assert SymbolicTimeout is not TimeoutError

    def test_the_module_state_starts_clean(self) -> None:
        assert _budget._WORKER_RUNNING.is_set() is False
