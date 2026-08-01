"""A wall-clock budget for one symbolic attempt.

SymPy has no notion of "give up". `solveset` on a degree-40 polynomial does not
fail and does not hang forever — it takes about eighty seconds, measured, and
returns. To a learner that is indistinguishable from a broken library, and there
is nothing on screen to say otherwise.

Worth being precise about what this module is *not* for, because the first
investigation got it wrong. `analyze(besselj(2, x) - 1/x)` took 27 seconds, and
the obvious reading was "a symbolic step is slow". Profiling said otherwise: 66
of the 68 seconds were in `NumericFunction._exact`, evaluating point by point
through `subs().evalf()` because NumPy has no `besselj`. That was a *numeric*
cost and no deadline would have caught it; the fix was an mpmath tier in
`sampling.py`, which took the same call to 1.7 s. This module is for the other
thing — a single symbolic call that genuinely will not finish soon.

The module gives every symbolic attempt a deadline. When it expires the attempt
is abandoned and the caller falls back to the numeric path it already has for
the case where SymPy *declines* — so a timeout produces the same
`approximate=True` result, with a note saying which of the two happened. That is
not a new behaviour, it is the third branch of a rule the package already
follows: exact where SymPy can, approximate where it cannot, and say which.

**Why a thread and not a signal.** `signal.SIGALRM` is Unix-only and only works
on the main thread, which rules out Jupyter's and marimo's execution threads.
A worker thread works everywhere. Mathics3 reached the same conclusion for
`TimeConstrained` and maintains a fork of `stopit` for it; this is the same
design without the dependency, since the package needs one deadline in one
place rather than a general nesting-capable framework.

**Stopping the worker, not just abandoning it.** `Thread` has no `kill`, and
the naive version of this module simply walked away from an expired attempt.
That is what Mathics3 documents as its own limitation — the evaluation "continues
consuming resources" — and here it was worse than untidy: while an abandoned
worker was still inside SymPy, later attempts had to be refused (running SymPy
from two threads at once is a real hazard), so **one pathological expression
degraded every later call in the session** for as long as it took to finish. A
degree-40 polynomial poisoned `analyze(sin(x))` for eighty seconds.

So the worker is asked to unwind, via CPython's `PyThreadState_SetAsyncExc` —
the mechanism `stopit` is built on. It raises the exception in the target thread
at its next bytecode boundary, and SymPy's `solveset` unwinds from it in **28
milliseconds**, measured. The refusal window is therefore normally not a window
at all.

One cosmetic consequence is unavoidable. The injected exception can be delivered
inside a weakref-clearing callback, which runs during garbage collection where no
frame of ours exists to catch it; CPython prints "Exception ignored in ..." and
discards it. Nothing is corrupted — the object simply stays in a `WeakSet` — but
the message looks alarming and cannot be suppressed from here.

Injection is not a guarantee: a thread blocked inside a C extension reaches no
bytecode boundary and will not notice. When that happens the worker is left
alone and later attempts are refused until it exits. Running a second SymPy
operation beside a worker whose state is unknown would risk its process-global
caches. A future isolated-process backend can make that refusal unnecessary:
unlike a thread, a process can be terminated safely even while native code runs.
"""

from __future__ import annotations

import ctypes
import threading
from typing import Any, Callable, Final, TypeVar

__all__ = ["SymbolicTimeout", "DEFAULT_BUDGET", "within_budget", "set_budget", "get_budget"]

T = TypeVar("T")


class SymbolicTimeout(Exception):
    """A symbolic attempt ran out of its wall-clock budget.

    Deliberately not a subclass of anything in
    :data:`mathslate.core._failure.SYMBOLIC_FAILURE`: the call sites treat
    "SymPy declined" and "SymPy ran out of time" the same way but report them
    differently, and a shared base class would make that distinction
    unavailable. It is also deliberately not the built-in ``TimeoutError``,
    which is an ``OSError`` — the one family the failure tuples exist to keep
    letting through.
    """


#: Seconds one symbolic attempt may take before the numeric path takes over.
#:
#: Five is chosen against measurement, not taste. Nothing in the corpus comes
#: near it — the slowest `plot()` is 1.7 s and the slowest `analyze()` 3.9 s once
#: the mpmath tier is in place — so on ordinary work the budget never fires at
#: all. It fires on `solveset` of a degree-40 polynomial, which takes eighty
#: seconds, and brings that `analyze()` in at 5.7 s. A learner waiting five
#: seconds is waiting; a learner waiting eighty has concluded it is broken.
DEFAULT_BUDGET: Final[float] = 5.0

#: Seconds allowed for an expired worker to unwind after being asked to. SymPy's
#: `solveset` on a degree-40 polynomial takes 28 ms, so half a second is generous
#: without making a stuck worker cost the caller real time.
_UNWIND_GRACE: Final[float] = 0.5

#: Stack the worker thread runs on. SymPy's solvers recurse a few hundred frames
#: deep (`decompogen` on a high-degree polynomial), and a thread's default stack
#: is far smaller than the main thread's — 1 MB on Windows. There the C stack
#: overflows *before* Python's own recursion counter reaches its (catchable)
#: limit, and a C stack overflow is fatal: it takes the whole interpreter down,
#: not just the one call. Reserving 64 MB — address space, not committed memory —
#: gives the recursion the room the main thread would have had, so a deep solve
#: either finishes or raises a catchable RecursionError that the budget handles
#: like any other refusal. This is why the budget must not simply be `signal`
#: on the main thread; a worker is the only portable option, and a worker needs
#: this.
_WORKER_STACK: Final[int] = 64 * 1024 * 1024

_budget: float | None = DEFAULT_BUDGET
_WORKER_RUNNING: threading.Event = threading.Event()
#: The most recent worker, kept only so a later call can ask it to unwind.
_last_worker: threading.Thread | None = None
#: SymPy's caches are process-global. Serialize independent callers instead of
#: mistaking a legitimate concurrent call for a stale worker and cancelling it.
_CALL_LOCK: threading.Lock = threading.Lock()
#: A budgeted operation may itself use another budgeted helper. Such a nested
#: call already runs under the outer deadline and must execute inline: starting
#: another worker would deadlock on ``_CALL_LOCK``.
_WORKER_CONTEXT: threading.local = threading.local()
#: Set when an injection was sent but the worker had not stopped yet, so the
#: cache repair still has to happen. See :func:`_repair_sympy_caches`.
_REPAIR_PENDING: threading.Event = threading.Event()


def get_budget() -> float | None:
    """The current per-attempt budget in seconds, or ``None`` when disabled."""
    return _budget


def set_budget(seconds: float | None) -> None:
    """Set the per-attempt budget. ``None`` disables it entirely.

    Disabling is the right call for a batch script that would rather wait than
    approximate, and it is what the test suite uses to assert that the exact
    answers are still exact when nothing is rushing them.
    """
    global _budget
    if seconds is not None and not seconds > 0:
        raise ValueError(f"a budget must be positive or None, not {seconds!r}")
    _budget = None if seconds is None else float(seconds)


def within_budget(
    operation: Callable[..., T], *args: Any, seconds: float | None = -1.0, **kwargs: Any
) -> T:
    """Run ``operation``, raising :class:`SymbolicTimeout` if it takes too long.

    ``seconds`` defaults to the module budget; pass a number to override it for
    one call, or ``None`` to run inline with no budget at all.

    The result — or the exception the operation raised — is passed through
    untouched, so wrapping a call never changes what a successful call does.
    """
    if getattr(_WORKER_CONTEXT, "active", False):
        # The outer worker already owns the deadline. Starting another worker
        # would wait on the lock held by the caller that is joining this one.
        return operation(*args, **kwargs)

    # SymPy mutates shared caches even for reads. Queue independent calls so one
    # caller can never inject a timeout into another caller's valid worker.
    with _CALL_LOCK:
        return _within_budget_serial(operation, *args, seconds=seconds, **kwargs)


def _within_budget_serial(
    operation: Callable[..., T], *args: Any, seconds: float | None, **kwargs: Any
) -> T:
    """Implementation of :func:`within_budget`, while ``_CALL_LOCK`` is held."""
    global _last_worker
    _settle_pending_repair()
    limit = _budget if seconds == -1.0 else seconds
    if limit is None:
        # Explicitly unlimited: the caller asked to wait, so wait.
        return operation(*args, **kwargs)
    if _WORKER_RUNNING.is_set() and not _unwind(_last_worker):
        # An earlier attempt survived injection and is still inside SymPy. This
        # one cannot be given a worker of its own without running SymPy from two
        # threads at once, and it must not run inline because nothing could then
        # stop it. Decline — but only for as long as that worker really lives.
        raise SymbolicTimeout(
            f"{name_of(operation)} was not attempted: an earlier symbolic step "
            "is still running and would not unwind"
        )

    box: dict[str, Any] = {}

    def work() -> None:
        _WORKER_CONTEXT.active = True
        try:
            try:
                box["value"] = operation(*args, **kwargs)
            except BaseException as error:  # noqa: BLE001 - re-raised on the caller's thread
                box["error"] = error
            finally:
                _WORKER_RUNNING.clear()
        except BaseException:  # noqa: BLE001 - see below
            # An injected exception is delivered at the next bytecode boundary,
            # which can be inside the `finally` above, after the handler that was
            # meant to absorb it has already run. Without this it escapes into
            # thread teardown and surfaces as an unraisable exception.
            _WORKER_RUNNING.clear()
        finally:
            _WORKER_CONTEXT.active = False

    worker = threading.Thread(target=work, daemon=True, name="mathslate-symbolic")
    _last_worker = worker
    _WORKER_RUNNING.set()
    _start_on_a_big_stack(worker)
    worker.join(limit)
    if worker.is_alive():
        _unwind(worker)
        raise SymbolicTimeout(f"{name_of(operation)} exceeded its {limit:g}s budget")
    if "error" in box:
        raise box["error"]
    return box["value"]  # type: ignore[no-any-return]


def _start_on_a_big_stack(worker: threading.Thread) -> None:
    """Start ``worker`` on a large stack — see :data:`_WORKER_STACK`.

    ``threading.stack_size`` is process-global and read at ``start()``, so it is
    set only across the start and then put back, to avoid enlarging the stack of
    threads the user's own program creates. Not every platform honours it; where
    it raises, the worker simply starts on the default stack.
    """
    try:
        previous = threading.stack_size(_WORKER_STACK)
    except (ValueError, RuntimeError):
        worker.start()
        return
    try:
        worker.start()
    finally:
        try:
            threading.stack_size(previous)
        except (ValueError, RuntimeError):  # pragma: no cover - platform-specific
            pass


def name_of(operation: Callable[..., Any]) -> str:
    """A readable name for an operation, for the timeout message."""
    return str(getattr(operation, "__name__", operation))


def _unwind(worker: threading.Thread | None) -> bool:
    """Ask ``worker`` to raise and stop. True once it is no longer running.

    ``PyThreadState_SetAsyncExc`` sets a pending exception on the target thread,
    which CPython delivers at its next bytecode boundary. Pure-Python recursion
    — which is what SymPy's search functions are — reaches one almost
    immediately. A thread blocked inside a C extension does not, and this
    returns ``False`` for it.
    """
    if worker is None or not worker.is_alive():
        return True
    ident = worker.ident
    if ident is not None:
        _inject_timeout(ident)
    worker.join(_UNWIND_GRACE)
    stopped = not worker.is_alive()
    if stopped:
        _try_repair()
    else:
        # Still unwinding. Repair is owed either way — an injection that lands a
        # second late corrupts a cache write just as thoroughly as one that lands
        # at once — so it is deferred rather than skipped. Making it conditional
        # on the grace period would make correctness depend on machine load.
        _REPAIR_PENDING.set()
    return stopped


def _inject_timeout(ident: int) -> bool:
    """Request one async exception, undoing the request if CPython over-applies it.

    CPython documents three outcomes: zero thread states changed, exactly one
    changed, or (on internal failure) more than one changed. The last case must
    be rolled back immediately; leaving several unrelated threads marked for a
    timeout would be substantially worse than missing this deadline.
    """
    affected = _set_async_exception(ident, SymbolicTimeout)
    if affected > 1:
        _set_async_exception(ident, None)
        return False
    return affected == 1


def _set_async_exception(ident: int, exception: type[BaseException] | None) -> int:
    """Thin, replaceable wrapper around CPython's private thread-state API."""
    argument = None if exception is None else ctypes.py_object(exception)
    return int(
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident), argument)
    )


def _settle_pending_repair() -> None:
    """Do a repair owed by an earlier interruption, once its worker is gone."""
    if _REPAIR_PENDING.is_set() and not _WORKER_RUNNING.is_set():
        _REPAIR_PENDING.clear()
        _try_repair()


def _try_repair() -> None:
    """Repair the caches, never letting that failure replace the real error."""
    try:
        _repair_sympy_caches()
    except Exception:  # noqa: BLE001 - the timeout is the news, not this
        pass


def _repair_sympy_caches() -> None:
    """Discard SymPy's memo caches after an interrupted computation.

    This is not tidiness. The interruption lands at an arbitrary bytecode
    boundary, which can be **in the middle of a cache write**, and SymPy's caches
    are module-level state shared with the calling thread. Observed once the
    injection was added: a `solveset` stopped inside `rootoftools` left
    `_complexes_cache` holding a key whose value was never stored, and the next
    unrelated lookup raised `KeyError` from deep inside SymPy — a failure with no
    visible connection to the timeout that caused it.

    Clearing costs the warm cache, which is exactly the thing that might be
    wrong. Both entry points are needed: `CRootOf` keeps its own.
    """
    try:
        from sympy.core.cache import clear_cache
        from sympy.polys.rootoftools import CRootOf

        CRootOf.clear_cache()
        clear_cache()
    except Exception:  # noqa: BLE001 - repair must not raise over the real error
        pass
