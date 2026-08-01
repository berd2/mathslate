"""Pure-Python core: independent of frontend and plotting backend (PRD 6.1)."""

from __future__ import annotations

from . import binding, dispatch, sampling
from ._budget import DEFAULT_BUDGET, SymbolicTimeout
from ._budget import get_budget as get_symbolic_budget
from ._budget import set_budget as set_symbolic_budget

__all__ = [
    "binding",
    "dispatch",
    "sampling",
    # The symbolic wall-clock budget. Deliberately reachable from
    # `mathslate.core` rather than from `mathslate` itself: it is a knob for the
    # rare session that hit the limit and wants a different trade, not something
    # a learner is meant to discover, and the PRD 4 surface budget is full. The
    # note printed when the budget fires names this path in full, so nobody has
    # to go looking for it.
    "set_symbolic_budget",
    "get_symbolic_budget",
    "DEFAULT_BUDGET",
    "SymbolicTimeout",
]
