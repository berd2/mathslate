"""The suite must stay able to hear MathSlate's own warnings.

PRD 5.3 step 6 requires the element-wise fallback to be *loud*: a plot that
silently became 2000× slower is a plot the reader will blame on the library.
The mechanism is a ``RuntimeWarning``.

`conftest._quiet_numeric_warnings` exists because NumPy emits its own
``RuntimeWarning`` for every ``tan(x)`` and ``1/x`` in the corpus, and a suite
that prints two hundred of those is a suite nobody reads. It used to do that
with `simplefilter("ignore", RuntimeWarning)`, which also silenced the notice
the PRD requires — the same category, so no filter could tell them apart
without matching on the message.

This module pins the distinction, on the fixture's own data rather than on a
paraphrase of it: whatever `_NUMPY_NOISE` grows to, it must keep matching
NumPy's messages and keep *not* matching MathSlate's.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
import sympy as sp

from conftest import _NUMPY_NOISE

from mathslate.core.sampling import NumericFunction

#: What NumPy actually says. These are the messages the fixture is for.
NUMPY_MESSAGES: tuple[str, ...] = (
    "divide by zero encountered in divide",
    "invalid value encountered in sqrt",
    "overflow encountered in exp",
    "underflow encountered in multiply",
    "Mean of empty slice",
    "All-NaN slice encountered",
    "Degrees of freedom <= 0 for slice",
)


def _suppressed(message: str) -> bool:
    """Whether `_NUMPY_NOISE` hides a warning with this message.

    `filterwarnings(message=...)` matches the *start* of the message, case
    insensitively, as a regex — mirrored here rather than guessed.
    """
    return any(re.compile(pattern, re.IGNORECASE).match(message) for pattern in _NUMPY_NOISE)


class TestTheFilterIsNarrow:
    @pytest.mark.parametrize("message", NUMPY_MESSAGES)
    def test_numpys_arithmetic_chatter_is_hidden(self, message: str) -> None:
        assert _suppressed(message), (
            f"NumPy's {message!r} now reaches the report — either NumPy reworded "
            "it or _NUMPY_NOISE lost an entry"
        )

    def test_no_entry_matches_everything(self) -> None:
        """An empty or `.*` pattern would silently restore the old behaviour."""
        for pattern in _NUMPY_NOISE:
            assert pattern.strip() not in {"", ".*", ".+"}
        assert not _suppressed("vectorised evaluation failed; falling back")


class TestMathslatesOwnWarningsStayAudible:
    def test_the_fallback_notice_is_not_suppressed(self) -> None:
        """The exact text `NumericFunction` emits, taken from the source of truth."""
        symbol = sp.Symbol("x", real=True)
        function = NumericFunction(sp.zeta(symbol, 3), symbol)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            function(np.array([0.5, 1.5]))
        assert caught, "the fallback did not warn at all"
        emitted = str(caught[0].message)
        assert "element-wise" in emitted
        assert not _suppressed(emitted), (
            f"the fixture hides MathSlate's own required notice: {emitted!r}"
        )

    def test_a_sampling_warning_in_general_is_not_suppressed(self) -> None:
        for message in (
            "vectorised evaluation failed (TypeError: ...); falling back",
            "the curve was clipped to the 2nd-98th percentile",
            "3% of the grid was not a real number",
        ):
            assert not _suppressed(message)
