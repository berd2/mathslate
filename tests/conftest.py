"""Test-suite setup: keep Plotly from trying to open a browser."""

from __future__ import annotations

import warnings
from typing import Any, Iterator

import plotly.basedatatypes
import pytest


#: NumPy's arithmetic chatter, which every plot of `tan(x)` or `1/x` produces
#: by design. Matched on the message so that MathSlate's *own* RuntimeWarnings
#: still reach the test that is watching for them.
_NUMPY_NOISE: tuple[str, ...] = (
    "divide by zero",
    "invalid value",
    "overflow",
    "underflow",
    "Mean of empty slice",
    "All-NaN",
    "Degrees of freedom",
)


@pytest.fixture(autouse=True)
def _quiet_numeric_warnings() -> Iterator[None]:
    """Silence NumPy's numeric noise — and only that.

    Ignoring `RuntimeWarning` wholesale also hid the element-wise fallback
    notice, which PRD 5.3 step 6 requires to be loud. A test asserting the
    fallback is audible would then have passed for the wrong reason, or needed
    to fight this fixture to see anything at all.
    """
    with warnings.catch_warnings():
        for message in _NUMPY_NOISE:
            warnings.filterwarnings("ignore", message=message, category=RuntimeWarning)
        yield


@pytest.fixture(autouse=True)
def _isolate_sliders() -> Iterator[None]:
    """A slider binds its symbol for the session — including into the next test.

    That lifetime is deliberate (it is what makes PRD 5.2 rule 2 work without
    the user restating it), but it means one test's `slider(..., name="a")`
    silently turns `a` into a parameter everywhere afterwards. It has already
    done so once: a slider in the manual's examples reached
    `tests/test_input_validation.py` and changed what that module observed.
    """
    from mathslate.core import binding
    from mathslate.ui import interact

    def reset() -> None:
        interact.release_all()
        # `release_all()` only unbinds sliders. A test that calls
        # `binding.bind_parameter()` directly leaves a binding no slider owns,
        # which turned `a` into a parameter for whichever test ran next on the
        # same worker.
        binding._BOUND.clear()

    # Both sides, not just teardown. Teardown alone leaves the *first* test
    # exposed to anything module-level collection created, and leaves every
    # test exposed if an earlier autouse fixture raises before this one's
    # teardown is reached.
    reset()
    try:
        yield
    finally:
        reset()


@pytest.fixture()
def headless_show(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace ``Figure.show()`` so generated code can run inside pytest."""
    shown: list[Any] = []

    def fake_show(self: Any, *args: Any, **kwargs: Any) -> None:
        shown.append(self)

    monkeypatch.setattr(plotly.basedatatypes.BaseFigure, "show", fake_show)
    return shown
