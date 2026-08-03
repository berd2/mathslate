"""Frontend adapters (PRD 6.2) — the key v0.1 architectural decision.

Coupling to marimo would buy reactive sliders and cost Colab, which is the
largest zero-installation funnel for a high school student. So the frontend is
detected at runtime and ``pip install mathslate`` pulls in none of them.

+---------------------------+---------------------------------+
| Environment               | ``slider()`` implementation     |
+===========================+=================================+
| marimo                    | ``mo.ui.slider`` (reactive)     |
| Jupyter / Colab           | ``ipywidgets``                  |
| other / static export     | Plotly animation frames         |
+---------------------------+---------------------------------+

v0.1 ships the *detection*; the widgets themselves land in v0.5 together with
``slider()`` and ``animate()``.
"""

from __future__ import annotations

import importlib.util
import sys
from enum import Enum
from typing import Final

__all__ = ["Frontend", "detect_frontend", "frontend_report"]


class Frontend(Enum):
    """Which notebook, if any, is hosting this session."""

    MARIMO = "marimo"
    JUPYTER = "jupyter"
    COLAB = "colab"
    PLAIN = "plain"


_MODULE_AVAILABLE_CACHE: Final[dict[str, bool]] = {}


def _available(module: str) -> bool:
    if module not in _MODULE_AVAILABLE_CACHE:
        _MODULE_AVAILABLE_CACHE[module] = importlib.util.find_spec(module) is not None
    return _MODULE_AVAILABLE_CACHE[module]


def detect_frontend() -> Frontend:
    """Detect the host environment without importing any frontend package."""
    if "marimo" in sys.modules:
        marimo = sys.modules["marimo"]
        running = getattr(marimo, "running_in_notebook", None)
        if running is None or bool(running()):
            return Frontend.MARIMO
    if "google.colab" in sys.modules:
        return Frontend.COLAB
    # JupyterLite's Pyodide kernel runs IPython in a Web Worker, but its shell
    # is not named ``ZMQInteractiveShell``. It is still a live Jupyter kernel.
    if sys.platform == "emscripten":
        return Frontend.JUPYTER
    shell = _ipython_shell()
    if shell in {"ZMQInteractiveShell"}:
        return Frontend.JUPYTER
    return Frontend.PLAIN


def _ipython_shell() -> str | None:
    get_ipython = getattr(sys.modules.get("IPython", None), "get_ipython", None)
    if get_ipython is None:
        return None
    shell = get_ipython()
    return type(shell).__name__ if shell is not None else None


def frontend_report() -> str:
    """Human-readable line about the detected environment and its extras."""
    frontend = detect_frontend()
    widget = {
        Frontend.MARIMO: "mo.ui.slider (reactive)",
        Frontend.COLAB: "ipywidgets" if _available("ipywidgets") else "ipywidgets (not installed)",
        Frontend.JUPYTER: "ipywidgets" if _available("ipywidgets") else "ipywidgets (not installed)",
        Frontend.PLAIN: "Plotly animation frames (self-contained HTML)",
    }[frontend]
    return (
        f"frontend: {frontend.value} | interactive widgets: {widget} | "
        f"range_controls(): {_range_controls_status(frontend)}"
    )


def _range_controls_status(frontend: Frontend) -> str:
    """Why a bare plot() would or would not show range_controls() (PRD §19.4).

    ``ipywidgets`` alone is not the whole answer: live resampling also needs
    Plotly's ``FigureWidget``, which on Plotly >= 6 additionally needs
    ``anywidget`` — a second install the "interactive widgets" line above does
    not check, and the one most likely to be the actual reason nothing showed.
    Imported lazily: :mod:`.range_controls` builds the sidebar and so imports
    this module for :func:`detect_frontend`, and that is the direction the two
    should depend in — frontend detection knows nothing about widgets. This one
    call back up is what a lazy import is for.
    """
    from . import range_controls

    if not range_controls.get_range_controls():
        return "off — set_range_controls(True) to turn back on"
    if frontend not in (Frontend.JUPYTER, Frontend.COLAB):
        return f"not shown on {frontend.value} — needs a Jupyter or Colab kernel"
    if not range_controls.deps_available():
        missing = "anywidget" if _available("ipywidgets") else "ipywidgets"
        if sys.platform == "emscripten":
            return (
                f"not shown — {missing} is not installed; run "
                "`import piplite; await piplite.install(['nbformat', "
                "'ipywidgets', 'anywidget'])` once"
            )
        return f"not shown — {missing} is not installed (pip install mathslate[jupyter])"
    return "ready"
