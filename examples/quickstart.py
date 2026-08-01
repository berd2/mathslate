"""The MathSlate quickstart, as one script.

This file is the single source of truth for the three-environment check
(PRD v0.1 acceptance criterion 5). ``tests/test_environments.py`` runs it as a
plain script, inside a Jupyter kernel and inside marimo, then compares the
numbers the three produce.

Run it directly to see the same thing in a terminal::

    python examples/quickstart.py
"""

from __future__ import annotations

import json
from typing import Any

from mathslate import Abs, cos, exp, floor, plot, sin, sqrt, t, tan, x


def quickstart() -> dict[str, Any]:
    """Every v0.1 feature, once. Returns a fingerprint of what was drawn."""
    fingerprint: dict[str, Any] = {}

    # 1. The first graph: no ranges, no options, no documentation.
    first = plot(sin(x) / x, verbose=False)
    fingerprint["first_graph"] = _digest(first)

    # 2. The functions that separate a real tool from a plotting wrapper.
    for name, expr in (
        ("tan", tan(x)),
        ("reciprocal", 1 / x),
        ("floor", floor(x)),
        ("sqrt", sqrt(x)),
        ("sign", x / Abs(x)),
    ):
        fingerprint[name] = _digest(plot(expr, verbose=False))

    # 3. The one rule: a list overlays, a tuple is one vector-valued object.
    fingerprint["overlay"] = _digest(plot([sin(x), cos(x)], verbose=False))
    fingerprint["parametric"] = _digest(plot((cos(t), sin(t)), verbose=False))

    # 4. Context-aware axes: pi ticks here, plain numbers there.
    fingerprint["pi_ticks"] = list(plot(sin(x), verbose=False).plotly.layout.xaxis.ticktext or [])
    fingerprint["numeric_ticks"] = plot(exp(x), verbose=False).plotly.layout.xaxis.ticktext is None

    # 5. Progressive unboxing: the equivalent plain Python.
    fingerprint["code_length"] = len(plot(sin(x) / x, verbose=False).python())

    # 6. Properties of the object — explicit, never automatic (v0.5).
    report = plot(x**3 - 3 * x, verbose=False).analyze()
    fingerprint["analysis"] = {
        "roots": [round(v, 9) for v in report.roots],
        "maxima": [round(v, 9) for v in report.maxima],
        "minima": [round(v, 9) for v in report.minima],
        "inflections": [round(v, 9) for v in report.inflections],
        "symmetry": report.symmetry.kind,
        # A property that is exact in one host and sampled in another would be
        # a parity failure worth hearing about, so the label is compared too.
        "approximate": report.approximate,
    }

    return fingerprint


def _digest(result: Any) -> dict[str, Any]:
    """A small, exactly comparable summary of a plot."""
    sample = result.plan.series[0].sample
    return {
        "kind": result.plan.kind,
        "points": result.plan.total_points,
        "finite": sample.finite_count,
        "breakpoints": [round(b, 9) for b in sample.breakpoints],
        "y_range": [round(v, 9) for v in result.plan.y_range] if result.plan.y_range else None,
    }


FINGERPRINT: dict[str, Any] = quickstart()

if __name__ == "__main__":
    print(json.dumps(FINGERPRINT, indent=2, sort_keys=True))
