"""Turning a SymPy expression back into runnable source text.

Two things emit code — :mod:`mathslate.codegen` for plots and
:mod:`mathslate.core.analysis` for ``analyze()`` — and both need the same two
answers: what does this expression look like as Python, and which names must be
imported for that text to run. Keeping one copy is what stops the two emitters
disagreeing about, say, whether ``sqrt`` needs importing.

Pure SymPy and string handling. It imports no plotting library, so ``core`` can
use it without breaking the layering of PRD 6.1.
"""

from __future__ import annotations

import re

import sympy as sp

__all__ = ["expr_source", "import_block"]

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def expr_source(expr: sp.Expr, symbol_names: set[str]) -> tuple[str, set[str]]:
    """SymPy expression → source text plus the sympy names it needs imported."""
    text = sp.sstr(expr)
    needed = {
        name
        for name in _IDENTIFIER.findall(text)
        if name not in symbol_names and hasattr(sp, name)
    }
    return text, needed


def import_block(
    needed: set[str], *, numpy: bool = True, sympy: bool = True, plotly: bool = True
) -> str:
    """The import header for emitted code, carrying only what is used."""
    lines: list[str] = []
    if numpy:
        lines.append("import numpy as np")
    if sympy:
        lines.append("import sympy as sp")
    if plotly:
        lines.append("import plotly.graph_objects as go")
    if needed:
        lines.append(f"from sympy import {', '.join(sorted(needed))}")
    return "\n".join(lines)
