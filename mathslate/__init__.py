"""MathSlate — a mathematical workspace that grows with you.

    from mathslate import *
    plot(sin(x)/x)

Everything symbolic is SymPy's, unwrapped and re-exported. MathSlate adds
eight callables and one guarantee: the graph is right, and you can always ask
it what the plain Python was.
"""

from __future__ import annotations

# --- SymPy re-exports (zero MathSlate code) -------------------------------
# PRD 6.3: thin wrappers around identically named SymPy functions are not
# wrapped at all. What follows is a plain re-export.
import sympy as sympy
from sympy import (  # noqa: F401
    E,
    Abs,
    Eq,
    Float,
    Function,
    I,
    Integer,
    Matrix,
    Piecewise,
    Rational,
    Sum,
    Symbol,
    acos,
    apart,
    asin,
    atan,
    binomial,
    cancel,
    cbrt,
    ceiling,
    cos,
    cosh,
    cot,
    csc,
    diff,
    exp,
    expand,
    factor,
    factorial,
    floor,
    gcd,
    integrate,
    lambdify,
    limit,
    log,
    nsimplify,
    nsolve,
    oo,
    pi,
    root,
    sec,
    series,
    sign,
    simplify,
    sin,
    sinh,
    solve,
    solveset,
    sqrt,
    symbols,
    tan,
    tanh,
    together,
    trigsimp,
)

from . import errors as errors
from .api import (
    analyze,
    animate,
    dataset,
    frontend_report,
    get_plot_size,
    get_range_controls,
    get_verbose,
    plot,
    polar,
    set_plot_size,
    set_range_controls,
    set_verbose,
    show_python,
    slider,
    table,
)
from .core.analysis import Analysis
from .core.data import Dataset
from .core.tables import Table
from .result import PlotResult

__version__: str = "0.1.5"

# --- predefined symbols (PRD 6.3) -----------------------------------------
# Open decision 3: `import *` is permitted so that these exist, but
# show_python() always emits the explicit `x = sp.symbols('x')`.
x, y, z, t, n, k = symbols("x y z t n k", real=True)
theta = Symbol("theta", real=True)

#: New public symbols introduced by MathSlate — the API-surface budget.
NEW_API: tuple[str, ...] = (
    "plot",
    "polar",
    "slider",
    "animate",
    "table",
    "analyze",
    "show_python",
    "dataset",
    "set_verbose",
    "get_verbose",
    "set_range_controls",
    "get_range_controls",
    "set_plot_size",
    "get_plot_size",
    "frontend_report",
    "PlotResult",
    "Analysis",
    "Table",
    "Dataset",
)

__all__ = [
    # MathSlate
    *NEW_API,
    "errors",
    "sympy",
    "__version__",
    # predefined symbols
    "x", "y", "z", "t", "n", "k", "theta",
    # SymPy re-exports
    "E", "Abs", "Eq", "Float", "Function", "I", "Integer", "Matrix", "Piecewise",
    "Rational", "Sum", "Symbol", "acos", "apart", "asin", "atan", "binomial",
    "cancel", "cbrt", "ceiling", "cos", "cosh", "cot", "csc", "diff", "exp",
    "expand", "factor", "factorial", "floor", "gcd", "integrate", "lambdify",
    "limit", "log", "nsimplify", "nsolve", "oo", "pi", "root", "sec", "series",
    "sign", "simplify", "sin", "sinh", "solve", "solveset", "sqrt", "symbols",
    "tan", "tanh", "together", "trigsimp",
]
