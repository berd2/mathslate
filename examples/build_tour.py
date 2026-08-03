"""Build the guided tour notebooks from one list of cells.

There are two tour notebooks — ``mathslate_tour.py`` for marimo and
``mathslate_tour.ipynb`` for Jupyter — and they have to be the same notebook or
they are worse than useless. So neither is written by hand: the cells live here
once, this script emits the marimo file, and ``marimo export ipynb`` derives the
Jupyter one from that.

The marimo cell signatures (``def _(a, b):`` … ``return (c,)``) are computed by
reading each cell with the same AST rules marimo uses, rather than being typed
out and hoped over. Getting one wrong is a dataflow error the reader would meet
instead of a graph.

Run it from the project root::

    python examples/build_tour.py
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from textwrap import dedent, indent

HERE = Path(__file__).resolve().parent
MARIMO_FILE = HERE / "mathslate_tour.py"
JUPYTER_FILE = HERE / "mathslate_tour.ipynb"

MARIMO_VERSION = "0.23.15"


def md(text: str) -> str:
    """A prose cell."""
    body = dedent(text).strip()
    return f'mo.md(\n    r"""\n{indent(body, "    ")}\n    """\n)'


# --------------------------------------------------------------------------
# the tour
# --------------------------------------------------------------------------

CELLS: list[str] = [
    # marimo is reactive and would resolve `mo` whatever the file order, but
    # the Jupyter twin runs top to bottom, so the imports genuinely have to
    # come first. `check_order` below enforces it for every name.
    """
    import marimo as mo
    import numpy as np
    """,
    md(
        """
        # MathSlate — a guided tour

        Run every cell. Each one is a claim you can check with your own eyes:
        the graph is right, the inferred settings are stated out loud, and the
        plain Python behind it is one call away.

        This notebook exercises the whole of v1.0 — 2D curves, discontinuities,
        3D, sliders, `analyze()`, tables, data fitting, worksheets and the
        optional AI layer. Nothing here needs a network.
        """
    ),
    md(
        """
        ## 0. Importing

        marimo rejects `from mathslate import *` before the cell runs — it has
        to know statically which names a cell defines in order to build its
        reactive graph. So the names come in explicitly. Everything else in the
        tour is identical in either notebook.
        """
    ),
    """
    from mathslate import (
        plot, polar, analyze, table, slider, animate, dataset, show_python,
        set_verbose, get_verbose, frontend_report,
        sin, cos, tan, exp, log, sqrt, Abs, floor, sign, pi, Eq, Matrix,
        diff, integrate, limit, solve, solveset, nsolve, simplify, symbols, Rational,
        factor, expand, apart, cancel, together, trigsimp, gcd,
        x, y, z, t, n, k, theta,
    )
    from mathslate.ui import release, release_all
    import mathslate

    print(f"MathSlate {mathslate.__version__} · {frontend_report()}")
    """,
    # -- 1. the first graph -------------------------------------------------
    md(
        """
        ## 1. The first graph

        No `symbols`, no `lambdify`, no `linspace`, no `figure`, no `show`.
        `x` already exists, and `sin` is SymPy's own — re-exported, not wrapped.
        """
    ),
    """
    first = plot(sin(x)/x)
    first
    """,
    md(
        """
        Read the line it printed. `sin(x)/x` is **undefined at x = 0**, and the
        curve is cut there rather than drawn through a point that does not
        exist. Everything the call decided for you is in that one line.
        """
    ),
    """
    print(first.summary())
    for note in first.notes:
        print(" ·", note)
    """,
    # -- 2. the hard functions ----------------------------------------------
    md(
        """
        ## 2. The functions that separate a tool from a wrapper

        Look for what is **not** there: no near-vertical line joining the top
        of one branch to the bottom of the next. Those lines are an artefact of
        joining consecutive samples without asking whether the function is
        continuous between them. Most plotters draw them.
        """
    ),
    "plot(tan(x))",
    "plot(1/x)",
    "plot(floor(x))",
    "plot(sqrt(x))",
    "plot(x/Abs(x))",
    md(
        """
        The evidence, rather than the picture: where the line was cut, and
        which stretches of the window the function is actually real on.
        """
    ),
    """
    for hard in (tan(x), 1/x, floor(x), sqrt(x), x/Abs(x), 1/(x**2 - 1)):
        cuts = plot(hard, verbose=False).plan.series[0].sample
        print(f"{str(hard):12s} cuts={len(cuts.breakpoints):2d}  real on {cuts.domain_intervals}")
    """,
    md(
        """
        A steep curve is **not** a discontinuity, and MathSlate tells them
        apart by squeezing the interval and watching whether the jump survives.
        `atan(1000x)` rises almost vertically and is not cut anywhere.
        """
    ),
    """
    from sympy import atan

    steep = plot(atan(1000*x), (x, -1, 1), verbose=False)
    print("cuts in atan(1000x):", steep.plan.series[0].sample.breakpoints)
    steep
    """,
    # -- 3. context-aware axes ----------------------------------------------
    md(
        """
        ## 3. Axes that read the expression

        Trigonometry gets π ticks. `exp` does not, because π has nothing to do
        with it. Nobody asked for either.
        """
    ),
    """
    print("sin ticks :", list(plot(sin(x), verbose=False).plotly.layout.xaxis.ticktext))
    print("exp ticks :", plot(exp(x), verbose=False).plotly.layout.xaxis.ticktext)
    plot(sin(x))
    """,
    md(
        """
        A log scale is **suggested, never applied**. A learner reading a
        log-scaled plot without realising it is worse off than one reading an
        awkward linear plot.
        """
    ),
    """
    growth = plot(exp(x), verbose=False)
    print([note for note in growth.notes if "log scale" in note])
    plot(exp(x), yscale="log")
    """,
    # -- 4. the one rule ----------------------------------------------------
    md(
        """
        ## 4. The one rule

        A **list** means *several things together*. A **tuple** means *one
        vector-valued object*. Same two expressions, different bracket,
        different picture.
        """
    ),
    "plot([sin(x), cos(x), sin(x) + cos(x)])",
    "plot((cos(t), sin(t)))",
    md(
        """
        `r = f(θ)` is written identically to `y = f(x)`, so inference cannot
        decide it in principle — and MathSlate refuses to guess. You say it.
        """
    ),
    "polar(1 + cos(t))",
    "polar(sin(3*t))",
    md("Your own numbers work too, as do plain Python functions."),
    """
    print(plot([2.0, 4.0, 8.0, 16.0, 32.0], verbose=False).plan.kind)
    print(plot(([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0]), verbose=False).plan.kind)
    plot(np.tanh, (x, -5, 5))
    """,
    # -- 5. show_python -----------------------------------------------------
    md(
        """
        ## 5. `show_python()` — the point of the whole project

        Not pseudocode. Paste it into a file and it runs, and it writes out
        every decision MathSlate made quietly: the real domain in pieces, the
        cuts, the y-window, the π ticks.
        """
    ),
    "_ = plot(tan(x), verbose=False).show_python()",
    md("It really runs — here it is, executed in a namespace holding nothing."),
    """
    emitted = {}
    exec(plot(sin(x)/x, verbose=False).python(), emitted)
    print(type(emitted["fig"]).__name__, "with", len(emitted["fig"].data[0].x), "points")
    """,
    # -- 6. escape hatches --------------------------------------------------
    md(
        """
        ## 6. Nothing here is a one-way door

        Every result hands you the underlying objects, so you can stop using
        MathSlate at any moment without losing work.
        """
    ),
    """
    hatch = plot(sin(x)/x, verbose=False)
    print("sympy :", hatch.sympy)
    print("numpy :", hatch.numpy[0].shape, hatch.numpy[1].shape)
    print("plotly:", type(hatch.plotly).__name__)
    print("integral of it from 1 to 2 :", integrate(hatch.sympy, (x, 1, 2)).evalf(6))
    """,
    md("And the SymPy re-exports are SymPy — there is nothing to unlearn."),
    """
    print(diff(sin(x)*exp(x), x))
    print(solve(x**2 - 5*x + 6, x))
    print(limit(sin(x)/x, x, 0))
    print(simplify((x**2 - 1)/(x - 1)))
    """,
    # -- 7. solving and algebra ---------------------------------------------
    md(
        """
        ## 7. Solving and algebra — SymPy's engine, right here

        MathSlate is a way of *seeing* mathematics, not a computer-algebra
        system: every symbolic computation is delegated to SymPy and the
        functions are re-exported unchanged. So the whole solving and algebra
        toolkit is already imported, and everything below is ordinary SymPy —
        usable, unchanged, in any Python project.
        """
    ),
    md(
        """
        ### Solving

        `solve()` takes an expression (read as `= 0`) or an `Eq`, and returns
        the solutions.
        """
    ),
    """
    print("expression (= 0) :", solve(x**2 - 5*x + 6, x))
    print("an equation      :", solve(Eq(x**2, 2), x))
    print("a trig equation  :", solve(sin(x) - Rational(1, 2), x))
    """,
    md("A **system** is a list of equations and the unknowns to solve for."),
    """
    print("two lines meet   :", solve([x + y - 3, x - y - 1], [x, y]))
    print("a circle & a line:", solve([x**2 + y**2 - 1, y - x], [x, y]))
    """,
    md(
        """
        `solveset()` returns the *whole* solution set — every branch of a
        periodic equation, not just one. `nsolve()` finds a single root
        numerically from a starting guess, for equations with no closed form.
        """
    ),
    """
    print("solveset(sin x)  :", solveset(sin(x), x))
    print("nsolve(x - cos x):", nsolve(x - cos(x), x, 0.5))
    """,
    md(
        """
        ### Algebra

        `factor`, `expand` and `gcd` do what they say, over polynomials in any
        number of symbols.
        """
    ),
    """
    print("factor  :", factor(x**3 - x))
    print("expand  :", expand((x + 1)**3))
    print("gcd     :", gcd(x**2 - 1, x**2 - x))
    """,
    md(
        """
        Rational expressions have their own verbs — `apart` for partial
        fractions, `cancel` and `together` for the two directions of a common
        denominator — and `trigsimp` for the trigonometric identities.
        """
    ),
    """
    print("apart   :", apart(1/(x**2 - 1)))
    print("cancel  :", cancel((x**2 - 1)/(x - 1)))
    print("together:", together(1/x + 1/y))
    print("trigsimp:", trigsimp(sin(x)**2 + cos(x)**2))
    """,
    md(
        """
        The point of re-exporting rather than wrapping: whatever SymPy hands
        back is an ordinary expression, so it flows straight into a plot with no
        conversion.
        """
    ),
    """
    alg_model = factor(x**3 - 6*x**2 + 11*x - 6)
    print("factored:", alg_model, " roots:", solve(alg_model, x))
    plot(alg_model, (x, 0, 4), verbose=False)
    """,
    md(
        """
        There are thousands of SymPy functions and only the common ones are
        re-exported at the top level. The whole library is one attribute away,
        as `mathslate.sympy` — so nothing is ever out of reach.
        """
    ),
    """
    from mathslate import sympy

    print("linsolve via mathslate.sympy :", sympy.linsolve([x + y - 3, x - y - 1], [x, y]))
    print("a Taylor series              :", sympy.series(sin(x), x, 0, 6))
    """,
    # -- 8. analyze ---------------------------------------------------------
    md(
        """
        ## 8. `analyze()` — properties of the object

        Explicit, never automatic: running property detection on every plot
        would be slow and noisy. It reports what a function **is**, never how a
        result was derived — step-by-step derivation is permanently out of
        scope.
        """
    ),
    """
    cubic = analyze(x**3 - 3*x)
    print(cubic.text())
    """,
    """
    print(analyze(1/x).text())
    """,
    """
    print(analyze(sin(x)).text())
    """,
    md(
        """
        Where SymPy can answer exactly it does; where it cannot, the answer is
        labelled approximate rather than quietly presented as exact.
        """
    ),
    """
    for subject in (x**3 - 3*x, Abs(x), (x**2 - 1)/(x - 1)):
        found = analyze(subject)
        print(f"{str(subject):18s} approximate={found.approximate}")
    """,
    md("`analyze()` is on the result object too, and it renders as a panel."),
    "plot(x**3 - 3*x, verbose=False).analyze()",
    # -- 8. table -----------------------------------------------------------
    md(
        """
        ## 9. `table()` — the same function as numbers
        """
    ),
    """
    values = table(sin(x)/x, (x, -1, 1), rows=9)
    print(values.text())
    values
    """,
    # -- 9. sliders ---------------------------------------------------------
    md(
        """
        ## 10. Interactive mathematics

        You never write a callback. A `Slider` is not a `Symbol`, but it
        answers SymPy's `_sympy_()` hook with one, so `amp*sin(x)` is an
        ordinary expression and everything downstream stays unaware a widget
        was involved.

        Drag the control under the plot.
        """
    ),
    """
    release_all()
    amp = slider(-3, 3, default=1, name="amp")
    waves = plot(amp*sin(x))
    print("frames:", len(waves.plotly.frames), " interactive:", waves.interactive)
    waves
    """,
    md("`animate()` is the same picture with a play button."),
    "animate(amp*sin(x))",
    md(
        """
        The slider's symbol is a **parameter**, not an axis — that is the
        binding rule doing its job without being told.
        """
    ),
    """
    print("axis symbol:", plot(amp*sin(x), verbose=False).plan.symbol.name)
    print("frame positions:", amp.values()[:5], "...")
    """,
    md(
        """
        ### The one thing to know about sliders

        A slider binds its symbol for the **whole session**, not just the cell.
        That is what makes the line above work without restating anything — and
        it means a slider named after a symbol you also want to plot *over* is
        a collision. MathSlate refuses rather than drawing the frozen point:
        """
    ),
    """
    clash = slider(0, 10, default=3, name="t")
    try:
        plot((cos(t), sin(t)))
    except Exception as error:
        print(type(error).__name__)
        print(error)
    """,
    md(
        """
        Any of the three remedies works. Releasing **that** slider rather than
        every slider matters here: marimo runs independent cells in whatever
        order the graph allows, so a bare `release_all()` could land before the
        cells above that still need `amp`. Naming the slider makes the
        dependency explicit, and the notebook order-proof.
        """
    ),
    """
    release(clash)
    print("after release:", plot((cos(t), sin(t)), verbose=False).plan.kind)
    print("amp is untouched:", plot(amp*sin(x), verbose=False).plan.symbol.name)
    """,
    # -- 10. three dimensions ----------------------------------------------
    md(
        """
        ## 11. Three dimensions

        One expression with two free symbols is a surface — that is the
        dispatch contract, not a special case. Rotate it with the mouse.
        """
    ),
    "plot(x*y)",
    md("The same object, flat. This is the other case inference cannot decide."),
    "plot(x*y, kind='contour')",
    "plot(sin(sqrt(x**2 + y**2)), (x, -8, 8), (y, -8, 8))",
    md(
        """
        A surface is ruled with grid lines by default — the sense of curvature a
        bare colour gradient loses. `mesh=False` returns the smooth look. A pole
        is bounded automatically so it cannot wall off the rest; `zlim=` sets the
        window yourself.
        """
    ),
    "plot(x*y, mesh=False, title='mesh=False')",
    "plot(1/(x*y), zlim=(-6, 6), title='a pole, bounded by zlim')",
    md("An equation rather than an expression is an implicit curve."),
    "plot(Eq(x**2 + y**2, 4))",
    "plot(Eq(x**2 - y**2, 1))",
    md("Three components sharing one symbol is a space curve; sharing two, a surface."),
    "plot((cos(t), sin(t), t), (t, 0, 12))",
    "plot((cos(t)*sin(z), sin(t)*sin(z), cos(z)))",
    # -- 11. linear algebra -------------------------------------------------
    md(
        """
        ## 12. Linear algebra

        A matrix draws as what it does to the plane, with its real
        eigenvectors marked.
        """
    ),
    """
    transform = plot(Matrix([[2, 1], [1, 3]]))
    print("eigen pairs:", [(round(value, 4), vector.round(4).tolist())
                           for value, vector in transform.plan.eigen])
    transform
    """,
    "plot(Matrix([[0, -1], [1, 0]]), title='A rotation has no real eigenvector')",
    # -- 12. data -----------------------------------------------------------
    md(
        """
        ## 13. `dataset()` — the bridge from symbolic to data

        Everything so far started from an expression. Real work usually starts
        from measurements, and `fit()` is where the two meet: you write the
        model as you would on paper and get **the same expression with its
        parameters filled in**, still a SymPy object.
        """
    ),
    """
    a, b, c = symbols("a b c", real=True)
    readings = dataset({
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [1.1, 2.9, 5.2, 6.8, 9.1, 11.0],
    })
    print(readings.describe())
    """,
    """
    straight = readings.fit(a*x + b)
    print(straight.describe())
    print("still SymPy:", straight.expr, "| derivative:", diff(straight.expr, x))
    """,
    "plot([straight.expr], (x, 0, 5), title='the fitted line')",
    md(
        """
        A model linear **in its parameters** is solved exactly. Anything else
        is refined by damped least squares from several starting points — which
        is what lets a model with a pole in it converge at all.
        """
    ),
    """
    curve_x = np.linspace(0.1, 5.0, 60)
    decay = dataset({"x": curve_x, "y": 2.5*np.exp(-0.7*curve_x) + 0.4})
    print(decay.fit(a*exp(b*x) + c).describe())

    poles = dataset({"x": curve_x, "y": 1.0/(curve_x + 0.5) + 0.2})
    print(poles.fit(a/(x + b) + c).describe())
    """,
    md("A fitted model is an ordinary expression, so `analyze()` applies to it."),
    """
    print(analyze(decay.fit(a*exp(b*x) + c).expr).text())
    """,
    md("Statistics: distributions of a column."),
    """
    rng = np.random.default_rng(7)
    sample = rng.normal(loc=10.0, scale=2.0, size=400)
    plot(sample, kind="hist", title="400 draws from a normal")
    """,
    "plot(sample, kind='box')",
    # -- 13. worksheets -----------------------------------------------------
    md(
        """
        ## 14. Classroom mode — one file you can hand out

        Plots, tables, analyses and prose on one page. Plotly is embedded
        **once** however many figures the page holds, and the result opens with
        no network and nothing installed — including its sliders, which is why
        they are built from frames rather than notebook widgets.
        """
    ),
    """
    from mathslate.classroom import worksheet

    handout = worksheet([
        "Where does sin(x)/x go at zero?",
        ("The graph", plot(sin(x)/x, verbose=False)),
        ("The numbers", table(sin(x)/x, (x, -1, 1), rows=7)),
        ("The properties", analyze(sin(x)/x)),
    ], title="Limits", subtitle="A one-page handout")
    handout
    """,
    md(
        """
        Displaying it shows a card, not the page: a whole HTML document in an
        output cell would restyle the notebook around it. `preview()` puts it
        in a sandboxed frame, and `save()` writes the file.
        """
    ),
    """
    mo.Html(handout.preview(height=420))
    """,
    md("`handout.save('limits.html')` writes it. Left commented so the tour writes nothing."),
    # -- 14. the AI layer ---------------------------------------------------
    md(
        """
        ## 15. The optional AI assistant

        Entirely optional and entirely offline-safe: the core never imports it,
        no provider is bundled, and generated code is shown before it can run.
        The panel guides first-time setup, reports progress and errors, and can
        remember a key in this computer's secure credential manager.
        """
    ),
    """
    from mathslate.ai import assistant, ask

    assistant("plot the tangent over one period")
    """,
    md(
        """
        The panel is the easiest first step. After setup, the programmatic API
        is equally short:

        ```python
        ask("plot the tangent over one period").run()
        ```

        The example is shown rather than run automatically, so opening the
        tour never sends a request or consumes provider quota.
        """
    ),
    md(
        """
        Writing code is one job. Most of the module does the other one —
        reading what MathSlate has **already** worked out. That needs no
        arithmetic from a model, so it cannot go wrong the same way:

        ```python
        plot(sin(x), 0, 6.28)          # TypeError: a range is (symbol, lo, hi)
        explain()                      # ...and here is the corrected line

        describe(analyze(x**3 - 3*x))  # prose about roots already solved
        ask("now on a log scale", about=drawn)   # a follow-up with an "it"
        draft.repair(failure)          # a second try, error as evidence
        suggest_model(readings)        # the data picks the curve to fit
        ```

        `explain()` reads the exception Python just reported, so after a failed
        cell the call is simply `explain()`. The error message is the evidence
        rather than the model's memory of MathSlate — and when MathSlate raised
        it, the model is told to trust it rather than re-diagnose it.
        """
    ),
    md(
        """
        Two of these need no provider at all, because the computing is
        MathSlate's and only the wording would have been the model's.
        `facts()` is exactly what `describe()` would send — the honest answer
        to "what did you share?" — and every property carries whether it was
        *solved* or *sampled*:
        """
    ),
    """
    from mathslate.ai import facts

    report = facts(analyze(x**3 - 3*x))
    (report["roots"]["exact"], report["roots"]["approximate"])
    """,
    md(
        """
        And `fit_evidence()` is the measurement behind `suggest_model()`: a
        model family is whatever transform straightens the data, so MathSlate
        measures the straightening rather than asking a model to guess. On
        exponential readings `log(y) ~ x` lands on 1.000 while the others sit
        near 0.94 — an answer, not an opinion.
        """
    ),
    """
    from mathslate.ai import fit_evidence

    _xs = np.linspace(1.0, 5.0, 20)
    _readings = dataset({"x": _xs, "y": 2 * np.exp(0.7 * _xs)})
    _straightness = fit_evidence(_readings)["straightness"]
    max((n for n, r in _straightness.items() if r),
        key=lambda n: _straightness[n]["r"])
    """,
    md(
        """
        Everything above points outward — MathSlate asking a model.
        `mathslate.ai.tools` points inward: an agent such as Claude or ChatGPT
        hands MathSlate an expression and gets a **computed** answer instead of
        a plausible recollection of one, with `approximate` and the method
        attached so it knows how far to trust each line.

        Arguments arriving from a model are given exactly the trust a generated
        suggestion is given — none. Each one is put through the same allowlist
        and the same isolated process before anything is evaluated.
        """
    ),
    """
    from mathslate.ai import call, tool_names

    (tool_names(),
     call("mathslate_analyze", {"expression": "x**2 - 2"})["roots"]["exact"])
    """,
    # -- 15. when it refuses ------------------------------------------------
    md(
        """
        ## 16. What it refuses, and how it says so

        An error message is part of the interface. Each of these is a case
        where guessing would produce something you did not ask for.
        """
    ),
    """
    p, q, r_sym = symbols("p q r", real=True)
    attempts = [
        ("three free symbols", lambda: plot(p*q*r_sym)),
        ("a range for a symbol that is not there", lambda: plot(sin(x), (q, -1, 1))),
        ("an unknown option", lambda: plot(sin(x), kind="bar")),
        ("an unplottable object", lambda: plot({"not": "plottable"})),
        ("a model that cannot be separated", lambda: dataset({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}).fit(a*b*x)),
        ("fitting a column against itself", lambda: dataset({"v": [1.0, 2.0, 3.0]}).fit(a*x + b)),
    ]
    for label, attempt in attempts:
        try:
            attempt()
            print(f"{label:42s} -> no error")
        except Exception as caught:
            print(f"{label:42s} -> {type(caught).__name__}: {str(caught).splitlines()[0][:70]}")
    """,
    # -- 16. closing --------------------------------------------------------
    md(
        """
        ## 17. Where you are

        Every plot on this page can tell you its plain Python, hand you its
        SymPy expression, and give you its NumPy arrays. That is the whole
        design: the entry barrier of a graphing calculator, and no ceiling.

        - `docs/tutorial.md` — the guided introduction in prose
        - `docs/manual.md` — every option and the exact guarantees
        - `mathslate_prd_0.3.md` — why it is shaped this way
        """
    ),
    """
    print("verbosity is switchable:", get_verbose())
    set_verbose(False)
    quiet = plot(sin(x))
    print("silent:", quiet.summary())
    set_verbose(True)
    """,
]


# --------------------------------------------------------------------------
# marimo cell signatures, computed rather than typed
# --------------------------------------------------------------------------


def analyse(code: str) -> tuple[set[str], set[str]]:
    """``(defined, referenced)`` for one cell, marimo's rules in miniature."""
    tree = ast.parse(dedent(code))
    defined: set[str] = set()
    referenced: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                referenced.add(node.id)
            else:
                defined.add(node.id)
        elif isinstance(node, ast.alias):
            defined.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    return defined, referenced


def check_order(parsed: list[tuple[set[str], set[str]]]) -> None:
    """No cell may use a name a later cell defines.

    marimo would resolve a forward reference from its graph, but the Jupyter
    export runs top to bottom and would raise `NameError`. Requiring one order
    to satisfy both is what keeps the two notebooks the same notebook.
    """
    first_defined: dict[str, int] = {}
    for index, (defined, _) in enumerate(parsed):
        for name in defined:
            first_defined.setdefault(name, index)

    for index, (_, referenced) in enumerate(parsed):
        for name in sorted(referenced):
            where = first_defined.get(name)
            if where is not None and where > index:
                raise SystemExit(
                    f"cell {index} uses {name!r}, which cell {where} defines. "
                    "Move the definition earlier — the Jupyter twin runs in "
                    "file order."
                )


def build_marimo() -> str:
    parsed = [analyse(cell) for cell in CELLS]
    check_order(parsed)

    available: dict[str, int] = {}
    parameters: list[list[str]] = []
    for index, (defined, referenced) in enumerate(parsed):
        parameters.append(sorted(name for name in referenced if name in available))
        for name in defined:
            available.setdefault(name, index)

    # A name is returned only if a later cell asks for it.
    returns: list[list[str]] = []
    for index, (defined, _) in enumerate(parsed):
        wanted = {
            name
            for name in defined
            if any(name in parsed[later][1] for later in range(index + 1, len(parsed)))
            and available.get(name) == index
        }
        returns.append(sorted(wanted))

    lines = [
        "# Generated by examples/build_tour.py — edit the cells there, not here.",
        "import marimo",
        "",
        f'__generated_with = "{MARIMO_VERSION}"',
        'app = marimo.App(width="medium")',
        "",
    ]
    for cell, params, produced in zip(CELLS, parameters, returns):
        signature = ", ".join(params)
        lines.append("")
        lines.append("@app.cell")
        lines.append(f"def _({signature}):")
        lines.append(indent(dedent(cell).strip(), "    "))
        if produced:
            body = ", ".join(produced)
            lines.append(f"    return ({body},)" if len(produced) == 1 else f"    return ({body})")
        else:
            lines.append("    return")
        lines.append("")
    lines += ["", 'if __name__ == "__main__":', "    app.run()", ""]
    return "\n".join(lines)


def main() -> int:
    MARIMO_FILE.write_text(build_marimo(), encoding="utf-8")
    print(f"wrote {MARIMO_FILE.relative_to(HERE.parent)} ({len(CELLS)} cells)")

    completed = subprocess.run(
        [
            sys.executable, "-m", "marimo", "export", "ipynb",
            str(MARIMO_FILE), "-o", str(JUPYTER_FILE), "--sort", "top-down",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        print(completed.stdout, completed.stderr, sep="\n")
        return completed.returncode

    _name_the_kernel()
    print(f"wrote {JUPYTER_FILE.relative_to(HERE.parent)}")
    return 0


def _name_the_kernel() -> None:
    """Give the exported notebook a kernelspec.

    ``marimo export ipynb`` leaves the metadata empty, and a notebook with no
    kernelspec makes Jupyter stop and ask which kernel to use before the reader
    has seen anything. Naming the ordinary Python 3 kernel removes the prompt.
    """
    import json

    notebook = json.loads(JUPYTER_FILE.read_text(encoding="utf-8"))
    notebook.setdefault("metadata", {}).update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }
    )
    JUPYTER_FILE.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
