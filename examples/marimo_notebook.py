"""MathSlate in marimo — a runnable example notebook.

Two of marimo's rules shape this file, and both are marimo's rather than
MathSlate's:

**A cell can only use names some cell returned.** marimo builds its dependency
graph statically from those signatures, so a cell referencing a name nobody
returned never runs at all — it does not fail loudly, it simply does nothing.
That is why every cell below ends with an explicit ``return``.

**A figure renders by being the cell's last expression, not via ``.show()``.**
``.show()`` is Plotly's behaviour for a *script*: it opens a separate browser
window. In a notebook that loses the inline output and the reactive re-render.
``PlotResult`` implements ``_repr_html_``, so returning it is all it takes.

Run it with::

    marimo edit examples/marimo_notebook.py
"""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # marimo rejects `import *` at parse time — it cannot know statically which
    # names that would define. Naming them is the documented workaround.
    from mathslate import (
        dataset,
        diff,
        exp,
        limit,
        plot,
        sin,
        solve,
        symbols,
        x,
    )

    return dataset, diff, exp, limit, mo, plot, sin, solve, symbols, x


@app.cell
def _(plot, sin, x):
    # The figure is this cell's last expression, so marimo renders it inline.
    # `f.show()` here would open a browser window and leave the cell blank.
    f = plot(sin(x), (x, 0, 6.28))
    f
    return


@app.cell
def _(diff, exp, limit, sin, solve, x):
    # None of these are MathSlate functions — they are SymPy, re-exported
    # unchanged, so what you learn here works in any Python project.
    print(diff(sin(x) * exp(x), x))
    print(solve(x**2 - 5 * x + 6, x))
    print(limit(sin(x) / x, x, 0))
    return


@app.cell
def _(dataset, symbols, x):
    # The bridge from data back to symbols: the fitted model comes back as an
    # ordinary SymPy expression with its parameters filled in.
    readings = dataset({"x": [0, 1, 2, 3, 4], "y": [1.0, 3.1, 4.9, 7.2, 8.9]})
    c, d = symbols("c d", real=True)
    found = readings.fit(c * x + d)
    print(found.describe())
    return (found,)


@app.cell
def _(diff, found, x):
    # Which is the point: everything else still applies to the result.
    print(diff(found.expr, x))
    return


@app.cell
def _(found, plot, x):
    plot(found.expr, (x, -1, 5))
    return


@app.cell
def _(mo):
    # A live range control (PRD §19), the marimo way. In Jupyter/Colab this is
    # what `result.range_controls()` builds by hand from ipywidgets; here it is
    # two ordinary `mo.ui.number()` values, because marimo already re-runs any
    # cell that reads `.value` when either one changes — no MathSlate wrapper
    # needed. Unlike Plotly's own drag-to-zoom, moving these re-samples the
    # curve over the new domain, so narrowing the window draws it at full
    # resolution instead of cropping the points already there.
    x_min = mo.ui.number(start=-20, stop=20, value=-10, label="x min")
    x_max = mo.ui.number(start=-20, stop=20, value=10, label="x max")
    mo.hstack([x_min, x_max])
    return x_max, x_min


@app.cell
def _(plot, sin, x, x_max, x_min):
    plot(sin(x) / x, (x, x_min.value, x_max.value))
    return


if __name__ == "__main__":
    app.run()
