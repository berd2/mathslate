# MathSlate Tutorial

**A one-sitting walkthrough, from your first graph to reading real Python.**

You need Python and about thirty minutes. You do not need to know NumPy,
Matplotlib, or what a `lambda` is. Everything here runs in Jupyter, Colab,
marimo, or a plain terminal.

If you want the exhaustive description of every option instead, that is the
[reference manual](manual.md). This page is the tour.

---

## 0. Install

```bash
pip install mathslate
```

For a local Jupyter learning setup with interactive graph controls, the Gemini
assistant and remembered API keys, install the complete set once:

```bash
pip install "mathslate[starter]"
```

This brings in `ipywidgets`, `anywidget`, `google-genai` and `keyring`; you do
not install or import those packages separately. From a checkout, the matching
one-shot setup scripts are `scripts/quickstart.ps1` on Windows and
`scripts/quickstart.sh` on macOS/Linux.

---

## 1. Your first graph

```python
from mathslate import *

plot(sin(x)/x)
```

That is the whole thing. No `import numpy`, no `symbols`, no `linspace`, no
`figure`, no `show`.

> **In marimo, write the imports out.** marimo rejects the star import before
> the cell runs, reporting that importing symbols with it is
> *not allowed in marimo*, and the cell does not execute. It is not optional
> and it is not about MathSlate; see §1.1. Use this in your first cell instead:
>
> ```python
> from mathslate import (
>     plot, polar, analyze, show_python, set_verbose, get_verbose,
>     slider, animate, table, dataset, frontend_report,
>     sin, cos, tan, atan, exp, log, sqrt, Abs, floor, pi,
>     diff, integrate, limit, solve, solveset, nsolve, simplify, symbols,
>     factor, expand, apart, cancel, together, trigsimp, gcd,
>     Eq, Integer, Rational, Matrix,
>     x, y, z, t, n, k, theta,
> )
> ```
>
> Everything else in this tutorial then works unchanged.

Two things are worth noticing.

**`x` already exists.** MathSlate predefines the symbols mathematicians
actually write: `x, y, z, t, n, k` and `theta`. You never had to declare them.

**`sin` is SymPy's `sin`, not a MathSlate invention.** MathSlate re-exports
SymPy rather than wrapping it, so everything you learn here is knowledge about
SymPy — a library you will still be using in ten years.

And a third thing, which is the interesting one: **`sin(x)/x` is undefined at
`x = 0`**, and the graph knows. The curve is cut there rather than being drawn
straight through a point that does not exist. We will come back to why that
matters in §5.

### 1.1 Why `import *` is banned in marimo

Worth two minutes, because the reason is genuinely interesting and it is the
one place MathSlate's convenience collides with a notebook's design.

marimo is *reactive*: change a cell, and every cell that depends on it re-runs
by itself. To do that it has to know, before running anything, which names each
cell **defines** and which it **uses** — that is how it builds the dependency
graph between cells. It works this out by reading the code, not by executing
it.

`from mathslate import *` defeats that completely. The set of names it defines
is whatever the module happens to contain, which cannot be known without
importing it. So marimo refuses at parse time, which is why you saw *"Cell not
run"* — the cell was rejected before a single line of it executed.

It is a hard rule with no opt-out, and it applies to every library, not just
this one. Jupyter, Colab and plain Python have no such constraint because they
never build the graph.

The fix is the explicit import above, and it is worth noticing that this is
the direction MathSlate wants you to go anyway. Naming what you import is what
real Python code does; `import *` exists here only so that a beginner's first
line is short. marimo just makes you grow out of it a little sooner.

---

## 2. Read the line it printed

Every `plot()` prints one line:

```text
curve | x ∈ [-10, 10] | 411 samples | 1 discontinuity handled
  · singularities at x = 0
```

Read it as: *"I decided this was a single 2D curve. I chose the window
`-10` to `10` for you. I used 411 sample points, not evenly spaced. I found one
place where the function breaks, and I cut the line there."*

This line is not logging noise. It is the whole teaching strategy of MathSlate
in one sentence: **it tells you about the decisions it made on your behalf, so
that you can start making them yourself.** Every number in that line is
something you can take control of, and the rest of this tutorial is mostly
about doing exactly that.

If it gets in your way:

```python
set_verbose(False)
plot(sin(x))          # silent now

set_verbose(True)
```

You can also read it back from the result instead of the console:

```python
f = plot(sin(x)/x)
print(f.summary())
```

---

## 3. Several curves, and the one rule

You will want more than one curve. There is exactly one rule to remember, and
it is the rule Python already taught you:

> A **list** means *several separate things*.
> A **tuple** means *one thing with several components*.

So a list overlays curves:

```python
plot([sin(x), cos(x), sin(x) + cos(x)])
```

and a tuple is a single parametric curve, where the two expressions are the
`x` and `y` of one moving point:

```python
plot((cos(t), sin(t)))
```

The first draws three curves. The second draws a circle. Same two expressions
could have gone in either bracket, and the bracket is what tells MathSlate
which you meant.

That one rule covers most of the dispatch contract. The
[manual](manual.md#3-the-dispatch-contract) has the full table.

---

## 4. Choose your own window

MathSlate picked `-10` to `10`. When you want something else, say so — a
range is a tuple of `(symbol, from, to)`:

```python
plot(sin(x), (x, 0, 6.28))
```

This is the first place where the printed line pays off: you saw
`x ∈ [-10, 10]`, you realised the window was a decision, and now you are making
it yourself. That is the pattern for everything else too.

A few other things you can ask for:

```python
plot(exp(-x**2), (x, -3, 3), title="A bell curve", label="Gaussian")
```

```python
plot(sin(x), points=2000)
```

`points` sets how many samples to start from. You will rarely need it — the
next section explains why.

---

## 5. The hard functions

This is the part that actually matters, and the reason MathSlate exists rather
than being a five-line wrapper around a plotting library.

Try these, all with no arguments:

```python
plot(tan(x))
```

```python
plot(1/x)
```

```python
plot(floor(x))
```

```python
plot(sqrt(x))
```

```python
plot(x/Abs(x))
```

Look at what did **not** happen.

`tan(x)` did not draw near-vertical lines connecting the top of one branch to
the bottom of the next. Those lines are not part of the graph of the tangent —
they are an artefact of a plotter joining consecutive sample points without
asking whether the function is continuous between them. Most tools draw them.

`1/x` is cut at zero, not joined across it. `floor(x)` is cut at every single
integer, so you get the staircase you were promised rather than a zigzag.
`sqrt(x)` is simply not drawn to the left of zero, because it is not real
there — it does not fade out or draw at zero, it stops. And `x/Abs(x)` jumps
from `-1` to `+1` without a line down the middle.

You can see what it found:

```python
f = plot(tan(x))
print(f.plan.series[0].sample.breakpoints)
```

```python
f = plot(sqrt(x))
print(f.plan.series[0].sample.domain_intervals)
```

The first prints the six poles of the tangent inside the window. The second
prints `((0.0, 10.0),)` — the only stretch of the window where `sqrt(x)` is a
real number.

**How it knows.** In outline: it asks SymPy for the poles and for the real
domain before sampling anything, so those come out exact rather than guessed.
Then it samples adaptively — more points where the curve bends, fewer where it
is straight — and cuts the line at every break. For jumps SymPy cannot see
(`floor`, `sign`, `Piecewise`), it uses a neat trick: squeeze the interval
around the suspicious jump and watch what the jump does. A real discontinuity
keeps its size no matter how far you zoom in. A merely steep slope flattens
out. That one test handles every step function without MathSlate ever needing
to know the word "floor".

The [manual](manual.md#6-sampling-the-technical-heart) has the full algorithm
if you want it.

One more thing happened quietly:

```python
f = plot(tan(x))
print(f.plan.y_range)
```

Without a chosen y-window, a single pole shooting off to 10⁷ would squash the
entire rest of the curve into a flat line at zero. MathSlate picks a window
from the middle of the data — so you see the tangent, not a horizontal smear.

---

## 6. π on the axis

Look at the tick labels here:

```python
plot(sin(x))
```

and then at these:

```python
plot(exp(x))
```

The first is labelled in multiples of π — `-2π`, `-3π/2`, `-π`, and so on —
because that is how anyone reading a trigonometric graph thinks. The second
gets ordinary numbers, because π has nothing to do with `exp`.

Nobody asked for either. MathSlate looks at what is *in* the expression.

What it will **not** do is quietly switch you to a logarithmic scale. Try:

```python
f = plot(exp(x))
print(f.notes)
```

It notices that `exp(x)` spans many orders of magnitude, and it *suggests* a
log scale — in the notes, in words. It does not apply one. A learner looking at
a log-scaled plot without knowing it is worse off than one looking at an
awkward linear plot. When you decide you want it, you ask:

```python
plot(exp(x), yscale="log")
```

---

## 7. Circles, spirals and flowers

Parametric curves you have already met:

```python
plot((cos(t), sin(t)))
```

Note the default range: for parametric curves involving trigonometry,
MathSlate uses `0` to `2π` — one full turn — rather than `-10` to `10`.

A spiral is the same idea with a growing radius:

```python
plot((t*cos(t), t*sin(t)), (t, 0, 20))
```

Then there is polar, `r = f(θ)`, and here MathSlate deliberately refuses to
guess. Written down, `1 + cos(t)` is a perfectly ordinary function of one
variable; nothing in it says whether you meant a wave or a cardioid. So you
say it:

```python
polar(1 + cos(t))
```

```python
polar(sin(3*t))
```

A cardioid and a three-petalled rose. Honesty about what cannot be inferred is
a design rule here, not an oversight — the manual calls out both places it
applies.

---

## 8. Your own numbers

Not everything is an expression. Lists of numbers work:

```python
plot([2.0, 4.0, 8.0, 16.0, 32.0])
```

Pairs of lists work as x-against-y:

```python
plot(([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0]))
```

And so do plain Python functions:

```python
import numpy as np

plot(np.tanh, (x, -5, 5))
```

Note that these all follow the same one rule from §3 — a list of numbers is
several numbers; a tuple of two lists is one x-y dataset.

---

## 9. The point of all this

Here is the feature the whole project is built around.

```python
plot(sin(x)/x).show_python()
```

You get back the plain NumPy + SymPy + Plotly program that produces the same
picture:

```text
# Equivalent code — this runs exactly as printed.
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from sympy import sin

x = sp.symbols('x', real=True)

expr = sin(x)/x
fn = sp.lambdify(x, expr, 'numpy')
...
```

Read that comment on the first line literally. It is not an illustration, not
pseudocode, and not "roughly what happens". You can paste it into a file and
run it. Every example in this tutorial and in the manual is executed by the
test suite, and so is the output of `show_python()` for every kind of plot —
if it ever stopped running, the build would fail.

This is your first `lambdify` and your first `linspace`, shown to you at the
moment you are ready to ask what they are rather than on line one of the
tutorial where they would have been noise.

Notice what it *includes*. It does not emit a naive `linspace(-10, 10, 1000)`
and hope. It emits the domain pieces, the line-breaking, the y-window — every
decision MathSlate made for you, written out as code you can now edit. Try it
on a hard one:

```python
plot(tan(x)).show_python()
```

You wanted the string rather than the printout:

```python
code = plot(sin(x)).python()
print(len(code))
```

---

## 10. Taking your work elsewhere

Nothing here is a one-way door. Every result hands you the underlying objects:

```python
f = plot(sin(x)/x)

f.plotly      # the Plotly Figure — yours to mutate
f.sympy       # the SymPy expression
f.numpy       # the sampled (x, y) arrays
```

So you can keep using MathSlate for the parts it is good at and drop into the
real library the moment you need something it does not do:

```python
f = plot(sin(x)/x)
f.plotly.update_layout(title="Now it is a Plotly problem")
```

```python
f = plot(sin(x))
xs, ys = f.numpy
print(xs.shape, ys.max().round(3))
```

```python
f = plot(sin(x)/x)
print(integrate(f.sympy, (x, 1, 2)).evalf(6))
```

That last one is worth a pause: `f.sympy` is an ordinary SymPy expression, so
the whole of SymPy applies to it. `solve`, `diff`, `integrate`, `limit`,
`series`, `simplify` and friends are all re-exported by MathSlate and are all
just SymPy:

```python
print(diff(sin(x)*exp(x), x))
print(solve(x**2 - 5*x + 6, x))
print(limit(sin(x)/x, x, 0))
```

MathSlate did not write any of those. It has no opinion about them. That is
the point — there is nothing to unlearn.

---

## 11. When MathSlate is unsure, it asks

Give it something genuinely ambiguous and see what happens:

```python raises=AmbiguousAxisError
a, b, c = symbols("a b c", real=True)
plot(a*b*c)
```

Three free symbols, none of them named like an axis, and no indication which
one is horizontal. Rather than picking one and drawing something you did not
ask for, it raises an error whose message *is* the question, and which shows
you the exact call that answers it.

```python
a, b, c = symbols("a b c", real=True)
plot(a*b*c, (a, -5, 5), parameters={b: 2, c: 3})
```

`parameters` freezes the symbols that are *not* the axis at a value. A
`slider()` does the same thing without you saying so.

When it is not ambiguous, it uses a convention: `x, y, z` before `t, u, v`
before `r, theta`, then alphabetical. So in `a*sin(x)`, `x` is the axis and `a`
is not — which is what you meant.

Two free symbols is not an error, though — it is a surface:

```python
plot(x*y)
```

and `kind="contour"` shows the same object flat. An equation draws the curve
where its two sides agree, and a three-component tuple is a space curve:

```python
plot(Eq(x**2 + y**2, 4))
```

```python
plot((cos(t), sin(t), t), (t, 0, 12))
```

---

## 12. Where your own numbers come in

Everything so far started from an expression. Real work usually starts from
measurements, and this is where the two meet:

```python
readings = dataset({"x": [0, 1, 2, 3, 4], "y": [1.0, 3.1, 4.9, 7.2, 8.9]})
c, d = symbols("c d", real=True)
found = readings.fit(c*x + d)
print(found.describe())
```

You wrote the model the way you would on paper, and what comes back is **the
same expression with its numbers filled in** — still SymPy, so everything you
learned in the last nine sections works on it:

```python
print(diff(found.expr, x))
```

`plot(readings)` draws the columns, `kind="hist"` asks about their shape
instead, and a 2×2 `Matrix` is drawn as the transformation it performs — with
its eigenvectors, the directions it only stretches.

```python
print(plot(Matrix([[2, 1], [1, 3]]), verbose=False).plan.kind)
```

## 13. Handing it to someone else

```python
from mathslate.classroom import worksheet

page = worksheet([
    "Where does sin(x)/x go at zero?",
    ("The graph", plot(sin(x)/x, verbose=False)),
    ("The numbers", table(sin(x)/x, (x, -1, 1), rows=5)),
], title="Limits")
print(len(page), "sections")
```

`page.save("handout.html")` writes one file that opens with no network and
nothing installed — sliders included.

There is also an optional assistant, `mathslate.ai`, which turns a question
into MathSlate code. It needs an API key and one of three provider packages;
MathSlate itself needs none of that and never reaches for it.

---

## 14. Coming from Mathematica?

If your instinct for "how do I plot this" was trained on Wolfram Language,
here is the same handful of tasks in both. The underlying idea is identical —
you write the mathematics, the tool infers the picture — so most of what
transfers is instinct, not syntax.

| Task | Mathematica | MathSlate |
|---|---|---|
| A curve | `Plot[Sin[x], {x, 0, 2 Pi}]` | `plot(sin(x))` — range is optional, inferred |
| Several curves overlaid | `Plot[{Sin[x], Cos[x]}, {x, a, b}]` | `plot([sin(x), cos(x)])` |
| A parametric curve | `ParametricPlot[{Cos[t], Sin[t]}, {t, 0, 2 Pi}]` | `plot((cos(t), sin(t)))` |
| Polar | `PolarPlot[1 + Cos[t], {t, 0, 2 Pi}]` | `polar(1 + cos(t))` |
| A surface | `Plot3D[x*y, {x, a, b}, {y, c, d}]` | `plot(x*y)` — inferred from two free symbols |
| An implicit curve | `ContourPlot[x^2 + y^2 == 4, {x, a, b}, {y, c, d}]` | `plot(Eq(x**2 + y**2, 4))` |
| Symbol declaration | none needed — `x` is symbolic by default | `x, y, z, t, n, k, theta` are predeclared; anything else: `symbols("a")` |
| Solve | `Solve[x^2 - 4 == 0, x]` | `solve(x**2 - 4, x)` — the same SymPy function |
| Differentiate | `D[Sin[x], x]` | `diff(sin(x), x)` |
| Integrate | `Integrate[Sin[x], x]` | `integrate(sin(x), x)` |
| Limit | `Limit[Sin[x]/x, x -> 0]` | `limit(sin(x)/x, x, 0)` |
| Properties of a function | read off the graph, or several separate calls (`Solve`, `D`, `Limit`, …) | `analyze(sin(x)/x)` — roots, extrema, asymptotes, symmetry, in one call |
| An interactive parameter | `Manipulate[Plot[a*Sin[x], {x, 0, 2Pi}], {a, 1, 3}]` | `a = slider(1, 3); plot(a*sin(x))` — axis vs. parameter is inferred, not declared |
| Fitting data | `FindFit[data, model, pars, x]` | `dataset(data).fit(model)` — returns the model as a SymPy expression, params filled in |
| Function call syntax | square brackets, capitalized: `Sin[x]` | parentheses, lowercase — ordinary Python: `sin(x)` |
| Indexing | 1-based, `list[[1]]` | 0-based, `list[0]` — ordinary Python |
| Sharing interactive work | a `.nb` notebook, or CDF/Player for read-only viewing | `worksheet(...).save("handout.html")` — one file, any browser, no viewer needed |
| Seeing "what it actually computed" | plot options change the picture; no code is shown | `show_python()` prints the exact NumPy/SymPy/Plotly that drew it |
| Cost and openness | commercial license | free; MIT license |
| What you learn transfers to | Wolfram Language only | Python, NumPy, SymPy and Plotly — usable everywhere, forever |

Three differences are worth more than a row each.

**Nothing here is a MathSlate-only skill.** `sin`, `solve`, `diff`, `integrate`
and `limit` above are not MathSlate functions with Mathematica-like names —
they *are* SymPy, imported unchanged. Learn them here and you already know
them in any Python project, ten years from now, with no translation step.

**MathSlate infers less than you might expect, on purpose.** Mathematica's
`Plot` silently drops branches it cannot handle and needs `Exclusions -> None`
or `Exclusions -> Automatic` tuned by hand for a genuinely hard function.
`plot(tan(x))` cuts every pole correctly with no option at all (§5) — but ask
for something inference cannot resolve, like two equally axis-like symbols,
and MathSlate raises `AmbiguousAxisError` naming the exact call that answers
it (§11), rather than guessing and drawing the wrong thing.

**Every escape hatch is real.** `f.plotly` is a genuine `plotly.graph_objects`
figure, `f.numpy` is genuine arrays, `f.sympy` is a genuine SymPy expression —
none of them wrapped, none of them requiring MathSlate to keep working. If
MathSlate does not do what you need, you are already holding the tool that
does, with zero conversion.

---

## Where to go next

- The [reference manual](manual.md) — every option, the full dispatch
  contract, the sampling algorithm, and the exact guarantees.
- The [PRD](../mathslate_prd_0.3.md) — why the project is shaped this way,
  and what it deliberately refuses to do.
- `examples/quickstart.py` — everything in this tutorial as one runnable file.

One thing that is **not** coming, ever: step-by-step worked solutions.
MathSlate shows you what a function *is* and what its properties *are*. It does
not narrate the algebra, because there is no honest way to do that for
arbitrary expressions and a dishonest one would be worse than nothing.
