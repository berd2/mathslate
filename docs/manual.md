# MathSlate Reference Manual

**Version 0.1** · complete description of the public API, the inference rules,
and the sampling algorithm.

For a guided introduction, read the [tutorial](tutorial.md) first. This
document is organised for lookup, not for reading front to back.

Every `python` block below is executed by the test suite
(`tests/test_docs.py`), in the order it appears, in one shared namespace that
starts with `from mathslate import *`.

---

## Contents

1. [Installation and requirements](#1-installation-and-requirements)
2. [The public surface](#2-the-public-surface)
3. [The dispatch contract](#3-the-dispatch-contract)
4. [`plot()`](#4-plot)
5. [Symbol-to-axis binding](#5-symbol-to-axis-binding)
6. [Sampling: the technical heart](#6-sampling-the-technical-heart)
7. [Axes](#7-axes)
8. [`PlotResult`](#8-plotresult)
9. [`show_python()`](#9-show_python)
10. [`analyze()`](#10-analyze)
11. [`slider()` and `animate()`](#11-slider-and-animate)
12. [`table()` and HTML export](#12-table-and-html-export)
13. [`dataset()`, statistics and linear algebra](#13-dataset-statistics-and-linear-algebra)
14. [Errors](#14-errors)
15. [Frontend adapters](#15-frontend-adapters)
16. [Milestones](#16-milestones)
17. [Internals map](#17-internals-map)
18. [Guarantees and non-guarantees](#18-guarantees-and-non-guarantees)

---

## 1. Installation and requirements

```bash
pip install mathslate
```

Python 3.10 or newer. Runtime dependencies are `sympy`, `numpy` and `plotly`,
and nothing else — installing MathSlate never pulls in a notebook frontend.
This is a tested property, not an intention.

Optional extras:

| Extra | Installs | For |
|---|---|---|
| `mathslate[jupyter]` | `ipywidgets`, `anywidget` | Jupyter and Colab widgets |
| `mathslate[marimo]` | `marimo` | marimo reactive widgets (v0.5) |
| `mathslate[ai-gemini]` | `google-genai`, `keyring` | Gemini assistant and remembered local API keys |
| `mathslate[starter]` | Jupyter Lab, notebook widgets, Gemini and `keyring` | complete local learning setup |
| `mathslate[dev]` | `pytest` | running the test suite |

For a first local Jupyter setup, install everything needed for interactive
plots, the Gemini assistant, and persistent local keys in one command:

```bash
pip install "mathslate[starter]"
```

The repository's `scripts/quickstart.ps1` (Windows) and
`scripts/quickstart.sh` (macOS/Linux) create an isolated environment, run this
installation, write a starter notebook and open Jupyter Lab. `anywidget` and
`keyring` are included automatically; users never import them directly.

```python
import mathslate

print(mathslate.__version__)
```

### Importing

`from mathslate import *` is supported **except in marimo** (see below), and is
the intended usage elsewhere, because the predefined symbols (`x`, `y`, `z`, …)
have to come from somewhere. This is a deliberate exception to the usual Python
advice; the compensating rule is that `show_python()` always emits the explicit
`x = sp.symbols('x', real=True)`, so the star import never hides anything from
you permanently.

The namespaced form works identically, and everywhere:

```python
import mathslate as ms

f = ms.plot(ms.sin(ms.x))
```

#### marimo forbids `import *`

marimo rejects the star import at **parse** time, so the cell is not run at
all:

```text
line 1 SyntaxError: Importing symbols with `import *` is not allowed in marimo.
```

This is marimo's rule, not MathSlate's, and there is no flag that disables it.
marimo is a reactive notebook: it determines statically which names each cell
defines and uses, in order to build the dependency graph that decides what
re-runs when. `import *` makes the set of defined names undecidable without
executing the module, so it cannot be admitted.

Use an explicit import in your first cell. This is the line the tutorial gives,
and it is checked against marimo's own parser by `tests/test_marimo_imports.py`:

```python
from mathslate import (
    plot, polar, analyze, show_python, set_verbose, get_verbose,
    slider, animate, table, dataset, frontend_report,
    sin, cos, tan, atan, exp, log, sqrt, Abs, floor, pi,
    diff, integrate, limit, solve, solveset, nsolve, simplify, symbols,
    factor, expand, apart, cancel, together, trigsimp, gcd,
    Eq, Integer, Rational, Matrix,
    x, y, z, t, n, k, theta,
)
```

Nothing else in this manual changes; only the import line does.

| Environment | `import *` | Explicit import | `import mathslate as ms` |
|---|---|---|---|
| Jupyter | ✅ | ✅ | ✅ |
| Colab | ✅ | ✅ | ✅ |
| plain Python | ✅ | ✅ | ✅ |
| marimo | ❌ parse error | ✅ | ✅ |

---

## 2. The public surface

MathSlate introduces **19 public names**, against a budget of 20 (PRD §20).
Everything else it exposes is SymPy, re-exported unchanged.

### New in MathSlate

| Name | Status | Purpose |
|---|---|---|
| `plot()` | v0.1 | unified dispatch — §4 |
| `polar()` | v0.1 | `r = f(θ)`, which inference cannot detect — §4.4 |
| `show_python()` | v0.1 | equivalent plain Python — §9 |
| `set_verbose()` / `get_verbose()` | v0.1 | the one-line inference report — §4.6 |
| `frontend_report()` | v0.1 | detected environment — §15 |
| `PlotResult` | v0.1 | the type every `plot()` returns — §8 |
| `analyze()` | v0.5 | properties of an expression — §10 |
| `Analysis` | v0.5 | the type `analyze()` returns — §10 |
| `slider()` | v0.5 | a parameter the reader can vary — §11 |
| `animate()` | v0.5 | the same, with a play button — §11 |
| `table()` | v0.5 | tabulated values — §12 |
| `Table` | v0.5 | the type `table()` returns — §12 |
| `dataset()` | v1.0 | the bridge from symbolic to data — §13 |
| `Dataset` | v1.0 | the type `dataset()` returns — §13 |
| `set_range_controls()` / `get_range_controls()` | v1.5 | whether a bare `plot()` shows `range_controls()` by default — §4.13, §21 |
| `set_plot_size()` / `get_plot_size()` | v1.6 | the size every later plot takes — §4.12b |

```python
import mathslate

print(len(mathslate.NEW_API))
```

### Re-exported from SymPy

These are **the SymPy functions themselves**, not wrappers:

```python
import sympy, mathslate

print(mathslate.solve is sympy.solve, mathslate.diff is sympy.diff)
```

`solve`, `simplify`, `expand`, `factor`, `apart`, `cancel`, `together`,
`trigsimp`, `nsimplify`, `diff`, `integrate`, `limit`, `series`, `solveset`,
`nsolve`, `lambdify`, `Matrix`, `Symbol`, `symbols`, `Function`, `Piecewise`,
`Eq`, `Rational`, `Integer`, `Float`, `Sum`, `factorial`, `binomial`, `gcd`,
`sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `asin`, `acos`, `atan`, `sinh`,
`cosh`, `tanh`, `exp`, `log`, `sqrt`, `cbrt`, `root`, `Abs`, `sign`, `floor`,
`ceiling`, `pi`, `E`, `I`, `oo`.

The full SymPy library is also reachable as `mathslate.sympy`, so a function not
in the list above is one attribute away rather than out of reach.

#### Solving and algebra

MathSlate does not compute; SymPy does (PRD §2.2). The point of re-exporting the
engine rather than wrapping it is that these are the real functions, they behave
exactly as SymPy documents, and whatever they return flows straight back into a
plot. This is a map of what is there, not a reimplementation of SymPy's own
reference.

**Solving.** `solve()` reads a bare expression as `= 0`, or takes an `Eq`; a
list of equations with a list of unknowns is a system:

```python
print(solve(x**2 - 5*x + 6, x))              # [2, 3]
print(solve(Eq(x**2, 2), x))                 # [-sqrt(2), sqrt(2)]
print(solve([x + y - 3, x - y - 1], [x, y])) # {x: 2, y: 1}
```

`solveset()` returns the entire solution set, including every branch of a
periodic equation; `nsolve()` finds one root numerically from a starting guess,
for equations with no closed form:

```python
print(solveset(sin(x), x))          # 2*n*pi and 2*n*pi + pi over the integers
print(nsolve(x - cos(x), x, 0.5))   # 0.739085133215161
```

**Algebra.** The polynomial and rational verbs, and the trig identities:

| Function | Does |
|---|---|
| `factor(e)` | factor over the rationals |
| `expand(e)` | multiply out |
| `gcd(a, b)` | greatest common divisor of two polynomials |
| `apart(e)` | partial-fraction decomposition |
| `cancel(e)` | reduce a ratio to lowest terms |
| `together(e)` | combine over a common denominator |
| `trigsimp(e)` | simplify using trig identities |
| `simplify(e)` | the general-purpose simplifier |

```python
print(factor(x**3 - x))              # x*(x - 1)*(x + 1)
print(expand((x + 1)**3))            # x**3 + 3*x**2 + 3*x + 1
print(gcd(x**2 - 1, x**2 - x))       # x - 1
print(apart(1/(x**2 - 1)))           # -1/(2*(x + 1)) + 1/(2*(x - 1))
print(trigsimp(sin(x)**2 + cos(x)**2))  # 1
```

Because the result is an ordinary expression, there is never a conversion step
between algebra and picture:

```python
model = factor(x**3 - 6*x**2 + 11*x - 6)
print(model, "→ roots", solve(model, x))
_ = plot(model, (x, 0, 4), verbose=False)
```

What is deliberately **not** here is a step-by-step derivation of any of these —
`solve()` gives the answer, not the working. That is the one scope line MathSlate
holds (§10.4, PRD §2.2); SymPy has no general worked-solution engine, and a
partial one would mislead more than it taught.

### Predefined symbols

`x`, `y`, `z`, `t`, `n`, `k`, `theta` — all created with `real=True`, which is
what makes domain and singularity analysis give textbook answers rather than
complex-plane ones.

```python
print(x.is_real, theta.name)
```

Nothing stops you making your own; they are ordinary SymPy symbols.

```python
u, v = symbols("u v", real=True)
f = plot(u**2, (u, -3, 3))
```

---

## 3. The dispatch contract

`plot()` infers what you meant. The mapping is a **documented contract**, not
an implementation detail — it will not change without a version bump.

> **The one rule.** A `list` means *several things together*.
> A `tuple` means *one vector-valued object*.

| Input form | Free symbols | Result | Status |
|---|---|---|---|
| `Expr` | 1 | 2D curve | v0.1 |
| `Expr` | 0 | horizontal line + message | v0.1 |
| `list[Expr]` | 1 shared | curves overlaid | v0.1 |
| `tuple[Expr, Expr]` | 1 shared | 2D parametric curve | v0.1 |
| `callable` | — | numeric sampling | v0.1 |
| array-like | — | data series | v0.1 |
| `(xdata, ydata)` | — | scatter / line | v0.1 |
| `Expr` | 2 | surface, contour toggle | v0.5 |
| `Eq(lhs, rhs)` | 2 | implicit curve | v0.5 |
| `tuple[Expr × 3]` | 1 shared | 3D space curve | v0.5 |
| `tuple[Expr × 3]` | 2 shared | parametric surface | v0.5 |
| `Dataset` | — | its columns, first one across | v1.0 |
| `Matrix` (2×2) | — | the transformation it performs | v1.0 |
| inequality | 2 | filled region | v1.5 |
| inequality | 1 | shaded bands on the axis | v1.5 |

Every row is implemented. A dispatch row is never silently mis-drawn as
something else.

### The kinds

`result.plan.kind` reports which row matched:

| `kind` | Meaning |
|---|---|
| `"curve"` | one expression, one free symbol |
| `"curves"` | several expressions overlaid |
| `"constant"` | expression with no free symbols |
| `"parametric"` | `(x(t), y(t))` |
| `"polar"` | `r = f(θ)`, via `polar()` |
| `"callable"` | a plain Python function |
| `"data"` | arrays of numbers |
| `"surface"` | one expression, two free symbols |
| `"contour"` | the same, flat — `kind="contour"` |
| `"implicit"` | the zero level of `Eq(lhs, rhs)` |
| `"space"` | `(x(t), y(t), z(t))` |
| `"hist"`, `"box"` | the shape of a column — `kind="hist"` |
| `"linalg"` | a 2×2 matrix as a transformation |
| `"psurface"` | `(x(u,v), y(u,v), z(u,v))` |
| `"region"` | where an inequality holds, in the plane — §4.10 |
| `"band"` | where an inequality holds, on the axis — §4.10 |

```python
print(plot(sin(x)).plan.kind)
print(plot([sin(x), cos(x)]).plan.kind)
print(plot((cos(t), sin(t))).plan.kind)
print(plot(Integer(3)).plan.kind)
print(plot([1.0, 2.0, 3.0]).plan.kind)
```

### Inputs accepted for an expression

`Expr`, a `str` (passed through `sympify`), or a plain number.

```python
print(plot("sin(x)/x").plan.kind)
print(plot(2.5).plan.kind)
```

### What inference cannot decide, ever

Two cases are ambiguous *in principle*, so MathSlate requires you to say which
you meant rather than guessing:

- **Polar.** `r = f(θ)` is written identically to `y = f(x)`. Use
  [`polar()`](#44-polar).
- **Surface versus contour.** Both are correct pictures of the same object.
  The default is surface; `kind="contour"` switches (v0.5).

---

## 4. `plot()`

```text
plot(obj, *ranges, polar=False, kind=None, label=None, title=None,
     yscale=None, show_legend=None, points=None, exclusions=None,
     mesh=True, ticks=None, width=None, height=None,
     xlim=None, ylim=None, zlim=None, controls=None,
     verbose=None, parameters=()) -> PlotResult
```

### 4.1 `obj` and `*ranges`

`obj` is anything in the dispatch table. Each range is a
`(symbol, lo, hi)` tuple.

```python
f = plot(sin(x), (x, 0, 6.28))
print(f.plan.param_range)
```

A malformed range is rejected immediately rather than misread. These raise the
ordinary Python exceptions, not `MathSlateError` — a badly shaped argument is a
programming mistake, not a mathematical one:

```python raises=TypeError
plot(sin(x), (x, 1.0))
```

```python raises=ValueError
plot(sin(x), (x, 1.0, 1.0))
```

### 4.2 `label`, `title`, `show_legend`

```python
f = plot(sin(x), label="the sine", title="A wave", show_legend=True)
print(f.plotly.data[0].name, "/", f.plotly.layout.title.text)
```

`show_legend` defaults to `True` when more than one curve is drawn.

### 4.3 `kind`

Overrides how samples are drawn.

| Value | Effect |
|---|---|
| `None` | infer: markers for small data series, lines otherwise |
| `"line"`, `"curve"` | draw as a connected line |
| `"scatter"` | draw as markers |
| `"surface"` | draw as a 3D surface (the default for two free symbols) |
| `"contour"` | draw the same object flat — §4.11 |
| anything else | `UnsupportedInputError` |

```python
print(plot(sin(x), kind="scatter").plotly.data[0].mode)
print(plot([1.0, 2.0, 3.0], kind="line").plotly.data[0].mode)
```

```python raises=UnsupportedInputError
plot(sin(x), kind="bar")
```

### 4.4 `polar` and `polar()`

`polar(expr, ...)` is `plot(expr, polar=True, ...)` and accepts everything
`plot()` does. The expression is read as the radius `r` and the axis symbol as
the angle.

```python
f = polar(1 + cos(t))
print(f.plan.kind, f.plan.param_range[1].__round__(3))
```

Polar and parametric plots get an equal-aspect y-axis, so a circle is round.

### 4.5 `yscale`

`None` (default), `"linear"`, or `"log"`. Anything else is rejected.

```python
print(plot(exp(x), yscale="log").plotly.layout.yaxis.type)
```

```python raises=UnsupportedInputError
plot(exp(x), yscale="semilog")
```

A log scale is **never applied automatically**, only suggested — see §7.2.
Setting `yscale="log"` also disables y-clipping (§6.5), because the two are
solving the same problem.

A log axis cannot show zero or negative values, and Plotly drops them without
comment. The option is still honoured — you asked for it — but the consequence
is stated in the notes rather than left as a half-drawn curve:

```python
print([n for n in plot(sin(x), yscale="log").notes if "cannot show" in n])
```

### 4.6 `verbose`

Per-call override of the global setting. `None` follows
`set_verbose()`; `True` or `False` forces it.

```python
set_verbose(False)
f = plot(sin(x))          # prints nothing
print(get_verbose())
set_verbose(True)
```

The report is also available without printing:

```python
print(plot(tan(x), verbose=False).summary())
```

Format:

```text
<kind> | <symbol> ∈ [<lo>, <hi>] | <n> samples | <n> discontinuities handled
```

The last field appears only when there were any. A further
`| element-wise evaluation` appears if the vectorised fast path failed (§6.6).

Notes — domain warnings, singularity lists, the log-scale suggestion — print
below the summary line and are readable as `result.notes`.

### 4.7 `points`

Sets the starting number of uniform samples before adaptive refinement.
Default 200. Raising it also raises the point cap if needed. Fewer than 2
points cannot describe a line, so it is rejected rather than rounded up to the
default behind your back.

```python
print(plot(sin(x), points=1000).plan.total_points >= 1000)
```

```python raises=UnsupportedInputError
plot(sin(x), points=0)
```

On a two-variable plot — a surface, contour, implicit curve, region or
parametric surface — it means samples **per axis** of the grid instead:

```python
print(plot(x*y, points=30, verbose=False).plan.series[0].sample.z.shape)   # (30, 30)
```

That is a different scale from a curve's on purpose, and it is capped: 2000
points along a line is 2000 evaluations, and a 2000×2000 grid is four million.
Left unset, each kind keeps its own default — 60 per axis for a surface, 200
for a region, which is higher because a region is judged by its boundary and a
surface by its interior.

```python
print(plot(x*y, verbose=False).plan.series[0].sample.z.shape)              # (60, 60)
print(plot(x**2 + y**2 < 1, verbose=False).plan.series[0].sample.z.shape)  # (200, 200)
```

§4.13's `Samples` slider is this number, live.

### 4.8 `exclusions`

Automatic discontinuity detection (§6) is what makes `plot(tan(x))` right where
a plotting wrapper is wrong. But detection with no override is a black box the
moment it disagrees with you, so there is a way to say otherwise. Mathematica
calls the same argument `Exclusions`.

**Name places to cut**, as well as wherever detection already cuts:

```python
cut = plot(sin(x), (x, -6, 6), exclusions=[0, pi/2], verbose=False)
print(sorted(cut.plan.series[0].sample.breakpoints))
```

Symbolic values are welcome — `exclusions=[pi/2]` is the natural thing to write
— and a place outside the window is ignored rather than refused, because
narrowing the range after naming a cut is not a mistake.

**Or stop the numeric probe entirely** with `exclusions=False`, for a curve that
is genuinely continuous but steep enough to look otherwise:

```python
print(len(plot(floor(x), (x, -3, 3), verbose=False).plan.series[0].sample.breakpoints))
print(len(plot(floor(x), (x, -3, 3), exclusions=False, verbose=False).plan.series[0].sample.breakpoints))
```

`False` switches off **only** the bisection probe of §6.4. Symbolic
singularities and the real domain still apply, because those are SymPy's answer
rather than a guess:

```python
poles = plot(tan(x), (x, -4, 4), exclusions=False, verbose=False)
print(len(poles.plan.series[0].sample.breakpoints))
```

`exclusions=True` is rejected: detection is already on, so it would mean
nothing, and an argument that silently does nothing is a bug.

```python raises=UnsupportedInputError
plot(sin(x), exclusions=True)
```

Both forms reach `show_python()`, so the emitted code cuts the line in the same
places the figure does.

### 4.9 `parameters`

Symbols that are *not* axes. Accepts a mapping of symbol to value (the symbol
is frozen at that value), or a bare sequence of symbols (excluded from axis
selection only).

```python
a = symbols("a", real=True)
f = plot(a*sin(x), parameters={a: 3.0})
print(f.plan.symbol.name, round(float(f.numpy[1].max()), 3))
```

The bare-sequence form removes a symbol from the axis candidates but does not
give it a value, so if it is still in the expression there is nothing to
evaluate. That is refused, with the call that fixes it:

```python raises=UnsupportedInputError
plot(a*sin(x), parameters=[a])
```

A symbol is either an axis or a parameter. Asking for both is a contradiction,
not a preference:

```python raises=UnsupportedInputError
plot(a*sin(x), (a, -1, 1), parameters={a: 2})
```

This is the binding point `slider()` will use in v0.5, where the slider
supplies the value the bare-sequence form is missing.

### 4.10 Inequalities: regions and bands

`Eq(lhs, rhs)` asks where two expressions are **equal**, which is a curve, and
§4.11 draws it. `lhs < rhs` asks where one **exceeds** the other, which is an
area — a different question, so a different picture. How many free symbols there
are decides which one, exactly as it does for an ordinary expression:

```python
plot(x**2 + y**2 < 1)          # two symbols → the filled unit disc
plot(sin(x) > 0, (x, -6, 6))   # one symbol  → shaded intervals of x
```

`And`, `Or` and the `&` / `|` operators compose them, and the whole compound is
evaluated in one pass — `lambdify` prints `And` as `numpy.logical_and`:

```python
print(plot((x**2 + y**2 < 4) & (y > x), verbose=False).plan.kind)
```

**Where the relation has no truth value, nothing is claimed.** This is subtler
than it looks: a comparison against `NaN` is *false*, in NumPy and in plain
Python alike, so `sqrt(x*y) > 1` would otherwise report the two quadrants where
`x*y < 0` as *outside the region* — a definite statement about a place where the
inequality means nothing. Those points are left blank instead, and counted:

```python
print([n for n in plot(sqrt(x*y) > 1, verbose=False).notes if "truth value" in n])
```

#### One variable: the solution set, and the curve that explains it

With one free symbol you get the intervals where the inequality holds, shaded,
**plus the curve of `lhs - rhs`**. The strip says where the answer is; the curve
crossing zero at the same places says why.

```python
answer = plot(x**2 - 4 < 0, (x, -4, 4), verbose=False)
print(answer.plan.bands, answer.plan.bands_exact)
```

`bands_exact` is the same honesty rule as everywhere else: `True` means
`solveset` solved the inequality, `False` means the intervals were sampled and
their edges refined by bisection.

**`solveset` is checked, not trusted.** For a periodic inequality it returns the
principal period *and does not say so*: `solveset(sin(x) > 0, x, Interval(-6, 6))`
is `(0, π)`, silently missing `(-6, -π)`. The range argument does not prevent it.
So the closed form is only used when sampling cannot find a solution outside it —
otherwise a confident wrong answer would reach the screen, which is the one
failure mode this library must not have.

```python
periodic = plot(sin(x) > 0, (x, -6, 6), verbose=False)
print(len(periodic.plan.bands), periodic.plan.bands_exact)
```

A compound inequality over a single variable is refused rather than guessed at —
give it two free symbols to draw as a region, or plot the parts one at a time.

```python raises=UnsupportedInputError
plot((x > 0) & (x < 1), (x, -2, 2))
```

### 4.11 Two variables: surfaces, contours and implicit curves

One expression with two free symbols is a surface (PRD §5.1), and
`kind="contour"` shows the same object flat. Both read the same grid, so the
toggle changes the picture and nothing else.

```python
print(plot(x*y).plan.kind, plot(x*y, kind="contour").plan.kind)
```

An equation is the curve where its two sides agree:

```python
print(plot(Eq(x**2 + y**2, 4)).plan.kind)
```

A three-component tuple is a space curve over one parameter and a parametric
surface over two:

```python
print(plot((cos(t), sin(t), t), (t, 0, 12)).plan.kind)
print(plot((cos(t)*cos(theta), cos(t)*sin(theta), sin(t)),
           (t, -1.5, 1.5), (theta, 0, 6.28)).plan.kind)
```

The default window is `(-5, 5)` on each axis, not the `(-10, 10)` a curve gets:
a 60×60 grid over the wider range resolves far less than 200 points along a
line. Both windows are reported.

```python
print(plot(x*y, verbose=False).summary())
```

**What two variables cannot promise.** §6.1 and §6.2 ask SymPy exactly where a
curve is real and where it blows up. Neither question has a usable answer in
two variables — `continuous_domain` takes a single symbol, and there is no
two-variable `singularities`. So a surface is evaluated on a grid, anything
non-real becomes a hole rather than a wrong value, and the z range is clipped on
the same principle as §6.5. That is less than a curve gets, and the notes say so
rather than implying otherwise:

```python
print([n for n in plot(sqrt(x*y), verbose=False).notes if "real number" in n])
```

The clip bounds **the z axis as well as the colour scale**. Clipping only the
colours leaves the geometry unchanged, so a pole 3481 units tall on a surface
whose features live within 14 is still drawn as a 3481-unit wall: the camera
zooms out to contain it, and everything the plot was made to show flattens into
a sheet at the bottom. Both come from the same range, so the emitted code
carries it too:

```python
poles = plot(1/(x*y), verbose=False)
print(poles.plan.series[0].sample.z_range)
print(tuple(poles.plotly.layout.scene.zaxis.range))
```

A surface that needs no clipping keeps an automatic axis — bounding one that
did not ask for it would crop the picture:

```python
print(plot(x + y, verbose=False).plotly.layout.scene.zaxis.range)
```

### 4.12 `mesh` and the view window (`xlim`, `ylim`, `zlim`)

Two controls for how a plot *looks*, as opposed to what it contains.

**`mesh`** rules a 3D surface with grid lines, the way Mathematica's `Plot3D`
does — a bare colour gradient reads as flatter than the surface is. On by
default; `mesh=False` returns the smooth look. Flat kinds (a curve, a contour, a
region) have no surface to rule and ignore it.

```python
print(plot(x*y, verbose=False).plotly.data[0].contours.x.show)          # True
print(plot(x*y, mesh=False, verbose=False).plotly.data[0].contours.x.show)  # None
```

`mesh` also takes a number instead of `True`, naming how many lines to draw
per axis — Plotly's own automatic spacing (what `True` used before this)
picks a "nice round number" the way an axis chooses tick marks, which reads
as sparse next to the actual sample grid:

```python
print(plot(x*y, verbose=False).plotly.data[0].contours.x.size)          # the default spacing
print(plot(x*y, mesh=40, verbose=False).plotly.data[0].contours.x.size)  # denser
```

The spacing is computed from the surface's own extent, so `mesh=40` means
forty lines whether the domain spans 2 units or 2000 — a fixed spacing would
not.

**`xlim`, `ylim`, `zlim`** set the view window. This is not the domain: the range
argument `(x, -10, 10)` decides where the function is *sampled*, while these
decide what the axis *shows*. They differ whenever they should — to undo the
automatic y-clip and look straight at a pole, to zoom, or to line two figures up
on one scale.

```python
zoomed = plot(sin(x), (x, -10, 10), xlim=(-3, 3), verbose=False)
print(tuple(zoomed.plotly.layout.xaxis.range))           # (-3.0, 3.0)
print(zoomed.plan.series[0].sample.x.min() < -9)         # still sampled wide: True
```

`ylim` overrides the automatic percentile clip (§6.5), which is how you ask to
see a pole the clip would otherwise hide; `zlim` does the same for a surface's
z axis:

```python
_ = plot(tan(x), (x, -4, 4), ylim=(-50, 50), verbose=False)
_ = plot(1/(x*y), zlim=(-10, 10), verbose=False)
```

Each is a `(low, high)` pair with `low < high`, and each is reproduced by
`show_python()`. A malformed window is refused rather than guessed at:

```python raises=UnsupportedInputError
plot(sin(x), xlim=(5, 1))
```

### 4.12a `ticks` — how many labels an axis carries

Left alone, the count is Plotly's. `ticks=n` caps it at `n` per axis, `ticks=False`
removes the labels, and `ticks=None` (the default) restores the automatic choice.
The number is a **ceiling, not a target**: Plotly still picks round positions, it
just stops before it passes that many.

```python
print(plot(exp(x), ticks=5, verbose=False).plotly.layout.xaxis.nticks)          # 5
print(plot(exp(x), ticks=False, verbose=False).plotly.layout.xaxis.showticklabels)  # False
```

On a 3D plot it reaches all three scene axes, and there it matters more than it
does in 2D. Plotly re-lays a flat axis's ticks against its pixel length every
time the view changes; a scene's labels are positioned in the projection and
nothing recomputes them, so they crowd as the camera comes in. This is the only
lever for that:

```python
print(plot(x*y, ticks=4, verbose=False).plotly.layout.scene.zaxis.nticks)  # 4
```

Fewer than three leaves the axis without a readable scale — two labels are its
endpoints and nothing between them — so it is refused rather than drawn:

```python raises=UnsupportedInputError
plot(sin(x), ticks=2)
```

In 2D you rarely need it: zooming re-labels itself (§4.13).

### 4.12b `width` and `height` — how big the figure is

A plot is `DEFAULT_HEIGHT` (520px) tall unless it is told otherwise. Plotly's
own default is 450, which is a dashboard tile's height; a notebook cell is the
full width of the page and the graph is the thing being read, and once the
sidebar of §4.13 takes its fifth of the width, 450 reads as a letterbox.

```python
from mathslate.render.options import DEFAULT_HEIGHT
print(plot(sin(x), verbose=False).plotly.layout.height == DEFAULT_HEIGHT)  # True
print(plot(sin(x), height=800, verbose=False).plotly.layout.height)        # 800
```

**Width is deliberately unset by default.** Plotly reads an unset width as
"measure the container", which is what lets a figure fill its cell; a pixel
width leaves a gap beside it on a wide screen and clips it on a narrow one.
Set it when you want a fixed size and mean it:

```python
print(plot(sin(x), verbose=False).plotly.layout.width)                 # None
print(plot(sin(x), width=1000, verbose=False).plotly.layout.width)     # 1000
```

`set_plot_size()` moves the default for every later plot, which is the version
you want at the top of a notebook rather than on every cell. Naming one
dimension leaves the other alone, and `reset=True` restores both:

```python
from mathslate import set_plot_size, get_plot_size

set_plot_size(height=720)
print(plot(sin(x), verbose=False).plotly.layout.height)         # 720
print(plot(sin(x), height=300, verbose=False).plotly.layout.height)  # 300 — a call still wins
set_plot_size(reset=True)
print(get_plot_size())                                          # (None, 520)
```

Both are reproduced by `show_python()`: a figure and a program said to build it
must not come out different sizes.

### 4.12c `controls` — turning the sidebar off for one plot

The range-control sidebar (§4.13) earns its fifth of the width on a curve you
are exploring, and not on one you are only looking at. `controls=False` gives
the plain figure back for this plot; `controls=True` asks for it even when
`set_range_controls(False)` has turned it off notebook-wide.

```text
plot(sin(x), controls=False)     # the whole cell width is the graph
plot(sin(x))                     # figure + sidebar, the default
```

It declines the *default display*, not the method: `.range_controls()` called
by name still builds the widget. And it changes nothing about the figure, so
`show_python()` emits the same program either way — where a plot is shown is
not part of what it is.

Use `set_range_controls(False)` when you want that for the whole notebook;
`controls=` is the per-plot override of it.

### 4.13 Live range controls — moving the window after the figure exists

`xlim`/`ylim` choose a window once, and Plotly's own drag-to-zoom only crops
the points already there — narrow into a tenth of a wide domain and you see a
tenth of the original point density, not a closer look. `range_controls()`
resamples instead: moving X or Y calls `plot()` again over the new window, so
narrowing draws at full resolution.

In Jupyter or Colab, with `pip install mathslate[jupyter]` (`ipywidgets` and,
for Plotly ≥ 6's `FigureWidget`, `anywidget` — both come with that one
extra), this is what a bare `plot(...)` shows by default at a cell's end —
no method call needed:

```text
plot(sin(x)/x, (x, -10, 10))          # the figure, with x/y min·max boxes
                                        # beside it; moving one resamples
```

The sidebar also has `Auto Y`, `Reset`, X/Y `in`/`out` buttons, and two
sliders: `Ticks` — §4.12a's `ticks=`, live, with the bottom of its track
meaning "leave the count to Plotly" — and `Samples`, which is §4.7's `points=`.
On a true 3D surface, `Auto Y` becomes `Auto Z`, and Z `in`/`out` buttons
appear too.

`Samples` is worth dragging *down* as well as up. Coarsening a curve to 50
shows the adaptive pass its own scaffolding — where it chose to put extra
points and where it did not — which is what makes "adaptive" something you can
see rather than something this manual asserts. On a two-variable plot the
slider moves the grid's samples per axis instead, on its own smaller scale.
Whatever you choose survives the next X or Y edit, so a window move does not
quietly put the density back.

Zooming re-labels the horizontal axis, which is the other half of that control
and the half you should not have to think about. Ticks chosen once for the
initial domain are wrong as soon as the window moves: a π axis zoomed into a
third of a period keeps the one label that still falls inside, and a numeric
one grows a digit per decade of zoom while Plotly goes on asking for the same
dozen ticks, until they overlap. So the choice is re-made from the window on
screen — π ticks re-fitted to a finer or coarser multiple, dropped for plain
numbers below about a quarter-period where no multiple of π is worth showing,
and the numeric count thinned to what its own labels have room for. A 3D scene
cannot do this (nothing there recomputes on a camera move), which is why
`ticks=` exists for it.

`Auto Y` fits the vertical window to the X range currently in the boxes: the
automatic clip where a pole makes one necessary, and otherwise the extent of
the values actually drawn. It clears an `ylim` you passed to `plot()` rather
than fitting inside it — asking to auto-fit is asking for a different window
than the one you fixed. The boxes always hold the axis's own units, which on
`yscale="log"` means powers of ten, the same convention `ylim` follows there.

Call it explicitly to keep the widget from a result you printed with
`verbose=False`, or from a plot with view controls already set —
`width=`/`height=` set the figure's own pixel size the same way
`.plotly.update_layout(width=..., height=...)` always could, in the one place
you're already looking to size this particular widget:

```text
result = plot(sin(x)/x, (x, -10, 10), verbose=False)
result.range_controls(width=800, height=500)
```

Left unset, the figure keeps a size already given it, or otherwise defaults
to one that keeps the number-box sidebar to roughly a fifth of the total
width. The split is a percentage on both sides, not a fixed pixel count, so
the figure and sidebar always cover the notebook's actual full width between
them. It is not a draggable divider — there is no drag handle between the
figure and the sidebar; resize by calling `range_controls()` again with new
numbers.

A surface, contour or region resamples both axes; an ordinary curve derives Y
from X, so only X resamples and Y becomes a `ylim` view update instead — no
per-point resolution to recover on an axis nothing was sampled along. A real
3D surface (`surface`/`psurface`, not the flat `contour`/`region`) gets two
more boxes, `z min`/`z max`: Z is the surface's *output*, not a domain, so
moving them only ever re-applies `zlim` — cheap, no resample, same as Y on an
ordinary curve.

If you do not want the sidebar at all, `controls=False` (§4.12c) turns it off
for one plot and `set_range_controls(False)` for the whole notebook.

Where the domain symbol is not the horizontal axis — a parametric, polar or 3D
space curve, whose x and y are both outputs of one parameter — the first row is
labelled with that parameter's name (`t`, say) rather than `x`. Moving it draws
more or less of the curve; it does not crop the view. A space curve also offers
Z controls.

marimo needs no wrapper: its own `mo.ui.number()` composed directly with
`plot()` gets the same live resampling for free, because marimo re-runs any
cell that reads `.value` on every change.

```text
x_min = mo.ui.number(start=-20, stop=20, value=-10, label="x min")
x_max = mo.ui.number(start=-20, stop=20, value=10, label="x max")
mo.hstack([x_min, x_max])
```

```text
plot(sin(x)/x, (x, x_min.value, x_max.value))
```

A full working version is `examples/marimo_notebook.py`. Outside a live
kernel — a plain script, or an exported HTML file — there is no host to push
a resample into, so `xlim`/`ylim` remain how a window is chosen there.

**Nothing showing?** `frontend_report()` says exactly why:

```python
from mathslate import frontend_report
print(frontend_report())
```
```text
frontend: jupyter | interactive widgets: ipywidgets | range_controls(): ready
```

The last field names the specific reason when it is not `ready`: the host
isn't Jupyter or Colab, `ipywidgets` or `anywidget` isn't installed, or
`set_range_controls(False)` turned it off. That last one is the one knob
this feature has —

```python
from mathslate import set_range_controls, get_range_controls

set_range_controls(False)          # every later plot() shows the plain figure
print(get_range_controls())        # False
set_range_controls(True)           # back to the default
```

— a global switch rather than a `plot()` keyword, because it is a standing
preference ("I don't want this today") and not a per-figure choice; call
`.range_controls()` on one result instead when the choice is per-figure.

---

## 5. Symbol-to-axis binding

Given `a*sin(x)`, how does MathSlate know `x` is the axis and `a` is not? Four
rules, applied in order.

**1. An explicit range wins.**

```python
b = symbols("b", real=True)
print(plot(b*x, (b, -2, 2), parameters={x: 1.0}).plan.symbol.name)
```

It can only win among symbols the expression actually has, though. A range for
a symbol that is not in what you are plotting cannot be what you meant, and
making it the axis would leave the real variable free and the curve undefined
everywhere:

```python raises=UnsupportedInputError
c = symbols("c", real=True)
plot(sin(x), (c, -1, 1))
```

Two ranges for one symbol are refused for the same reason — one of them would
have to be discarded, and you would not be told which:

```python raises=UnsupportedInputError
plot(sin(x), (x, -1, 1), (x, -2, 2))
```

Naming the axes does not give the *other* symbols values, and a curve with an
unbound symbol in it is not a number anywhere. Two symbols are a surface
(§4.11), so this only bites from three:

```python raises=UnsupportedInputError
a = symbols("a", real=True)
plot(a*x*y, (x, -5, 5), (y, -5, 5))
```

Freeze the extra one with `parameters=`, which is what the error asks for:

```python
print(plot(a*x*y, (x, -5, 5), (y, -5, 5), parameters={a: 2}).plan.kind)
```

**2. Bound parameters are never axis candidates.** See §4.8.

**3. Otherwise, conventional order.**

```text
x, y, z  →  t, u, v  →  r, theta, phi  →  then alphabetically
```

```python
from mathslate.core import binding

p, q = symbols("p q", real=True)
print([s.name for s in binding.sort_by_convention([q, theta, p, t, x])])
```

**4. If it is still ambiguous, ask — do not guess.**

```python raises=AmbiguousAxisError
c, d, e = symbols("c d e", real=True)
plot(c*d*e)
```

The exception's `.question` is the message and `.candidates` lists the symbol
names. In a library there is nobody to prompt, so the question is delivered as
the error text, including the exact call that resolves it.

```python
from mathslate.errors import AmbiguousAxisError

c, d, e = symbols("c d e", real=True)
try:
    plot(c*d*e)
except AmbiguousAxisError as error:
    print(error.candidates)
```

### Default ranges

| Situation | Default |
|---|---|
| explicit curve | `(-10, 10)` |
| parametric or polar, trigonometric components | `(0, 2π)` — one full turn |
| parametric or polar, otherwise | `(-10, 10)` |

```python
print(plot((cos(t), sin(t))).plan.param_range[1].__round__(3))
print(plot((t, t**2)).plan.param_range)
```

---

## 6. Sampling: the technical heart

`plot(tan(x))` drawing no spurious vertical lines is the feature that separates
MathSlate from a plotting wrapper. Everything else in this manual is
convenience. Six steps, in order.

### 6.1 Symbolic singularity detection

Poles come from `sympy.calculus.singularities`, never from numeric guessing.
Infinite families (`ImageSet` over the integers, as trigonometric poles
produce) are enumerated within the window.

```python
from mathslate.core import sampling

info = sampling.describe_domain(tan(x), x, -10.0, 10.0)
print([round(p, 4) for p in info.singular_points])
```

### 6.2 Real domain computation

The real domain comes from `sympy.calculus.util.continuous_domain`, and
nothing is ever sampled outside it. The window is then cut open at every
singular point, giving the list of continuous pieces.

```python
from mathslate.core import sampling

print(sampling.describe_domain(sqrt(x), x, -10.0, 10.0).intervals)
print(sampling.describe_domain(sqrt(x**2 - 1), x, -10.0, 10.0).intervals)
print(sampling.describe_domain(1/x, x, -10.0, 10.0).intervals)
```

When SymPy cannot decide — it raises `NotImplementedError` on `floor` and on
`Piecewise` — MathSlate falls back to the full window and **says so** in the
notes rather than pretending to knowledge it does not have.

```python
f = plot(floor(x), verbose=False)
print(f.plan.series[0].sample.domain_intervals)
print([n for n in f.notes if "domain" in n])
```

### 6.3 Adaptive subdivision

Start from `points` uniform samples per piece. Repeatedly insert midpoints
into every segment whose three-point interior angle is sharper than the
threshold, so detail lands where the curve bends. Bounded by `max_depth` and
`max_points`.

Both axes are normalised by the *visible* scale before the angle is measured,
which is what makes the criterion behave the same for `sin(x)` and for
`exp(x)`.

```python
straight = plot(2*x + 1, verbose=False).plan.total_points
wiggly = plot(sin(5*x), verbose=False).plan.total_points
print(straight, wiggly, wiggly > straight)
```

### 6.4 Line breaking

A `NaN` is inserted at every discontinuity, and Plotly is told
`connectgaps=False`. That, and nothing else, is what prevents the false
vertical lines.

Two sources of breaks:

- **Symbolic**, from §6.1 — exact.
- **Numeric**, for jumps SymPy cannot see: `floor`, `ceiling`, `sign`,
  `Piecewise`, and anything built from them.

The numeric test is worth understanding, because it is what lets MathSlate
handle step functions without ever naming one. Take the largest gaps between
adjacent samples and bisect around each. **A genuine discontinuity keeps its
gap size as the bracketing interval shrinks; a steep but continuous slope does
not.** So the probe compares gap *decay*, not gap size.

```python
print(len(plot(floor(x), verbose=False).plan.series[0].sample.breakpoints))
print(plot(atan(1000*x), (x, -1, 1), verbose=False).plan.series[0].sample.breakpoints)
```

Nineteen breaks for the staircase; none at all for a curve that merely rises
steeply. Probing is capped at `max_jump_probes` per curve.

**Parametric and polar curves get all of this too.** The continuous pieces are
the *intersection* of both components' domains — a point of the curve is
drawable only where `x(t)` and `y(t)` both are — and the poles are their union.
The jump probe watches both coordinates, because a jump in `x` breaks the line
just as surely as one in `y`, and the cut blanks both.

```python
f = plot((tan(t), t))
print(len(f.plan.series[0].sample.breakpoints))
print(plot((sqrt(t), t), (t, -5, 5)).plan.series[0].sample.domain_intervals)
```

### 6.5 Y-axis clipping

A single pole reaching 10⁷ would flatten every other feature to a horizontal
smear. So a visible y-window is chosen:

1. discard samples within 2% of the window width of any pole;
2. take the 2nd and 98th percentiles — read off a *uniform* grid, because
   adaptive refinement piles points up near poles and would otherwise drag the
   percentiles with it;
3. widen that window threefold, since the bare percentile band is too tight to
   read;
4. clamp back to values the curve actually reaches, so bounded functions are
   not padded with empty space;
5. apply only when the data really runs away, or when poles are known.

```python
tan_range = plot(tan(x), verbose=False).plan.y_range
sinc_range = plot(sin(x)/x, verbose=False).plan.y_range
print(None if tan_range is None else [round(v, 2) for v in tan_range])
print(None if sinc_range is None else [round(v, 2) for v in sinc_range])
print(plot(sin(x), verbose=False).plan.y_range)
```

**Horizontally, only where x can run away.** For `y = f(x)` the x extent is the
range you asked for and there is nothing to choose. A parametric or polar curve
has no such guarantee — `x(t)` reaches a pole exactly as `y(t)` does — so it
gets the same treatment on both axes, and neither when it does not need one.

```python
print(plot((tan(t), t), verbose=False).plan.x_range is not None)
print(polar(1 + cos(t), verbose=False).plan.x_range)
```

The tangent gets a readable window; `sin(x)/x` keeps its true extent because
of step 4; the sine is left entirely alone because it never runs away.

### 6.6 Vectorised evaluation

Expressions are evaluated with `lambdify(modules="numpy")`. Complex results are
turned into `NaN` rather than silently taking a real part, so a function
outside its real domain leaves a gap instead of a wrong curve.

If the vectorised path fails, MathSlate falls back to element-wise evaluation
**loudly** — a `RuntimeWarning`, a note on the result, and an extra field in
the summary line. It never degrades silently.

Element-wise means three tiers, tried in order: `lambdify(modules="math")`, the
NumPy function point by point, and finally SymPy's own `evalf`. That last tier
is what makes the special functions reachable — neither NumPy nor `math`
carries `zeta`, `Si` or `besselj`, so without it the fallback could not rescue
anything NumPy had just failed on.

```python
import warnings
from sympy import zeta

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    f = plot(zeta(x), (x, 2, 5), verbose=False)
print(f.plan.series[0].sample.vectorized, "element-wise" in f.summary())
```

### 6.7 `SamplingConfig`

The defaults follow the PRD. They are reachable if you need them:

```python
from mathslate.core.sampling import DEFAULT_CONFIG

print(DEFAULT_CONFIG.initial_points, DEFAULT_CONFIG.max_depth, DEFAULT_CONFIG.max_points)
print(DEFAULT_CONFIG.angle_threshold, DEFAULT_CONFIG.clip_percentiles)
```

| Field | Default | Meaning |
|---|---|---|
| `initial_points` | 200 | uniform samples before refinement |
| `max_depth` | 8 | refinement passes |
| `max_points` | 5000 | hard cap per curve |
| `angle_threshold` | 177.5° | interior angle below which a triple is "curved" |
| `clip_percentiles` | `(2.0, 98.0)` | y-window percentiles |
| `max_jump_probes` | 200 | numeric discontinuity probes per curve |

### 6.8 `SampleResult`

`result.plan.series[i].sample` for the raw output of all of the above.

| Attribute | Meaning |
|---|---|
| `x`, `y` | the sampled arrays, `NaN` at every break |
| `t` | the parameter values, for parametric curves |
| `breakpoints` | where the line was cut |
| `domain_intervals` | continuous pieces of the real domain in the window |
| `y_range` | the chosen visible window, or `None` |
| `x_range` | the same, horizontally — parametric and polar only (§6.5) |
| `vectorized` | whether the fast path held |
| `n_points`, `finite_count` | sizes |

```python
sample = plot(1/x, verbose=False).plan.series[0].sample
print(sample.breakpoints, sample.domain_intervals, sample.vectorized)
```

---

## 7. Axes

### 7.1 π-aware ticks

When an expression contains a trigonometric function or `pi`, the horizontal
axis is labelled in multiples of π. The spacing is chosen from π/4, π/2, π,
2π, 4π, 8π so that between 5 and 17 ticks land in the window.

```python
print(list(plot(sin(x), verbose=False).plotly.layout.xaxis.ticktext))
print(plot(exp(x), verbose=False).plotly.layout.xaxis.ticktext)
```

Not applied to parametric, polar, data or callable plots, where the horizontal
axis is not the expression's variable.

### 7.2 The log-scale suggestion

If an expression is dominated by `exp` or `log` and its positive values span
three or more orders of magnitude, MathSlate adds a note suggesting a log
scale. It does not apply one. A learner reading a log-scaled plot without
realising it is worse off than one reading an awkward linear plot.

```python
print([n for n in plot(exp(x), verbose=False).notes if "log scale" in n])
```

---

## 8. `PlotResult`

Everything `plot()` returns.

### Escape hatches

| Attribute | Type | Notes |
|---|---|---|
| `.plotly` / `.figure` | `go.Figure` | yours to mutate; MathSlate does not own it |
| `.sympy` | `Expr`, tuple, or `None` | `None` for raw data |
| `.numpy` | `(ndarray, ndarray)` | the sampled `x` and `y` of the first series |

```python
f = plot(sin(x)/x, verbose=False)
print(type(f.plotly).__name__, f.sympy, f.numpy[0].shape)
```

```python
f = plot([sin(x), cos(x)], verbose=False)
print(f.sympy)
print(plot([1.0, 2.0], verbose=False).sympy)
```

### Inspection

| Member | Returns |
|---|---|
| `.plan` | the resolved `PlotPlan` — kind, symbol, range, series, config |
| `.notes` | tuple of the messages printed under the summary |
| `.summary()` | the one-line inference report |

```python
f = plot(tan(x), verbose=False)
print(f.plan.kind, f.plan.symbol.name, f.plan.param_range, f.plan.total_points)
```

### Output

| Member | Effect |
|---|---|
| `.python()` | the equivalent code, as a string |
| `.show_python()` | prints it and returns it |
| `.show()` | delegates to `Figure.show()` |
| `._repr_mimebundle_()` | notebook rendering, delegated to the figure |

### `PlotPlan`

| Attribute | Meaning |
|---|---|
| `kind` | which dispatch row matched |
| `series` | list of `Series`, each with `.name`, `.sample`, `.expr` |
| `symbol` | the axis symbol |
| `param_range` | `(lo, hi)` |
| `exprs` | the expressions plotted |
| `y_range` | the union of the series' visible windows |
| `total_points` | samples across all series |
| `notes`, `config` | as above |

---

## 9. `show_python()`

The flagship. Given that the project's purpose is for you to become fluent in
real Python, this is not a secondary convenience.

```python
f = plot(sin(x)/x, verbose=False)
code = f.python()
print(code.splitlines()[0])
```

### The guarantee

**The emitted code runs, verbatim.** No pseudocode, no ellipses, no "roughly
this". This is enforced: the test suite executes `show_python()` output for
every plot kind under `exec` and asserts a figure comes out.

```python
f = plot(tan(x), verbose=False)
namespace = {}
exec(f.python(), namespace)
print(type(namespace["fig"]).__name__)
```

### What it emits

Not a naive `linspace`. It writes out every decision MathSlate made silently:

- explicit `x = sp.symbols('x', real=True)` — always, even under `import *`
- `sp.lambdify(..., 'numpy')`
- the continuous pieces of the real domain, sampled separately and joined with
  `NaN`
- jump discontinuities not already at a piece boundary, cut explicitly
- the y-window, with a comment explaining why it exists
- π tick values and labels, when they apply
- a targeted `from sympy import ...` covering exactly the names used

```python
code = plot(sqrt(x), verbose=False).python()
print("pieces = [(0.0, 10.0)]" in code)
```

Per kind: curves and overlaid curves emit the piecewise sampler; parametric and
polar emit two `lambdify` calls over the parameter, and the same piecewise
sampler over `t`, so a cut MathSlate made is a cut the emitted code makes;
**callables and data emit their samples as literal arrays**, evenly reduced to
2000 points if there are more.

```python
print("pieces = [" in polar(tan(t), verbose=False).python())
```

A plain Python function cannot be written into source — `np.sin` prints
as `sin`, and a lambda has no usable name at all — so emitting a call to it
would produce code that runs only in the namespace it came from. Literals
always run.

### What is reproduced

`show_python()` reproduces the figure, not merely *a* figure. Every call is
built twice in the test suite — once through MathSlate, once by executing the
emitted source in a clean namespace — and the two figures are compared property
by property:

| Reproduced | Deliberately omitted |
|---|---|
| trace x/y data, mode, name, `connectgaps` | margins |
| template, title, `showlegend` | hover mode and hover template |
| x-axis range, π tickvals/ticktext | axis titles |
| axis tick density (`ticks`) | zeroline styling |
| y-axis range and type (`log`) | |

The omitted column is chrome: emitting it would bloat the teaching code without
changing the picture. Everything that changes the picture is in the left
column. Both lists are exported as `mathslate.render.options.REPRODUCED` and
`NOT_REPRODUCED`, and the test suite asserts against them, so this table cannot
quietly go stale.

Options you pass to `plot()` are part of that guarantee:

```python
f = plot(sin(x), kind="scatter", yscale="linear", title="Points", show_legend=True)
namespace = {}
exec(f.python(), namespace)
print(namespace["fig"].data[0].mode == f.plotly.data[0].mode)
print(namespace["fig"].layout.title.text == f.plotly.layout.title.text)
```

### The one limitation

For expression-based plots the emitted program uses a **uniform grid**, not
MathSlate's adaptive one. Reproducing the adaptive sampler verbatim would mean
emitting the sampler itself, which would defeat the purpose. So the curve is
the same curve to plotting accuracy, not array-identical: the suite requires
each curve to lie within 2% of the visible diagonal of the other, measured
point-to-segment, inside the window you actually see.

Callable and data plots have no such gap — their samples are emitted exactly.

---

## 10. `analyze()`

```text
analyze(obj, *ranges, parameters=()) -> Analysis
result.analyze()                    -> Analysis
```

The properties of an expression: roots, extrema, inflection points, symmetry,
periodicity, asymptotes, discontinuities and monotonic intervals.

**Never automatic.** Property detection is slow enough and noisy enough that
running it on every `plot()` would spoil both. You ask for it.

```python
report = analyze(x**3 - 3*x)
print(report.roots.describe())
print(report.maxima.describe(), "/", report.minima.describe())
```

From a plot, the window analysed is **the one you are looking at** — the roots
of `sin(x)` depend entirely on where you looked:

```python
print(len(plot(sin(x), (x, 0, 3.5), verbose=False).analyze().roots))
print(len(plot(sin(x), (x, -10, 10), verbose=False).analyze().roots))
```

### 10.1 Exact, or labelled approximate

Every property is attempted symbolically first. `solveset` answers with a
`FiniteSet` when it has really solved the equation and a `ConditionSet` when it
has merely restated it — that is the signal to fall back to sampling.

A numeric answer is **never presented as an exact one**. It carries
`approximate=True` all the way to the printed panel, because a reader who
cannot tell a proof from a sample is worse off than one who was told nothing.

```python
exact = analyze(x**3 - 3*x)
sampled = analyze(x - cos(x))
print(exact.approximate, exact.roots.symbolic)
print(sampled.approximate, sampled.roots.describe())
```

### 10.1a The third reason an answer is approximate: time

SymPy has no notion of giving up. `solveset` on a degree-40 polynomial does not
fail and does not hang — it takes about **eighty seconds** and then answers,
which to a reader is indistinguishable from a broken library.

So every symbolic step gets a five-second budget. When it expires, that property
takes the numeric path it would have taken had SymPy *declined*, and the report
says which of the two happened — because they are not the same thing. SymPy
declining is a fact about the mathematics; running out of five seconds is a fact
about this computer, and a bigger budget may well turn the same answer exact.

```python
big = sum(Integer(i + 1)*x**i for i in range(41))
print([n for n in analyze(big).notes if "budget" in n])
```

Nothing in ordinary use comes near the limit — the slowest plot in the test
corpus is 1.7 s and the slowest `analyze()` 3.9 s — so on everyday work the
budget never fires and exact answers stay exact.

Change it when you would rather wait than approximate. It is reached through
`mathslate.core` rather than the top level, because it is a knob for the rare
session that hit the limit, not something a learner should have to meet:

```python
from mathslate.core import get_symbolic_budget, set_symbolic_budget

set_symbolic_budget(30)          # seconds
print(get_symbolic_budget())
set_symbolic_budget(None)        # no limit at all
print(get_symbolic_budget())
set_symbolic_budget(5.0)         # back to the default
```

One expired step disables the symbolic attempts for the *rest of that*
`analyze()` call. Eight properties over one expression means about a dozen
symbolic steps, and paying five seconds for each would turn a five-second limit
into a minute-long wait for the same answer; the first timeout is evidence about
the expression, not about that one step.

The portable implementation uses a worker thread. Native code that does not
return to a Python bytecode boundary cannot be interrupted immediately; while
such a worker remains alive, later symbolic attempts are refused rather than run
concurrently against SymPy's process-global caches. Isolating symbolic work in a
restartable process is the long-term route to a hard deadline for native code.

### 10.2 What each property means

| Field | Type | Notes |
|---|---|---|
| `.roots` | `Points` | zeros inside the window, within the real domain |
| `.maxima` / `.minima` | `Points` | stationary points that actually turn |
| `.inflections` | `Points` | where concavity changes, not merely `f'' = 0` |
| `.discontinuities` | `Points` | symbolic poles, plus jumps found by §6.4's probe |
| `.asymptotes` | `tuple[Asymptote, ...]` | vertical, horizontal and oblique |
| `.symmetry` | `Symmetry` | `even`, `odd`, `neither` or `unknown` |
| `.periodicity` | `Periodicity` | the period, or `None` |
| `.increasing` / `.decreasing` | intervals | split at every turn, pole and domain edge |
| `.notes` | `tuple[str, ...]` | what could not be computed, and why |
| `.sympy` | `Expr` | the escape hatch — the expression itself |
| `.approximate` | `bool` | whether *any* line came from sampling |
| `.provenance()` | pairs | which call produced each property — §10.4 |
| `.python()` | `str` | those calls as runnable source — §10.4 |

Three cases are worth knowing because the naive answer is wrong in each:

```python
print(analyze(x**4).inflections.describe())      # f'' = 12x**2 is 0 at the origin
print(analyze(x**3).maxima.describe())           # stationary, but never turns
print(analyze(sin(x)).asymptotes)                # oscillating: bounded, not settling
```

`x**4` is concave up throughout, `x**3` rises throughout, and `sin(x)` has no
horizontal asymptote — SymPy reports its limit at infinity as
`AccumBounds(-1, 1)`, which is bounded but is not a value the curve approaches.

### 10.3 Reading the report

`Points` behaves like a sequence, and every part can describe itself:

```python
roots = analyze(x**2 - 1).roots
print(len(roots), list(roots), roots.describe())
```

`.text()` is the whole panel as plain text; in a notebook the object renders as
a **collapsed** panel, which is what PRD §5.6 asks for.

```python
print(analyze(1/x).text())
```

### 10.4 "How did you get this?"

There is an honest answer and a dishonest one, and the difference is worth
being precise about.

**The honest one — what was actually run.** Every property records the call
that produced it, and `.python()` emits that as runnable source, exactly as
`show_python()` does for a plot (§9):

```python
report = analyze(x**3 - 3*x)
for label, how in report.provenance():
    print(f"{label}: {how}")
```

```python
print(analyze(x**2 - 1).python())
```

The emitted code **runs verbatim and reproduces the same values** — that is
the same guarantee §9 makes. Properties that were sampled rather than solved
emit their values as literals with a comment naming the search, because
emitting a `solveset` call that could not solve the thing would be a lie that
happens to run.

**The dishonest one — a worked solution.** MathSlate does not produce one, and
this is not a gap waiting to be filled. SymPy's only step-by-step engine is
`sympy.integrals.manualintegrate`, which covers integration and gives up
(`DontKnowRule`) on ordinary cases like `sqrt(x**3 + 1)`. `analyze()` performs
no integration at all — it solves, differentiates and takes limits, and SymPy
exposes no derivation trace for any of the three. A narration of "how" would
therefore have to be *invented*, which is precisely the LLM-generated
derivation PRD §2.2 forbids by name.

So: the receipt is real and always available; the tutorial is not offered
because it could not be honest. See §16 and PRD §2.2. There is no `explain()`
here in any spelling, and the test suite asserts as much about this object
specifically.

---

## 11. `slider()` and `animate()`

```text
slider(start, stop, step=None, *, default=None, name=None, label=None) -> Slider
animate(obj, *ranges, over=None, **plot_options)                       -> PlotResult
```

A parameter the reader can vary. **You never write a callback.**

```python
a = slider(-3, 3, default=1, name="a")
f = plot(a*sin(x))
print(f.interactive, len(f.plotly.frames))
```

`animate()` is the same picture with a play button:

```python
print([b.label for b in animate(a*sin(x)).plotly.layout.updatemenus[0].buttons])
```

### 11.1 How one line does that

A `Slider` is not a `Symbol` — it carries a range, a step and a current value —
but it answers SymPy's `_sympy_()` hook with its symbol. So `a*sin(x)` builds
an ordinary expression in an ordinary symbol, and nothing downstream needs to
know a widget was involved.

```python
print((a*sin(x)).free_symbols == {a.symbol, x})
```

`plot()` then treats that symbol as a parameter rather than an axis. That is
[§5 rule 2](#5-symbol-to-axis-binding), and it now happens by itself:

```python
print(plot(a*sin(x), verbose=False).plan.symbol.name)
```

The precedence is the same as everywhere else — an explicit range wins over a
slider, and an explicit `parameters=` entry wins over the slider's own value:

```python
print(plot(sin(a), (a.symbol, -1, 1), verbose=False).plan.symbol.name)
```

### 11.2 Why Plotly's own control is the default

PRD §6.2 assigns `mo.ui.slider` to marimo, `ipywidgets` to Jupyter and Colab,
and Plotly animation frames to "other / static export". MathSlate uses the
**frames everywhere**, for three reasons:

- they behave identically in all three hosts, so a notebook is portable;
- they need no frontend package at all, which is [§1](#1-installation-and-requirements)'s promise;
- they survive being written to one HTML file, which is what a teacher hands out.

The cost is that the positions are computed in advance rather than on demand.
`slider(0, 1)` draws 21 frames; `slider(0, 1, 0.25)` draws exactly the five you
asked for.

```python
b = slider(0, 1, 0.25, name="b")
print(b.values())
```

For live recomputation instead, `Slider.widget()` returns the host's own
control — `mo.ui.slider` under marimo, `ipywidgets.FloatSlider` under Jupyter
and Colab. Neither package is a dependency, so it raises where there is no
notebook.

### 11.3 Several sliders

One parameter is animated; the rest hold their current values. Plotly's frames
are a one-dimensional sequence, and a grid over several sliders would multiply
out into thousands of pre-computed curves. `animate(..., over=b)` picks which
one runs, and the report says what was held.

```python
c = slider(1, 2, name="c")
print([n for n in plot(a*c*sin(x), verbose=False).notes if "holds" in n])
```

### 11.4 The moving picture is what `show_python()` emits

A still frame would not be "the same result" (§9). The emitted code carries
every frame and builds the same Plotly control:

```python
f = plot(a*sin(x), verbose=False)
namespace = {}
exec(f.python(), namespace)
print(len(namespace["fig"].frames) == len(f.plotly.frames))
```

### 11.5 A slider lasts the whole session

Creating a slider binds its symbol as a parameter until something releases it.
That lifetime is deliberate — it is what lets `plot(a*sin(x))` know that `a` is
a parameter without being told again in every call — but it means the binding
outlives the cell that made it.

So a slider named after a symbol you also want to plot *over* is a collision:

```python raises=UnsupportedInputError
from mathslate.ui import release_all

release_all()
time = slider(0, 10, default=3, name="t")
plot((cos(t), sin(t)))
```

`t` is now a parameter everywhere, so the circle would freeze to a single
point. Rather than draw that, MathSlate refuses and names the three ways out:

```python
from mathslate.ui import release_all

# 1. plot over it anyway — an explicit range outranks a slider
print(plot((cos(t), sin(t)), (t, 0, 6.283), verbose=False).plan.kind)

# 2. let the slider go
release_all()
print(plot((cos(t), sin(t)), verbose=False).plan.kind)
```

The third is to give the slider a name of its own, which is usually what was
meant: `slider(0, 10, name="time")`.

`release(a_slider)` frees one; `release_all()` frees every slider in the
session. Both live in `mathslate.ui`. Binding only one of two symbols is not a
collision — it is the ordinary case, and a surface simply becomes a curve:

```python
from mathslate.ui import release_all

release_all()
height = slider(0, 3, default=2, name="y")
print(plot(x*y, verbose=False).plan.kind)
release_all()
```

---

## 12. `table()` and HTML export

```text
table(obj, *ranges, rows=11, label=None, parameters=()) -> Table
result.table(rows=11)                                   -> Table
result.to_html(path=None, *, standalone=True)           -> str
```

### 12.1 The same function, read as numbers

A graph shows shape; a table shows value. You reach for the second when
checking a hand-worked answer.

```python
print(table(sin(x), (x, 0, 1), rows=3).text())
```

A list gives several columns, and the axis is chosen by the same rules as
everywhere else (§5):

```python
print(table([sin(x), cos(x)], (x, 0, 1), rows=3).headers())
```

From a plot, it tabulates the window you are looking at:

```python
print(plot(sin(x), (x, 0, 2), verbose=False).table(rows=3).inputs)
```

A point where the expression is not a real number is **blank**, not zero — a
table that printed a number there would be lying about the mathematics:

```python
f = table(sqrt(x), (x, -1, 1), rows=5)
print("—" in f.text(), f.notes[0][:20])
```

The escape hatches are the ones you already know, and `.python()` emits the
plain NumPy that prints the same rows:

```python
inputs, values = table([sin(x), cos(x)], (x, 0, 1), rows=4).numpy
print(inputs.shape, values.shape)
```

### 12.2 One file you can hand to a class

`to_html()` writes the figure as a **single self-contained page**: Plotly
itself is embedded, so it opens with no network and nothing installed.

```python
page = plot(sin(x), verbose=False).to_html()
print(len(page) > 1_000_000, "<script src=" not in page)
```

`to_html(path)` also writes it to disk. `standalone=False` links Plotly from a
CDN instead — a much smaller file that needs the network.

This is why §11's slider is built from frames carried inside the figure rather
than from a notebook widget: a widget needs a live kernel, and a file handed to
a class does not have one. The interactive plot survives the trip.

```python
b = slider(-2, 2, default=0, name="handout")
print("addFrames" in plot(b*sin(x), verbose=False).to_html())
```

---

## 13. `dataset()`, statistics and linear algebra

```text
dataset(source, columns=None, encoding=None) -> Dataset
Dataset.fit(model, x=None, y=None, symbol=None, guess=None) -> FitResult
```

### 13.1 The bridge

Everything before this section starts from an expression. Real work usually
starts from measurements, and `fit()` is where the two meet.

```python
readings = dataset({"x": [0, 1, 2, 3, 4], "y": [1.0, 3.1, 4.9, 7.2, 8.9]})
c, d = symbols("c d", real=True)
found = readings.fit(c*x + d)
print(found.describe())
```

The point is what comes back. `found.expr` is **an ordinary SymPy expression** —
the model with its parameters filled in — so everything else in MathSlate works
on it with no conversion:

```python
print(diff(found.expr, x))
print(analyze(found.expr).roots.describe())
```

A model **linear in its parameters** is solved exactly by least squares —
`c*x**2 + d` counts, because linearity in `x` is not what matters. Anything
else is refined iteratively from `guess` (or from a small set of starts chosen
from the data):

```python
growth = dataset({"x": [0, 1, 2, 3], "y": [2.0, 4.0, 8.0, 16.0]})
print(growth.fit(c*exp(d*x)).describe())
```

The iteration is **damped** least squares (Levenberg–Marquardt), and only keeps
a step that lowered the cost. That matters for any model with a pole in it:
`c/(x + d)` from an undamped full step routinely throws `d` past the pole on
the first iteration and never returns. It also runs from several starting
points, because damping keeps a step from bolting but cannot move the search
into a different basin — `c*exp(d*x)` started at `d = -3` descends neatly into
a flat region and would report the flat region.

```python
print(dataset({"x": [1, 2, 3, 4], "y": [1.0, 0.5, 0.3333, 0.25]}).fit(c/(x + d)).describe())
```

`.residuals` and `.r_squared` come back with it, so the fit can be judged
rather than trusted. On constant data `r_squared` is `None` rather than 1 or 0:
R² compares against "predict the mean", and there is nothing there to beat.

Two things are refused rather than answered badly: a model whose parameters
cannot be separated at all (`c*d*x` — only the product moves the curve), and a
fit whose independent and dependent columns are the same, which would always
"succeed" with R² = 1 and mean nothing.

### 13.2 Building one

From a mapping, a 2-D array, or a CSV whose first row is the column names.

```python
print(dataset({"t": [0, 1], "v": [2.0, 4.0]}).names)
```

A cell that is not a number becomes `NaN` rather than an error — a
hand-exported sheet routinely has a note in it, and refusing the whole file
over one cell helps nobody.

```python
print(readings.describe())
```

#### CSV encodings

A CSV exported from Excel is usually not UTF-8: it is cp949 on a Korean
Windows, cp1252 on much of Europe, and UTF-8 *with a byte-order mark* even when
it is UTF-8. All of those are read without being asked about — UTF-8 first,
then the encodings spreadsheets write, then `latin-1` last:

```python
from pathlib import Path
from tempfile import mkdtemp

sheet = Path(mkdtemp()) / "측정값.csv"
sheet.write_bytes("시간,측정값\n0,1\n1,3\n2,5\n".encode("cp949"))

korean = dataset(str(sheet))          # nothing said about the encoding
print(korean.names, korean["측정값"].tolist())
```

Name the encoding when you know it, or when the guess picked wrongly. Doing so
turns the guessing off, so a mismatch then fails with the file name and this
argument in the message, rather than with a bare `UnicodeDecodeError` from
inside the standard library:

```text
dataset("readings.csv", encoding="cp949")
dataset("exported.csv", encoding="utf-16")
```

UTF-16 and UTF-32 have to be named. They are full of NUL bytes, and a NUL byte
is what MathSlate uses to tell a text file from a binary one — without that
check `latin-1` accepts *any* byte sequence, so handing the reader a PNG
produced a one-column dataset named `\x89PNG` and no error at all:

```python raises=UnsupportedInputError
picture = Path(mkdtemp()) / "chart.csv"
picture.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
dataset(str(picture))
```

### 13.3 Plotting it

A `Dataset` reads like a spreadsheet: first column across, the rest up, and the
axis titles come from the column names.

```python
print(plot(readings, verbose=False).plan.kind)
```

`kind="hist"` and `kind="box"` ask about the shape of each column instead,
which is a different question:

```python
print(plot(readings, kind="hist", verbose=False).plan.kind)
print(plot([1.0, 2.0, 2.0, 3.0, 9.0], kind="box", verbose=False).plan.kind)
```

Histograms are overlaid, not stacked: stacking two answers a question about
their sum, which is not what was asked.

### 13.4 A matrix as what it does

A matrix of numbers on a page is not a picture of anything. What makes it mean
something is seeing where it sends things, so a 2×2 `Matrix` is drawn as the
unit square, its image, and the eigenvectors — the directions the
transformation only stretches.

```python
f = plot(Matrix([[2, 1], [1, 3]]), verbose=False)
print(f.plan.kind, len(f.plan.eigen))
print([n for n in f.notes if "determinant" in n])
```

A rotation says so rather than drawing nothing:

```python
print([n for n in plot(Matrix([[0, -1], [1, 0]]), verbose=False).notes if "turns" in n])
```

The aspect ratio is locked, or the distortion you see would be the plot's
rather than the matrix's.

### 13.5 The optional assistant

```python
from mathslate.ai import ask, PROVIDERS
print([p.name for p in PROVIDERS])
```

For notebooks, the guided panel is the easiest starting point:

```python
from mathslate.ai import assistant
assistant("plot the tangent over one period")
```

It selects a provider, links to its API-key page, accepts a masked key, reports
waiting/success/authentication/quota/timeout/empty-response states, and shows
the generated code before the separate **Validate & Run** action. On a local
computer, **Remember on this device** stores the key with the operating-system
credential manager. JupyterLite deliberately disables persistent key storage.

`mathslate.ai` turns a question into MathSlate code. Three things are true of
it by construction:

- **The core never imports it.** PRD §8 requires MathSlate to work fully
  offline, because school and corporate networks block AI endpoints. `import
  mathslate` does not reach this module and the provider SDKs are imported
  inside the call that needs them. The test suite asserts both in a subprocess.
- **It returns code; it does not run it.** `ask(...)` gives you a `Suggestion`
  you read. `Suggestion.run()` is explicit and validates an allowlisted,
  expression-oriented subset before running with restricted builtins. Imports,
  file/network access, dunder inspection, loops and indirect calls are refused.
  `run(unsafe=True)` restores unrestricted Python and is only for code you trust.
- **No provider is bundled.** Claude, OpenAI and Gemini are supported; install
  one and set its key. The key may be an environment variable or an explicit
  argument; an explicit key does not also need to be copied into the
  environment. With none configured, the error says exactly that.

```text
ask("plot sine", provider="openai", api_key="sk-...")
configure(provider="openai", api_key="sk-...")
configure(provider="gemini", api_key="...", remember=True)
```

**An explicit key needs an explicit provider.** A key is a credential for one
company, and nothing in the string reliably says which; sending it to the wrong
one would hand your secret to a business you have no account with. So when more
than one provider SDK is installed, `api_key=` without `provider=` is refused
rather than guessed at. With exactly one installed there is no ambiguity and it
is used.

`configure()` treats `None` as "leave as it was", so it cannot clear anything.
`forget()` is how a key leaves the session; `forget(persistent=True)` also
removes a remembered key from the operating-system credential manager.
`configured()` reports what is held without ever printing the key:

An explicit session key stays with the provider it was configured for. Changing
the provider without supplying its matching key never carries the old key
across: the new provider must already have its environment variable set, or the
change is refused. Likewise, a per-call `api_key=` does not silently inherit a
different session provider; name `provider=` with it when several SDKs are
installed.

```python
from mathslate.ai import configure, configured, forget

configure(model="some-model")
print(configured()["model"])
forget()
print(configured())
```

```python
from mathslate.ai import system_prompt
print(len(system_prompt()) < 4000)
```

The prompt is generated from the real dispatch contract, so a new plot kind
cannot be added without the assistant being told in the same commit. It is
short because the API is small — which is the argument for keeping the API
small (PRD goal 5).

#### 13.5.1 Asking for something other than code

`ask()` writes code you then read. The rest of the module divides along one
line: whether the model is being asked to **produce** something you must check,
or to **read** something MathSlate has already computed. The second is the safer
job, and most of what follows is on that side.

**`explain()` — what went wrong, and the line that fixes it.** A beginner's
first mistakes are shape mistakes, and MathSlate already answers them precisely;
what is missing is turning the paragraph back into code.

```text
plot(sin(x), 0, 6.28)
# TypeError: a range must be written (symbol, lo, hi); got 0

from mathslate.ai import explain
explain()
```

With no argument it reads the exception Python just reported, so the call after
a failed cell is simply `explain()`. Pass an exception to explain one you
caught, or `code=` to have a snippet reviewed without running it.

The error message is the evidence rather than the model's memory of MathSlate.
When MathSlate raised it the message is authoritative and the model is told to
turn it into a corrected line; when it came from elsewhere — a `SyntaxError`
from `solve(x**2 - 4 = 0)`, a `TypeError` from inside SymPy — the model is
reasoning from general Python knowledge and is asked to say so. Which half
applies is decided by *which frame raised*, not by the exception's class: a
mis-shaped range raises a plain `TypeError` on purpose (§4.1), and that is
exactly the case this exists for.

**`describe()` — what the answer means.** The roots are already solved and the
discontinuities already found. Only the sentence saying what they amount to is
missing, and that needs no arithmetic:

```text
from mathslate.ai import describe
describe(analyze(x**3 - 3*x))
describe(plot(tan(x)), "why is the line broken?")
```

The model is given the finished numbers and never computes. `facts()` is what
it is given, and is worth reading on its own — it is the honest answer to "what
did you send?":

```python
from mathslate.ai import facts

report = facts(analyze(x**3 - 3*x))
print(report["roots"]["exact"], report["roots"]["approximate"])
```

Every property carries `approximate` and the `method` behind it, so a solved
answer and a sampled one are never presented alike. This is the one place in
the module where a model's reply is shown as an answer rather than as a draft —
and `Analysis.rows()`, `Table.text()` and `PlotResult.summary()` remain the same
facts written by hand, needing no network at all.

**`Suggestion.repair()` — a second try, with the error as evidence.** A model
writing against a half-remembered API gets closer once it is shown what actually
happened:

```text
draft = ask("plot the tangent")
try:
    draft.run()
except MathSlateError as failure:
    better = draft.repair(failure)
```

The loop is deliberately left open. `repair()` returns a new `Suggestion` rather
than running it, because retrying is the part worth automating and skipping the
look is not. The original question travels with the error, so a repair that
quietly solves an easier problem is refused by the prompt.

**`ask(..., about=result)` — a follow-up that has an *it*.**

```text
drawn = plot(sin(x)/x)
ask("show the same thing on a log scale", about=drawn)
```

What is sent is the same computed summary `facts()` returns. The context is
*named* rather than collected: nothing about your session leaves the machine
because a question was asked near it. `assistant(question, about=...)` threads
the same context through the panel.

**`suggest_model()` — which curve to fit.** Choosing the model is the step
before the mathematics, and the one a beginner has least to go on. The shape is
in the numbers, so MathSlate measures it rather than asking the model to guess:
a family is whatever transform straightens the data.

```python
import numpy as np
from mathslate.ai import fit_evidence

xs = np.linspace(1.0, 5.0, 20)
readings = dataset({"x": xs, "y": 2 * np.exp(0.7 * xs)})

straightness = fit_evidence(readings)["straightness"]
print(max((name for name, r in straightness.items() if r),
          key=lambda name: straightness[name]["r"]))
```

On exponential data `log(y) ~ x` is 1.000 while the others sit near 0.94, so the
family is a measurement rather than an opinion. Each figure reports `rows_used`
against `of`, because `log` drops every non-positive value: on data crossing
zero the exponential correlation describes the positive tail alone, and a tail
is easily straighter than the whole. `suggest_model(readings)` hands those
numbers to the model, which names the family and writes the `.fit(...)` call.

#### 13.5.2 MathSlate as a tool for an outside agent

Everything above points outward — MathSlate asks a model. `mathslate.ai.tools`
points inward: an agent hands MathSlate an expression and gets computed
answers instead of a plausible recollection of them.

```python
from mathslate.ai import call, tool_names

print(tool_names())
print(call("mathslate_analyze", {"expression": "x**2 - 2"})["roots"]["exact"])
```

`tool_schemas("anthropic")` and `tool_schemas("openai")` emit the definitions in
either provider's shape, generated from one description. `mathslate_plot`
returns no image — a summary, the notes MathSlate attached, and the equivalent
plain program.

Arguments arrive from a model and reach SymPy, where a string is evaluated as
code, so a tool call is given exactly the trust a generated suggestion is given:
none. Each request is assembled into MathSlate source, put through the same
allowlist as a suggestion, and run in the same isolated process under the same
wall-clock budget. Symbol names must be identifiers and bounds are re-emitted as
numbers, so neither can carry source. A refusal comes back as `{"error": ...}`
rather than raising, because the caller is an agent that can read it and correct
itself.

### 13.6 Worksheets

```python
from mathslate.classroom import worksheet

page = worksheet([
    "Where does sin(x)/x go at zero?",
    ("The graph", plot(sin(x)/x, verbose=False)),
    ("The numbers", table(sin(x)/x, (x, -1, 1), rows=5)),
], title="Limits")
print(len(page))
```

Plots, tables, analyses and paragraphs on one page, in the order you wrote
them. `page.save("handout.html")` writes it. Plotly is embedded **once**
however many figures the page holds, so ten plots is not a 40 MB file, and the
result opens with no network and nothing installed — including its sliders,
which is why §11.8 built them from frames rather than notebook widgets.

For a smaller file on a networked machine, use `standalone=False` consistently
with `PlotResult.to_html()`:

```python
small = worksheet([plot(sin(x), verbose=False)], standalone=False)
print("cdn.plot.ly" in small.html())         # versioned Plotly CDN
page.preview(standalone=False)              # small notebook output
```

The default remains `standalone=True`; CDN pages need a network connection.

Text is escaped, not interpreted: a teacher's `<` is a less-than sign.

**Three ways to look at it, and they are not interchangeable.** `html()` is a
whole document — doctype, `<head>`, a stylesheet, a copy of Plotly — which is
right for a file and wrong for a notebook cell, where the browser would drop
the outer tags, keep the stylesheet, and restyle the notebook around it.

| Call | Gives you |
|---|---|
| `page.save(path)` | the file to hand out |
| `page.preview()` | the page inside an `<iframe>`, safe in a notebook |
| displaying `page` | a short card listing what is on it |

```python
from mathslate.classroom import worksheet

page = worksheet([("A graph", plot(sin(x), verbose=False))], title="One plot")
print(page.preview(height=400).startswith("<iframe"))
print("<!DOCTYPE" not in page._repr_html_())
```

---

## 14. Errors

All inherit from `mathslate.errors.MathSlateError`.

| Error | Raised when | Typical fix |
|---|---|---|
| `AmbiguousAxisError` | more free symbols than axes and no way to choose | give a range, or bind with `parameters=` |
| `UnsupportedInputError` | input matches no dispatch row, a bad option value, or a range or parameter that cannot mean anything | check §3, §4 and §5 |
| `NotYetImplementedError` | a documented API scheduled for a later milestone | none remain — §16 |
| `SamplingError` | nothing finite to draw | check the range and the real domain |

`AmbiguousAxisError` additionally carries `.question` and `.candidates`.

```python
from mathslate.errors import MathSlateError, SamplingError

try:
    plot(sqrt(-1 - x**2), (x, -1, 1))
except SamplingError as error:
    print(isinstance(error, MathSlateError), str(error)[:30])
```

```python raises=UnsupportedInputError
plot({"not": "plottable"})
```

```python raises=UnsupportedInputError
plot(sin(x), points=0)
```

---

## 15. Frontend adapters

Coupling to one notebook would buy reactive widgets and cost the environments
that need no installation, which are the ones a student is most likely to have.
So the frontend is detected at runtime, and installing MathSlate pulls in none
of them.

| Environment | `slider()` implementation (v0.5) |
|---|---|
| marimo | `mo.ui.slider`, reactive |
| Jupyter / Colab | `ipywidgets` |
| anything else | Plotly animation frames, self-contained HTML |

```python
from mathslate.ui import Frontend, detect_frontend

print(detect_frontend() in set(Frontend))
print(frontend_report())
```

Detection imports nothing: it inspects `sys.modules` and the IPython shell
class. Nothing in the numeric path consults it, so results are identical in
every environment — checked in CI by running one fingerprint through plain
Python, a Jupyter kernel and marimo.

---

## 16. Milestones

Every documented function exists. The ones not yet built raise
`NotYetImplementedError` naming their milestone, so following the API list
never produces a mysterious `AttributeError`.

| API | Milestone |
|---|---|
| `analyze()` | **v0.5 — shipped**, see §10 |
| `slider()`, `animate()` | **v0.5 — shipped**, see §11 |
| surfaces, contours, implicit curves, 3D space curves | **v0.5 — shipped**, see §4.11 |
| `table()`, single-file HTML export | **v0.5 — shipped**, see §12 |
| `dataset()`, statistics, linear algebra | **v1.0 — shipped**, see §13 |
| optional AI assistant | **v1.0 — shipped**, `mathslate.ai` |
| classroom worksheets | **v1.0 — shipped**, `mathslate.classroom` |

**Nothing is deferred.** Every name in §2 does what it says.

### Permanently out of scope

There is no `explain()`, no "show steps", no generated worked solutions — in
any spelling. SymPy provides no general step-by-step engine, and a partial one
would be worse than none. `analyze()` reports *properties of an object*
(roots, extrema, inflection points, symmetry, periodicity, asymptotes,
monotonic intervals), never a derivation.

MathSlate is also not a CAS, not a notebook, not Wolfram-compatible, not a
grading tool, not a high-performance numerics library, and not a GUI equation
editor.

---

## 17. Internals map

Stable enough to reference; not part of the public API contract.

| Module | Responsibility |
|---|---|
| `mathslate/api.py` | the public surface |
| `mathslate/core/dispatch.py` | input → plot kind, and the resolved `PlotPlan` |
| `mathslate/core/binding.py` | symbol → axis, ranges, conventional order |
| `mathslate/core/sampling.py` | the six-step algorithm of §6 |
| `mathslate/core/_sets.py` | SymPy `Set` algebra → plain float intervals |
| `mathslate/render/plotly_backend.py` | plan → `go.Figure`; **the only Plotly import** |
| `mathslate/render/options.py` | figure-level decisions, shared with `codegen` so the two cannot disagree |
| `mathslate/render/axes.py` | π ticks, log-scale hint |
| `mathslate/codegen.py` | `show_python()` |
| `mathslate/result.py` | `PlotResult` and the escape hatches |
| `mathslate/ui/adapters.py` | runtime frontend detection |
| `mathslate/_text.py` | encoding-safe console output |
| `mathslate/errors.py` | the error hierarchy |

`core` depends on SymPy and NumPy only — never on Plotly or on a frontend.
Plotly is confined to one file so that swapping the backend later stays cheap.

---

## 18. Guarantees and non-guarantees

### Guaranteed

- **The dispatch contract in §3 is stable.** It will not change without a
  version bump.
- **`show_python()` output runs**, for every plot kind, verbatim.
- **No line is ever drawn across a discontinuity.** Checked over a
  200-function corpus and a 30-case hand-checked review, both in the suite.
- **Installing MathSlate installs no frontend.** Checked by a clean-install job
  in CI.
- **Every result exposes `.sympy`, `.plotly` and `.numpy`** where they apply.
- **Results are identical across notebook environments.**
- **The console never crashes on a non-UTF-8 terminal.** Decorative Unicode
  transliterates to ASCII when the stream cannot encode it.

### Not guaranteed

- **Array-identical output between MathSlate and `show_python()`** — §9.
- **Exact sample counts.** They depend on the adaptive sampler and may change
  between versions; the *picture* is what is stable.
- **Performance beyond educational scale.** The design target is ≤10⁶ points
  with no perceptible lag; MathSlate is not a high-performance numerics
  library.
- **Symbolic analysis of arbitrarily exotic expressions.** Where SymPy cannot
  decide a domain, MathSlate falls back to numeric probing and says so in the
  notes. Read the notes.

---

## See also

- [Tutorial](tutorial.md) — the guided introduction.
- [PRD](../mathslate_prd_0.3.md) — the design rationale, the non-goals, and
  the implementation status with acceptance-criteria evidence.
- [`README`](../README.md) — the short version.
