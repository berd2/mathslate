# MathSlate

**A mathematical workspace that grows with you.**

A mathematical workspace that orchestrates SymPy, NumPy and Plotly so that a
learner can go from their first graph to real scientific computing without ever
changing tools.

```python
from mathslate import *

plot(sin(x)/x)
```

```
curve | x ∈ [-10, 10] | 411 samples | 1 discontinuity handled
  · singularities at x = 0
```

That is the whole first lesson. No `symbols`, no `lambdify`, no `linspace`, no
`figure`, no `show`. When you are ready for those, ask:

```python
plot(sin(x)/x).show_python()
```

and MathSlate prints the plain NumPy + SymPy + Plotly program that would have
produced the same picture — including the parts it did quietly on your behalf.

---

## Status — v1.0, "growth environment" — complete

| Feature | State |
|---|---|
| 2D `plot()`: single curve, overlaid curves, parametric curves | ✅ |
| Adaptive sampling, singularities, domains, line breaking | ✅ |
| π-aware contextual axes | ✅ |
| SymPy re-exports and predefined symbols | ✅ |
| `show_python()` | ✅ |
| Jupyter / Colab / marimo | ✅ |
| 200-function corpus and regression tests | ✅ 200/200 |
| `analyze()` — first of v0.5 | ✅ |
| `slider()` / `animate()` — second of v0.5 | ✅ |
| 3D: surfaces, contours, implicit curves, space curves | ✅ |
| `table()` and single-file HTML export | ✅ |
| `dataset()`, statistics, linear algebra | ✅ |
| Optional AI assistant, classroom worksheets | ✅ |

Nothing is deferred. Every documented name does what it says.

## Install

```bash
pip install mathslate
```

Nothing frontend-specific comes with it. Add what your notebook needs:

```bash
pip install "mathslate[jupyter]"
```

```bash
pip install "mathslate[marimo]"
```

> **marimo users:** marimo rejects `import *` at parse time — it needs to know
> statically which names each cell defines in order to build its reactive
> graph. Import explicitly instead; everything else is identical. See the
> [manual](docs/manual.md#marimo-forbids-import-).
>
> ```python
> from mathslate import plot, polar, sin, cos, tan, exp, sqrt, x, y, t
> ```

## The one rule

`plot()` infers what you meant. There is exactly one thing to memorise:

- a **list** means *several things together*
- a **tuple** means *one vector-valued object*

```python
plot([sin(x), cos(x)])     # two curves, overlaid
plot((sin(t), cos(t)))     # one parametric curve
```

Everything else follows the dispatch contract:

| Input | Free symbols | Result |
|---|---|---|
| `Expr` | 1 | 2D curve |
| `Expr` | 0 | horizontal line + message |
| `list[Expr]` | 1 shared | curves overlaid |
| `tuple[Expr, Expr]` | 1 shared | 2D parametric curve |
| `callable` | — | numeric sampling |
| array-like | — | data series |
| `(xdata, ydata)` | — | scatter |
| `Expr` | 2 | surface, `kind="contour"` to flatten |
| `Eq(lhs, rhs)` | 2 | implicit curve |
| `tuple[Expr × 3]` | 1 or 2 | space curve / parametric surface |

Two cases inference cannot decide in principle, so you say them out loud:

```python
polar(1 + cos(t))                # r = f(θ) is indistinguishable from y = f(x)
plot(x*y, kind="contour")        # surface vs. contour
```

Every call reports what it inferred, in one line. That line is not logging: it
is where you first learn that parameters you never wrote exist.

## Which symbol becomes the axis

1. an explicit range wins — `plot(a*sin(x), (a, -1, 1))`
2. bound parameters are never axes — `plot(a*sin(x), parameters={a: 3})`
3. otherwise the conventional order: `x, y, z` → `t, u, v` → `r, θ` → alphabetical
4. if it is still ambiguous, MathSlate asks instead of guessing

## The hard part: discontinuities

`plot(tan(x))` drawing no spurious vertical lines is the feature that separates
MathSlate from a plotting wrapper. Everything else here is convenience.

```python
plot(tan(x))      # broken at every pole, y-window from the 2nd–98th percentile
plot(1/x)         # broken at 0
plot(floor(x))    # broken at every integer
plot(sqrt(x))     # never sampled outside its real domain
plot(x/abs(x))    # broken at 0
```

How, in order:

1. **Symbolic singularities** from `sympy.calculus.singularities` — never guessed.
2. **Real domain** from `sympy.calculus.util.continuous_domain` — never sampled outside it.
3. **Adaptive subdivision**: 200 uniform points, then midpoints wherever three
   adjacent points bend, to depth 8 and at most 5000 points.
4. **Line breaking**: NaN at every discontinuity. Jumps SymPy cannot see
   (`floor`, `sign`, `Piecewise`) are found by bisection — a genuine jump keeps
   its size as the interval shrinks, a steep slope does not.
5. **Y-clipping**: the visible window comes from the 2nd–98th percentile with
   near-pole samples excluded.
6. **Vectorised evaluation** via `lambdify(modules="numpy")`, falling back to
   element-wise evaluation loudly — never silently. The last element-wise tier
   is SymPy's own `evalf`, which is what makes `zeta`, `Si` and `besselj`
   plottable at all: no numeric backend carries them.

All six apply to parametric and polar curves as well. There the continuous
pieces are the intersection of both components' domains, the jump probe watches
`x(t)` as well as `y(t)`, and the window is clipped horizontally too — `x(t)`
can reach a pole in a way `x` never can when it is the axis.

```python
plot((tan(t), t))    # broken at t = π/2 and 3π/2, not drawn out to 10¹⁶
polar(tan(t))        # the same, through the polar reduction
```

## What a function *is*: `analyze()`

Explicit, never automatic — property detection is too slow and too noisy to run
on every plot.

```python
analyze(x**3 - 3*x)      # roots -sqrt(3), 0, sqrt(3); max at -1; min at 1
analyze(1/x)             # x = 0 vertical; y = 0 at both ends; odd
plot(tan(x)).analyze()   # analyses the window you are looking at
```

Roots, extrema, inflection points, symmetry, periodicity, asymptotes,
discontinuities and monotonic intervals. Exact where SymPy can solve it,
sampled where it cannot — and the sampled lines are **labelled approximate**,
because an approximation dressed as a proof is worse than no answer.

It reports what a function *is*. It never narrates how the answer was reached.

## Interactive: `slider()`

Bind a parameter and it becomes something the reader can drag. **You never
write a callback.**

```python
a = slider(-3, 3, default=1, name="a")
plot(a*sin(x))       # x is the axis, a is the parameter — inferred
animate(a*sin(x))    # the same, with a play button
```

The control is Plotly's own, carried inside the figure, so it needs no frontend
package and survives export to one self-contained HTML file — which is the
point if you are handing it to a class. `Slider.widget()` gives you
`mo.ui.slider` or `ipywidgets` instead when you want live recomputation.

## Your own numbers: `dataset()`

The bridge from symbolic to data. Write the model as you would on paper; get
**the same expression back with its parameters filled in**, still SymPy.

```python
readings = dataset({"x": [0, 1, 2, 3], "y": [1.0, 3.1, 4.9, 7.2]})
found = readings.fit(a*x + b)     # a = 2.05, b = 0.98, R² = 0.999
diff(found.expr, x)               # still an expression — everything works on it
```

Linear in the parameters is solved exactly; anything else is refined
iteratively. `.residuals` and `.r_squared` come back so the fit can be judged
rather than trusted.

```python
plot(readings)                    # the columns
plot(readings, kind="hist")       # their shape
plot(Matrix([[2, 1], [1, 3]]))    # a matrix as what it does, with eigenvectors
```

## Numbers, and one file to hand out

```python
table(sin(x), (x, 0, 1))          # the same function, read as values
plot(a*sin(x)).to_html("lesson.html")
```

`to_html()` embeds Plotly itself, so the page opens with no network and nothing
installed — sliders included. That is why the slider is built from frames
carried inside the figure rather than a notebook widget: a widget needs a live
kernel, and a file handed to a class does not have one.

## Escape hatches

Peeling the wrapper off costs nothing:

```python
f = plot(sin(x)/x)
f.plotly     # the Plotly Figure — yours to mutate
f.sympy      # the expression
f.numpy      # the sampled (x, y) arrays
f.python()   # the equivalent code, as a string
```

## Optional: the assistant, and worksheets

```python
from mathslate.ai import ask
print(ask("plot the tangent over one period").code)   # you read it, then run it
```

Claude, OpenAI or Gemini — install one and set its key, either as the provider's
environment variable or with `api_key=` on `ask()`/`configure()`. **The core never
imports any of it** and works fully offline, which is the point on a school
network; the test suite asserts that in a subprocess. `ask()` returns code, it
does not execute it. `Suggestion.run()` validates a restricted MathSlate subset
by default; unrestricted Python requires the explicit `unsafe=True` escape hatch.

```python
from mathslate.classroom import worksheet
worksheet([
    "Where does sin(x)/x go at zero?",
    ("The graph", plot(sin(x)/x)),
    ("The numbers", table(sin(x)/x, (x, -1, 1))),
], title="Limits", path="handout.html")
```

Plotly is embedded once however many figures the page holds.
Pass `standalone=False` to `worksheet(...)`, `.html()`, `.save()` or `.preview()`
to link the versioned Plotly CDN and keep networked outputs small.

## What MathSlate is not

- Not a CAS. All symbolic computation is SymPy's.
- **Not a step-by-step derivation engine.** There is no `explain()`, no "show
  steps", no generated worked solutions. SymPy provides no general
  step-by-step engine, and a partial one would be worse than none.
- Not a notebook or editor. It runs inside marimo and Jupyter.
- Not Wolfram Language compatible.
- Not a grading or LMS tool.
- Not high-performance numerics — educational scale (≤10⁶ points).
- Not a GUI equation editor. Input is Python code.

## Development

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev,jupyter,marimo]"
```

```bash
python -m pytest -q
```

The suite covers each PRD acceptance criterion in its own class, plus the
200-function corpus, a 30-case discontinuity review, a three-environment parity
check, and every example in `docs/`.

## Try it

A guided tour of everything in v1.0, as a notebook you run yourself — 95 cells
of curves, discontinuities, 3D, sliders, `analyze()`, tables, data fitting,
worksheets and refusals. Nothing in it needs a network.

```bash
marimo edit examples/mathslate_tour.py
```

```bash
jupyter lab examples/mathslate_tour.ipynb
```

The two are the same notebook: both are generated from
[`examples/build_tour.py`](examples/build_tour.py), and the test suite executes
the Jupyter one end to end on every run, so a tour cell that stops working
fails the build.

## Documentation

- **[Tutorial](docs/tutorial.md)** — a one-sitting walkthrough, from your first
  graph to reading real Python. Start here.
- **[Reference manual](docs/manual.md)** — every option, the full dispatch
  contract, the sampling algorithm, and the exact guarantees.
- [`mathslate_prd_0.3.md`](mathslate_prd_0.3.md) — the product requirements
  document, with an implementation-status section kept up to date.

Every `python` example in both documents is executed by the test suite in
document order, so the manuals cannot drift from the code.
