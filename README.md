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

![plot-2D](https://raw.githubusercontent.com/berd2/mathslate/master/docs/images/plot-2D.png)

That is the whole first lesson. No `symbols`, no `lambdify`, no `linspace`, no
`figure`, no `show`. When you are ready for those, ask:

```python
plot(sin(x)/x).show_python()
```

and MathSlate prints the plain NumPy + SymPy + Plotly program that would have
produced the same picture — including the parts it did quietly on your behalf.

---

## Install

Pick whichever of these three matches you.

### 1. You already have a Python environment

```bash
pip install mathslate
```

Nothing frontend-specific comes with it. If you already have Jupyter and only
need plotting controls, add its notebook extra:

```bash
pip install "mathslate[jupyter]"   # pulls in ipywidgets + anywidget
pip install "mathslate[marimo]"
```

For the complete local learning setup — Jupyter Lab, interactive graph
controls, Gemini assistant, and secure API-key storage — use this one command:

```bash
pip install "mathslate[starter]"
```

`anywidget`, `keyring`, and the Gemini SDK are installed by the extras; there
is no need to find or install them one by one.

The assistant is optional: `pip install mathslate` stays fully usable for
plotting, analysis, tables, fitting and exports without an API key, an AI SDK,
or a network connection.

### 2. New to Python, or you'd rather not fiddle

From a checkout, after installing [uv](https://docs.astral.sh/uv/) once, one script creates an
isolated environment, installs Jupyter Lab, interactive controls, Gemini AI
and secure key storage, writes a starter notebook, and opens Jupyter Lab:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if `uv --version` already works
bash scripts/quickstart.sh
```

```powershell
irm https://astral.sh/uv/install.ps1 | iex   # skip if `uv --version` already works
scripts\quickstart.ps1
```

What that script runs, spelled out — useful if you'd rather type it yourself
or adapt it:

```bash
uv venv .venv-mathslate
uv pip install --python .venv-mathslate ".[starter]"
.venv-mathslate/bin/python -m jupyter lab
```

(`.venv-mathslate/Scripts/python.exe` on Windows. The environment is named
rather than the usual `.venv` so running this inside a clone cannot replace a
development environment already there.)

The one line every notebook needs, at the top of the first cell:

```python
from mathslate import *
```

That single import is *the whole setup* — `plot`, `analyze`, `sin`, `x` and
everything else in this README come from it. `ipywidgets`/`anywidget` are
never imported directly; MathSlate uses them internally for `slider()` and
range controls. The quickstart setup installs the Gemini assistant and secure
key storage too, so no further package installation is needed before calling
`assistant()`.

Swap the extra for `marimo` and the last line for `-m marimo edit` to use
marimo instead.

### 3. Just evaluating — no install at all

A live, in-browser preview (JupyterLite, no local setup) runs at
**https://berd2.github.io/mathslate/**. It's slower than a local install —
everything runs client-side via Pyodide — and needs one extra cell before the
first import:

```python
import piplite
await piplite.install("mathslate")
```

Treat it as a five-minute look, not a working environment; move to path 1 or 2
once you're sold.

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

![plot-3D](https://raw.githubusercontent.com/berd2/mathslate/master/docs/images/plot-3D.png)

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

Axes read the expression: trigonometry gets π ticks, `exp` does not. They also
read the *window*, which is the part you notice only when it is missing — zoom
in and the labels are re-fitted rather than left where the first draw put them.
A finer multiple of π while one still fits, plain numbers once none does, and
fewer of them as a deep zoom makes each one longer.

```python
plot(x*y, ticks=4)      # or cap them yourself — which is the only lever in 3D,
                        # where nothing re-lays labels as the camera comes in
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

## How big, and how much chrome

In Jupyter a bare `plot()` shows the figure with a live range-control sidebar
beside it. Both halves of that are yours to change:

```python
plot(sin(x), height=800)          # this plot, taller
plot(sin(x), controls=False)      # this plot, no sidebar — the cell is all graph
set_plot_size(height=720)         # every later plot, once at the top of a notebook
set_range_controls(False)         # every later plot, plain figure
```

Height defaults to 520px rather than Plotly's 450 — a notebook cell is not a
dashboard tile, and a fifth of the width goes to the sidebar. Width is left
unset on purpose so the figure fills its cell; set it when you want a fixed
size and mean it.

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

Writing code is one job. Most of the module does the other one — reading what
MathSlate has already worked out, which needs no arithmetic from a model and so
cannot be got wrong the same way:

```python
plot(sin(x), 0, 6.28)            # TypeError: a range must be (symbol, lo, hi)
explain()                        # ...and here is the corrected line

describe(analyze(x**3 - 3*x))    # prose about roots that are already solved
ask("now on a log scale", about=drawn)   # a follow-up that has an "it"
draft.repair(failure)            # a second try, with the error as evidence
suggest_model(readings)          # the transform that straightens the data picks the fit
```

`explain()` reads the exception Python just reported, so the call after a failed
cell is simply `explain()` — and the error message is the evidence, not the
model's memory of MathSlate. `describe()` is given the finished numbers and told
never to compute; every property it sees carries whether it was *solved* or
*sampled*.

Pointing the other way, `mathslate.ai.tools` makes MathSlate a tool an outside
agent can call, so Claude or ChatGPT gets computed roots rather than a plausible
recollection of them — with `approximate` and the method attached, so it knows
how far to trust each one. Arguments from a model are given exactly the trust a
generated suggestion is given: none.

See [the manual](docs/manual.md#1351-asking-for-something-other-than-code) for
the whole of it.

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

A guided tour of everything in v1.0, as a notebook you run yourself — 124 cells
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
- **[Understanding MathSlate](docs/guide.md)** — the ideas behind the examples:
  what `plot()` does in five steps, the three rules worth remembering, a
  task-to-function map, and how to read the common errors.
- **[Reference manual](docs/manual.md)** — every option, the full dispatch
  contract, the sampling algorithm, and the exact guarantees.
- [`docs/design/mathslate_prd_0.3.md`](docs/design/mathslate_prd_0.3.md) — the product requirements
  document, with an implementation-status section kept up to date.

Every `python` example in both documents is executed by the test suite in
document order, so the manuals cannot drift from the code.
