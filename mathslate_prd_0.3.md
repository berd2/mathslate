# PRD — MathSlate v0.3

**Package**: `mathslate` · **Import**: `import mathslate` · **PyPI**: available as of this writing

> **One-line definition**: A mathematical workspace that orchestrates SymPy, NumPy, and Plotly so that a learner can go from their first graph to real scientific computing without ever changing tools.

> **Tagline**: *A mathematical workspace that grows with you.*

---

## 0. Document History

| Version | Changes |
|---|---|
| 0.1 | Vision, philosophy, draft API |
| 0.2 | Added non-goals, success metrics, `plot()` dispatch contract, symbol-to-axis binding rules, frontend adapter architecture, adaptive sampling spec, roadmap, risks. Reduced API surface. Renamed `Guess()` → `analyze()`. |
| 0.3 | Named the project **MathSlate**. **Removed `explain()` / step-by-step derivation from scope entirely** (now an explicit non-goal). Added v0.1 acceptance criteria. Translated to English for coding-agent consumption. |
| 0.3.1 | **v0.1 implemented.** Added §11 Implementation Status with acceptance-criteria evidence. Resolved open decisions 1, 3 and 5 (§9); the rest still open. No change to scope, non-goals or the dispatch contract. |
| 0.3.2 | Code review of v0.1: §5.3 was reaching explicit curves only, and `plot()` accepted arguments that could not mean anything — §11.4 items 12–14. |
| 0.3.3 | **v0.5 started.** `analyze()` shipped — §11.7. No change to scope, non-goals or the dispatch contract. |
| 0.3.4 | **v0.5 complete.** `slider()`/`animate()` (§11.8), 3D and implicit curves (§11.9), `table()` and HTML export (§11.10). §2.2 re-examined against SymPy and upheld — §11.7. API surface 14 of the permitted 15. |
| 0.3.5 | **v1.0 complete.** `dataset()`, statistics and linear algebra, the optional AI assistant, and classroom mode defined as worksheet export — §13. Open decision 4 resolved. API surface 15 of the permitted 15. |
| 0.3.6 | Code review of v1.0 — §14. Two defects that produced wrong output or leaked a credential, four that were latent. §11 subsections reordered into numeric order; they had been appended out of sequence across three milestones. |
| 0.3.7 | Guided tour notebooks for marimo and Jupyter — §15. Generated from one cell list, executed by the suite. Building them found two ordering hazards and one error-message defect. |
| 0.3.8 | `plot()` drew nothing in marimo — §15.1. `PlotResult` delegated only `_repr_mimebundle_`, which Plotly leaves empty outside a Jupyter kernel. Every displayable type is now checked through the host's own formatter. |
| 0.3.11 | Polish round — §18. `mesh=True` rules 3D surfaces with grid lines (the Plot3D look); `xlim`/`ylim`/`zlim` set the view window as distinct from the sampling domain; the guided tour gained a Solving-and-algebra section; the manual documents the re-exported solve/factor/expand/gcd engine with a support table. Zero new top-level symbols. |
| 0.3.10 | **v1.5 — benchmarked against Mathics3 (§17).** Inequality regions and bands added to the dispatch contract; a wall-clock budget for symbolic steps; `exclusions=` as the override §5.3 was missing. Three of the reviewed proposals were declined or deferred on measured grounds, and the most valuable finding was not among them. §4's API metric now counts `plot()` keywords as well as top-level symbols. |
| 0.3.9 | Second review round — §16. Sixteen risks assessed; **five were not defects**, two of those refuted by measurement. Real findings: a dead example notebook, a warning filter that hid the notice §5.3 requires, surface clipping that reached the colours but not the geometry, and CSV reading that failed on any non-UTF-8 export. The `except Exception` sites are now a named, measured set rather than the narrower list proposed, which would have crashed on five real inputs. |

---

## 1. Problem Statement

A high school student who wants to explore mathematics visually has three options today, and all three are flawed.

| Tool | Strength | Flaw |
|---|---|---|
| Desmos / GeoGebra | Near-zero barrier to entry | Leads nowhere. Must be abandoned to learn programming. |
| Mathematica | Superb capability and defaults | Expensive, closed ecosystem; Wolfram Language transfers nowhere else. |
| Python (SymPy + Matplotlib) | Unlimited ceiling, lifelong skill | First graph requires `symbols`, `lambdify`, `linspace`, `figure`, `plot`, `show` — plumbing, not mathematics. |

**The hypothesis**: if the entry barrier of the third option is lowered to match the first, a learner can travel from high school algebra to research computing without ever switching tools.

The point is not "an easy API." Easy APIs are common, and most are discarded the moment the user outgrows them. The point is **continuity**.

---

## 2. Goals and Non-Goals

### 2.1 Goals

1. A first graph is possible without reading documentation.
2. Mathematically difficult functions (discontinuities, poles, restricted domains, rapid growth) render correctly with no user intervention.
3. The user can always see what the equivalent real Python was.
4. Peeling off the wrapper breaks nothing — zero lock-in.
5. The API is easy for an AI to generate correctly.

### 2.2 Non-Goals — **the scope defense line**

- ❌ **Building a CAS.** All symbolic computation is delegated to SymPy without exception.
- ❌ **Step-by-step derivations / worked solutions / tutoring explanations.** Out of scope for this project. SymPy provides no general step-by-step engine; a partial implementation would be worse than none. Do not add an `explain()` API, a "show steps" affordance, or LLM-generated derivations.
- ❌ **Building a notebook or editor.** Through v1.0, MathSlate runs inside marimo and Jupyter.
- ❌ **Wolfram Language syntax compatibility.**
- ❌ **Grading, LMS, or learning-management features.**
- ❌ **High-performance numerics.** Target scale is educational (≤10⁶ points). "No perceptible lag" is sufficient.
- ❌ **A GUI equation editor.** Input is Python code.
- ❌ **Reimplementing SymPy or Plotly functions.** Re-export them or expose them directly.

---

## 3. Users

### P1 — High school student (primary)
Learning calculus and trigonometry. Little or no Python.
*Success scenario*: "I want to see what happens to sin(x)/x at x = 0" → interactive graph in under 30 seconds.

### P2 — Undergraduate STEM student
Knows some NumPy and Matplotlib. The friction is boilerplate, every time.
*Success scenario*: gets a symbolic result while solving a problem set, then moves straight into numerical simulation in the same session.

### P3 — Teacher
Builds interactive classroom material and distributes it as a single HTML file.

### P4 — Researcher / engineer (secondary)
Fast sanity checks. Pulls the result straight out into Plotly or NumPy for their own pipeline.

---

## 4. Success Metrics

This is an educational project, so the metrics are quality gates, not usage counts.

| Metric | Target |
|---|---|
| Time to first success (P1, no docs) | 2D graph within 5 min; slider interaction within 15 min |
| Curriculum coverage | ≥95% of a 200-function high-school test corpus render correctly from a bare `plot(f)` |
| Discontinuity review | All 30 hand-checked cases pass (`tan`, `1/x`, `floor`, `sqrt` domain, `x/abs(x)`, etc.) |
| `show_python()` fidelity | 100% of emitted code runs verbatim and reproduces the same result |
| New API surface | ≤20 public symbols **and** ≤20 `plot()` keyword arguments, excluding SymPy re-exports — see §17.8, raised in §20 |
| Escape hatch completeness | Every returned object exposes `.sympy` / `.plotly` / `.numpy` where applicable |

---

## 5. Core Features

### 5.1 A single intelligent `plot()` — dispatch contract

Automatic inference is the right direction, but without a specification it becomes a black box the user cannot predict. The following table is a **documented contract**, not an implementation detail.

| Input form | Free symbols | Result |
|---|---|---|
| `Expr` | 1 | 2D curve |
| `Expr` | 2 | surface (default), with a contour toggle |
| `Expr` | 0 | horizontal line + informational message |
| `Eq(lhs, rhs)` | 2 | implicit curve |
| inequality | 2 | filled region (v1.5) |
| inequality | 1 | shaded bands on the axis (v1.5) |
| `list[Expr]` | 1 shared | multiple curves **overlaid** |
| `tuple[Expr, Expr]` | 1 shared | 2D parametric curve |
| `tuple[Expr × 3]` | 1 shared | 3D space curve |
| `tuple[Expr × 3]` | 2 shared | parametric surface |
| `callable` | — | numeric sampling |
| array-like | — | data series |
| `(xdata, ydata)` | — | scatter / line |

**There is exactly one rule to memorize**: a `list` means "several things together"; a `tuple` means "one vector-valued object." This matches ordinary Python intuition.

**Cases where inference is impossible in principle** — acknowledge honestly and require an explicit argument:
- Polar: `r = f(θ)` is indistinguishable from `y = f(x)`. → `plot(1 + cos(t), polar=True)`
- Surface vs. contour: default to surface; `kind="contour"` to switch.

**Every `plot()` call reports what it inferred, in one line.**

```
plot(x*y)
# → surface | x ∈ [-5, 5], y ∈ [-5, 5] | 60×60 samples   [view as contour]
```

This line is not logging. It is **the first step of progressive unboxing**: it is where the user first learns that parameters they never wrote exist, which is what leads them to start writing those parameters explicitly.

### 5.2 Symbol-to-axis binding rules

In `plot(a*sin(x))`, how does the system know `x` is the axis and `a` is a parameter? Without this rule the entire slider feature is unstable.

Resolution order:
1. An explicit range wins: `plot(expr, (x, -10, 10))`.
2. Symbols bound by `slider()` and friends are **excluded from axis candidates** and treated as parameters.
3. Remaining free symbols are sorted by convention: `x, y, z` → `t, u, v` → `r, θ` → then alphabetical.
4. If more remain than are needed, do not raise — **ask**: "3 free symbols found (x, y, a). Which should be the axes?"

### 5.3 Adaptive sampling and discontinuity handling — **the technical heart**

`plot(tan(x))` producing no spurious vertical lines is the single feature that separates "a Matplotlib wrapper" from "a tool that understands mathematics." Everything else in this document is convenience; this is differentiation.

Algorithm:

1. **Symbolic singularity detection.** Use `sympy.calculus.singularities()` to find poles first. Do not guess numerically.
2. **Domain computation.** Use `sympy.calculus.util.continuous_domain()` to get the real domain and never sample outside it (`sqrt(x)`, `log(x)`).
3. **Adaptive subdivision.** Start with 200 uniform points; recursively subdivide wherever the angle formed by three adjacent points falls below threshold. Max depth 8, total point cap 5000.
4. **Break the line.** Insert `None` at singularities and at NaN/inf regions so segments are not connected across them.
5. **Y-axis clipping.** Choose the y-range from the 2nd–98th percentile of samples, excluding those near poles.
6. **Vectorization.** Prefer `lambdify(modules="numpy")`; fall back to element-wise evaluation on failure — loudly, never silently.

### 5.4 Context-aware axes

- Expression contains trigonometric functions or `pi` → tick marks at multiples of π (interval auto-selected from π/4 to 2π based on range width).
- `exp` / `log` dominant → **suggest** a log scale; do not apply it automatically. A learner viewing a log-scaled plot without realizing it is harmful.
- Otherwise → standard numeric ticks.
- **The window, not the domain, decides.** All of the above are re-answered when the reader zooms: a π interval is re-fitted, dropped for numbers once no multiple of π lands often enough, and a numeric count is thinned to what its own labels have room for (Plotly rounds each label to the tick spacing and offers no offset line, so a deep zoom far from the origin spells its position out in full). A 3D scene positions labels in the projection and recomputes nothing on a camera move, which is what `ticks=` is for.
- `ticks=n` caps labels per axis, `ticks=False` removes them, `ticks=None` is the automatic choice. A ceiling rather than a target, and reproduced by `show_python()`.

### 5.5 `show_python()` — the actual mechanism of growth

In earlier drafts, "progressive unboxing" existed only as a philosophy. Here it is a **feature**. Given that the project's stated purpose is for the user to become comfortable with Python over time, this is not a secondary convenience — it is the flagship.

Every result object can return the equivalent code that produced it.

```python
f = plot(sin(x)/x)
f.show_python()
```

```python
# Equivalent code
import numpy as np, sympy as sp, plotly.graph_objects as go
x = sp.symbols('x')
expr = sp.sin(x)/x
fn = sp.lambdify(x, expr, "numpy")
xs = np.linspace(-10, 10, 1000)
ys = fn(xs)
fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines"))
fig.update_layout(template="plotly_white")
fig.show()
```

Requirements:
- The emitted code **must actually run** — this is a success metric, not decoration. No pseudocode.
- In the UI, expose it as a collapsed "Show Python" control next to the result.
- This is the user's first `lambdify` and first `linspace`. Hidden at first; revealed when they are ready.

### 5.6 `analyze()`

Explicit call, never automatic. Running property detection on every plot is slow and noisy.

```python
plot(x**3 - 3*x).analyze()
```

Detects: roots, extrema, inflection points, symmetry, periodicity, asymptotes, discontinuities, monotonic intervals.

- Compute exactly via SymPy wherever symbolically possible.
- On symbolic failure, fall back to numerical approximation and **label the result as approximate**.
- Present as a collapsed panel beside the plot.

Note: `analyze()` reports *properties of the object*. It does not explain *how* results were derived — see §2.2.

### 5.7 Interactive mathematics

```python
a = slider(-5, 5, default=1)
plot(a*sin(x))
```

The user never writes a callback binding. Implementation is per §6.2.

---

## 6. Architecture

### 6.1 Layers

```
mathslate/
├── core/            # Pure Python. Independent of frontend and plotting backend
│   ├── dispatch.py      # plot() input → plot kind inference
│   ├── binding.py       # symbol → axis / parameter binding
│   ├── sampling.py      # adaptive sampling, singularities, domains
│   └── analysis.py      # analyze()
├── render/          # → Plotly Figure (replaceable backend)
├── ui/              # frontend adapters (§6.2)
├── ai/              # optional. natural language → code. separate extras
└── api.py           # public surface + SymPy re-exports
```

The linear stack from v0.1 (`SymPy → NumPy → Plotly → Marimo`) does not describe the real dependency graph. `core` must not be tightly coupled to any of them.

### 6.2 Frontend adapters — **the key v0.1 decision**

Coupling to marimo gives reactive sliders for free but **loses Colab**. For a high school student, the largest funnel by far is the environment that requires no installation.

Therefore, runtime-detected adapters:

| Environment | `slider()` implementation |
|---|---|
| marimo | `mo.ui.slider` (reactive) |
| Jupyter / Colab | `ipywidgets` |
| Other / static export | Plotly animation frames (self-contained single HTML) |

`pip install mathslate` must not force a frontend dependency. Provide `mathslate[marimo]` and `mathslate[jupyter]` extras.

### 6.3 API surface — deliberately tiny

Most of the originally proposed API (`solve`, `simplify`, `expand`, `factor`, `diff`, `integrate`, `limit`, `series`, `matrix`) consists of thin wrappers around identically named SymPy functions. Under the "minimize new object types" principle, these should not be wrapped at all — they should be re-exported.

```python
# Re-exported (zero MathSlate code)
solve, simplify, expand, factor, diff, integrate, limit, series, Matrix, ...
x, y, z, t, n, k   # predefined symbols

# Genuinely new API — the complete list
plot()          # unified dispatch
polar()         # the case inference cannot resolve
slider()        # adapter
animate()
table()
analyze()
show_python()   # also a method on every result object
dataset()
```

**Eight new public symbols.** This is what makes the project tractable: the remaining real work concentrates entirely on inference, sampling, and rendering.

The display layer (automatic LaTeX rendering, pretty matrices) is already handled by SymPy's `_repr_latex_`. Supplement only the gaps via `_repr_mimebundle_`.

---

## 7. Roadmap

### v0.1 — "One graph, done properly" (4–6 weeks) — ✅ **shipped**, see §11

Scope:
- 2D `plot()`: single curve, overlaid curves, parametric curves
- **Adaptive sampling + discontinuity / singularity / domain handling** ← top priority
- π-aware contextual axes
- SymPy re-exports + predefined symbols
- `show_python()`
- Works in Jupyter, Colab, and marimo
- 200-function test corpus + visual regression tests

Excluded: 3D, sliders, AI, `analyze()`.

**Acceptance criteria (definition of done):**
1. `plot(tan(x))`, `plot(1/x)`, `plot(floor(x))`, `plot(sqrt(x))`, `plot(x/abs(x))` all render correctly with no arguments and no spurious connecting lines.
2. `plot(sin(x))` produces π-multiple tick labels; `plot(exp(x))` produces numeric ticks.
3. ≥95% pass rate on the 200-function corpus.
4. Every `show_python()` output in the test suite executes verbatim under `exec` without error.
5. Identical results across all three notebook environments in CI.
6. `pip install mathslate` pulls in no frontend dependency.

### v0.5 — "Exploration environment" — ✅ **shipped**, see §11.6
- 3D surface / contour / parametric surface / implicit
- `slider()`, `animate()` across all three adapters
- `analyze()` — ✅ shipped, see §11.7
- `table()`, single-file HTML export — ✅ shipped, see §11.10

### v1.0 — "Growth environment" — ✅ **shipped**, see §13
- `dataset()` — the bridge from symbolic to data — ✅
- Statistics and linear algebra visualization — ✅
- Optional AI assistant (separate extras; core remains fully offline) — ✅
- Classroom mode — ✅, **defined as worksheet export**; see §13

### v1.5 — "Benchmarked" — ✅ **shipped**, see §17

Scope came from reviewing Mathics3 rather than from the original plan.

- Inequality regions and bands — two new dispatch rows — ✅
- A wall-clock budget for symbolic steps, so `analyze()` degrades instead of
  blocking — ✅
- `exclusions=`, the override §5.3 lacked — ✅
- An mpmath evaluation tier: 15× on the special functions, and the actual cause
  of the slowness the budget was proposed for — ✅

Declined with reasons recorded (§17.7): NetworkX graph plotting, a third-party
plugin system. Deferred: vector and stream fields, browser-hosted frontends.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| `plot()` inference confuses users | Freeze the dispatch table as a documented contract; always display what was inferred; provide `kind=` override |
| marimo coupling costs Colab users | Adapter architecture (§6.2) |
| Adaptive sampling underestimated in difficulty | Make it the entire focus of v0.1; defer everything else |
| SymPy symbolic operations take seconds | Timeout + numeric fallback + progress indicator |
| Scope creep | Use §2.2 non-goals and the ≤20 symbol API cap as gates |
| AI dependency blocked on school or corporate networks | Core must work fully offline; AI ships as extras |
| Non-UTF-8 console turns the first `plot()` into a traceback | Encoding-aware output; found and fixed during v0.1 — §11.4 item 4 |
| Bisection jump detection mistakes a very steep curve for a jump | The probe compares gap *decay*, not gap size, and is bounded — §11.4 item 2 |
| SymPy version drift silently changes singularity/domain results | The 30-case discontinuity review pins exact cut locations — §11.3 |
| A host rejects usage the docs recommend, and CI never notices because it imports modules rather than typing cells | Documented usage is extracted **from the docs** and checked against the host's own parser — §11.4 item 9 |
| A rendering option reaches the figure but not `show_python()`, quietly breaking "reproduces the same result" | Both renderers consume one shared `RenderOptions`; every kind × option pair is compared figure-to-figure — §11.4 item 10 |
| A dispatch row is covered by tests but only on inputs that never exercise §5.3, so the sampler is untested there while the coverage table reads green | Every kind that draws a curve carries a hard case — a pole, a restricted domain and a step function — not only a smooth one; §11.4 item 12 |
| An argument is accepted, silently ignored or mis-bound, and fails later as a message about something else | Ranges, `parameters=` and `points=` are validated where they are read, and the message names the call that fixes it — §11.4 item 13 |

---

## 9. Open Decisions

1. ~~**Target curriculum**~~ — **decided during v0.1: curriculum-agnostic.** The 200-function corpus (`tests/corpus.py`) is assembled from function *families* that appear in every secondary syllabus (polynomial, rational, radical, exponential, logarithmic, trigonometric, absolute value, step, piecewise, composition), so it stays valid whichever curriculum is later targeted. Revisit only if a specific curriculum is adopted for marketing.
2. **Localization scope**: whether error messages and UI text need Korean i18n. *Still open.* v0.1 ships English-only messages; a related issue was found and fixed — see §11.4.
3. ~~**`from mathslate import *` policy**~~ — **decided: permit `import *`, and `show_python()` always emits the explicit `x = sp.symbols('x', real=True)`.** The proposed compromise was adopted verbatim and is pinned by a test. **Amended after a user report:** the decision is unimplementable in one of the three frontends §6.2 requires — **marimo rejects `import *` at parse time** and there is no opt-out, because a reactive notebook must know statically which names a cell defines. So `import *` is the documented default in Jupyter, Colab and plain Python, and an explicit import line is the documented default in marimo. See §11.4 item 10.
4. ~~**AI layer placement**~~ — **decided: extras within the core package.** `mathslate/ai/` is a few hundred lines; a separate distribution would cost more in version skew than it saves. `pip install 'mathslate[ai]'`, and the core imports none of it (§13).
5. ~~**Plotting backend**~~ — **decided: fixed on Plotly through v1.0.** The recommendation was adopted. The cost of reversing it is bounded by construction: `mathslate/render/plotly_backend.py` is the only module that imports Plotly.
6. **Namespace verification**: PyPI is clear, but GitHub org name and domain availability still need checking. *Still open* — outside the code.

---

## 10. Appendix — Changes from v0.1

| v0.1 | v0.3 | Rationale |
|---|---|---|
| Working title "MathStudio" | **MathSlate** | Original name collides with an existing mobile CAS app; "slate" also encodes the workspace-not-notebook positioning |
| Parallel `plot`, `plot3`, `surface`, `contour` | Single `plot()` + dispatch contract | Adopts the design note's intent, but pins the specification as a contract |
| `Guess()` | `analyze()`, explicit call | Old name implied inaccuracy; automatic execution is noise |
| `Explain()` as a headline feature | **Removed from scope entirely** (§2.2) | SymPy has no general step-by-step engine; a partial implementation is worse than none |
| Progressive unboxing (philosophy) | `show_python()` (feature) | Directly serves the project's stated purpose, so it becomes the flagship |
| Marimo in the stack | Frontend adapters | Preserves Colab accessibility |
| ~20 API functions | 8 new + SymPy re-exports | The logical consequence of "minimize new object types" |
| AI-first | AI-optional, offline core | Classroom and corporate network constraints |
| (none) | Non-goals, success metrics, acceptance criteria, risks | Required PRD elements |

---

## 11. Implementation Status

*Last updated: 2026-07-28. Reflects the state of the repository at PRD 0.3.1.*

### 11.1 What shipped

**v0.1 — "One graph, done properly" — complete.**
**v0.5 — "Exploration environment" — complete.**
**v1.0 — "Growth environment" — complete.** 1070 tests pass.

```
mathslate/
├── core/
│   ├── dispatch.py    plot() input → plot kind inference (§5.1 contract)
│   ├── binding.py     symbol → axis / parameter binding (§5.2)
│   ├── sampling.py    adaptive sampling, singularities, domains (§5.3)
│   ├── _sets.py       SymPy Set algebra → plain float intervals
│   ├── surfaces.py    two-variable sampling (§5.1's remaining rows)
│   ├── tables.py      table() — the same function as numbers
│   ├── data.py        dataset() — the bridge from symbolic to data
│   └── analysis.py    analyze() — properties of an expression (§5.6)
├── render/
│   ├── plotly_backend.py   plan → go.Figure; the only Plotly import
│   ├── options.py          figure-level decisions shared with codegen
│   └── axes.py             π-aware ticks, log-scale hint (§5.4)
├── ui/
│   ├── adapters.py    runtime frontend detection (§6.2)
│   └── interact.py    slider() / animate() (§5.7)
├── ai/                optional. natural language -> code. extras only
├── classroom.py       worksheet() — one page you can hand out
├── codegen.py         show_python() (§5.5)
├── result.py          PlotResult + escape hatches
├── _text.py           encoding-safe console output (see §11.4)
└── api.py             the public surface (§6.3)

docs/
├── tutorial.md        guided walkthrough, first graph → show_python()
└── manual.md          complete reference: API, contract, algorithm, guarantees
```

**User documentation ships with v0.1.** Both documents are English (§9 open
decision 2 remains open for UI strings; these are developer-facing prose).
Every `python` block in `docs/` is executed by `tests/test_docs.py` in document
order, in one namespace per document, exactly as a reader following along would
experience it. Blocks that are *meant* to fail declare it in the fence
(`` ```python raises=AmbiguousAxisError ``). A documentation example that stops
working fails the build — the same standard `show_python()` is held to.

Every module §6.1 anticipated now exists. `core/analysis.py` arrived with v0.5
item 1 (§11.7) and `ai/` with v1.0 (§13).

### 11.2 v0.1 acceptance criteria — evidence

| # | Criterion | Status | Where it is checked |
|---|---|---|---|
| 1 | `tan`, `1/x`, `floor`, `sqrt`, `x/abs(x)` render bare, with no spurious connecting lines | ✅ | `tests/test_acceptance.py::TestCriterion1NoSpuriousLines` |
| 2 | `plot(sin(x))` gets π ticks, `plot(exp(x))` gets numeric ticks | ✅ | `tests/test_acceptance.py::TestCriterion2ContextualAxes` |
| 3 | ≥95% pass rate on the 200-function corpus | ✅ **200/200 (100%)** | `tests/test_corpus.py`, corpus in `tests/corpus.py` |
| 4 | Every `show_python()` output executes verbatim under `exec` | ✅ | `tests/test_acceptance.py::TestCriterion4ShowPythonRuns`, `tests/test_codegen.py` |
| 5 | Identical results across all three notebook environments | ✅ | `tests/test_environments.py` — one fingerprint compared across plain Python, a Jupyter kernel (`nbclient`) and marimo |
| 6 | `pip install mathslate` pulls in no frontend dependency | ✅ | `tests/test_acceptance.py::TestCriterion6NoFrontendDependency` + a clean-install job in `.github/workflows/ci.yml` |

### 11.3 Success metrics (§4) — measured

| Metric | Target | Measured |
|---|---|---|
| Curriculum coverage | ≥95% of 200 functions | **100%** (200/200) |
| Discontinuity review | all 30 hand-checked cases | **30/30**, pinned by exact cut locations and domain-piece counts in `tests/test_discontinuity_review.py`; parametric and polar poles in `tests/test_parametric_sampling.py` (§11.4 item 12) |
| `show_python()` fidelity | 100% runs verbatim **and reproduces the same result** | **100%** on both halves: every plot kind executes in a clean namespace, and the resulting figure is compared property by property against the one MathSlate drew — `tests/test_codegen_fidelity.py`, 32 cases (see §11.4 items 10 and 12) |
| New API surface | ≤15 public symbols | **12** (`mathslate.NEW_API`) |
| Escape hatch completeness | `.sympy` / `.plotly` / `.numpy` everywhere | ✅ on every `PlotResult` |
| Time to first success | 2D graph in 5 min without docs | not measured — needs real users, not a test suite |

### 11.4 Deviations and decisions taken during implementation

Each of these is a place where the PRD did not fully determine the answer.

1. **§5.2 rule 4 — "do not raise, *ask*".** In a library there is no one to ask,
   so the question is delivered as `AmbiguousAxisError`, whose message *is* the
   question and includes the exact call that resolves it. When a UI layer
   exists (v0.5), it can catch this and render a real prompt.

2. **§5.3 step 4 — breaking the line at jumps SymPy cannot see.** `floor`,
   `sign` and `Piecewise` defeat `singularities()` and `continuous_domain()`
   alike (both raise `NotImplementedError` on `floor`). Rather than
   special-casing those heads, v0.1 uses a **bisection probe**: a genuine jump
   keeps its size as the bracketing interval shrinks; a steep-but-continuous
   slope does not. That single test handles every step function, `sign`, and
   `Piecewise` without naming any of them, and correctly leaves
   `atan(1000·x)` unbroken.

3. **§5.3 step 5 — y-axis clipping.** The bare 2nd–98th percentile window turned
   out to be too tight to read (`tan(x)` clipped to ±2.8). Implemented as:
   exclude samples within 2% of the window width of any pole, take the
   percentiles on a *uniform* re-read of the curve (adaptive refinement piles
   points up near poles and would drag the percentiles with it), widen the
   result threefold, then clamp back to values the curve actually reaches. That
   last clamp is what stops bounded curves such as `sin(x)/x` being padded with
   empty space.

4. **Console encoding — a real bug, found on the target platform.** The
   inference report (`x ∈ [-10, 10]`, `·`) and the generated code header (`—`)
   crashed with `UnicodeEncodeError` on a Windows console using a legacy code
   page — turning a learner's very first `plot()` into a traceback. Fixed in
   `mathslate/_text.py`: the stream's encoding is checked up front and the
   decorative characters transliterate to ASCII when it cannot take them
   (`∈`→`in`, `π`→`pi`, `—`→`--`). Notebooks still get the real Unicode.
   Covered by `tests/test_text.py`. **This is worth carrying into §8 as a
   risk**: the primary user (P1, a high school student on a school Windows
   machine) is exactly the person most likely to hit a non-UTF-8 console.

5. **`parameters=` on `plot()`.** §5.2 rule 2 requires slider-bound symbols to
   be excluded from axis candidates, but `slider()` is v0.5. To make the rule
   real and testable now, `plot()` accepts `parameters={a: 3}` — the symbols
   are excluded from axis selection and frozen at their value. This is the hook
   `slider()` plugs into in v0.5; no separate mechanism will be needed.

6. **Default ranges.** `[-10, 10]` for explicit curves; `[0, 2π]` for
   parametric and polar curves whose components involve trigonometric
   functions, because one full turn is far more useful than a symmetric numeric
   window. Not specified in the PRD; documented in the README.

7. **Deferred API is present, not absent.** `slider()`, `animate()`, `table()`,
   `analyze()`, `dataset()`, `Eq(...)`, 3-tuples and `kind="surface"` all exist
   and raise `NotYetImplementedError` naming their milestone. A user who
   follows the PRD's API list never hits an `AttributeError` and never wonders
   whether they typed something wrong.

8. **`kind=` and `yscale=` were accepted but inert.** Both were in `plot()`'s
   signature while writing the manual, and neither did anything for the values
   it claimed to take. `kind="scatter"` / `"line"` now actually select the
   trace mode, and `yscale` rejects anything other than `"linear"` and
   `"log"` instead of silently ignoring it. Writing the reference manual is
   what surfaced this: an option that cannot be documented truthfully is a bug.

9. **`import *` does not work in marimo — a conflict the PRD did not notice.**
   §9 decision 3 permits `from mathslate import *`, and §6.2 requires marimo as
   a first-class frontend. These are incompatible: marimo raises
   `ImportStarError` while *parsing* the cell, so the cell never runs. The
   reason is structural, not a marimo bug — a reactive notebook derives its
   cell dependency graph from the names each cell statically defines, and
   `import *` makes that set undecidable without executing the module. There is
   no flag to disable it.

   Reported by a user hitting it in a real marimo session, not caught by CI:
   the three-environment parity check (§11.2 criterion 5) executes a *module*
   in each host, which never exercises how a user types the import.

   Resolution, documentation-only — no API change: `import *` stays the
   documented default for Jupyter, Colab and plain Python; marimo gets a
   documented explicit-import line, carried in both the tutorial and the
   manual. `tests/test_marimo_imports.py` reads that line **out of the docs**
   (rather than restating it) and checks it against marimo's own parser, that
   it imports successfully, and that it covers every name the tutorial's
   examples use — which immediately caught two omissions. Worth noting that
   marimo's constraint pushes users toward explicit imports, which is the
   direction §9 decision 3 wanted anyway.

10. **`show_python()` reproduced the data but not the figure — caught in code
    review.** The success metric in §4 is "runs verbatim **and reproduces the
    same result**", and only the first half was being tested. Four figure-level
    options (`kind`, `yscale`, `title`, `show_legend`) reached the Plotly
    backend but not the code generator, so `plot(sin(x), kind="scatter")` drew
    markers and emitted `lines`; `yscale="log"` on a data or callable plot drew
    a log axis and emitted a linear one. Separately, callable plots emitted a
    call to the function *by name*, which cannot work: `np.sin` prints as
    `sin` (`NameError` in a clean namespace) and a lambda prints as `<lambda>`
    (`SyntaxError`).

    Root cause was structural, so the fix is too: figure-level decisions now
    live in `render/options.py` as a `RenderOptions` object that imports no
    plotting library, and **both** renderers — the Plotly backend and the code
    generator — call the same methods on it. An option that reaches one now
    necessarily reaches the other. Callable and data plots emit their samples
    as literal arrays, evenly reduced rather than truncated past 2000 points.

    The missing test was the real defect. `tests/test_codegen_fidelity.py`
    builds every plot twice — once through MathSlate, once by executing the
    emitted source in a clean namespace — and compares the two figures property
    by property, over every dispatch kind crossed with every render option.
    The compared and omitted property lists are exported as
    `render.options.REPRODUCED` / `NOT_REPRODUCED` so the manual cannot drift
    from them. Curves are compared point-to-segment inside the visible window,
    which is the only comparison that is meaningful for parametric and polar
    curves (x is not monotone) and across the NaN cuts of a broken curve.

    Writing that test immediately found two further defects nobody had
    reported: `plot(3)` emitted code that crashed, because `lambdify` returns a
    scalar for a constant expression and `np.concatenate` rejects a 0-d array
    (fixed with `np.broadcast_to`, which is exactly the trap a learner meets on
    their own first `lambdify`); and an over-long data series was **truncated**
    to its first 2000 points rather than reduced evenly, silently emitting a
    different picture.

11. **`polar()` shipped in v0.1**, though §7 places it after the v0.1 list. It is
   a four-line reduction to the parametric case that was already built, and it
   closes one of the two "inference cannot decide this" holes in §5.1 rather
   than leaving it open for a release.

12. **§5.3 reached explicit curves only — found in a second code review.**
    `sample_parametric()` was a bare `linspace` plus adaptive refinement: no
    `continuous_domain`, no `singularities`, no bisection probe, no NaN cuts,
    no window clipping. So `plot((tan(t), t))` drew a line out to 10¹⁶ with
    zero breakpoints — precisely the failure acceptance criterion 1 forbids for
    `tan(x)`, one dispatch row over. Item 11 above compounded it: `polar()`
    ships in v0.1 and runs entirely through this path, so `polar(tan(θ))` and
    `polar(1/cos(θ))` were affected too.

    Why no test caught it: every parametric and polar case in the suite —
    including the fidelity matrix of §11.4 item 10, which was otherwise
    exhaustive — used a bounded, everywhere-continuous curve (`(cos t, sin t)`,
    the cardioid). The matrix crossed every *option* with every kind, but only
    ever asked the easy *question* of two of those kinds. Coverage of the
    dispatch table is not coverage of the algorithm.

    Fixed by giving parametric sampling the whole of §5.3: pieces are the
    intersection of both components' domains and poles their union, the jump
    probe watches both coordinates (a jump in `x(t)` breaks the line as surely
    as one in `y(t)`, and `(floor(t), t)` has jumps in nothing else), and the
    cut blanks both. One addition the PRD does not anticipate, because it
    cannot arise for `y = f(x)`: the window is clipped **horizontally** as well.
    When x is the axis its extent is the range the user asked for; when it is
    `x(t)` it can run away exactly as y can. `show_python()` emits the
    piecewise sampler over `t` to match, and the four hard cases are now in the
    fidelity matrix.

13. **Arguments that could not mean anything were accepted.** The standard is
    item 8's — an option that cannot be documented truthfully is a bug — and
    these had not been held to it. Each was accepted and then either ignored or
    carried far enough downstream to fail with a message about the wrong thing:
    `plot(sin(x), (a, -1, 1))` made `a` the axis, left `x` free, and reported an
    empty domain; `parameters=[a]` (the §5.2 rule 2 form, documented in the
    manual) excluded `a` from the axes but left it in the expression, reporting
    the same; `points=0` silently sampled 200; `plot([])` raised `IndexError`
    out of the middle of dispatch; a string that failed to sympify was reported
    as an unknown type, though a string is a documented input; `yscale="log"`
    on a curve crossing zero let Plotly drop half of it without comment. All
    now name the problem and the call that fixes it. This matters more here
    than in most libraries: P1 (§3) cannot tell "I typed something meaningless"
    from "mathematics is hard" unless the error says which.

    The worst of the set was an error whose *advice* was broken. `plot(x*y)`
    hits the surface milestone (§5.1) and told the reader to give an explicit
    range; doing exactly that named `x` as the axis, left `y` free, and died in
    the sampler — the library walked its user into a second error. Naming an
    axis does not give the other symbols values, so the message now recommends
    `plot(expr, (x, -5, 5), parameters={y: 1})`, which draws one slice of the
    surface, and a test executes the recommendation rather than matching its
    text.

14. **The element-wise fallback could not rescue what NumPy had failed on.**
    §5.3 step 6 requires a fall back to element-wise evaluation, and the
    fallback ran through `lambdify(modules="math")`, which carries no more
    special functions than NumPy does. So `zeta`, `Si` and `besselj` reached
    the fallback, failed again, and surfaced as "produced no finite values" —
    an expression SymPy can evaluate exactly, reported as undefined. A third
    tier calling SymPy's own `evalf` fixes it. Separately, "loudly, never
    silently" was not being delivered: the `RuntimeWarning` was raised inside
    the sampler's own `simplefilter("ignore", RuntimeWarning)`, which is there
    to quiet NumPy's overflow chatter and swallowed this too. It is now
    re-emitted outside that block.

### 11.5 Risks discovered while building — now folded into §8

Three risks that the original §8 did not anticipate were found during v0.1 and
have been added to that table: a non-UTF-8 console, a false positive in jump
detection, and SymPy version drift. All three are mitigated and
regression-tested; `atan(1000·x)` is the pinned case for the second.

### 11.6 v0.5 — complete

In dependency order, as planned at the close of v0.1:

1. ✅ **`analyze()` → `core/analysis.py`** — shipped, see §11.7.
2. ✅ **`slider()` / `animate()` → `ui/interact.py`** — shipped, see §11.8.
3. ✅ **3D: surface, contour, parametric surface, implicit** — shipped, §11.9.
4. ✅ **`table()` and single-file HTML export** — shipped, §11.10.

### 11.7 `analyze()` — v0.5 item 1

All eight properties §5.6 names are implemented in `core/analysis.py`: roots,
extrema, inflection points, symmetry, periodicity, asymptotes, discontinuities
and monotonic intervals. `analyze(expr)` and `result.analyze()` are both
available; from a result the window analysed is the one on screen, because the
roots of `sin(x)` depend entirely on where the reader looked.

**The exact/approximate split is the whole design.** §5.6 requires exact
computation wherever symbolically possible and a *labelled* numeric fallback
otherwise. The signal turned out to be clean: `solveset` returns a `FiniteSet`
when it has really solved the equation and a `ConditionSet` when it has merely
restated it. Treating the latter as an answer would report "no roots" for
`x - cos(x)`, which has one. A numeric answer carries `approximate=True` out to
the printed panel; `tests/test_analysis.py` asserts the label on both sides of
the split, since an approximation presented as a proof is the same defect as a
wrong answer.

Five places where the obvious implementation is wrong, each now pinned by a
test:

1. **A zero of `f''` is not an inflection point.** `x**4` has `f'' = 12x**2`,
   which vanishes at the origin while the curve stays concave up throughout.
   Concavity must be seen to change.
2. **A stationary point is not an extremum.** `x**3` is stationary at the
   origin and rises on both sides. Worse, `x - cos(x)` has slope `1 + sin(x)`,
   which only *touches* zero — and the second-derivative test says `6e-17`
   rather than `0` there, because substituting the float `-1.5707963` into
   `cos(x)` is not the same as substituting `-pi/2`. The exact location is used
   where it is known, and a negligible curvature counts as zero.
3. **`AccumBounds` is not a limit.** SymPy reports the end behaviour of
   `sin(x)` as `AccumBounds(-1, 1)` — bounded, but not a value the curve
   approaches. Read as one, it gives the sine a horizontal asymptote.
4. **A vertical asymptote must be probed at the exact pole.**
   `limit(tan(x), x, 1.5707963)` is a large finite number; only
   `limit(tan(x), x, pi/2)` is infinite.
5. **A pole is not a root, and a stretch of zeros is not a hundred roots.**
   `1/x` changes sign across the origin without ever being zero — searching per
   continuous piece excludes it structurally. `floor(x)` is zero across all of
   `[0, 1)`, so the numeric search reports where each stretch begins and says
   that is what it did.

`diff(floor(x), x)` needed care of its own: it does not raise, it returns an
unevaluated `Derivative`, which `lambdify` cannot print at all. That is a
restatement rather than a derivative and is rejected, so the properties that
depend on it are reported as not computed instead of crashing.

**A stale test found along the way.** §11.4 item 9 describes
`tests/test_marimo_imports.py` as reading the documented import line out of the
docs "rather than restating it", and checking it covers "every name the
tutorial's examples use". The first half was true; the second was not — the
list of names was hardcoded in the test. So when the tutorial gained
`analyze()`, the documented marimo import line did not, and the test that
exists to prevent exactly this said nothing. The names are now extracted from
the documents' own code blocks with `ast`, which immediately found five further
omissions in the manual (`Eq`, `Integer`, `atan`, `frontend_report`, `slider`).
A hand-kept list is a second copy of the truth; this one had already drifted.

**Scope.** `analyze()` reports properties of the object and never how they were
obtained (§2.2). `tests/test_api_surface.py` asserts the absence of `explain`,
`steps`, `derive` and friends on the `Analysis` object specifically, because
that object is where such a thing would be most tempting to add.

**§2.2 was re-examined and stands — but the honest half of "how" now ships.**
The question was whether a meaningful description of *how* a result was
reached is possible after all, since §2.2 excluded it on the judgement that it
was not. Checked against SymPy rather than from memory:

- SymPy's **only** step-by-step engine is
  `sympy.integrals.manualintegrate`, which returns a rule tree (`PartsRule`,
  `SinRule`, …). It gives up with `DontKnowRule` on ordinary cases —
  `sqrt(x**3 + 1)`, `gamma(x)` — which is the "partial implementation is worse
  than none" of §2.2, demonstrated.
- It covers integration. **`analyze()` performs no integration.** It solves,
  differentiates and takes limits, and SymPy exposes no derivation trace for
  any of the three.

So a narration of how would have to be *invented by the model*, which is
exactly the LLM-generated derivation §2.2 forbids by name. The ban is correct
and unchanged.

What *is* knowable exactly, and was simply not being reported, is the
**receipt**: which call was made, whether it solved or fell back, and what
witnessed each classification. That is not a derivation, and it is already the
project's idiom — it is `show_python()` (§5.5, the flagship) applied to
analysis. Every property now carries a `method`, `Analysis.provenance()`
returns them, and `Analysis.python()` emits the calls as source that runs
verbatim and reproduces the same values. Sampled properties emit literals with
a comment naming the search, because emitting a `solveset` call that could not
solve the thing would be a lie that happens to run.

Writing it found a defect of the kind §5.5 exists to catch: the emitted
templates hardcoded `x`, and a post-hoc string patch could not fix that
(`sp.diff(expr, x)` has no trailing comma to match), so every expression in
`t`, `u` or `theta` emitted code that died with a `NameError`. The templates
now carry the report's own symbol, and a test executes the output for
non-`x` symbols.

**Known limitation.** §8 lists "timeout + numeric fallback" as the mitigation
for slow symbolic operations. Every symbolic step here is independently
guarded and falls back, but there is no hard timeout: a real one needs threads
or signals, and signals do not work off the main thread. A pathological
expression can therefore still make `analyze()` slow. It is an explicit call,
never automatic, which bounds the damage.

**API surface.** 13 of the 15 permitted public names are now used (`Analysis`
is the new one).

---

### 11.8 `slider()` / `animate()` — v0.5 item 2

§5.7's requirement is that **the user never writes a callback binding**, in all
three environments of §6.2. Two things had to be true for `a = slider(-5, 5)`
followed by `plot(a*sin(x))` to work.

**A slider has to survive being put inside a SymPy expression.** `Slider` is
not a `Symbol` — it carries a range, a step and a current value — but it
answers `_sympy_()` with its symbol, which is the hook `sympify` looks for. So
`a*sin(x)` builds an ordinary expression in an ordinary symbol and nothing
downstream learns a widget was involved. Only plain-number arithmetic (`a*2`)
needs explicit dunders, because `int` cannot sympify us.

**`plot()` has to bind that symbol as a parameter, not an axis.** That is §5.2
rule 2, which until now the user had to restate by hand via `parameters=`
(§11.4 item 5). A registry in `core/binding.py` — plain `symbol -> float`, so
`core` still knows nothing about widgets — now makes it automatic. Precedence
is unchanged and tested: an explicit range beats a slider, and an explicit
`parameters=` entry beats the slider's current value.

**Deviation from §6.2's table, deliberately.** §6.2 assigns `mo.ui.slider` to
marimo, `ipywidgets` to Jupyter/Colab, and Plotly animation frames to "other /
static export". MathSlate uses the **frames everywhere**: they behave
identically in all three hosts, they need no frontend package at all — which is
acceptance criterion 6 — and they survive being written to a single HTML file,
which is what §3's teacher (P3) hands out and what §7 lists as a v0.5
deliverable in its own right. The cost is that positions are pre-computed
rather than recomputed on demand, so `Slider.widget()` returns the host's
native control for readers who want the latter. Neither package became a
dependency.

**Scope taken honestly.** One parameter animates; others hold their values and
the report says so. Plotly's frames are a one-dimensional sequence, and a grid
over several sliders would multiply out into thousands of pre-computed curves.

`show_python()` emits the frames, not a still picture — a still one would not
be "the same result" (§4) for a figure that moves.

**A lifetime problem, found by the test suite.** A slider binds its symbol for
the session, which is right in a notebook and is what makes rule 2 work without
restatement. It also means one test's `slider(name="a")` silently turned `a`
into a parameter for every test after it — and it did: a slider in the manual's
own examples reached `tests/test_input_validation.py` and changed what that
module observed. `interact.release_all()` exists to draw the line, and
`conftest.py` calls it between tests. Worth noting because the same surprise is
available to a user who re-runs a notebook cell.

### 11.9 3D and implicit curves — v0.5 item 3

The four remaining rows of the §5.1 table are implemented: `Expr` with two free
symbols is a surface with `kind="contour"` as the documented toggle,
`Eq(lhs, rhs)` is an implicit curve, and a three-component tuple is a space
curve over one parameter or a parametric surface over two. Dispatch had already
routed each row to its own milestone-named error, so each became a local
change, as §11.6 predicted.

**Two variables cannot promise what one variable does, and the code says so.**
§5.3 steps 1 and 2 ask SymPy exactly where a curve is real and exactly where it
blows up. Neither question has a usable answer here: `continuous_domain` takes
a single symbol, and there is no two-variable `singularities`. So
`core/surfaces.py` evaluates on a grid, turns everything non-real or non-finite
into `NaN` so Plotly leaves a hole rather than drawing a wrong surface, and
clips the colour range on the principle of step 5 — a surface with a pole is
worse off than a curve with one, because the colour scale collapses as well as
the axis. The reduction in guarantees is reported in the notes (`"50% of the
grid is not a real number"`), not glossed over.

An implicit curve is drawn as the single zero level of a contour. That is the
only honest rendering on a grid: there is no parametrisation to sample, and no
guarantee the solution set is even a curve.

**Defaults.** `(-5, 5)` on each axis rather than a curve's `(-10, 10)`: a 60×60
grid over the wider window resolves far less than 200 adaptive points along a
line. §5.1's own example reports "60×60 samples", which is where the grid size
comes from.

**Two bugs the tests caught, both about what counts as an axis.**

1. `plot(a*sin(x))` with a slider became a *surface over x and a*. Surface
   detection ran before `_apply_parameters` folded in the slider registry, so
   it saw two free symbols where the substituted expression has one. The
   headline example of §5.7 — the feature shipped one item earlier — was broken
   by the next item. Bound parameters are now subtracted at detection time.
2. `plot(x*y, (x, -1, 1), (y, -2, 2))` fell through to the curve planner,
   because the test asked for a still-free symbol and both had ranges. Two
   ranges say "surface" as plainly as two free symbols do.

`show_python()` reproduces the grid *exactly* for every two-variable kind — a
uniform grid has nothing adaptive about it, so §11.4 item 10's documented
"same curve, not array-identical" caveat does not apply. It is emitted as a
`meshgrid` plus one `lambdify` rather than as 3600 literals, which is both
shorter and closer to what a reader would write.

### 11.10 `table()` and HTML export — v0.5 item 4

`table()` evaluates over an evenly spaced window and returns a `Table` with the
usual escape hatches (`.numpy`, `.sympy`), a text rendering, a notebook
rendering and a `.python()` that prints the same rows. `result.table()`
tabulates the window on screen, as `result.analyze()` does.

A cell where the expression is not a real number is **blank**, not zero. A
table that printed a number there would be lying about the mathematics, and the
count of blanks goes in the notes.

`result.to_html(path)` writes the figure as one self-contained page with Plotly
embedded — no network, nothing installed. That is P3's requirement from §3
("distributes it as a single HTML file"), and it is the reason §11.8 chose
Plotly frames over a notebook widget: a widget needs a live kernel and a file
handed to a class does not have one. The two decisions are one decision, and
the export test asserts an interactive plot keeps its slider through the trip.

**API surface: 14 of the permitted 15** (`Table` is the new name). One left.

---

## 12. v0.5 retrospective

All four items of §11.6 shipped. Three notes worth carrying into v1.0.

**Each item broke the one before it, and the tests caught it every time.**
Surfaces turned `plot(a*sin(x))` — the headline example of §5.7, shipped one
item earlier — into a surface over `x` and `a`, because surface detection ran
before slider parameters were folded in. `analyze()` reached
`tests/test_input_validation.py` through the global slider registry and changed
what that module observed. Neither was found by review; both were found by a
test written for something else.

**The documentation tests earn their keep more than any other kind.** Executing
every block in `docs/` caught four separate cases of the docs promising a
`NotYetImplementedError` that the code no longer raised. And the marimo import
extraction (§11.7) caught six omissions across the milestone — `analyze`,
`animate`, `table`, `dataset`, `Eq`, `Integer` — each of which would have been
a `NameError` for a marimo reader following the manual.

**The one thing still not measured** is §4's first row: "time to first success,
P1, no docs". It needs real users, not a test suite, and it is the metric the
whole design is aimed at.

## 13. v1.0 — "Growth environment"

All four roadmap items shipped. Two needed a decision the PRD had deliberately
left open, and both were put to the project owner rather than guessed.

### 13.1 `dataset()` — the bridge

The one-line roadmap entry is "the bridge from symbolic to data", and the
bridge is `Dataset.fit`. The model is written the way it would be on paper —
`a*x + b`, `a*exp(b*x)` — and what comes back is **the same expression with its
parameters substituted**, still a SymPy object. So a fit can be differentiated,
solved, analysed and plotted by everything built in v0.1 and v0.5, with no
conversion and without leaving the symbolic world. `core/data.py` is otherwise
plumbing that gets you to that method.

Linearity **in the parameters** is what decides the algorithm, not linearity in
the variable: `a*x**2 + b` has an exact least-squares answer and gets one,
while `a*exp(b*x)` is refined by Gauss–Newton. `.residuals` and `.r_squared`
come back either way so a fit can be judged rather than trusted, and on
constant data `r_squared` is `None` rather than 1 or 0 — R² compares against
"predict the mean", and there is nothing there to beat.

### 13.2 Statistics and linear algebra

`kind="hist"` and `kind="box"` ask about the shape of a column rather than its
order. Histograms are overlaid rather than stacked, because stacking two
answers a question about their sum that nobody asked.

A 2×2 `Matrix` is a new dispatch row, drawn as **what it does**: the unit
square, its image, and the real eigenvectors. A matrix of numbers on a page is
not a picture of anything; where it sends things is. The determinant is
reported with its meaning ("areas are scaled by 5"), a rotation says it has no
real eigenvectors rather than drawing nothing, and the aspect ratio is locked —
otherwise the distortion the reader sees is the plot's rather than the matrix's.

### 13.3 The AI layer — open decision 4, resolved

**Decision: extras within the core package.** `mathslate/ai/` is a few hundred
lines, and a separate distribution would cost more in version skew than it
saves. Three backends — Claude, OpenAI, Gemini — behind one small protocol, at
the project owner's direction; none is bundled and none is a runtime dependency.

Three properties, and the first two are absolute:

- **The core never imports it.** §8's risk line requires MathSlate to work
  fully offline because school and corporate networks block AI endpoints.
  `import mathslate` does not reach `mathslate.ai`, and the provider SDKs are
  imported *inside* the call that needs them, so even `import mathslate.ai` is
  free. Both are asserted in a subprocess, because properties of import graphs
  rot silently.
- **It returns code; it does not run it.** A model writing Python that the
  library executes unseen is the one thing a learner cannot check, and checking
  is the point of the project — `show_python()` exists so the real Python is
  always visible. `Suggestion.run()` exists and is one line, but it is a line
  the user types.
- **The system prompt is generated from `dispatch.KINDS`**, so a new plot kind
  cannot be added without the assistant being told in the same commit. It is
  about 2 kB, which is the measurement §4's goal 5 ("the API is easy for an AI
  to generate correctly") was asking for: the prompt is short because the API is
  small.

### 13.4 Classroom mode — defined as worksheet export

The roadmap named "classroom mode" and defined it nowhere. Rather than invent a
meaning, the choice went to the project owner, who chose **worksheet export** —
which is also the only concrete thing the PRD says about classroom use, in §3's
P3: "builds interactive classroom material and distributes it as a single HTML
file".

`mathslate/classroom.py` composes plots, tables, analyses and paragraphs into
one page. Plotly is embedded **once** however many figures the page holds, so
ten plots is not a 40 MB file, and an interactive plot keeps its slider through
the trip — which is the payoff for §11.8's choice of frames over notebook
widgets, made three milestones earlier for this reason. Text is escaped rather
than interpreted, so a teacher's `<` is a less-than sign.

Nothing in it collects, transmits or grades anything. §2.2 rules out grading
and LMS features, and a handout is not a loophole in that.

### 13.5 The API surface is now full

**15 of the permitted 15** (§4). `Dataset` was the last one available, which is
why `worksheet()` lives in `mathslate.classroom` and `ask()` in `mathslate.ai`
rather than at the top level. The budget did its job: it forced the two
optional, non-core features into namespaces where they belong anyway, and any
future addition now has to displace something.

### 13.6 What is still not measured

§4's first row — "time to first success, P1, no docs: 2D graph within 5 min" —
remains the one success metric with no evidence behind it. It needs real
learners, not a test suite. Every other row in that table is now asserted
somewhere in `tests/`.

---

## 14. v1.0 code review

A review of the shipped v1.0, requested after the milestone closed. The test
suite was green at 1084 tests throughout, so everything below is something the
suite was not looking at. Two findings produced wrong output or leaked a
credential; four were latent. The package version stays at `0.1.0` by decision
of the project owner — it tracks the distribution, not this document.

### 14.1 A slider could silently eat the last axis

`slider()` binds its symbol for the whole session — deliberately, since that is
what makes §5.2 rule 2 work without the user restating it (§11.8). But
`_apply_parameters` folded every bound symbol in without checking that an axis
survived. So:

```python
time = slider(0, 10, default=3, name="t")   # an ordinary thing to call a slider
plot((cos(t), sin(t)))                       # meant to be a circle
```

drew two hundred samples of the single point `(cos 3, sin 3)`, reported
`kind="parametric"`, and issued no note. Naming a slider `x` turned every later
`plot(sin(x))` into a horizontal line the same way.

Rule 2 removes symbols from the axis candidates; it was never meant to empty
the pool. `_reject_axis_eating_sliders` now refuses when freezing the
control-bound symbols would leave fewer axes than the plot kind needs, and
names the three ways out: an explicit range (which already outranked the
slider), `release_all()`, or a different slider name. Binding one of two
symbols is untouched — a surface legitimately becoming a curve is the ordinary
case, and is now pinned by a test that says so.

Explicit `parameters=` is exempt: that is the user speaking in the same call.

**The remedy was unreachable.** `release()` and `release_all()` existed but were
exported from neither `mathslate.ui` nor documented anywhere; the only code
using them was the test suite's own isolation fixture, which had been added
after this exact class of leak was observed between test modules. They are now
part of `mathslate.ui` and documented in manual §11.5.

### 14.2 An explicit API key went to whichever provider was first

`resolve_provider` consulted `api_key` only *after* returning the first
configured provider, so on a machine with `ANTHROPIC_API_KEY` in the
environment and both SDKs installed:

```python
ask("...", api_key="sk-proj-an-openai-key")   # -> Anthropic's client
```

the user's OpenAI credential was placed in an Authorization header and sent to
Anthropic. With no environment variable set it picked "first installed", with
the same result.

Nothing in a key string reliably identifies its service, and this project's own
rule for that situation is to ask rather than guess (§5.2 rule 4). An explicit
`api_key` with more than one provider SDK installed is now refused with the
candidates named; with exactly one there is no ambiguity and it is used; naming
the provider always works.

Related: `configure()` treats `None` as "leave as it was", which left no way to
remove a key from a session. Added `forget()`, and `configured()` which reports
what is held without printing the key.

The same rule applies across calls: a key configured for one provider is never
carried into another provider when `configure()` changes the default. A
per-call key also does not inherit a session provider implicitly when several
SDKs make its destination ambiguous.

### 14.3 A worksheet restyled the notebook it was shown in

`Worksheet._repr_html_` returned `html()` — a complete 4.8 MB document with a
doctype, a `<head>`, a `<style>` carrying a `body { max-width: … }` rule, and
an embedded copy of Plotly. A notebook injects that into the page: the browser
discards the outer tags and **keeps the stylesheet**, so displaying a worksheet
restyled the notebook around it until reload, and wrote 4.8 MB into the
`.ipynb` on every save.

`save()` and `html()` are unchanged — a file wants a whole document.
`_repr_html_` is now a short card naming the items, and `preview()` puts the
page in a sandboxed `<iframe>`, which is the document boundary a whole document
needs.

### 14.4 The non-linear fit was undamped

`core/data.py` described its iteration as "Levenberg–Marquardt-ish". It was
plain Gauss–Newton: full linearised step, no damping, no line search. On
`a/(x + b) + c` the first step threw `b` from 1 to −0.23, across the pole, and
the search never recovered — then a transient rank-deficiency 200 iterations
later **raised**, discarding the best point it had already found.

Now genuinely damped: the step solves `(JᵀJ + λ·diag(JᵀJ))·s = Jᵀr`, is kept
only if it lowered the cost, and λ adapts either way. Rank-deficiency mid-search
is no longer fatal — it is where the search is standing, not a property of the
model — while rank-deficiency at *every* starting point still refuses, because
`a*b*x` really cannot separate `a` from `b`.

Damping stops a step bolting but cannot cross into another basin, so the fit
also runs from a few deterministic starts drawn from the data's scale;
`a*exp(b*x)` from `b = −3` used to descend into a flat region and report it.
The model and its derivatives are now lambdified **once** with the parameters
as arguments instead of being `subs`-ed and recompiled every iteration, which
is what made several starts affordable — the whole thing runs in milliseconds.

### 14.5 The fidelity net had not been extended for two milestones

`tests/test_codegen_fidelity.py` was built during the v0.1 review precisely to
stop `show_python()` drifting from the figure, and it caught five defects then.
Every one of its 31 cases was still a v0.1 kind: surfaces, contours, implicit
curves, space curves, parametric surfaces, matrices, histograms, box plots and
animation frames had **no fidelity case at all**.

The implementations turned out to be correct, which is luck rather than
process. Fourteen cases added, plus frame-level comparison for sliders, plus
`TestEveryDispatchKindIsCovered`, which fails when a member of
`dispatch.KINDS` has no case — so the net now extends itself by refusing to
stay behind. Writing it immediately exposed that space curves are sampled
adaptively but emitted on a uniform grid, so they are compared by shape rather
than element-wise, exactly as 2D curves already were.

### 14.6 Smaller things

- `Dataset` was `frozen=True` around a mutable `dict` of writable arrays, so
  the length and finiteness checks in `__post_init__` could be voided a line
  later. Columns are now copied on construction — the caller's array stays
  theirs — and handed back as read-only views behind a `MappingProxyType`.
- Fitting a single-column dataset fitted a column against itself and reported
  R² = 1. Refused.
- A CSV containing only a newline raised a bare `IndexError`, and duplicate
  header names silently collapsed into one column. Both now behave: a
  `MathSlateError` with a sentence, and unique names such as `a`, `a_1`, `b`.
  Generated names are reserved too, so `x,x,x_1` cannot collide a second time.
- `classroom.Item` was annotated as a string literal rather than a type alias.

### 14.7 What this round says about the process

Both §14.1 and §14.2 are the same shape: a global that outlives the call, and
no test asking what happens to the *next* call. The slider registry's leak had
already been observed once — between test modules — and answered with a
fixture that hid it from the suite instead of a guard that stopped it reaching
users. A fixture that exists to contain a bug is a bug report.

§14.5 is the counterpart: a safety net that was not extended when the thing it
protects grew. The remedy is the same in both cases and now in place — the
coverage test that fails on an uncovered kind, and regression tests that pin
the session behaviour rather than isolating it away.

---

## 15. The guided tour notebooks

Reviews kept finding defects the 1084-test suite was green through. Every one
of them (§14) shared a shape: the tests asked whether a *call* was right, and
never whether a *session* was. So v1.0 gains the one check a test suite
structurally cannot make — a person running the thing.

`examples/mathslate_tour.py` (marimo) and `examples/mathslate_tour.ipynb`
(Jupyter) are a 95-cell tour of everything shipped: the first graph, the five
hard functions with their cut locations printed as evidence, π axes, the
list/tuple rule, polar, `show_python()` executed in an empty namespace, the
escape hatches, `analyze()`, `table()`, sliders and animation, 3D surfaces,
contours, implicit curves, space curves, parametric surfaces, matrices as
transformations, `dataset()` with linear, exponential and pole-bearing fits,
histograms and box plots, a worksheet, the AI layer's availability report, and
a section that deliberately triggers six refusals so the error messages are on
the page too.

**They are the same notebook, by construction.** Both are generated from
`examples/build_tour.py`, where the cells live once; the marimo file is emitted
from that list and `marimo export ipynb` derives the Jupyter one. Two hand-kept
notebooks would disagree within a week.

Three things fell out of building it, which is the point of building it:

- **marimo cell signatures cannot be typed by hand.** They are computed from
  each cell with the same AST rules marimo uses. The first attempt put the
  title cell before the imports — harmless in marimo, which resolves from its
  graph, and a `NameError` in the Jupyter twin, which runs in file order. The
  generator now refuses any cell that uses a name a later cell defines, so one
  ordering satisfies both.
- **A reactive notebook has no order between independent cells.** The tour's
  slider section originally recovered from the name collision of §14.1 with a
  bare `release_all()`, in a cell with no dependency on the cells still using
  the live slider — so marimo was free to run it first. It releases *that*
  slider by name now. This is the §14.1 hazard reappearing one layer up, in the
  documentation of the fix for it.
- **`AmbiguousAxisError` said "Which 2 should be the axis?"** and offered a
  one-range example for a two-axis problem. Visible only because the tour
  prints its refusals.

`tests/test_tour_notebooks.py` executes the Jupyter notebook end to end and
fails on any cell that raises, asserts the marimo graph has no duplicate
definitions or cycles, and checks the generated files are current against
`build_tour.py`. The tour cannot rot silently, and it cannot drift from the
library — but the reason it exists is still the reading, not the assertions.

### 15.1 The tour found the defect it was built to find

Reported by the project owner on the first run, in marimo:

```python
plot(tan(x))
```

printed its inference report and drew **no graph at all**. Every cell of the
tour "ran"; the suite was green; nothing was on the page.

`PlotResult` delegated only `_repr_mimebundle_` to its figure. Plotly returns
`{}` from that method unless a Jupyter kernel has activated the mimetype
renderer — so in Jupyter the plot appeared, and in marimo, which is not a
kernel, nothing did. marimo prefers `_repr_html_`, which Plotly implements and
which works everywhere; `PlotResult` simply did not offer it.

The rule the fix encodes: **a result displays wherever the figure it wraps
would, so it offers every hook the figure offers** rather than picking one.
`Dataset` had no display hook at all and rendered as `<Dataset 6 rows: x, y>`;
it now draws its summary as a table.

This is §14.7's shape once more, and worth naming precisely because the tour
was supposed to be the answer to it:

- **The suite only ever asked one host.** `tests/test_codegen_fidelity.py`
  compares figures object-to-object, never through a host's formatter. The
  Jupyter parity check (§11.2 criterion 5) compares *numbers*, not renderings.
  Nothing in 1286 tests asked "does anything appear on the screen".
- **The tour itself was verified with the same blind spot.** It was checked by
  executing the Jupyter twin and asserting no cell raised — and separately by
  running `marimo export html` and confirming it exited zero. Exit zero is not
  a picture. The marimo export at that point was 218 KB and contained **no
  plots**; it is 2.2 MB and contains 28 now.

Two checks close it. `tests/test_notebook_display.py` puts every displayable
type — plots of all nine kinds, `Analysis`, `Table`, `Dataset`, `Worksheet` —
through marimo's own formatter and fails on a rendering too small to be a
picture. `TestItRunsInMarimoToo` renders the whole tour as marimo does and
counts the plots, tables, analyses and animation frames on the page.

Writing the second one produced its own small lesson: the first version
searched the export for `<table` and found none, which looks exactly like
"nothing rendered". marimo escapes cell output into an iframe `srcdoc` and then
into JSON, so the tag arrives as `\u003Ctable`. A test that cannot tell a
rendering from an encoding is not a check on rendering, so the helper now
unescapes before it looks.

---

## 16. Second review round — sixteen risks assessed

A second review, submitted as three general risk statements and thirteen
numbered code observations. The value of this round was not the fix count. It
was that **five of the sixteen turned out not to be defects**, and finding that
out took measurement rather than reading — two of them contradicted claims that
looked obviously true.

The rule applied throughout: a claim about behaviour gets checked against
running code before anything is changed. A fix applied to a non-defect is not
free; it is new code, new surface, and in one case below it would have
introduced a bug that does not currently exist.

### 16.1 The verdicts

| # | Observation | Verdict |
|---|---|---|
| G1 | `exec` in `suggest.py`, and AI-suggested code being run | **By design** — `Suggestion.run()` is explicit-only and documented as such (§13.3) |
| G2 | Symbolic work blocks with no timeout | **Known limitation** — already recorded in §11.7; a real fix needs threads or signals |
| G3 | Global slider state can collide across symbols | **Fixed upstream** — `release()` / `release_all()` and manual §11.5 |
| 1 | `suggest.py` `run()` executes model output | Same as G1 |
| 2 | An API key could route to the wrong provider | **Fixed upstream** — `resolve_provider(name, *, api_key)` refuses to guess |
| 3 | `examples/marimo_notebook.py` calls `f.show()` | **Real** — and worse than reported; see §16.2 |
| 4 | `analysis.py` imports `sampling._detect_jumps` | **Real** — see §16.3 |
| 5 | The conftest `RuntimeWarning` filter is too broad | **Real** — see §16.4 |
| 6 | `_isolate_sliders` only cleans up on teardown | **Real** — the first test in a session was unprotected |
| 7 | Surfaces get no domain or singularity detection | **Partly real** — the gap is documented (§6.3), but the clip reached only the colours; see §16.5 |
| 8 | `bound_parameters()` is recomputed per plot | **Not a defect — measured**; see §16.6 |
| 9 | `except Exception` should be an explicit list | **Real, and the proposed list was wrong**; see §16.7 |
| 10 | `_starting_points` explodes with parameter count | **Claim incorrect — measured**; see §16.6 |
| 11 | `str()` around an already-`str` return | **Real** — trivially removed |
| 12 | `read_csv` cannot read cp949 | **Real** — see §16.8 |
| 13 | A marimo slider binds for the whole session | Same as G3 |

Test count over the round: 1332 → 1387, all passing.

### 16.2 The example notebook was dead code

Item 3 reported `f.show()`, which was correct: in a notebook that opens a
separate browser window and leaves the cell blank. But the file had a second
defect nobody had reported, because nothing had ever run it.

One cell used the name `dataset`, and **no cell in the file imported it**.
marimo resolves its dataflow graph statically from what each cell defines, so an
unresolvable name does not raise — the cell is simply never scheduled. Half the
notebook did nothing, silently, and had presumably done nothing since it was
written.

The tour notebooks (§15) had a graph-soundness check. This file had no test at
all, which is the entire explanation. It now has
`tests/test_example_notebook.py`: the graph resolves, no cell calls `.show()`,
and `marimo export html` produces a page with plots on it. Both original defects
were confirmed to fail those checks before the fixes were kept.

### 16.3 A private name with four callers is not private

`analyze()` and the sampler need the same bisection jump probe — `singularities()`
cannot see a discontinuity that is not a pole, so both have to run it.
`analysis.py` was reaching for `sampling._detect_jumps`.

Two core modules sharing one algorithm is fine. Spelling the shared thing as
private is not: the underscore stops being a statement about visibility and
becomes a comment that is no longer true, and the next person to rename it will
not know a second module depends on it. The name is now `detect_jumps`, in
`__all__`, with the reason for its publicness in its docstring, and a test
asserts both callers agree on what it returns.

### 16.4 A filter that hid the thing it was protecting

§5.3 step 6 requires the element-wise fallback to be **loud** — a plot that
silently became 2000× slower is a plot the reader blames on the library. The
mechanism is a `RuntimeWarning`.

The conftest fixture silenced `RuntimeWarning` wholesale, because NumPy emits
one for every `tan(x)` and `1/x` in the corpus. Same category, so no
category-level filter could separate them. It now matches on the *message*, and
the five notices that legitimately fire during a suite run are visible in the
report again. `tests/test_warning_policy.py` pins the distinction against the
fixture's own data, so the list can grow without quietly re-swallowing
MathSlate's own warnings.

### 16.5 Clipping the colour is not clipping the surface

Item 7 said surfaces get no domain or singularity detection. That is true, and
it is already written down as a reduction in guarantees (§6.3): there is no
two-variable `singularities`, so a surface is sampled on a grid and non-real
points become holes.

Checking how the existing mitigation behaved found a real defect underneath it.
The percentile clip computed a z range and then used it for `cmin`/`cmax` only.
The **geometry** was untouched, so `plot(1/(x*y))` still contained a spike 3481
units tall on a surface whose features live within 14. Plotly zooms out to
contain the spike, and everything the plot was made to show flattens into a
sheet at the bottom — the colour scale is now beautifully graded across a
picture of nothing.

The range now also sets `scene.zaxis_range`, in the figure and in the emitted
code, and a surface that needs no clipping keeps an automatic axis.

### 16.6 Two claims that measurement refuted

Both of these looked correct. Neither was.

**Item 8 — `bound_parameters()` per plot.** Measured at **0.09 µs**, against
an 88 ms `plot()`: 0.0001% of the call. Caching it would also introduce a
staleness bug that does not exist today, because the value changes when a slider
moves. The right change is no change.

**Item 10 — `_starting_points` explodes.** It does not. The count is bounded at
nine regardless of parameter count: 1 → 3, 2 → 9, 3 → 9, 5 → 9, 8 → 9. The
function does not have the shape the review attributed to it.

Recording these is the point of the section. A review's job is to raise
candidates; deciding one needs code changed is a separate step, and skipping it
would have cost a cache invalidation bug and a rewrite of a function that was
already correct.

### 16.7 The exception list, measured instead of guessed

Item 9 asked for the ten `except Exception` sites in `analysis.py` to name their
types, suggesting `NotImplementedError`, `ValueError`, `TypeError` and
`SympifyError`. The style preference is right. The list is not.

Running every operation the module performs over three dozen pathological
inputs produces nine types, and **five of them are outside that list**:

| Raised in practice | In the proposed list? | Example |
|---|---|---|
| `NotImplementedError` | yes | `continuous_domain(floor(x))` |
| `TypeError`, `ValueError` (incl. `SympifyError`) | yes | `solveset` of a `Matrix` |
| `AttributeError` | no | `limit(Eq(x, 2), x, 0)` |
| `NameError` | no | `lambdify(DiracDelta(x))` |
| `KeyError` | no | `lambdify(zoo*x)` |
| `SyntaxError` | no | `lambdify` printing unparseable source |
| `RecursionError` | no | `periodicity(Sum(...))` |

Narrowing to the four would have converted today's graceful fallback into a
crash on each of the other five. So the set is named rather than narrowed, in
`core/_failure.py`, with the evidence in the docstring and
`tests/test_failure_taxonomy.py` re-running a slice of the survey so a SymPy
upgrade that raises something new fails in one place instead of in a learner's
notebook.

Naming it still buys the thing the item was after, in the other direction:
`MemoryError` and `OSError` are no longer caught. Running out of memory inside
`solveset` was being reported as **"this function has no roots"** — a lie about
mathematics caused by a fact about the computer. A second tuple,
`EVALUATION_FAILURE`, covers the numeric sites; measurement showed it to be a
subset of the symbolic one, which is unsurprising once you notice the last
evaluation tier is SymPy's own `evalf`.

### 16.8 A CSV from a Korean Excel

`read_csv` did `read_bytes().decode("utf-8")`. A sheet exported from Excel on a
Korean Windows is cp949, so this met the audience of §2.1 with a raw
`UnicodeDecodeError` from inside the standard library: no file name, no
suggestion, and no hint that an encoding argument might exist. cp1252 from a
European export and a UTF-8 BOM from Excel failed the same way.

There is now a fallback chain (`utf-8-sig`, `cp949`, `cp1252`, `latin-1`), an
`encoding=` argument on `dataset()` and `read_csv()`, and a message that names
both the file and the argument when an explicitly named encoding does not fit.

Writing the test for the failure path found one more thing. `latin-1` decodes
*any* byte sequence, so the chain could never conclude "this is not a text
file" — handed a PNG, the reader produced a one-column dataset named `\x89PNG`
and raised nothing at all. A NUL byte anywhere in the file is now taken as the
evidence that it is binary, which is why UTF-16 has to be named explicitly
rather than guessed.

### 16.9 What this round says about the suite

Every defect found here was invisible to 1332 passing tests, and they fall into
two groups.

**Things no test executed.** The example notebook is the clearest case: it was
broken from the day it was written, in a way that produces no error, and nothing
ran it. §15 drew the same conclusion about the tour and fixed it for the tour
only — the smaller example was left behind.

**Things a test could see but was told to ignore.** The warning filter is the
sharp one. The suite had tests asserting the fallback notice is audible, and
they passed, because `pytest.warns` overrides filters. The notice was inaudible
everywhere else in the suite and nobody could tell. A fixture that suppresses a
category the project itself uses is not test hygiene; it is a hole with a
docstring.

---

## 17. v1.5 — benchmarked against Mathics3

A review of [Mathics3](https://github.com/Mathics3) as a source of features
worth copying. Four candidates were proposed; the most valuable finding was none
of them, and two of the four premises did not survive checking.

### 17.1 What Mathics3 actually has

The proposal credited Mathics3 with `RegionPlot`, `VectorPlot` and `StreamPlot`.
It has none of them. Its drawing builtins are:

| File | Plotting classes |
|---|---|
| `builtin/drawing/plot_plot.py` | `_Plot`, `LogPlot`, `Plot`, `ParametricPlot`, `PolarPlot` |
| `builtin/drawing/plot_plot3d.py` | `_Plot3D`, `DensityPlot`, `Plot3D` |
| `builtin/drawing/plot.py` | `ColorData`, `ColorDataFunction`, `Histogram` |

There is not even a `ContourPlot`. Those three names are **Wolfram
documentation**, not Mathics3 code, so there was no implementation to benchmark —
only a specification. The feature ideas are still good; the evidence for them
was not what it appeared to be.

`Mathics3-Module-networkx` is likewise not a visualisation module. It wraps
nineteen Wolfram graph *functions* (`FindShortestPath`, `VertexConnectivity`,
`EdgeDelete`, …) and renders through its own `GraphBox`. "One-line `plot(G)`" is
a thin slice of it.

### 17.2 The finding that was not on the list

Mathics3 maintains **forks of `stopit` and `python-timed-threads`**, for one
stated purpose: implementing `TimeConstrained`. That is the same problem §16
recorded as a known limitation and deferred. Mathics3 also documents its own
partial solution honestly — the abandoned evaluation "continues consuming
resources" — which turned out to matter, see §17.4.

This ranked above all four proposals, and implementing it produced the round's
main correction (§17.3).

### 17.3 The premise was right, the diagnosis was wrong

The case for a timeout was `analyze(besselj(2, x) - 1/x)` taking 27 seconds. The
obvious reading — "a symbolic step is slow" — was wrong, and a profile said so:

```
68.0s  analyze_expression
66.4s    NumericFunction._elementwise
64.9s      NumericFunction._exact          × 12942 calls
```

Sixty-six of sixty-eight seconds were **numeric**. NumPy has no `besselj`, so
evaluation fell to the last-resort tier, which rebuilt and re-evaluated the whole
expression tree per point via `subs().evalf()`. No deadline would have caught it;
a deadline would have made the plot approximate and still slow.

The fix was an **mpmath tier** ahead of that one. mpmath is what `evalf` calls
underneath, so going to it directly costs nothing in accuracy and skips the tree
walk. Measured per 400 points:

| Expression | `subs().evalf()` | mpmath | Speed-up |
|---|---|---|---|
| `besselj(2, x)` | 0.270 s | 0.006 s | 45× |
| `Si(x)` | 0.350 s | 0.004 s | 87× |
| `gamma(x)` | 0.108 s | 0.004 s | 27× |
| `zeta(x, 3)` | 0.310 s | 0.159 s | 2× |

End to end, with no new dependency (mpmath is already SymPy's):

| `analyze(...)` | Before | After |
|---|---|---|
| `besselj(2, x) - 1/x` | 29.5 s | 1.7 s |
| `LambertW(x) - cos(x)` | 18.0 s | 4.9 s |
| `sin(x)·zeta(x, 2)` | 16.1 s | 3.9 s |
| `x - Si(x)` | 10.0 s | 0.6 s |
| `gamma(x) - x³` | 7.5 s | 1.3 s |

The lesson is the same one §16.6 recorded: a measured symptom does not identify
its cause. The 27 seconds were real; the story about them was not.

### 17.4 The budget, and what stopping a thread costs

A deadline is still worth having — `solveset` on a degree-40 polynomial takes
eighty seconds, which no numeric tier addresses. `core/_budget.py` gives each
symbolic attempt five seconds and falls back to the numeric path already used
when SymPy declines, with a note distinguishing the two: SymPy declining is a
fact about mathematics, running out of five seconds is a fact about the computer.

Five is measured, not chosen: nothing in the corpus reaches two seconds, so on
ordinary work it never fires.

Building it produced three problems in sequence, each found by measurement.

**A thread cannot be killed, so the first version abandoned expired ones.** Since
SymPy cannot safely run from two threads at once, later attempts then had to be
refused — and one bad expression made `analyze(sin(x))` approximate for eighty
seconds afterwards. A guard that degrades everything after it is not a guard.

**So the worker is asked to unwind**, via `PyThreadState_SetAsyncExc` — the
mechanism `stopit` is built on. SymPy's `solveset` unwinds from it in **28 ms**,
measured, so Mathics3's documented "keeps consuming resources" limitation does
not apply here. Total time for the degree-40 case: 74 s → 5.7 s.

**Injection lands at an arbitrary bytecode boundary, including mid-cache-write.**
Observed once it was added: a `solveset` stopped inside `rootoftools` left
`_complexes_cache` holding a key whose value never arrived, and the next
*unrelated* lookup raised `KeyError` from inside SymPy — a failure with no
visible connection to the timeout that caused it. So SymPy's caches are cleared
after any interruption that landed, and the repair is deferred rather than
skipped when the worker has not finished unwinding yet, so correctness does not
depend on machine load.

One residue is documented rather than fixed: the injected exception can be
delivered inside a weakref-clearing callback, where no frame of ours exists to
catch it. CPython prints "Exception ignored in …" and discards it. Nothing is
corrupted; it cannot be suppressed from inside the library.

One expired step disables symbolic attempts for the rest of that `analyze()`
call. Twelve steps each paying five seconds would turn a five-second limit into a
minute-long wait for the same answer.

### 17.5 Inequalities — the one proposal worth taking

`plot(x**2 + y**2 < 1)` was a refusal whose error message already described the
feature: *"plot() draws Eq(lhs, rhs) as an implicit curve, but not an
inequality."* Two rows now:

| Input | Free symbols | Result |
|---|---|---|
| inequality | 2 | filled region (`kind="region"`) |
| inequality | 1 | shaded bands on the axis (`kind="band"`) |

`Eq` asks where two expressions are *equal*, which is a curve; `<` asks where one
*exceeds* the other, which is an area. Same shape of input, different question,
so a separate planner.

Implementation was cheap because `lambdify` vectorises relations already —
`And` prints as `logical_and.reduce`, a comparison as `less` — so a compound
region evaluates in one pass over the existing grid.

Two things were not cheap.

**"Not real" is not "outside".** A comparison against `NaN` is *false*, in NumPy
and in Python alike, so `sqrt(x*y) > 1` would report the quadrants where
`x*y < 0` as outside the region — a definite claim about a place where the
inequality has no truth value. The operands are therefore evaluated separately
and those points left blank, and counted in the notes.

**`solveset` returns the principal period of a periodic inequality and does not
say so.** `solveset(sin(x) > 0, x, Interval(-6, 6))` is `Interval.open(0, pi)`,
silently missing `(-6, -pi)`; the range argument does not prevent it. Reporting
that as exact would put a confident wrong answer on the screen, which is the one
failure mode this project cannot have. So the closed form is **checked**: it is
used only when sampling cannot find a solution outside it, and otherwise the
sampled intervals are used with their edges refined by bisection and
`bands_exact=False`. This is the same trust-but-verify shape as the
`ConditionSet` rule in §11.7 — and it holds regardless of why solveset came back
short.

Curriculum fit was the deciding argument. 부등식의 영역 and 부등식의 해 are both
standard Korean high-school topics, which no other proposal on the list was.

### 17.6 `exclusions=` — closing a gap in §4

Automatic discontinuity detection is the project's differentiator (§5.3). It had
no override. `plot()` accepted `points=` and nothing else, so a reader who found
detection wrong had no recourse at all — which contradicts §4's escape-hatch
completeness metric. Mathematica's `Exclusions` is the same idea.

* `exclusions=[a, b]` — cut here as well.
* `exclusions=False` — stop the numeric probe. Symbolic singularities and the
  real domain still apply, because those are SymPy's answer rather than a guess.

Writing it found a fidelity bug in the first attempt: named places were recorded
in `breakpoints` but never inserted into the sampled arrays, so `show_python()`
cut the line where the figure did not. `_insert_breaks` was receiving only the
newly *detected* jumps.

### 17.7 Not taken, and why

| Proposal | Verdict |
|---|---|
| Vector / stream fields | **Deferred.** Slots are free and Plotly has the traces (`create_quiver`, `create_streamline`, `Streamtube`, `Cone`). But the proposed spelling `plot([f, g], kind="vector")` breaks §5.1's one rule — a vector field is *one* vector-valued object, so it is a tuple. The consistent form is a 2-tuple with **two ranges**, using the precedent §16 established for surfaces. Audience is P2, not P1, so it contributes nothing to §4's high-school corpus metric. |
| NetworkX graphs | **Declined.** Three reasons, in order of weight: `spring_layout` is randomly initialised, so `show_python()` could not reproduce the same figure without emitting a seed — a direct conflict with §4's 100% fidelity metric; it needs a new optional dependency, making `plot(G)` a function that works or not depending on the environment; and discrete graph theory is outside P1's curriculum. Mathics3 shipping it as a separate module is the same conclusion. |
| Third-party plugin system | **Premature.** `mathslate.ai` and `mathslate.classroom` are in-tree extras; Mathics3's `pymathics.*` is a runtime-discovered plugin namespace. That is the right design once third-party authors exist. There are none. `sympy.stats` can be an in-tree extra when it is wanted. |
| JupyterLite / Observable frontends | **Noted, not scheduled.** Zero-install browser execution serves P1 and P3 directly, and single-file HTML export covers only the distribution half. SymPy under Pyodide is heavy and it touches §6.1; investigation, not a milestone. |
| Doctests in docstrings | **Already stronger here.** `tests/test_docs.py` executes every `python` block in `docs/`, in document order, in one namespace — the reader's own experience. Nothing to copy. |

### 17.8 §4's API-surface metric was mismeasuring

Raised by the project owner: is the 15-symbol cap a real constraint or an
arbitrary one?

The number is arbitrary. The forcing function is not — every v1.0 feature was
pushed through a dispatch row or an optional module *because* of it, and without
it the path of least resistance is `region_plot()`, `vector_plot()`,
`graph_plot()`: Mathematica's design, with roughly six thousand symbols, which is
what this project differentiates against.

But the metric only counts top-level names, and `plot()` already had eleven
keyword arguments before this milestone added a twelfth. Keyword growth is
surface growth and the metric could not see it. Raising the cap while leaving
that unmeasured would be the worst of both.

So the cap stays at 15 and the metric is restated to count both. All three items
in this milestone cost **zero** new public symbols: the budget is reached through
`mathslate.core`, the region is a dispatch row, and `exclusions` is a keyword.

| Metric | Target |
|---|---|
| New API surface | ≤15 public symbols **and** ≤15 `plot()` keyword arguments, excluding SymPy re-exports |

### 17.9 Score

Tests: 1387 → 1602. Suite green.

Of six proposals reviewed, one was implemented as proposed (inequalities), one
was implemented in a different form than proposed (the timeout — where the
measurement contradicted the reasoning behind it), two were declined on
measurable grounds, and two were deferred. The item that mattered most was not
on the list at all.

---

## 18. Polish round — 3D look, view control, and the SymPy engine surfaced

Four requests, all in the same spirit: the mathematics was right, but the tool
did not always *look* or *read* like the commercial programs it is measured
against. None of these adds a top-level symbol.

### 18.1 3D surfaces were too plain

Mathematica's `Plot3D` rules its surfaces with grid lines, and that is not
decoration — a bare colour gradient reads as flatter than the surface is, and
the eye recovers curvature from the way the grid bends. Ours were smooth.

Plotly's `go.Surface` already carries the mechanism: `contours.x.show` and
`contours.y.show` draw lines at constant x and constant y, and both together are
a grid rather than parallel ribbons. `mesh=True` (the default now) turns them on
in a subtle translucent grey with the hover-highlight suppressed; `mesh=False`
returns the smooth look. It applies to both `surface` and `psurface`, and is
reproduced by `show_python()`.

Defaulting it *on* changes every existing 3D figure, which is the point — the
complaint was precisely that the default looked unfinished.

### 18.2 The view window was not the domain, but you could only set the domain

A range argument `(x, -10, 10)` sets where a function is **sampled**. What the
axis **shows** is a separate thing, and until now the only way to change it was
to reach into the returned Plotly object. Two are not the same: the automatic
y-clip (§16.5) deliberately hides a pole, sampling runs wider than the eye wants
to see, and two figures meant for comparison want a shared scale.

`xlim`, `ylim`, `zlim` set the window directly, matplotlib-style. Each is a
`(low, high)` pair, validated and refused if malformed; each overrides the
automatic choice for that axis — `ylim` is how you ask to see the pole the clip
hides — and each is carried into `show_python()`. `zlim` overrides the 3D
z-clip, and `xlim`/`ylim` also zoom a flat region or contour.

### 18.3 The tour was all plotting

`examples/mathslate_tour.py` showed the drawing and almost nothing of the engine
underneath it. A new §7, *Solving and algebra*, covers `solve` (expressions,
equations, systems), `solveset` and `nsolve`, and the algebra verbs —
`factor`, `expand`, `gcd`, `apart`, `cancel`, `together`, `trigsimp` — and ends
by feeding a factored polynomial straight into a plot, because the whole reason
to re-export rather than wrap is that no conversion step exists. It closes by
reaching a non-re-exported function (`linsolve`) through `mathslate.sympy`, so a
reader learns nothing is out of reach.

### 18.4 Algebra support was undocumented

The manual listed the re-exported names and said no more. A reader could not
tell, without trying, whether `factor` was MathSlate's or SymPy's or supported
at all. The "Re-exported from SymPy" section now has a *Solving and algebra*
subsection: a support table for the algebra verbs, runnable examples for each,
and the one honest boundary stated plainly — `solve()` returns the answer, not
the worked steps, because SymPy has no general step engine and a partial one
would mislead (§2.2). Both the English and Korean manuals carry it, and every
example is executed by `tests/test_docs.py`.

The general principle behind requests 3 and 4: **information about SymPy is
information the user needs, even when it is not MathSlate's own.** A tool that
delegates its whole computational core should say so and show it, not leave the
seam invisible.

### 18.5 The metric caught its own blind spot, again

Adding `mesh`, `xlim`, `ylim`, `zlim` brought `plot()` to fourteen keyword
arguments — against the ≤15 restated in §17.8. That restatement, made one
milestone ago to count keywords and not only top-level symbols, is what made the
cost of this round visible in advance rather than after the fact. Fourteen of
fifteen; the budget now has one slot left, which is information worth having.
`tests/test_api_surface.py` enforces both halves.

### 18.6 Score

Tests: 1602 → 1643. Suite green. Zero new top-level symbols; four new `plot()`
keywords; two new documentation subsections, both executed by the suite.

## 19. Live range controls — the gap `xlim` still left open

§18.2 gave `plot()` a window separate from the domain. It did not give the
*reader* a way to move that window without editing code, and it was pointed
out that Plotly's own drag-to-zoom does not fill the gap either: dragging a
box only crops the points already computed. Zoom into a tenth of a wide
domain and you see a tenth of the original point density — jagged, not
closer. Excel's own axis dialog was the reference point: change the range,
get a chart drawn for that range, not a screenshot crop of the old one.

### 19.1 Why this is not a `plot()` keyword

`xlim`/`ylim`/`zlim` choose a window once, at call time. What was asked for
here is a window the reader keeps moving after the figure exists — which
needs a live host to push a new sample into an already-displayed figure, not
another argument to a function that already returned. It belongs on the
result, not on the call.

### 19.2 `PlotResult.range_controls()`

Four number boxes (`x min`, `x max`, `y min`, `y max`) next to the figure;
changing one re-samples through `PlotPlan`'s own domain and redraws in place.
The rebuild (`PlotResult._replotted`) is plan-shape aware rather than
uniform: a surface, contour or region has two real domains (`plan.axes`), so
both X and Y resample; an ordinary curve derives Y from X, so only X
resamples and Y becomes a `ylim` view update — cheap, and correct, since
there is no per-point Y resolution to recover on a curve where nothing was
sampled along Y in the first place. This is exactly the asymmetry that was
asked for: X finer without touching Y, or the reverse.

Needs a running Jupyter or Colab kernel with `ipywidgets` (and, on Plotly
≥ 6, `anywidget` — `FigureWidget` moved onto it and the failure otherwise
surfaces only at first use, so it is caught and named explicitly). marimo was
deliberately not given a wrapper: `mo.ui.number()` composed directly with
`plot()` already gets the same live resampling for free, because marimo
re-runs the cell reactively on every change — which is precisely what this
method does by hand for ipywidgets. Building a second abstraction over that
would fight marimo's model rather than use it. A plain script or an exported
HTML file has no kernel to push an update through, so `xlim`/`ylim` remain
their way to choose a window there.

### 19.4 Made the default, not an opt-in nobody finds

An opt-in method has an adoption problem baked in: a reader has to already
know `range_controls()` exists to type it. The request that followed §19.2
landed within the same session — make it the default — and the honest reason
to do that is the same one that made `mesh=True` the default in §18.1: the
feature is the point, and a feature nobody discovers does not deliver it.

The hook is `PlotResult._ipython_display_`. IPython calls this instead of
`_repr_html_`/`_repr_mimebundle_` when it is defined — not merged with them,
instead of — so it can choose exclusively rather than adding a second
representation next to the old one. Bare `plot(...)` at a cell's end now
shows `range_controls()` whenever `_wants_live_range_controls()` says yes
(Jupyter or Colab, `ipywidgets`/`anywidget` importable, a plan with a domain
to resample) and the plain figure — pixel-identical to before this
round — everywhere that is false: marimo, a plain script, missing deps, or
raw data. `_live_range_deps_available()` checks the capability the same way
`range_controls()` itself would (an empty `FigureWidget`, cheap, no clone of
the real figure) so the two never disagree about when it is safe.

marimo was checked, not assumed, before relying on this: its own display
resolution (`repr_formatters.py`) looks for `_repr_html_` then
`_repr_mimebundle_` and never for `_ipython_display_`, so defining the hook
does not touch marimo's rendering path at all — it keeps using `_repr_html_`
exactly as §11's fix left it. `_ipython_display_` also does not exist as a
concept outside an initialized `InteractiveShell`: `IPython.display.display()`
just prints `repr()` until one is running, which is what `examples/quickstart.py`
gets under `python -m pytest` and a plain script — so the hook is provably
inert everywhere it should be, not merely untriggered in the cases tested.

This did cost a design constraint worth stating: the API-surface budget is
full — `NEW_API` is at 15 of 15 (§18.5) — so a global on/off switch in the
style of `set_verbose()` was not available without retiring something else to
make room, which was not this round's call to make. The default is therefore
automatic and capability-driven rather than user-toggleable; the existing
escape hatches (`.plotly`, `.figure`, explicit `range_controls()` from a
result printed with `verbose=False`) are what is available until a future
round decides the budget is worth spending on a toggle.

### 19.5 Score

Zero new top-level symbols; zero new `plot()` keywords — this is a method
(and now a display hook) on `PlotResult`, so the §17.8 budget is untouched.
`tests/test_view_and_mesh.py` covers the resample/view-clip split on a curve
and on a surface, the refusal for raw data and for a non-notebook host, and
(best-effort, like `Slider.widget()`) the ipywidgets path itself, including
the default-display wiring verified against a real, initialized
`InteractiveShell` rather than the plain-script no-op path alone.

---

## 20. The API-surface cap raised to 20

§17.8 asked, at v1.5, whether the 15-symbol cap was a real constraint or an
arbitrary one, and concluded the number was arbitrary but the forcing
function was not — so it kept the number and restated the metric to also
count `plot()`'s keywords. Two rounds later that restatement was doing exactly
what it was for: §18.5 landed at fourteen of fifteen keywords, and §19.4 hit
the symbol side of the same wall directly — `NEW_API` sat at 15 of 15, so a
`set_range_controls()`-style toggle had nowhere to go without retiring
something else first, and the round shipped without one for exactly that
reason.

Raised by the project owner directly: repeatedly letting a fixed number decide
what ships is not the same discipline §17.8 argued for. §17.8's own case for
the cap was that it forces a feature through a dispatch row or an optional
module *instead of* a new top-level name — a design pressure, not a
population limit — and a project that has now hit the ceiling twice in three
rounds is measuring the number, not the pressure. The fix is not to remove the
gate; it is to stop it from being the thing that decides scope by accident.

The cap is **20 public symbols and 20 `plot()` keyword arguments**, both
counted the way §17.8 defined them (`mathslate.NEW_API`; `plot()`'s
keyword-or-positional parameters excluding `obj`). Nothing was added to
either list to reach this milestone — the change is the ceiling, not the
count, which stays at 15 symbols and 14 `plot()` keywords exactly where §19.5
left it. `tests/test_api_surface.py` asserts the new numbers.

| Metric | Target |
|---|---|
| New API surface | ≤20 public symbols **and** ≤20 `plot()` keyword arguments, excluding SymPy re-exports |

The five slots this opens are not spent here — a `set_range_controls()`
toggle (§19.4's deferred item) is the obvious first claim on one of them, but
that is a future round's decision, made against its own evidence, the same way
this round's was.

---

## 21. `set_range_controls()` — the deferred item, claimed

The next round in the same session, and the report was concrete: a reader
running `plot(...)` in their own Jupyter session was not seeing
`range_controls()` and could not tell why. Two separate needs came out of
that — a toggle, and a diagnosis — and both are cheaper solved together than
apart.

### 21.1 The toggle

`set_range_controls(bool)` / `get_range_controls() -> bool`, the same shape as
`set_verbose()`/`get_verbose()` (§4.6) and living next to
`_wants_live_range_controls()` in `mathslate/result.py` so the one function
that decides is the one function that reads the flag. It is a *global*
switch, on the same reasoning `set_verbose()` already established: a standing
preference ("not today") rather than a per-figure choice, which is what
`.range_controls()` called by name on one result is for. Capability still
gates it either way — turning the flag on does not draw a widget where
`ipywidgets`/`anywidget` are missing or the host is not Jupyter/Colab.

Two of the five slots §20 opened: `NEW_API` goes from 15 to 17.

### 21.2 The diagnosis — "nothing showed" needs a reason, not a bool

`_wants_live_range_controls()` is a single `bool`; asking it directly answers
"is it on", not "why is it off", which was the actual question. `frontend_report()`
already existed for exactly this class of question (PRD §15) — it names the
detected host and whether `ipywidgets` is there — so it gained a third field
rather than a new function costing a symbol:

```
frontend: colab | interactive widgets: ipywidgets | range_controls(): not shown — anywidget is not installed (pip install mathslate[jupyter])
```

The four states it distinguishes are the four ways §19.4's condition can be
false: the toggle is off, the host is not Jupyter/Colab, `ipywidgets` is
missing, or (the one the "interactive widgets" field cannot see, and the most
likely actual cause of the report that opened this round) Plotly ≥ 6's
`FigureWidget` needs `anywidget` and it is not installed alongside
`ipywidgets`. `_range_controls_status()` lives in `mathslate.ui.adapters`
next to `frontend_report()`, importing `mathslate.result` lazily — the two
modules already import each other's names at call time elsewhere in this
pairing (§19), and a module-level import here would be the cycle that pattern
exists to avoid.

### 21.3 Score

`NEW_API`: 15 → 17, still 3 under the 20 §20 opened. Zero new `plot()`
keywords. `tests/test_api_surface.py` covers the toggle and that the report
names why; `tests/test_view_and_mesh.py` covers the report's four states
directly.

### 21.4 The diagnosis found a real gap, not a misunderstanding

The reader's own `frontend_report()` output confirmed it: `anywidget` was the
missing piece, and `pip install mathslate[jupyter]` — the fix every message
in §19 and §21.2 pointed to — did not actually install it. `jupyter` in
`pyproject.toml` listed only `ipywidgets>=8.0`; `anywidget` had been a second,
undocumented-as-required step ever since §19.2 introduced `FigureWidget`, and
every "just `pip install mathslate[jupyter]`" message was quietly wrong for
any Plotly >= 6 install.

Fixed at the source rather than in the message: `jupyter = ["ipywidgets>=8.0",
"anywidget>=0.9"]`. One extra now installs everything `range_controls()`
needs, and the messages in `mathslate/result.py` and this document were
reworded to say so rather than to name `anywidget` as a separate install.

---

## 22. `range_controls()` — layout and size

Two more requests on the same widget, both about how it sits on screen rather
than what it computes.

### 22.1 Beside the figure, not above it

The controls were an `HBox` of four boxes stacked in a `VBox` on top of the
`FigureWidget` — reasonable as the first working version, wrong as the
placement to keep: it makes the figure shorter every time the reader looks at
the numbers driving it, which is the opposite of what a "look closely at
this" control should do. They are now a `VBox` beside the figure in an outer
`HBox` — `[figure_widget, controls]` — so widening the window is not also
narrowing the plot. `bundle.children`'s order flipped accordingly
(`figure_widget, controls = bundle.children`), which is why every existing
test that unpacked it needed the same one-line fix.

### 22.2 `width=` / `height=`

`.plotly.update_layout(width=..., height=...)` already sized any figure — an
existing escape hatch, not a gap — but a reader sizing the *live* widget had
to know to reach for it before calling `range_controls()`, in a different
place from the call that builds the thing they were looking at. `range_controls(width=, height=)`
sets `figure_widget.layout.width`/`height` directly in that call, refusing a
non-positive value the same way `points=` refuses one (§4).

Whether it survives a resample was a real question, not an assumption: each
`_redraw()` rebuilds the *plan* from scratch via `_replotted()`, and if
rebuilding the *figure* the same way had lost the size, this would have been
a worse regression than the one it fixed. It does not, and measurably so —
`FigureWidget.layout.update()` merges into the existing layout rather than
replacing it, so a property the new layout dict does not mention keeps its
prior value. Verified directly (`tests/test_view_and_mesh.py`
`test_the_size_survives_a_resample`) rather than assumed from reading Plotly's
source, because a merge-vs-replace question is exactly the kind avoidably
wrong on paper and cheap to just run.

Left unset, the widget keeps whatever size the figure already had — including
one set with `.plotly.update_layout()` before the call, which this is a
shortcut for, not a replacement of.

### 22.3 Score

Zero new top-level symbols, zero new `plot()` keywords, zero `NEW_API`
change — both are `range_controls()`'s own keyword-only parameters, which the
§17.8/§20 budget was never defined to count. `tests/test_view_and_mesh.py`
gained a class covering the swapped layout, both dimensions, survival across
a resample, the unset-stays-as-is case, and the refusal.

### 22.4 §22.1 shipped a real regression, corrected the same day

The very split §22.1 introduced was reported wrong before the round closed:
the sidebar had grown to roughly 40% of the widget's width, not the small
strip the layout change was supposed to add, and the four boxes inside it
were narrow enough that the editable number was not visible at all — only
the label showed.

Both traced to one wrong assumption. §22.1's first attempt set
`figure_widget.layout.flex = "4 1 auto"`, intending a CSS flex-grow split;
`figure_widget.layout` is not a CSS box, it is Plotly's own chart-layout
object — the same one `width=`/`height=` above already write to — and
`FigureWidget` shadows the ordinary ipywidgets `layout` trait a plain widget
would expose for that, so there was no flex-grow to set on it in the first
place. The immediate symptom was a `ValueError` from Plotly's own property
validation on `flex`, not a wrong-looking split — the split itself would have
been left to the browser's default, unconstrained flexbox distribution
between a widget with a pixel size (the figure) and one without, which is
the ~40% actually observed. The box width regression was independent and
simpler: `_RANGE_CONTROLS_BOX_WIDTH` (then 120px) had to hold both the "x
min"-style label and an editable number, and a description label at
ipywidgets' default reserved width left the number no room at all.

Fixed by giving up on a flexible split. `_RANGE_CONTROLS_SIDEBAR_WIDTH` (a
fixed 190px) and `_RANGE_CONTROLS_FIGURE_WIDTH` (4x that, 760px, applied only
when the figure has no width of its own already — from this call or from an
earlier `.plotly.update_layout()`) are chosen together, so the ratio is
correct by construction rather than by trusting a browser's flexbox algorithm
to divide space between two widgets with different sizing capabilities. The
label's reserved width dropped to 45px, leaving comfortably over 80px of the
now-165px box for the number itself. Pinned by
`test_left_unset_the_default_split_is_close_to_80_20` and
`test_the_number_inside_each_box_has_room_to_be_read`, both written to fail
on the exact regression rather than only on the mechanism that caused it, so
a future change that reaches the same wrong ratio a different way still gets
caught.

**Drag-to-resize was asked about directly, and the honest answer is no.**
ipywidgets has no built-in split-pane widget, and building one is a real
piece of UI work — a custom `anywidget` component with its own JS, not a
`Layout` property — disproportionate to what this round asked for. The two
sides resize together by calling `range_controls(width=, height=)` again
with new numbers; there is no drag handle between them.

---

## 23. Z, mesh density, and a second layout bug in the same feature

Three more requests, landing before §22 had settled: Z should be
controllable on a 3D surface the same way X/Y are; the mesh (§18.1) reads too
sparse to adjust for; and — reported again — a fixed-width split still left
roughly 15% of the widget blank, because the figure and the sidebar's pixel
widths did not sum to whatever the actual notebook was rendering at.

### 23.1 The real bug behind §22.4's fix

§22.4 diagnosed the *ratio* correctly (sidebar too wide) and picked a
*mechanism* that could not deliver it in general: `_RANGE_CONTROLS_SIDEBAR_
WIDTH` and `_RANGE_CONTROLS_FIGURE_WIDTH` were both **pixel** counts, chosen
to sum to 950px. That is exactly 80/20 when the container happens to be
950px wide, and increasingly wrong — with the shortfall showing up as blank
space on the right, which is what was reported — as the actual notebook
width diverges from 950px. Two pixel numbers can be *tuned* to look right at
one width; they cannot *be* right at every width, because nothing forces
them to sum to the container's actual size.

The fix is a percentage on both sides instead, which does have that
property by construction: `_RANGE_CONTROLS_SIDEBAR_PERCENT = 20` on the
sidebar and `100 - 20` on a `Box` wrapped around the figure, so the two
always cover the full width regardless of what that width actually is.
`FigureWidget` still cannot take a CSS width directly (§22.4), which is why
the figure is wrapped in a plain `Box` rather than sized itself — the
wrapper carries the percentage, and Plotly's own `autosize` (active whenever
`width=`/`height=` were never given) fills whatever pixel width that wrapper
resolves to, the same way a lone figure filled its cell before any of this
existed. An explicit `width=`/`height=` still means exactly that many
pixels: the wrapper is left unconstrained in that case, so a deliberate
request is not stretched or squeezed to fit a percentage it never asked for.

The one thing this costs: `range_controls()`'s returned bundle nests one
level deeper than before (`figure_container, controls = bundle.children`,
where `figure_container.children[0]` is the actual `FigureWidget`) — every
test written against the old shape needed the same one-line unwrap, so
`tests/test_view_and_mesh.py` gained a `_unwrap()` helper rather than
repeating it.

### 23.2 Z controls on a real 3D surface

Two more boxes, `z min`/`z max`, appear only where there is a real Z to
window: `plan.kind in {"surface", "psurface"}`. `contour`, `implicit` and
`region` are two-variable too (`plan.axes` is set for all five §7 kinds) but
flat — `go.Heatmap`/`go.Contour`, no Z — so they stay at four boxes.

Z is the surface's *output*, computed from the X/Y grid, not sampled along
an axis of its own — the same relationship §19.2 already gave Y on an
ordinary curve. Moving Z therefore never resamples; it only ever replaces
`zlim` and redraws, the cheap path `_replotted()` already had.

Seeding the box's initial value found a second bug on the way: it first read
`sample.z_range`, which is the *active auto-clip* (PRD §16.5) and is `None`
on precisely the well-behaved surfaces that have nothing to clip — the
opposite of "no Z axis". A smooth surface like `x*y` would have opened with
`z min`/`z max` reading a placeholder unrelated to the surface on screen.
Fixed by falling back to the sample's own data extent
(`sample.z.min()`/`.max()`) when no clip is active, so the box always opens
on a number that means something.

### 23.3 Mesh density

`mesh=True` used to leave Plotly's `contours.x`/`.y` `size` unset, which
picks a "nice round number" spacing the way an axis chooses tick marks —
usually far fewer lines than the 60-point sample grid, and reported as
visibly sparse. `mesh` now also accepts a positive `int`, naming a line
count instead of just switching the grid on; `True` uses a new
`DEFAULT_MESH_LINES = 24`. The spacing is computed from the sample's own
X/Y extent (`span / count`), so the same `mesh=20` means the same twenty
lines whether the surface spans 2 units or 2000 — a fixed `size` would not
have that property. `bool` being a subclass of `int` in Python means the
bool case has to be checked first, or `True`/`False` would themselves be
read as a line count of 1 or 0; `_mesh_option()` in `mathslate/api.py` does,
and refuses anything under 2 with a message rather than drawing an
unreadable one-line "grid".

Writing it surfaced a real fidelity bug, not a new one: `codegen.py` had its
own hand-written `_mesh_kwarg()`, parallel to `RenderOptions.surface_
contours()` rather than calling it, and it never emitted `size` at all —
`show_python()` had been drawing Plotly's sparser automatic spacing
regardless of what the real figure showed, silently, for as long as `mesh`
has existed (§18.1). Fixed by having `_mesh_kwarg()` call
`options.surface_contours(plan)` — the same method the real figure is built
from — and format its result, rather than reconstructing the dict by hand.
This is exactly the failure mode `mathslate/render/options.py`'s own module
docstring names as the reason it exists ("every option only one of them
reads is a broken promise waiting to happen"); the module existing did not
prevent it here because `_mesh_kwarg()` did not call into it at all.

### 23.4 Score

Zero new top-level symbols. `mesh` widens from `bool` to `bool | int` on an
existing `plot()`/`animate()` keyword rather than adding a new one, so the
keyword count is also unchanged. `range_controls()` gained no new
parameters — Z detection and seeding are automatic. `tests/test_view_and_
mesh.py` gained `TestMeshDensity` (default spacing, an explicit count,
domain-scaling, refusals, `show_python()` fidelity on both `surface` and
`psurface`) and `TestRangeControlsOnA3DSurface` (box count by kind, both
Z-seeding paths, the no-resample property, Z surviving a later X change, the
mid-edit guard), and the §22.4 tests were rewritten against the percentage
split rather than deleted.

### 23.5 The percentage split alone did not make the figure grow

Reported immediately after §23.1 shipped: the sidebar now correctly held its
20%, but the figure did not expand into its 80% — it sat at whatever size it
already was, with the rest of the wrapper's width empty. §23.1's reasoning
had a gap: `figure_widget.layout.width = None` (Plotly's `autosize`) stops
the chart from *fighting* a container size, it does not make Plotly.js
*measure* one on its own. The lever that does that is `config.responsive`,
a Plotly.js setting that is not part of `layout` at all — it lives in the
separate config object `FigureWidget` exposes as `_config`, which is why it
was not sitting next to `width=`/`height=` in the first pass. Set with
`figure_widget._config = {**figure_widget._config, "responsive": True}`
(reassigning the dict, not mutating it in place, because that is what makes
a traitlets `Dict` trait notice a change and sync it to the frontend) —
only in the unset-size case, since `responsive` resizes the chart to its
container regardless of `layout.width`/`height`, which would silently
override an explicit `width=`/`height=` request otherwise.

### 23.6 `Auto Y` was reading the clip, not the curve

Reported as "the values it computes do not look right". They were not being
computed at all on most plots. `_autoscale_y` read `plan.y_range` and
returned when it was `None` — but `plan.y_range` is the *auto-clip*
(§16.5), and `_clip_range` deliberately returns `None` wherever a curve is
well behaved enough not to need clipping. So the button was dead on
`sin(x)`, `x**2` and most of the 200-function corpus, and alive only on the
poles — exactly backwards from where fitting Y is easy. `Auto Z` had had the
right shape since §23.2: ask `options` for the rendered window, then fall
back to measuring the data. `Auto Y` now makes the same two-step call
through a shared `_data_span`, and three consequences of the same root
confusion went with it:

- **The seed values.** The boxes started at `plan.y_range or (-1.0, 1.0)`,
  so `plot(x**2, (x, -10, 10))` drew 0..100 while its boxes read -1..1. They
  are not decorative: `_redraw` pushes them back as `ylim` on the next edit,
  so touching X cropped the plot to a window the reader never chose.
- **`ylim` could not be escaped.** `_replotted(y_range=None)` means "leave
  this axis alone", which is a different request from "clear it", and only
  the first was expressible. Auto Y on a plot built with `ylim=` fitted
  itself to the very window it was being asked to leave. `auto_y`/`auto_z`
  flags now say the second thing.
- **The stale axis range.** A fitted window is *absent* from the fresh
  figure's layout, and `Layout.update` leaves what it does not mention
  alone, so the previous hand-typed range survived the redraw and the boxes
  disagreed with the plot. §23.2 had already hit this for Z and written the
  range on explicitly; Y needed the same, to `layout.yaxis` or
  `layout.scene.yaxis` depending on the kind.

One pre-existing bug surfaced while fixing these: on `yscale="log"` Plotly
reads an axis range as powers of ten, and `RenderOptions.y_range` already
honours that for `ylim`, but the boxes were seeded from raw sample values.
`plot(exp(x), yscale="log")` would have asked for a window up to 10**22026
on the first edit. `_fit_window` converts before measuring, over the
positive samples a log axis can show at all.

No new symbols, no new parameters — `_data_span`, `_fit_window` and the two
flags are private. `tests/test_view_and_mesh.py` gained
`TestAutoYFitsTheCurrentXDomain`: a clipless curve fitting at all, the seeds
matching what is drawn, a pole keeping its clip in preference to the raw
extent, `ylim`/`zlim` escape, and the log-axis units.

## 24. A project-wide review, and the bug the ignored test was holding

Four tests had been failing for long enough to be treated as scenery, and
`--ignore=tests/test_tour_notebooks.py` in CI meant two of them never ran
there at all. Between them they were hiding one user-facing crash and three
defects in the suite's own honesty.

### 24.1 A slider-driven plot raised on display, in every notebook

`plot(a*sin(x))` at a cell's end raised `ValueError: Figure Widgets do not
support frames` — not in some corner, but in Jupyter, for the whole of §11's
interactive feature. §19.4 made `range_controls()` the default display and
routed everything resamplable into `go.FigureWidget`, and a slider's
positions are pre-rendered *frames* carried inside the figure, which is
precisely what `FigureWidget` refuses to accept. The two features want
opposite things from the figure: frames are what let a slider survive
`to_html()` with no kernel (§11), and resampling would have to rebuild every
frame on each keystroke.

`_wants_live_range_controls()` now excludes an interactive result, so the
default display falls back to the plain figure — which is the right answer,
since the slider's control is already inside it. Asked for by name,
`range_controls()` raises a `MathSlateError` explaining the conflict rather
than letting Plotly's own `ValueError` out. The `except ImportError` around
the `FigureWidget` construction had always been too narrow; the fix is to
not reach it, not to widen the catch.

### 24.2 The test that would have caught it was excluded from CI

`tests/test_tour_notebooks.py` is the only place the Jupyter tour runs in a
real kernel, and it is the only thing in the suite that exercises
`_ipython_display_` end to end. CI ignored the whole file to keep the
*marimo* tour's expensive export and generator-synchronisation checks from
failing the library build — a sound decision applied with too broad a
brush. The three marimo classes are deselected by name now, and
`TestItActuallyRuns` runs with everything else.

Its second assertion was stale on top of that: it counted
`application/vnd.plotly.v1+json` outputs and demanded fifteen, but §19.4
means a plot arrives as a widget view wherever ipywidgets is installed. It
reported "only 8 plots rendered" for a notebook that had drawn 32. Both
mime types count now.

### 24.3 Two tests were reading the developer's machine, not the code

- `test_no_ai_package_is_a_runtime_dependency` read `pyproject.toml` with
  `read_text()` and no encoding, so it decoded with the *locale* codec and
  died on any machine whose default is not UTF-8 (cp949, cp1252) as soon as
  that file grew a non-ASCII character. The library itself was already
  clean — every `read_text`/`write_text` in `mathslate/` passes
  `encoding=`; only the test had the hole.
- `test_it_says_exactly_what_is_missing` called `ask()` and asserted on the
  refusal, without forcing the state it was describing. Where an SDK
  happened to be installed the message correctly named the missing *key*
  rather than the package, and the assertion failed. Worse, on a machine
  with a real key in the environment nothing would have raised at all:
  `ask()` would have reached `backend.complete()` and billed a live request
  to whoever ran the tests. Both halves are monkeypatched now.

### 24.4 The annotations were invisible to everyone but us

422 of 424 functions were fully annotated, and no consumer could see one of
them: without a `py.typed` marker (PEP 561) mypy and pyright treat the whole
package as untyped and silently degrade every imported name to `Any`. The
marker is added, the wheel is confirmed to carry it, and
`tests/test_api_surface.py` now pins both halves — the file ships, and
nothing unannotated ships with it. The two stragglers (`Points.__iter__`,
`_join`) lost their `type: ignore` comments and gained real types.

### 24.5 What the review did not find

Recorded because a review that only lists faults says nothing about
coverage. The AST allowlist in `mathslate/ai/suggest.py` holds up: node
types, attributes, calls and builtins are all allowlisted rather than
denied, `__builtins__` is overwritten *after* a caller-supplied namespace is
merged so it cannot be restored, `dataset()` is swapped for a path-refusing
shim, and a wall-clock budget backstops expressions that are cheap to parse
and ruinous to evaluate. Every string that reaches HTML — worksheets,
tables, datasets, analyses, suggestions — goes through `html.escape` at the
interpolation site. No bare `except`, no mutable default arguments, no
unencoded file I/O in the package.

## 25. Restricted execution moved out of process — review of the change

`Suggestion.run()` no longer execs validated code on the caller's thread. It
pickles the source to a child interpreter, runs it there under a
`subprocess` timeout, and pickles the assigned names back. This is the right
direction: the previous wall-clock guard was `within_budget`, which enforces
a deadline by injecting an asynchronous exception into a thread that is
inside NumPy, SymPy or Plotly C code — safe for pure-Python SymPy loops,
which is what it was built for, and not something to rely on for code a
model wrote. A process can simply be killed.

Verified while reviewing: `print()` (an allowed builtin) is captured and
replayed rather than corrupting the pickle stream; importing the package in
the child writes nothing to stdout, so the stream stays clean; `set_budget
(None)` in the child cannot reach the parent's module state; `Dataset`
needed the new `__reduce__` because `MappingProxyType` will not pickle; and
a refusal raised inside the child (`dataset('secrets.csv')`) still arrives
as an `UnsupportedInputError`. Process startup costs about 1.4s of the 10s
budget, which leaves the documented headroom intact.

Three defects found and fixed:

- **`if namespace:` should have been `is not None`.** A non-empty namespace
  was correctly refused, but `run({})` fell through the truthiness test and
  was silently ignored — and, since results now come back from another
  process, the caller's dict was no longer filled in place either. Code
  doing `ns = {}; s.run(ns); ns["y"]` went from working to `KeyError` with
  nothing said. An empty dict is still a caller asking to be given results
  in their object, so it is refused with the same message.
- **A host that cannot fork leaked a raw `OSError`.** Only
  `TimeoutExpired` was caught. Pyodide/JupyterLite has no process creation
  and a hardened container may refuse it, so a method whose contract is
  `UnsupportedInputError` could raise something else entirely.
- **The timeout test had become a tautology.** It monkeypatched
  `_run_restricted` to raise `UnsupportedInputError("execution budget")` and
  then asserted `UnsupportedInputError` matching "execution budget" — it
  would have passed against a `run()` with no budget at all. The original
  had a real reason to test at the mechanism boundary (a live `9**9**9`
  would leave a memory-hungry thread loose in the session); process
  isolation removes that reason, because the bomb is confined to a child
  that gets killed. It now runs the real expression under a shortened
  budget.

### 25.1 The escape hatches did not survive the boundary

A `PlotResult` that crossed it was not the object it had been. Plotly's own
`__reduce__` goes through `to_dict()`, and Plotly 6 encodes numeric arrays
there into the `{"dtype", "bdata"}` typed-array spec plotly.js reads off the
wire, so `result.plotly.data[0].x` came back a `dict` where in-process it is
an `ndarray` — `np.asarray` on it raises `TypeError`. The figure still
rendered and `to_html()` still worked, and `.numpy` was never affected
because it reads the plan rather than the figure, so this was easy to miss;
but PRD 4 makes escape-hatch completeness a success metric, and "the Plotly
Figure — yours to mutate" has to mean the same thing on both sides. Nothing
pickled a result until §25 moved restricted execution out of process, so
this was newly *reachable* rather than newly broken.

`PlotResult.__reduce__` now decodes those specs back to arrays and rebuilds
the figure from the decoded dict. The alternative — regenerating it from
`plan` and `options` via `figure_from_plan` — was rejected on the same
promise it was meant to keep: it would silently discard whatever the caller
had done to `.plotly` in order to fix how the figure travels. Decoding keeps
the mutation and reverses only the transport encoding.

Keying on `dtype`/`bdata`/`shape` is safe because that spec is plotly.js's
wire format rather than an implementation detail of one release, and the
short codes (`f8`, `i4`, …) are NumPy's own, so no translation table is
needed. `frombuffer` gives a read-only view of the decoded bytes, so the
array is copied — in-process trace data is writable, and a clone that is not
would be a second, quieter version of the same defect.

Checked across every kind the library draws — curve, discontinuous (the NaN
breaks compare equal), overlaid, parametric, polar, surface with its 2-D
`shape` key, contour, implicit, space curve, raw data, matrix, and a
21-frame slider figure — plus a caller-applied `update_layout` and the
`.sympy`/`.numpy`/`.plan`/`.python()` hatches on the clone.
`tests/test_api_surface.py::TestTheEscapeHatchesSurvivePickling` pins it;
two of its five fail against the unfixed code and the other three guard the
parts that were already right.

## 26. Figure height, size control, and turning the sidebar off

A Jupyter user's report, and all three parts of it are about the same thing:
how much of a notebook cell the graph gets.

### 26.1 The default height

Nothing set a height, so every figure was Plotly's own 450px. That number is
chosen for a dashboard tile — one panel among several. A notebook cell is the
full width of the page and the graph is the thing being read, and since §19.4
a fifth of that width goes to the range-control sidebar. 450 left the curve in
a letterbox.

`DEFAULT_HEIGHT = 540` — that default times 1.2, enough to stop the squeeze
without pushing the report line under the fold on a laptop screen.

Applied at `figure_from_plan`'s gate rather than inside each of the six figure
builders, and emitted at `generate_code`'s, because "how tall is a plot" has
no per-kind answer and six copies of one answer is six chances for a seventh
builder to forget it. `REPRODUCED` gains `figure width/height`: a figure and
the program `show_python()` says builds it must not come out different sizes.

### 26.2 `width` / `height` / `set_plot_size()`

Per call, and notebook-wide, in the shape §21.1 already established for
`set_range_controls()` — a standing preference stated once at the top should
not have to be repeated per cell, and a call still wins over it.

**Width stays unset by default**, and that is the substantive decision here.
Plotly reads an unset width as "measure the container", which is what makes a
figure fill its cell; a pixel width leaves a gap beside it on a wide screen and
clips it on a narrow one — the trap §23.1 had already fallen into with the
sidebar split, and re-introducing it as a default would have undone that fix.

Two of the three remaining slots §20 left: `NEW_API` goes from 17 to 19. That
is one short of the cap, and the next round has to argue for its symbol rather
than assume it.

### 26.3 `controls=` — the per-plot sidebar switch

§21.1 reasoned that a *global* toggle was the right shape because the standing
preference is what a reader has, and `.range_controls()` by name covers the
other direction. The report shows the missing case: the sidebar earns its fifth
of the width on a curve being explored and not on one being looked at, which is
a per-plot judgement, and `set_range_controls(False)` is too blunt for it while
calling `.range_controls()` on every *other* plot is too tedious.

`controls=None|True|False` on `plot()`, resolved against the global rather than
replacing it. It is **not** a `RenderOptions` field: that module holds what both
renderers read, and neither renderer draws a sidebar. It decides what a notebook
cell displays, changes nothing about the figure, and is absent from the emitted
program — where a plot is shown is not part of what it is.

That makes it the first *display-only* keyword, so `_DISPLAY_KEYWORDS` names it
and the `animate()` keyword-coverage test reads that set. `animate()` does not
take it, and refuses it rather than ignoring it: a slider-driven figure never
shows the sidebar (its positions are pre-rendered frames), so accepting it there
would be an option that ignores the value it was given.

### 26.4 What the existing tests caught

Two regressions, both from tests written for earlier rounds:

* `range_controls(width=, height=)` stopped surviving a redraw. `_apply()`
  replaces the whole layout with the fresh figure's, and the fresh figure now
  carries a height — so a size that lived only on the widget was overwritten the
  first time a box was edited. The size is folded into `controller_options`
  instead, which every rebuild already carries.
* `plot()`'s keyword-coverage test failed on `controls`, which is the test doing
  its job: the exemption is now declared in `_DISPLAY_KEYWORDS` rather than the
  assertion being loosened.

## 27. Three from one Jupyter session

### 27.1 `Linear`/`Log` on two rows — and two bugs under it

Reported on `plot(sqrt(x))`, and the choice of function is the clue: `sqrt` is
non-negative, so it is one of the few plots where `can_log_y` holds and a
*second* scale button exists at all. Under it were two independent defects.

**The stylesheet was malformed.** Each rule in `range_controls` spans several
adjacent string literals, of which only the first was an f-string — so the
doubled braces CSS needs inside an f-string came out literally everywhere else.
Every rule shipped ending `}}` (a stray top-level `}`, which browsers recover
from) and the two written as `selector {{ … }}` shipped as a *nested block*,
which drops the whole declaration list. Those two were exactly the rules that
give a segment control's buttons `flex: 1 1 0`. Rewritten with `.format`, which
has no brace-doubling problem across concatenated literals, and a test asserts
no `{{`/`}}` survives and that the braces balance.

**Y scale was the last `ToggleButtons`.** §22 and §23 had each rewritten one
control — Mode, then Mesh, then Z scale — around the fact that ipywidgets'
`ToggleButtons` view is a *wrapping* flex row, in three separate copies of the
same fix. Y scale was never converted, so it was the one that could still wrap.
The fourth copy is written once, as `_segment`, and every control goes through
it. Each choice carries `(label, value)` explicitly: `Points` selects the trace
mode `scatter`, which no rule could derive from the caption, and a segment that
guessed would set a `kind` `RenderOptions.trace_mode` does not recognise and
quietly draw lines.

**And the control was one-way.** Found while testing the rewrite, present on
master: `_apply` merges the fresh layout with `layout.update()`, and a linear
axis is the *absence* of `yaxis.type` rather than a value — so `Log` took and
`Linear` did nothing, leaving a logarithmic axis wearing the linear window's
numbers. Cleared by name from the fresh figure.

### 27.2 520 rather than 540

§26.1 took Plotly's 450 × 1.2. Correct for the case that prompted it — a curve
beside the sidebar — and loose for a notebook full of surfaces and histograms,
which were never cramped. 520 keeps most of the gain and costs a scroll less
down a page of figures.

### 27.3 `Samples` — and `points=` finally reaching a grid

The sidebar moved the window and the drawing options but not the sampler, which
is the one thing PRD 5.3 is actually about. `_replotted` gains a `config=`, the
controller carries one beside its `RenderOptions`, and a `Samples` slider moves
it. Down as much as up: coarsening a curve to 50 shows the adaptive pass its own
scaffolding, which is what makes "adaptive" visible rather than asserted.

Building it surfaced a latent bug. `points=` set `initial_points`, and the
two-variable samplers never read it — `sample_surface` was called with its
default `resolution`, so `plot(x*y, points=30)` drew the stock 60×60 and said
nothing. A slider with the same gap would have been a control that does nothing
on half the plot kinds.

`SamplingConfig.grid_points` fixes both, and is deliberately a *second* field
rather than a reuse of `initial_points`: the scales differ, 2000 along a line
being 2000 evaluations where 2000 square is four million. `None` means "each
sampler's own default", which is what lets a surface keep 60 and a region keep
the 200 its boundary needs. `points=` is capped at `_MAX_GRID_POINTS` on the way
in, and the slider offers 10–200 per axis on a grid against 20–2000 on a curve,
because one number that means different things depending on what is plotted is
not one control.
