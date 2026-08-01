# Task: Enable `range_controls()` inside marimo, and verify it live in a real browser

## Background you need (this session has no memory of the prior conversation)

`mathslate` (this repo) has a feature, `PlotResult.range_controls()` (see
`mathslate/result.py`), that returns a live ipywidgets/anywidget bundle: a
Plotly `FigureWidget` plus X/Y(/Z) number boxes that **resample** the plot
(not just crop the view) whenever a box's value changes. It also becomes the
*default* thing a bare `plot(...)` shows at a Jupyter/Colab cell's end
(`PlotResult._ipython_display_`). Read `mathslate_prd_0.3.md` sections
**§19 through §23** before touching anything — they document the exact
design decisions, prior bugs, and fixes this feature already went through,
in the same terse "why, not what" prose style the rest of the file uses.
Match that voice for anything you add.

Right now, `range_controls()` explicitly **refuses to run in marimo**:

```python
# mathslate/result.py, inside PlotResult.range_controls()
frontend = detect_frontend()
if frontend not in (Frontend.JUPYTER, Frontend.COLAB):
    raise UnsupportedInputError(
        "range_controls() needs a running Jupyter or Colab kernel. In "
        "marimo, compose mo.ui.number() with plot() directly for the "
        "same live resampling — marimo's reactivity already does what "
        "this method does by hand for ipywidgets. In a plain script or "
        "exported HTML, use xlim/ylim to choose the window up front."
    )
```

The reasoning at the time (§19.2) was "marimo has its own reactive
model — a cell that both creates a `mo.ui.*` element and reads its `.value`
never gets re-run by marimo when that element changes (this is enforced in
marimo's own runtime — see `Kernel.set_ui_element_value` in
`marimo/_runtime/runtime.py`, which explicitly does
`self.graph.get_referring_cells(name) - self.graph.get_defining_cells(name)`,
i.e. *never rerun the cell that created the name*), so `range_controls()`'s
whole `.observe()`-callback design can't work there — the user has to hand-roll
`mo.ui.number()` in one cell and read `.value` in a second cell."

**That reasoning conflates two different things**, and the second one turns
out to be wrong. There are two separate mechanisms in play:

1. marimo's own `mo.ui.*` reactivity (cell-rerun graph) — genuinely
   incompatible with a single self-contained cell, for the reason above.
   This part of the original reasoning is correct and not in question.
2. Whether a plain **ipywidgets** widget's own `.observe()`/comm mechanism
   (which is what `range_controls()` actually uses — `ipywidgets.FloatText`,
   `HBox`, `VBox`, and Plotly's `FigureWidget`, none of which are `mo.ui.*`)
   works *at all* inside marimo. The code above assumes it categorically
   does not. It does not check.

Investigating marimo's installed source turned up this, in
`marimo/_plugins/ui/_impl/anywidget/init.py` (path may differ slightly by
version — find the real one in your environment and read it before
proceeding):

```python
def init_marimo_widget(w: ipywidgets.Widget) -> None:
    ...
    w.comm = MarimoComm(
        comm_id=w._model_id,
        comm_manager=WIDGET_COMM_MANAGER,
        target_name="jupyter.widgets",   # the *standard* ipywidgets protocol name
        data={"state": state, "buffer_paths": buffer_paths, "method": "open"},
        ...
    )
    w.comm.on_msg(w._handle_msg)   # wires incoming browser messages to
                                     # ipywidgets' own standard handler —
                                     # the same one that fires .observe()
```

and, in `marimo/_output/formatters/ipywidgets_formatters.py`:

```python
class IPyWidgetsFormatter(FormatterFactory):
    def register(self) -> None:
        import ipywidgets
        ipywidgets.Widget.on_widget_constructed(init_marimo_widget)
```

Read literally: **every** `ipywidgets.Widget` instance constructed while a
marimo kernel context is active — not just `mo.ui.*` elements, not just
`anywidget`-based ones — gets wired to a working `MarimoComm` using the
*standard* ipywidgets protocol, with incoming messages routed to
`Widget._handle_msg`, which is exactly what updates a traitlet and fires
`.observe()` callbacks in ordinary ipywidgets usage. This is a first-class,
intentional marimo compatibility feature (it exists specifically so plain
ipywidgets code — not written with marimo's own reactivity in mind — still
works when displayed inside marimo), not an accident or an edge case.

`init_marimo_widget` calls `get_context()` and does nothing if that raises
`ContextNotInitializedError` — meaning this mechanism is **only active
inside a real, running marimo kernel** (`marimo edit` / `marimo run`
actually serving a page to a browser). It could not be verified end-to-end
in the environment that did this investigation, because that environment
had no browser and no live marimo kernel — only a static, headless install
of the `marimo` package. **You have both**, which is the entire reason this
task is being handed to you instead of finished remotely.

## The hypothesis to test

If the guard above is relaxed to also allow `Frontend.MARIMO`, then
`result.range_controls()` — called once, as the trailing expression of a
**single marimo cell** — should render as a live widget, and typing a new
value into (e.g.) the "x min" box should cause the figure to actually
resample, with no second cell and no `mo.ui.number()` anywhere. This would
make marimo behave identically to Jupyter/Colab for this feature, for free,
by deleting a restriction rather than adding new code.

This is a hypothesis, not a conclusion. Falsify it properly if it's wrong —
do not force a partial success into looking like a full one.

## What to change (if the hypothesis holds — see verification steps first)

Work from the latest `master` (this repo already has PRD §19–§23 and all of
`range_controls()` merged in). Create a branch for this work (check what
naming convention recent branches on this repo use, e.g.
`claude/code-review-improvements-*`, and follow it, or ask the user).

1. **`mathslate/result.py`**, inside `PlotResult.range_controls()`:
   - Change the frontend check to also accept `Frontend.MARIMO`.
   - Rewrite the `UnsupportedInputError` message: it can no longer say
     "In marimo, compose mo.ui.number()..." as if that's the only option in
     marimo — marimo is now a directly-supported frontend, same as
     Jupyter/Colab. The message should now only fire for `Frontend.PLAIN`
     (a plain script / no live kernel at all), and should say so precisely.
   - Update the method's docstring: it currently frames marimo purely
     through the `mo.ui.number()` composition path (see the existing prose
     citing "marimo re-runs the cell reactively... which is exactly what
     this method emulates by hand for ipywidgets"). That framing is now
     backwards for the direct-call case — `range_controls()` works in
     marimo directly via marimo's own ipywidgets comm compatibility
     (`Widget.on_widget_constructed(init_marimo_widget)`), not by emulating
     marimo's reactivity. Keep the `mo.ui.number()` composition documented
     as a valid *alternative* (it still works, and some readers may prefer
     marimo's native idiom over an ipywidgets bundle rendered inside it),
     not as marimo's only path.
   - **Do not** add `Frontend.MARIMO` to `_wants_live_range_controls()`
     (the gate behind the *automatic* default-display hook,
     `_ipython_display_`). That hook is IPython-specific
     (`InteractiveShell.instance().display_formatter`) and marimo's own
     display resolution never looks for `_ipython_display_` at all — it was
     verified (in the prior session, and re-verify it if you want certainty)
     that marimo's formatter chain checks `_repr_html_` then
     `_repr_mimebundle_` only. Adding `MARIMO` there would be dead code:
     marimo will never reach it through that path. A marimo user gets the
     live widget by writing `result.range_controls()` (or
     `plot(...).range_controls()`) as their cell's trailing expression —
     which is the normal, idiomatic way marimo renders anything — not by
     `plot(...)` alone auto-upgrading itself the way it does in Jupyter.
     Leave a short comment explaining this so the next reader doesn't
     "fix" it by adding MARIMO there.

2. **`mathslate/ui/adapters.py`**, `_range_controls_status()` (feeds
   `frontend_report()`): this function describes readiness for the
   *automatic default display* specifically, which is unchanged by this
   work (still Jupyter/Colab-only, see above) — it should not need a
   functional change. Read it and confirm that's still accurate; if the
   wording implies marimo can never show `range_controls()` at all
   (rather than "not automatically"), soften it so it isn't misleading now
   that an explicit call works there.

3. **`examples/marimo_notebook.py`**: it currently has an `mo.ui.number()` +
   `plot()` two-cell example (added in a prior round, PRD §19.2/4.13). Keep
   it (it's still a valid, marimo-native pattern some readers will prefer),
   but add a new cell demonstrating the single-cell form:
   ```python
   result = plot(sin(x)/x, (x, -10, 10), verbose=False)
   result.range_controls()
   ```
   as the cell's trailing expression. Confirm both patterns actually render
   and work when you run this notebook live (see verification below) —
   don't just add the cell and assume.

4. **`docs/manual.md`** / **`docs/manual_ko.md`**, §4.13 ("Live range
   controls"): currently states marimo needs the `mo.ui.number()` composition
   because it "needs no wrapper" — rewrite to lead with the single-cell form
   now that it works, and keep the `mo.ui.number()` composition as a
   documented alternative for readers who want marimo's own idiom. Update
   both language versions; they must stay in sync (existing convention in
   this file).

5. **`mathslate_prd_0.3.md`**: add a new numbered section (check the last
   section number in the file — likely §24 by the time you read this, but
   verify) documenting this change in the same voice as §19–§23: what was
   asked, why the original restriction existed, what was actually found in
   marimo's source (quote it, the way §23 quotes Plotly internals), how it
   was verified (be exact about what automated tests covered vs. what
   required a live browser — see below), and the outcome. If the hypothesis
   turned out to be *false*, write that section describing the real failure
   mode instead — this file's whole value is being an honest record, not a
   changelog of successes only.

6. **Tests** — `tests/test_view_and_mesh.py`: there is an existing test,
   `TestLiveRangeResampling::test_a_plain_script_is_told_to_use_marimo_or_xlim`,
   that asserts calling `range_controls()` on a non-notebook host raises an
   error mentioning "marimo" as the suggested alternative. Once marimo is a
   supported frontend rather than alternative advice, this test's premise
   changes — update it to match the new message, and add a
   `Frontend.MARIMO`-stubbed test class mirroring the existing
   `_colab` fixture pattern already used throughout this file (search for
   `monkeypatch.setitem(sys.modules, "google.colab", ...)` for the pattern,
   and `test_marimo_is_unaffected_because_it_never_looks_for_the_hook`, which
   already shows how to stub `sys.modules["marimo"]` with a
   `running_in_notebook` attribute so `detect_frontend()` reports MARIMO).
   These headless tests can only prove the widget *constructs* without
   raising under a simulated marimo frontend — they cannot prove the live
   comm round-trip works, because `init_marimo_widget` requires a real
   marimo kernel context. Say so in the test docstrings; don't imply more
   coverage than there is.

## Verification — this is the actual point of doing this locally

Do not consider this done on the strength of the headless tests passing.
The whole reason this was handed to a local session is to get the one piece
of evidence that couldn't be gathered before: **does a value typed into a
`range_controls()` box, inside a real running marimo notebook viewed in a
real browser, actually cause the figure to resample?**

1. Run the full existing test suite first, unmodified, to confirm your
   starting point is clean:
   `python -m pytest -q --ignore=tests/test_tour_notebooks.py`

2. Make the code changes above.

3. Run the (now-updated) test suite again. All headless tests must pass.

4. **Live check, with a real browser** — this environment has Chromium
   available (Playwright-drivable) and a real `marimo` install, so drive
   this with Playwright rather than describing it as untestable:
   - Start `marimo edit examples/marimo_notebook.py` (or a scratch copy)
     as a subprocess, pointed at a local port.
   - Open it with Playwright.
   - Locate the cell whose trailing expression is `result.range_controls()`;
     confirm the widget actually renders (a Plotly chart plus number boxes
     visible in the page, not an error traceback in the cell's output).
   - Programmatically fill a new value into the "x min" box (Playwright
     `fill()`/`press("Enter")` or `blur()` as appropriate for how ipywidgets'
     `FloatText` commits a value in the browser) and wait briefly.
   - Confirm the plot actually changed: e.g. screenshot before/after and
     compare, or (more reliably) read back some property of the rendered
     Plotly figure via the page's own JS state if accessible, or add a
     temporary visible marker (like a title reflecting the current x-range)
     to the test notebook cell so success/failure is unambiguous from a
     screenshot alone. Take and keep screenshots either way — they're the
     evidence for the PRD entry and for the user.
   - Also spot-check: a 3D surface's z-controls, and that moving Y on an
     ordinary curve does *not* change the point count (i.e. the same
     resample-vs-view-clip distinction documented in §19/§23 still holds
     inside marimo) — at least one such case, not just the simplest one.
   - If interaction fails, or the cell errors, or nothing updates: capture
     the browser console output and the marimo server's stdout/stderr, and
     the exact symptom. This is a real, useful outcome — write it up
     honestly in §24 (or whatever section number), revert the guard change
     in `range_controls()` back to Jupyter/Colab-only, and leave the
     `mo.ui.number()` composition as marimo's documented path, same as
     before. Do not leave the guard relaxed if it does not actually work —
     that would replace a clear error message with a silent or confusing
     failure for every marimo user who hits it.

5. Only if the live check in step 4 genuinely succeeds: finish the
   documentation (PRD §24, manuals, example notebook), run the full test
   suite once more, commit, and push, following this repo's normal
   fast-forward branch → master workflow (check recent commit history with
   `git log --oneline -20` to confirm the pattern before pushing to master).
   Do not merge to master without the user's confirmation if that is not
   already how this session is configured to work.

## Do not

- Do not mark this done based on "the widget object was constructed and no
  exception was raised" — that was already checkable without a browser and
  is not what this task is for.
- Do not silently drop the `mo.ui.number()` composition path from the docs
  even if the direct call works — it remains valid, and some readers
  legitimately prefer marimo's own idiom over an ipywidgets bundle.
- Do not guess at marimo internals from this prompt's quoted source without
  re-reading the actual installed version in your environment first — marimo
  is under active development and the exact file/function names may have
  moved by the time you run this.
