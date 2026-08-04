"""The live range-control sidebar (PRD §19, §22, §23).

This is one method's worth of code — :func:`build`, which
:meth:`mathslate.result.PlotResult.range_controls` delegates to — and it lived
inside ``result.py`` until it was 60% of that file. Everything here exists only
to construct the ipywidgets sidebar and keep it in step with the figure beside
it; nothing in ``result.py`` needs any of it in order to *be* a result.

:func:`build` reaches for ``PlotResult``'s private attributes rather than a
public accessor apiece. The two modules are one unit split for size, not two
layers with a contract between them, and Python has no way to say so — so this
line is drawn here instead: the split was a move, and inventing a public
surface as part of it would have made "nothing changed" impossible to check.

Importing this module pulls in no frontend package. ``ipywidgets`` is imported
inside :func:`build`, where a missing one becomes a message rather than an
``ImportError`` at ``import mathslate``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go

from ..errors import MathSlateError, UnsupportedInputError
from ..render.options import DEFAULT_MESH_LINES
from .adapters import Frontend, detect_frontend

if TYPE_CHECKING:
    from ..result import PlotResult

__all__ = ["build", "deps_available", "set_range_controls", "get_range_controls"]


#: Whether a bare `plot(...)` shows `range_controls()` by default where it
#: can (PRD §19.4). On by default — that default is the point of §19.4;
#: `set_range_controls(False)` is the escape hatch §20 opened a slot for.
_RANGE_CONTROLS_DEFAULT: bool = True

#: The sidebar's share of the total width (PRD §23.1) — a percentage on both
#: it and the figure's wrapping `Box`, rather than pixels on one or both
#: (§22.4's fixed-pixel split, which left blank space once the container was
#: wider than the pixel total). `FigureWidget.layout` cannot take a CSS width
#: at all (it is Plotly's chart config, not a CSS box), which is why the
#: figure is wrapped rather than sized directly.
_RANGE_CONTROLS_SIDEBAR_PERCENT: int = 20
#: Floor below which the sidebar stops shrinking with the percentage split,
#: protecting legibility on a narrow window the same way §22.4 protected the
#: default ratio on a normal one.
_RANGE_CONTROLS_MIN_SIDEBAR_WIDTH: str = "240px"
#: ``FigureWidget.layout`` is Plotly's figure layout, not the ipywidgets CSS
#: layout trait.  The class lets the small stylesheet in ``range_controls``
#: size the widget root itself when the plot has no explicit pixel width.
_RANGE_CONTROLS_FIGURE_CLASS: str = "mathslate-range-controls-figure"
_RANGE_CONTROLS_INPUT_CLASS: str = "mathslate-range-controls-input"
_RANGE_CONTROLS_COMPACT_CLASS: str = "mathslate-range-controls-compact"


def set_range_controls(value: bool) -> None:
    """Turn `range_controls()` as `plot()`'s default display on or off (PRD §21).

    Capability still gates it either way — Jupyter/Colab, `ipywidgets` and
    (on Plotly >= 6) `anywidget` installed, a plot with a domain to resample —
    this only controls whether a bare `plot(...)` *reaches for* the widget
    when all of that is true. Off returns to the plain figure everywhere,
    which is what every plot showed before §19.4 and is still what
    `.range_controls()` called by name gives you regardless of this setting.
    """
    global _RANGE_CONTROLS_DEFAULT
    _RANGE_CONTROLS_DEFAULT = bool(value)


def get_range_controls() -> bool:
    """Whether a bare `plot(...)` currently reaches for `range_controls()`."""
    return _RANGE_CONTROLS_DEFAULT


def _data_span(arrays: Iterable[Any]) -> tuple[float, float] | None:
    """The finite extent of ``arrays`` — what "fit to the data" means.

    ``Auto Y``/``Auto Z`` reach for this when no automatic clip applies, which
    on a well-behaved curve is the usual case rather than the exception. The
    extent is exact rather than padded, because these numbers are also what
    the sidebar's boxes display: a window wider than the values it claims to
    bound would be a number with no relationship to what is on screen.

    A constant function collapses to a single value, and Plotly draws a
    zero-height window as an empty axis, so that one case is opened out.
    ``None`` comes back when there is nothing finite to measure, and the
    caller then leaves the view alone rather than building one from NaN.
    """
    lows: list[float] = []
    highs: list[float] = []
    for array in arrays:
        values = np.asarray(array, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            lows.append(float(finite.min()))
            highs.append(float(finite.max()))
    if not lows:
        return None
    low, high = min(lows), max(highs)
    if high <= low:
        margin = max(abs(low), 1.0) * 0.05
        return (low - margin, high + margin)
    return (low, high)


def _fit_window(arrays: Iterable[Any], *, log: bool) -> tuple[float, float] | None:
    """:func:`_data_span`, in the units the axis is actually labelled in.

    Plotly reads a log axis's range as powers of ten, and ``ylim`` already
    follows that convention there (`RenderOptions.y_range`). A window measured
    off the raw samples would therefore ask for 10**22026 on something as
    ordinary as ``plot(exp(x), yscale="log")`` — so the samples are converted
    first, over the positive values a log axis can show at all.
    """
    if not log:
        return _data_span(arrays)
    exponents: list[Any] = []
    for array in arrays:
        values = np.asarray(array, dtype=float)
        drawable = values[np.isfinite(values) & (values > 0.0)]
        if drawable.size:
            exponents.append(np.log10(drawable))
    return _data_span(exponents)


def _can_use_log_scale(arrays: Iterable[Any]) -> bool:
    """Whether every drawable value is positive enough for a truthful log view."""
    found = False
    for array in arrays:
        values = np.asarray(array, dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        if np.any(finite <= 0.0):
            return False
        found = True
    return found


def deps_available() -> bool:
    """Whether :func:`build` could build its widget now.

    A cheap, side-effect-free stand-in for actually building one: an empty
    ``FigureWidget`` costs nothing to construct and fails exactly when a real
    one would — including Plotly >= 6's extra need for ``anywidget`` — without
    cloning the real figure just to find out.
    """
    if importlib.util.find_spec("ipywidgets") is None:
        return False
    try:
        go.FigureWidget()
    except ImportError:
        return False
    return True


def build(
    result: PlotResult, *, width: int | None = None, height: int | None = None
) -> Any:
    """Live view and display controls beside this plot (PRD §19, §22, §23).

    This is the missing half of PRD §18's ``xlim``/``ylim``: those decide
    the *initial* window, and Plotly's own drag-to-zoom only ever crops
    what was already sampled there. These controls instead call
    :meth:`_replotted` on every change, so narrowing X to look closely at
    one feature draws it at full resolution rather than showing fewer of
    the original points — the Excel-style "edit the axis, get a redrawn
    chart" behaviour, not a screenshot crop. They sit beside the figure,
    not above it, so widening the window is not also narrowing the plot.
    A 2D curve additionally gets scale and line/points switches. A true
    3D surface (§23.2) gets two more boxes, a Z-scale switch, and mesh
    on/off plus density controls. Z is the function's *output*, not a
    domain, so moving its bounds only ever re-applies ``zlim``, never a
    resample. Log choices are offered only when every drawable value is
    positive, avoiding a view that silently discards part of a graph.

    ``width``/``height`` set the figure's own pixel size — the same
    ``layout.width``/``layout.height`` :meth:`plotly` always exposed, in
    the one place a reader is already looking to size this widget. They
    survive every later resample: only a ``FigureWidget.layout.update()``
    happens on change, never a fresh construction, and Plotly's own update
    leaves properties the new layout does not mention alone.

    Left unset, the figure keeps a size already given it (e.g. by
    ``.plotly.update_layout(...)`` before this call) or otherwise fills
    whatever room is left once the sidebar takes its own fixed share
    (§23.1) — a percentage on both sides, so together they always cover
    the full width regardless of how wide the notebook actually is, which
    a fixed-pixel split (§22.4's first fix) could not promise. There is
    no drag handle between the two; resize by calling this again with a
    new ``width``/``height``.

    Needs a running Jupyter or Colab kernel with ``ipywidgets`` installed,
    because updating a *displayed* figure in place needs a live host to
    push the new trace data to. marimo gets the same live resampling for
    free by composing ``mo.ui.number()`` directly with :func:`plot`
    instead — marimo re-runs the cell reactively on every change, which is
    exactly what this method emulates by hand for ipywidgets. A plain
    script or an exported HTML file has no kernel to push updates through,
    so ``xlim``/``ylim`` remain their way to choose a window.

    This is what a bare ``plot(...)`` shows by default at a cell's end
    whenever the three conditions above hold — see :meth:`_ipython_display_`.
    Call it directly only to keep the widget instead of the report, or to
    build one from a result that was printed with ``verbose=False``.
    """
    plan = result._plan
    if not result._resamplable():
        # A property of the plot itself, so it applies regardless of host —
        # checked before the environment questions below can obscure it.
        raise UnsupportedInputError(
            f"a {plan.kind} plot has no symbolic domain; range_controls() "
            "needs a plotted expression, not raw data."
        )
    if result.interactive:
        # `go.FigureWidget` rejects a figure carrying frames outright, so
        # without this the reader would get Plotly's own ValueError — and,
        # before §19.4's default display learned to skip these, got it
        # merely for evaluating `plot(a*sin(x))` in a cell. The two
        # controls also want incompatible things from the figure: a
        # slider's positions are pre-rendered frames inside it (§11, which
        # is what lets one survive `to_html()`), and resampling would have
        # to rebuild every one of them on each keystroke.
        raise UnsupportedInputError(
            "range_controls() cannot drive a slider-driven figure: its "
            "positions are frames carried inside it, and Plotly's "
            "FigureWidget does not accept frames. Use xlim/ylim for the "
            "window, or Slider.widget() for live recomputation."
        )
    for name, value in (("width", width), ("height", height)):
        if value is not None and not int(value) > 0:
            raise UnsupportedInputError(
                f"range_controls({name}={value!r}) must be a positive "
                "number of pixels."
            )

    frontend = detect_frontend()
    if frontend not in (Frontend.JUPYTER, Frontend.COLAB):
        raise UnsupportedInputError(
            "range_controls() needs a running Jupyter or Colab kernel. In "
            "marimo, compose mo.ui.number() with plot() directly for the "
            "same live resampling — marimo's reactivity already does what "
            "this method does by hand for ipywidgets. In a plain script or "
            "exported HTML, use xlim/ylim to choose the window up front."
        )
    try:
        import ipywidgets as widgets
    except ImportError as error:
        if sys.platform == "emscripten":
            warnings.warn(
                "range_controls() is unavailable because this JupyterLite "
                "kernel has no ipywidgets; showing the regular Plotly graph.",
                RuntimeWarning,
                stacklevel=2,
            )
            return result._figure
        raise UnsupportedInputError(
            "ipywidgets is not installed. `pip install mathslate[jupyter]`."
        ) from error
    try:
        # Plotly >= 6 rebuilt FigureWidget on `anywidget`, which
        # `mathslate[jupyter]` installs alongside ipywidgets (PRD §21) —
        # this only fires for an environment that has ipywidgets some
        # other way, without it.
        figure_widget = go.FigureWidget(result._figure)
    except ImportError as error:
        if sys.platform == "emscripten":
            warnings.warn(
                "range_controls() is unavailable because this JupyterLite "
                "kernel has no anywidget; showing the regular Plotly graph.",
                RuntimeWarning,
                stacklevel=2,
            )
            return result._figure
        raise UnsupportedInputError(
            "this Plotly version's FigureWidget also needs `anywidget`. "
            "`pip install mathslate[jupyter]` installs both."
        ) from error
    # `FigureWidget.layout` is Plotly's chart-layout object (it is what
    # width=/height= below set) — the same name a plain ipywidgets box
    # uses for its own CSS box model, but not the same thing, and
    # `FigureWidget` shadows the trait so there is no separate CSS layout
    # to reach on it directly (§22.4's first attempt tried exactly that
    # and failed). A percentage split is put on a plain `Box` wrapped
    # around the figure instead, sized against the sidebar's own
    # percentage so the two always sum to the full width with nothing
    # left over (§23.1) — pixels alone could not do that, because a
    # container wider or narrower than the pixel total either clips or
    # leaves a gap, which is what a fixed-pixel split (§22.4) actually
    # shipped.
    if width is not None:
        figure_widget.layout.width = int(width)
    if height is not None:
        figure_widget.layout.height = int(height)
    # Height is independent of width.  Supplying ``height=`` alone must
    # still let the graph use all of the horizontal room beside the
    # sidebar; only an actual pixel *width* opts out of that behaviour.
    if width is None and figure_widget.layout.width is None:
        # No size from the call and none from an earlier
        # `.plotly.update_layout(...)` either: fill whatever the wrapping
        # Box below is given, which is what made a lone figure fill 100%
        # of its cell before this method added a sidebar next to it.
        # `layout.width = None` alone does not do this — that only tells
        # Plotly not to *fight* a container size, and Plotly.js still
        # falls back to its own default pixel size unless something
        # tells it to actually measure one. `config.responsive` is that
        # something; it is not a `layout` property (which is why it is
        # not alongside `width=`/`height=` above) — it lives in the
        # separate Plotly.js config object FigureWidget exposes as
        # `_config`, and reassigning the dict is what a traitlets Dict
        # needs to notice the change and sync it to the frontend.
        figure_widget._config = {**figure_widget._config, "responsive": True}
        figure_widget.add_class(_RANGE_CONTROLS_FIGURE_CLASS)
        # Plotly's responsive option shrinks a widget that is too wide,
        # but FigureWidget itself keeps its 700px intrinsic CSS width on
        # a wide notebook.  Its ``layout`` attribute cannot set CSS (it
        # is the Plotly chart layout), so give its widget root a scoped
        # CSS width instead.  Put the style widget first so Jupyter
        # applies the rule before it creates the FigureWidget view.
        figure_style = widgets.HTML(
            value=(
                "<style>"
                f".{_RANGE_CONTROLS_FIGURE_CLASS} {{ width: 100% !important; }}"
                "</style>"
            ),
            layout=widgets.Layout(display="none"),
        )

        def _autosize_after_first_view(change: dict[str, Any]) -> None:
            """Measure the CSS-expanded widget once its browser view exists.

            The stylesheet above can reach the browser just after
            FigureWidget's first Plotly render.  Plotly then still holds
            its 700px initial measurement until the browser is resized.
            ``_view_count`` is synchronized when the frontend has made
            the view, so this relayout is the programmatic equivalent of
            that otherwise-required browser resize.
            """
            if change["new"] <= 0:
                return
            figure_widget.unobserve(_autosize_after_first_view, names="_view_count")
            figure_widget.update_layout(autosize=True)

        figure_widget.observe(_autosize_after_first_view, names="_view_count")
        figure_container = widgets.Box(
            [figure_style, figure_widget],
            # ``HBox`` otherwise sizes itself to its contents in several
            # notebook frontends.  Let this box grow into the space left
            # by the sidebar, and allow it to shrink below Plotly's
            # intrinsic width rather than leaving unused room on its
            # right.  The percentage remains the intended 80/20 split;
            # flex makes that split occupy the outer box's actual width.
            layout=widgets.Layout(
                width=f"{100 - _RANGE_CONTROLS_SIDEBAR_PERCENT}%",
                flex="1 1 0",
                min_width="0",
            ),
        )
    else:
        # An explicit size (from this call or an earlier one) is a
        # request for exactly that many pixels; stretching its wrapper
        # to a percentage would just leave blank space beside it instead
        # of respecting the number asked for.
        figure_container = widgets.Box([figure_widget])

    resamples_y = plan.axes is not None
    # Every y number below — the seeds, what the boxes push back as
    # ``ylim``, and what Auto Y fits — is in the axis's own units, and on
    # a log axis Plotly's are powers of ten.
    log_y = result._options.log_y
    x_lo, x_hi = plan.param_range or (0.0, 1.0)
    if resamples_y:
        y_lo, y_hi = plan.second_range
    else:
        # These boxes are pushed back as ``ylim`` the moment any *other*
        # box is edited, so a starting value unrelated to the window on
        # screen does not merely look wrong — it crops the plot to itself
        # on the first X change. ``plan.y_range`` alone cannot supply one:
        # it is the auto-clip, unset on any curve well behaved enough not
        # to need clipping (and never set at all on a space curve, whose
        # ``y`` is a plotted coordinate rather than a domain). Ask
        # ``options`` for the window actually rendered, then measure the
        # drawn values when there is no window because none was needed.
        window = result._options.y_range(plan) or _fit_window(
            (series.sample.y for series in plan.series), log=log_y
        )
        y_lo, y_hi = window if window is not None else (-1.0, 1.0)
    has_z = result._has_z_axis()
    if has_z:
        # z_range is the *active auto-clip*, unset on a well-behaved
        # surface with nothing to clip — the box still needs a starting
        # number, so it falls back to the surface's own data range
        # rather than a value with no relationship to what is on screen.
        clipped = result._options.z_range(plan) or _data_span(
            [plan.series[0].sample.z]
        )
        z_lo, z_hi = clipped if clipped is not None else (0.0, 1.0)
    else:
        z_lo, z_hi = (0.0, 0.0)

    # Pair each axis' endpoints in one compact row: ``x [min] ~ [max]``.
    # The input widths deliberately leave room for both labels and their
    # borders, avoiding the one- or two-pixel horizontal overflow some
    # ipywidgets themes otherwise create.
    box_style = {"description_width": "0px"}
    box_layout = widgets.Layout(
        width="calc(50% - 12px)", flex="1 1 0", min_width="0"
    )
    # Label the first row with what it actually moves. On a curve or a surface
    # the domain symbol *is* the horizontal axis, so "x" is both true and the
    # more familiar of the two names. On a parametric, polar or space curve it
    # is not: those boxes hold the parameter's window, and calling it "x" told
    # the reader they were cropping the horizontal axis while they were in fact
    # drawing more or less of the curve.
    domain_label = (
        plan.symbol.name
        if plan.symbol is not None and plan.kind in {"parametric", "polar", "space"}
        else "x"
    )
    x_min = widgets.FloatText(
        value=x_lo, description="", step=None,
        style=box_style, layout=box_layout,
    )
    x_max = widgets.FloatText(
        value=x_hi, description="", step=None,
        style=box_style, layout=box_layout,
    )
    y_min = widgets.FloatText(
        value=y_lo, description="", step=None, style=box_style, layout=box_layout,
    )
    y_max = widgets.FloatText(
        value=y_hi, description="", step=None, style=box_style, layout=box_layout,
    )
    boxes = [x_min, x_max, y_min, y_max]
    if has_z:
        z_min = widgets.FloatText(
            value=z_lo, description="", step=None, style=box_style, layout=box_layout,
        )
        z_max = widgets.FloatText(
            value=z_hi, description="", step=None, style=box_style, layout=box_layout,
        )
        boxes += [z_min, z_max]

    for box in boxes:
        box.add_class(_RANGE_CONTROLS_INPUT_CLASS)

    def _range_row(label: str, lower: Any, upper: Any) -> Any:
        return widgets.HBox(
            [
                widgets.Label(label, layout=widgets.Layout(width="16px")),
                lower,
                widgets.Label(
                    "~", layout=widgets.Layout(width="8px", margin="0", padding="0")
                ),
                upper,
            ],
            layout=widgets.Layout(
                width="calc(100% - 4px)", margin="0 2px", align_items="center"
            ),
        )

    range_rows = [
        _range_row(domain_label, x_min, x_max),
        _range_row("y", y_min, y_max),
    ]
    if has_z:
        range_rows.append(_range_row("z", z_min, z_max))

    controls_style = widgets.HTML(
        value=(
            "<style>"
            f".{_RANGE_CONTROLS_INPUT_CLASS} {{ "
            "box-sizing: border-box; max-width: 100% !important; "
            "min-width: 0 !important; }}"
            f".{_RANGE_CONTROLS_INPUT_CLASS} input {{ "
            "box-sizing: border-box; max-width: 100% !important; "
            "min-width: 0 !important; }}"
            f".{_RANGE_CONTROLS_COMPACT_CLASS} {{ "
            "box-sizing: border-box; max-width: 100% !important; "
            "min-width: 0 !important; overflow: hidden; }}"
            f".{_RANGE_CONTROLS_COMPACT_CLASS} .widget-button, "
            f".{_RANGE_CONTROLS_COMPACT_CLASS} .widget-toggle-button {{ "
            "box-sizing: border-box; min-width: 0 !important; "
            "padding-left: 3px !important; padding-right: 3px !important; }}"
            # ipywidgets ToggleButtons uses a wrapping flex row by
            # default.  Two 72px buttons can therefore split at a theme's
            # extra border/gap pixel even inside a 148px controller slot.
            # Segment controls must stay a single compact row.
            f".{_RANGE_CONTROLS_COMPACT_CLASS}.widget-toggle-buttons {{ "
            "display: flex !important; flex-flow: row nowrap !important; "
            "flex-wrap: nowrap !important; }}"
            f".{_RANGE_CONTROLS_COMPACT_CLASS}.widget-toggle-buttons "
            "> .widget-toggle-button {{ flex: 1 1 0 !important; "
            "width: 50% !important; }}"
            f".{_RANGE_CONTROLS_COMPACT_CLASS}.widget-toggle-buttons "
            "> .widget-toggle-button:only-child {{ width: 100% !important; }}"
            "</style>"
        ),
        layout=widgets.Layout(display="none"),
    )

    # These options belong to this displayed controller, not the original
    # PlotResult. Every redraw starts from them so a changed scale, mode or
    # mesh density survives the next X/Y/Z range edit.
    controller_options = result._options
    z_scale = "linear"
    thickness_percent = 100
    has_flat_trace_mode = not resamples_y and not has_z
    # A parametric space curve has a Z axis, but its visible geometry is
    # still a line whose width can be adjusted like a 2D trace.
    has_trace_thickness = has_flat_trace_mode or plan.kind == "space"
    has_surface_mesh = plan.kind in {"surface", "psurface"}
    can_log_y = has_flat_trace_mode and _can_use_log_scale(
        series.sample.y for series in plan.series
    )
    can_log_z = has_z and _can_use_log_scale(
        [plan.series[0].sample.z]
    )

    syncing = False

    def _apply_thickness() -> None:
        """Scale trace widths, retaining each plot family's default width."""
        if not has_trace_thickness:
            return
        factor = thickness_percent / 100
        line_width = 4 if plan.kind == "space" else 2
        with figure_widget.batch_update():
            for trace in figure_widget.data:
                trace.line.width = line_width * factor
                if trace.type == "scatter":
                    trace.marker.size = 6 * factor

    def _apply(fresh: PlotResult) -> None:
        with figure_widget.batch_update():
            figure_widget.data = []
            figure_widget.add_traces(fresh.plotly.data)
            figure_widget.layout.update(fresh.plotly.layout.to_plotly_json())
            if has_z:
                figure_widget.layout.scene.zaxis.type = z_scale
        _apply_thickness()

    def _redraw(_change: dict[str, Any] | None = None) -> None:
        nonlocal controller_options
        if syncing:
            return
        if not all(math.isfinite(box.value) for box in boxes):
            return  # wait for a finite value instead of leaking a callback traceback
        if x_max.value <= x_min.value or y_max.value <= y_min.value:
            return  # a mid-edit, inconsistent pair — wait for both to settle
        if has_z and z_max.value <= z_min.value:
            return
        try:
            fresh = result._replotted(
                x_range=(x_min.value, x_max.value),
                y_range=(y_min.value, y_max.value),
                z_range=(z_min.value, z_max.value) if has_z else None,
                render_options=controller_options,
            )
        except (MathSlateError, ValueError, OverflowError):
            return  # e.g. a range too narrow for this expression to sample
        controller_options = fresh._options
        _apply(fresh)

    for box in boxes:
        box.observe(_redraw, names="value")

    def _set_pairs(*pairs: tuple[Any, Any, float, float]) -> None:
        """Update related fields once, then redraw once."""
        nonlocal syncing
        syncing = True
        try:
            for lower, upper, lo, hi in pairs:
                lower.value, upper.value = lo, hi
        finally:
            syncing = False
        _redraw()

    def _zoom(lower: Any, upper: Any, factor: float) -> None:
        middle = 0.5 * (lower.value + upper.value)
        half_span = 0.5 * (upper.value - lower.value) * factor
        _set_pairs((lower, upper, middle - half_span, middle + half_span))

    def _autoscale_y(_button: Any) -> None:
        """Fit a curve's Y view to the values in its current X domain.

        Only a flat kind ever gets here: a 3D one puts ``Auto Z`` in this
        button's place, so there is no Z to carry through the resample and
        no ``scene`` axis to write the result to.
        """
        if resamples_y:
            return  # on surfaces Y is an input domain, not a view axis
        nonlocal controller_options, syncing
        try:
            fresh = result._replotted(
                x_range=(x_min.value, x_max.value),
                auto_y=True,
                render_options=controller_options,
            )
        except (MathSlateError, ValueError, OverflowError):
            return
        # ``plan.y_range`` is the *auto-clip* — unset precisely when the
        # curve is well behaved enough not to need one, which is most
        # curves. Reading it alone (what this used to do) therefore left
        # the button dead on `sin(x)`, `x**2` and every other bounded
        # function: exactly the cases where fitting Y is easiest. Ask
        # ``options`` first so a genuine clip still wins, then fall back
        # to the values actually drawn — the same two-step ``Auto Z``
        # already made below.
        fitted = fresh._options.y_range(fresh.plan)
        if fitted is None:
            fitted = _fit_window(
                (series.sample.y for series in fresh.plan.series),
                log=controller_options.log_y,
            )
        if fitted is None:
            return
        syncing = True
        try:
            y_min.value, y_max.value = fitted
        finally:
            syncing = False
        controller_options = fresh._options
        _apply(fresh)
        # A fitted Y window may be *absent* from the fresh figure's layout
        # (nothing to clip, no ylim), and ``Layout.update`` leaves what it
        # does not mention alone — so the stale hand-typed range would
        # survive the redraw and the boxes would disagree with the plot.
        # Write it on explicitly. Only a flat kind reaches here (a 3D one
        # gets Auto Z in this button's place), so the axis is the
        # cartesian one, never ``scene.yaxis``.
        figure_widget.layout.yaxis.range = fitted

    def _autoscale_z(_button: Any) -> None:
        """Fit a surface's Z view to its current X/Y domain."""
        nonlocal controller_options, syncing
        try:
            fresh = result._replotted(
                x_range=(x_min.value, x_max.value),
                y_range=(y_min.value, y_max.value),
                auto_z=True,
                render_options=controller_options,
            )
        except (MathSlateError, ValueError, OverflowError):
            return
        fitted = fresh._options.z_range(fresh.plan)
        if fitted is None:
            fitted = _data_span([fresh.plan.series[0].sample.z])
        if z_scale == "log":
            fitted = _fit_window([fresh.plan.series[0].sample.z], log=True)
        if fitted is None:
            return
        syncing = True
        try:
            z_min.value, z_max.value = fitted
        finally:
            syncing = False
        controller_options = fresh._options
        _apply(fresh)
        # An automatic surface has no ``scene.zaxis.range`` entry in its
        # fresh layout. ``Layout.update`` preserves an old manual entry
        # in that case, so replace it explicitly when Auto Z clears one.
        figure_widget.layout.scene.zaxis.range = fitted

    def _reset(_button: Any) -> None:
        pairs: list[tuple[Any, Any, float, float]] = [
            (x_min, x_max, x_lo, x_hi),
            (
                y_min,
                y_max,
                math.log10(y_lo) if controller_options.log_y else y_lo,
                math.log10(y_hi) if controller_options.log_y else y_hi,
            ),
        ]
        if has_z:
            pairs.append((
                z_min,
                z_max,
                math.log10(z_lo) if z_scale == "log" else z_lo,
                math.log10(z_hi) if z_scale == "log" else z_hi,
            ))
        _set_pairs(*pairs)

    def _change_y_scale(change: dict[str, Any]) -> None:
        """Keep the Y boxes in the units Plotly expects for the new scale."""
        nonlocal controller_options, syncing
        if change.get("name") != "value" or change["new"] == (
            "log" if controller_options.log_y else "linear"
        ):
            return
        if change["new"] == "log":
            bounds = (math.log10(y_min.value), math.log10(y_max.value))
            controller_options = replace(
                controller_options, yscale="log", ylim=bounds
            )
        else:
            bounds = (10.0 ** y_min.value, 10.0 ** y_max.value)
            controller_options = replace(
                controller_options, yscale=None, ylim=bounds
            )
        syncing = True
        try:
            y_min.value, y_max.value = bounds
        finally:
            syncing = False
        _redraw()

    def _change_trace_mode(change: dict[str, Any]) -> None:
        nonlocal controller_options
        if change.get("name") != "value":
            return
        controller_options = replace(controller_options, kind=change["new"])
        _redraw()

    def _change_z_scale(change: dict[str, Any]) -> None:
        """Switch a 3D scene's Z axis without resampling its surface."""
        nonlocal controller_options, syncing, z_scale
        if change.get("name") != "value" or change["new"] == z_scale:
            return
        if change["new"] == "log":
            bounds = (math.log10(z_min.value), math.log10(z_max.value))
        else:
            bounds = (10.0 ** z_min.value, 10.0 ** z_max.value)
        z_scale = change["new"]
        controller_options = replace(controller_options, zlim=bounds)
        syncing = True
        try:
            z_min.value, z_max.value = bounds
        finally:
            syncing = False
        _redraw()

    def _change_mesh(change: dict[str, Any]) -> None:
        nonlocal controller_options
        if change.get("name") != "value":
            return
        mesh_density.disabled = change["new"] == "off"
        mesh = False if change["new"] == "off" else int(mesh_density.value)
        controller_options = replace(controller_options, mesh=mesh)
        _redraw()

    def _change_mesh_density(change: dict[str, Any]) -> None:
        nonlocal controller_options
        if change.get("name") != "value" or not mesh_on.value:
            return
        controller_options = replace(controller_options, mesh=int(change["new"]))
        _redraw()

    def _change_thickness(change: dict[str, Any]) -> None:
        nonlocal thickness_percent
        if change.get("name") != "value":
            return
        thickness_percent = int(change["new"])
        _apply_thickness()

    y_scale_options = [("Linear", "linear")]
    if can_log_y or controller_options.log_y:
        y_scale_options.append(("Log", "log"))
    y_scale_switch = widgets.ToggleButtons(
        options=y_scale_options,
        value="log" if controller_options.log_y else "linear",
        description="",
        style={"button_width": "72px"},
        layout=widgets.Layout(width="148px", min_width="0"),
    )
    y_scale_switch.observe(_change_y_scale, names="value")

    def _exclusive_toggle(
        change: dict[str, Any], selected: Any, other: Any,
        value: str, callback: Any,
    ) -> None:
        """Make two fixed-width ToggleButtons behave like one selector."""
        if change.get("name") != "value":
            return
        if not change["new"]:
            if not other.value:
                selected.value = True  # one choice must remain active
            return
        if other.value:
            other.value = False
        callback({"name": "value", "new": value})

    trace_mode = controller_options.trace_mode(plan)
    line_toggle = widgets.ToggleButton(
        value=trace_mode != "markers", description="Line",
        layout=widgets.Layout(width="72px", min_width="0"),
    )
    points_toggle = widgets.ToggleButton(
        value=trace_mode == "markers", description="Points",
        layout=widgets.Layout(width="72px", min_width="0"),
    )
    trace_mode_switch = widgets.HBox(
        [line_toggle, points_toggle],
        layout=widgets.Layout(width="148px", min_width="0", overflow="hidden"),
    )
    trace_mode_switch.add_class(_RANGE_CONTROLS_COMPACT_CLASS)
    line_toggle.observe(
        lambda change: _exclusive_toggle(
            change, line_toggle, points_toggle, "line", _change_trace_mode
        ), names="value",
    )
    points_toggle.observe(
        lambda change: _exclusive_toggle(
            change, points_toggle, line_toggle, "scatter", _change_trace_mode
        ), names="value",
    )

    # Use the same fixed HBox segment controls as Mode and Mesh.  The
    # ipywidgets ToggleButtons view can wrap its inner buttons under a 3D
    # controller's narrower flex layout, leaving ``Linear`` and ``Log`` on
    # separate lines despite the parent having enough visible width.
    z_linear = widgets.ToggleButton(
        value=True, description="Linear",
        layout=widgets.Layout(width="72px", min_width="0"),
    )
    z_scale_buttons = [z_linear]
    if can_log_z:
        z_log = widgets.ToggleButton(
            value=False, description="Log",
            layout=widgets.Layout(width="72px", min_width="0"),
        )
        z_scale_buttons.append(z_log)
        z_linear.observe(
            lambda change: _exclusive_toggle(
                change, z_linear, z_log, "linear", _change_z_scale
            ), names="value",
        )
        z_log.observe(
            lambda change: _exclusive_toggle(
                change, z_log, z_linear, "log", _change_z_scale
            ), names="value",
        )
    z_scale_switch = widgets.HBox(
        z_scale_buttons,
        layout=widgets.Layout(width="148px", min_width="0", overflow="hidden"),
    )

    initial_mesh = controller_options.mesh
    mesh_enabled = bool(initial_mesh)
    mesh_density = widgets.IntSlider(
        value=(
            int(initial_mesh)
            if isinstance(initial_mesh, int) and not isinstance(initial_mesh, bool)
            else DEFAULT_MESH_LINES
        ),
        min=2,
        max=50,
        step=1,
        description="",
        style={"description_width": "0px"},
        disabled=not mesh_enabled,
        continuous_update=False,
        layout=widgets.Layout(width="calc(100% - 56px)", flex="1 1 0", min_width="0"),
    )
    mesh_on = widgets.ToggleButton(
        value=mesh_enabled, description="On",
        layout=widgets.Layout(width="72px", min_width="0"),
    )
    mesh_off = widgets.ToggleButton(
        value=not mesh_enabled, description="Off",
        layout=widgets.Layout(width="72px", min_width="0"),
    )
    mesh_switch = widgets.HBox(
        [mesh_on, mesh_off],
        layout=widgets.Layout(width="148px", min_width="0", overflow="hidden"),
    )
    mesh_on.observe(
        lambda change: _exclusive_toggle(
            change, mesh_on, mesh_off, "on", _change_mesh
        ), names="value",
    )
    mesh_off.observe(
        lambda change: _exclusive_toggle(
            change, mesh_off, mesh_on, "off", _change_mesh
        ), names="value",
    )
    mesh_density.observe(_change_mesh_density, names="value")
    for compact_control in (
        y_scale_switch,
        z_scale_switch,
        mesh_switch,
        mesh_density,
    ):
        compact_control.add_class(_RANGE_CONTROLS_COMPACT_CLASS)

    thickness_slider = widgets.IntSlider(
        value=100,
        min=50,
        max=250,
        step=10,
        description="",
        readout_format="d",
        continuous_update=False,
        layout=widgets.Layout(flex="1 1 0", min_width="0"),
    )
    thickness_slider.observe(_change_thickness, names="value")
    thickness_slider.add_class(_RANGE_CONTROLS_COMPACT_CLASS)

    compact_button = widgets.Layout(width="100%", min_width="0")
    if has_z:
        auto_axis = widgets.Button(
            description="Fit Z", tooltip="Fit Z to the current X/Y range",
            layout=compact_button,
        )
        auto_axis.on_click(_autoscale_z)
    else:
        auto_axis = widgets.Button(
            description="Fit Y", tooltip="Fit Y to the current X range",
            layout=compact_button, disabled=resamples_y,
        )
        auto_axis.on_click(_autoscale_y)
    reset = widgets.Button(
        description="Reset", tooltip="Restore the original ranges", layout=compact_button
    )
    domain_button = domain_label.upper() if domain_label == "x" else domain_label
    x_in = widgets.Button(
        description=f"[{domain_button}]−", tooltip="Shrink the sampled domain range", layout=compact_button
    )
    x_out = widgets.Button(
        description=f"[{domain_button}]+", tooltip="Expand the sampled domain range", layout=compact_button
    )
    y_in = widgets.Button(
        description="[Y]−", tooltip="Shrink the Y range", layout=compact_button
    )
    y_out = widgets.Button(
        description="[Y]+", tooltip="Expand the Y range", layout=compact_button
    )
    reset.on_click(_reset)
    x_in.on_click(lambda _button: _zoom(x_min, x_max, 0.5))
    x_out.on_click(lambda _button: _zoom(x_min, x_max, 2.0))
    y_in.on_click(lambda _button: _zoom(y_min, y_max, 0.5))
    y_out.on_click(lambda _button: _zoom(y_min, y_max, 2.0))
    if has_z:
        z_in = widgets.Button(
            description="[Z]−", tooltip="Shrink the Z range", layout=compact_button
        )
        z_out = widgets.Button(
            description="[Z]+", tooltip="Expand the Z range", layout=compact_button
        )
        z_in.on_click(lambda _button: _zoom(z_min, z_max, 0.5))
        z_out.on_click(lambda _button: _zoom(z_min, z_max, 2.0))

    range_area = widgets.VBox(
        range_rows, layout=widgets.Layout(width="100%", overflow="hidden")
    )
    # Keep the two visual rows predictable: zoom-out controls follow the
    # fit action, and their matching zoom-in controls follow Reset.
    action_buttons = [auto_axis, x_out, y_out]
    if has_z:
        action_buttons.append(z_out)
    action_buttons.extend([reset, x_in, y_in])
    if has_z:
        action_buttons.append(z_in)
    action_columns = 4 if has_z else 3
    action_button_layout = widgets.Layout(
        width=f"calc({100 / action_columns}% - 4px)",
        flex="0 1 auto",
        min_width="0",
    )
    for button in action_buttons:
        button.layout = action_button_layout
    action_area: Any = widgets.VBox(
        [
            widgets.HBox(
                action_buttons[:action_columns],
                layout=widgets.Layout(
                    width="calc(100% - 6px)", margin="0 3px",
                    justify_content="space-between", overflow="hidden",
                ),
            ),
            widgets.HBox(
                action_buttons[action_columns:],
                layout=widgets.Layout(
                    width="calc(100% - 6px)", margin="0 3px",
                    justify_content="space-between", overflow="hidden",
                ),
            ),
        ],
        layout=widgets.Layout(width="100%", overflow="hidden"),
    )
    action_area.add_class(_RANGE_CONTROLS_COMPACT_CLASS)
    for button in action_buttons:
        button.add_class(_RANGE_CONTROLS_COMPACT_CLASS)

    display_controls: list[Any] = []
    if has_flat_trace_mode:
        # A one-choice scale selector is visual noise. Keep a tiny static
        # value in its place, while reserving the full two-way toggle for
        # plots whose values can actually be shown logarithmically.
        scale_control: Any = y_scale_switch
        if len(y_scale_options) == 1:
            scale_control = widgets.Label(
                "Linear",
                layout=widgets.Layout(width="54px", min_width="0"),
            )
        display_controls.append(
            widgets.VBox(
                [
                    widgets.HBox(
                        [
                            widgets.Label("Scale", layout=widgets.Layout(width="44px")),
                            scale_control,
                        ],
                        layout=widgets.Layout(width="calc(100% - 4px)", align_items="center"),
                    ),
                    widgets.HBox(
                        [
                            widgets.Label("Mode", layout=widgets.Layout(width="44px")),
                            trace_mode_switch,
                        ],
                        layout=widgets.Layout(width="calc(100% - 4px)", align_items="center"),
                    ),
                ],
                layout=widgets.Layout(width="100%"),
            )
        )
    elif has_z:
        display_controls.append(
            widgets.HBox(
                [
                    widgets.Label("Z scale", layout=widgets.Layout(width="50px")),
                    z_scale_switch,
                ],
                layout=widgets.Layout(width="calc(100% - 4px)", align_items="center"),
            )
        )
    surface_controls: list[Any] = []
    if has_surface_mesh:
        surface_controls.extend([
            widgets.HBox(
                [
                    widgets.Label("Mesh", layout=widgets.Layout(width="50px")),
                    mesh_switch,
                ],
                layout=widgets.Layout(width="calc(100% - 4px)", align_items="center"),
            ),
            widgets.HBox(
                [
                    widgets.Label("Density", layout=widgets.Layout(width="56px")),
                    mesh_density,
                ],
                layout=widgets.Layout(width="calc(100% - 4px)", align_items="center"),
            ),
        ])

    thickness_controls: list[Any] = []
    if has_trace_thickness:
        thickness_controls.append(
            widgets.HBox(
                [
                    widgets.Label("Thickness", layout=widgets.Layout(width="70px")),
                    thickness_slider,
                    widgets.Label("%", layout=widgets.Layout(width="12px")),
                ],
                layout=widgets.Layout(
                    width="calc(100% - 4px)", margin="0 2px", align_items="center"
                ),
            )
        )

    # The sidebar's own percentage, plus the floor `min_width` protects
    # once the container is too narrow for a percentage split to stay
    # legible — the same problem §22.4 fixed for the figure/sidebar
    # ratio, applied here to the lower bound instead of the target.
    controls = widgets.VBox(
        [
            controls_style,
            range_area,
            action_area,
            *display_controls,
            *thickness_controls,
            *surface_controls,
        ],
        layout=widgets.Layout(
            width=f"{_RANGE_CONTROLS_SIDEBAR_PERCENT}%",
            flex=f"0 0 {_RANGE_CONTROLS_SIDEBAR_PERCENT}%",
            min_width=_RANGE_CONTROLS_MIN_SIDEBAR_WIDTH,
            margin="0 0 0 12px",
        ),
    )
    # ``HBox`` is an inline-sized widget unless its own width is set.
    # Giving the outer row the notebook's full width is what gives the
    # responsive Plotly figure a real 80%-wide container to measure.
    return widgets.HBox(
        [figure_container, controls],
        layout=widgets.Layout(
            width="100%",
            display="flex",
            flex_flow="row nowrap",
            align_items="stretch",
        ),
    )
