"""The object every ``plot()`` returns.

Escape hatch completeness is a success metric (PRD 4): every result exposes
``.sympy``, ``.plotly`` and ``.numpy``, so peeling the wrapper off costs the
user nothing.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import sympy as sp

from . import codegen
from ._text import safe_print
from .core.analysis import Analysis, analyze_expression
from .core.dispatch import PlotPlan, build_plan
from .core.tables import DEFAULT_ROWS, Table, tabulate
from .errors import MathSlateError, UnsupportedInputError
from .render import plotly_backend
from .render.options import RenderOptions
from .ui.adapters import Frontend, detect_frontend

__all__ = ["PlotResult", "set_range_controls", "get_range_controls"]

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
_RANGE_CONTROLS_MIN_SIDEBAR_WIDTH: str = "180px"
#: ``FigureWidget.layout`` is Plotly's figure layout, not the ipywidgets CSS
#: layout trait.  The class lets the small stylesheet in ``range_controls``
#: size the widget root itself when the plot has no explicit pixel width.
_RANGE_CONTROLS_FIGURE_CLASS: str = "mathslate-range-controls-figure"
_RANGE_CONTROLS_INPUT_CLASS: str = "mathslate-range-controls-input"


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


class PlotResult:
    """A drawn plot, plus every door back out to the underlying libraries."""

    def __init__(
        self,
        plan: PlotPlan,
        figure: go.Figure,
        options: RenderOptions | None = None,
        frames: tuple[list[PlotPlan], tuple[float, ...], str] | None = None,
        frame_play: bool = False,
    ) -> None:
        self._plan: PlotPlan = plan
        self._figure: go.Figure = figure
        #: The drawing options, so ``python()`` emits the figure you are seeing.
        self._options: RenderOptions = options or RenderOptions()
        #: ``(one plan per slider position, the positions, the label)`` when a
        #: slider is driving this figure. ``show_python()`` needs all three, or
        #: it would emit a still picture for something that moves.
        self._frames = frames
        #: ``animate()`` has play/pause controls; slider-driven ``plot()`` does not.
        self._frame_play = bool(frame_play)

    @property
    def frames(self) -> tuple[list[PlotPlan], tuple[float, ...], str] | None:
        """The per-position plans behind an interactive figure, if any."""
        return self._frames

    @property
    def interactive(self) -> bool:
        """Whether a slider is driving this figure."""
        return self._frames is not None

    # -- escape hatches ----------------------------------------------------

    @property
    def plotly(self) -> go.Figure:
        """The Plotly figure. Mutate it freely; MathSlate does not own it."""
        return self._figure

    #: Alias for people who think in Plotly rather than in MathSlate.
    @property
    def figure(self) -> go.Figure:
        return self._figure

    @property
    def sympy(self) -> sp.Expr | tuple[sp.Expr, ...] | None:
        """The expression(s) behind the plot, or ``None`` for raw data."""
        if not self._plan.exprs:
            return None
        if len(self._plan.exprs) == 1:
            return self._plan.exprs[0]
        return self._plan.exprs

    @property
    def numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """The sampled ``(x, y)`` arrays of the first series."""
        sample = self._plan.series[0].sample
        return sample.x, sample.y

    @property
    def plan(self) -> PlotPlan:
        """The resolved plan — kind, ranges, breakpoints, notes."""
        return self._plan

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(self._plan.notes)

    # -- progressive unboxing ---------------------------------------------

    def python(self) -> str:
        """Return the equivalent plain-Python source as a string."""
        if self._frames is not None:
            plans, steps, label = self._frames
            return codegen.generate_frames_code(
                plans, steps, label, self._options, play=self._frame_play
            )
        return codegen.generate_code(self._plan, self._options)

    def show_python(self) -> str:
        """Print the equivalent plain Python (PRD 5.5) and return it."""
        code = self.python()
        safe_print(code)
        return code

    def summary(self) -> str:
        """The one-line inference report."""
        return plotly_backend.describe_plan(self._plan)

    # -- properties of the object (PRD 5.6) --------------------------------

    def analyze(self) -> Analysis:
        """Report the properties of the curve on screen (PRD 5.6).

        The window analysed is the one you are looking at, not an arbitrary
        default: the roots of ``sin(x)`` depend entirely on where you looked.

        Only a single ``y = f(x)`` has these properties to report. Overlaid
        curves have them one at a time, and a parametric curve, a callable or
        raw data has no expression to solve at all — each says so rather than
        guessing which one you meant.
        """
        if self._plan.kind in {"parametric", "polar"}:
            raise UnsupportedInputError(
                f"a {self._plan.kind} curve has no single y = f(x) to analyse. "
                "Its components are available as .sympy if you want to analyse "
                "one of them: analyze(f.sympy[0])."
            )
        if not self._plan.exprs:
            raise UnsupportedInputError(
                f"a {self._plan.kind} plot has no expression to analyse — "
                "analyze() solves symbolically, and there is nothing here to solve."
            )
        if len(self._plan.exprs) > 1:
            raise UnsupportedInputError(
                f"this plot draws {len(self._plan.exprs)} curves; analyze() describes "
                "one at a time. Pick one, e.g. analyze(f.sympy[0])."
            )
        # Ask what analyse() actually needs — one real-valued expression — rather
        # than naming the kinds that happen not to have one. A `region` stores the
        # inequality itself in `exprs`, so a kind-by-kind guard let it through and
        # SymPy raised a raw TypeError from inside `limit()`. Asking the question
        # directly means the next kind is covered without anyone remembering.
        subject = self._plan.exprs[0]
        if not isinstance(subject, sp.Expr):
            raise UnsupportedInputError(
                f"a {self._plan.kind} plot draws where {sp.sstr(subject)} is true, "
                "which is a set rather than a function — there is no y = f(x) whose "
                "roots or extrema analyze() could report. Analyse a side of it "
                "instead, e.g. analyze(lhs - rhs)."
            )
        assert self._plan.symbol is not None and self._plan.param_range is not None
        return analyze_expression(
            self._plan.exprs[0], self._plan.symbol, self._plan.param_range
        )

    def table(self, rows: int = DEFAULT_ROWS) -> Table:
        """The values behind the curve, over the window you are looking at."""
        if self._plan.two_variable or self._plan.kind in {"space", "parametric", "polar"}:
            raise UnsupportedInputError(
                f"a {self._plan.kind} plot has no single column of values; "
                "table() tabulates y = f(x)."
            )
        if not self._plan.exprs:
            raise UnsupportedInputError(
                f"a {self._plan.kind} plot has no expression to evaluate. Its "
                "numbers are already yours as .numpy."
            )
        assert self._plan.symbol is not None and self._plan.param_range is not None
        return tabulate(
            list(self._plan.exprs), self._plan.symbol, self._plan.param_range, rows=rows
        )

    # -- live range controls (PRD §19) --------------------------------------

    def _replotted(
        self,
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
    ) -> "PlotResult":
        """A fresh result over a new domain — a resample, not a view-crop.

        ``xlim``/``ylim`` (PRD §18) only decide what the axes show of the
        points already computed; zooming into a tenth of that window still
        shows a tenth of the original point density. This rebuilds the sample
        at full resolution over the window actually asked for, which is what
        :meth:`range_controls` needs to be more than a wrapper around Plotly's
        own drag-to-zoom.

        Two-variable kinds (a surface, a contour, a region) have both axes as
        real domains, so both resample. Everywhere else Y is derived from X,
        not sampled independently, so ``y_range`` becomes a ``ylim`` view
        override instead — cheap, and correct: there is no "y resolution" to
        recover by resampling a value nothing there computed pointwise. A
        surface's Z is the same story one level up (PRD §23): it is the
        function's *output*, computed from the X/Y grid rather than sampled
        along its own axis, so ``z_range`` is always a ``zlim`` view override
        and never triggers a resample by itself.
        """
        plan = self._plan
        for axis, bounds in (("x", x_range), ("y", y_range), ("z", z_range)):
            if bounds is not None and not all(math.isfinite(value) for value in bounds):
                raise UnsupportedInputError(
                    f"{axis} range must contain finite numbers, not {bounds!r}."
                )
        if not plan.exprs:
            raise UnsupportedInputError(
                f"a {plan.kind} plot has no expression to resample; its range "
                "cannot be changed live."
            )
        # Keep the tuple form for a space curve: ``build_plan`` uses that
        # shape (three expressions plus one parameter range) to distinguish a
        # single 3D curve from several ordinary 2D curves.
        obj: object = plan.exprs[0] if len(plan.exprs) == 1 else tuple(plan.exprs)

        if plan.axes is not None:
            x_symbol, y_symbol = plan.axes
            x_lo, x_hi = x_range if x_range is not None else plan.param_range
            y_lo, y_hi = y_range if y_range is not None else plan.second_range
            ranges: tuple[object, ...] = ((x_symbol, x_lo, x_hi), (y_symbol, y_lo, y_hi))
            options = replace(self._options, zlim=z_range) if z_range is not None else self._options
        elif plan.symbol is not None:
            x_lo, x_hi = x_range if x_range is not None else plan.param_range
            ranges = ((plan.symbol, x_lo, x_hi),)
            options = replace(
                self._options,
                ylim=y_range if y_range is not None else self._options.ylim,
                zlim=(
                    z_range
                    if plan.kind == "space" and z_range is not None
                    else self._options.zlim
                ),
            )
        else:
            raise UnsupportedInputError(
                f"a {plan.kind} plot has no symbolic domain to resample; range "
                "controls only apply to a plotted expression."
            )

        # ``space`` is an inferred internal kind, not a public ``kind=``
        # value accepted by ``build_plan``. Re-infer it from the three
        # components and their one shared parameter when resampling.
        kind = None if plan.kind == "space" else plan.kind
        new_plan = build_plan(obj, *ranges, polar=plan.polar, kind=kind, config=plan.config)
        figure = plotly_backend.figure_from_plan(new_plan, options)
        return PlotResult(new_plan, figure, options)

    def _resamplable(self) -> bool:
        """Whether this plot has a symbolic domain :meth:`range_controls` can move."""
        plan = self._plan
        return plan.symbol is not None or plan.axes is not None

    def _has_z_axis(self) -> bool:
        """Whether this plot has a reader-adjustable Z view (surface or space curve).

        Not ``sample.z_range is not None`` — that field is the *active
        auto-clip*, ``None`` precisely on a well-behaved surface with nothing
        to clip, which is the opposite of what "has a Z axis" should mean.
        ``contour``/``implicit``/``region`` are two-variable too but flat
        (``go.Heatmap``/``go.Contour``, no Z), so the kind itself — the same
        check ``plotly_backend.py`` and ``codegen.py`` already make to choose
        ``go.Surface`` — is what's actually being asked here.
        """
        return self._plan.kind in {"surface", "psurface", "space"}

    def range_controls(
        self, *, width: int | None = None, height: int | None = None
    ) -> Any:
        """Live X/Y(/Z) number boxes that resample this plot (PRD §19, §22, §23).

        This is the missing half of PRD §18's ``xlim``/``ylim``: those decide
        the *initial* window, and Plotly's own drag-to-zoom only ever crops
        what was already sampled there. These controls instead call
        :meth:`_replotted` on every change, so narrowing X to look closely at
        one feature draws it at full resolution rather than showing fewer of
        the original points — the Excel-style "edit the axis, get a redrawn
        chart" behaviour, not a screenshot crop. They sit beside the figure,
        not above it, so widening the window is not also narrowing the plot.
        A true 3D surface (§23.2) gets two more boxes, ``z min``/``z max`` —
        Z is the function's *output*, not a domain, so moving them only ever
        re-applies ``zlim``, never a resample.

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
        plan = self._plan
        if not self._resamplable():
            # A property of the plot itself, so it applies regardless of host —
            # checked before the environment questions below can obscure it.
            raise UnsupportedInputError(
                f"a {plan.kind} plot has no symbolic domain; range_controls() "
                "needs a plotted expression, not raw data."
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
                return self._figure
            raise UnsupportedInputError(
                "ipywidgets is not installed. `pip install mathslate[jupyter]`."
            ) from error
        try:
            # Plotly >= 6 rebuilt FigureWidget on `anywidget`, which
            # `mathslate[jupyter]` installs alongside ipywidgets (PRD §21) —
            # this only fires for an environment that has ipywidgets some
            # other way, without it.
            figure_widget = go.FigureWidget(self._figure)
        except ImportError as error:
            if sys.platform == "emscripten":
                warnings.warn(
                    "range_controls() is unavailable because this JupyterLite "
                    "kernel has no anywidget; showing the regular Plotly graph.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._figure
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
        x_lo, x_hi = plan.param_range or (0.0, 1.0)
        y_lo, y_hi = plan.second_range if resamples_y else (plan.y_range or (-1.0, 1.0))
        if plan.kind == "space":
            # A space curve's ``y`` is a plotted coordinate, not a second
            # domain.  Unlike a 2D curve, PlotPlan has no derived ``y_range``
            # for it, so the generic fallback of -1..1 would immediately crop
            # curves such as a trefoil when Auto Z redraws the figure.
            y_data = plan.series[0].sample.y
            y_lo, y_hi = float(np.nanmin(y_data)), float(np.nanmax(y_data))
        has_z = self._has_z_axis()
        if has_z:
            # z_range is the *active auto-clip*, unset on a well-behaved
            # surface with nothing to clip — the box still needs a starting
            # number, so it falls back to the surface's own data range
            # rather than a value with no relationship to what is on screen.
            clipped = self._options.z_range(plan)
            if clipped is not None:
                z_lo, z_hi = clipped
            else:
                z_data = plan.series[0].sample.z
                z_lo, z_hi = float(np.nanmin(z_data)), float(np.nanmax(z_data))
        else:
            z_lo, z_hi = (0.0, 0.0)

        # A narrow overall box (§22.1) squeezed the label and the editable
        # number into the same few pixels, and the number lost; shrinking the
        # label's reserved width instead leaves the number room to be read.
        # The box itself fills the sidebar (`width="100%"`) rather than a
        # fixed pixel count, so it tracks whatever width the percentage
        # sidebar below actually renders at.
        box_style = {"description_width": "45px"}
        # Leave a few pixels of breathing room inside the sidebar.  Some
        # ipywidgets themes include a border in a 100%-wide FloatText, making
        # the final field spill horizontally and create a page scrollbar.
        box_layout = widgets.Layout(width="calc(100% - 8px)", min_width="0")
        domain_label = plan.symbol.name if plan.kind == "space" and plan.symbol else "x"
        x_min = widgets.FloatText(
            value=x_lo, description=f"{domain_label} min", step=None, style=box_style, layout=box_layout
        )
        x_max = widgets.FloatText(
            value=x_hi, description=f"{domain_label} max", step=None, style=box_style, layout=box_layout
        )
        y_min = widgets.FloatText(
            value=y_lo, description="y min", step=None, style=box_style, layout=box_layout
        )
        y_max = widgets.FloatText(
            value=y_hi, description="y max", step=None, style=box_style, layout=box_layout
        )
        boxes = [x_min, x_max, y_min, y_max]
        if has_z:
            z_min = widgets.FloatText(
                value=z_lo, description="z min", step=None, style=box_style, layout=box_layout
            )
            z_max = widgets.FloatText(
                value=z_hi, description="z max", step=None, style=box_style, layout=box_layout
            )
            boxes += [z_min, z_max]

        for box in boxes:
            box.add_class(_RANGE_CONTROLS_INPUT_CLASS)

        controls_style = widgets.HTML(
            value=(
                "<style>"
                f".{_RANGE_CONTROLS_INPUT_CLASS} {{ "
                "box-sizing: border-box; max-width: 100% !important; "
                "min-width: 0 !important; }}"
                f".{_RANGE_CONTROLS_INPUT_CLASS} input {{ "
                "box-sizing: border-box; max-width: 100% !important; "
                "min-width: 0 !important; }}"
                "</style>"
            ),
            layout=widgets.Layout(display="none"),
        )

        syncing = False

        def _apply(fresh: PlotResult) -> None:
            with figure_widget.batch_update():
                figure_widget.data = []
                figure_widget.add_traces(fresh.plotly.data)
                figure_widget.layout.update(fresh.plotly.layout.to_plotly_json())

        def _redraw(_change: dict[str, Any] | None = None) -> None:
            if syncing:
                return
            if not all(math.isfinite(box.value) for box in boxes):
                return  # wait for a finite value instead of leaking a callback traceback
            if x_max.value <= x_min.value or y_max.value <= y_min.value:
                return  # a mid-edit, inconsistent pair — wait for both to settle
            if has_z and z_max.value <= z_min.value:
                return
            try:
                fresh = self._replotted(
                    x_range=(x_min.value, x_max.value),
                    y_range=(y_min.value, y_max.value),
                    z_range=(z_min.value, z_max.value) if has_z else None,
                )
            except (MathSlateError, ValueError, OverflowError):
                return  # e.g. a range too narrow for this expression to sample
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
            """Fit a curve's Y view to the values in its current X domain."""
            if resamples_y:
                return  # on surfaces Y is an input domain, not a view axis
            try:
                fresh = self._replotted(
                    x_range=(x_min.value, x_max.value),
                    y_range=None,
                    z_range=(z_min.value, z_max.value) if has_z else None,
                )
            except (MathSlateError, ValueError, OverflowError):
                return
            fitted = fresh.plan.y_range
            if fitted is None:
                return
            nonlocal syncing
            syncing = True
            try:
                y_min.value, y_max.value = fitted
            finally:
                syncing = False
            _apply(fresh)

        def _autoscale_z(_button: Any) -> None:
            """Fit a surface's Z view to its current X/Y domain."""
            try:
                fresh = self._replotted(
                    x_range=(x_min.value, x_max.value),
                    y_range=(y_min.value, y_max.value),
                    z_range=None,
                )
            except (MathSlateError, ValueError, OverflowError):
                return
            fitted = fresh._options.z_range(fresh.plan)
            if fitted is None:
                z_data = fresh.plan.series[0].sample.z
                fitted = (float(np.nanmin(z_data)), float(np.nanmax(z_data)))
            nonlocal syncing
            syncing = True
            try:
                z_min.value, z_max.value = fitted
            finally:
                syncing = False
            _apply(fresh)
            # An automatic surface has no ``scene.zaxis.range`` entry in its
            # fresh layout. ``Layout.update`` preserves an old manual entry
            # in that case, so replace it explicitly when Auto Z clears one.
            figure_widget.layout.scene.zaxis.range = fitted

        def _reset(_button: Any) -> None:
            pairs: list[tuple[Any, Any, float, float]] = [
                (x_min, x_max, x_lo, x_hi),
                (y_min, y_max, y_lo, y_hi),
            ]
            if has_z:
                pairs.append((z_min, z_max, z_lo, z_hi))
            _set_pairs(*pairs)

        compact_button = widgets.Layout(width="50%", min_width="0")
        if has_z:
            auto_axis = widgets.Button(
                description="Auto Z", tooltip="Fit Z to the current X/Y range",
                layout=compact_button,
            )
            auto_axis.on_click(_autoscale_z)
        else:
            auto_axis = widgets.Button(
                description="Auto Y", tooltip="Fit Y to the current X range",
                layout=compact_button, disabled=resamples_y,
            )
            auto_axis.on_click(_autoscale_y)
        reset = widgets.Button(
            description="Reset", tooltip="Restore the original ranges", layout=compact_button
        )
        domain_button = domain_label.upper() if domain_label == "x" else domain_label
        x_in = widgets.Button(
            description=f"{domain_button} in", tooltip="Zoom in on the sampled domain", layout=compact_button
        )
        x_out = widgets.Button(
            description=f"{domain_button} out", tooltip="Zoom out on the sampled domain", layout=compact_button
        )
        y_in = widgets.Button(
            description="Y in", tooltip="Zoom in on the Y range", layout=compact_button
        )
        y_out = widgets.Button(
            description="Y out", tooltip="Zoom out on the Y range", layout=compact_button
        )
        reset.on_click(_reset)
        x_in.on_click(lambda _button: _zoom(x_min, x_max, 0.5))
        x_out.on_click(lambda _button: _zoom(x_min, x_max, 2.0))
        y_in.on_click(lambda _button: _zoom(y_min, y_max, 0.5))
        y_out.on_click(lambda _button: _zoom(y_min, y_max, 2.0))
        actions = [
            widgets.HBox([auto_axis, reset], layout=widgets.Layout(width="100%")),
            widgets.HBox([x_in, x_out], layout=widgets.Layout(width="100%")),
            widgets.HBox([y_in, y_out], layout=widgets.Layout(width="100%")),
        ]
        if has_z:
            z_in = widgets.Button(
                description="Z in", tooltip="Zoom in on the Z range", layout=compact_button
            )
            z_out = widgets.Button(
                description="Z out", tooltip="Zoom out on the Z range", layout=compact_button
            )
            z_in.on_click(lambda _button: _zoom(z_min, z_max, 0.5))
            z_out.on_click(lambda _button: _zoom(z_min, z_max, 2.0))
            actions.append(
                widgets.HBox([z_in, z_out], layout=widgets.Layout(width="100%"))
            )

        # The sidebar's own percentage, plus the floor `min_width` protects
        # once the container is too narrow for a percentage split to stay
        # legible — the same problem §22.4 fixed for the figure/sidebar
        # ratio, applied here to the lower bound instead of the target.
        controls = widgets.VBox(
            [controls_style, *boxes, *actions],
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

    # -- display and export ------------------------------------------------

    def show(self, *args: Any, **kwargs: Any) -> None:
        self._figure.show(*args, **kwargs)

    def to_html(self, path: str | Path | None = None, *, standalone: bool = True) -> str:
        """Write the figure to one self-contained HTML file (PRD 3, P3).

        ``standalone`` embeds Plotly itself, so the file opens with no network
        and no install — which is the whole point for a teacher handing it to a
        class, and why a slider is drawn from frames carried inside the figure
        (§11) rather than from a widget that needs a live kernel.

        Returns the HTML either way; ``path`` also writes it.
        """
        html = self._figure.to_html(
            include_plotlyjs=True if standalone else "cdn",
            full_html=True,
            auto_play=False,
        )
        if path is not None:
            Path(path).write_text(html, encoding="utf-8")
        return html

    def _wants_live_range_controls(self) -> bool:
        """Whether a bare ``plot(...)`` should show :meth:`range_controls` (§19.4)."""
        return (
            _RANGE_CONTROLS_DEFAULT
            and self._resamplable()
            and detect_frontend() in (Frontend.JUPYTER, Frontend.COLAB)
            and _live_range_deps_available()
        )

    def _ipython_display_(self) -> None:
        """What evaluating this at a cell's end actually shows.

        IPython calls this instead of the ``_repr_*_`` methods below when it
        is defined, which is what makes the choice exclusive rather than a
        second thing merged into the same display. In Jupyter or Colab, with
        ``ipywidgets``/``anywidget`` installed and a plot that has a domain to
        move, that choice is :meth:`range_controls` (PRD §19.4) — the point of
        a live X/Y control nobody has to call by name is that it is what
        appears without being asked for. Everywhere else — marimo (whose own
        reactivity already gets the same thing for free, see §19.2), a plain
        script, missing deps, or raw data with nothing to resample — this
        shows exactly what plot() has always shown: the plain figure.
        """
        from IPython.display import HTML, display

        if self._wants_live_range_controls():
            display(self.range_controls())
            return
        try:
            display(self._figure)
        except ValueError as error:
            # A partially initialized Pyodide kernel can render HTML but has
            # not yet installed nbformat, which Plotly's MIME renderer checks
            # before producing output. Give the learner the normal graph and
            # a warning instead of exposing that implementation traceback.
            if sys.platform != "emscripten" or "nbformat" not in str(error):
                raise
            warnings.warn(
                "JupyterLite has not loaded nbformat yet; showing a regular "
                "Plotly graph without range controls.",
                RuntimeWarning,
                stacklevel=2,
            )
            display(HTML(self._figure.to_html(full_html=False, include_plotlyjs="cdn")))

    # A result must display wherever the figure it wraps would, so it offers
    # every hook the figure offers rather than picking one.
    #
    # Picking one was the bug: only `_repr_mimebundle_` was delegated, and
    # Plotly returns `{}` from it unless a Jupyter kernel has activated the
    # mimetype renderer. In Jupyter that is true and the plot appeared; in
    # marimo it is not, so `plot(tan(x))` printed its report and drew nothing.
    # `_repr_html_` is what actually renders outside a kernel, and it is the
    # hook marimo prefers — `_ipython_display_` above pre-empts both of these
    # in an IPython kernel, but marimo never looks for it, so these two are
    # still what marimo (and any other `_repr_html_`/`_repr_mimebundle_`-only
    # host) renders through.

    def _repr_html_(self) -> str:
        return self._figure._repr_html_()

    def _repr_mimebundle_(self, *args: Any, **kwargs: Any) -> Any:
        return self._figure._repr_mimebundle_(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<PlotResult {self.summary()}>"


def _live_range_deps_available() -> bool:
    """Whether :meth:`PlotResult.range_controls` could build its widget now.

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
