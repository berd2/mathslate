"""The object every ``plot()`` returns.

Escape hatch completeness is a success metric (PRD 4): every result exposes
``.sympy``, ``.plotly`` and ``.numpy``, so peeling the wrapper off costs the
user nothing.
"""

from __future__ import annotations

import base64
import importlib.util
import math
import sys
import warnings
from collections.abc import Iterable
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
from .render.options import DEFAULT_MESH_LINES, RenderOptions
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


def _restore_arrays(value: Any) -> Any:
    """Undo Plotly's typed-array encoding, in place of nothing else.

    ``{"dtype": "f8", "bdata": <base64>}`` — plus ``"shape"`` once the array
    has more than one dimension — is plotly.js's own wire format, which is
    why it is safe to key on: it is the shape ``to_dict()`` is contracted to
    emit, not an implementation detail of one version. The short codes are
    NumPy's own (``f8``, ``i4``, …), so they need no translation table.

    Anything else is walked and returned unchanged, so a figure with no
    numeric arrays survives untouched.
    """
    if isinstance(value, dict):
        if "bdata" in value and "dtype" in value:
            # `frombuffer` views the decoded bytes read-only; copy so the
            # result is as mutable as the array `plot()` puts there itself.
            array = np.frombuffer(
                base64.b64decode(value["bdata"]), dtype=str(value["dtype"])
            ).copy()
            shape = value.get("shape")
            if shape:
                array = array.reshape(
                    [int(size) for size in str(shape).split(",") if size.strip()]
                )
            return array
        return {key: _restore_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_arrays(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_arrays(item) for item in value)
    return value


def _rebuild_plot_result(
    figure_spec: dict[str, Any],
    plan: PlotPlan,
    options: RenderOptions,
    frames: tuple[list[PlotPlan], tuple[float, ...], str] | None,
    frame_play: bool,
) -> "PlotResult":
    """Reconstruct a pickled :class:`PlotResult`. See its ``__reduce__``."""
    return PlotResult(plan, go.Figure(figure_spec), options, frames, frame_play)


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

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Pickle the figure with its arrays intact (PRD 4, §25.1).

        Plotly's own ``__reduce__`` goes through ``to_dict()``, which encodes
        every numeric array into the ``{"dtype", "bdata"}`` typed-array spec
        that plotly.js reads off the wire. A figure rebuilt from that renders
        perfectly and is *not* the object the escape-hatch promise describes:
        ``result.plotly.data[0].x`` comes back a dict, so ``np.asarray`` on it
        raises where it worked before. Nothing pickled a result until
        restricted AI execution moved to a separate process (§25); this makes
        that boundary lossless rather than merely functional.

        The specs are decoded back to arrays rather than the figure being
        rebuilt from ``plan``/``options``, because ``plotly`` is documented as
        "yours to mutate" — regenerating it would silently discard whatever
        the caller did to the figure in order to fix how it travels.
        """
        return (
            _rebuild_plot_result,
            (
                _restore_arrays(self._figure.to_dict()),
                self._plan,
                self._options,
                self._frames,
                self._frame_play,
            ),
        )

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
        auto_y: bool = False,
        auto_z: bool = False,
        render_options: RenderOptions | None = None,
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

        ``auto_y``/``auto_z`` *clear* that view override rather than replacing
        it, which passing ``None`` cannot do: ``None`` means "leave this axis
        as it was", and the two are different requests. Auto Y on a plot built
        with an explicit ``ylim`` would otherwise fit itself to the very window
        it is being asked to escape.
        """
        plan = self._plan
        base_options = render_options or self._options
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
            if auto_z:
                options = replace(base_options, zlim=None)
            elif z_range is not None:
                options = replace(base_options, zlim=z_range)
            else:
                options = base_options
        elif plan.symbol is not None:
            x_lo, x_hi = x_range if x_range is not None else plan.param_range
            ranges = ((plan.symbol, x_lo, x_hi),)
            if auto_y:
                ylim = None
            elif y_range is not None:
                ylim = y_range
            else:
                ylim = base_options.ylim
            if auto_z:
                zlim = None
            elif plan.kind == "space" and z_range is not None:
                zlim = z_range
            else:
                zlim = base_options.zlim
            options = replace(base_options, ylim=ylim, zlim=zlim)
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
        plan = self._plan
        if not self._resamplable():
            # A property of the plot itself, so it applies regardless of host —
            # checked before the environment questions below can obscure it.
            raise UnsupportedInputError(
                f"a {plan.kind} plot has no symbolic domain; range_controls() "
                "needs a plotted expression, not raw data."
            )
        if self.interactive:
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
        # Every y number below — the seeds, what the boxes push back as
        # ``ylim``, and what Auto Y fits — is in the axis's own units, and on
        # a log axis Plotly's are powers of ten.
        log_y = self._options.log_y
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
            window = self._options.y_range(plan) or _fit_window(
                (series.sample.y for series in plan.series), log=log_y
            )
            y_lo, y_hi = window if window is not None else (-1.0, 1.0)
        has_z = self._has_z_axis()
        if has_z:
            # z_range is the *active auto-clip*, unset on a well-behaved
            # surface with nothing to clip — the box still needs a starting
            # number, so it falls back to the surface's own data range
            # rather than a value with no relationship to what is on screen.
            clipped = self._options.z_range(plan) or _data_span(
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
        domain_label = plan.symbol.name if plan.kind == "space" and plan.symbol else "x"
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
        controller_options = self._options
        z_scale = "linear"
        thickness_percent = 100
        has_flat_trace_mode = not resamples_y and not has_z
        has_surface_mesh = plan.kind in {"surface", "psurface"}
        can_log_y = has_flat_trace_mode and _can_use_log_scale(
            series.sample.y for series in plan.series
        )
        can_log_z = has_z and _can_use_log_scale(
            [plan.series[0].sample.z]
        )

        syncing = False

        def _apply_thickness() -> None:
            """Scale the standard 2D line and marker sizes by one percentage."""
            if not has_flat_trace_mode:
                return
            factor = thickness_percent / 100
            with figure_widget.batch_update():
                for trace in figure_widget.data:
                    trace.line.width = 2 * factor
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
                fresh = self._replotted(
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
                fresh = self._replotted(
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
                fresh = self._replotted(
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
        if has_flat_trace_mode:
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
            and not self.interactive
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
            # not yet installed nbformat, which Plotly's MIME renderer checks.
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
