"""The object every ``plot()`` returns.

Escape hatch completeness is a success metric (PRD 4): every result exposes
``.sympy``, ``.plotly`` and ``.numpy``, so peeling the wrapper off costs the
user nothing.
"""

from __future__ import annotations

import base64
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
from .core.dispatch import REQUESTABLE_KINDS, PlotPlan, build_plan
from .core.tables import DEFAULT_ROWS, Table, tabulate
from .errors import UnsupportedInputError
from .render import plotly_backend
from .render.options import RenderOptions
from .ui import range_controls as range_controls_widget
from .ui.adapters import Frontend, detect_frontend

# The two knobs live with the widget they turn on and off, and are re-exported
# here because `plot()`'s result is where a reader looks for them and because
# `mathslate.api` has imported them from this module since §21 named them.
from .ui.range_controls import get_range_controls, set_range_controls

__all__ = ["PlotResult", "set_range_controls", "get_range_controls"]


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
    controls: bool | None = None,
) -> "PlotResult":
    """Reconstruct a pickled :class:`PlotResult`. See its ``__reduce__``.

    ``controls`` is last and defaults, so a result pickled by an older version
    — the AI subprocess boundary sends these across (§25) — still unpickles.
    """
    return PlotResult(
        plan, go.Figure(figure_spec), options, frames, frame_play, controls
    )



class PlotResult:
    """A drawn plot, plus every door back out to the underlying libraries."""

    def __init__(
        self,
        plan: PlotPlan,
        figure: go.Figure,
        options: RenderOptions | None = None,
        frames: tuple[list[PlotPlan], tuple[float, ...], str] | None = None,
        frame_play: bool = False,
        controls: bool | None = None,
    ) -> None:
        self._plan: PlotPlan = plan
        self._figure: go.Figure = figure
        #: The drawing options, so ``python()`` emits the figure you are seeing.
        self._options: RenderOptions = options or RenderOptions()
        #: Whether this plot shows the range-control sidebar, or ``None`` to
        #: follow ``set_range_controls()``. Not a ``RenderOptions`` field: that
        #: module holds what *both renderers* read, and neither renderer draws
        #: a sidebar — this decides what a notebook cell displays, and has no
        #: bearing on the figure or on the program ``show_python()`` emits.
        self._controls: bool | None = controls
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
                self._controls,
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

        The result is a still picture, never a slider-driven one — which is why
        a slider-driven plot is refused here rather than resampled. Carrying
        ``_frames`` across would be worse than dropping them: the fresh figure
        is built by ``figure_from_plan`` and holds no frames at all, so the
        result would call itself :attr:`interactive`, draw one still curve, and
        have ``show_python()`` emit ``generate_frames_code`` for the *old*
        domain's plans. Resampling every frame instead is the alternative, and
        it is the one :meth:`range_controls` already declines: it would rebuild
        the whole sequence on every keystroke.
        """
        plan = self._plan
        base_options = render_options or self._options
        if self.interactive:
            raise UnsupportedInputError(
                "a slider-driven plot cannot be resampled one window at a "
                "time: its positions are pre-rendered frames, and a fresh "
                "sample would leave the figure and those frames describing "
                "different domains. Use xlim/ylim for the window, or rebuild "
                "with animate(..., (x, lo, hi))."
            )
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
        # What to hand back to ``build_plan``. A relation-driven plan says so
        # itself, because its ``exprs`` hold the difference rather than the
        # relation and would rebuild as a plain curve or surface. Otherwise the
        # expressions are the object — kept as a tuple where there are several,
        # since that shape is what distinguishes one parametric curve from
        # several ordinary 2D ones.
        obj: object
        if plan.relation is not None:
            obj = plan.relation
        else:
            obj = plan.exprs[0] if len(plan.exprs) == 1 else tuple(plan.exprs)

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

        # Most kinds are *inferred* from the object's shape and are not values
        # ``build_plan`` accepts by name — ``space`` from three components over
        # one parameter, ``parametric`` from two, ``polar`` from the flag,
        # ``implicit``/``band``/``region`` from the relation. Handing such a
        # kind back raises ``unknown kind=...``, which the sidebar cannot tell
        # from a mid-edit value and so swallows: the controls simply stop
        # working. Pass through only what can be asked for — which is also the
        # only place the distinction matters, ``contour`` being a genuine
        # choice that re-inference would lose — and let the rest be re-derived.
        kind = plan.kind if plan.kind in REQUESTABLE_KINDS else None
        new_plan = build_plan(obj, *ranges, polar=plan.polar, kind=kind, config=plan.config)
        figure = plotly_backend.figure_from_plan(new_plan, options)
        return PlotResult(new_plan, figure, options, controls=self._controls)

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
        the original points.

        Needs a running Jupyter or Colab kernel with ``ipywidgets`` installed.
        See :func:`mathslate.ui.range_controls.build`, which this delegates to,
        for what the sidebar offers each kind of plot and why.

        This is what a bare ``plot(...)`` shows by default at a cell's end
        whenever that is possible — see :meth:`_ipython_display_`. Call it
        directly only to keep the widget instead of the report, or to build one
        from a result that was printed with ``verbose=False``.
        """
        return range_controls_widget.build(self, width=width, height=height)

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
        wanted = (
            range_controls_widget.get_range_controls()
            if self._controls is None
            else self._controls
        )
        return (
            wanted
            and self._resamplable()
            and not self.interactive
            and detect_frontend() in (Frontend.JUPYTER, Frontend.COLAB)
            and range_controls_widget.deps_available()
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


