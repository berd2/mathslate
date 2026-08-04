"""Figure-level options, decided once and consumed by both renderers.

There are two renderers for every plot: :mod:`mathslate.render.plotly_backend`,
which builds the figure you see, and :mod:`mathslate.codegen`, which writes the
plain-Python program that builds the same figure. ``show_python()`` promises
those two agree.

Every option that only one of them reads is a broken promise waiting to happen,
so the decisions live here — in a module that imports no plotting library — and
both sides call the same methods.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.dispatch import PlotPlan
from ..errors import UnsupportedInputError

__all__ = [
    "RenderOptions",
    "REPRODUCED",
    "NOT_REPRODUCED",
    "MESH_LINE",
    "REGION_FILL",
    "DEFAULT_MESH_LINES",
    "DEFAULT_HEIGHT",
    "set_plot_size",
    "get_plot_size",
]

#: Figure properties ``show_python()`` guarantees to reproduce.
REPRODUCED: tuple[str, ...] = (
    "trace x/y data",
    "trace mode",
    "connectgaps",
    "trace name",
    "template",
    "title",
    "showlegend",
    "xaxis range",
    "xaxis tickvals/ticktext",
    "axis tick density",
    "figure width/height",
    "yaxis range",
    "yaxis type",
    "zaxis range",
    "surface mesh contours",
)

#: Chrome the emitted code deliberately leaves out, to stay readable.
NOT_REPRODUCED: tuple[str, ...] = (
    "margins",
    "hovermode",
    "hovertemplate",
    "axis titles",
    "zeroline styling",
)


#: The subtle grey used for surface grid lines. Translucent so it reads as a
#: mesh laid over the colour, the way Plot3D draws it, not as a wireframe that
#: competes with the surface underneath.
MESH_LINE: str = "rgba(40, 40, 40, 0.35)"

#: Fill for a shaded region or band. Translucent on purpose: a band lies *over*
#: the curve that explains it, and an opaque fill would hide the thing the reader
#: is meant to connect it to. Lives here rather than in either renderer because
#: both of them draw it, which is this module's whole reason to exist.
REGION_FILL: str = "rgba(31, 119, 180, 0.35)"

#: Grid lines per axis when ``mesh=True`` names no number of its own (PRD
#: §23.3). Plotly's own automatic interval, left unset, picks a "nice round
#: number" spacing the way an axis chooses its ticks — usually far fewer
#: lines than the underlying sample grid, and reported as visibly sparse.
DEFAULT_MESH_LINES: int = 24

#: How tall a figure is when nothing asks for a particular height.
#:
#: Plotly's own default is 450px, chosen for a dashboard tile. A notebook cell
#: is the whole width of the page and the graph is the thing being read, not a
#: panel beside other panels — and once the range-control sidebar takes a fifth
#: of the width, 450 leaves a curve squeezed into a letterbox. This is that
#: default times 1.2, which is enough to stop the squeeze without pushing the
#: report line under the fold on a laptop screen.
DEFAULT_HEIGHT: int = 540

#: The size a plot takes when it is given none. ``height`` starts at
#: :data:`DEFAULT_HEIGHT`; ``width`` starts unset, and staying unset is what
#: lets a figure fill the cell it is in (a pixel width would leave a gap beside
#: it on a wide screen and clip it on a narrow one, which is the trap §23.1
#: already fell into with the sidebar split).
_DEFAULT_SIZE: dict[str, int | None] = {"width": None, "height": DEFAULT_HEIGHT}


def set_plot_size(
    width: int | None = None, height: int | None = None, *, reset: bool = False
) -> None:
    """Set the size every later plot takes unless it names its own.

    The notebook-wide twin of ``plot(width=..., height=...)``, for the reader
    who wants taller graphs generally rather than on one call — the same shape
    ``set_range_controls`` has, and for the same reason: a preference stated
    once at the top of a notebook should not have to be repeated on every cell.

    ``width=None``/``height=None`` leave that dimension alone rather than
    clearing it, so raising the height does not silently drop a width set a
    moment ago. ``reset=True`` restores both to the shipped defaults.
    """
    if reset:
        _DEFAULT_SIZE.update(width=None, height=DEFAULT_HEIGHT)
        return
    if width is not None:
        _DEFAULT_SIZE["width"] = _positive_pixels(width, "width")
    if height is not None:
        _DEFAULT_SIZE["height"] = _positive_pixels(height, "height")


def get_plot_size() -> tuple[int | None, int | None]:
    """The ``(width, height)`` a bare ``plot()`` currently uses."""
    return _DEFAULT_SIZE["width"], _DEFAULT_SIZE["height"]


def _positive_pixels(value: object, name: str) -> int:
    """A size is a positive whole number of pixels or it is a mistake."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsupportedInputError(
            f"{name} must be a number of pixels; got {value!r}."
        )
    if not value > 0:
        raise UnsupportedInputError(
            f"{name}={value!r} is not a size a figure can be drawn at; "
            "give a positive number of pixels, or None for the default."
        )
    return int(value)


@dataclass(frozen=True)
class RenderOptions:
    """How the figure is drawn, as opposed to what was sampled."""

    title: str | None = None
    yscale: str | None = None
    show_legend: bool | None = None
    kind: str | None = None
    #: Grid lines on a 3D surface (PRD §17: the Plot3D "sense of curvature").
    #: On by default because a bare colour gradient reads as flatter than the
    #: surface is; ``mesh=False`` returns the smooth look. An ``int`` instead
    #: of ``True`` names how many lines to draw per axis (PRD §23.3) — Plotly's
    #: own automatic spacing, used when it is left as ``True``, reads as
    #: sparse next to the actual sample density. Ignored by every 2D kind,
    #: which has no surface to rule.
    mesh: bool | int = True
    #: Axis tick labels, in the same three-valued shape as ``mesh``. ``None``
    #: leaves the count to Plotly, an ``int`` caps it — a *maximum* per axis,
    #: not a target, since asking for twenty labels does not make twenty of
    #: them legible — and ``False`` removes them, which is what a figure being
    #: shown for its shape rather than read off wants. The cap is the only
    #: lever a 3D scene has: Plotly re-lays 2D ticks against the axis's pixel
    #: length on every zoom but positions scene labels in the projection, so
    #: they crowd as the camera comes in and nothing recomputes them.
    ticks: bool | int | None = None
    #: Figure size in pixels. ``None`` on either falls back to whatever
    #: :func:`set_plot_size` last established — the height to
    #: :data:`DEFAULT_HEIGHT`, the width to unset, which is what lets the
    #: figure fill the cell rather than sit at a fixed size inside it.
    width: int | None = None
    height: int | None = None
    #: View-window overrides. A range like ``(x, -10, 10)`` sets the *domain* —
    #: where the function is sampled; these set the *window* — what the axis
    #: shows. They differ whenever the two should: to undo the automatic y-clip
    #: and see a pole, to zoom, or to line two figures up on a shared scale.
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    zlim: tuple[float, float] | None = None

    def trace_mode(self, plan: PlotPlan) -> str:
        """``"lines"`` or ``"markers"``. ``kind`` overrides; otherwise infer."""
        if self.kind == "scatter":
            return "markers"
        if self.kind in {"line", "curve"}:
            return "lines"
        # Sparse raw data reads better as points than as a line joining them.
        if plan.kind == "data" and plan.series[0].sample.n_points <= 200:
            return "markers"
        return "lines"

    def figure_size(self) -> tuple[int | None, int | None]:
        """``(width, height)`` in pixels — this call's, or the notebook's.

        Resolved here rather than at the point of use so the figure and the
        program ``show_python()`` emits cannot end up different sizes, and so
        that ``set_plot_size()`` applies to a plot built before it was called
        only if that plot is redrawn — which is what a *default* means.
        """
        width, height = get_plot_size()
        return (
            width if self.width is None else self.width,
            height if self.height is None else self.height,
        )

    def tick_limit(self) -> int | None:
        """The reader's ``ticks=`` as a Plotly ``nticks``, or ``None`` for auto.

        ``True`` and ``False`` both come back as ``None``: neither is a count,
        and ``False`` is answered by :meth:`tick_labels_visible` instead. The
        ``bool`` check has to come first — ``bool`` is a subclass of ``int``,
        so ``ticks=False`` would otherwise be read as a cap of zero, which
        Plotly takes as "choose freely" and which is the opposite of what was
        asked. (The same trap ``mesh`` fell into; see ``api._mesh_option``.)
        """
        if isinstance(self.ticks, bool) or self.ticks is None:
            return None
        return int(self.ticks)

    def tick_labels_visible(self) -> bool:
        return self.ticks is not False

    def legend_visible(self, plan: PlotPlan) -> bool:
        if self.show_legend is not None:
            return self.show_legend
        return len(plan.series) > 1

    @property
    def log_y(self) -> bool:
        return self.yscale == "log"

    def y_range(self, plan: PlotPlan) -> tuple[float, float] | None:
        """The clipped y-window, unless a log axis is already solving that.

        An explicit ``ylim`` wins over both: the reader asking for a window is
        asking to see exactly it, pole and all, so the automatic clip steps
        aside. On a log axis the numbers are read as powers of ten, which is
        Plotly's own convention for a log range.
        """
        if self.ylim is not None:
            return self.ylim
        if self.log_y:
            return None
        return plan.y_range

    def x_range(self, plan: PlotPlan) -> tuple[float, float] | None:
        """The horizontal window, or ``None`` to let Plotly choose.

        For an explicit curve x *is* the parameter, so the window is the range
        the user asked for. A parametric or polar curve has no such guarantee —
        ``x(t)`` can run to a pole — so it gets the same percentile clipping as
        the vertical axis, and nothing at all when it does not need it. Raw
        data always chooses its own extent. An explicit ``xlim`` overrides all
        of that.
        """
        if self.xlim is not None:
            return self.xlim
        if plan.kind in {"parametric", "polar"}:
            return plan.x_range
        if plan.kind == "data":
            return None
        return plan.param_range

    def z_range(self, plan: PlotPlan) -> tuple[float, float] | None:
        """The z-window for a 3D surface — the auto-clip, or ``zlim`` if given.

        The auto-clip bounds the *view* rather than the data (PRD §16.5), so a
        single pole cannot rear up as a wall that hides the surface behind it.
        ``zlim`` replaces it when the reader wants a particular window instead.
        """
        if self.zlim is not None:
            return self.zlim
        return getattr(plan.series[0].sample, "z_range", None)

    def surface_contours(self, plan: PlotPlan) -> dict[str, dict[str, object]] | None:
        """The ``contours=`` payload that draws the grid, or ``None`` when off.

        Both `x` and `y` families are shown, which is what makes a *grid* rather
        than a set of parallel ribbons. ``highlight=False`` suppresses the bold
        line Plotly otherwise sweeps under the cursor — lively, and noise here.

        ``size`` — the spacing between lines, in data units — is what actually
        controls density; left unset, Plotly picks its own "nice round number"
        interval the way an axis chooses tick spacing, which reads as sparse
        next to a 60-point sample grid. Computed from the sample's own X/Y
        extent so ``mesh=True``'s line count means the same thing on a surface
        spanning 2 units as one spanning 2000.
        """
        if not self.mesh:
            return None
        target = self.mesh if isinstance(self.mesh, int) and not isinstance(self.mesh, bool) else DEFAULT_MESH_LINES
        sample = plan.series[0].sample
        return dict(
            x=_mesh_axis(float(sample.x.min()), float(sample.x.max()), target),
            y=_mesh_axis(float(sample.y.min()), float(sample.y.max()), target),
        )

    def wants_pi_ticks(self, plan: PlotPlan) -> bool:
        """Only where the horizontal axis really is the expression's variable."""
        return plan.kind not in {"parametric", "polar", "data", "callable"}


def _mesh_axis(low: float, high: float, target_lines: int) -> dict[str, object]:
    """One axis's ``contours`` entry, with ``size`` set unless there is no span."""
    line: dict[str, object] = dict(show=True, color=MESH_LINE, width=1, highlight=False)
    span = high - low
    if span > 0 and target_lines > 0:
        # Plotly otherwise chooses its own contour start/end levels.  A
        # ``size`` by itself is therefore not a requested line count: on
        # many ranges it snaps to the same few "nice" levels regardless of
        # ``mesh=5`` versus ``mesh=50``.  Pin all three values to the sampled
        # extent so the requested density is visible and predictable.
        line.update(start=low, end=high, size=span / target_lines)
    return line
