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

__all__ = [
    "RenderOptions",
    "REPRODUCED",
    "NOT_REPRODUCED",
    "MESH_LINE",
    "REGION_FILL",
    "DEFAULT_MESH_LINES",
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
