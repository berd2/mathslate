"""Plot plan → Plotly ``Figure``.

The backend is deliberately the only module that imports Plotly, so swapping
it out later (PRD open decision 5) touches one file.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go

from ..core.dispatch import PlotPlan
from . import axes
from .options import REGION_FILL, RenderOptions

__all__ = [
    "figure_from_plan",
    "figure_with_frames",
    "describe_plan",
    "pi_ticks_for",
]

TEMPLATE: str = "plotly_white"



def figure_from_plan(plan: PlotPlan, options: RenderOptions | None = None) -> go.Figure:
    """Build the Plotly figure. NaNs in the data are what break the lines.

    Every decision here that ``show_python()`` must also make lives on
    ``options``, so the two cannot drift apart.
    """
    options = options or RenderOptions()
    if plan.two_variable:
        return _apply_size(_two_variable_figure(plan, options), options)
    if plan.kind == "space":
        return _apply_size(_space_curve_figure(plan, options), options)
    if plan.kind in {"hist", "box"}:
        return _apply_size(_distribution_figure(plan, options), options)
    if plan.kind == "linalg":
        return _apply_size(_linalg_figure(plan, options), options)

    figure = go.Figure(data=_flat_traces(plan, options))

    figure.update_layout(
        template=TEMPLATE,
        showlegend=options.legend_visible(plan),
        margin=dict(l=60, r=20, t=40 if options.title else 20, b=50),
        title=options.title,
        hovermode="closest",
    )
    labels = plan.axis_labels or (_x_title(plan), "y")
    figure.update_xaxes(title_text=labels[0], zeroline=True, zerolinewidth=1)
    figure.update_yaxes(title_text=labels[1], zeroline=True, zerolinewidth=1)

    _apply_view_window(figure, plan, options)

    if options.log_y:
        figure.update_yaxes(type="log")

    ticks = pi_ticks_for(plan, options)
    if ticks is not None:
        figure.update_xaxes(tickmode="array", tickvals=list(ticks.values), ticktext=list(ticks.text))
    _apply_tick_density(figure, options)

    if plan.kind in {"parametric", "polar"}:
        figure.update_yaxes(scaleanchor="x", scaleratio=1)

    if plan.kind == "band":
        _shade_bands(figure, plan)

    return _apply_size(figure, options)


def _apply_size(figure: go.Figure, options: RenderOptions) -> go.Figure:
    """The figure's pixel size — every kind, in one place.

    Applied at the one gate every figure leaves through rather than inside each
    builder, because "how tall is a plot" has no per-kind answer and six copies
    of it would eventually disagree. A dimension left ``None`` is not written
    at all: Plotly reads an unset width as "measure the container", which is
    what makes a figure fill its notebook cell, and writing an explicit number
    there would trade that for a fixed size that is wrong on every other
    screen.
    """
    width, height = options.figure_size()
    if width is not None:
        figure.update_layout(width=int(width))
    if height is not None:
        figure.update_layout(height=int(height))
    return figure


def _apply_tick_density(figure: go.Figure, options: RenderOptions) -> None:
    """The reader's ``ticks=`` on a flat figure's two axes.

    A cap only bounds the *numeric* branch: under ``tickmode="array"`` — the π
    labels set just above — Plotly draws the array it was given and ignores
    ``nticks`` entirely. Hiding the labels still works there, which is the
    behaviour ``ticks=False`` is actually asked for.
    """
    limit = options.tick_limit()
    if limit is not None:
        figure.update_xaxes(nticks=limit)
        figure.update_yaxes(nticks=limit)
    if not options.tick_labels_visible():
        figure.update_xaxes(showticklabels=False)
        figure.update_yaxes(showticklabels=False)


def _scene_tick_density(options: RenderOptions) -> dict[str, object]:
    """The same, as ``scene=`` entries — the only tick lever 3D offers.

    Plotly re-lays a 2D axis's ticks against its pixel length on every zoom.
    A scene's are positioned in the projection instead and nothing recomputes
    them as the camera moves, so on a 3D plot this is not a refinement over an
    automatic count that already works: it is the whole of the control.
    """
    scene: dict[str, object] = {}
    limit = options.tick_limit()
    for axis in ("xaxis", "yaxis", "zaxis"):
        if limit is not None:
            scene[f"{axis}_nticks"] = limit
        if not options.tick_labels_visible():
            scene[f"{axis}_showticklabels"] = False
    return scene


def _flat_traces(plan: PlotPlan, options: RenderOptions) -> list[go.Scatter]:
    """Render the data traces of an ordinary 2-D plan, without its layout."""
    mode = options.trace_mode(plan)
    return [
        go.Scatter(
            x=series.sample.x,
            y=series.sample.y,
            mode=mode,
            name=series.name,
            connectgaps=False,
            hovertemplate="%{x:.4g}, %{y:.4g}<extra>" + series.name + "</extra>",
        )
        for series in plan.series
    ]


def _shade_bands(figure: go.Figure, plan: PlotPlan) -> None:
    """Shade the intervals where an inequality holds, behind its curve.

    Drawn as layout shapes rather than traces: a shape spans the whole y axis
    however the axis is later rescaled, which is what "this interval of x" means.
    A filled trace would have to pick a height, and would then be wrong the
    moment the reader zoomed.

    `layer="below"` matters. The curve is the explanation of the band — it
    crosses zero exactly where the shading starts — and a translucent fill drawn
    on top of it dulls the one line the reader is meant to follow.
    """
    for start, end in plan.bands:
        # `add_vrect` spans the plot height itself; naming `yref` as well makes
        # Plotly build "paper domain" and reject it.
        figure.add_vrect(
            x0=start, x1=end, fillcolor=REGION_FILL, line_width=0, layer="below"
        )


def _apply_view_window(
    figure: go.Figure, plan: PlotPlan, options: RenderOptions
) -> None:
    """Put the axis windows on ``figure`` — automatic, or whatever was asked for.

    Every flat kind gets this, not the ones somebody remembered: `xlim` used to
    be silently ignored by `hist`, `box` and `linalg` because their builders
    never asked. `x_range`/`y_range` return ``None`` where a kind has no
    automatic window, so asking is always safe and the answer is theirs alone.
    """
    x_range = options.x_range(plan)
    if x_range is not None:
        figure.update_xaxes(range=list(x_range))
    y_range = options.y_range(plan)
    if y_range is not None:
        figure.update_yaxes(range=list(y_range))


def _axis_names(plan: PlotPlan) -> tuple[str, str]:
    if plan.axes is not None:
        return plan.axes[0].name, plan.axes[1].name
    return "x", "y"


def _two_variable_figure(plan: PlotPlan, options: RenderOptions) -> go.Figure:
    """Surface, contour, or the zero level of one (an implicit curve)."""
    sample = plan.series[0].sample
    first, second = _axis_names(plan)
    figure = go.Figure()

    contours = options.surface_contours(plan)
    if plan.kind == "psurface":
        figure.add_trace(
            go.Surface(
                x=sample.x, y=sample.y, z=sample.z, showscale=False,
                contours=contours,
            )
        )
    elif plan.kind == "surface":
        figure.add_trace(
            go.Surface(
                x=sample.x,
                y=sample.y,
                z=sample.z,
                showscale=False,
                cmin=sample.z_range[0] if sample.z_range else None,
                cmax=sample.z_range[1] if sample.z_range else None,
                contours=contours,
            )
        )
    elif plan.kind == "region":
        # A Heatmap of the 1/0 truth grid, with 0 drawn as fully transparent so
        # the region is what appears and its complement is simply the page.
        # `Contour` was the alternative and it is wrong here: it interpolates
        # between levels, which invents a soft edge for a boundary that is not
        # soft — the set either contains a point or it does not.
        figure.add_trace(
            go.Heatmap(
                x=sample.x,
                y=sample.y,
                z=sample.z,
                colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, REGION_FILL]],
                zmin=0.0,
                zmax=1.0,
                showscale=False,
                hoverongaps=False,
                hovertemplate=f"{first}=%{{x:.4g}}<br>{second}=%{{y:.4g}}<extra></extra>",
            )
        )
    elif plan.kind == "implicit":
        # Only the level where lhs - rhs is zero: that curve *is* the answer,
        # and every other level is a different equation.
        figure.add_trace(
            go.Contour(
                x=sample.x,
                y=sample.y,
                z=sample.z,
                contours=dict(start=0.0, end=0.0, size=1.0, coloring="none", showlabels=False),
                line=dict(width=2),
                showscale=False,
                hoverinfo="skip",
            )
        )
    else:
        figure.add_trace(
            go.Contour(
                x=sample.x,
                y=sample.y,
                z=sample.z,
                showscale=True,
                colorbar=dict(title=""),
                zmin=sample.z_range[0] if sample.z_range else None,
                zmax=sample.z_range[1] if sample.z_range else None,
            )
        )

    figure.update_layout(
        template=TEMPLATE,
        showlegend=False,
        margin=dict(l=60, r=20, t=40 if options.title else 20, b=50),
        title=options.title,
    )
    if plan.kind in {"surface", "psurface"}:
        scene = dict(
            xaxis_title=first if plan.kind == "surface" else "x",
            yaxis_title=second if plan.kind == "surface" else "y",
            zaxis_title="z",
        )
        # Clipping the colour range alone leaves the *geometry* spiking: a pole
        # 3481 tall on a surface whose features live within 14 is a wall that
        # hides everything behind it. 2D solves the same problem by bounding
        # the view (`update_yaxes(range=...)`), not the data; the scene's z axis
        # is the same lever one dimension up. `zlim` overrides the auto-clip.
        limits = options.z_range(plan)
        if limits is not None:
            scene["zaxis_range"] = list(limits)
        scene.update(_scene_tick_density(options))
        figure.update_layout(scene=scene)
    else:
        figure.update_xaxes(title_text=first)
        figure.update_yaxes(title_text=second)
        # A region or contour is flat, so its axes take a view window like any
        # 2D plot. The domain is the grid; xlim/ylim zoom what is shown of it.
        _apply_view_window(figure, plan, options)
        _apply_tick_density(figure, options)
    return figure


def _distribution_figure(plan: PlotPlan, options: RenderOptions) -> go.Figure:
    """A histogram or box plot — the shape of the numbers, not their order."""
    figure = go.Figure()
    for series in plan.series:
        values = series.sample.y[np.isfinite(series.sample.y)]
        if plan.kind == "hist":
            figure.add_trace(go.Histogram(x=values, name=series.name, opacity=0.75))
        else:
            figure.add_trace(go.Box(y=values, name=series.name, boxmean=True))

    labels = plan.axis_labels or ("value", "count")
    figure.update_layout(
        template=TEMPLATE,
        showlegend=options.legend_visible(plan),
        margin=dict(l=60, r=20, t=40 if options.title else 20, b=50),
        title=options.title,
        # Overlaid rather than stacked: stacking two histograms answers a
        # question about their sum, which is not what was asked.
        barmode="overlay",
    )
    figure.update_xaxes(title_text=labels[0])
    figure.update_yaxes(title_text=labels[1])
    _apply_view_window(figure, plan, options)
    _apply_tick_density(figure, options)
    return figure


def _linalg_figure(plan: PlotPlan, options: RenderOptions) -> go.Figure:
    """A 2x2 matrix drawn as what it does to the plane."""
    figure = go.Figure()
    before, after = plan.series[0], plan.series[1]
    figure.add_trace(
        go.Scatter(
            x=before.sample.x,
            y=before.sample.y,
            mode="lines",
            name=before.name,
            line=dict(dash="dot", width=2),
            fill="toself",
            opacity=0.35,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=after.sample.x,
            y=after.sample.y,
            mode="lines",
            name=after.name,
            line=dict(width=3),
            fill="toself",
            opacity=0.35,
        )
    )
    for value, vector in plan.eigen:
        figure.add_trace(
            go.Scatter(
                x=[0.0, float(vector[0] * value)],
                y=[0.0, float(vector[1] * value)],
                mode="lines+markers",
                name=f"eigenvector (λ={value:.4g})",
                line=dict(width=2),
            )
        )

    labels = plan.axis_labels or ("x", "y")
    figure.update_layout(
        template=TEMPLATE,
        showlegend=options.legend_visible(plan),
        margin=dict(l=60, r=20, t=40 if options.title else 20, b=50),
        title=options.title,
    )
    figure.update_xaxes(title_text=labels[0], zeroline=True, zerolinewidth=1)
    figure.update_yaxes(
        title_text=labels[1],
        zeroline=True,
        zerolinewidth=1,
        # Equal aspect, or the shape's distortion is the plot's, not the matrix's.
        scaleanchor="x",
        scaleratio=1,
    )
    _apply_view_window(figure, plan, options)
    _apply_tick_density(figure, options)
    return figure


def _space_curve_figure(plan: PlotPlan, options: RenderOptions) -> go.Figure:
    """A 3D space curve. NaNs break it exactly as they break a 2D one."""
    sample = plan.series[0].sample
    figure = go.Figure(
        go.Scatter3d(
            x=sample.x,
            y=sample.y,
            z=sample.z,
            mode="lines" if options.trace_mode(plan) == "lines" else "markers",
            name=plan.series[0].name,
            connectgaps=False,
            line=dict(width=4),
        )
    )
    scene: dict[str, object] = dict(xaxis_title="x", yaxis_title="y", zaxis_title="z")
    if options.xlim is not None:
        scene["xaxis_range"] = list(options.xlim)
    if options.ylim is not None:
        scene["yaxis_range"] = list(options.ylim)
    if options.zlim is not None:
        scene["zaxis_range"] = list(options.zlim)
    scene.update(_scene_tick_density(options))
    figure.update_layout(
        template=TEMPLATE,
        showlegend=options.legend_visible(plan),
        margin=dict(l=0, r=0, t=40 if options.title else 20, b=0),
        title=options.title,
        scene=scene,
    )
    return figure


def figure_with_frames(
    plans: Sequence[PlotPlan],
    steps: Sequence[float],
    label: str,
    options: RenderOptions | None = None,
    *,
    play: bool = False,
) -> go.Figure:
    """One figure showing ``plans[i]`` at slider position ``steps[i]``.

    The control is Plotly's own, so no callback is written anywhere (PRD 5.7)
    and nothing needs installing: the frames travel inside the figure and
    survive being written to a single HTML file.

    The axes are locked to the union over every frame. Letting Plotly
    autoscale per frame would make the curve appear to stay still while the
    axis numbers slid past it, which is the opposite of what the reader is
    trying to see.
    """
    options = options or RenderOptions()
    if not plans:
        raise ValueError("no frames to draw")

    figure = figure_from_plan(plans[0], options)
    figure.frames = [
        _frame_from_plan(plan, options, name=_frame_name(index))
        for index, plan in enumerate(plans)
    ]

    controls: list[dict[str, object]] = [
        {
            "active": 0,
            "currentvalue": {"prefix": f"{label} = "},
            "pad": {"t": 40},
            "steps": [
                {
                    # The label is what the reader sees; the animate target is
                    # the frame's own name. They must not be the same string:
                    # `_step_name` rounds for display, so a slider over a narrow
                    # range (1.0 to 1.0002) rounds every position to "1", and if
                    # that were also the frame name every step would animate to
                    # the first frame — the slider would move and the plot would
                    # not. The index is unique per frame; the value is only shown.
                    "label": _step_name(value),
                    "method": "animate",
                    "args": [[_frame_name(index)], _TRANSITION],
                }
                for index, value in enumerate(steps)
            ],
        }
    ]
    figure.update_layout(sliders=controls, margin=dict(l=60, r=20, t=40, b=110))
    if play:
        figure.update_layout(updatemenus=[_PLAY_BUTTONS])

    span = _union_range(plans, options)
    if span[0] is not None:
        figure.update_xaxes(range=list(span[0]))
    if span[1] is not None:
        figure.update_yaxes(range=list(span[1]))
    return figure


def _frame_name(index: int) -> str:
    """A frame's animation id — unique per frame, unlike its display label."""
    return str(index)


def _frame_from_plan(
    plan: PlotPlan, options: RenderOptions, *, name: str
) -> go.Frame:
    """A frame with the same trace kinds and count as the plan's real figure.

    A frame used to be rebuilt as a list of 2-D ``Scatter`` traces regardless
    of what the base figure was. That happened to work for one ordinary curve,
    but a surface became ``Surface -> Scatter``, a region ``Heatmap -> Scatter``
    and an overlay lost its later traces. Building through the same renderer as
    the base figure makes the dispatch contract apply to frames too.

    Bands are the one kind whose answer also lives in layout shapes. Carry only
    those shapes in the frame; carrying the whole per-frame layout would undo
    the union window below and make the axes jump during animation.
    """
    if plan.two_variable:
        data = list(_two_variable_figure(plan, options).data)
    elif plan.kind == "space":
        data = list(_space_curve_figure(plan, options).data)
    elif plan.kind in {"hist", "box"}:
        data = list(_distribution_figure(plan, options).data)
    elif plan.kind == "linalg":
        data = list(_linalg_figure(plan, options).data)
    else:
        data = _flat_traces(plan, options)

    frame_layout: go.Layout | None = None
    if plan.kind == "band":
        shape_holder = go.Figure()
        _shade_bands(shape_holder, plan)
        frame_layout = go.Layout(shapes=shape_holder.layout.shapes)
    return go.Frame(
        data=data,
        layout=frame_layout,
        name=name,
    )


_TRANSITION: dict[str, object] = {
    "mode": "immediate",
    "frame": {"duration": 300, "redraw": False},
    "transition": {"duration": 0},
}

_PLAY_BUTTONS: dict[str, object] = {
    "type": "buttons",
    "showactive": False,
    "x": 0.05,
    "y": -0.18,
    "xanchor": "right",
    "yanchor": "top",
    "buttons": [
        {
            "label": "play",
            "method": "animate",
            "args": [
                None,
                {
                    "mode": "immediate",
                    "fromcurrent": True,
                    "frame": {"duration": 120, "redraw": False},
                    "transition": {"duration": 0},
                },
            ],
        },
        {
            "label": "pause",
            "method": "animate",
            "args": [[None], _TRANSITION],
        },
    ],
}


def _step_name(value: float) -> str:
    return _num(value)


def _union_range(
    plans: Sequence[PlotPlan], options: RenderOptions
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """The window that holds every frame, so the axes do not jump."""
    xs = [r for r in (options.x_range(plan) for plan in plans) if r is not None]
    ys = [r for r in (options.y_range(plan) for plan in plans) if r is not None]
    if not ys:
        # No frame asked for clipping, but the curves still have to share one
        # window or the reader cannot compare them.
        finite = [
            v
            for plan in plans
            for series in plan.series
            for v in series.sample.y[np.isfinite(series.sample.y)]
        ]
        if finite:
            low, high = float(min(finite)), float(max(finite))
            pad = max((high - low) * 0.05, 1e-9)
            ys = [(low - pad, high + pad)]
    x_span = (min(r[0] for r in xs), max(r[1] for r in xs)) if xs else None
    y_span = (min(r[0] for r in ys), max(r[1] for r in ys)) if ys else None
    return x_span, y_span


def _x_title(plan: PlotPlan) -> str:
    if plan.kind in {"parametric", "polar"}:
        return "x"
    if plan.symbol is not None:
        return plan.symbol.name
    return "x"


def pi_axis(plan: PlotPlan, options: RenderOptions | None = None) -> bool:
    """Whether π labels belong on this plan's horizontal axis *at all*.

    Window-independent, unlike :func:`pi_ticks_for`, which also asks whether a
    multiple of π lands often enough in one particular window. The live sidebar
    needs the two questions separated: a reader zooming into a third of a period
    has not stopped plotting a trig function, so the axis is still a π axis —
    it is only this window that no π spacing fits, and the answer for it is
    numbers until the reader zooms back out.
    """
    options = options or RenderOptions()
    return options.wants_pi_ticks(plan) and axes.uses_pi_ticks(plan.exprs)


def pi_ticks_for(plan: PlotPlan, options: RenderOptions | None = None) -> axes.PiTicks | None:
    """The π ticks for this plan, or ``None``. Shared with ``codegen``."""
    if not pi_axis(plan, options):
        return None
    if plan.param_range is None:
        return None
    return axes.pi_ticks(*plan.param_range)


def describe_plan(plan: PlotPlan) -> str:
    """The one-line 'here is what I inferred' report (PRD 5.1).

    This line is not logging: it is where a learner first discovers that
    parameters they never wrote exist.
    """
    parts: list[str] = [plan.kind]
    if plan.param_range is not None:
        name = plan.symbol.name if plan.symbol is not None else "x"
        lo, hi = plan.param_range
        window = f"{name} ∈ [{_num(lo)}, {_num(hi)}]"
        if plan.axes is not None and plan.second_range is not None:
            second_lo, second_hi = plan.second_range
            window += (
                f", {plan.axes[1].name} ∈ [{_num(second_lo)}, {_num(second_hi)}]"
            )
        parts.append(window)
    if plan.two_variable:
        grid = plan.series[0].sample.z
        parts.append(f"{grid.shape[1]}×{grid.shape[0]} samples")
    elif plan.kind == "linalg":
        parts.append(f"{len(plan.eigen)} real eigenvector"
                     + ("" if len(plan.eigen) == 1 else "s"))
    elif plan.kind in {"hist", "box"}:
        parts.append(
            f"{len(plan.series)} column" + ("" if len(plan.series) == 1 else "s")
        )
        parts.append(f"{plan.total_points} values")
    else:
        parts.append(f"{plan.total_points} samples")
    breaks = sum(len(s.sample.breakpoints) for s in plan.series)
    if breaks:
        parts.append(f"{breaks} discontinuit{'y' if breaks == 1 else 'ies'} handled")
    if any(not s.sample.vectorized for s in plan.series):
        parts.append("element-wise evaluation")
    return " | ".join(parts)


def _num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4g}"


def figure_y_data(plan: PlotPlan) -> np.ndarray:
    """All y samples across series — used for the log-scale hint."""
    if not plan.series:
        return np.array([], dtype=np.float64)
    if plan.two_variable:
        return np.concatenate([np.ravel(s.sample.z) for s in plan.series])
    return np.concatenate([np.ravel(s.sample.y) for s in plan.series])
