"""``show_python()`` — the actual mechanism of growth (PRD 5.5).

The emitted code **must run**. That is a success metric, not decoration, so
this module never emits pseudocode, ellipses or hand-waving comments. What it
emits is the plain NumPy + SymPy + Plotly program a competent user would have
written by hand to get the same picture — including the parts MathSlate did
silently: restricting to the real domain, cutting the line at singularities,
clipping the y-axis and labelling π ticks.
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import sympy as sp
from plotly.utils import PlotlyJSONEncoder

from .core import _source
from .core.dispatch import PlotPlan, Series
from .core.surfaces import comparison_operands
from .render import plotly_backend
from .render.options import MESH_LINE, REGION_FILL, RenderOptions
from .render.plotly_backend import pi_ticks_for

__all__ = ["generate_code", "generate_frames_code"]

#: Points per continuous piece in the emitted (uniform-grid) code.
EMITTED_POINTS: int = 1000
#: Cap on literal arrays written into the emitted code, to keep it readable.
MAX_LITERAL_POINTS: int = 2000


def generate_code(plan: PlotPlan, options: RenderOptions | None = None) -> str:
    """Return runnable Python equivalent to ``plan`` drawn with ``options``."""
    options = options or RenderOptions()
    if plan.two_variable:
        return _two_variable_code(plan, options)
    if plan.kind == "space":
        return _space_curve_code(plan, options)
    if plan.kind in {"hist", "box"}:
        return _distribution_code(plan, options)
    if plan.kind == "linalg":
        return _linalg_code(plan, options)
    if plan.kind == "data":
        return _data_code(plan, options)
    if plan.kind == "callable":
        return _callable_code(plan, options)
    if plan.kind in {"parametric", "polar"}:
        return _parametric_code(plan, options)
    return _curve_code(plan, options)


def generate_frames_code(
    plans: Sequence[PlotPlan],
    steps: Sequence[float],
    label: str,
    options: RenderOptions | None = None,
    *,
    play: bool = False,
) -> str:
    """Runnable Python for a slider-driven figure.

    A still picture would not be the same result: the figure on screen moves.
    So the frames are emitted too — as literal samples, one set per slider
    position. That is the same choice the callable and data emitters make, and
    for the same reason. Re-deriving them would mean emitting a loop that
    substitutes into the expression and re-runs the adaptive sampler, which is
    the sampler itself, which §9 of the manual says is not what this is for.
    """
    options = options or RenderOptions()
    payload = _frames_payload(plans, options)
    encoded = json.dumps(
        payload, cls=PlotlyJSONEncoder, ensure_ascii=True, separators=(",", ":")
    )
    body: list[str] = [
        f"# One frame per position of {label}. Plotly's own slider drives them,",
        "# so there is no callback anywhere and the file stands alone.",
        "# The trace payloads are literal samples: this preserves overlays, 3-D",
        "# traces and regions without copying MathSlate's sampler into this file.",
        "import json",
        f"payload = json.loads({encoded!r})",
        "frames = []",
        f"positions = {_fmt_list(list(steps))}",
    ]
    body += [
        "",
        "# A frame's name is its unique animation id (the index); the position",
        "# is only the label shown on the slider. They must differ: a narrow",
        "# range formats every position to the same string, and reusing that as",
        "# the name would make every step animate to the first frame.",
        "for index, item in enumerate(payload['frames']):",
        "    frames.append({",
        "        'data': item['data'],",
        "        'layout': item.get('layout') or None,",
        "        'name': str(index),",
        "    })",
        "",
        "fig = go.Figure(data=payload['frames'][0]['data'],",
        "                layout=payload['base_layout'], frames=frames)",
        "fig.update_layout(",
        "    sliders=[{",
        "        'active': 0,",
        f"        'currentvalue': {{'prefix': {label + ' = '!r}}},",
        "        'pad': {'t': 40},",
        "        'steps': [",
        "            {'label': f'{p:g}', 'method': 'animate',",
        "             'args': [[str(i)], {'mode': 'immediate',",
        "                                 'frame': {'duration': 300, 'redraw': False},",
        "                                 'transition': {'duration': 0}}]}",
        "            for i, p in enumerate(positions)",
        "        ],",
        "    }],",
        ")",
    ]
    if options.title is not None:
        body.append(f"fig.update_layout(title={options.title!r})")
    if play:
        # Keep this sourced from the real renderer so the emitted controls
        # cannot drift from animate()'s controls.
        from .render.plotly_backend import _PLAY_BUTTONS

        body.append(f"fig.update_layout(updatemenus={[_PLAY_BUTTONS]!r})")
    span = _frames_window(plans, options)
    if span[0] is not None:
        body.append(f"fig.update_xaxes(range={_fmt_list(list(span[0]))})")
    if span[1] is not None:
        body.append(f"fig.update_yaxes(range={_fmt_list(list(span[1]))})")
    body.append("fig.show()")
    return _assemble(set(), body, sympy_needed=False)


def _frames_payload(
    plans: Sequence[PlotPlan], options: RenderOptions
) -> dict[str, object]:
    """JSON-ready traces for every frame, rendered through the real backend."""
    base = plotly_backend.figure_from_plan(plans[0], options)
    frames: list[dict[str, object]] = []
    for index, plan in enumerate(plans):
        frame = plotly_backend._frame_from_plan(
            plan, options, name=plotly_backend._frame_name(index)
        )
        item: dict[str, object] = {
            "data": [trace.to_plotly_json() for trace in frame.data],
        }
        if frame.layout and frame.layout.shapes:
            item["layout"] = {
                "shapes": [shape.to_plotly_json() for shape in frame.layout.shapes]
            }
        frames.append(item)
    return {
        "base_layout": base.layout.to_plotly_json(),
        "frames": frames,
    }


def _frames_window(
    plans: Sequence[PlotPlan], options: RenderOptions
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Must match the backend's union, or the two pictures differ."""
    from .render.plotly_backend import _union_range

    return _union_range(list(plans), options)


# --------------------------------------------------------------------------
# expression → Python source
# --------------------------------------------------------------------------


def _expr_source(expr: sp.Expr, symbol_names: set[str]) -> tuple[str, set[str]]:
    """SymPy expression → source text plus the sympy names it needs imported."""
    return _source.expr_source(expr, symbol_names)


def _import_block(needed: set[str], *, sympy_needed: bool = True) -> str:
    return _source.import_block(needed, sympy=sympy_needed)


def _segments_source(series: Series, fallback: tuple[float, float]) -> list[tuple[float, float]]:
    intervals = list(series.sample.domain_intervals)
    return intervals or [fallback]


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "np.nan"
    if abs(value - round(value)) < 1e-12:
        return f"{value:.1f}"
    return repr(round(float(value), 12))


def _fmt_list(values: Sequence[float]) -> str:
    return "[" + ", ".join(_fmt(v) for v in values) + "]"


def _fmt_pairs(pairs: Sequence[tuple[float, float]]) -> str:
    return "[" + ", ".join(f"({_fmt(lo)}, {_fmt(hi)})" for lo, hi in pairs) + "]"


# --------------------------------------------------------------------------
# per-kind emitters
# --------------------------------------------------------------------------


def _curve_code(plan: PlotPlan, options: RenderOptions) -> str:
    assert plan.symbol is not None and plan.param_range is not None
    name = plan.symbol.name
    needed: set[str] = set()
    body: list[str] = []

    body.append(f"{name} = sp.symbols({name!r}, real=True)")
    body.append("")

    trace_names: list[str] = []
    for index, series in enumerate(plan.series):
        assert series.expr is not None
        source, extra = _expr_source(series.expr, {name})
        needed |= extra
        suffix = "" if len(plan.series) == 1 else str(index + 1)
        segments = _segments_source(series, plan.param_range)
        per_piece = max(EMITTED_POINTS // max(len(segments), 1), 50)
        body += [
            f"expr{suffix} = {source}",
            f"fn{suffix} = sp.lambdify({name}, expr{suffix}, 'numpy')",
            "",
            "# The expression is continuous on each of these pieces and nowhere",
            "# else. Sampling them separately and joining with a NaN is what stops",
            "# Plotly drawing a line across a pole or outside the real domain.",
            f"pieces{suffix} = {_fmt_pairs(segments)}",
            f"xs{suffix}, ys{suffix} = [], []",
            f"for lo, hi in pieces{suffix}:",
            "    inset = (hi - lo) * 1e-9",
            f"    xp = np.linspace(lo + inset, hi - inset, {per_piece})",
            # lambdify returns a scalar for a constant expression, so broadcast
            # back to the grid shape before using it.
            f"    yp = np.broadcast_to(np.asarray(fn{suffix}(xp), dtype=float), xp.shape)",
            f"    xs{suffix} += [xp, np.array([np.nan])]",
            f"    ys{suffix} += [yp, np.array([np.nan])]",
            f"xs{suffix} = np.concatenate(xs{suffix})",
            f"ys{suffix} = np.concatenate(ys{suffix})",
            f"ys{suffix} = np.where(np.isfinite(ys{suffix}), ys{suffix}, np.nan)",
        ]
        edges = {v for segment in segments for v in segment}
        jumps = [
            b
            for b in series.sample.breakpoints
            if _inside(b, plan.param_range) and not any(abs(b - e) < 1e-9 for e in edges)
        ]
        if jumps:
            body += [
                "",
                "# Jump discontinuities: cut the line exactly where it jumps.",
                f"for b in {_fmt_list(jumps)}:",
                f"    i = np.searchsorted(xs{suffix}, b)",
                f"    xs{suffix} = np.insert(xs{suffix}, i, b)",
                f"    ys{suffix} = np.insert(ys{suffix}, i, np.nan)",
            ]
        body.append("")
        trace_names.append(suffix)

    mode = options.trace_mode(plan)
    traces = ", ".join(
        f"go.Scatter(x=xs{s}, y=ys{s}, mode={mode!r}, connectgaps=False, "
        f"name={plan.series[i].name!r})"
        for i, s in enumerate(trace_names)
    )
    body.append(f"fig = go.Figure([{traces}])")
    body += _layout_lines(plan, options)
    body.append("fig.show()")

    return _assemble(needed, body)


def _parametric_code(plan: PlotPlan, options: RenderOptions) -> str:
    assert plan.symbol is not None and plan.param_range is not None
    name = plan.symbol.name
    series = plan.series[0]
    fx, fy = series.components
    needed: set[str] = set()
    sx, extra = _expr_source(fx, {name})
    needed |= extra
    sy, extra = _expr_source(fy, {name})
    needed |= extra

    segments = _segments_source(series, plan.param_range)
    per_piece = max(EMITTED_POINTS // max(len(segments), 1), 50)
    body = [
        f"{name} = sp.symbols({name!r}, real=True)",
        f"x_of = sp.lambdify({name}, {sx}, 'numpy')",
        f"y_of = sp.lambdify({name}, {sy}, 'numpy')",
        "",
        f"# Both components are continuous on each of these pieces of {name} and",
        "# nowhere else. Sampling them separately and joining with a NaN is what",
        "# stops Plotly drawing a line across a pole.",
        f"pieces = {_fmt_pairs(segments)}",
        "ts, xs, ys = [], [], []",
        "for lo, hi in pieces:",
        "    inset = (hi - lo) * 1e-9",
        f"    tp = np.linspace(lo + inset, hi - inset, {per_piece})",
        # broadcast_to covers a constant component, where lambdify returns a scalar.
        "    xp = np.broadcast_to(np.asarray(x_of(tp), dtype=float), tp.shape)",
        "    yp = np.broadcast_to(np.asarray(y_of(tp), dtype=float), tp.shape)",
        "    ts += [tp, np.array([np.nan])]",
        "    xs += [xp, np.array([np.nan])]",
        "    ys += [yp, np.array([np.nan])]",
        "ts = np.concatenate(ts)",
        "xs = np.concatenate(xs)",
        "ys = np.concatenate(ys)",
        "xs = np.where(np.isfinite(xs), xs, np.nan)",
        "ys = np.where(np.isfinite(ys), ys, np.nan)",
    ]

    edges = {v for segment in segments for v in segment}
    jumps = [
        b
        for b in series.sample.breakpoints
        if _inside(b, plan.param_range) and not any(abs(b - e) < 1e-9 for e in edges)
    ]
    if jumps:
        body += [
            "",
            f"# Jumps in x or in y: cut the line at the {name} where it jumps.",
            f"for b in {_fmt_list(jumps)}:",
            "    i = np.searchsorted(ts, b)",
            "    ts = np.insert(ts, i, b)",
            "    xs = np.insert(xs, i, np.nan)",
            "    ys = np.insert(ys, i, np.nan)",
        ]

    body += [
        "",
        f"fig = go.Figure(go.Scatter(x=xs, y=ys, mode={options.trace_mode(plan)!r}, "
        f"connectgaps=False, name={series.name!r}))",
    ]
    body += _layout_lines(plan, options)
    body.append("fig.update_yaxes(scaleanchor='x', scaleratio=1)")
    body.append("fig.show()")
    return _assemble(needed, body)


def _callable_code(plan: PlotPlan, options: RenderOptions) -> str:
    """Emit the samples as literals.

    A plain Python function cannot be written into source code: its name may be
    qualified (``np.sin`` prints as ``sin``), unbound, or absent entirely
    (``<lambda>``). Emitting a call to it would produce code that only runs in
    the namespace it came from — which is not what ``show_python()`` promises.
    So the sampled points go in as literals, which always run.
    """
    series = plan.series[0]
    sample = series.sample
    xs, ys, reduced = _array_literals(sample.x, sample.y)
    header = [
        f"# {series.name} is your own Python function, so it cannot be written",
        "# into this file. These are the samples MathSlate took from it.",
    ]
    if reduced:
        header.append(
            f"# (evenly reduced to {MAX_LITERAL_POINTS} points to stay readable)"
        )
    body = [
        *header,
        f"xs = np.array({xs})",
        f"ys = np.array({ys})",
        "",
        _figure_line(plan, options, ("xs", "ys"), series.name),
    ]
    body += _layout_lines(plan, options)
    body.append("fig.show()")
    return _assemble(set(), body, sympy_needed=False)


def _two_variable_code(plan: PlotPlan, options: RenderOptions) -> str:
    """Surfaces, contours and implicit curves.

    Unlike a curve, the grid is emitted as the *call that builds it* rather
    than as literals: 60x60 is 3600 numbers, and a `meshgrid` plus one
    `lambdify` is both shorter and closer to what the reader would write. It is
    also exactly reproducible, because a uniform grid has nothing adaptive
    about it — the caveat in §9 of the manual does not apply here.
    """
    series = plan.series[0]
    sample = series.sample
    assert plan.axes is not None and plan.second_range is not None
    first, second = plan.axes[0].name, plan.axes[1].name
    resolution = sample.z.shape[0]

    if plan.kind == "psurface":
        return _parametric_surface_code(plan, options)
    if plan.kind == "region":
        return _region_code(plan, options)

    assert series.expr is not None
    source, needed = _expr_source(series.expr, {first, second})
    lo, hi = plan.param_range or (-5.0, 5.0)
    vlo, vhi = plan.second_range

    body = [
        f"{first}, {second} = sp.symbols({first + ' ' + second!r}, real=True)",
        f"expr = {source}",
        f"fn = sp.lambdify(({first}, {second}), expr, 'numpy')",
        "",
        f"{first}s = np.linspace({_fmt(lo)}, {_fmt(hi)}, {resolution})",
        f"{second}s = np.linspace({_fmt(vlo)}, {_fmt(vhi)}, {resolution})",
        f"grid_{first}, grid_{second} = np.meshgrid({first}s, {second}s)",
        f"z = np.asarray(fn(grid_{first}, grid_{second}), dtype=complex)",
        "# Anything not a real number becomes a hole rather than a wrong value.",
        "z = np.where(np.abs(z.imag) > 1e-9 * np.maximum(np.abs(z.real), 1.0),",
        "             np.nan, z.real)",
        "",
    ]
    body += _two_variable_trace(plan, sample, first, second, options)
    body += _two_variable_layout(plan, options, first, second)
    body.append("fig.show()")
    return _assemble(needed, body)


def _region_code(plan: PlotPlan, options: RenderOptions) -> str:
    """A filled inequality region.

    Separate from the surface path because the grid holds *truth values*, and the
    two things that follow from that both differ. A relation lambdifies to
    `logical_and.reduce` / `less` rather than to arithmetic, so there is no
    complex part to discard; and "not real here" has to be found from the
    operands, because a comparison against NaN comes back False and would
    otherwise be indistinguishable from being genuinely outside the region.
    """
    series = plan.series[0]
    sample = series.sample
    assert plan.axes is not None and plan.second_range is not None
    assert series.expr is not None
    first, second = plan.axes[0].name, plan.axes[1].name
    resolution = sample.z.shape[0]
    lo, hi = plan.param_range or (-5.0, 5.0)
    vlo, vhi = plan.second_range

    source, needed = _expr_source(series.expr, {first, second})
    operands = comparison_operands(series.expr)
    operand_sources = [_expr_source(o, {first, second})[0] for o in operands]
    for operand in operands:
        needed |= _expr_source(operand, {first, second})[1]

    body = [
        f"{first}, {second} = sp.symbols({first + ' ' + second!r}, real=True)",
        f"relation = {source}",
        f"test = sp.lambdify(({first}, {second}), relation, 'numpy')",
        "",
        f"{first}s = np.linspace({_fmt(lo)}, {_fmt(hi)}, {resolution})",
        f"{second}s = np.linspace({_fmt(vlo)}, {_fmt(vhi)}, {resolution})",
        f"grid_{first}, grid_{second} = np.meshgrid({first}s, {second}s)",
        f"inside = np.where(test(grid_{first}, grid_{second}), 1.0, 0.0)",
    ]
    if operand_sources:
        body += [
            "",
            "# A comparison against a NaN is False, which is indistinguishable from",
            "# being outside the region — so the operands decide where there is no",
            "# truth value at all, and those points become holes.",
            f"for side in [{', '.join(operand_sources)}]:",
            f"    values = sp.lambdify(({first}, {second}), side, 'numpy')"
            f"(grid_{first}, grid_{second})",
            "    values = np.broadcast_to(np.asarray(values, dtype=complex),"
            f" grid_{first}.shape)",
            "    real = np.where(np.abs(values.imag) > 1e-9 *"
            " np.maximum(np.abs(values.real), 1.0), np.nan, values.real)",
            "    inside = np.where(np.isfinite(real), inside, np.nan)",
        ]
    body += [
        "",
        "# 0 is drawn fully transparent, so the region is what appears.",
        f"fig = go.Figure(go.Heatmap(x={first}s, y={second}s, z=inside,",
        f"    colorscale=[[0.0, 'rgba(0,0,0,0)'], [1.0, {REGION_FILL!r}]],",
        "    zmin=0.0, zmax=1.0, showscale=False, hoverongaps=False))",
    ]
    body += _two_variable_layout(plan, options, first, second)
    body.append("fig.show()")
    return _assemble(needed, body)


def _mesh_kwarg(options: RenderOptions, plan: PlotPlan) -> str:
    """The ``contours=`` source that draws surface grid lines, or ``""``.

    Calls :meth:`RenderOptions.surface_contours` — the same method the real
    figure builds from — rather than reformatting ``MESH_LINE``/``mesh`` by
    hand. A hand-written version here used to omit the line spacing
    entirely, so `show_python()` always drew Plotly's sparser automatic
    interval no matter how dense `mesh=<int>` (PRD §23.3) asked for.
    """
    contours = options.surface_contours(plan)
    if contours is None:
        return ""

    def _axis(spec: dict[str, object]) -> str:
        size = spec.get("size")
        extra = f", size={_fmt(size)}" if size is not None else ""
        return f"dict(show=True, color={MESH_LINE!r}, width=1, highlight=False{extra})"

    return f", contours=dict(x={_axis(contours['x'])}, y={_axis(contours['y'])})"


def _two_variable_trace(
    plan: PlotPlan, sample: object, first: str, second: str, options: RenderOptions
) -> list[str]:
    limits = getattr(sample, "z_range", None)
    if plan.kind == "surface":
        bounds = (
            f", cmin={_fmt(limits[0])}, cmax={_fmt(limits[1])}" if limits else ""
        )
        return [
            f"fig = go.Figure(go.Surface(x={first}s, y={second}s, z=z, "
            f"showscale=False{bounds}{_mesh_kwarg(options, plan)}))"
        ]
    if plan.kind == "implicit":
        return [
            "# Only the level where lhs - rhs is zero: every other level is a",
            "# different equation.",
            f"fig = go.Figure(go.Contour(x={first}s, y={second}s, z=z,",
            "    contours=dict(start=0.0, end=0.0, size=1.0, coloring='none'),",
            "    line=dict(width=2), showscale=False))",
        ]
    bounds = f", zmin={_fmt(limits[0])}, zmax={_fmt(limits[1])}" if limits else ""
    return [
        f"fig = go.Figure(go.Contour(x={first}s, y={second}s, z=z, "
        f"showscale=True{bounds}))"
    ]


def _two_variable_layout(
    plan: PlotPlan, options: RenderOptions, first: str, second: str
) -> list[str]:
    layout = f"template='plotly_white', showlegend=False"
    if options.title is not None:
        layout += f", title={options.title!r}"
    lines = [f"fig.update_layout({layout})"]
    if plan.kind in {"surface", "psurface"}:
        names = ("x", "y") if plan.kind == "psurface" else (first, second)
        scene = (
            f"xaxis_title={names[0]!r}, yaxis_title={names[1]!r}, zaxis_title='z'"
        )
        limits = options.z_range(plan)
        if limits is not None:
            # Same lever as a 2D y-range: bound the view, not the data, so one
            # pole cannot hide the surface behind a wall. `zlim` overrides it.
            scene += f", zaxis_range={_fmt_list(list(limits))}"
        scene += _scene_tick_kwargs(options)
        lines.append(f"fig.update_layout(scene=dict({scene}))")
    else:
        lines.append(f"fig.update_xaxes(title_text={first!r})")
        lines.append(f"fig.update_yaxes(title_text={second!r})")
        lines += _view_window_lines(plan, options)
        lines += _tick_density_lines(options)
    return lines


def _parametric_surface_code(plan: PlotPlan, options: RenderOptions) -> str:
    series = plan.series[0]
    assert plan.axes is not None and plan.second_range is not None
    first, second = plan.axes[0].name, plan.axes[1].name
    resolution = series.sample.z.shape[0]
    lo, hi = plan.param_range or (-5.0, 5.0)
    vlo, vhi = plan.second_range

    needed: set[str] = set()
    sources: list[str] = []
    for component in series.components:
        text, extra = _expr_source(component, {first, second})
        needed |= extra
        sources.append(text)

    body = [
        f"{first}, {second} = sp.symbols({first + ' ' + second!r}, real=True)",
        f"fx = sp.lambdify(({first}, {second}), {sources[0]}, 'numpy')",
        f"fy = sp.lambdify(({first}, {second}), {sources[1]}, 'numpy')",
        f"fz = sp.lambdify(({first}, {second}), {sources[2]}, 'numpy')",
        "",
        f"{first}s = np.linspace({_fmt(lo)}, {_fmt(hi)}, {resolution})",
        f"{second}s = np.linspace({_fmt(vlo)}, {_fmt(vhi)}, {resolution})",
        f"grid_{first}, grid_{second} = np.meshgrid({first}s, {second}s)",
        "# broadcast_to covers a component that does not use both parameters.",
        f"shape = grid_{first}.shape",
        f"xs = np.broadcast_to(np.asarray(fx(grid_{first}, grid_{second}), float), shape)",
        f"ys = np.broadcast_to(np.asarray(fy(grid_{first}, grid_{second}), float), shape)",
        f"zs = np.broadcast_to(np.asarray(fz(grid_{first}, grid_{second}), float), shape)",
        "",
        f"fig = go.Figure(go.Surface(x=xs, y=ys, z=zs, showscale=False"
        f"{_mesh_kwarg(options, plan)}))",
    ]
    body += _two_variable_layout(plan, options, first, second)
    body.append("fig.show()")
    return _assemble(needed, body)


def _space_curve_code(plan: PlotPlan, options: RenderOptions) -> str:
    """A 3D space curve: the parametric emitter with a third component."""
    assert plan.symbol is not None and plan.param_range is not None
    name = plan.symbol.name
    series = plan.series[0]
    needed: set[str] = set()
    sources: list[str] = []
    for component in series.components:
        text, extra = _expr_source(component, {name})
        needed |= extra
        sources.append(text)

    segments = _segments_source(series, plan.param_range)
    per_piece = max(EMITTED_POINTS // max(len(segments), 1), 50)
    layout = (
        f"template='plotly_white', showlegend={options.legend_visible(plan)}"
    )
    if options.title is not None:
        layout += f", title={options.title!r}"
    body = [
        f"{name} = sp.symbols({name!r}, real=True)",
        f"x_of = sp.lambdify({name}, {sources[0]}, 'numpy')",
        f"y_of = sp.lambdify({name}, {sources[1]}, 'numpy')",
        f"z_of = sp.lambdify({name}, {sources[2]}, 'numpy')",
        "",
        f"# All three components are continuous on these pieces of {name}.",
        f"pieces = {_fmt_pairs(segments)}",
        "xs, ys, zs = [], [], []",
        "for lo, hi in pieces:",
        "    inset = (hi - lo) * 1e-9",
        f"    tp = np.linspace(lo + inset, hi - inset, {per_piece})",
        "    xs += [np.broadcast_to(np.asarray(x_of(tp), float), tp.shape), np.array([np.nan])]",
        "    ys += [np.broadcast_to(np.asarray(y_of(tp), float), tp.shape), np.array([np.nan])]",
        "    zs += [np.broadcast_to(np.asarray(z_of(tp), float), tp.shape), np.array([np.nan])]",
        "xs, ys, zs = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)",
        "",
        f"fig = go.Figure(go.Scatter3d(x=xs, y=ys, z=zs, "
        f"mode={options.trace_mode(plan)!r},",
        f"                             connectgaps=False, name={series.name!r},",
        "                             line=dict(width=4)))",
        f"fig.update_layout({layout},",
        "    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'"
        f"{_scene_tick_kwargs(options)}))",
        "fig.show()",
    ]
    return _assemble(needed, body)


def _data_code(plan: PlotPlan, options: RenderOptions) -> str:
    """Every column, not just the first — a `Dataset` can bring several."""
    mode = options.trace_mode(plan)
    body: list[str] = []
    traces: list[str] = []
    for index, series in enumerate(plan.series):
        suffix = "" if len(plan.series) == 1 else str(index + 1)
        xs, ys, reduced = _array_literals(series.sample.x, series.sample.y)
        if reduced:
            body.append(
                f"# {series.name}: evenly reduced to {MAX_LITERAL_POINTS} points."
            )
        body += [f"xs{suffix} = np.array({xs})", f"ys{suffix} = np.array({ys})"]
        traces.append(
            f"go.Scatter(x=xs{suffix}, y=ys{suffix}, mode={mode!r}, "
            f"connectgaps=False, name={series.name!r})"
        )
    body += ["", f"fig = go.Figure([{', '.join(traces)}])"]
    body += _layout_lines(plan, options)
    if plan.axis_labels is not None:
        body.append(f"fig.update_xaxes(title_text={plan.axis_labels[0]!r})")
        body.append(f"fig.update_yaxes(title_text={plan.axis_labels[1]!r})")
    body.append("fig.show()")
    return _assemble(set(), body, sympy_needed=False)


def _distribution_code(plan: PlotPlan, options: RenderOptions) -> str:
    """Histograms and box plots. Measurements are literals — they came from
    outside, and there is no expression that would regenerate them."""
    body: list[str] = ["fig = go.Figure()"]
    for series in plan.series:
        values = series.sample.y[np.isfinite(series.sample.y)]
        # Counts, quartiles and outliers depend on every observation. The
        # line-plot downsampler cannot be reused here without changing the
        # statistic the figure reports.
        listed = _fmt_list(list(values))
        if plan.kind == "hist":
            body.append(
                f"fig.add_trace(go.Histogram(x=np.array({listed}), "
                f"name={series.name!r}, opacity=0.75))"
            )
        else:
            body.append(
                f"fig.add_trace(go.Box(y=np.array({listed}), "
                f"name={series.name!r}, boxmean=True))"
            )
    labels = plan.axis_labels or ("value", "count")
    layout = (
        f"template='plotly_white', showlegend={options.legend_visible(plan)}, "
        "barmode='overlay'"
    )
    if options.title is not None:
        layout += f", title={options.title!r}"
    body += [
        "",
        f"fig.update_layout({layout})",
        f"fig.update_xaxes(title_text={labels[0]!r})",
        f"fig.update_yaxes(title_text={labels[1]!r})",
    ]
    body += _view_window_lines(plan, options)
    body.append("fig.show()")
    return _assemble(set(), body, sympy_needed=False)


def _linalg_code(plan: PlotPlan, options: RenderOptions) -> str:
    """A 2x2 matrix as what it does to the plane.

    Emitted as the matrix product rather than as the resulting corners: `M @
    square` is the whole idea, and writing out the eight numbers it produces
    would hide it.
    """
    assert plan.matrix is not None
    rows = ", ".join(
        "[" + ", ".join(_fmt(v) for v in row) + "]" for row in plan.matrix
    )
    layout = (
        f"template='plotly_white', showlegend={options.legend_visible(plan)}"
    )
    if options.title is not None:
        layout += f", title={options.title!r}"
    body = [
        f"M = np.array([{rows}])",
        "square = np.array([[0.0, 1.0, 1.0, 0.0, 0.0],",
        "                   [0.0, 0.0, 1.0, 1.0, 0.0]])",
        "image = M @ square",
        "",
        "fig = go.Figure()",
        "fig.add_trace(go.Scatter(x=square[0], y=square[1], mode='lines',",
        f"                         name={plan.series[0].name!r},",
        "                         line=dict(dash='dot', width=2),",
        "                         fill='toself', opacity=0.35))",
        "fig.add_trace(go.Scatter(x=image[0], y=image[1], mode='lines',",
        f"                         name={plan.series[1].name!r},",
        "                         line=dict(width=3), fill='toself', opacity=0.35))",
    ]
    for value, vector in plan.eigen:
        body += [
            "fig.add_trace(go.Scatter(",
            f"    x=[0.0, {_fmt(float(vector[0] * value))}], "
            f"y=[0.0, {_fmt(float(vector[1] * value))}],",
            f"    mode='lines+markers', name={f'eigenvector (λ={value:.4g})'!r},",
            "    line=dict(width=2)))",
        ]
    body += [
        "",
        f"fig.update_layout({layout})",
        "fig.update_xaxes(title_text='x', zeroline=True)",
        "# Equal aspect, or the distortion you see is the plot's, not the matrix's.",
        "fig.update_yaxes(title_text='y', zeroline=True, scaleanchor='x', scaleratio=1)",
    ]
    body += _view_window_lines(plan, options)
    body.append("fig.show()")
    return _assemble(set(), body, sympy_needed=False)


def _array_literals(
    x: np.ndarray, y: np.ndarray, limit: int = MAX_LITERAL_POINTS
) -> tuple[str, str, bool]:
    """Both arrays as source text, evenly reduced if they are unwieldy.

    Reducing *evenly* rather than truncating matters: taking the first N points
    would silently emit a different picture from the one on screen.
    """
    if x.size > limit:
        index = np.unique(np.linspace(0, x.size - 1, limit).astype(int))
        return _fmt_list(list(x[index])), _fmt_list(list(y[index])), True
    return _fmt_list(list(x)), _fmt_list(list(y)), False


def _figure_line(
    plan: PlotPlan, options: RenderOptions, arrays: tuple[str, str], name: str
) -> str:
    mode = options.trace_mode(plan)
    return (
        f"fig = go.Figure(go.Scatter(x={arrays[0]}, y={arrays[1]}, mode={mode!r}, "
        f"connectgaps=False, name={name!r}))"
    )


# --------------------------------------------------------------------------
# shared layout emission
# --------------------------------------------------------------------------


def _view_window_lines(plan: PlotPlan, options: RenderOptions) -> list[str]:
    """The emitted twin of `plotly_backend._apply_view_window`."""
    lines: list[str] = []
    x_range = options.x_range(plan)
    if x_range is not None:
        lines.append(f"fig.update_xaxes(range={_fmt_list(list(x_range))})")
    y_range = options.y_range(plan)
    if y_range is not None:
        lines.append(f"fig.update_yaxes(range={_fmt_list(list(y_range))})")
    return lines


def _tick_density_lines(options: RenderOptions) -> list[str]:
    """The emitted twin of `plotly_backend._apply_tick_density`.

    ``ticks=`` is in ``REPRODUCED``, so leaving it out here would be the exact
    breach this module exists to prevent: a figure the reader is looking at and
    a program said to build it that disagree about what is on the axes.
    """
    lines: list[str] = []
    limit = options.tick_limit()
    if limit is not None:
        lines.append("# `nticks` is a ceiling, not a count: Plotly still picks")
        lines.append("# round numbers, it just stops before it passes this many.")
        for axis in ("x", "y"):
            lines.append(f"fig.update_{axis}axes(nticks={limit})")
    if not options.tick_labels_visible():
        for axis in ("x", "y"):
            lines.append(f"fig.update_{axis}axes(showticklabels=False)")
    return lines


def _scene_tick_kwargs(options: RenderOptions) -> str:
    """The same, as ``scene=dict(...)`` keywords for a 3D figure."""
    parts: list[str] = []
    limit = options.tick_limit()
    for axis in ("xaxis", "yaxis", "zaxis"):
        if limit is not None:
            parts.append(f"{axis}_nticks={limit}")
        if not options.tick_labels_visible():
            parts.append(f"{axis}_showticklabels=False")
    return "".join(f", {part}" for part in parts)


def _layout_lines(plan: PlotPlan, options: RenderOptions) -> list[str]:
    """Every figure property listed in ``render.options.REPRODUCED``."""
    layout = f"template='plotly_white', showlegend={options.legend_visible(plan)}"
    if options.title is not None:
        layout += f", title={options.title!r}"
    lines = [f"fig.update_layout({layout})"]

    x_range = options.x_range(plan)
    if x_range is not None:
        lines.append(f"fig.update_xaxes(range={_fmt_list(list(x_range))})")

    y_range = options.y_range(plan)
    if y_range is not None:
        lines.append(
            "# The 2nd-98th percentile window: without it a single pole would "
            "flatten the whole curve."
        )
        lines.append(f"fig.update_yaxes(range={_fmt_list(list(y_range))})")

    if options.log_y:
        lines.append("fig.update_yaxes(type='log')")

    ticks = pi_ticks_for(plan, options)
    if ticks is not None:
        lines.append(
            f"fig.update_xaxes(tickmode='array', tickvals={_fmt_list(list(ticks.values))}, "
            f"ticktext={list(ticks.text)!r})"
        )
    lines += _tick_density_lines(options)

    if plan.bands:
        how = "solved exactly" if plan.bands_exact else "found by sampling"
        lines += [
            f"# Where the inequality holds ({how}). A shape spans the whole y axis",
            "# however it is later rescaled, which is what an interval of x means.",
            f"for start, end in {_fmt_pairs(plan.bands)}:",
            f"    fig.add_vrect(x0=start, x1=end, fillcolor={REGION_FILL!r},",
            "                  line_width=0, layer='below')",
        ]
    return lines




def _assemble(needed: set[str], body: list[str], *, sympy_needed: bool = True) -> str:
    header = [
        "# Equivalent code — this runs exactly as printed.",
        _import_block(needed, sympy_needed=sympy_needed),
        "",
    ]
    return "\n".join(header + body).rstrip() + "\n"


def _inside(value: float, span: tuple[float, float]) -> bool:
    return span[0] < value < span[1]
