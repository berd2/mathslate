"""Every MathSlate object must draw itself in every host, not just in Jupyter.

`plot(tan(x))` in marimo printed its inference report and drew nothing. The
cause was a single delegated hook: `PlotResult` offered `_repr_mimebundle_`
and Plotly returns `{}` from that unless a Jupyter kernel has switched on the
mimetype renderer. In a kernel it works; in marimo it does not, and the suite
only ever asked the kernel.

So this module asks each host's own formatter, rather than assuming that one
working representation implies the others.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from mathslate import (
    Eq,
    Matrix,
    analyze,
    cos,
    dataset,
    plot,
    polar,
    sin,
    slider,
    t,
    table,
    tan,
    x,
    y,
    z,
)
from mathslate.classroom import worksheet
from mathslate.ui import release_all

#: Everything a reader can put at the end of a cell.
DISPLAYABLE: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("curve", lambda: plot(tan(x), verbose=False)),
    ("curves", lambda: plot([sin(x), cos(x)], verbose=False)),
    ("parametric", lambda: plot((cos(t), sin(t)), verbose=False)),
    ("polar", lambda: polar(1 + cos(t), verbose=False)),
    ("data", lambda: plot([1.0, 2.0, 3.0], verbose=False)),
    ("hist", lambda: plot([1.0, 2.0, 2.0, 3.0] * 6, kind="hist", verbose=False)),
    ("surface", lambda: plot(x * y, verbose=False)),
    ("contour", lambda: plot(x * y, kind="contour", verbose=False)),
    ("implicit", lambda: plot(Eq(x**2 + y**2, 4), verbose=False)),
    ("space", lambda: plot((cos(t), sin(t), t), verbose=False)),
    ("psurface", lambda: plot((cos(t) * sin(z), sin(t) * sin(z), cos(z)), verbose=False)),
    ("linalg", lambda: plot(Matrix([[2, 1], [1, 3]]), verbose=False)),
    ("analysis", lambda: analyze(x**3 - 3 * x)),
    ("table", lambda: table(sin(x), (x, 0, 1), rows=4)),
    ("dataset", lambda: dataset({"p": [1.0, 2.0, 3.0], "q": [3.0, 4.0, 5.0]})),
    ("worksheet", lambda: worksheet([("A graph", plot(sin(x), verbose=False))])),
)
IDS = [label for label, _ in DISPLAYABLE]

#: Below this, whatever came back is a stub rather than a rendering.
MINIMUM_USEFUL = 200


@pytest.fixture()
def animated() -> Any:
    release_all()
    control = slider(-2, 2, default=1, name="amp")
    result = plot(control * sin(x), verbose=False)
    yield result
    release_all()


class TestMarimoDrawsThem:
    """marimo prefers `text/html` and has no formatter registered for our types."""

    @pytest.fixture(scope="class")
    def formatter(self) -> Any:
        pytest.importorskip("marimo")
        from marimo._output.formatters.formatters import register_formatters
        from marimo._output.formatting import get_formatter

        register_formatters()
        return get_formatter

    @pytest.mark.parametrize(("label", "build"), DISPLAYABLE, ids=IDS)
    def test_it_produces_a_rendering(
        self, label: str, build: Callable[[], Any], formatter: Any
    ) -> None:
        obj = build()
        found = formatter(obj)
        assert found is not None, f"marimo has no way to display a {label}"
        mime, data = found(obj)
        assert len(str(data)) > MINIMUM_USEFUL, (
            f"marimo rendered a {label} as {len(str(data))} bytes of {mime} — "
            "that is a stub, not a picture"
        )

    def test_an_animated_plot_carries_its_frames(
        self, animated: Any, formatter: Any
    ) -> None:
        """A slider is frames plus a control, and both must reach the page.

        Plotly emits them as `Plotly.addFrames(...)` and a `sliders` layout
        entry rather than under a literal "frames" key, so those are what to
        look for.
        """
        _mime, rendered = formatter(animated)(animated)
        assert len(animated.plotly.frames) > 1
        for token in ("Plotly.newPlot", "addFrames", "sliders", "steps"):
            assert token in str(rendered), f"the rendering lost {token}"

    def test_a_plot_renders_as_html_not_an_empty_mimebundle(
        self, formatter: Any
    ) -> None:
        """The exact defect: `{}` from `_repr_mimebundle_` outside a kernel."""
        result = plot(tan(x), verbose=False)
        mime, data = formatter(result)(result)
        assert mime == "text/html"
        assert len(str(data)) > 1000


class TestTheHooksMatchTheFigureTheyWrap:
    """A result must display wherever its figure would — so it offers the same hooks."""

    def test_plot_results_offer_every_hook_the_figure_does(self) -> None:
        result = plot(sin(x), verbose=False)
        for hook in ("_repr_html_", "_repr_mimebundle_"):
            assert hasattr(result.plotly, hook), f"plotly dropped {hook}"
            assert hasattr(result, hook), f"PlotResult does not delegate {hook}"

    def test_the_html_is_the_figures_own(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Compared by delegation, not by string: Plotly mints a fresh div id
        on every call, so two renderings of one figure never match."""
        result = plot(sin(x), verbose=False)
        monkeypatch.setattr(
            result.plotly, "_repr_html_", lambda: "<div>the figure spoke</div>"
        )
        assert result._repr_html_() == "<div>the figure spoke</div>"

    def test_the_html_is_self_contained_enough_to_draw(self) -> None:
        rendered = plot(tan(x), verbose=False)._repr_html_()
        assert "<div" in rendered and "Plotly" in rendered


class TestJupyterStillWorks:
    """The path that already worked must keep working."""

    def test_the_mimebundle_is_still_delegated(self) -> None:
        result = plot(sin(x), verbose=False)
        assert result._repr_mimebundle_() == result.plotly._repr_mimebundle_()

    @pytest.mark.parametrize(("label", "build"), DISPLAYABLE, ids=IDS)
    def test_ipython_finds_a_representation(
        self, label: str, build: Callable[[], Any]
    ) -> None:
        """IPython looks for the same hooks; at least one must answer usefully."""
        obj = build()
        answers = [
            getattr(obj, hook)()
            for hook in ("_repr_html_", "_repr_mimebundle_")
            if hasattr(obj, hook)
        ]
        assert any(len(str(answer)) > MINIMUM_USEFUL for answer in answers), (
            f"a {label} has no useful representation for a notebook"
        )
