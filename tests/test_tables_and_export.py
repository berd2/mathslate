"""PRD 11.6 step 4 — `table()` and single-file HTML export.

A graph shows shape; a table shows value. A learner checking a hand-worked
answer needs the second, and reaches for it exactly when the plot stops being
enough.

The export is the other half of the same milestone and answers to P3 in PRD §3:
a teacher who "distributes it as a single HTML file". That requirement is why
§11's slider is drawn from frames carried inside the figure rather than from a
widget needing a live kernel — the two decisions are the same decision, and
`TestTheFileReallyStandsAlone` is where it is checked rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sympy as sp

import mathslate as ms
from mathslate import Eq, cos, plot, sin, slider, sqrt, t, table, x, y
from mathslate.core.tables import DEFAULT_ROWS, Table
from mathslate.errors import UnsupportedInputError
from mathslate.ui import interact


@pytest.fixture(autouse=True)
def _clean_sliders():
    yield
    interact.release_all()


class TestTheValues:
    def test_the_columns_are_the_axis_and_the_expression(self) -> None:
        assert table(sin(x), (x, 0.0, 1.0), rows=3).headers() == ("x", "sin(x)")

    def test_the_rows_are_evenly_spaced_and_include_both_ends(self) -> None:
        values = table(sin(x), (x, 0.0, 1.0), rows=5).inputs
        assert values[0] == 0.0 and values[-1] == 1.0
        assert np.allclose(np.diff(values), 0.25)

    def test_the_values_are_right(self) -> None:
        found = table(sin(x), (x, 0.0, 1.0), rows=3)
        assert np.allclose(found.columns[0].values, np.sin([0.0, 0.5, 1.0]))

    def test_a_list_becomes_several_columns(self) -> None:
        found = table([sin(x), cos(x)], (x, 0.0, 1.0), rows=3)
        assert found.headers() == ("x", "sin(x)", "cos(x)")
        assert len(found.columns) == 2

    def test_the_default_row_count_is_odd(self) -> None:
        """So a symmetric range has a row exactly at its centre."""
        assert DEFAULT_ROWS % 2 == 1
        assert 0.0 in table(sin(x)).inputs


class TestPointsWithNoValue:
    def test_they_are_blank_not_zero(self) -> None:
        """A table that printed 0 there would be lying about the mathematics."""
        found = table(sqrt(x), (x, -1.0, 1.0), rows=5)
        assert not np.isfinite(found.columns[0].values[0])
        assert "—" in found.text()

    def test_and_the_table_says_how_many(self) -> None:
        found = table(sqrt(x), (x, -1.0, 1.0), rows=5)
        assert any("blank" in note for note in found.notes)

    def test_a_complete_column_says_nothing(self) -> None:
        assert table(sin(x), (x, 0.0, 1.0), rows=5).notes == ()


class TestItFollowsTheSameRules:
    def test_the_axis_is_chosen_by_convention(self) -> None:
        a = sp.Symbol("a", real=True)
        assert table(a * x, parameters={a: 2.0}, rows=3).symbol == x

    def test_an_explicit_range_wins(self) -> None:
        found = table(sin(x), (x, 2.0, 4.0), rows=3)
        assert found.inputs[0] == 2.0 and found.inputs[-1] == 4.0

    def test_a_free_parameter_is_refused(self) -> None:
        a = sp.Symbol("a", real=True)
        with pytest.raises(UnsupportedInputError, match="still free"):
            table(a * x, (x, 0.0, 1.0), rows=3)

    def test_too_few_rows_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="at least 2 rows"):
            table(sin(x), rows=1)


class TestFromAPlot:
    def test_it_tabulates_the_window_on_screen(self) -> None:
        found = plot(sin(x), (x, 0.0, 2.0), verbose=False).table(rows=3)
        assert found.inputs[0] == 0.0 and found.inputs[-1] == 2.0

    @pytest.mark.parametrize(
        ("build", "match"),
        [
            (lambda: plot((cos(t), sin(t)), verbose=False), "no single column"),
            (lambda: plot(x * y, verbose=False), "no single column"),
            (lambda: plot([1.0, 2.0, 3.0], verbose=False), "no expression"),
        ],
        ids=["parametric", "surface", "data"],
    )
    def test_what_has_no_column_of_values_says_so(self, build: object, match: str) -> None:
        with pytest.raises(UnsupportedInputError, match=match):
            build().table()  # type: ignore[operator]


class TestTheEscapeHatches:
    def test_numpy_gives_the_arrays(self) -> None:
        inputs, values = table([sin(x), cos(x)], (x, 0.0, 1.0), rows=4).numpy
        assert inputs.shape == (4,)
        assert values.shape == (4, 2)

    def test_sympy_gives_the_expressions(self) -> None:
        assert table(sin(x), (x, 0.0, 1.0), rows=3).sympy == sin(x)
        assert table([sin(x), cos(x)], (x, 0.0, 1.0), rows=3).sympy == (sin(x), cos(x))


class TestShowPython:
    def test_it_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        exec(compile(table(sin(x), (x, 0.0, 1.0), rows=4).python(), "<c>", "exec"), {})  # noqa: S102

    def test_it_prints_the_same_numbers(self, capsys: pytest.CaptureFixture[str]) -> None:
        found = table(sin(x), (x, 0.0, 1.0), rows=4)
        exec(compile(found.python(), "<c>", "exec"), {})  # noqa: S102
        printed = capsys.readouterr().out.split()
        assert float(printed[0]) == pytest.approx(0.0)
        assert float(printed[-1]) == pytest.approx(float(np.sin(1.0)), rel=1e-5)


class TestPresentation:
    def test_the_text_table_is_aligned(self) -> None:
        lines = table(sin(x), (x, 0.0, 1.0), rows=3).text().splitlines()
        assert set(lines[1]) <= {"-", " "}
        assert len({len(line.rstrip()) for line in lines[:2]}) == 1

    def test_it_renders_in_a_notebook(self) -> None:
        assert "<table>" in table(sin(x), (x, 0.0, 1.0), rows=3)._repr_html_()

    def test_the_repr_says_its_shape(self) -> None:
        assert "3 rows x 2" in repr(table([sin(x), cos(x)], (x, 0.0, 1.0), rows=3))


class TestTheFileReallyStandsAlone:
    """P3 hands this to a class; it has to open with no network and no install."""

    def test_nothing_is_fetched_from_outside(self, tmp_path: Path) -> None:
        import re

        page = plot(sin(x), verbose=False).to_html()
        assert re.search(r'<script[^>]*\ssrc="', page) is None

    def test_plotly_itself_is_embedded(self) -> None:
        assert len(plot(sin(x), verbose=False).to_html()) > 1_000_000

    def test_the_cdn_variant_is_small_and_says_where_it_points(self) -> None:
        page = plot(sin(x), verbose=False).to_html(standalone=False)
        assert len(page) < 1_000_000
        assert "cdn.plot.ly" in page

    def test_it_writes_the_file(self, tmp_path: Path) -> None:
        destination = tmp_path / "lesson.html"
        returned = plot(sin(x), verbose=False).to_html(destination)
        assert destination.exists()
        assert destination.read_text(encoding="utf-8") == returned

    def test_an_interactive_figure_keeps_its_slider(self, tmp_path: Path) -> None:
        """The reason §11 chose frames over a widget: a widget needs a kernel."""
        a = slider(-2, 2, default=0, name="export_a")
        page = plot(a * sin(x), verbose=False).to_html()
        assert "addFrames" in page
        import re

        assert re.search(r'<script[^>]*\ssrc="', page) is None

    def test_a_surface_exports_too(self) -> None:
        assert len(plot(x * y, verbose=False).to_html()) > 1_000_000
