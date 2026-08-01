"""A whole HTML document does not belong in a notebook output cell.

``Worksheet.html()`` is a complete page: doctype, ``<head>``, a ``<style>``
carrying a ``body`` rule, and an embedded copy of Plotly. Returning it from
``_repr_html_`` did not sandbox any of that — the browser drops the outer tags
and keeps the stylesheet, so displaying a worksheet restyled the notebook
around it and wrote ~5 MB into the ``.ipynb`` every time.
"""

from __future__ import annotations

import pytest

from mathslate import analyze, plot, sin, table, x
from mathslate.classroom import Worksheet, worksheet
from mathslate.errors import UnsupportedInputError


@pytest.fixture()
def page() -> Worksheet:
    return worksheet(
        [
            "Some prose for the reader.",
            ("The graph", plot(sin(x), verbose=False)),
            ("The numbers", table(sin(x), (x, 0, 1), rows=5)),
            ("The properties", analyze(sin(x))),
        ],
        title="Waves",
    )


class TestTheNotebookRepresentation:
    def test_it_is_a_fragment_not_a_document(self, page: Worksheet) -> None:
        rendered = page._repr_html_()
        for forbidden in ("<!DOCTYPE", "<html", "<head", "<body"):
            assert forbidden not in rendered, f"{forbidden} leaked into the cell"

    def test_it_carries_no_page_wide_stylesheet(self, page: Worksheet) -> None:
        """A `body { ... }` rule in an output cell restyles the whole notebook."""
        assert "<style" not in page._repr_html_()

    def test_it_is_small(self, page: Worksheet) -> None:
        assert len(page._repr_html_()) < 4000

    def test_it_says_what_is_on_the_page(self, page: Worksheet) -> None:
        rendered = page._repr_html_()
        assert "Waves" in rendered
        assert "The graph" in rendered and "The numbers" in rendered
        assert ".save(" in rendered and ".preview(" in rendered

    def test_headings_are_escaped(self) -> None:
        page = worksheet([("<script>alert(1)</script>", "text")])
        assert "<script>" not in page._repr_html_()


class TestPreview:
    def test_it_is_an_iframe(self, page: Worksheet) -> None:
        assert page.preview().startswith("<iframe")

    def test_the_document_is_escaped_into_srcdoc(self, page: Worksheet) -> None:
        rendered = page.preview()
        assert "srcdoc=" in rendered
        assert "<!DOCTYPE" not in rendered[:400]
        assert "&lt;!DOCTYPE" in rendered

    def test_the_height_is_settable(self, page: Worksheet) -> None:
        assert "height:320px" in page.preview(height=320)

    def test_the_cdn_preview_does_not_embed_the_bundle(self, page: Worksheet) -> None:
        rendered = page.preview(standalone=False)
        assert "cdn.plot.ly" in rendered
        assert len(rendered) < 1_000_000


class TestTheFileIsUnchanged:
    def test_save_still_writes_a_whole_document(self, page: Worksheet, tmp_path) -> None:
        destination = page.save(tmp_path / "handout.html")
        written = destination.read_text(encoding="utf-8")
        assert written.startswith("<!DOCTYPE html>")
        assert "<style>" in written and "</html>" in written

    def test_plotly_is_embedded_exactly_once(self, tmp_path) -> None:
        page = worksheet(
            [plot(sin(x), verbose=False), plot(sin(2 * x), verbose=False)]
        )
        written = page.html()
        assert written.count("plotly.js v") <= 1
        assert len(written) > 1_000_000, "the bundle should be there for offline use"

    def test_the_cdn_variant_is_small_and_versioned(self, page: Worksheet) -> None:
        written = page.html(standalone=False)
        assert "cdn.plot.ly" in written
        assert "plotly-latest" not in written
        assert len(written) < 1_000_000

    def test_the_constructor_can_choose_cdn_for_html_and_save(
        self, tmp_path
    ) -> None:
        page = worksheet(
            [plot(sin(x), verbose=False)],
            standalone=False,
        )
        assert "cdn.plot.ly" in page.html()
        destination = page.save(tmp_path / "small.html")
        assert "cdn.plot.ly" in destination.read_text(encoding="utf-8")

    def test_a_save_call_can_override_the_page_default(self, page: Worksheet, tmp_path) -> None:
        destination = page.save(tmp_path / "small.html", standalone=False)
        written = destination.read_text(encoding="utf-8")
        assert "cdn.plot.ly" in written
        assert len(written) < 1_000_000

    def test_an_empty_worksheet_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="nothing to hand out"):
            Worksheet().html()

    def test_it_rejects_what_it_cannot_render(self) -> None:
        with pytest.raises(UnsupportedInputError, match="does not know what to do"):
            Worksheet().add(object())
