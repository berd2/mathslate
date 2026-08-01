"""``worksheet()`` — several things on one page you can hand out (PRD 7, v1.0).

PRD §3's third persona "builds interactive classroom material and distributes
it as a single HTML file". §11.10 already does that for one figure; this does
it for a lesson — plots, tables, analyses and prose, in the order you wrote
them, in one file.

**Self-contained by construction.** Plotly is embedded once, no matter how many
figures the page holds, so the file opens with no network and nothing
installed. That is also why a MathSlate slider is built from frames carried
inside the figure (§11.8): a worksheet with a slider on it still works after
being emailed, and a notebook widget would not.

The output is a plain HTML document with no JavaScript of its own beyond what
Plotly needs. Nothing here collects anything, phones anywhere, or grades
anybody — §2.2 rules out grading and LMS features, and a handout is not a
loophole in that.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Sequence

from .core.analysis import Analysis
from .core.tables import Table
from .errors import UnsupportedInputError
from .result import PlotResult

__all__ = ["Worksheet", "worksheet", "Item"]

#: Anything a worksheet knows how to put on the page.
Item = PlotResult | Table | Analysis | str

_STYLE = """\
:root { color-scheme: light dark; }
body { font: 16px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       max-width: 52rem; margin: 2rem auto; padding: 0 1rem; }
h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.15rem; margin: 2rem 0 0.4rem; }
.meta { opacity: 0.6; font-size: 0.9rem; margin-bottom: 2rem; }
.item { margin: 1rem 0 2rem; }
table { border-collapse: collapse; }
th, td { border-bottom: 1px solid rgba(128,128,128,0.3); padding: 3px 10px; }
pre { overflow-x: auto; padding: 8px; background: rgba(128,128,128,0.12);
      border-radius: 4px; }
details > summary { cursor: pointer; }
@media print { .item { break-inside: avoid; } }
"""


@dataclass
class Worksheet:
    """A page under construction. Add items in the order they should appear."""

    title: str = "MathSlate worksheet"
    subtitle: str = ""
    items: list[tuple[str, object]] = field(default_factory=list)
    #: Embed Plotly for an offline handout; False links the versioned CDN.
    standalone: bool = True

    def add(self, item: object, heading: str = "") -> "Worksheet":
        """Append one plot, table, analysis or paragraph. Returns self."""
        if not isinstance(item, (PlotResult, Table, Analysis, str)):
            raise UnsupportedInputError(
                f"a worksheet holds plots, tables, analyses and text; it does "
                f"not know what to do with {type(item).__name__}."
            )
        self.items.append((heading, item))
        return self

    def html(self, *, standalone: bool | None = None) -> str:
        """The whole page, offline by default or linked to Plotly's CDN."""
        if not self.items:
            raise UnsupportedInputError("an empty worksheet has nothing to hand out.")

        blocks: list[str] = []
        offline = self.standalone if standalone is None else standalone
        # Plotly is ~4 MB. Included with the first figure and referenced by
        # every one after it, so a ten-plot worksheet is not a 40 MB file.
        first_figure = True
        for heading, item in self.items:
            include_plotlyjs: bool | Literal["cdn"] = (
                (True if offline else "cdn") if first_figure else False
            )
            body, used_plotly = _render(item, include_plotlyjs=include_plotlyjs)
            first_figure = first_figure and not used_plotly
            head = f"<h2>{html.escape(heading)}</h2>" if heading else ""
            blocks.append(f"<section class='item'>{head}{body}</section>")

        subtitle = (
            f"<div class='meta'>{html.escape(self.subtitle)}</div>"
            if self.subtitle
            else f"<div class='meta'>{date.today().isoformat()}</div>"
        )
        return (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            "<meta charset='utf-8'>\n"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
            f"<title>{html.escape(self.title)}</title>\n"
            f"<style>\n{_STYLE}</style>\n</head>\n<body>\n"
            f"<h1>{html.escape(self.title)}</h1>\n{subtitle}\n"
            + "\n".join(blocks)
            + "\n</body>\n</html>\n"
        )

    def save(self, path: str | Path, *, standalone: bool | None = None) -> Path:
        """Write the page and return where it went."""
        destination = Path(path)
        destination.write_text(self.html(standalone=standalone), encoding="utf-8")
        return destination

    def preview(
        self, height: int = 640, *, standalone: bool | None = None
    ) -> str:
        """The page in an ``<iframe>``, for looking at it inside a notebook.

        :meth:`html` is a whole document — doctype, ``<head>``, a ``<style>``
        with a ``body`` rule, and a 4 MB copy of Plotly. Handing that to a
        notebook's output cell does not sandbox it: the browser drops the outer
        tags and keeps the stylesheet, so the worksheet's page styling silently
        restyles the notebook around it, and the Plotly bundle is written into
        the ``.ipynb`` on every save. An iframe is a document boundary, which
        is exactly what a whole document needs.
        """
        srcdoc = html.escape(self.html(standalone=standalone), quote=True)
        return (
            f"<iframe srcdoc=\"{srcdoc}\" style='width:100%;height:{int(height)}px;"
            "border:1px solid rgba(128,128,128,0.3);border-radius:4px'"
            " sandbox='allow-scripts'></iframe>"
        )

    def _repr_html_(self) -> str:
        """A short card, not the page.

        Rendering the whole handout on every ``repr`` would put a Plotly bundle
        into the notebook each time the object is echoed. Call :meth:`preview`
        to actually look at it, or :meth:`save` to hand it out.
        """
        headings = "".join(
            f"<li>{html.escape(heading) if heading else '<em>(no heading)</em>'} "
            f"<span style='opacity:0.6'>— {type(item).__name__}</span></li>"
            for heading, item in self.items
        )
        return (
            f"<div style='padding:8px'><strong>{html.escape(self.title)}</strong>"
            f"<span style='opacity:0.6'> — worksheet, {len(self.items)} items</span>"
            f"<ol style='margin:6px 0'>{headings}</ol>"
            "<div style='opacity:0.6;font-size:0.9em'>"
            ".preview() to see it &nbsp;·&nbsp; .save(path) to hand it out</div></div>"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"<Worksheet {self.title!r}: {len(self.items)} items>"


def _render(
    item: object, *, include_plotlyjs: bool | Literal["cdn"]
) -> tuple[str, bool]:
    """``(html, whether this item carried the Plotly bundle)``."""
    if isinstance(item, PlotResult):
        return (
            item.plotly.to_html(
                include_plotlyjs=include_plotlyjs,
                full_html=False,
                auto_play=False,
            ),
            True,
        )
    if isinstance(item, (Table, Analysis)):
        return item._repr_html_(), False
    if isinstance(item, str):
        return _paragraphs(item), False
    raise UnsupportedInputError(f"cannot render {type(item).__name__}.")


def _paragraphs(text: str) -> str:
    """Blank-line-separated text, escaped. No Markdown, deliberately.

    A worksheet is written in a Python string beside the mathematics; adding a
    Markdown dependency to style it would cost more than it returns, and
    escaping everything means a teacher's ``<`` is a less-than sign.
    """
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "".join(f"<p>{html.escape(part)}</p>" for part in parts)


def worksheet(
    items: Sequence[object] | None = None,
    title: str = "MathSlate worksheet",
    subtitle: str = "",
    path: str | Path | None = None,
    *,
    standalone: bool = True,
) -> Worksheet:
    """Build a handout from plots, tables, analyses and text.

    Pass ``(heading, item)`` pairs to caption a section, or bare items for no
    heading. With ``path``, the file is written before the worksheet returns.

    Examples
    --------
    >>> from mathslate import plot, table, sin, x                 # doctest: +SKIP
    >>> from mathslate.classroom import worksheet                 # doctest: +SKIP
    >>> page = worksheet([                                        # doctest: +SKIP
    ...     "Where does sin(x)/x go at zero?",
    ...     ("The graph", plot(sin(x)/x, verbose=False)),
    ...     ("The numbers", table(sin(x)/x, (x, -1, 1), rows=9)),
    ... ], title="Limits", path="handout.html")
    """
    page = Worksheet(title=title, subtitle=subtitle, standalone=standalone)
    for entry in items or ():
        if (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], str)
        ):
            page.add(entry[1], heading=entry[0])
        else:
            page.add(entry)
    if path is not None:
        page.save(path)
    return page
