"""``examples/marimo_notebook.py`` must actually work when a reader opens it.

The tour notebooks have had this check for a while; the small standalone
example had none, and it was broken the whole time in two independent ways:

* One cell took ``dataset`` as a parameter that no cell returned. marimo
  resolves its dataflow graph from those signatures, so an unresolvable name
  does not raise — the cell is simply never scheduled. Half the notebook was
  dead and nothing said so.
* Another called ``f.show()``, which is Plotly's *script* behaviour: it opens a
  separate browser window. Inline, the cell rendered nothing.

Neither failure raises, which is exactly why they need asserting. Both are
invisible to any check that only imports the file.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "marimo_notebook.py"


def _graph() -> Any:
    pytest.importorskip("marimo", reason="marimo is an optional extra")
    from marimo._ast.app import InternalApp

    spec = importlib.util.spec_from_file_location("example_notebook", NOTEBOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return InternalApp(module.app).graph


class TestTheFileIsAMarimoNotebook:
    def test_it_exists(self) -> None:
        assert NOTEBOOK.is_file()

    def test_the_documented_command_names_this_file(self) -> None:
        text = NOTEBOOK.read_text(encoding="utf-8")
        assert "marimo edit examples/marimo_notebook.py" in text


class TestEveryCellCanActuallyRun:
    def test_no_cell_needs_a_name_that_no_cell_provides(self) -> None:
        """The original bug: an unresolved input silently disables the cell."""
        graph = _graph()
        provided = set(graph.definitions) | set(dir(builtins))
        dangling = {
            name
            for cell in graph.cells.values()
            for name in cell.refs
            if name not in provided
        }
        assert dangling == set(), (
            f"no cell returns {sorted(dangling)}, so marimo will never schedule "
            "the cells that ask for them — they fail silently, not loudly"
        )

    def test_no_name_is_defined_twice(self) -> None:
        graph = _graph()
        duplicates = {n: len(w) for n, w in graph.definitions.items() if len(w) > 1}
        assert duplicates == {}, f"marimo forbids defining a name twice: {duplicates}"

    def test_there_are_no_cycles(self) -> None:
        assert list(_graph().cycles) == []

    def test_no_cell_uses_a_star_import(self) -> None:
        """marimo rejects it at parse time, so the cell would never run."""
        for node in ast.walk(ast.parse(NOTEBOOK.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert "*" not in [alias.name for alias in node.names]


class TestFiguresRenderInline:
    def test_no_cell_calls_show(self) -> None:
        """`.show()` opens a browser window and leaves the cell blank."""
        offenders = [
            node.func.attr
            for node in ast.walk(ast.parse(NOTEBOOK.read_text(encoding="utf-8")))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "show"
        ]
        assert offenders == [], (
            "a notebook renders a figure by making it the cell's last "
            "expression; .show() opens a separate window instead"
        )

    def test_marimo_draws_the_plots(self, tmp_path: Path) -> None:
        """Running without raising is not the same as drawing something."""
        pytest.importorskip("marimo", reason="marimo is an optional extra")
        import subprocess

        source = tmp_path / NOTEBOOK.name
        destination = tmp_path / "rendered.html"
        shutil.copy2(NOTEBOOK, source)
        completed = subprocess.run(
            [
                sys.executable, "-m", "marimo", "export", "html",
                str(source), "-o", str(destination),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
        if completed.returncode != 0 or not destination.exists():
            pytest.fail(f"marimo could not render it:\n{completed.stderr[-2000:]}")
        drawn = destination.read_text(encoding="utf-8").lower().count("plotly-graph-div")
        assert drawn >= 2, (
            f"only {drawn} plots rendered — a reader would open this notebook "
            "and see prose with no figures"
        )
