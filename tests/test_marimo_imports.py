"""marimo forbids `import *`, so the documented marimo import line must work.

marimo builds its reactive dataflow graph by reading, statically, which names
each cell defines. `import *` makes that set unknowable without running the
cell, so marimo rejects it at parse time — the cell never runs. That is
marimo's rule, not MathSlate's, and there is no flag to turn it off.

PRD §6.2 requires marimo to be a supported frontend and §9 decision 3 permits
`import *`, so the two only coexist if an explicit import line is documented
and kept working. This module pins the workaround to the documents that
promise it: the import line is *read out of the docs*, not restated here, so
the two cannot drift apart.
"""

from __future__ import annotations

import ast
import builtins
import re
from pathlib import Path

import pytest

pytest.importorskip("marimo", reason="marimo is an optional extra")

from marimo._ast.errors import ImportStarError  # noqa: E402
from marimo._ast.visitor import ScopedVisitor  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"
_FENCE = re.compile(r"^```python[^\n]*\n(?P<body>.*?)^```$", re.MULTILINE | re.DOTALL)


def _documented_import(document: str) -> str:
    """The explicit-import block a marimo user is told to write."""
    # The tutorial nests it in a blockquote; strip the quote markers first.
    text = "\n".join(
        line[2:] if line.startswith("> ") else line.rstrip(">")
        for line in (DOCS / document).read_text(encoding="utf-8").splitlines()
    )
    blocks = [m.group("body") for m in _FENCE.finditer(text) if "from mathslate import (" in m.group("body")]
    assert blocks, f"{document} does not show a marimo-safe import"
    return blocks[0]


def _defines(source: str) -> set[str]:
    visitor = ScopedVisitor()
    visitor.visit(ast.parse(source))
    return set(visitor.defs)


def test_star_import_is_rejected_by_marimo() -> None:
    """The behaviour the user hits. If this ever stops raising, relax the docs."""
    with pytest.raises(ImportStarError, match=r"import \*"):
        _defines("from mathslate import *")


def test_both_documents_explain_it() -> None:
    for document in ("tutorial.md", "manual.md"):
        text = (DOCS / document).read_text(encoding="utf-8")
        assert "not allowed in marimo" in text, f"{document} does not warn about it"


def test_the_two_documents_agree_on_the_workaround() -> None:
    assert _documented_import("tutorial.md") == _documented_import("manual.md")


@pytest.mark.parametrize("document", ["tutorial.md", "manual.md"])
def test_the_documented_import_is_accepted_by_marimo(document: str) -> None:
    defined = _defines(_documented_import(document))
    assert {"plot", "sin", "x"} <= defined


@pytest.mark.parametrize("document", ["tutorial.md", "manual.md"])
def test_the_documented_import_actually_imports(document: str) -> None:
    """Accepted by marimo is not enough; the names have to exist."""
    namespace: dict[str, object] = {}
    exec(_documented_import(document), namespace)  # noqa: S102
    for name in ("plot", "polar", "sin", "x", "theta", "integrate"):
        assert name in namespace, name


def _names_needed(document: str) -> set[str]:
    """Every name the document's examples use without defining first.

    Extracted, not listed. A hand-kept list is a second copy of the truth and
    goes stale the moment an example gains a call — which is exactly what
    happened when the tutorial started using `analyze()`: the import line did
    not carry it, and the restated list did not notice.
    """
    text = (DOCS / document).read_text(encoding="utf-8")
    provided: set[str] = set(dir(builtins))
    needed: set[str] = set()
    for match in _FENCE.finditer(text):
        body = match.group("body")
        if "from mathslate import (" in body:
            continue  # the workaround itself, not an example
        try:
            tree = ast.parse(body)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in provided:
                    needed.add(node.id)
            elif isinstance(node, ast.Name):
                provided.add(node.id)
            elif isinstance(node, ast.alias):
                provided.add((node.asname or node.name).split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                provided.add(node.name)
            elif isinstance(node, ast.arg):
                provided.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                provided.add(node.name)  # `except E as error` binds a str, not a Name
            elif isinstance(node, ast.comprehension) and isinstance(node.target, ast.Name):
                provided.add(node.target.id)
    return needed - provided


@pytest.mark.parametrize("document", ["tutorial.md", "manual.md"])
def test_the_documented_import_covers_the_documents_own_examples(document: str) -> None:
    """A marimo reader must be able to run the document with that one line."""
    namespace: dict[str, object] = {}
    exec(_documented_import(document), namespace)  # noqa: S102
    missing = sorted(name for name in _names_needed(document) if name not in namespace)
    assert missing == [], f"the documented marimo import omits {missing}"


def test_the_namespaced_form_is_accepted_too() -> None:
    assert _defines("import mathslate as ms") == {"ms"}
