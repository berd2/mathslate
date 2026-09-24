"""The manuals must not lie.

Every ``python`` code block in ``docs/`` is executed, in document order, in one
namespace per document — exactly as a reader following along would experience
it. A block that is *meant* to fail says so in its fence::

    ```python raises=AmbiguousAxisError
    plot(a*b)
    ```

Blocks that are illustrative rather than runnable use a non-``python`` fence
(``text``, ``console``, ``bash``). A block that needs an optional extra names
the module, and is skipped where it is not installed::

    ```python requires=ipywidgets
    assistant("plot the tangent")
    ```
"""

from __future__ import annotations

import builtins
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest

import mathslate
from mathslate import errors

DOCS = Path(__file__).resolve().parents[1] / "docs"
_FENCE = re.compile(r"^```(?P<info>[^\n]*)\n(?P<body>.*?)^```$", re.MULTILINE | re.DOTALL)


@dataclass(frozen=True)
class Block:
    document: str
    index: int
    line: int
    code: str
    raises: str | None
    requires: str | None = None

    @property
    def label(self) -> str:
        return f"{self.document}:{self.line}"


def _blocks(path: Path) -> Iterator[Block]:
    text = path.read_text(encoding="utf-8")
    for index, match in enumerate(_FENCE.finditer(text)):
        info = match.group("info").strip()
        if not info.split(" ")[0] == "python":
            continue
        raises = requires = None
        for token in info.split(" ")[1:]:
            key, _, value = token.partition("=")
            if key == "raises":
                raises = value
            elif key == "requires":
                requires = value
        yield Block(
            document=path.name,
            index=index,
            line=text[: match.start()].count("\n") + 1,
            code=match.group("body"),
            raises=raises,
            requires=requires,
        )


def _documents() -> list[Path]:
    return sorted(DOCS.glob("*.md")) if DOCS.is_dir() else []


def test_the_docs_directory_exists() -> None:
    assert _documents(), "no documentation found in docs/"


@pytest.mark.parametrize("path", _documents(), ids=lambda p: p.name)
def test_every_example_runs(path: Path, headless_show: list[Any]) -> None:
    """One namespace per document, blocks executed in the order they are read."""
    namespace: dict[str, Any] = {}
    exec("from mathslate import *", namespace)  # noqa: S102
    mathslate.set_verbose(False)
    try:
        for block in _blocks(path):
            _run(block, namespace)
    finally:
        mathslate.set_verbose(True)


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _run(block: Block, namespace: dict[str, Any]) -> None:
    if block.requires and not _installed(block.requires):
        return  # an optional extra this environment does not have
    compiled = compile(block.code, f"<{block.label}>", "exec")
    if block.raises is None:
        try:
            exec(compiled, namespace)  # noqa: S102
        except Exception as error:  # noqa: BLE001 - report where, not just what
            raise AssertionError(
                f"example at {block.label} failed: {type(error).__name__}: {error}"
                f"\n---\n{block.code}---"
            ) from error
        return

    expected = getattr(errors, block.raises, None) or getattr(builtins, block.raises, None)
    assert isinstance(expected, type) and issubclass(expected, BaseException), (
        f"{block.label}: unknown error name {block.raises!r}"
    )
    with pytest.raises(expected):
        exec(compiled, namespace)  # noqa: S102


@pytest.mark.parametrize("path", _documents(), ids=lambda p: p.name)
def test_every_document_has_examples(path: Path) -> None:
    assert list(_blocks(path)), f"{path.name} contains no runnable examples"
