"""PRD v0.1 acceptance criterion 5 — identical results in all three notebooks.

The same script (``examples/quickstart.py``) is executed three ways:

* as plain Python              — the Colab/script baseline
* inside a Jupyter kernel      — via ``nbclient``, which is what Colab runs too
* inside marimo               — via marimo's own script runner

and the fingerprints must match exactly. The notebook environments are
optional extras, so each test skips cleanly when its host is not installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = PROJECT_ROOT / "examples" / "quickstart.py"


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    """The fingerprint produced by importing the module in this interpreter."""
    from examples.quickstart import FINGERPRINT

    return FINGERPRINT


def _requires(module: str) -> None:
    pytest.importorskip(module, reason=f"{module} is an optional extra")


def test_the_quickstart_runs_as_a_plain_script(baseline: dict[str, Any]) -> None:
    completed = subprocess.run(
        [sys.executable, str(QUICKSTART)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=PROJECT_ROOT,
        check=True,
    )
    assert json.loads(completed.stdout) == baseline


def test_jupyter_agrees(baseline: dict[str, Any]) -> None:
    _requires("nbclient")
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r})"),
            nbformat.v4.new_code_cell(
                "import json\n"
                "from examples.quickstart import FINGERPRINT\n"
                "print(json.dumps(FINGERPRINT, sort_keys=True))"
            ),
        ]
    )
    NotebookClient(notebook, timeout=600, kernel_name="python3").execute()
    printed = _stdout_of(notebook.cells[-1])
    assert json.loads(printed) == baseline


def test_marimo_agrees(baseline: dict[str, Any]) -> None:
    _requires("marimo")
    completed = subprocess.run(
        [sys.executable, "-m", "marimo", "run", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=PROJECT_ROOT,
    )
    if completed.returncode != 0:
        pytest.skip("the marimo CLI is not runnable in this environment")

    # marimo notebooks are ordinary Python modules; executing one under the
    # marimo import machinery is what proves nothing in MathSlate needs a
    # different code path there.
    script = (
        "import json, marimo\n"
        f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from examples.quickstart import FINGERPRINT\n"
        "assert marimo.running_in_notebook() is False\n"
        "print(json.dumps(FINGERPRINT, sort_keys=True))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=PROJECT_ROOT,
        check=True,
    )
    assert json.loads(result.stdout) == baseline


def test_the_frontend_adapter_sees_each_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection must not need the frontend to be the *running* one."""
    from mathslate.ui import Frontend, detect_frontend

    assert detect_frontend() is Frontend.PLAIN

    stub = type(sys)("google.colab")
    monkeypatch.setitem(sys.modules, "google.colab", stub)
    assert detect_frontend() is Frontend.COLAB


def test_the_frontend_adapter_recognizes_jupyterlite(monkeypatch: pytest.MonkeyPatch) -> None:
    from mathslate.ui import Frontend, detect_frontend

    monkeypatch.setattr(sys, "platform", "emscripten")
    assert detect_frontend() is Frontend.JUPYTER


def test_jupyterlite_report_gives_a_piplite_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mathslate.ui import adapters, range_controls

    monkeypatch.setattr(sys, "platform", "emscripten")
    monkeypatch.setattr(range_controls, "deps_available", lambda: False)
    monkeypatch.setattr(adapters, "_available", lambda _name: False)
    assert "await piplite.install" in adapters.frontend_report()


def test_jupyterlite_missing_widgets_warns_and_returns_a_plain_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mathslate import plot, sin, x

    monkeypatch.setattr(sys, "platform", "emscripten")
    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    graph = plot(sin(x), verbose=False)
    with pytest.warns(RuntimeWarning, match="showing the regular Plotly graph"):
        assert graph.range_controls() is graph.plotly


def _stdout_of(cell: Any) -> str:
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream" and output.get("name") == "stdout":
            return str(output["text"])
    raise AssertionError(f"the notebook cell printed nothing: {cell.get('outputs')}")
