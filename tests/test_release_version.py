"""scripts/release_version.py: the release tag must name the checkout's version."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import mathslate

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_version.py"


def _load():
    spec = importlib.util.spec_from_file_location("release_version", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_reads_the_same_version_the_package_reports() -> None:
    assert _load().checkout_version() == mathslate.__version__


def test_it_works_from_outside_the_checkout(tmp_path: Path) -> None:
    """The documented release command must not depend on the working directory."""
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == mathslate.__version__


def test_the_matching_tag_passes() -> None:
    assert _load().main(["--check-tag", f"v{mathslate.__version__}"]) == 0


@pytest.mark.parametrize("tag", ["v0.0.0", mathslate.__version__, "v" + mathslate.__version__ + ".post1"])
def test_any_other_tag_fails(tag: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert _load().main(["--check-tag", tag]) == 1
    assert "does not match" in capsys.readouterr().err
