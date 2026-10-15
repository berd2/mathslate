"""20편(JupyterLite)용 사이트를 로컬에서 빌드한다 — `.github/workflows/deploy-lite.yml`의 빌드 단계와 같다.

    python -m tools.tutorial_video.build_lite          # → tools/tutorial_video/out/lite-site

jupyterlite-pyodide-kernel 0.8은 Python 3.12 이상을 요구해서, 3.12 인터프리터(`python3.12`)로 별도 가상환경(`out/.lite-venv`)을 만든다.
Pyodide 런타임·패키지는 실행할 때 CDN에서 내려받는다(사이트에 들어 있지 않다) — 인터넷이 필요하다.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
SITE = OUT / "lite-site"
VENV = OUT / ".lite-venv"


def _run(cmd: list[str], **kw) -> None:
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    py312 = shutil.which("python3.12")
    if not py312:
        print("python3.12가 필요하다(jupyterlite-pyodide-kernel 0.8 요구).")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    if not (VENV / "bin" / "jupyter").exists():
        _run([py312, "-m", "venv", str(VENV)])
        _run([str(VENV / "bin" / "pip"), "install", "-q", "-r", str(ROOT / "lite" / "requirements.txt")])
    with tempfile.TemporaryDirectory(prefix="mathslate-lite-") as tmp:
        work = Path(tmp) / "lite"
        shutil.copytree(ROOT / "lite", work)
        (work / "files").mkdir()
        (work / "wheels").mkdir()
        _run([str(VENV / "bin" / "python"), "-m", "build", "--wheel", "--outdir", str(work / "wheels")], cwd=ROOT)
        shutil.copy(ROOT / "examples" / "mathslate_tour.ipynb", work / "files")
        shutil.copy(ROOT / "lite" / "mathslate_playground.ipynb", work / "files")
        wheel = next((work / "wheels").glob("mathslate-*.whl"))
        shutil.rmtree(SITE, ignore_errors=True)
        _run(
            [str(VENV / "bin" / "jupyter"), "lite", "build", "--contents", "files", "--output-dir", str(SITE), "--piplite-wheels", str(wheel.relative_to(work))],
            cwd=work, env={"PATH": f"{VENV / 'bin'}:/usr/bin:/bin", "HOME": tmp},
        )
    print(f"✓ {SITE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
