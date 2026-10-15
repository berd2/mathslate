"""격리된 Jupyter Lab 서버. 사용자의 설정·확장·최근 파일을 건드리지 않도록 설정·데이터·런타임 디렉터리를 임시 폴더로 돌린다.

셀렉터는 `SELECTORS`에 모은다 — Jupyter Lab 버전이 바뀌어 DOM이 달라지면 여기만 고친다(계획서 §8).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SELECTORS = {
    "notebook": ".jp-Notebook",
    "cell": ".jp-Notebook .jp-Cell",
    "code_cell": ".jp-Notebook .jp-CodeCell",
    "input_editor": ".jp-InputArea-editor .cm-content",
    "output": ".jp-OutputArea",
    "output_text": ".jp-OutputArea-output",
    "plotly": ".js-plotly-plot",
    "stream": ".jp-RenderedText",
}

# 새 설정 디렉터리에서도 첫 화면이 깨끗하도록 끈다: 공지·뉴스 알림, 자동 완성 팝업(계획서 §9-2·7).
USER_SETTINGS = {
    "@jupyterlab/apputils-extension:notification": {"fetchNews": "false", "doNotDisturbMode": True},
    "@jupyterlab/apputils-extension:themes": {"theme": "JupyterLab Light"},
    "@jupyterlab/completer-extension:manager": {"autoCompletion": False},
    "@jupyterlab/extensionmanager-extension:plugin": {"enabled": False},
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def jupyter_lab(workdir: Path | None = None, files: dict[str, str] | None = None) -> Iterator[tuple[str, Path]]:
    """Jupyter Lab을 띄우고 (기본 URL, 작업 폴더)를 돌려준다. 끝나면 서버와 임시 폴더를 지운다."""
    root = Path(tempfile.mkdtemp(prefix="mathslate-video-"))
    work = workdir or (root / "work")
    work.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {}).items():
        (work / name).write_text(content, encoding="utf-8")
    config, data, runtime = root / "config", root / "data", root / "runtime"
    for d in (config, data, runtime):
        d.mkdir()
    settings_dir = config / "lab" / "user-settings"
    for plugin, values in USER_SETTINGS.items():
        pkg, name = plugin.split(":")
        target = settings_dir / pkg
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{name}.jupyterlab-settings").write_text(json.dumps(values), encoding="utf-8")
    port = _free_port()
    token = "video"
    env = {
        **os.environ,
        "JUPYTER_CONFIG_DIR": str(config),
        "JUPYTER_DATA_DIR": str(data),
        "JUPYTER_RUNTIME_DIR": str(runtime),
        "IPYTHONDIR": str(root / "ipython"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    log = (root / "server.log").open("w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "jupyterlab", "--no-browser", f"--port={port}", "--ServerApp.ip=127.0.0.1",
            f"--ServerApp.token={token}", f"--ServerApp.root_dir={work}", "--ServerApp.open_browser=False", "--allow-root",
            "--LabApp.news_url=", "--LabApp.check_for_updates_class=jupyterlab.NeverCheckForUpdate",
        ],
        env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 60
        while True:
            try:
                urllib.request.urlopen(f"{base}/api?token={token}", timeout=2).read()
                break
            except Exception:
                if proc.poll() is not None or time.time() > deadline:
                    raise RuntimeError(f"Jupyter Lab이 뜨지 않았다:\n{(root / 'server.log').read_text()[-2000:]}")
                time.sleep(0.5)
        yield f"{base}/lab?token={token}&reset", work
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        shutil.rmtree(root, ignore_errors=True)
