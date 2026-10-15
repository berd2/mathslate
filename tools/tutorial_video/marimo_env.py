"""격리된 marimo 서버(`marimo edit --headless`). 빈 노트북 파일 하나를 임시 폴더에 만들어 연다. 계획서 §10 21편."""

from __future__ import annotations

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

BLANK = '''import marimo

app = marimo.App()


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
'''


@contextmanager
def marimo_edit() -> Iterator[str]:
    root = Path(tempfile.mkdtemp(prefix="mathslate-video-marimo-"))
    nb = root / "notebook.py"
    nb.write_text(BLANK, encoding="utf-8")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = (root / "server.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "marimo", "edit", str(nb), "--headless", "--no-token", "--port", str(port), "--host", "127.0.0.1"],
        cwd=root, stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "XDG_CONFIG_HOME": str(root / "cfg"), "MARIMO_SKIP_UPDATE_CHECK": "1"},
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 60
        while True:
            try:
                urllib.request.urlopen(base, timeout=2).read(10)
                break
            except Exception:
                if proc.poll() is not None or time.time() > deadline:
                    raise RuntimeError(f"marimo가 뜨지 않았다:\n{(root / 'server.log').read_text()[-1500:]}")
                time.sleep(0.5)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        shutil.rmtree(root, ignore_errors=True)
