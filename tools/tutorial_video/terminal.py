"""터미널 장면(00편). 계획서 §9-6.

실제 명령을 pty에서 돌려 출력(바이트)과 시각을 기록해 두고(`out/captures/*.json`), 영상에서는 그 기록을 xterm.js 터미널에 **빨리 감아** 재생한다.
설치처럼 몇 분 걸리는 명령은 가속 재생이 정직한 표현이다 — 출력 내용은 실제 명령이 낸 그대로이고 시간만 줄인다.
xterm.js는 jsDelivr(npm)에서 한 번 받아 `out/.cache/`에 둔다.
"""

from __future__ import annotations

import base64
import json
import os
import pty
import select
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
CAPTURES = OUT / "captures"
CACHE = OUT / ".cache"
COLS, ROWS = 92, 20
XTERM = "5.3.0"
ASSETS = {
    "xterm.js": f"https://cdn.jsdelivr.net/npm/xterm@{XTERM}/lib/xterm.js",
    "xterm.css": f"https://cdn.jsdelivr.net/npm/xterm@{XTERM}/css/xterm.css",
}


def capture(name: str, cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None, timeout: float = 900) -> dict:  # noqa: PLR0913
    """cmd를 pty에서 실행해 [(시각, base64 바이트)]를 기록한다. 이미 기록이 있으면 그것을 쓴다."""
    CAPTURES.mkdir(parents=True, exist_ok=True)
    path = CAPTURES / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    master, slave = pty.openpty()
    import fcntl, struct, termios

    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    full_env = {**os.environ, "TERM": "xterm-256color", "COLUMNS": str(COLS), "LINES": str(ROWS), **(env or {})}
    proc = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave, env=full_env, cwd=cwd, close_fds=True)
    os.close(slave)
    events: list[list] = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        r, _, _ = select.select([master], [], [], 0.2)
        if r:
            try:
                data = os.read(master, 4096)
            except OSError:
                break
            if not data:
                break
            events.append([round(time.time() - t0, 3), base64.b64encode(data).decode()])
        elif proc.poll() is not None:
            break
    if proc.poll() is None:  # 끝나지 않는 명령(서버)은 기록 시간이 다 되면 끊는다
        proc.terminate()
    proc.wait(timeout=30)
    os.close(master)
    result = {"cmd": cmd, "returncode": proc.returncode, "duration": round(time.time() - t0, 1), "events": events}
    path.write_text(json.dumps(result))
    return result


def asset(name: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"xterm-{XTERM}-{name}"
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(ASSETS[name], timeout=60).read())
    return path.read_text(encoding="utf-8")


def page_html() -> str:
    """영상 화면 높이(720 - 자막 막대 76)에 꽉 차는 xterm 터미널 페이지."""
    return f"""<!doctype html><meta charset="utf-8"><style>{asset('xterm.css')}
html,body{{margin:0;height:100%;background:#0b1020}}
#t{{position:absolute;left:24px;top:16px;right:24px;bottom:92px}}
</style><div id="t"></div><script>{asset('xterm.js')}</script>
<script>
const term = new Terminal({{cols:{COLS}, rows:{ROWS}, fontSize:22, fontFamily:'DejaVu Sans Mono, monospace', cursorBlink:true,
  theme:{{background:'#0b1020', foreground:'#e5e7eb'}}}});
term.open(document.getElementById('t'));
window.term = term;
window.feed = b64 => term.write(Uint8Array.from(atob(b64), c => c.charCodeAt(0)));
</script>"""


def sanitize_url(text: str) -> str:
    """화면에 띄울 로그에서 임시 토큰·경로를 가린다."""
    import re

    text = re.sub(r"token=[0-9a-f]+|token=video", "token=…", text)
    return text


if __name__ == "__main__":
    print("터미널 기록 도우미 — 러너가 호출한다.")
    sys.exit(0)
