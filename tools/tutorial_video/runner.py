"""튜토리얼 영상 러너 — 시나리오(YAML) 하나를 격리된 Jupyter Lab에서 실행해 .webm 과 .ko.srt 를 만든다. 계획서 §7.

    .venv-video/bin/python -m tools.tutorial_video.runner tools/tutorial_video/scenarios/01_first_plot.yml
    .venv-video/bin/python -m tools.tutorial_video.runner tools/tutorial_video/scenarios/*.yml

Chromium 경로는 PLAYWRIGHT_CHROMIUM_EXECUTABLE (없으면 Playwright 기본 탐색). 화면은 1280×720이다 — Jupyter Lab의 글자가
작아 보이지 않도록 1920×1080 영상의 150% 확대와 같은 효과를 낸다(계획서 §6).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from .jupyter_env import SELECTORS as S
from .jupyter_env import jupyter_lab
from .marimo_env import marimo_edit
from . import terminal as term
from .scenario import Scenario, load

OUT_DIR = Path(__file__).resolve().parent / "out"
VIEWPORT = {"width": 1280, "height": 720}
CAPTION_H = 76

def overlay_script(shrink_css: str | None = None) -> str:
    """자막 막대·가짜 커서를 심는 초기화 스크립트. shrink_css는 앱 영역을 막대 높이만큼 줄이는 CSS(Lab은 #main, 기본값)."""
    shrink_css = shrink_css if shrink_css is not None else f"#main{{height:calc(100% - {CAPTION_H}px)!important}}"
    return f"""
(() => {{
  if (window.top !== window) return;  // iframe(예: 10편에서 연 handout.html) 안에는 막대를 또 만들지 않는다
  const H = {CAPTION_H};
  const install = () => {{
    if (document.getElementById('__cap')) return;
    // 자막 막대는 앱 '아래' 별도 영역이다 — Jupyter Lab 상태 표시줄을 가리지 않게 Lab 메인 영역(#main)만 줄인다.
    // (body를 줄이거나 transform을 주면 Lab의 절대 배치가 깨진다.)
    const st = document.createElement('style');
    st.textContent = `{shrink_css}`;
    document.head.append(st);
    const bar = document.createElement('div');
    bar.id = '__cap';
    bar.style.cssText = `position:fixed;left:0;right:0;bottom:0;height:${{H}}px;display:flex;align-items:center;justify-content:center;`
      + 'background:#111827;color:#fff;font:600 26px/1.3 "Noto Sans CJK KR","Noto Sans KR","Malgun Gothic",sans-serif;text-align:center;padding:0 24px;z-index:2147483647';
    const cur = document.createElement('div');
    cur.id = '__cur';
    cur.style.cssText = 'position:fixed;width:22px;height:22px;border-radius:50%;background:rgba(56,189,248,.55);border:2px solid #0ea5e9;pointer-events:none;z-index:2147483647;left:-40px;top:-40px;transform:translate(-50%,-50%)';
    document.body.append(bar, cur);
    window.addEventListener('mousemove', e => {{ cur.style.left = e.clientX + 'px'; cur.style.top = e.clientY + 'px'; }}, true);
    window.dispatchEvent(new Event('resize'));
  }};
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install); else install();
}})();
"""


OVERLAY_SCRIPT = overlay_script()


def _ffmpeg() -> str:
    for candidate in (os.environ.get("FFMPEG"), "ffmpeg"):
        if candidate and shutil.which(candidate):
            return candidate
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError("디코더가 있는 ffmpeg가 필요하다: FFMPEG 환경변수, PATH, 또는 pip install imageio-ffmpeg") from error


def _srt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _set_caption(page: Page, text: str) -> None:
    page.evaluate("t => { const b = document.getElementById('__cap'); if (b) b.textContent = t; }", text)


def _move_cursor(page: Page, x: float, y: float) -> None:
    page.mouse.move(x, y, steps=18)


def _cell(page: Page, index: int):
    return page.locator(S["code_cell"]).nth(index)


def _ensure_cell(page: Page, index: int) -> None:
    """index번째 셀이 있게 한다. 마지막 셀을 실행하면 Lab이 새 셀을 만들어 주므로 보통은 이미 있다."""
    while page.locator(S["code_cell"]).count() <= index:
        page.keyboard.press("Escape")
        page.keyboard.press("b")
        page.wait_for_timeout(200)


def _do_cell(page: Page, args: dict) -> None:
    index = args.get("index", 0)
    _ensure_cell(page, index)
    editor = _cell(page, index).locator(S["input_editor"])
    editor.scroll_into_view_if_needed()
    box = editor.bounding_box()
    assert box, "셀 입력칸의 위치를 알 수 없다"
    _move_cursor(page, box["x"] + min(box["width"] / 2, 300), box["y"] + box["height"] / 2)
    editor.click()
    delay = args.get("type_delay_ms", 45)
    page.keyboard.type(args["code"], delay=delay)
    page.wait_for_timeout(400)


def _prompt_done(page: Page, index: int, timeout_ms: int) -> None:
    prompt = _cell(page, index).locator(".jp-InputPrompt")
    page.wait_for_function(
        "el => /\\[\\d+\\]/.test(el.textContent || '')", arg=prompt.element_handle(), timeout=timeout_ms
    )


def _do_run(page: Page, args: dict, state: dict) -> None:
    index = args.get("index", state["last_cell"])
    timeout_ms = int(args.get("timeout", 60) * 1000)
    page.keyboard.press("Shift+Enter")
    wait_start = state["clock"]()
    _prompt_done(page, index, timeout_ms)
    cell = _cell(page, index)
    wait = args.get("wait_output", "any")
    if wait == "figure" or args.get("expect_figure"):
        cell.locator(S["plotly"]).first.wait_for(timeout=timeout_ms)
    if "expect_text" in args:
        page.wait_for_function(
            "([el, t]) => (el.textContent || '').includes(t)",
            arg=[cell.locator(S["output"]).first.element_handle(), args["expect_text"]],
            timeout=timeout_ms,
        )
    if "expect_error" in args:
        # 실행이 끝난 직후에는 오류 출력이 아직 그려지기 전일 수 있다 — 나타날 때까지 기다린다.
        try:
            page.wait_for_function(
                "([el, t]) => (el.textContent || '').includes(t)",
                arg=[cell.locator(S["output"]).first.element_handle(), args["expect_error"]],
                timeout=10_000,
            )
        except Exception as error:  # noqa: BLE001
            text = cell.locator(S["output"]).first.inner_text()
            raise AssertionError(f"셀 {index}의 출력에 {args['expect_error']!r}가 없다:\n{text[:400]}") from error
    # 가속 대상: 실행 키를 누른 뒤 출력이 나올 때까지의 대기(compose.py가 이 구간을 빠르게 돌린다).
    state["waits"].append((wait_start, state["clock"]()))
    # 실행한 셀을 화면 위쪽에 맞춘다 — 입력 코드가 맨 위, 출력(그림)이 그 아래로 보인다.
    cell.evaluate("el => el.scrollIntoView({block: 'start'})")
    page.wait_for_timeout(int(args.get("settle_ms", 800)))
    state["last_cell"] = index + 1


def _do_highlight(page: Page, args: dict, state: dict) -> None:
    index = args.get("cell", state["last_cell"] - 1)
    cell = _cell(page, index)
    seconds = args.get("seconds", 2.5)
    if "output_line" in args:
        found = cell.locator(S["output"]).first.element_handle().evaluate(
            """(root, needle) => {
              const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
              for (let n = walker.nextNode(); n; n = walker.nextNode()) {
                const i = (n.textContent || '').indexOf(needle);
                if (i < 0) continue;
                const r = document.createRange(); r.setStart(n, i); r.setEnd(n, i + needle.length);
                const b = r.getBoundingClientRect();
                const d = document.createElement('div'); d.className = '__hl';
                d.style.cssText = `position:fixed;left:${b.left - 4}px;top:${b.top - 3}px;width:${b.width + 8}px;height:${b.height + 6}px;border:3px solid #f59e0b;border-radius:4px;pointer-events:none;z-index:2147483646`;
                document.body.append(d);
                return true;
              }
              return false;
            }""",
            args["output_line"],
        )
        if not found:
            raise AssertionError(f"셀 {index}의 출력에 {args['output_line']!r}가 없다")
    else:
        target = cell.locator(args.get("selector", S["output"])).first
        target.evaluate("el => { el.dataset.hlold = el.style.outline; el.style.outline = '3px solid #f59e0b'; el.style.outlineOffset = '2px'; }")
    page.wait_for_timeout(int(seconds * 1000))
    page.evaluate("() => document.querySelectorAll('.__hl').forEach(e => e.remove())")
    if "output_line" not in args:
        target.evaluate("el => { el.style.outline = el.dataset.hlold || ''; }")


def _do_drag_slider(page: Page, args: dict, state: dict) -> None:
    """Plotly 슬라이더의 손잡이를 레일의 `to`(0~1) 위치까지 마우스로 끈다. 구간을 잘게 나눠 천천히 움직인다."""
    cell = _cell(page, args.get("cell", state["last_cell"] - 1))
    grip = cell.locator(".slider-grip-rect").first
    rail = cell.locator(".slider-rail-rect").first
    grip.scroll_into_view_if_needed()
    g, r = grip.bounding_box(), rail.bounding_box()
    assert g and r, "슬라이더 손잡이를 찾지 못했다"
    gx, gy = g["x"] + g["width"] / 2, g["y"] + g["height"] / 2
    _move_cursor(page, gx, gy)
    page.mouse.down()
    steps = max(10, int(args.get("seconds", 3) * 25))
    target = r["x"] + r["width"] * args["to"]
    for i in range(1, steps + 1):
        page.mouse.move(gx + (target - gx) * i / steps, gy)
        page.wait_for_timeout(35)
    page.mouse.up()
    page.wait_for_timeout(600)


def _do_click(page: Page, args: dict, state: dict) -> None:
    cell = _cell(page, args.get("cell", state["last_cell"] - 1))
    target = cell.locator(args["selector"]).first
    target.scroll_into_view_if_needed()
    box = target.bounding_box()
    assert box, f"{args['selector']}의 위치를 알 수 없다"
    _move_cursor(page, box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    target.click()
    page.wait_for_timeout(int(args.get("settle_ms", 500)))


def _do_new_tab(page: Page, args: dict, base: str) -> None:
    """만든 HTML 파일을 '새 탭'처럼 보여 준다. Playwright는 탭마다 따로 녹화하므로 같은 화면 위에 탭 막대+iframe 오버레이를 띄운다."""
    root, token = base
    url = f"{root}/files/{args['file']}?token={token}"
    page.evaluate(
        """([url, title]) => {
          const o = document.createElement('div'); o.id = '__tab';
          o.style.cssText = 'position:fixed;left:0;right:0;top:0;bottom:0;padding-bottom:76px;box-sizing:border-box;background:#dee1e6;z-index:2147483645;display:flex;flex-direction:column';
          const bar = document.createElement('div');
          bar.style.cssText = 'height:40px;display:flex;align-items:flex-end;padding:0 8px;gap:8px;font:14px sans-serif';
          bar.innerHTML = `<div style="background:#fff;padding:8px 16px;border-radius:8px 8px 0 0;max-width:260px;white-space:nowrap">📄 ${title}</div>`
            + `<div style="flex:1;background:#fff;margin:0 0 4px;padding:6px 12px;border-radius:14px;color:#555;white-space:nowrap;overflow:hidden">file:///${title}</div>`;
          const f = document.createElement('iframe'); f.src = url; f.style.cssText = 'flex:1;border:0;background:#fff';
          o.append(bar, f); document.body.append(o);
        }""",
        [url, args["file"]],
    )
    frame = page.frame_locator("#__tab iframe")
    frame.locator(S["plotly"]).first.wait_for(timeout=30_000)
    page.wait_for_timeout(1200)
    if args.get("hover", True):
        box = page.locator("#__tab iframe").bounding_box()
        assert box
        y = box["y"] + box["height"] * 0.55
        _move_cursor(page, box["x"] + box["width"] * 0.25, y)
        for i in range(30):
            page.mouse.move(box["x"] + box["width"] * (0.25 + 0.4 * i / 30), y)
            page.wait_for_timeout(60)
    page.wait_for_timeout(int(args.get("seconds", 3) * 1000))
    page.evaluate("() => document.getElementById('__tab')?.remove()")
    page.wait_for_timeout(500)


def _term_write(page: Page, data: bytes) -> None:
    import base64

    page.evaluate("b => window.feed(b)", base64.b64encode(data).decode())


def _term_type(page: Page, text: str, delay_ms: int = 55) -> None:
    for ch in text:
        _term_write(page, ch.encode())
        page.wait_for_timeout(delay_ms)
    page.wait_for_timeout(350)
    _term_write(page, b"\r\n")


def _real_capture(name: str) -> dict:
    """명령 기록을 만든다(이미 있으면 재사용). 설치는 새 가상환경에서 실제 `pip install "mathslate[starter]"`를 돌린다."""
    if name == "pip_install":
        tmp = Path(tempfile.mkdtemp(prefix="mathslate-pip-"))
        try:
            subprocess.run([sys.executable, "-m", "venv", str(tmp / "v")], check=True)
            return term.capture(name, [str(tmp / "v" / "bin" / "pip"), "install", "--no-cache-dir", "mathslate[starter]"], env={"PIP_DISABLE_PIP_VERSION_CHECK": "1"}, timeout=1200)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if name == "jupyter_lab":
        tmp = Path(tempfile.mkdtemp(prefix="mathslate-lab-"))
        try:
            env = {"JUPYTER_CONFIG_DIR": str(tmp / "c"), "JUPYTER_DATA_DIR": str(tmp / "d"), "JUPYTER_RUNTIME_DIR": str(tmp / "r")}
            return term.capture(name, [sys.executable, "-m", "jupyterlab", "--no-browser", "--allow-root", f"--ServerApp.root_dir={tmp}"], env=env, timeout=12)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    raise KeyError(name)


def _do_terminal(page: Page, args: dict) -> None:
    """`terminal: {type, capture, target_seconds, keep: [정규식…]}` — 명령을 치고 실제 기록을 가속 재생한다."""
    cap = _real_capture(args["capture"])
    if args.get("prompt", True):
        _term_write(page, b"\x1b[1;32m$\x1b[0m ")
    _term_type(page, args["type"])
    events = cap["events"]
    if "keep" in args:  # 서버 로그처럼 내용이 긴 출력은 보여 줄 줄만 남긴다(내용은 실제 출력 그대로, 임시 경로·토큰은 가린다)
        import base64

        text = b"".join(base64.b64decode(b) for _, b in events).decode("utf-8", "replace")
        ansi = re.compile(r"\x1b\[[0-9;]*m")
        lines = [
            term.sanitize_url(ln)
            for ln in text.replace("\r", "").split("\n")
            if any(re.search(k, ansi.sub("", ln)) for k in args["keep"])  # 색 코드를 뺀 글자로 고르고, 출력은 색이 든 원본으로
        ]
        for ln in lines:
            _term_write(page, (ln.rstrip() + "\r\n").encode())
            page.wait_for_timeout(120)
    else:
        import base64

        # 기록 시각을 target_seconds에 맞게 줄이고(긴 공백은 0.4초로 접는다), 0.1초 간격 묶음으로 한꺼번에 쓴다.
        factor = max(1.0, cap["duration"] / args.get("target_seconds", 12))
        scaled, prev, clock = [], 0.0, 0.0
        for t, b in events:
            clock += min((t - prev) / factor, 0.4)
            prev = t
            scaled.append((clock, base64.b64decode(b)))
        batches: dict[int, bytes] = {}
        for clock, data in scaled:
            batches[int(clock / 0.1)] = batches.get(int(clock / 0.1), b"") + data
        last = 0
        for slot in sorted(batches):
            page.wait_for_timeout(max(0, (slot - last) * 100))
            last = slot
            _term_write(page, batches[slot])
    _term_write(page, b"\x1b[1;32m$\x1b[0m ")
    page.wait_for_timeout(int(args.get("hold", 2.0) * 1000))


def _do_browser(page: Page, state: dict) -> None:
    """터미널 화면에서 브라우저(Jupyter Lab)로 넘어간다."""
    page.goto(state["lab_url"])
    page.wait_for_selector(S["code_cell"], timeout=60_000)
    page.wait_for_function("() => /Idle/.test(document.body.innerText)", timeout=60_000)
    page.wait_for_timeout(1000)


def _do_step(page: Page, step: dict, state: dict) -> None:
    (action, args), = step.items()
    if action == "cell":
        _do_cell(page, args)
        state["last_cell"] = args.get("index", 0)
    elif action == "run":
        _do_run(page, args or {}, state)
    elif action == "highlight":
        _do_highlight(page, args, state)
    elif action == "drag_slider":
        _do_drag_slider(page, args, state)
    elif action == "click":
        _do_click(page, args, state)
    elif action == "new_tab":
        _do_new_tab(page, args, state["base"])
    elif action == "terminal":
        _do_terminal(page, args)
    elif action == "browser":
        _do_browser(page, state)
    elif action == "scroll":
        _cell(page, args["cell"]).scroll_into_view_if_needed()
        page.wait_for_timeout(500)
    elif action == "wait":
        page.wait_for_timeout(int(float(args) * 1000))
    else:
        raise NotImplementedError(f"아직 없는 동작: {action}")


def _marimo_editor(page: Page, index: int):
    return page.locator(".cm-content").nth(index)


def _do_marimo_step(page: Page, step: dict, state: dict) -> None:
    """marimo 편집 화면용 동작. 셀 입력칸은 `.cm-content`, 그림은 `.js-plotly-plot`(마지막 식이 그대로 출력된다)."""
    (action, args), = step.items()
    if action == "cell":
        editor = _marimo_editor(page, args.get("index", 0))
        editor.scroll_into_view_if_needed()
        box = editor.bounding_box()
        assert box, "셀 입력칸의 위치를 알 수 없다"
        _move_cursor(page, box["x"] + min(box["width"] / 2, 300), box["y"] + box["height"] / 2)
        editor.click()
        if args.get("replace"):
            page.keyboard.press("Control+a")
            page.keyboard.press("Delete")
        page.keyboard.type(args["code"], delay=args.get("type_delay_ms", 45))
        page.wait_for_timeout(400)
    elif action == "run":
        timeout_ms = int(args.get("timeout", 60) * 1000)
        wait_start = state["clock"]()
        page.keyboard.press("Shift+Enter")
        if args.get("wait_output") == "figure":
            # marimo는 HTML 출력을 iframe 안에 그린다.
            page.frame_locator("iframe").last.locator(S["plotly"]).first.wait_for(timeout=timeout_ms)
        if "expect_text" in args:
            page.get_by_text(args["expect_text"]).first.wait_for(timeout=timeout_ms)
        page.wait_for_timeout(int(args.get("settle_ms", 1000)))
        state["waits"].append((wait_start, state["clock"]()))
    elif action == "highlight":
        target = page.get_by_text(args["text"]).first if "text" in args else page.locator(args["selector"]).first
        target.evaluate("el => { el.dataset.hlold = el.style.outline; el.style.outline = '3px solid #f59e0b'; el.style.outlineOffset = '3px'; }")
        page.wait_for_timeout(int(args.get("seconds", 2.5) * 1000))
        target.evaluate("el => { el.style.outline = el.dataset.hlold || ''; }")
    elif action == "wait":
        page.wait_for_timeout(int(float(args) * 1000))
    else:
        raise NotImplementedError(f"marimo에서 아직 없는 동작: {action}")


def _create_notebook(base_url: str) -> str:
    """서버 API로 빈 노트북(코드 셀 하나, 커널 지정)을 만든다 — 영상에는 안 찍힌다. kernelspec이 없으면 Lab이 커널 선택 창을 띄운다."""
    token = base_url.split("token=")[1].split("&")[0]
    root = base_url.split("/lab")[0]
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ""}],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    name = "notebook.ipynb"
    body = json.dumps({"type": "notebook", "format": "json", "content": notebook}).encode()
    request = urllib.request.Request(
        f"{root}/api/contents/{name}?token={token}", data=body, method="PUT", headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(request).read()
    return name


def _trim_start(webm: Path, seconds: float) -> bool:
    """앞 seconds초(준비 구간)를 잘라낸다. 디코더가 있는 ffmpeg가 필요하다."""
    tmp = webm.with_suffix(".trim.webm")
    result = subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{seconds:.3f}", "-i", str(webm), "-c:v", "libvpx", "-b:v", "2M", "-an", str(tmp)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ! 앞부분을 자르지 못했다(원본 유지): {result.stderr[-200:]}")
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(webm)
    return True


@contextmanager
def _static_site(site: Path):
    """JupyterLite 사이트를 `python -m http.server`로 연다."""
    import socket

    if not (site / "lab" / "index.html").exists():
        raise RuntimeError(f"JupyterLite 사이트가 없다: {site}\n먼저 python -m tools.tutorial_video.build_lite")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(site)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1).read(10)
                break
            except Exception:
                time.sleep(0.25)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()


def _fetch_externally(route) -> None:
    """Pyodide 런타임·패키지는 CDN에서 오는데, Chromium의 워커 모듈 로드는 이 환경의 프록시 인증서 오류를 무시하지 못한다.
    그래서 외부 요청을 Playwright의 네트워크(오류 무시)로 대신 가져와 채워 준다. 일반 사용자 환경에서는 필요 없다."""
    try:
        route.fulfill(response=route.fetch(timeout=180_000))
    except Exception:  # noqa: BLE001
        route.abort()


def run_scenario(scenario: Scenario, headed: bool = False) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_dir = OUT_DIR / f".rec-{scenario.stem}"
    shutil.rmtree(video_dir, ignore_errors=True)
    exe = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None
    print(f"▶ {scenario.stem} — {scenario.title}")
    marimo = scenario.env == "marimo"
    lite = scenario.env == "jupyterlite"
    terminal = scenario.env == "terminal"
    with ExitStack() as stack:
        if lite:
            base = stack.enter_context(_static_site(Path(os.environ.get("MATHSLATE_LITE_SITE", OUT_DIR / "lite-site"))))
            overlay = OVERLAY_SCRIPT
        elif marimo:
            base = stack.enter_context(marimo_edit())
            overlay = overlay_script("")  # marimo는 화면이 스크롤되는 문서라 앱 영역을 줄이지 않고 막대를 겹쳐 둔다
        else:  # jupyterlab, terminal(마지막에 Lab으로 넘어간다)
            url = stack.enter_context(jupyter_lab())[0]
            name = _create_notebook(url)
            overlay = OVERLAY_SCRIPT
        pw = stack.enter_context(sync_playwright())
        browser = pw.chromium.launch(executable_path=exe, headless=not headed, args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        # marimo는 Plotly를 CDN에서 불러온다. 이 환경의 프록시 인증서를 Chromium이 믿지 않아 그림이 안 그려지므로 인증서 오류를 무시한다(일반 사용자 환경에서는 필요 없다).
        context = browser.new_context(viewport=VIEWPORT, record_video_dir=str(video_dir), record_video_size=VIEWPORT, ignore_https_errors=True)
        context.add_init_script(overlay)
        if lite:
            context.route(lambda u: u.startswith("http") and not u.startswith("http://127.0.0.1"), _fetch_externally)
        page = context.new_page()
        t0 = time.time()
        cues: list[tuple[float, float, str]] = []
        try:
            if terminal:
                # 시작은 터미널 화면. 브라우저(Lab)로 넘어가는 `browser` 단계에서 Lab 주소로 이동한다.
                page.set_content(term.page_html())
                page.wait_for_function("() => !!window.term", timeout=30_000)
                page.evaluate(overlay)  # set_content가 문서를 갈아 끼워 초기화 스크립트가 만든 자막 막대가 사라진다 — 다시 심는다
                page.wait_for_timeout(800)
                root = url.split("/lab")[0]
                token = url.split("token=")[1].split("&")[0]
            elif lite:
                page.goto(f"{base}/lab/index.html?reset")
                page.wait_for_selector(".jp-LauncherCard", timeout=90_000)
                page.locator('.jp-LauncherCard[data-category="Notebook"]').first.click()
                page.wait_for_selector(S["code_cell"], timeout=90_000)
                # Pyodide 커널이 뜰 때까지 — 영상에서는 이 구간을 잘라낸다.
                page.wait_for_function("() => /Idle/.test(document.body.innerText)", timeout=300_000)
                page.wait_for_timeout(1500)
                root, token = base, ""
            elif marimo:
                page.goto(base)
                page.wait_for_selector(".cm-content", timeout=60_000)
                page.wait_for_timeout(1500)
                root, token = base, ""
            else:
                root = url.split("/lab")[0]
                token = url.split("token=")[1].split("&")[0]
                page.goto(f"{root}/lab/tree/{name}?token={token}&reset")
                page.wait_for_selector(S["code_cell"], timeout=60_000)
                # 커널이 준비될 때까지(오른쪽 위 표시가 Idle) — 영상에서는 이 구간을 잘라낸다.
                page.wait_for_function("() => (document.querySelector('.jp-Notebook-ExecutionIndicator')?.title || '').includes('Idle') || /Idle/.test(document.body.innerText)", timeout=60_000)
                page.wait_for_timeout(1500)
            ready = time.time() - t0
            state = {"last_cell": 0, "base": (root, token), "waits": [], "clock": lambda: time.time() - t0 - ready}
            if terminal:
                state["lab_url"] = f"{root}/lab/tree/{name}?token={token}&reset"
            do_step = _do_marimo_step if marimo else _do_step
            for scene in scenario.scenes:
                start = time.time() - t0 - ready
                _set_caption(page, scene.caption)
                for step in scene.steps:
                    do_step(page, step, state)
                cues.append((start, time.time() - t0 - ready, scene.caption))
        except Exception:
            page.screenshot(path=str(OUT_DIR / f"{scenario.stem}.fail.png"))
            raise
        finally:
            context.close()
            browser.close()
    recorded = next(video_dir.glob("*.webm"))
    target = OUT_DIR / f"{scenario.stem}.webm"
    shutil.move(str(recorded), target)
    shutil.rmtree(video_dir, ignore_errors=True)
    _trim_start(target, ready)
    srt = "\n".join(f"{i}\n{_srt_time(a)} --> {_srt_time(b)}\n{text}\n" for i, (a, b, text) in enumerate(cues, 1))
    (OUT_DIR / f"{scenario.stem}.ko.srt").write_text(srt, encoding="utf-8")
    (OUT_DIR / f"{scenario.stem}.waits.json").write_text(json.dumps({"waits": state["waits"]}), encoding="utf-8")
    print(f"  ✓ {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", nargs="+", type=Path)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args(argv)
    failed = 0
    for path in args.scenarios:
        try:
            run_scenario(load(path), headed=args.headed)
        except Exception as error:  # noqa: BLE001 - 시나리오 하나의 실패가 나머지를 막지 않게
            failed += 1
            print(f"  ✗ [{path.stem}] {type(error).__name__}: {error}")
    if failed:
        print(f"✗ {failed}개 시나리오 실패.")
        return 2
    print("✓ 전체 완료.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
