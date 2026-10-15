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
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from .jupyter_env import SELECTORS as S
from .jupyter_env import jupyter_lab
from .scenario import Scenario, load

OUT_DIR = Path(__file__).resolve().parent / "out"
VIEWPORT = {"width": 1280, "height": 720}
CAPTION_H = 76

OVERLAY_SCRIPT = f"""
(() => {{
  if (window.top !== window) return;  // iframe(예: 10편에서 연 handout.html) 안에는 막대를 또 만들지 않는다
  const H = {CAPTION_H};
  const install = () => {{
    if (document.getElementById('__cap')) return;
    // 자막 막대는 앱 '아래' 별도 영역이다 — Jupyter Lab 상태 표시줄을 가리지 않게 Lab 메인 영역(#main)만 줄인다.
    // (body를 줄이거나 transform을 주면 Lab의 절대 배치가 깨진다.)
    const st = document.createElement('style');
    st.textContent = `#main{{height:calc(100% - ${{H}}px)!important}}`;
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
    elif action == "scroll":
        _cell(page, args["cell"]).scroll_into_view_if_needed()
        page.wait_for_timeout(500)
    elif action == "wait":
        page.wait_for_timeout(int(float(args) * 1000))
    else:
        raise NotImplementedError(f"아직 없는 동작: {action}")


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


def run_scenario(scenario: Scenario, headed: bool = False) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_dir = OUT_DIR / f".rec-{scenario.stem}"
    shutil.rmtree(video_dir, ignore_errors=True)
    exe = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or None
    print(f"▶ {scenario.stem} — {scenario.title}")
    with jupyter_lab() as (url, _work), sync_playwright() as pw:
        name = _create_notebook(url)
        browser = pw.chromium.launch(executable_path=exe, headless=not headed, args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        context = browser.new_context(viewport=VIEWPORT, record_video_dir=str(video_dir), record_video_size=VIEWPORT)
        context.add_init_script(OVERLAY_SCRIPT)
        page = context.new_page()
        t0 = time.time()
        cues: list[tuple[float, float, str]] = []
        try:
            root = url.split("/lab")[0]
            token = url.split("token=")[1].split("&")[0]
            page.goto(f"{root}/lab/tree/{name}?token={token}&reset")
            page.wait_for_selector(S["code_cell"], timeout=60_000)
            # 커널이 준비될 때까지(오른쪽 위 표시가 Idle) — 영상에서는 이 구간을 잘라낸다.
            page.wait_for_function("() => (document.querySelector('.jp-Notebook-ExecutionIndicator')?.title || '').includes('Idle') || /Idle/.test(document.body.innerText)", timeout=60_000)
            page.wait_for_timeout(1500)
            ready = time.time() - t0
            state = {"last_cell": 0, "base": (root, token)}
            for scene in scenario.scenes:
                start = time.time() - t0 - ready
                _set_caption(page, scene.caption)
                for step in scene.steps:
                    _do_step(page, step, state)
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
