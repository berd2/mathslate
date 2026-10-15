"""합성 — out/NN_이름.webm 을 mp4 로 바꾸고, 커널·그림이 나오길 기다리는 구간을 가속한다. 계획서 §6.

    .venv-video/bin/python -m tools.tutorial_video.compose            # out/ 의 모든 영상
    .venv-video/bin/python -m tools.tutorial_video.compose 01 05      # 이름에 01·05가 든 것만

러너가 남긴 `.waits.json`(실행 키를 누른 뒤 출력이 나올 때까지의 구간)이 `MIN_WAIT` 초 넘으면, 그 구간의 첫 `KEEP`초는 그대로 두고
나머지를 `SPEEDUP`배속으로 돌린다. 자막(.ko.srt)은 줄어든 시간에 맞춰 다시 쓴다. 영상에 구워진 자막 막대는 그대로다.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from .runner import OUT_DIR, _ffmpeg, _srt_time

SPEEDUP = 6.0
MIN_WAIT = 2.0  # 이보다 짧은 대기는 그대로 둔다
KEEP = 0.8  # 대기 구간 앞부분은 원래 속도로 보여 준다(누르는 순간이 보이도록)


def _parse_srt(path: Path) -> list[tuple[float, float, str]]:
    def t(x: str) -> float:
        m = re.match(r"(\d+):(\d+):(\d+),(\d+)", x)
        assert m
        return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000

    out = []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = block.split("\n")
        a, b = lines[1].split(" --> ")
        out.append((t(a), t(b), " ".join(lines[2:])))
    return out


def _segments(duration: float, waits: list[list[float]]) -> list[tuple[float, float, float]]:
    """(시작, 끝, 배속) 목록. 겹치지 않게 시간순으로."""
    segs: list[tuple[float, float, float]] = []
    cursor = 0.0
    for start, end in sorted(map(tuple, waits)):
        if end - start < MIN_WAIT or start < cursor:
            continue
        fast_from = start + KEEP
        if fast_from >= end:
            continue
        segs.append((cursor, fast_from, 1.0))
        segs.append((fast_from, end, SPEEDUP))
        cursor = end
    segs.append((cursor, duration, 1.0))
    return [s for s in segs if s[1] - s[0] > 0.02]


def _duration(ffmpeg: str, path: Path) -> float:
    out = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    assert m, out[-300:]
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])


def _remap(t: float, segs: list[tuple[float, float, float]]) -> float:
    """원본 시각 t → 가속 뒤 시각."""
    out = 0.0
    for a, b, speed in segs:
        if t >= b:
            out += (b - a) / speed
        elif t > a:
            out += (t - a) / speed
            break
        else:
            break
    return out


def compose(stem: str) -> Path:
    ffmpeg = _ffmpeg()
    webm = OUT_DIR / f"{stem}.webm"
    waits_path = OUT_DIR / f"{stem}.waits.json"
    waits = json.loads(waits_path.read_text())["waits"] if waits_path.exists() else []
    duration = _duration(ffmpeg, webm)
    segs = _segments(duration, waits)
    parts, labels = [], []
    for i, (a, b, speed) in enumerate(segs):
        factor = "" if speed == 1.0 else f",setpts=(PTS-STARTPTS)/{speed}"
        parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS{factor}[v{i}]")
        labels.append(f"[v{i}]")
    graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(segs)}:v=1:a=0,fps=25,format=yuv420p[out]"
    target = OUT_DIR / f"{stem}.mp4"
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(webm), "-filter_complex", graph, "-map", "[out]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(target)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-600:])
    srt_in = OUT_DIR / f"{stem}.ko.srt"
    if srt_in.exists():
        cues = _parse_srt(srt_in)
        text = "\n".join(
            f"{i}\n{_srt_time(_remap(a, segs))} --> {_srt_time(_remap(b, segs))}\n{c}\n" for i, (a, b, c) in enumerate(cues, 1)
        )
        (OUT_DIR / f"{stem}.mp4.ko.srt").write_text(text, encoding="utf-8")
    saved = duration - _duration(ffmpeg, target)
    print(f"  ✓ {target.name}  ({duration:.1f}s → {duration - saved:.1f}s, 대기 가속 {len([s for s in segs if s[2] > 1])}곳)")
    return target


def main(argv: list[str] | None = None) -> int:
    keys = (argv if argv is not None else sys.argv[1:])
    stems = sorted(p.stem for p in OUT_DIR.glob("*.webm") if not p.stem.endswith(".fail"))
    stems = [s for s in stems if not keys or any(k in s for k in keys)]
    if not stems:
        print("합성할 .webm이 없다(먼저 runner).")
        return 2
    for stem in stems:
        print(f"▶ {stem}")
        compose(stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
