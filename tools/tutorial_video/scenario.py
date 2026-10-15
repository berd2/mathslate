"""시나리오(YAML) 로드와 코드 참조 해석. 계획서 §4.

영상에 치는 코드는 시나리오가 따로 적지 않고 문서의 코드 블록을 가리킨다(`code_from`). 문서가 바뀌면 영상 코드도 같이 바뀐다 —
`tests/test_docs.py`가 문서의 모든 python 블록을 실행하므로 영상에 치는 코드도 실제로 도는 코드다.
블록은 앞의 HTML 주석 앵커 `<!-- video:이름 -->`로 가리킨다(블록 번호는 문서를 고치면 깨진다).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DOCS = Path(__file__).resolve().parents[2] / "docs"
_ANCHOR = re.compile(r"<!--\s*video:(?P<name>[\w-]+)\s*-->\s*\n```python[^\n]*\n(?P<body>.*?)^```", re.MULTILINE | re.DOTALL)


class ScenarioError(ValueError):
    pass


def code_block(ref: str, part: int | None = None) -> str:
    """`tutorial_ko.md#video:이름` → 그 앵커 뒤의 python 블록. `part`는 빈 줄로 나뉜 덩어리의 번호(그 덩어리만)."""
    doc, _, anchor = ref.partition("#")
    name = anchor.removeprefix("video:")
    path = DOCS / doc
    if not path.is_file():
        raise ScenarioError(f"문서가 없다: {path}")
    for match in _ANCHOR.finditer(path.read_text(encoding="utf-8")):
        if match.group("name") == name:
            body = match.group("body").strip("\n")
            if part is None:
                return body
            chunks = [c.strip("\n") for c in re.split(r"\n\s*\n", body) if c.strip()]
            if part >= len(chunks):
                raise ScenarioError(f"{ref}에는 덩어리가 {len(chunks)}개뿐이다(part={part}).")
            return chunks[part]
    raise ScenarioError(f"{doc}에 앵커 <!-- video:{name} -->가 없다(또는 python 블록이 바로 뒤에 오지 않는다).")


@dataclass
class Scene:
    caption: str
    steps: list[dict[str, Any]]


@dataclass
class Scenario:
    path: Path
    title: str
    output: str
    scenes: list[Scene] = field(default_factory=list)
    setup: dict[str, Any] = field(default_factory=dict)
    env: str = "jupyterlab"  # jupyterlab | marimo (계획서 §4)

    @property
    def stem(self) -> str:
        return Path(self.output).stem


def _resolve_step(step: dict[str, Any]) -> dict[str, Any]:
    if len(step) != 1:
        raise ScenarioError(f"단계는 동작 하나만 가져야 한다: {step}")
    (action, args), = step.items()
    if action == "cell" and "code_from" in args:
        args = dict(args)
        args["code"] = code_block(args.pop("code_from"), args.pop("part", None))
    return {action: args}


def load(path: Path) -> Scenario:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("title", "output", "scenes"):
        if key not in raw:
            raise ScenarioError(f"{path.name}: '{key}'가 없다.")
    scenes = [Scene(caption=s["caption"], steps=[_resolve_step(st) for st in s.get("steps", [])]) for s in raw["scenes"]]
    return Scenario(path=path, title=raw["title"], output=raw["output"], scenes=scenes, setup=raw.get("setup", {}), env=raw.get("env", "jupyterlab"))
