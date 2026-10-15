# 사용법 영상 — 계획 문서와 자동 생성 (MathSlate)

사람이 노트북을 조작하며 녹화하지 않는다. **계획(이 문서) → 시나리오 파일 → 러너 → 영상** 순서로 만든다.
MathSlate는 GUI 앱이 아니라 **Python 라이브러리**이므로, 영상의 "화면"은 **Jupyter Lab 노트북**(그리고 JupyterLite·터미널)이다.
Nemo1API의 Qt 합성도, Slingshot의 웹앱 녹화도 그대로 쓸 수 없다 — 이 과제만의 요소는 **셀에 코드를 한 글자씩 치는 연출**과
**커널이 계산하는 동안의 대기**다.

```
docs/design/TUTORIAL_VIDEO_PLAN.md                ← 무엇을 보여줄지 (이 문서)
tools/tutorial_video/scenario.py             ← 시나리오 로드·검증 (YAML)
tools/tutorial_video/scenarios/*.yml         ← 챕터당 시나리오 하나
tools/tutorial_video/jupyter_env.py          ← 격리된 Jupyter Lab 서버를 띄우고 내린다
tools/tutorial_video/runner.py               ← 실행 진입점 (Playwright for Python)
tools/tutorial_video/compose.py              ← 대기 구간 가속·자막 구움·mp4 변환 (ffmpeg)
tools/tutorial_video/out/*.mp4, *.srt        ← 결과 (git에 넣지 않는다)
requirements-video.txt                       ← playwright, pyyaml, jupyterlab, (starter extra)
```

> **상태: 러너(M0·M1) 구현됨, 01~05·09·11편 시나리오가 돈다.** `tools/tutorial_video/`에 `jupyter_env.py`·`scenario.py`·`runner.py`가 있고 실제로 돌려 webm·`.ko.srt`와 화면을 확인했다(§9 "구현하며 확인한 것"). 합성(`compose.py` — mp4·대기 가속)과 20·21·00편은 아직 없다. §10 카드는 `README.md`·`docs/tutorial_ko.md`·`examples/mathslate_tour.py`를 읽고 쓴 것이다.

## 0. 왜 이 방식인가

| | 이 과제에서 |
|---|---|
| 영상의 화면 | 사용자가 실제로 쓰는 **Jupyter Lab**(`pip install "mathslate[starter]"`가 설치하는 것). 같은 코드가 marimo·JupyterLite에서도 돌지만 기준은 Jupyter Lab. |
| 녹화 | Playwright(Python)로 브라우저를 구동하고 `record_video_dir`로 녹화한다. **자막·가짜 커서는 페이지에 DOM으로 심는다**(Slingshot 방식). |
| 조작 | 노트북 셀을 **클릭 → 코드를 한 글자씩 입력(`keyboard.type`, 글자당 40~70ms) → `Shift+Enter` 실행 → 출력이 나타날 때까지 대기.** 사람이 치는 것처럼 보이는 것이 이 챕터들의 핵심 연출이다. |
| 코드의 정본 | 영상에 치는 코드는 **`docs/tutorial_ko.md`·`README.md`의 코드 블록과 같아야 한다.** 시나리오가 코드를 따로 적지 않고 `code_from: tutorial_ko.md#1` 처럼 문서의 블록을 가리킨다(§4) — 문서가 바뀌면 영상도 바뀐다. |
| 출력 검증 | 출력은 **결정적**이다(sympy가 심볼릭으로 푼다). `plot(sin(x)/x)`가 항상 `curve \| x ∈ [-10, 10] \| 411 samples \| 1 discontinuity handled` 를 낸다(`README`). 러너는 출력 텍스트가 **기대 문자열을 포함하는지** 확인해, 라이브러리 동작이 바뀌면 영상 생성이 실패하게 한다 — 영상이 낡은 동작을 보여 주는 것을 막는 점검이 된다. |
| 시작 상태 | 임시 폴더에 **새 Jupyter 설정·데이터·런타임 디렉터리**를 만들어 서버를 띄운다(사용자의 설정·확장·최근 파일을 건드리지 않는다). 첫 화면이 "처음 설치한 사용자의 Jupyter Lab"이다. |
| 설치 챕터 | 터미널에서 `pip install`을 치는 장면은 브라우저가 아니라 **터미널 화면**이다. 터미널 녹화 도구(§9-6)로 따로 만든다. |
| 결과물 | Playwright webm → `compose.py`가 mp4 + `.srt`로 변환. 대기 구간은 가속(§6). |

## 1. 역할 분담

| 단계 | 누가 | 결과물 |
|---|---|---|
| 챕터 계획 | 사람(기획) | 이 문서 §2·§10 |
| 시나리오 작성 | 사람 또는 Claude Code | `scenarios/NN_이름.yml` |
| 영상 생성 | 러너 + 합성 | `out/NN_이름.mp4`, `.ko.srt` |
| 검토 | 사람 | 카드의 `상태` 갱신 |

## 2. 챕터 구성

`docs/tutorial_ko.md`의 순서를 따른다 — 영상을 보고 같은 절을 읽는 흐름이다.

### 파트 1 — Jupyter Lab (00~19)

| # | 제목 | 튜토리얼 | 길이(안) | 환경 | 상태 |
|---|---|---|---|---|---|
| 00 | 설치와 첫 실행 (`mathslate[starter]`, Jupyter Lab 열기) | §0 | 40초 | 터미널 + Lab | 계획 |
| 01 | 첫 그래프 — `plot(sin(x)/x)`와 출력 메시지 읽기 | §1·§2 | 45초 | Lab | 구현 |
| 02 | 여러 곡선과 단 하나의 규칙 (리스트 vs 튜플) | §3 | 40초 | Lab | 구현 |
| 03 | 범위 지정과 옵션 — `(x, 0, 6.28)`, `title=`, `points=` | §4 | 40초 | Lab | 구현 |
| 04 | 그리기 어려운 함수 — `tan(x)`·`1/x`·`floor(x)` | §5 | 45초 | Lab | 구현 |
| 05 | 매개변수 곡선·극좌표·3D (원, 나선, 꽃, 곡면) | §7·§11 | 45초 | Lab | 구현 |
| 06 | `analyze()` — 함수가 어떤 것인지 읽기 | README | 40초 | Lab | 구현 |
| 07 | `slider()`와 `animate()` — 끌어서 보기 | README | 40초 | Lab | 구현 |
| 08 | 내 데이터: 리스트·쌍, `dataset()`, `fit`, `table()` | §8·§12 | 45초 | Lab | 구현 |
| 09 | 탈출구 — `show_python()`이 보여 주는 NumPy+SymPy+Plotly 코드 | §9 | 40초 | Lab | 구현 |
| 10 | 다른 사람에게 전달 — `worksheet()`와 HTML 한 파일 | §13 | 40초 | Lab + 브라우저 탭 | 구현 |
| 11 | 헷갈릴 때는 묻는다 — 모호한 축과 `parameters` | §11 | 35초 | Lab | 구현 |

### 파트 2 — 다른 환경 (20~29)

| # | 제목 | 길이(안) | 환경 | 상태 |
|---|---|---|---|---|
| 20 | 설치 없이 체험 — JupyterLite | 40초 | 정적 사이트(Pyodide) | 계획 |
| 21 | marimo에서 쓰기 (`import *` 대신 명시 import) | 35초 | marimo | 계획 |

### 예약
- 30번대: AI 도우미(`ask()`·`explain()`·`describe()`·`suggest_model()`·`draft.repair()` — 모두 `mathslate.ai`) — **API 키와 네트워크가 필요**해 지금은 카드를 만들지 않는다. 모의 공급자가 있어야 결정적으로 찍힌다(§9-5).
- 40번대: Mathematica에서 넘어오셨나요?(§14), 외부에서 작업 계속하기(§10) — 코드 비교 중심이라 정적 슬라이드가 더 낫다.

## 3. 챕터 카드 쓰는 법

- **목표는 한 문장, 영상 하나에 하나.**
- **치는 코드는 문서의 코드 블록 그대로**(§0). 영상용으로 코드를 줄이거나 바꾸지 않는다.
- **자막은 코드가 하는 일을 말한다**(`plot()`이 무엇을 추론했는지). 출력 메시지는 영상에서 **읽어 주는 것이 핵심**이므로 해당 줄을 `highlight`로 강조한다.
- 장면 하나는 3~8초. 코드를 치는 시간(글자 수 × 지연)이 길면 장면을 나누거나 한 셀을 짧게 쓴다. 한 셀 40자 이내가 자연스럽다.
- **그림이 나오는 시간**을 자막에 쓰지 않는다. 커널·플로틀리 로딩 시간은 PC마다 다르다.
- 데모 값은 **문서 코드의 값**이지 임의의 값이 아니다. `dataset()`용 숫자는 문서의 예제 값 그대로.

## 4. 시나리오 파일 형식

```yaml
title: 첫 그래프
subtitle: plot(sin(x)/x) 한 줄과 출력 메시지
output: 01_first_plot.mp4
env: jupyterlab                         # jupyterlab | jupyterlite | marimo | terminal
video: { width: 1920, height: 1080, speed_up_waits: 6 }
setup:
  notebook: blank                       # blank | file:<경로>  — 빈 노트북으로 시작
  kernel: python3
scenes:
  - caption: 첫 셀에 import 한 줄이 전부입니다
    steps:
      - cell: {index: 0, code_from: "tutorial_ko.md#1.a", type_delay_ms: 45}     # from mathslate import *
      - run: {wait_output: any}
  - caption: plot() 한 줄로 그래프가 나옵니다
    steps:
      - cell: {index: 1, code_from: "tutorial_ko.md#1.b"}                          # plot(sin(x)/x)
      - run: {wait_output: figure, expect_text: "1 discontinuity handled", timeout: 60}
      - highlight: {output_line: "singularities at x = 0", seconds: 3}
```

### 동작

| 동작 | 예 | 하는 일 |
|---|---|---|
| `cell` | `cell: {index: 1, code_from: "…", type_delay_ms: 45}` | 셀을 선택(없으면 아래에 새 셀을 추가)하고 코드를 한 글자씩 친다. `code`로 직접 적을 수도 있지만 **문서 참조(`code_from`)가 기본**이다. |
| `run` | `run: {wait_output: figure, expect_text: "…", timeout: 60}` | `Shift+Enter` → 출력이 나타날 때까지 기다린다. `wait_output`: `any`(아무 출력) · `text`(`expect_text` 포함) · `figure`(Plotly 그림). **`expect_text`가 안 맞으면 시나리오가 실패한다**(§0). 의도한 오류를 보여 줄 때는 `expect_error: AmbiguousAxisError`(출력에 그 예외 이름이 있어야 통과). |
| `highlight` | `highlight: {output_line: "…", seconds: 3}` / `{selector: "…"}` | 출력의 해당 줄이나 요소에 주황 테두리 |
| `scroll` | `scroll: {to: output, index: 1}` | 셀 출력이 화면에 보이게 스크롤 |
| `drag_slider` | `drag_slider: {label: "a", from: 1, to: -2, seconds: 3}` | Plotly 슬라이더 컨트롤을 마우스로 끈다(보간 이동 + 가짜 커서). |
| `new_tab` | `new_tab: {file: "handout.html"}` | 생성된 HTML을 새 탭에서 열어 보여 준다(10편). 창 전환 자막을 띄운다. |
| `wait` / `caption` / `title` | — | Slingshot과 같다. |
| `terminal` | `terminal: {cmd: 'pip install "mathslate[starter]"', speed_up: 8}` | 터미널 챕터 전용(§9-6) |

### 코드 참조 형식

`code_from: "tutorial_ko.md#1.b"`는 문서의 "§1의 두 번째 코드 블록"이다. 블록 번호가 아닌 **블록 앞의 HTML 주석 앵커**(`<!-- video:1-plot -->`)를 쓰면
문서가 편집돼도 깨지지 않는다 — 문서에 앵커를 더하는 작은 변경이 필요하다(§11 M0). 앵커가 없는 블록을 가리키면 러너가 실패한다.

## 5. 시작 상태와 데모 자료

| 키 | 예 | 하는 일 |
|---|---|---|
| `notebook` | `blank` | 임시 폴더에 빈 노트북을 만들어 연다. |
| `notebook: file:examples/mathslate_tour.ipynb` | — | 가이드 투어 노트북을 복사해서 연다(06편 등에서 한 셀씩 실행). |
| `files` | `[sales.csv]` | 노트북 폴더에 데모 파일을 둔다(08편). |
| `theme` | `JupyterLab Light` | 테마를 고정한다(설정 파일로). |

데모 자료: 08편의 `dataset()`용 값은 `docs/tutorial_ko.md` §8의 예제 값을 그대로 쓴다. 파일이 필요하면 **가상 측정값**(예: 가상의 낙하 실험 시간·거리)만 쓴다. 실제 학생 데이터 금지.

## 6. 녹화 설정

| 항목 | 값 |
|---|---|
| 기본 | 1920×1080. Jupyter Lab의 글자가 작게 보이므로 브라우저 확대(`document.body.style.zoom` 또는 Lab 설정 글꼴 크기)를 **125~150%**로 고정한다. |
| 대기 가속 | 커널 시작·Plotly 첫 로딩·`pip install` 같은 대기는 러너가 **구간 표시**(시작·끝 시각)를 기록하고, `compose.py`가 그 구간을 6~8배속으로 처리한다(화면에 "⏩ 로딩 중" 표시). 사용자가 보는 건 "기다리는 시간 압축"이며, **계산이 끝나 그림이 나오는 순간은 실시간으로 보여 준다.** |
| 자막 파일 | `.ko.srt` (가속 구간 반영 후의 시각) |

## 7. 실행

```bash
python -m venv .venv-video
.venv-video/bin/pip install -e ".[starter]" -r requirements-video.txt     # Windows: .venv-video\Scripts\
.venv-video/bin/playwright install chromium ffmpeg

.venv-video/bin/python -m tools.tutorial_video.runner tools/tutorial_video/scenarios/01_first_plot.yml
.venv-video/bin/python -m tools.tutorial_video.runner tools/tutorial_video/scenarios/*.yml
.venv-video/bin/python -m tools.tutorial_video.compose out/       # 가속·mp4·srt
```

- 러너는 포트와 토큰을 정해 **임시 설정 디렉터리로 `jupyter lab --no-browser`를 띄우고**(`JUPYTER_CONFIG_DIR`/`JUPYTER_DATA_DIR`/`JUPYTER_RUNTIME_DIR`을 임시 폴더로) 끝나면 내린다.
- **헤드리스 Chromium에서 Plotly 3D(WebGL)가 그려지는지**는 구현 때 확인한다(§9-3). 안 그려지면 `--use-gl=swiftshader`류 옵션 또는 `--headed`.
- 결과는 `out/`(git 제외).

## 8. 화면이 바뀌었을 때

- **라이브러리 동작이 바뀌면** `expect_text`가 맞지 않아 러너가 어느 장면·셀에서 실패했는지 알린다 — 이것이 영상이 낡았다는 신호다. 문서도 같이 고친다.
- **Jupyter Lab 버전이 바뀌면** DOM 클래스가 달라질 수 있다. 러너의 셀렉터를 한 곳(`jupyter_env.py`의 `SELECTORS`)에 모아 두어 거기만 고친다.
- 자막이 화면과 맞는지는 사람이 본다.

## 9. 알아 둘 점 (확인한 것과 구현 때 확인할 것)

확인한 것(문서·코드로):
- `plot()`이 출력하는 한 줄(예: `curve | x ∈ [-10, 10] | 411 samples | 1 discontinuity handled` + `· singularities at x = 0`)은 심볼릭 계산에서 나오므로 결정적이다(`README`·튜토리얼 §2).
- `slider()`의 컨트롤은 **Plotly 자체 컨트롤이 그림 안에 들어 있다**(`README`) — ipywidgets 없이 HTML 한 파일로도 동작한다. 따라서 드래그는 Plotly의 SVG 슬라이더를 마우스로 끌면 된다.
- `worksheet(...)`는 `path="handout.html"`로 HTML 한 파일을 만들고 Plotly를 한 번만 내장한다(`standalone` 기본).
- marimo는 `import *`를 거부한다 — 21편의 요점이다(튜토리얼 §1.1).
- 라이브러리의 핵심(그리기·`analyze`·표·적합·내보내기)은 AI·네트워크 없이 돈다(`README`). **`explain`·`describe`·`suggest_model`·`ask`는 `mathslate.ai`**(공급자·키 필요)라 01~11편에는 넣지 않는다.

구현하며 확인한 것(M0·M1, 01~05·09·11편):
- **계획서가 `docs/`에 있으면 테스트가 깨진다.** `tests/test_docs.py`는 `docs/*.md`마다 실행 가능한 python 블록이 있어야 한다고 요구한다 — 그래서 이 계획서를 `docs/design/`(그 테스트가 훑지 않는 하위 폴더)로 옮겼다. (처음 올린 커밋 `07f69b9`부터 그 테스트가 실패해 있었다.)
- **§9-1 괄호 자동 닫기: 문제없다.** `keyboard.type`으로 `plot(sin(x)/x)`를 치면 그대로 완성된다. 따옴표도 같다(`title="A bell curve"`). **§9-2·7**은 `jupyter_env.py`의 사용자 설정(공지·뉴스·자동완성 끔)으로 막았고 첫 화면은 깨끗하다.
- **§9-3 Plotly는 헤드리스 Chromium에서 그려진다 — 3D 곡면(`plot(x*y)`)도 WebGL(SwiftShader, `--use-gl=swiftshader --enable-unsafe-swiftshader`)로 나온다.** 그림은 Plotly 단독이 아니라 anywidget 컨트롤(x/y 범위, `Fit Y`, `Samples` 등)이 붙은 위젯이다. 셀 출력의 `.js-plotly-plot`로 기다린다.
- **새 노트북은 `kernelspec`이 있어야 한다.** API로 빈 노트북을 만들 때 메타데이터에 커널 이름을 안 넣으면 Lab이 `Select Kernel` 창을 띄운다. 러너는 코드 셀 하나와 `python3` kernelspec이 든 `notebook.ipynb`를 `PUT /api/contents`로 만들어 연다.
- **자막 영역은 `body`가 아니라 `#main`을 줄여서 만든다.** `body`에 `transform`/`height`를 주면 Lab의 절대 배치가 깨졌다(툴바가 잘리고 자막 막대가 위로 올라갔다). `#main{height:calc(100% - 76px)}`와 하단 고정 막대로 상태 표시줄(`Idle`)을 그대로 두고 자막을 아래에 둔다.
- **서버는 `--allow-root`가 필요하다**(컨테이너는 root). 임시 디렉터리에서만 도는 격리 서버라 문제없다.
- **준비 구간 제거**: 노트북을 열고 커널이 `Idle`이 될 때까지의 구간을 영상 앞에서 ffmpeg로 잘라낸다(`imageio-ffmpeg` 필요 — `requirements-video.txt`).
- **실행 완료는 셀 프롬프트가 `[n]:`이 되는 것으로 안다**(`[*]`가 끝났을 때). 그 뒤에 `expect_text`/`expect_error`는 출력이 DOM에 그려질 때까지 기다려야 한다 — 실행이 끝난 직후엔 오류 출력이 아직 없는 때가 있다(11편에서 실제로 한 번 실패했다).
- **강조는 출력 텍스트의 한 구절을 `Range`로 찾아 그 위에 주황 테두리를 그린다**(`output_line`). ANSI 색이 입혀진 오류 메시지는 텍스트 노드가 갈려 있어 `AmbiguousAxisError: …`처럼 색이 바뀌는 경계를 넘는 구절은 못 찾는다 — 한 색 안의 구절(`3 free symbols found (a, b, c).`)을 쓴다.
- **코드는 문서 앵커에서 온다.** `docs/tutorial_ko.md`의 코드 블록 앞에 `<!-- video:이름 -->` 주석을 21개 더했다(`first-plot`, `tan`, `show-python`, `ambiguous` 등). `code_from: "tutorial_ko.md#video:이름"`, 블록을 빈 줄로 나눈 덩어리는 `part:`로 고른다(예: `first-plot`의 `part: 0`은 import 줄). 앵커가 없으면 러너가 시나리오 로드 때 실패한다.
- 첫 셀을 실행하면(커널이 `mathslate`를 가져오고 위젯을 준비하는 동안) 5~10초가 걸린다 — 아직 **가속 구간 처리(compose)가 없어 영상에 그대로 남는다**(01편 전체 약 21초, 05편 약 34초).
- 04편의 `tan`은 이 PC에서 `713 samples | 6 discontinuities handled`와 극점 6개를 낸다 — 자막은 숫자를 쓰지 않았다.

구현하며 확인한 것(06·07·08·10편):
- **`analyze()` 결과는 Lab에서 접힌 `▶ analyze(x**3 - 3*x)`로 나온다**(HTML `<details>`) — 펼쳐야 보인다. 펼치면 `print`(텍스트)와 달리 **표(`<tr>`)**로 나온다: 행마다 `roots`, `maxima`, `increasing on` …. 그래서 `output_line`(텍스트 구절) 대신 `highlight: {selector: "tr:has-text('roots')"}`를 쓴다. 그 아래 `how each was obtained`(또 하나의 `<details>`)를 펼치면 값마다 `solveset(...)`로 풀었는지, `simplify(f(-x) -+ f(x)) == 0`처럼 어떻게 구했는지가 나온다(§10 6번의 solved/sampled 구분).
- **`slider()`/`animate()`는 Plotly 자체 컨트롤이다**(ipywidgets 아님). 슬라이더 손잡이는 `.slider-grip-rect`, 레일은 `.slider-rail-rect` — 러너의 `drag_slider: {to: 0..1, seconds}`가 손잡이를 마우스로 레일 위치까지 끈다(구간을 잘게 나눠 천천히). 끌면 곡선이 21프레임 중 하나로 바뀌고 `s = 2.8` 같은 값이 슬라이더 위에 나온다. `animate()`는 `.updatemenu-button`의 `play`/`pause` 버튼이다(러너의 `click: {selector}`).
- **`new_tab`은 같은 화면 위 오버레이다.** Playwright는 탭(페이지)마다 따로 녹화하므로 새 탭을 열면 영상에 안 찍힌다. 그래서 탭 막대를 흉내 낸 띠(`📄 handout.html`, `file:///handout.html`) 아래에 `/files/handout.html`을 **iframe**으로 띄우고, 마우스를 그래프 위로 움직인 뒤 걷는다. handout은 Plotly를 내장한 단일 파일(약 4.8MB)이라 인터넷 없이 열린다.
- **자막 막대 초기화 스크립트는 최상위 문서에서만 돈다.** 처음엔 iframe 문서 안에서도 돌아 iframe 아래에 두 번째 자막 막대가 생겨(어두운 띠가 그림 아래 76px를 가렸다) 10편이 잘 보이지 않았다. `window.top !== window`면 건너뛴다.
- **문서에 없는 코드는 `code:`로 직접 적는다.** `analyze(x**3 - 3*x)`·`animate(s*sin(x))`·`table(sin(x), (x, 0, 1))`·`page.save("handout.html")`는 문서에 코드 블록이 아니라 표·산문으로만 있어서 시나리오에 적었다(실행해 확인했다). 나머지(`slider`·`fit-line`·`worksheet` 등)는 앵커로 문서 블록을 가리킨다 — 앵커를 `tutorial_ko.md`에 4개(`data-list`·`data-pair`·`fit-line`·`worksheet`), `guide_ko.md`에 1개(`slider`) 더했다.
- 8편의 `fit-line` 블록은 한 셀에 4줄이라 입력에 약 8초가 걸린다 — 영상 35초 중 20초 가까이가 입력·실행 대기다(compose 가속 대상).
- 10편의 Lab 파일 목록은 `handout.html`이 생겨도 새로고침 전에는 안 보여서 그 장면은 넣지 않았다(계획서 10편 2번).

구현 때 확인할 것:
1. **셀에 글자를 칠 때 괄호 자동 닫기.** CodeMirror 6은 `(`를 치면 `)`를 자동으로 넣고, 사용자가 `)`를 치면 덮어쓴다. `keyboard.type`이 같은 결과를 내는지(`plot(sin(x)/x)`가 그대로 완성되는지) 확인한다. 따옴표·대괄호도.
2. **자동완성 팝업.** Jupyter Lab의 완성은 Tab 키 기반이라 일반 입력에는 안 뜨지만, 설정에 따라 뜰 수 있다. 영상에 팝업이 안 보이게 설정을 고정한다(`jupyter_env.py`의 설정 덮어쓰기).
3. **Plotly 그림이 Lab에서 나오는 방식.** `_repr_mimebundle_`가 Plotly mime을 돌려주므로 Lab이 렌더러를 가져야 한다. `mathslate[starter]`로 설치한 환경에서 **그림이 실제로 나오는지**(그리고 3D 곡면은 WebGL로 나오는지)를 첫 편에서 확인한다.
4. **JupyterLite의 로딩 시간.** Pyodide 다운로드·`piplite.install("mathslate")`에 수십 초가 걸린다(`README`: "slower than a local install"). §6의 가속 구간으로 처리하고, 가능하면 로컬에서 `lite/`를 빌드해 정적 서버로 서빙한다(외부 네트워크에 기대지 않기 위해). Pyodide 자체가 CDN에서 오는지 번들되는지는 확인 필요.
5. **AI 도우미 챕터를 만들려면 모의 공급자가 필요하다.** `mathslate/ai/providers.py`에 공급자 목록(`available_providers`)이 있다 — 공급자를 바꿔 끼울 수 있는 구조인지 보고, 있다면 결정적 응답을 주는 모의 공급자를 영상 전용으로 쓴다. **실제 키를 영상에 노출하지 않는다.**
6. **터미널 챕터(00).** 브라우저 녹화가 아니다. 후보: ① VHS(charmbracelet) `.tape` 파일을 시나리오로 쓰고 mp4 출력, ② `asciinema` + `agg`. `pip install`은 수 분이 걸리므로 **미리 받아 둔 wheel 캐시**(`--find-links`)로 실행하거나 가속 구간으로 처리한다. 가속하면 "실제 설치 시간이 이만큼 걸리지 않는다"는 오해가 생기므로 화면에 ⏩ 표시를 둔다.
7. **Jupyter Lab 첫 실행 팝업**(공지·추천 확장). 새 설정 디렉터리에서는 공지 창이 뜰 수 있다 — `announcements` 확장 비활성 또는 설정으로 끈다.

## 10. 챕터 카드

코드 블록 번호(`§1.a`)는 `docs/tutorial_ko.md`의 절과 블록이다. 구현 때 앵커로 바꾼다(§4).

### 00. 설치와 첫 실행
- 목표: 한 줄로 설치하고 Jupyter Lab을 연다
- 환경: 터미널 → Lab
- 시나리오: `00_install.yml`
- 장면:
  1. 터미널에서 `pip install "mathslate[starter]"` (가속) — "한 줄로 설치합니다"
  2. 설치된 것 요약 자막(Jupyter Lab, 대화형 컨트롤, 도우미, 키 저장) — "Jupyter Lab과 필요한 것이 함께 설치됩니다"
  3. `jupyter lab` 실행 → 브라우저에 Lab 열림 — "`jupyter lab`으로 엽니다"
  4. 새 노트북 만들기 — "새 노트북을 엽니다"
  5. 첫 셀에 `from mathslate import *` — "이 한 줄이 설정의 전부입니다"

### 01. 첫 그래프
- 목표: `plot()` 한 줄로 그래프를 그리고 출력 메시지가 무슨 뜻인지 읽는다
- 시나리오: `01_first_plot.yml`
- 장면:
  1. `plot(sin(x)/x)` 입력·실행 — "`symbols`도 `linspace`도 `show`도 필요 없습니다"
  2. 그림이 나옴 — "`x`는 이미 정의되어 있습니다"
  3. 출력 한 줄 강조 `curve | x ∈ [-10, 10] | 411 samples | 1 discontinuity handled` — "MathSlate가 무엇을 판단했는지 한 줄로 알려 줍니다"
  4. `· singularities at x = 0` 강조 — "0에서 끊어지는 것을 알아냈습니다"
  5. `set_verbose(False)` 후 다시 `plot(sin(x))` — "이 메시지는 끌 수 있습니다"
  6. `f = plot(sin(x)/x); print(f.summary())` — "결과 객체에서 읽을 수도 있습니다"

### 02. 여러 곡선과 단 하나의 규칙
- 목표: 리스트는 "독립된 여러 개", 튜플은 "하나의 항목"이라는 규칙을 안다
- 시나리오: `02_list_vs_tuple.yml`
- 장면: (코드는 튜토리얼 §3 그대로)
  1. `plot([sin(x), cos(x), sin(x) + cos(x)])` — "리스트는 곡선 여러 개를 겹쳐 그립니다"
  2. 출력 메시지에서 곡선 수 강조 — "메시지에도 곡선 수가 나옵니다"
  3. `plot((cos(t), sin(t)))` — "튜플은 하나의 매개변수 곡선, 원 하나입니다"
  4. 두 결과를 나란히(스크롤) — "같은 식이라도 괄호가 의도를 알려 줍니다. 외울 것은 이것 하나입니다"

### 03. 범위 지정과 옵션
- 목표: 범위를 직접 정하고, 제목·점 수 같은 옵션을 안다
- 시나리오: `03_range_options.yml`
- 장면: (코드는 튜토리얼 §4 그대로)
  1. `plot(sin(x), (x, 0, 6.28))` — "범위는 `(기호, 시작, 끝)` 튜플로 정합니다"
  2. 출력 메시지의 `x ∈ [0, 6.28]` 강조 — "이제 메시지의 범위가 내가 정한 값입니다 (처음에는 자동 `[-10, 10]`이었습니다)"
  3. `plot(exp(-x**2), (x, -3, 3), title="A bell curve", label="Gaussian")` — "제목과 이름도 붙입니다"
  4. `plot(sin(x), points=2000)` — "점 개수도 정할 수 있지만 거의 필요 없습니다"
  5. 자막 — "그 이유는 04편에서: MathSlate가 점을 알아서 더 찍기 때문입니다"
- 비고: 축에 π가 표시되는 부분(튜토리얼 §6)은 이 편에 넣을지 04편 앞에 따로 둘지 구현 때 정한다(코드를 읽고 확정).

### 04. 그리기 어려운 함수
- 목표: 불연속 함수가 깨끗하게 그려지는 것을 본다
- 시나리오: `04_discontinuities.yml`
- 장면:
  1. `plot(tan(x))` — "`tan(x)`는 극점마다 끊어서 그립니다"
  2. 그림에서 수직선이 없는 것 강조 — "가짜 수직선이 없습니다"
  3. `plot(1/x)` — "`1/x`는 0에서 끊깁니다"
  4. `plot(floor(x))` — "`floor(x)`는 정수마다 끊깁니다"
  5. `plot(sqrt(x))` — "정의역 밖은 아예 계산하지 않습니다"
  6. 출력 메시지의 singularities 줄 강조 — "끊긴 곳은 심볼릭 계산으로 찾습니다 — 추측이 아닙니다"

### 05. 매개변수 곡선·극좌표·3D
- 목표: 원·나선·꽃 모양과 곡면·공간 곡선을 같은 방식으로 그린다
- 시나리오: `05_polar_3d.yml`
- 장면: (코드는 튜토리얼 §7·§11 그대로)
  1. `plot((cos(t), sin(t)))` — "한 바퀴가 기본 범위입니다 (`0`에서 `2π`)"
  2. `plot((t*cos(t), t*sin(t)), (t, 0, 20))` — "반지름이 커지면 나선입니다"
  3. `polar(1 + cos(t))` — "극좌표는 추측하지 않고 `polar()`로 말해 줍니다"
  4. `polar(sin(3*t))` — "세 잎 장미입니다"
  5. `plot(x*y)` — "자유 기호가 둘이면 곡면입니다"
  6. `plot(x*y, kind="contour")` — "`kind=`로 등고선으로 펼칩니다"
  7. `plot((cos(t), sin(t), t), (t, 0, 12))` — "세 요소 튜플은 공간 곡선입니다"
  8. 3D를 마우스로 돌려 보기 — "마우스로 돌려 볼 수 있습니다" (**WebGL이 안 되면 이 장면과 5~7을 빼고 2D만**, §9-3)

### 06. analyze()
- 목표: 함수의 성질을 한 번에 읽는다
- 시나리오: `06_analyze.yml`
- 장면:
  1. `analyze(x**3 - 3*x)` — "함수가 어떤 것인지 알려 줍니다"
  2. 결과 표(근·극값·증감 등)의 **풀었는지(solved)와 표집했는지(sampled)** 구분 강조 — "수치마다 계산으로 구한 것인지 표집으로 근사한 것인지 표시합니다"
  3. (`describe(analyze(...))`는 `mathslate.ai`의 AI 기능이라 **이 편에 넣지 않는다** — 30번대. `analyze()`의 표 자체가 읽는 대상이다)

### 07. slider()와 animate()
- 목표: 콜백 없이 매개변수를 끌어 그래프를 바꾼다
- 시나리오: `07_slider.yml`
- 장면:
  1. `a = slider(-3, 3, default=1, name="a")` — "매개변수를 슬라이더로 만듭니다"
  2. `plot(a*sin(x))` — "콜백을 쓸 필요가 없습니다"
  3. 슬라이더 드래그(`drag_slider`) — "끌면 그래프가 바뀝니다"
  4. `animate(a*sin(x))` — "재생 버튼이 붙은 버전도 있습니다"
  5. 재생 — (몇 초)

### 08. 내 데이터
- 목표: 숫자 리스트와 데이터 쌍을 그리고, 식을 데이터에 맞춰 본다
- 시나리오: `08_dataset.yml`
- 장면:
  1. `plot([2.0, 4.0, 8.0, 16.0, 32.0])` (튜토리얼 §8) — "숫자 리스트는 데이터 하나입니다"
  2. `plot(([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0]))` — "리스트 두 개의 쌍은 x-y 데이터입니다"
  3. 자막 — "§3의 규칙 그대로입니다: 숫자 리스트는 여러 숫자, 두 리스트의 튜플은 데이터셋 하나"
  4. `d = dataset(...)` → `d.fit(a*x + b)` (`api.py`의 `dataset()` 도큐스트링 예) — "식을 쓰면 데이터에 맞춰 줍니다"
  5. 맞춘 매개변수 출력 강조 — "구한 값이 나옵니다"
  6. `table(...)` — "표로도 봅니다" (`table()`의 실제 호출 형태는 문서에서 확인)
- 주의: `fit`은 AI 없이 도는 `Dataset.fit`이다(README: "fitting … without an API key"). `suggest_model()`은 `mathslate.ai`라 넣지 않는다.

### 09. 탈출구 — show_python()
- 목표: MathSlate가 대신 해 준 일을 일반 코드로 확인한다
- 시나리오: `09_show_python.yml`
- 장면:
  1. `plot(sin(x)/x).show_python()` — "`show_python()`을 붙이면"
  2. 출력된 NumPy+SymPy+Plotly 코드 스크롤 — "같은 그림을 만드는 일반 코드가 나옵니다"
  3. 코드 일부 강조(조용히 처리한 부분: 불연속 처리 등) — "MathSlate가 조용히 해 준 일도 코드로 보입니다"
  4. (선택) 그 코드를 새 셀에 붙여 실행 — "그대로 실행해도 같은 그림이 나옵니다"
- 주의: 출력된 코드는 **실제 `show_python()`의 출력**이어야 한다 — 시나리오가 코드를 적지 않는다. 그대로 실행하는 장면은 같은 그림이 나오는지 러너가 확인한다.

### 10. 다른 사람에게 전달
- 목표: 그래프와 설명을 HTML 한 파일로 묶어 건넨다
- 시나리오: `10_worksheet.yml`
- 장면:
  1. `worksheet([...], title="Limits", path="handout.html")` (README 예) — "문제와 그래프를 한 파일로 묶습니다"
  2. 파일이 만들어진 것(Lab 파일 목록에 `handout.html`) — "HTML 한 파일이 생깁니다"
  3. 새 탭에서 열기(`new_tab`) — "받는 사람은 브라우저로 엽니다"
  4. 그래프 마우스 올리기·확대 — "그래프는 그대로 상호작용이 됩니다"
  5. 자막 — "Plotly가 한 번만 내장되어 인터넷 없이도 열립니다" (README: `standalone` 기본)

### 11. 헷갈릴 때는 묻는다
- 목표: 모호한 입력에서 MathSlate가 추측하지 않고 질문하는 것을 보고, 매개변수로 해결한다
- 시나리오: `11_ambiguity.yml`
- 장면: (코드는 튜토리얼 §11 그대로)
  1. `a, b, c = symbols("a b c", real=True)` / `plot(a*b*c)` — "자유 기호가 셋이고 어느 것이 축인지 알 수 없습니다"
  2. 오류(`AmbiguousAxisError`) 출력 강조(`run: {expect_error: AmbiguousAxisError}`) — "추측하지 않고, 메시지 자체가 질문입니다"
  3. 메시지 속 정확한 호출 강조 — "해결할 수 있는 호출까지 보여 줍니다"
  4. `plot(a*b*c, (a, -5, 5), parameters={b: 2, c: 3})` — "`parameters`는 축이 아닌 기호의 값을 고정합니다"
  5. `plot(a*sin(x))` — "모호하지 않으면 관례를 따릅니다: `x`가 축, `a`는 아닙니다"
- 주의: **`explain()`은 이 편에 넣지 않는다.** `mathslate.ai`의 AI 기능이라(공급자·API 키 필요) 결정적으로 찍을 수 없다 — 30번대(§9-5). 오류 클래스 이름은 `expect_error`로 확인한다.

### 20. 설치 없이 체험 — JupyterLite
- 목표: 설치 없이 브라우저에서 바로 써 본다
- 환경: JupyterLite(Pyodide)
- 시나리오: `20_jupyterlite.yml`
- 장면:
  1. 사이트 열기(`https://berd2.github.io/mathslate/` 또는 로컬 빌드) — "설치 없이 브라우저에서 바로 씁니다"
  2. 로딩(가속 구간) — "처음에는 로딩이 오래 걸립니다"
  3. `await piplite.install("mathslate")` — "첫 셀에서 한 번 설치합니다"
  4. `from mathslate import *`, `plot(sin(x)/x)` — "그다음은 같습니다"
  5. 자막 — "빨리 둘러볼 때만 쓰고, 계속 쓰려면 설치하세요"

### 21. marimo
- 목표: marimo에서 `import *` 대신 이름을 명시한다
- 환경: marimo
- 시나리오: `21_marimo.yml`
- 장면:
  1. `marimo edit` 화면 — "marimo에서도 씁니다"
  2. `from mathslate import *` 시도 → 거부 메시지 — "marimo는 `import *`를 허용하지 않습니다"
  3. 명시 import(`from mathslate import plot, sin, x`) — "필요한 이름을 직접 가져옵니다"
  4. `plot(sin(x))` — "나머지는 똑같습니다"
- 주의: marimo는 별도 서버라 러너에 `env: marimo` 구동기가 필요하다(§11 M5).

## 11. 구현 순서(마일스톤)

1. **M0 — 문서 앵커와 Spike(반나절~하루).** `docs/tutorial_ko.md`의 코드 블록에 `<!-- video:… -->` 앵커를 더하고, 격리된 Jupyter Lab을 띄워 **한 셀을 치고 실행해 Plotly 그림이 headless Chromium에서 나오는지**(§9-1·§9-3)를 확인한다. 이것이 막히면 방식을 바꾼다(예: `--headed` + Xvfb).
2. **M1 — 러너 뼈대.** `jupyter_env.py`(격리 서버), `scenario.py`, `runner.py`의 `cell`·`run`·`highlight`·자막·가짜 커서. 01편으로 검증한다.
3. **M2 — 합성.** `compose.py`: 대기 구간 가속, mp4·srt.
4. **M3 — 02~06·09·11편.** 셀 입력·출력 확인만 쓰는 편들.
5. **M4 — 07·08·10편.** 슬라이더 드래그, 데이터, 새 탭.
6. **M5 — 20·21편, 00편.** JupyterLite(로컬 빌드), marimo, 터미널 녹화.
7. **예약 — 30번대(AI 도우미).** 모의 공급자 설계 후.
