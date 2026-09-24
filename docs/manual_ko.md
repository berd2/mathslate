# MathSlate 레퍼런스 매뉴얼

**버전 0.1** · 공개 API, 추론 규칙 및 샘플링 알고리즘에 대한 완전한 설명.

이 문서는 처음부터 끝까지 읽기보다는 찾아보기 용도로 작성되었습니다. 단계별 안내가 필요하다면 [튜토리얼](tutorial_ko.md)을 먼저 읽어보세요.

아래의 모든 `python` 코드 블록은 테스트 스위트(`tests/test_docs.py`)에 의해 문서에 나타난 순서대로 실행되며, 모두 `from mathslate import *`로 시작하는 하나의 공유 네임스페이스에서 실행됩니다.

---

## 목차

 1. [설치 및 요구 사항](#1-%EC%84%A4%EC%B9%98-%EB%B0%8F-%EC%9A%94%EA%B5%AC-%EC%82%AC%ED%95%AD)
 2. [공개 API 표면](#2-%EA%B3%B5%EA%B0%9C-api-%ED%91%9C%EB%A9%B4)
 3. [디스패치 규칙](#3-%EB%94%94%EC%8A%A4%ED%8C%A8%EC%B9%98-%EA%B7%9C%EC%B9%99)
 4. `plot()`
 5. [기호와 축 바인딩](#5-%EA%B8%B0%ED%98%B8%EC%99%80-%EC%B6%95-%EB%B0%94%EC%9D%B8%EB%94%A9)
 6. [샘플링: 기술적 핵심](#6-%EC%83%98%ED%94%8C%EB%A7%81-%EA%B8%B0%EC%88%A0%EC%A0%81-%ED%95%B5%EC%8B%AC)
 7. [축](#7-%EC%B6%95)
 8. `PlotResult`
 9. `show_python()`
10. `analyze()`
11. `slider()`[ 및 ](#11-slider-%EB%B0%8F-animate)`animate()`
12. `table()`[ 및 HTML 내보내기](#12-table-%EB%B0%8F-html-%EB%82%B4%EB%B3%B4%EB%82%B4%EA%B8%B0)
13. `dataset()`[, 통계 및 선형 대수](#13-dataset-%ED%86%B5%EA%B3%84-%EB%B0%8F-%EC%84%A0%ED%98%95-%EB%8C%80%EC%88%98)
14. [오류](#14-%EC%98%A4%EB%A5%98)
15. [프론트엔드 어댑터](#15-%ED%94%84%EB%A1%A0%ED%8A%B8%EC%97%94%EB%93%9C-%EC%96%B4%EB%8C%91%ED%84%B0)
16. [마일스톤](#16-%EB%A7%88%EC%9D%BC%EC%8A%A4%ED%86%A4)
17. [내부 구조 맵](#17-%EB%82%B4%EB%B6%80-%EA%B5%AC%EC%A1%B0-%EB%A7%B5)
18. [보장 사항 및 보장하지 않는 사항](#18-%EB%B3%B4%EC%9E%A5-%EC%82%AC%ED%95%AD-%EB%B0%8F-%EB%B3%B4%EC%9E%A5%ED%95%98%EC%A7%80-%EC%95%8A%EB%8A%94-%EC%82%AC%ED%95%AD)

---

## 1. 설치 및 요구 사항

```bash
pip install mathslate
```

Python 3.10 이상이 필요합니다. 런타임 종속성은 `sympy`, `numpy`, `plotly`뿐이며, 다른 것은 설치되지 않습니다 — MathSlate를 설치할 때 노트북 프론트엔드가 자동으로 설치되는 일은 없습니다. 이는 설계 의도가 아니라 테스트로 검증된 속성입니다.

선택적 추가 패키지:

| 추가 패키지 | 설치되는 패키지 | 용도 |
| --- | --- | --- |
| `mathslate[jupyter]` | `ipywidgets` | Jupyter 및 Colab 위젯 (v0.5) |
| `mathslate[marimo]` | `marimo` | marimo 반응형 위젯 (v0.5) |
| `mathslate[dev]` | `pytest` | 테스트 스위트 실행 |

```python
import mathslate

print(mathslate.__version__)
```

### 임포트하기

**marimo를 제외한** 대부분의 환경에서는 `from mathslate import *`를 지원하며, 이것이 의도된 사용법입니다. 사전 정의된 기호들(`x`, `y`, `z`, …)을 어딘가에서 가져와야 하기 때문입니다. 이는 일반적인 Python 권장 사항에 대한 의도적인 예외입니다. 이를 보완하기 위해 `show_python()`은 항상 명시적으로 `x = sp.symbols('x', real=True)`를 출력하므로, 별표(*) 임포트가 영구적으로 여러분에게 무언가를 숨기지는 않습니다.

네임스페이스를 지정하는 방식도 동일하게 동작하며, 모든 곳에서 사용할 수 있습니다:

```python
import mathslate as ms

f = ms.plot(ms.sin(ms.x))
```

#### marimo는 `import *`를 금지합니다.

marimo는 **파싱(parse)** 시점에 별표 임포트를 거부하므로 셀 자체가 전혀 실행되지 않습니다:

```text
line 1 SyntaxError: Importing symbols with `import *` is not allowed in marimo.
```

이것은 MathSlate의 규칙이 아니라 marimo의 규칙이며, 이를 비활성화할 수 있는 옵션은 없습니다. marimo는 반응형 노트북입니다: 어떤 셀이 다시 실행될지 결정하는 종속성 그래프를 그리기 위해, 각 셀이 정의하고 사용하는 이름이 무엇인지 정적으로 파악해야 합니다. `import *`를 사용하면 모듈을 실제로 실행해 보기 전까지는 정의된 이름의 집합을 알 수 없으므로 허용되지 않습니다.

첫 번째 셀에서 명시적 임포트를 사용하세요. 이 코드는 튜토리얼에서 제공하는 코드와 같으며, `tests/test_marimo_imports.py`에서 marimo의 파서를 통해 검증됩니다:

```python
from mathslate import (
    plot, polar, analyze, show_python, set_verbose, get_verbose,
    slider, animate, table, dataset, frontend_report,
    sin, cos, tan, atan, exp, log, sqrt, Abs, floor, pi,
    diff, integrate, limit, solve, simplify, symbols,
    Eq, Integer, Matrix,
    x, y, z, t, n, k, theta,
)
```

이 매뉴얼의 다른 모든 내용은 그대로 유지되며, 오직 임포트 줄만 변경됩니다.

| 환경 | `import *` | 명시적 임포트 | `import mathslate as ms` |
| --- | --- | --- | --- |
| Jupyter | ✅ | ✅ | ✅ |
| Colab | ✅ | ✅ | ✅ |
| 일반 Python | ✅ | ✅ | ✅ |
| marimo | ❌ 파싱 에러 | ✅ | ✅ |

---

## 2. 공개 API 표면

MathSlate는 20개의 예산(PRD §20) 중 **19개의 공개 이름**을 도입합니다. 그 외에 노출되는 모든 것은 변경 없이 다시 내보내진(re-exported) SymPy입니다.

### MathSlate에서 새로 도입된 기능

| 이름 | 상태 | 목적 |
| --- | --- | --- |
| `plot()` | v0.1 | 통합 디스패치 — §4 |
| `polar()` | v0.1 | 추론으로 감지할 수 없는 `r = f(θ)` — §4.4 |
| `show_python()` | v0.1 | 동일하게 동작하는 순수 Python 코드 표시 — §9 |
| `set_verbose()` / `get_verbose()` | v0.1 | 한 줄 추론 보고서 표시 여부 설정 — §4.6 |
| `frontend_report()` | v0.1 | 감지된 환경 표시 — §15 |
| `PlotResult` | v0.1 | 모든 `plot()`이 반환하는 타입 — §8 |
| `analyze()` | v0.5 | 식의 속성 분석 — §10 |
| `Analysis` | v0.5 | `analyze()`가 반환하는 타입 — §10 |
| `slider()` | v0.5 | 독자가 변경할 수 있는 매개변수 — §11 |
| `animate()` | v0.5 | 슬라이더와 동일하나 재생 버튼 포함 — §11 |
| `table()` | v0.5 | 표 형태의 값 반환 — §12 |
| `Table` | v0.5 | `table()`이 반환하는 타입 — §12 |
| `dataset()` | v1.0 | 기호와 데이터를 연결하는 다리 — §13 |
| `Dataset` | v1.0 | `dataset()`이 반환하는 타입 — §13 |
| `set_range_controls()` / `get_range_controls()` | v1.5 | 기본 `plot()`이 `range_controls()`를 보여줄지 여부 — §4.13, §21 |
| `set_plot_size()` / `get_plot_size()` | v1.6 | 이후 모든 plot이 갖는 크기 — §4.12b |

```python
import mathslate

print(len(mathslate.NEW_API))
```

### SymPy에서 다시 내보내진 기능

이들은 래퍼(wrapper)가 아니라 **SymPy 함수 그 자체**입니다:

```python
import sympy, mathslate

print(mathslate.solve is sympy.solve, mathslate.diff is sympy.diff)
```

`solve`, `simplify`, `expand`, `factor`, `apart`, `cancel`, `together`,
`trigsimp`, `nsimplify`, `diff`, `integrate`, `limit`, `series`, `solveset`,
`nsolve`, `lambdify`, `Matrix`, `Symbol`, `symbols`, `Function`, `Piecewise`,
`Eq`, `Rational`, `Integer`, `Float`, `Sum`, `factorial`, `binomial`, `gcd`,
`sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `asin`, `acos`, `atan`, `sinh`,
`cosh`, `tanh`, `exp`, `log`, `sqrt`, `cbrt`, `root`, `Abs`, `sign`, `floor`,
`ceiling`, `pi`, `E`, `I`, `oo`.

또한 `mathslate.sympy`를 통해 전체 SymPy 라이브러리에 접근할 수도 있습니다. 위 목록에 없는 함수도 손이 닿지 않는 것이 아니라 속성 하나 거리에 있습니다.

#### 풀이와 대수 (Solving and algebra)

MathSlate는 계산하지 않습니다. SymPy가 합니다(PRD §2.2). 엔진을 감싸지 않고 그대로 재내보내는 이유는, 이것들이 진짜 함수 그 자체이고, SymPy 문서 그대로 동작하며, 무엇을 반환하든 곧바로 그림으로 흘러들어가기 때문입니다. 아래는 어떤 기능이 있는지에 대한 안내이지 SymPy 레퍼런스의 재작성이 아닙니다.

**풀이(Solving).** `solve()`는 맨 식을 `= 0`으로 읽거나 `Eq`를 받습니다. 방정식 리스트와 미지수 리스트를 주면 연립방정식입니다:

```python
print(solve(x**2 - 5*x + 6, x))              # [2, 3]
print(solve(Eq(x**2, 2), x))                 # [-sqrt(2), sqrt(2)]
print(solve([x + y - 3, x - y - 1], [x, y])) # {x: 2, y: 1}
```

`solveset()`은 주기 방정식의 모든 분기를 포함해 해집합 전체를 반환합니다. `nsolve()`는 닫힌 형태가 없는 방정식에 대해 시작값에서 근 하나를 수치적으로 찾습니다:

```python
print(solveset(sin(x), x))          # 정수 전체에 대한 2*n*pi 및 2*n*pi + pi
print(nsolve(x - cos(x), x, 0.5))   # 0.739085133215161
```

**대수(Algebra).** 다항식·유리식 동사들, 그리고 삼각 항등식:

<!--octarine-table-cols:166,0-->
| 함수 | 하는 일 |
| --- | --- |
| `factor(e)` | 유리수 위에서 인수분해 |
| `expand(e)` | 전개 |
| `gcd(a, b)` | 두 다항식의 최대공약수 |
| `apart(e)` | 부분분수 분해 |
| `cancel(e)` | 유리식을 기약 형태로 |
| `together(e)` | 공통분모로 결합 |
| `trigsimp(e)` | 삼각 항등식으로 정리 |
| `simplify(e)` | 범용 단순화 |

```python
print(factor(x**3 - x))              # x*(x - 1)*(x + 1)
print(expand((x + 1)**3))            # x**3 + 3*x**2 + 3*x + 1
print(gcd(x**2 - 1, x**2 - x))       # x - 1
print(apart(1/(x**2 - 1)))           # -1/(2*(x + 1)) + 1/(2*(x - 1))
print(trigsimp(sin(x)**2 + cos(x)**2))  # 1
```

결과가 평범한 식이므로 대수와 그림 사이에 변환 단계가 없습니다:

```python
model = factor(x**3 - 6*x**2 + 11*x - 6)
print(model, "→ roots", solve(model, x))
_ = plot(model, (x, 0, 4), verbose=False)
```

여기에 의도적으로 **없는** 것은 이 과정들의 단계별 유도입니다 — `solve()`는 답을 주지 풀이 과정을 주지 않습니다. 이것이 MathSlate가 지키는 유일한 스코프 경계선이며(§10.4, PRD §2.2), SymPy에는 범용 풀이 과정 엔진이 없고 부분적인 것은 가르치기보다 오해를 낳습니다.

### 사전 정의된 기호

`x`, `y`, `z`, `t`, `n`, `k`, `theta` — 모두 `real=True`로 생성됩니다. 이 덕분에 정의역과 특이점 분석에서 복소 평면에서의 답이 아닌 교과서적인 답을 얻을 수 있습니다.

```python
print(x.is_real, theta.name)
```

물론 자신만의 기호를 만들 수도 있습니다. 이들은 모두 일반적인 SymPy 기호입니다.

```python
u, v = symbols("u v", real=True)
f = plot(u**2, (u, -3, 3))
```

---

## 3. 디스패치 규칙

`plot()`은 여러분의 의도를 추론합니다. 이 매핑은 단순한 구현 세부 사항이 아니라 **문서화된 규칙(contract)**이며, 버전 업그레이드 없이는 변경되지 않습니다.

> **단 하나의 규칙.** `list`는 *여러 개를 함께* 의미합니다.
> `tuple`은 *하나의 벡터 값 객체*를 의미합니다.

| 입력 형태 | 자유 기호 (Free symbols) | 결과 | 상태 |
| --- | --- | --- | --- |
| `Expr` | 1 | 2D 곡선 | v0.1 |
| `Expr` | 0 | 수평선 + 메시지 | v0.1 |
| `list[Expr]` | 1개 공유 | 중첩된 곡선들 | v0.1 |
| `tuple[Expr, Expr]` | 1개 공유 | 2D 매개변수 곡선 | v0.1 |
| `callable` | — | 수치 샘플링 | v0.1 |
| 배열 형태 (array-like) | — | 데이터 시리즈 | v0.1 |
| `(xdata, ydata)` | — | 산점도 (scatter) / 선형 | v0.1 |
| `Expr` | 2 | 곡면, 등고선 토글 가능 | v0.5 |
| `Eq(lhs, rhs)` | 2 | 음함수 곡선 | v0.5 |
| `tuple[Expr × 3]` | 1개 공유 | 3D 공간 곡선 | v0.5 |
| `tuple[Expr × 3]` | 2개 공유 | 매개변수 곡면 | v0.5 |
| `Dataset` | — | 각 열, 첫 번째 열은 가로축 | v1.0 |
| `Matrix` (2×2) | — | 해당 행렬이 수행하는 변환 | v1.0 |
| 부등식 | 2 | 색칠된 영역 | v1.5 |
| 부등식 | 1 | 축 위에 음영 처리된 구간 | v1.5 |

모든 행은 구현되어 있습니다. 특정 디스패치 행이 말없이 다른 것으로 잘못 그려지는 일은 없습니다.

### 종류 (`kind`)

`result.plan.kind`는 어떤 행과 일치했는지 보고합니다:

<!--octarine-table-cols:204,461-->
| `kind` | 의미 |
| --- | --- |
| `"curve"` | 하나의 식, 하나의 자유 기호 |
| `"curves"` | 중첩된 여러 개의 식 |
| `"constant"` | 자유 기호가 없는 식 |
| `"parametric"` | `(x(t), y(t))` |
| `"polar"` | `r = f(θ)`, `polar()`를 통해 |
| `"callable"` | 일반 Python 함수 |
| `"data"` | 숫자 배열 |
| `"surface"` | 하나의 식, 두 개의 자유 기호 |
| `"contour"` | 동일하지만 평면 형태 — `kind="contour"` |
| `"implicit"` | `Eq(lhs, rhs)`의 0 레벨 |
| `"space"` | `(x(t), y(t), z(t))` |
| `"hist"`, `"box"` | 열의 분포 형태 — `kind="hist"` |
| `"linalg"` | 변환을 나타내는 2×2 행렬 |
| `"psurface"` | `(x(u,v), y(u,v), z(u,v))` |
| `"region"` | 부등식이 성립하는 평면 영역 — §4.10 |
| `"band"` | 부등식이 성립하는 축 위 구간 — §4.10 |

```python
print(plot(sin(x)).plan.kind)
print(plot([sin(x), cos(x)]).plan.kind)
print(plot((cos(t), sin(t))).plan.kind)
print(plot(Integer(3)).plan.kind)
print(plot([1.0, 2.0, 3.0]).plan.kind)
```

### 식(Expression)으로 허용되는 입력

`Expr`, 문자열(`sympify`를 거침), 또는 단순한 숫자.

```python
print(plot("sin(x)/x").plan.kind)
print(plot(2.5).plan.kind)
```

### 원칙적으로 추론할 수 없는 경우

다음 두 가지 경우는 본질적으로 모호하기 때문에, MathSlate가 섣불리 추측하는 대신 명확히 의도를 말해주어야 합니다:

- **극좌표 (Polar).** `r = f(θ)`는 `y = f(x)`와 똑같이 표기됩니다. `polar()`를 사용하세요.
- **곡면(Surface) 대 등고선(contour).** 둘 다 동일한 객체에 대한 올바른 그림입니다. 기본값은 곡면(surface)이며, `kind="contour"`로 전환할 수 있습니다 (v0.5).

---

## 4. `plot()`

```text
plot(obj, *ranges, polar=False, kind=None, label=None, title=None,
     yscale=None, show_legend=None, points=None, exclusions=None,
     mesh=True, ticks=None, width=None, height=None,
     xlim=None, ylim=None, zlim=None, controls=None,
     verbose=None, parameters=()) -> PlotResult
```

### 4.1 `obj` 및 `*ranges`

`obj`는 디스패치 표에 있는 모든 것이 될 수 있습니다. 각 범위(range)는 `(기호, 최소, 최대)` 형태의 튜플입니다.

```python
f = plot(sin(x), (x, 0, 6.28))
print(f.plan.param_range)
```

형식이 잘못된 범위는 잘못 읽히는 대신 즉시 거부됩니다. 이들은 `MathSlateError`가 아니라 일반적인 Python 예외를 발생시킵니다. 인수의 형태가 잘못된 것은 수학적 실수가 아니라 프로그래밍 실수이기 때문입니다:

```python raises=TypeError
plot(sin(x), (x, 1.0))
```

```python raises=ValueError
plot(sin(x), (x, 1.0, 1.0))
```

### 4.2 `label`, `title`, `show_legend`

```python
f = plot(sin(x), label="the sine", title="A wave", show_legend=True)
print(f.plotly.data[0].name, "/", f.plotly.layout.title.text)
```

`show_legend`는 그리는 곡선이 2개 이상일 때 기본적으로 `True`가 됩니다.

### 4.3 `kind`

샘플이 그려지는 방식을 재정의(override)합니다.

| 값 | 효과 |
| --- | --- |
| `None` | 추론: 데이터 시리즈가 작으면 마커로, 그 외에는 선으로 그림 |
| `"line"`, `"curve"` | 연결된 선으로 그림 |
| `"scatter"` | 마커로 그림 |
| `"surface"` | 3D 곡면으로 그림 (자유 기호가 두 개인 경우의 기본값) |
| `"contour"` | 동일한 객체를 평면(등고선)으로 그림 — §4.11 |
| 그 외의 값 | `UnsupportedInputError` |

```python
print(plot(sin(x), kind="scatter").plotly.data[0].mode)
print(plot([1.0, 2.0, 3.0], kind="line").plotly.data[0].mode)
```

```python raises=UnsupportedInputError
plot(sin(x), kind="bar")
```

### 4.4 `polar` 및 `polar()`

`polar(expr, ...)`는 `plot(expr, polar=True, ...)`와 같으며 `plot()`이 허용하는 모든 것을 허용합니다. 식(expression)은 반지름 `r`로, 축 기호는 각도로 읽힙니다.

```python
f = polar(1 + cos(t))
print(f.plan.kind, f.plan.param_range[1].__round__(3))
```

극좌표 및 매개변수 플롯은 동등한 비율(equal-aspect)의 y축을 갖게 되어 원이 둥글게 그려집니다.

### 4.5 `yscale`

`None` (기본값), `"linear"`, 또는 `"log"`. 그 외의 값은 거부됩니다.

```python
print(plot(exp(x), yscale="log").plotly.layout.yaxis.type)
```

```python raises=UnsupportedInputError
plot(exp(x), yscale="semilog")
```

로그 스케일은 **절대 자동으로 적용되지 않고**, 오직 제안될 뿐입니다 — §7.2를 참조하세요. `yscale="log"`를 설정하면 y 클리핑(§6.5)도 비활성화됩니다. 둘 다 동일한 문제를 해결하기 위한 것이기 때문입니다.

로그 축은 0이나 음수 값을 표시할 수 없으며, Plotly는 아무런 코멘트 없이 이를 무시합니다. 옵션은 여전히 유지되지만(당신이 요청했으므로), 그 결과는 곡선이 반쯤 잘린 채 방치되는 것이 아니라 주의사항(notes)에 명시됩니다:

```python
print([n for n in plot(sin(x), yscale="log").notes if "cannot show" in n])
```

### 4.6 `verbose`

전역 설정(global setting)을 호출마다 재정의합니다. `None`이면 `set_verbose()`의 설정을 따르고, `True`나 `False`이면 강제 적용합니다.

```python
set_verbose(False)
f = plot(sin(x))          # 아무것도 출력하지 않음
print(get_verbose())
set_verbose(True)
```

보고서는 출력하지 않고도 확인할 수 있습니다:

```python
print(plot(tan(x), verbose=False).summary())
```

형식:

```text
<kind> | <symbol> ∈ [<lo>, <hi>] | <n> samples | <n> discontinuities handled
```

마지막 필드는 불연속성이 있을 때만 나타납니다. 벡터화된 빠른 경로가 실패한 경우(§6.6), `| element-wise evaluation`이 추가로 표시됩니다.

정의역 경고, 특이점 목록, 로그 스케일 제안 등의 주의사항(Notes)은 요약(summary) 줄 아래에 출력되며, `result.notes`를 통해 읽을 수 있습니다.

### 4.7 `points`

적응형 세분화(adaptive refinement) 이전에 사용할 초기 균일 샘플의 개수를 설정합니다. 기본값은 200입니다. 이 값을 높이면 필요한 경우 최대 포인트 제한(point cap)도 함께 올라갑니다. 점이 2개 미만이면 선을 표현할 수 없으므로, 암묵적으로 기본값으로 반올림하는 대신 곧바로 거부됩니다.

```python
print(plot(sin(x), points=1000).plan.total_points >= 1000)
```

```python raises=UnsupportedInputError
plot(sin(x), points=0)
```

두 변수 plot — 곡면, 등고선, 음함수 곡선, 영역, 매개변수 곡면 — 에서는 격자의 **축당** 샘플 수를 뜻합니다:

```python
print(plot(x*y, points=30, verbose=False).plan.series[0].sample.z.shape)   # (30, 30)
```

곡선과는 일부러 다른 척도이고, 상한이 있습니다. 선을 따라 2000점은 2000번의 평가지만 2000×2000 격자는 400만 번이기 때문입니다. 지정하지 않으면 각 종류가 자기 기본값을 유지합니다 — 곡면은 축당 60, 영역은 200입니다. 영역이 더 높은 이유는 영역은 *경계*로 판단하고 곡면은 *내부*로 판단하기 때문입니다.

```python
print(plot(x*y, verbose=False).plan.series[0].sample.z.shape)              # (60, 60)
print(plot(x**2 + y**2 < 1, verbose=False).plan.series[0].sample.z.shape)  # (200, 200)
```

§4.13의 `Samples` 슬라이더가 이 값을 실시간으로 조절합니다.

### 4.8 `exclusions`

자동 불연속점 검출(§6)은 `plot(tan(x))`이 옳게 그려지는 이유입니다. 하지만 검출에 재정의 수단이 없다면, 그 판단이 당신의 생각과 다른 순간부터 블랙박스가 됩니다. 그래서 직접 지정할 수 있게 했습니다. Mathematica의 `Exclusions`와 같은 인자입니다.

**끊을 위치를 직접 지정합니다** — 검출된 위치에 더해서:

```python
cut = plot(sin(x), (x, -6, 6), exclusions=[0, pi/2], verbose=False)
print(sorted(cut.plan.series[0].sample.breakpoints))
```

기호 값도 받습니다(`exclusions=[pi/2]`가 자연스러운 표기입니다). 창 밖의 위치는 거부하지 않고 무시합니다 — 위치를 지정한 뒤 범위를 좁히는 것은 실수가 아니기 때문입니다.

**또는 수치 탐색 자체를 끕니다**(`exclusions=False`). 실제로는 연속이지만 기울기가 급해서 불연속처럼 보이는 곡선에 씁니다:

```python
print(len(plot(floor(x), (x, -3, 3), verbose=False).plan.series[0].sample.breakpoints))
print(len(plot(floor(x), (x, -3, 3), exclusions=False, verbose=False).plan.series[0].sample.breakpoints))
```

`False`는 §6.4의 이분 탐색 **만** 끕니다. 기호적 특이점과 실수 정의역은 그대로 적용됩니다 — 그것들은 추측이 아니라 SymPy의 답이기 때문입니다:

```python
poles = plot(tan(x), (x, -4, 4), exclusions=False, verbose=False)
print(len(poles.plan.series[0].sample.breakpoints))
```

`exclusions=True`는 거부합니다. 검출은 이미 켜져 있으므로 아무 의미가 없고, 조용히 아무 일도 하지 않는 인자는 기능이 아니라 버그입니다.

```python raises=UnsupportedInputError
plot(sin(x), exclusions=True)
```

두 형태 모두 `show_python()`에 반영되므로, 출력된 코드도 그림과 같은 위치에서 선을 끊습니다.

### 4.9 `parameters`

축이 *아닌* 기호들입니다. 기호와 값의 매핑(해당 기호는 그 값으로 고정됨) 또는 단순한 기호 시퀀스(축 후보에서만 제외됨)를 허용합니다.

```python
a = symbols("a", real=True)
f = plot(a*sin(x), parameters={a: 3.0})
print(f.plan.symbol.name, round(float(f.numpy[1].max()), 3))
```

단순 시퀀스 형태는 축 후보에서 기호를 제거하지만 값을 부여하지는 않으므로, 식에 해당 기호가 남아있다면 평가할 대상이 없게 됩니다. 이 경우 해결 방법과 함께 명시적으로 거부됩니다:

```python raises=UnsupportedInputError
plot(a*sin(x), parameters=[a])
```

기호는 축이거나 매개변수 둘 중 하나입니다. 두 가지 모두를 요구하는 것은 선호가 아니라 모순입니다:

```python raises=UnsupportedInputError
plot(a*sin(x), (a, -1, 1), parameters={a: 2})
```

이것이 v0.5에서 `slider()`가 사용할 바인딩 지점이며, 단순 시퀀스 형태에서 누락된 값을 슬라이더가 제공하게 됩니다.

### 4.10 부등식: 영역과 구간

`Eq(lhs, rhs)`는 두 식이 **같아지는** 곳을 묻고, 그 답은 곡선입니다(§4.11). `lhs < rhs`는 한쪽이 다른 쪽을 **넘어서는** 곳을 묻고, 그 답은 영역입니다 — 다른 질문이므로 다른 그림입니다. 어느 쪽인지는 자유 기호의 개수가 결정하며, 이는 일반 수식과 똑같은 규칙입니다:

```python
plot(x**2 + y**2 < 1)          # 기호 2개 → 색칠된 단위원 내부 (부등식의 영역)
plot(sin(x) > 0, (x, -6, 6))   # 기호 1개 → x축 위 음영 구간 (부등식의 해)
```

`And`, `Or` 및 `&`, `|` 연산자로 결합할 수 있고, 결합된 전체가 한 번에 평가됩니다 — `lambdify`가 `And`를 `numpy.logical_and`로 출력하기 때문입니다:

```python
print(plot((x**2 + y**2 < 4) & (y > x), verbose=False).plan.kind)
```

**진리값이 없는 곳에 대해서는 아무것도 주장하지 않습니다.** 이건 보기보다 미묘합니다. NaN과의 비교는 NumPy에서도 순수 파이썬에서도 *거짓*이므로, 그대로 두면 `sqrt(x*y) > 1`이 `x*y < 0`인 두 사분면을 *영역 밖*이라고 보고하게 됩니다 — 부등식 자체가 의미를 갖지 않는 곳에 대한 단정입니다. 그런 점은 비워 두고, 비율을 알립니다:

```python
print([n for n in plot(sqrt(x*y) > 1, verbose=False).notes if "truth value" in n])
```

#### 한 변수: 해집합과 그것을 설명하는 곡선

자유 기호가 하나면 부등식이 성립하는 구간이 음영으로 표시되고, **여기에** `lhs - rhs`**의 곡선이 함께** 그려집니다. 음영은 답이 *어디* 있는지 말하고, 같은 지점에서 0을 지나는 곡선은 *왜* 그런지 말합니다.

```python
answer = plot(x**2 - 4 < 0, (x, -4, 4), verbose=False)
print(answer.plan.bands, answer.plan.bands_exact)
```

`bands_exact`는 다른 모든 곳과 같은 정직성 규칙입니다. `True`는 `solveset`이 부등식을 풀었다는 뜻이고, `False`는 구간을 샘플링으로 찾고 경계를 이분법으로 정밀화했다는 뜻입니다.

`solveset`**은 신뢰하지 않고 검증합니다.** 주기 부등식에 대해 `solveset`은 기본 주기만 반환하면서 *그 사실을 말하지 않습니다*. `solveset(sin(x) > 0, x, Interval(-6, 6))`은 `(0, π)`이고, `(-6, -π)`를 조용히 빠뜨립니다. 범위 인자를 줘도 막아 주지 않습니다. 그래서 닫힌 형태는 **샘플링이 그 밖에서 해를 찾지 못할 때만** 채택합니다. 그러지 않으면 확신에 찬 틀린 답이 화면에 오르는데, 이 라이브러리가 절대 허용하면 안 되는 유일한 실패 방식입니다.

```python
periodic = plot(sin(x) > 0, (x, -6, 6), verbose=False)
print(len(periodic.plan.bands), periodic.plan.bands_exact)
```

한 변수에 대한 복합 부등식은 추측하지 않고 거부합니다 — 자유 기호 두 개를 주어 영역으로 그리거나, 부분별로 나눠 그리십시오.

```python raises=UnsupportedInputError
plot((x > 0) & (x < 1), (x, -2, 2))
```

### 4.11 두 변수: 곡면, 등고선, 음함수 곡선

두 개의 자유 기호를 가진 단일 식은 곡면(PRD §5.1)이며, `kind="contour"`는 동일한 객체를 평면으로 보여줍니다. 두 경우 모두 동일한 격자(grid)를 읽기 때문에, 토글(toggle)은 단지 그림의 형태만 바꿉니다.

```python
print(plot(x*y).plan.kind, plot(x*y, kind="contour").plan.kind)
```

방정식(equation)은 양변이 일치하는 곡선을 그립니다:

```python
print(plot(Eq(x**2 + y**2, 4)).plan.kind)
```

세 가지 구성 요소로 된 튜플은 하나의 매개변수를 가지는 공간 곡선이 되거나, 두 개의 매개변수를 가지는 매개변수 곡면이 됩니다:

```python
print(plot((cos(t), sin(t), t), (t, 0, 12)).plan.kind)
print(plot((cos(t)*cos(theta), cos(t)*sin(theta), sin(t)),
           (t, -1.5, 1.5), (theta, 0, 6.28)).plan.kind)
```

이들의 기본 범위(window)는 곡선의 `(-10, 10)`이 아니라 각 축에서 `(-5, 5)`입니다. 더 넓은 범위에서 60×60 그리드를 사용하면 선을 따라 그리는 200개의 포인트보다 해상도가 훨씬 떨어지기 때문입니다. 두 범위 모두 보고됩니다.

```python
print(plot(x*y, verbose=False).summary())
```

**두 변수의 경우 보장할 수 없는 점.** §6.1과 §6.2는 SymPy에게 곡선이 정확히 어디에서 실수인지, 어디에서 폭발(blow up)하는지 묻습니다. 이 두 질문 모두 두 변수에 대해 사용할 수 있는 답을 제공하지 않습니다. `continuous_domain`은 단일 기호만 취하며, 두 변수에 대한 `singularities`는 존재하지 않습니다. 따라서 곡면은 그리드 상에서 평가되며, 실수가 아닌 모든 부분은 잘못된 값이 아니라 구멍(hole)이 되고, §6.5와 동일한 원칙에 따라 z 범위가 클리핑됩니다. 이는 곡선보다 제한된 처리를 받는 것이며, 이러한 사실을 숨기지 않고 주의사항(notes)에 명시합니다:

```python
print([n for n in plot(sqrt(x*y), verbose=False).notes if "real number" in n])
```

이 클리핑은 **색상 스케일뿐 아니라 z축까지** 제한합니다. 색상만 클리핑하면 형상(geometry)은 그대로 남으므로, 나머지 특징이 14 이내에 있는 곡면 위에 3481 높이의 극점이 있으면 여전히 3481 높이의 벽으로 그려집니다. 카메라는 그 벽을 담기 위해 줌아웃하고, 정작 보려던 것은 바닥에 깔린 평면이 되어버립니다. 둘은 같은 범위에서 나오므로 출력되는 코드에도 함께 담깁니다:

```python
poles = plot(1/(x*y), verbose=False)
print(poles.plan.series[0].sample.z_range)
print(tuple(poles.plotly.layout.scene.zaxis.range))
```

클리핑이 필요 없는 곡면은 축을 자동 설정 그대로 둡니다 — 요청하지 않은 곡면에 범위를 씌우면 그림이 잘려나가기 때문입니다:

```python
print(plot(x + y, verbose=False).plotly.layout.scene.zaxis.range)
```

### 4.12 `mesh`와 뷰 윈도우 (`xlim`, `ylim`, `zlim`)

플롯이 *무엇을 담느냐*가 아니라 *어떻게 보이느냐*를 조절하는 두 가지입니다.

**`mesh`**는 3D 곡면에 격자선을 그어, Mathematica의 `Plot3D`처럼 곡률의 느낌을 살립니다 — 색상 그라데이션만으로는 실제보다 평평해 보입니다. 기본값은 켜짐이고 `mesh=False`로 매끈한 모습으로 돌아갑니다. 평면 종류(곡선, 등고선, 영역)는 다스릴 표면이 없으므로 이 옵션을 무시합니다.

```python
print(plot(x*y, verbose=False).plotly.data[0].contours.x.show)          # True
print(plot(x*y, mesh=False, verbose=False).plotly.data[0].contours.x.show)  # None
```

`mesh`는 `True` 대신 숫자도 받는데, 축마다 몇 개의 선을 그릴지 정합니다 — 이전까지 `True`가 쓰던 Plotly의 자동 간격은 축 눈금을 고르듯 "적당히 떨어진 숫자"를 골라서, 실제 샘플 격자에 비해 듬성듬성해 보였습니다:

```python
print(plot(x*y, verbose=False).plotly.data[0].contours.x.size)          # 기본 간격
print(plot(x*y, mesh=40, verbose=False).plotly.data[0].contours.x.size)  # 더 조밀하게
```

간격은 곡면 자체의 범위로부터 계산되므로, `mesh=40`은 정의역이 2단위든 2000단위든 똑같이 40개의 선을 의미합니다 — 고정된 간격이었다면 그렇지 않았을 것입니다.

**`xlim`, `ylim`, `zlim`**는 뷰 윈도우를 설정합니다. 이것은 정의역이 아닙니다. 범위 인자 `(x, -10, 10)`은 함수를 *샘플링하는* 곳을 정하고, 이 옵션들은 축이 *보여주는* 것을 정합니다. 둘은 필요할 때 갈라집니다 — 자동 y-클립을 풀고 극점을 정면으로 보거나, 확대하거나, 두 그림을 같은 축척에 맞출 때입니다.

```python
zoomed = plot(sin(x), (x, -10, 10), xlim=(-3, 3), verbose=False)
print(tuple(zoomed.plotly.layout.xaxis.range))           # (-3.0, 3.0)
print(zoomed.plan.series[0].sample.x.min() < -9)         # 여전히 넓게 샘플링됨: True
```

`ylim`은 자동 백분위수 클립(§6.5)을 덮어써서, 클립이 감추던 극점을 보고 싶을 때 쓰는 수단입니다. `zlim`은 곡면의 z축에 대해 같은 일을 합니다:

```python
_ = plot(tan(x), (x, -4, 4), ylim=(-50, 50), verbose=False)
_ = plot(1/(x*y), zlim=(-10, 10), verbose=False)
```

각각은 `low < high`인 `(low, high)` 쌍이며 `show_python()`으로 재현됩니다. 잘못된 윈도우는 추측하지 않고 거부합니다:

```python raises=UnsupportedInputError
plot(sin(x), xlim=(5, 1))
```

### 4.12a `ticks` — 축에 붙는 눈금 라벨의 개수

그냥 두면 개수는 Plotly가 정합니다. `ticks=n`은 축당 `n`개로 제한하고, `ticks=False`는 라벨을 없애며, `ticks=None`(기본값)은 자동 선택으로 되돌립니다. 이 숫자는 **목표치가 아니라 상한**입니다 — Plotly는 여전히 반올림된 위치를 고르되, 그 개수를 넘기기 전에 멈춥니다.

```python
print(plot(exp(x), ticks=5, verbose=False).plotly.layout.xaxis.nticks)          # 5
print(plot(exp(x), ticks=False, verbose=False).plotly.layout.xaxis.showticklabels)  # False
```

3D에서는 scene의 세 축 모두에 적용되며, 2D에서보다 훨씬 중요합니다. Plotly는 평면 축의 눈금은 화면이 바뀔 때마다 축의 픽셀 길이에 맞춰 다시 배치하지만, scene의 라벨은 투영된 위치에 놓이고 아무것도 이를 다시 계산하지 않습니다 — 그래서 카메라를 당기면 글자가 겹칩니다. 이것이 그에 대한 유일한 수단입니다:

```python
print(plot(x*y, ticks=4, verbose=False).plotly.layout.scene.zaxis.nticks)  # 4
```

3개 미만은 축에 읽을 수 있는 눈금이 남지 않으므로 — 라벨 2개는 양 끝점이고 그 사이에 아무것도 없습니다 — 그리지 않고 거부합니다:

```python raises=UnsupportedInputError
plot(sin(x), ticks=2)
```

2D에서는 쓸 일이 드뭅니다: 확대·축소하면 스스로 다시 라벨을 붙입니다(§4.13).

### 4.12b `width`와 `height` — 그림의 크기

따로 지정하지 않으면 plot의 높이는 `DEFAULT_HEIGHT`(520px)입니다. Plotly 자체 기본값은 450인데, 이는 대시보드 타일의 높이입니다. 노트북 셀은 페이지 전체 너비이고 그래프는 여러 패널 중 하나가 아니라 읽으려는 대상 그 자체이며, §4.13의 사이드바가 너비의 5분의 1을 가져가고 나면 450은 우편함 투입구처럼 읽힙니다.

```python
from mathslate.render.options import DEFAULT_HEIGHT
print(plot(sin(x), verbose=False).plotly.layout.height == DEFAULT_HEIGHT)  # True
print(plot(sin(x), height=800, verbose=False).plotly.layout.height)        # 800
```

**너비는 기본적으로 일부러 지정하지 않습니다.** Plotly는 지정되지 않은 너비를 "컨테이너를 측정하라"는 뜻으로 읽으며, 그래서 그림이 셀을 가득 채웁니다. 픽셀 너비를 주면 넓은 화면에서는 옆에 빈 공간이 생기고 좁은 화면에서는 잘립니다. 고정 크기를 정말로 원할 때만 지정하세요:

```python
print(plot(sin(x), verbose=False).plotly.layout.width)                 # None
print(plot(sin(x), width=1000, verbose=False).plotly.layout.width)     # 1000
```

`set_plot_size()`는 이후 모든 plot의 기본값을 바꿉니다. 매 셀마다 쓰는 대신 노트북 맨 위에서 한 번 쓰고 싶을 때 이쪽을 씁니다. 한쪽 차원만 지정하면 다른 쪽은 그대로 두며, `reset=True`는 둘 다 되돌립니다:

```python
from mathslate import set_plot_size, get_plot_size

set_plot_size(height=720)
print(plot(sin(x), verbose=False).plotly.layout.height)         # 720
print(plot(sin(x), height=300, verbose=False).plotly.layout.height)  # 300 — 개별 호출이 우선
set_plot_size(reset=True)
print(get_plot_size())                                          # (None, 520)
```

둘 다 `show_python()`으로 재현됩니다. 그림과 그것을 만든다고 하는 프로그램의 크기가 서로 달라서는 안 되기 때문입니다.

### 4.12c `controls` — 특정 plot에서만 사이드바 끄기

범위 조절 사이드바(§4.13)는 탐색 중인 곡선에서는 너비의 5분의 1만큼의 값어치를 하지만, 그냥 보기만 하는 곡선에서는 그렇지 않습니다. `controls=False`는 이 plot에 대해 원래의 순수한 그림을 돌려주고, `controls=True`는 `set_range_controls(False)`로 노트북 전체에서 꺼놓았더라도 이 plot에서만 요청합니다.

```text
plot(sin(x), controls=False)     # 셀 너비 전체가 그래프
plot(sin(x))                     # 그림 + 사이드바, 기본값
```

이것은 *기본 표시 방식*을 사양하는 것이지 메서드를 없애는 것이 아닙니다. `.range_controls()`를 직접 호출하면 위젯은 여전히 만들어집니다. 그리고 그림 자체는 전혀 바뀌지 않으므로 `show_python()`은 어느 쪽이든 같은 프로그램을 내보냅니다 — plot이 *어디에* 표시되는지는 그 plot이 *무엇인지*의 일부가 아닙니다.

노트북 전체에 적용하고 싶으면 `set_range_controls(False)`를, 개별 plot에서 그것을 뒤집으려면 `controls=`를 쓰세요.

### 4.13 라이브 범위 컨트롤 — 그림을 그린 뒤에도 창을 움직이기

`xlim`/`ylim`은 창을 한 번 정할 뿐이고, Plotly 자체의 드래그 확대는 이미 계산된 점들을 잘라 보여줄 뿐입니다 — 넓은 정의역의 10분의 1로 확대해도 원래 점 밀도의 10분의 1을 볼 뿐, 더 자세히 보이는 게 아닙니다. `range_controls()`는 대신 다시 샘플링합니다: X나 Y를 움직이면 새 창으로 `plot()`을 다시 호출하므로, 좁힐수록 원래 해상도 그대로 그려집니다.

Jupyter나 Colab에서 `pip install mathslate[jupyter]`를 설치했으면(`ipywidgets`와, Plotly 6 이상의 `FigureWidget`에 필요한 `anywidget`까지 이 하나의 extra에 함께 들어있습니다), 이것이 셀 끝에서 그냥 `plot(...)`을 쓸 때 기본으로 보이는 화면입니다 — 메서드를 따로 부를 필요 없습니다:

```text
plot(sin(x)/x, (x, -10, 10))          # 그림 옆에 x/y min·max 입력창이 놓임;
                                        # 하나를 바꾸면 재샘플링
```

사이드바에는 `Auto Y`, `Reset`, X/Y `in`/`out` 버튼과 슬라이더 2개가 있습니다. `Ticks`는 §4.12a의 `ticks=`를 실시간으로 조절하며 트랙 맨 아래는 "개수를 Plotly에 맡김"을 뜻하고, `Samples`는 §4.7의 `points=`입니다. 진짜 3D 곡면에서는 `Auto Y`가 `Auto Z`로 바뀌고 Z `in`/`out` 버튼도 함께 나타납니다.

`Samples`는 올리는 것뿐 아니라 **내리는** 쪽으로도 써볼 만합니다. 곡선을 50까지 거칠게 만들면 적응형 샘플러의 골격이 드러납니다 — 어디에 점을 더 넣었고 어디에는 넣지 않았는지가 보이며, 이것이 "적응형"을 이 매뉴얼의 주장이 아니라 눈으로 확인할 수 있는 것으로 만듭니다. 두 변수 plot에서는 격자의 축당 샘플 수를 자체 척도로 조절합니다. 선택한 값은 이후 X·Y 편집에도 유지되므로, 창을 옮겼다고 밀도가 조용히 되돌아가지 않습니다.

확대·축소하면 가로축의 라벨이 다시 매겨지며, 이것이 그 컨트롤의 나머지 절반 — 신경 쓰지 않아도 되는 쪽 — 입니다. 처음 정의역에 맞춰 한 번 고른 눈금은 창이 움직이는 순간 틀린 것이 됩니다: π축을 주기의 3분의 1까지 확대하면 그 안에 남는 라벨 하나만 보이고, 숫자축은 확대 10배마다 자릿수가 하나씩 늘어나는데도 Plotly는 계속 같은 개수를 요구해 결국 글자가 겹칩니다. 그래서 화면에 보이는 창을 기준으로 선택을 다시 합니다 — π 눈금은 더 촘촘하거나 성긴 배수로 다시 맞추고, π의 배수를 보여줄 만하지 않은 약 1/4주기 아래에서는 일반 숫자로 내려가며, 숫자 개수는 그 라벨들이 실제로 들어갈 만큼으로 줄입니다. 3D scene은 이렇게 할 수 없어서(카메라를 움직여도 다시 계산되는 것이 없습니다) `ticks=`가 존재합니다.

`Auto Y`는 현재 박스에 들어 있는 X 범위에 맞춰 세로 창을 맞춥니다. 극점 때문에 자동 클리핑이 필요한 경우에는 그 클리핑 창을, 그렇지 않으면 실제로 그려진 값들의 범위를 사용합니다. `plot()`에 넘긴 `ylim`은 그 안에 맞추는 것이 아니라 해제합니다 — 자동 맞춤을 요청하는 것은 고정해 둔 창과는 다른 창을 원한다는 뜻이기 때문입니다. 박스는 항상 축 자체의 단위를 담으며, `yscale="log"`에서는 `ylim`과 동일한 규칙에 따라 10의 거듭제곱 지수가 됩니다.

`verbose=False`로 출력한 결과에서 위젯만 다시 얻고 싶거나, 뷰 컨트롤이 이미 정해진 plot에서 위젯을 명시적으로 부르려면 — `width=`/`height=`는 `.plotly.update_layout(width=..., height=...)`로 항상 가능했던 그림 크기 조절을, 지금 이 위젯을 다루는 바로 그 자리에서 할 수 있게 해줍니다:

```text
result = plot(sin(x)/x, (x, -10, 10), verbose=False)
result.range_controls(width=800, height=500)
```

지정하지 않으면 이미 설정된 크기를 그대로 유지하거나, 없으면 입력창 사이드바가 전체 너비의 약 5분의 1을 차지하도록 기본값이 적용됩니다. 이 분할은 (고정 픽셀이 아니라) 퍼센트 기준이라 그림과 사이드바를 합치면 항상 노트북의 실제 너비 전체를 채웁니다. 드래그로 조절 가능한 구분선은 아닙니다 — 그림과 사이드바 사이에 드래그 핸들은 없고, 크기를 바꾸려면 `range_controls()`를 새 값으로 다시 호출하면 됩니다.

곡면·등고선·영역은 두 축 모두 재샘플링되고, 일반 곡선은 Y가 X에서 유도되는 값이라 X만 재샘플링되고 Y는 `ylim` 뷰 조정으로 대체됩니다 — 애초에 값이 샘플링되지 않은 축에서 되찾을 해상도는 없기 때문입니다. 진짜 3D 곡면(`surface`/`psurface`, 평면인 `contour`/`region`은 제외)에는 `z min`/`z max` 입력창 2개가 더 생깁니다 — Z는 곡면의 *결과값*이지 정의역이 아니므로, 움직여도 `zlim`만 다시 적용될 뿐입니다 — 재샘플링 없이, 일반 곡선의 Y와 마찬가지로 저렴하게.

사이드바 자체가 필요 없으면 `controls=False`(§4.12c)로 이 plot에서만, `set_range_controls(False)`로 노트북 전체에서 끌 수 있습니다.

정의역 기호가 가로축이 아닌 경우 — x와 y가 모두 하나의 매개변수의 결과값인 매개변수 곡선, 극곡선, 3D 공간곡선 — 첫 행은 `x`가 아니라 그 매개변수의 이름(예: `t`)으로 표시됩니다. 이 값을 움직이면 곡선을 더 많이 또는 더 적게 그리는 것이지, 화면을 자르는 것이 아닙니다. 공간곡선은 Z 컨트롤도 함께 제공합니다.

marimo는 별도 래퍼가 필요 없습니다: `mo.ui.number()`를 `plot()`과 직접 조합하면 marimo가 `.value`를 읽는 셀을 값이 바뀔 때마다 다시 실행해 주므로 같은 라이브 재샘플링을 공짜로 얻습니다.

```text
x_min = mo.ui.number(start=-20, stop=20, value=-10, label="x min")
x_max = mo.ui.number(start=-20, stop=20, value=10, label="x max")
mo.hstack([x_min, x_max])
```

```text
plot(sin(x)/x, (x, x_min.value, x_max.value))
```

완전히 동작하는 예제는 `examples/marimo_notebook.py`에 있습니다. 살아있는 커널이 없는 곳 — 일반 스크립트나 내보낸 HTML 파일 — 에서는 재샘플링을 밀어 넣을 호스트가 없으므로, 창을 정하는 방법은 여전히 `xlim`/`ylim`입니다.

**아무것도 안 보이나요?** `frontend_report()`가 정확한 이유를 알려줍니다:

```python
from mathslate import frontend_report
print(frontend_report())
```
```text
frontend: jupyter | interactive widgets: ipywidgets | range_controls(): ready
```

마지막 항목이 `ready`가 아니면 구체적인 이유를 알려줍니다 — 호스트가 Jupyter/Colab이 아니거나, `ipywidgets`나 `anywidget`이 설치되어 있지 않거나, `set_range_controls(False)`로 꺼져 있는 경우입니다. 마지막 경우가 이 기능의 유일한 조절 스위치입니다 —

```python
from mathslate import set_range_controls, get_range_controls

set_range_controls(False)          # 이후의 모든 plot()이 일반 그림만 보여줌
print(get_range_controls())        # False
set_range_controls(True)           # 기본값으로 복귀
```

— `plot()` 키워드가 아니라 전역 스위치인 이유는, 이것이 "오늘은 이 기능을 원치 않는다"는 지속적인 선호이지 그림 하나마다 다른 선택이 아니기 때문입니다. 그림별로 다르게 하고 싶다면 대신 해당 결과에서 `.range_controls()`를 호출하세요.

---

## 5. 기호와 축 바인딩

`a*sin(x)`가 주어졌을 때, MathSlate는 어떻게 `x`가 축이고 `a`는 아니라는 것을 알 수 있을까요? 다음 네 가지 규칙을 순서대로 적용합니다.

**1. 명시적 범위가 우선합니다.**

```python
b = symbols("b", real=True)
print(plot(b*x, (b, -2, 2), parameters={x: 1.0}).plan.symbol.name)
```

하지만 식(expression)에 실제로 포함된 기호 중에서만 우선순위를 가질 수 있습니다. 그리고 그 범위가 해당 식에 포함되지 않은 기호를 가리킨다면 여러분이 의도한 것일 리 없으며, 그것을 축으로 만들면 실제 변수는 자유롭게 남아있게 되어 곡선이 어느 곳에서도 정의되지 않게 됩니다:

```python raises=UnsupportedInputError
c = symbols("c", real=True)
plot(sin(x), (c, -1, 1))
```

마찬가지로 한 기호에 대해 두 개의 범위를 지정하는 것도 거부됩니다. 둘 중 하나는 버려져야 하며, 어느 것이 버려졌는지 알 수 없게 되기 때문입니다:

```python raises=UnsupportedInputError
plot(sin(x), (x, -1, 1), (x, -2, 2))
```

축의 이름을 지정한다고 해서 *다른* 기호에 자동으로 값이 부여되지는 않으며, 바인딩되지 않은 기호가 포함된 곡선은 그 어느 곳에서도 값이 도출되지 않습니다. 기호가 두 개인 경우는 곡면(§4.11)이 되므로, 이 문제는 세 개부터 발생합니다:

```python raises=UnsupportedInputError
a = symbols("a", real=True)
plot(a*x*y, (x, -5, 5), (y, -5, 5))
```

남는 하나를 `parameters=`로 고정하세요. 에러 메시지도 이 방법을 안내합니다:

```python
print(plot(a*x*y, (x, -5, 5), (y, -5, 5), parameters={a: 2}).plan.kind)
```

**2. 바인딩된 매개변수는 축 후보가 될 수 없습니다.** §4.8을 참조하세요.

**3. 그 외에는 관례적 순서를 따릅니다.**

```text
x, y, z  →  t, u, v  →  r, theta, phi  →  그 다음에는 알파벳순
```

```python
from mathslate.core import binding

p, q = symbols("p q", real=True)
print([s.name for s in binding.sort_by_convention([q, theta, p, t, x])])
```

**4. 여전히 모호하다면, 추측하지 않고 직접 묻습니다.**

```python raises=AmbiguousAxisError
c, d, e = symbols("c d e", real=True)
plot(c*d*e)
```

이 예외(exception)의 `.question`은 메시지를 의미하며 `.candidates`는 기호 이름들을 나열합니다. 라이브러리 환경에서는 프롬프트를 띄울 수 없으므로, 질문 자체가 오류 메시지로 출력되며 이를 해결할 수 있는 정확한 호출 방식이 함께 표시됩니다.

```python
from mathslate.errors import AmbiguousAxisError

c, d, e = symbols("c d e", real=True)
try:
    plot(c*d*e)
except AmbiguousAxisError as error:
    print(error.candidates)
```

### 기본 범위 (Default ranges)

| 상황 | 기본값 |
| --- | --- |
| 명시적 곡선 | `(-10, 10)` |
| 매개변수 또는 극좌표, 삼각 함수 성분 포함 | `(0, 2π)` — 온전한 한 바퀴 |
| 매개변수 또는 극좌표, 그 외 | `(-10, 10)` |

```python
print(plot((cos(t), sin(t))).plan.param_range[1].__round__(3))
print(plot((t, t**2)).plan.param_range)
```

---

## 6. 샘플링: 기술적 핵심

`plot(tan(x))`에서 잘못된 수직선이 그려지지 않는다는 것이 바로 MathSlate와 단순 플로팅 래퍼의 차별점입니다. 이 매뉴얼의 다른 모든 것은 편의를 위한 것입니다. 다음의 6단계를 순서대로 거칩니다.

### 6.1 기호적 특이점 검출 (Symbolic singularity detection)

극점(poles)은 숫자 추측이 아니라 `sympy.calculus.singularities`에서 가져옵니다. (삼각 함수의 극점이 만들어내는 것처럼) 무한한 집합 형태(`ImageSet`)도 화면 범위(window) 내에서 계산됩니다.

```python
from mathslate.core import sampling

info = sampling.describe_domain(tan(x), x, -10.0, 10.0)
print([round(p, 4) for p in info.singular_points])
```

### 6.2 실수 정의역 계산 (Real domain computation)

실수 정의역은 `sympy.calculus.util.continuous_domain`에서 가져오며, 그 범위를 벗어나서 샘플링하는 일은 절대 없습니다. 화면 범위는 각 특이점에서 잘려나가, 연속적인 구간들의 리스트를 생성합니다.

```python
from mathslate.core import sampling

print(sampling.describe_domain(sqrt(x), x, -10.0, 10.0).intervals)
print(sampling.describe_domain(sqrt(x**2 - 1), x, -10.0, 10.0).intervals)
print(sampling.describe_domain(1/x, x, -10.0, 10.0).intervals)
```

SymPy가 판단할 수 없을 때 — `floor`나 `Piecewise`에서 `NotImplementedError`를 발생시키는 경우 — MathSlate는 화면 전체 범위를 사용하며, 가지고 있지도 않은 지식을 있는 척하는 대신 이 사실을 주의사항(notes)에 **명시적으로 알립니다**.

```python
f = plot(floor(x), verbose=False)
print(f.plan.series[0].sample.domain_intervals)
print([n for n in f.notes if "domain" in n])
```

### 6.3 적응형 세분화 (Adaptive subdivision)

각 구간(piece)당 `points`로 지정된 개수의 균일한 샘플에서 시작합니다. 세 개의 점으로 이루어진 내부 각도(interior angle)가 임계값보다 날카로운 모든 구간의 중간에 점을 반복적으로 삽입하여, 곡선이 구부러진 곳에 정밀도를 높입니다. 이는 `max_depth`와 `max_points`에 의해 제한됩니다.

각도를 측정하기 전에 두 축을 *가시적인(visible)* 스케일로 정규화합니다. 이렇게 해야 `sin(x)`와 `exp(x)` 모두에서 이 기준이 동일하게 작용합니다.

```python
straight = plot(2*x + 1, verbose=False).plan.total_points
wiggly = plot(sin(5*x), verbose=False).plan.total_points
print(straight, wiggly, wiggly > straight)
```

### 6.4 선 끊기 (Line breaking)

모든 불연속점에 `NaN`을 삽입하고, Plotly에 `connectgaps=False`를 전달합니다. 오직 이것만이 잘못된 수직선이 그려지는 것을 막아줍니다.

끊어지는 지점의 두 가지 출처:

- **기호적(Symbolic)**, §6.1에서 가져옴 — 정확함.
- **수치적(Numeric)**, SymPy가 볼 수 없는 점프를 탐지: `floor`, `ceiling`, `sign`, `Piecewise`, 그리고 이것들로 만들어진 모든 것.

수치적 테스트는 알아둘 만한 가치가 있습니다. 왜냐하면 MathSlate가 계단 함수를 이름조차 모른 채 다룰 수 있게 해주는 비결이기 때문입니다. 인접한 샘플 간의 갭(gap)이 가장 큰 부분들을 잡고 그 주변을 이등분(bisect)합니다. **진짜 불연속점은 구간이 줄어들어도 그 갭의 크기를 유지하지만, 가파르지만 연속적인 경사는 그렇지 않습니다.** 따라서 탐지기(probe)는 갭의 크기가 아니라 갭의 *감쇠(decay)*를 비교합니다.

```python
print(len(plot(floor(x), verbose=False).plan.series[0].sample.breakpoints))
print(plot(atan(1000*x), (x, -1, 1), verbose=False).plan.series[0].sample.breakpoints)
```

계단 함수에서는 19개의 끊김이 발생하지만, 그저 가파르게 오르는 곡선에서는 단 하나의 끊김도 발생하지 않습니다. 탐지(probing)는 곡선당 `max_jump_probes`로 제한됩니다.

**매개변수 및 극좌표 곡선도 이 모든 과정을 똑같이 적용받습니다.** 연속적인 조각들은 두 구성 요소 정의역의 *교집합(intersection)*이며(곡선상의 점은 `x(t)`와 `y(t)` 모두 그릴 수 있는 곳에서만 그릴 수 있음), 극점(poles)은 이 둘의 합집합입니다. `x`에서의 점프 또한 `y`의 점프와 마찬가지로 선을 끊어버리기 때문에, 점프 탐지기는 양쪽 좌표를 모두 관찰하고 끊어진 부분은 양쪽 모두를 공백으로 처리합니다.

```python
f = plot((tan(t), t))
print(len(f.plan.series[0].sample.breakpoints))
print(plot((sqrt(t), t), (t, -5, 5)).plan.series[0].sample.domain_intervals)
```

### 6.5 Y축 클리핑 (Y-axis clipping)

10⁷까지 치솟는 단일 극점은 다른 모든 특징을 가로줄 형태의 얼룩으로 찌그러뜨립니다. 따라서 다음과 같은 가시적인 y축 창(window)이 선택됩니다:

1. 어떤 극점이든 그 창 너비의 2% 내에 있는 샘플은 버립니다.
2. 2번째에서 98번째 백분위수(percentile)를 선택합니다. 이 값은 *균일한(uniform)* 그리드에서 읽어오는데, 적응형 세분화가 극점 주변에 점들을 쌓아 올려 백분위수 자체를 끌고 가는 것을 방지하기 위함입니다.
3. 그 창을 3배 넓힙니다. 단순히 백분위수만으로 된 범위는 읽기에는 너무 좁기 때문입니다.
4. 곡선이 실제로 도달하는 값으로 다시 제한(clamp)하여, 유계 함수(bounded functions)의 위아래가 빈 공간으로 패딩되지 않게 합니다.
5. 데이터가 정말로 치솟거나 극점이 알려진 경우에만 적용합니다.

```python
tan_range = plot(tan(x), verbose=False).plan.y_range
sinc_range = plot(sin(x)/x, verbose=False).plan.y_range
print(None if tan_range is None else [round(v, 2) for v in tan_range])
print(None if sinc_range is None else [round(v, 2) for v in sinc_range])
print(plot(sin(x), verbose=False).plan.y_range)
```

**가로 방향으로는 x가 치솟는 곳에서만 클리핑합니다.** `y = f(x)`에 대해 x의 범위는 사용자가 요청한 범위이며, 별도로 선택할 것이 없습니다. 매개변수나 극좌표 곡선에는 이런 보장이 없습니다 — `x(t)`가 극점에 도달하는 것은 `y(t)`와 정확히 똑같기 때문에, 두 축 모두에 동일한 처리를 하며, 굳이 필요하지 않은 경우에는 어느 축도 건드리지 않습니다.

```python
print(plot((tan(t), t), verbose=False).plan.x_range is not None)
print(polar(1 + cos(t), verbose=False).plan.x_range)
```

탄젠트는 읽기 편한 창을 얻고, `sin(x)/x`는 4단계의 클램핑 덕분에 실제 크기를 유지하며, 사인(sine) 함수는 치솟는 일이 없기 때문에 아예 건드리지 않습니다.

### 6.6 벡터화된 평가 (Vectorised evaluation)

식(Expression)은 `lambdify(modules="numpy")`로 평가됩니다. 복소수 결과는 조용히 실수부만 취하는 대신 `NaN`으로 변환되므로, 함수가 실수 정의역을 벗어나면 잘못된 곡선 대신 빈 공간(gap)이 남습니다.

벡터화 경로가 실패하면 MathSlate는 요소 단위(element-wise) 평가로 **요란하게(loudly)** 폴백합니다. `RuntimeWarning`을 발생시키고 결과 객체에 주의사항(note)을 남기며, 요약 줄(summary line)에 추가 필드를 표시합니다. 결코 조용히 성능을 저하시키지 않습니다.

요소 단위 평가에는 다음 순서로 시도되는 3가지 단계가 있습니다: `lambdify(modules="math")`, NumPy 함수를 점(point) 단위로 처리, 마지막으로 SymPy 자체의 `evalf`. 마지막 단계는 특수 함수에 도달할 수 있게 만들어 줍니다. NumPy나 `math` 모두 `zeta`, `Si`, `besselj`를 지원하지 않기 때문에, 이것이 없으면 폴백 과정조차 NumPy가 실패했던 것들을 구제할 수 없습니다.

```python
import warnings
from sympy import zeta

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    f = plot(zeta(x), (x, 2, 5), verbose=False)
print(f.plan.series[0].sample.vectorized, "element-wise" in f.summary())
```

### 6.7 `SamplingConfig`

기본값은 PRD를 따릅니다. 필요한 경우 직접 접근할 수 있습니다:

```python
from mathslate.core.sampling import DEFAULT_CONFIG

print(DEFAULT_CONFIG.initial_points, DEFAULT_CONFIG.max_depth, DEFAULT_CONFIG.max_points)
print(DEFAULT_CONFIG.angle_threshold, DEFAULT_CONFIG.clip_percentiles)
```

| 필드 | 기본값 | 의미 |
| --- | --- | --- |
| `initial_points` | 200 | 세분화 이전의 균일한 샘플 수 |
| `max_depth` | 8 | 세분화(refinement) 반복 횟수 |
| `max_points` | 5000 | 곡선당 절대 상한(hard cap) |
| `angle_threshold` | 177.5° | 세 개의 점이 "곡선"으로 간주되는 내부 각도 기준 |
| `clip_percentiles` | `(2.0, 98.0)` | y축 창(window) 설정을 위한 백분위수 |
| `max_jump_probes` | 200 | 곡선당 수치적 불연속점 탐지 횟수 |

### 6.8 `SampleResult`

위의 모든 과정을 거친 원시(raw) 결과는 `result.plan.series[i].sample`에 담깁니다.

| 속성 | 의미 |
| --- | --- |
| `x`, `y` | 샘플링된 배열, 끊어진 곳에서는 `NaN` |
| `t` | 매개변수 곡선에 대한 매개변수 값 |
| `breakpoints` | 선이 잘린 위치 |
| `domain_intervals` | 화면 범위 내 실수 정의역의 연속적인 구간들 |
| `y_range` | 선택된 가시적 범위(창), 또는 `None` |
| `x_range` | 가로축에 대해서도 동일 — 매개변수 및 극좌표 전용 (§6.5) |
| `vectorized` | 빠른 경로(fast path)가 유지되었는지 여부 |
| `n_points`, `finite_count` | 데이터 크기 |

```python
sample = plot(1/x, verbose=False).plan.series[0].sample
print(sample.breakpoints, sample.domain_intervals, sample.vectorized)
```

---

## 7. 축 (Axes)

### 7.1 π 인식 틱 (π-aware ticks)

식에 삼각함수나 `pi`가 포함된 경우 가로축은 π의 배수로 레이블링됩니다. 간격은 π/4, π/2, π, 2π, 4π, 8π 중에서 5\~17개의 틱(tick)이 창에 들어오도록 선택됩니다.

```python
print(list(plot(sin(x), verbose=False).plotly.layout.xaxis.ticktext))
print(plot(exp(x), verbose=False).plotly.layout.xaxis.ticktext)
```

가로축이 해당 식의 변수가 아닌 매개변수, 극좌표, 데이터 또는 호출 가능(callable) 플롯에는 적용되지 않습니다.

### 7.2 로그 스케일 제안

식이 `exp`나 `log`에 의해 지배되고 양수 값이 3자릿수(three or more orders of magnitude) 이상에 걸쳐 있는 경우, MathSlate는 로그 스케일을 제안하는 주의사항을 추가합니다. 하지만 이를 자동으로 적용하지는 않습니다. 깨닫지 못한 채 로그 스케일로 된 플롯을 읽는 학습자는 차라리 보기 불편한 선형 플롯을 읽는 학습자보다 상황이 더 나쁩니다.

```python
print([n for n in plot(exp(x), verbose=False).notes if "log scale" in n])
```

---

## 8. `PlotResult`

`plot()`이 반환하는 모든 것.

### 이스케이프 해치 (Escape hatches)

| 속성 | 타입 | 설명 |
| --- | --- | --- |
| `.plotly` / `.figure` | `go.Figure` | 변경 권한이 사용자에게 있음; MathSlate가 소유하지 않음 |
| `.sympy` | `Expr`, 튜플, 또는 `None` | 원본 데이터인 경우 `None` |
| `.numpy` | `(ndarray, ndarray)` | 첫 번째 시리즈의 샘플링된 `x`와 `y` |

```python
f = plot(sin(x)/x, verbose=False)
print(type(f.plotly).__name__, f.sympy, f.numpy[0].shape)
```

```python
f = plot([sin(x), cos(x)], verbose=False)
print(f.sympy)
print(plot([1.0, 2.0], verbose=False).sympy)
```

### 조사(Inspection)

| 멤버 | 반환값 |
| --- | --- |
| `.plan` | 해결된(resolved) `PlotPlan` — 종류, 기호, 범위, 시리즈, 설정 |
| `.notes` | 요약(summary) 줄 아래에 출력된 메시지들의 튜플 |
| `.summary()` | 한 줄로 된 추론 보고서 |

```python
f = plot(tan(x), verbose=False)
print(f.plan.kind, f.plan.symbol.name, f.plan.param_range, f.plan.total_points)
```

### 출력(Output)

| 멤버 | 효과 |
| --- | --- |
| `.python()` | 동일한 역할을 하는 코드를 문자열로 반환 |
| `.show_python()` | 코드를 출력하고 반환 |
| `.show()` | `Figure.show()`로 위임 |
| `._repr_mimebundle_()` | 노트북 렌더링, figure로 위임 |

### `PlotPlan`

| 속성 | 의미 |
| --- | --- |
| `kind` | 디스패치 테이블의 어느 행과 일치하는지 |
| `series` | `Series`의 리스트, 각각 `.name`, `.sample`, `.expr`를 가짐 |
| `symbol` | 축 기호 |
| `param_range` | `(최소, 최대)` |
| `exprs` | 그려진 식들 |
| `y_range` | 모든 시리즈의 가시적 창(visible windows)들의 합집합 |
| `total_points` | 모든 시리즈에 걸친 샘플 수 |
| `notes`, `config` | 상단 설명 참조 |

---

## 9. `show_python()`

플래그십(핵심 기능). 이 프로젝트의 목적이 사용자가 진짜 Python에 유창해지는 것이라는 점을 고려하면, 이것은 결코 부차적인 편의 기능이 아닙니다.

```python
f = plot(sin(x)/x, verbose=False)
code = f.python()
print(code.splitlines()[0])
```

### 보장 사항

**출력된 코드는 한 글자도 빠짐없이 완벽하게 실행됩니다.** 의사코드(pseudocode)도, 생략(ellipses)도, "대략 이런 식"도 아닙니다. 이 규칙은 엄격히 강제됩니다. 테스트 스위트는 모든 플롯 종류에 대해 `show_python()` 출력을 `exec`로 실행하며 그림이 제대로 나오는지 단언(assert)합니다.

```python
f = plot(tan(x), verbose=False)
namespace = {}
exec(f.python(), namespace)
print(type(namespace["fig"]).__name__)
```

### 무엇을 출력하는가

이 기능은 단순하게 `linspace`를 쓰지 않습니다. MathSlate가 조용히 결정했던 모든 것을 코드로 풀어냅니다:

- `import *` 상황에서도 명시적인 `x = sp.symbols('x', real=True)` 선언
- `sp.lambdify(..., 'numpy')`
- 별도로 샘플링하고 `NaN`으로 이어진, 실수 정의역의 연속적인 조각들
- 조각(piece)의 경계에 있지 않은, 별도로 잘라낸 점프(jump) 불연속점
- 왜 존재하는지 설명하는 주석과 함께 설정된 y축 창(window)
- 해당하는 경우 π 틱 값과 레이블
- 실제로 사용된 이름들만 정확히 커버하는 `from sympy import ...`

```python
code = plot(sqrt(x), verbose=False).python()
print("pieces = [(0.0, 10.0)]" in code)
```

플롯의 종류(kind)에 따라 다릅니다: 곡선 및 겹쳐진 곡선은 조각별(piecewise) 샘플러를 출력합니다; 매개변수 및 극좌표 곡선은 매개변수에 대한 두 개의 `lambdify` 호출을 출력하고, MathSlate가 만든 절단선(cut)과 동일하게 자르는 조각별 샘플러를 `t`에 대해 출력합니다; **callable과 데이터는 리터럴 배열 형태로 샘플을 직접 출력하며**, 포인트가 2000개를 넘으면 고르게 축소합니다.

```python
print("pieces = [" in polar(tan(t), verbose=False).python())
```

일반 Python 함수는 소스 코드로 적을 수 없습니다 — `np.sin`은 `sin`으로 출력되고, 람다(lambda)는 쓸만한 이름이 아예 없기 때문입니다. 그래서 이를 호출하는 코드를 출력하면 애초에 가져왔던 네임스페이스에서만 돌아가는 반쪽짜리 코드가 됩니다. 배열 리터럴(literal)은 어디서든 실행됩니다.

### 무엇을 재현하는가

`show_python()`은 그저 *어떤* 그림이 아니라, 정확히 같은 그림을 재현합니다. 테스트 스위트에서는 모든 호출을 두 번 빌드합니다 — 한 번은 MathSlate를 통해, 한 번은 깨끗한 네임스페이스에서 출력된 소스 코드를 실행해서 — 그리고 두 그림의 속성을 하나하나 비교합니다:

| 재현됨 (Reproduced) | 고의로 생략됨 (Deliberately omitted) |
| --- | --- |
| 트레이스 x/y 데이터, 모드, 이름, `connectgaps` | 여백 (margins) |
| 템플릿(template), 제목, `showlegend` | 호버(hover) 모드 및 호버 템플릿 |
| x축 범위, π tickvals/ticktext | 축 제목 |
| 축 눈금 밀도 (`ticks`) | zeroline 스타일링 |
| y축 범위 및 타입 (`log`) | |

생략된 열(column)은 장식(chrome)에 불과합니다. 이것들까지 코드로 출력하면 정작 배워야 할 코드가 장황해지고 그림에는 아무 변화가 없기 때문입니다. 그림을 실질적으로 바꾸는 모든 요소는 왼쪽 열에 있습니다. 두 리스트는 모두 `mathslate.render.options.REPRODUCED` 및 `NOT_REPRODUCED`로 내보내어지며, 테스트 스위트가 이를 검증하므로 조용히 규칙이 망가지는 일은 없습니다.

`plot()`에 전달한 옵션들도 이 보장 사항에 포함됩니다:

```python
f = plot(sin(x), kind="scatter", yscale="linear", title="Points", show_legend=True)
namespace = {}
exec(f.python(), namespace)
print(namespace["fig"].data[0].mode == f.plotly.data[0].mode)
print(namespace["fig"].layout.title.text == f.plotly.layout.title.text)
```

### 단 한 가지 한계점

식(expression) 기반의 플롯에 대해, 출력된 프로그램은 MathSlate의 적응형(adaptive) 그리드가 아닌 **균일(uniform) 그리드**를 사용합니다. 적응형 샘플러를 한 글자도 빠짐없이 그대로 재현하려면 샘플러 코드 자체를 출력해야 할 텐데, 이는 애초의 교육적 목적을 훼손할 것입니다. 따라서 출력된 곡선은 배열상 완벽히 동일하지는 않지만 시각적 정확성 측면에서는 동일한 곡선입니다. 테스트 스위트는 실제로 보이는 창(window) 내에서 두 곡선이 서로의 가시적 대각선 길이의 2% 오차 범위 내에 있도록 요구합니다(point-to-segment 측정 방식).

Callable 및 데이터 플롯은 이런 차이가 없습니다 — 그들의 샘플은 정확하게 그대로 출력됩니다.

---

## 10. `analyze()`

```text
analyze(obj, *ranges, parameters=()) -> Analysis
result.analyze()                    -> Analysis
```

식의 특성들: 근, 극점(extrema), 변곡점(inflection points), 대칭성, 주기성, 점근선, 불연속성, 그리고 단조 증가/감소 구간.

**절대 자동으로 실행되지 않습니다.** 특성 감지는 매번 `plot()`에 포함시키기에는 너무 느리고 노이즈가 많아 둘 다 망칠 수 있습니다. 사용자가 직접 요청해야 합니다.

```python
report = analyze(x**3 - 3*x)
print(report.roots.describe())
print(report.maxima.describe(), "/", report.minima.describe())
```

플롯에서 호출할 경우, 분석되는 범위는 **여러분이 보고 있는 창(window)**입니다 — `sin(x)`의 근은 전적으로 여러분이 어디를 바라보고 있는지에 달려 있습니다:

```python
print(len(plot(sin(x), (x, 0, 3.5), verbose=False).analyze().roots))
print(len(plot(sin(x), (x, -10, 10), verbose=False).analyze().roots))
```

### 10.1 정확한 값, 또는 '근사치' 표시

모든 속성은 우선 기호적으로(symbolically) 계산을 시도합니다. `solveset`은 방정식을 완벽히 풀었을 때는 `FiniteSet`을 반환하고, 그저 식을 재진술했을 뿐일 때는 `ConditionSet`을 반환합니다. 이것이 바로 샘플링(수치적 방법)으로 대체(fallback)하라는 신호입니다.

수치적으로 얻은 답은 **절대 정확한 값으로 포장되어 제시되지 않습니다**. 패널에 출력될 때까지 내내 `approximate=True` 속성을 달고 다닙니다. 증명된 값과 샘플링된 값을 구별하지 못하는 독자는 아예 아무 정보도 얻지 못한 독자보다 더 나쁜 상황에 빠지게 되기 때문입니다.

```python
exact = analyze(x**3 - 3*x)
sampled = analyze(x - cos(x))
print(exact.approximate, exact.roots.symbolic)
print(sampled.approximate, sampled.roots.describe())
```

### 10.1a 답이 근사가 되는 세 번째 이유: 시간

SymPy에는 "포기"라는 개념이 없습니다. 40차 다항식에 대한 `solveset`은 실패하지도, 영원히 멈추지도 않습니다 — **약 80초**가 걸린 뒤 답을 냅니다. 읽는 사람에게는 라이브러리가 고장난 것과 구별되지 않습니다.

그래서 모든 기호 연산 단계에 5초의 예산을 둡니다. 예산이 끝나면 그 속성은 SymPy가 *거부했을 때* 갔을 수치 경로를 대신 타고, 보고서는 둘 중 어느 쪽이었는지 밝힙니다 — 같은 일이 아니기 때문입니다. SymPy의 거부는 수학에 관한 사실이고, 5초를 넘긴 것은 이 컴퓨터에 관한 사실이며, 예산을 늘리면 같은 답이 정확해질 수 있습니다.

```python
big = sum(Integer(i + 1)*x**i for i in range(41))
print([n for n in analyze(big).notes if "budget" in n])
```

일상적인 사용은 이 한계에 근접하지 않습니다 — 테스트 코퍼스에서 가장 느린 plot이 1.7초, 가장 느린 `analyze()`가 3.9초입니다 — 따라서 평소에는 예산이 발동하지 않고 정확한 답은 정확하게 남습니다.

근사보다 기다리는 편이 나을 때 값을 바꿉니다. 최상위가 아니라 `mathslate.core`를 통해 접근하는데, 이는 한계에 부딪힌 드문 세션을 위한 조절 장치이지 학습자가 만나야 할 것이 아니기 때문입니다:

```python
from mathslate.core import get_symbolic_budget, set_symbolic_budget

set_symbolic_budget(30)          # 초
print(get_symbolic_budget())
set_symbolic_budget(None)        # 제한 없음
print(get_symbolic_budget())
set_symbolic_budget(5.0)         # 기본값으로 복귀
```

한 단계가 예산을 넘기면 *그* `analyze()` *호출의 나머지* 기호 시도는 건너뜁니다. 하나의 수식에 여덟 개 속성이면 기호 연산 단계는 십여 개이고, 각각에 5초를 지불하면 5초 제한이 1분짜리 대기로 바뀝니다. 첫 타임아웃은 그 한 단계가 아니라 그 수식에 대한 증거입니다.

이식 가능한 구현을 위해 작업자 스레드를 사용합니다. Python 바이트코드 경계로 돌아오지 않는 네이티브 코드는 즉시 중단할 수 없습니다. 그런 작업자가 살아 있는 동안에는 SymPy의 프로세스 전역 캐시를 두 스레드가 동시에 건드리지 않도록 이후 기호 연산을 거부합니다. 다시 시작할 수 있는 별도 프로세스로 기호 연산을 격리하는 것이 네이티브 코드에도 강제 시간 제한을 적용하기 위한 장기 방향입니다.

### 10.2 각 속성의 의미

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `.roots` | `Points` | 범위(window) 안의 실수 정의역 내에 있는 0(근)들 |
| `.maxima` / `.minima` | `Points` | 실제로 방향이 바뀌는 정류점(stationary points) |
| `.inflections` | `Points` | 단순히 `f'' = 0`인 곳이 아니라, 실제로 오목함/볼록함이 변하는 곳 |
| `.discontinuities` | `Points` | 기호적 극점(poles), 그리고 §6.4의 탐지기로 찾은 점프(jumps) |
| `.asymptotes` | `tuple[Asymptote, ...]` | 수직, 수평, 그리고 사선 점근선 |
| `.symmetry` | `Symmetry` | `even`(우함수), `odd`(기함수), `neither`(둘 다 아님), 또는 `unknown`(알 수 없음) |
| `.periodicity` | `Periodicity` | 주기, 또는 `None` |
| `.increasing` / `.decreasing` | intervals | 방향이 바뀌는 곳, 극점, 정의역 경계에서 분할된 구간들 |
| `.notes` | `tuple[str, ...]` | 무엇을 계산할 수 없었는지, 그리고 그 이유 |
| `.sympy` | `Expr` | 탈출구(escape hatch) — 식(expression) 그 자체 |
| `.approximate` | `bool` | *어느 하나라도* 샘플링에서 나온 값인지 여부 |
| `.provenance()` | pairs | 어떤 호출이 각 속성을 생성했는지 — §10.4 |
| `.python()` | `str` | 실행 가능한 소스 코드 형태의 호출들 — §10.4 |

순진한 접근법으로는 오답을 낼 수 있는, 꼭 알아두어야 할 세 가지 경우가 있습니다:

```python
print(analyze(x**4).inflections.describe())      # f'' = 12x**2는 원점에서 0이 됩니다
print(analyze(x**3).maxima.describe())           # 기울기가 0이 되지만 방향은 바뀌지 않습니다
print(analyze(sin(x)).asymptotes)                # 진동함: 수렴하는 것이 아니라 유계(bounded)일 뿐입니다
```

`x**4`는 항상 아래로 볼록(concave up)하고, `x**3`은 계속 증가하며, `sin(x)`는 수평 점근선이 없습니다 — SymPy는 무한대에서의 극한을 `AccumBounds(-1, 1)`로 보고하는데, 이는 범위가 제한되어 있다는 뜻이지 곡선이 어느 특정 값에 다가간다는 뜻이 아닙니다.

### 10.3 보고서 읽기

`Points`는 시퀀스(sequence)처럼 작동하며, 각 부분은 스스로를 설명할 수 있습니다:

```python
roots = analyze(x**2 - 1).roots
print(len(roots), list(roots), roots.describe())
```

`.text()`는 전체 패널을 일반 텍스트로 보여주며, 노트북 환경에서는 PRD §5.6의 요구사항에 따라 **접힌(collapsed)** 패널 형태로 렌더링됩니다.

```python
print(analyze(1/x).text())
```

### 10.4 "이 결과는 어떻게 나왔나요?"

이 질문에 대해서는 정직한 대답과 부정직한 대답이 있습니다. 이 차이를 명확히 아는 것이 중요합니다.

**정직한 대답 — 실제로 무엇이 실행되었는가.** 모든 속성은 어떤 호출을 통해 생성되었는지 그 기록을 저장하며, `.python()`은 `show_python()`이 플롯에 대해 수행하는 것과 정확히 똑같은 방식으로 이를 실행 가능한 소스 코드로 출력합니다(§9):

```python
report = analyze(x**3 - 3*x)
for label, how in report.provenance():
    print(f"{label}: {how}")
```

```python
print(analyze(x**2 - 1).python())
```

출력된 코드는 **원형 그대로 실행되며 똑같은 값을 재현합니다** — 이것은 §9와 같은 보장 사항입니다. `solveset`으로 해결하지 못해 샘플링으로 구한 속성들은 리터럴(literal)로 값을 출력하며, 탐색에 실패한 `solveset` 호출을 마치 성공한 것처럼 출력하는 거짓말을 피하기 위해 그 옆에 탐색 과정을 설명하는 주석을 남깁니다.

**부정직한 대답 — 풀이 과정(worked solution).** MathSlate는 이를 제공하지 않으며, 이것은 채워야 할 빈 구멍이 아닙니다. SymPy에서 제공하는 유일한 단계별 엔진은 `sympy.integrals.manualintegrate`인데, 이마저도 적분만 지원하며 `sqrt(x**3 + 1)` 같은 평범한 케이스에서도 포기해버립니다(`DontKnowRule`). `analyze()`는 적분을 전혀 수행하지 않습니다 — 방정식 풀이, 미분, 극한 계산만 수행하며 SymPy는 이 셋 중 어느 것에 대해서도 단계별 과정을 제공하지 않습니다. 따라서 "어떻게"에 대한 과정을 제공하려면 그 과정을 *만들어내야(invented)* 하는데, 이것이 바로 PRD §2.2에서 명시적으로 금지한 LLM 생성 풀이입니다.

결론: 작업 영수증(receipt)은 진짜이며 항상 볼 수 있습니다. 반면, 튜토리얼 형태의 설명(풀이 과정)은 정직하게 만들 수 없기 때문에 아예 제공되지 않습니다. §16과 PRD §2.2를 확인하세요. 어떤 형태로도 `explain()` 기능은 없으며, 테스트 스위트는 특히 이 객체에 대해 그러한 기능이 없음을 단언(assert)합니다.

---

## 11. `slider()` 및 `animate()`

```text
slider(start, stop, step=None, *, default=None, name=None, label=None) -> Slider
animate(obj, *ranges, over=None, **plot_options)                       -> PlotResult
```

독자가 직접 변경할 수 있는 매개변수입니다. **여러분은 콜백(callback)을 작성할 필요가 전혀 없습니다.**

```python
a = slider(-3, 3, default=1, name="a")
f = plot(a*sin(x))
print(f.interactive, len(f.plotly.frames))
```

`animate()`는 똑같은 그림에 재생 버튼을 추가합니다:

```python
print([b.label for b in animate(a*sin(x)).plotly.layout.updatemenus[0].buttons])
```

### 11.1 어떻게 한 줄로 동작하는가

`Slider`는 `Symbol`이 아니며(범위, 단계, 현재 값을 가짐), SymPy의 `_sympy_()` 후크(hook)에 응답할 때 자신의 기호(symbol)를 반환합니다. 따라서 `a*sin(x)`는 일반 기호를 사용하는 평범한 식을 만들며, 그 이후의 과정에서는 이것이 위젯과 연결되어 있었다는 사실을 알 필요가 전혀 없습니다.

```python
print((a*sin(x)).free_symbols == {a.symbol, x})
```

그 후 `plot()`은 이 기호를 축이 아닌 매개변수로 취급합니다. 이것이 [§5 규칙 2](#5-%EA%B8%B0%ED%98%B8%EC%99%80-%EC%B6%95-%EB%B0%94%EC%9D%B8%EB%94%A9)이며, 이제 알아서 적용됩니다:

```python
print(plot(a*sin(x), verbose=False).plan.symbol.name)
```

우선순위는 다른 모든 곳과 동일하게 적용됩니다 — 명시적인 범위 지정은 슬라이더에 우선하며, `parameters=`에 명시적으로 지정한 값은 슬라이더 자신의 값보다 우선합니다:

```python
print(plot(sin(a), (a.symbol, -1, 1), verbose=False).plan.symbol.name)
```

### 11.2 왜 Plotly 자체 컨트롤이 기본값인가

PRD §6.2는 marimo에는 `mo.ui.slider`를, Jupyter와 Colab에는 `ipywidgets`를, "그 외/정적 내보내기"에는 Plotly 애니메이션 프레임을 할당했습니다. MathSlate는 이 세 가지 이유로 **모든 곳에서 프레임 방식을 사용합니다**:

- 이 방식은 세 호스트(환경) 모두에서 동일하게 작동하므로 노트북 간 이동이 자유롭습니다.
- 어떤 프론트엔드 패키지도 추가로 필요하지 않습니다. 이것이 [§1](#1-%EC%84%A4%EC%B9%98-%EB%B0%8F-%EC%9A%94%EA%B5%AC-%EC%82%AC%ED%95%AD)의 약속입니다.
- 선생님이 학생들에게 배포하는 가장 핵심적인 형태인, 단일 HTML 파일로 출력해도 살아남습니다.

이 방식의 단점은 위치가 필요할 때마다 계산되는 것이 아니라, 미리 계산된다는 것입니다. `slider(0, 1)`은 21개의 프레임을 그리고, `slider(0, 1, 0.25)`는 여러분이 요청한 정확히 5개만 그립니다.

```python
b = slider(0, 1, 0.25, name="b")
print(b.values())
```

실시간으로 다시 계산하려면, `Slider.widget()`이 현재 호스트 환경의 자체 컨트롤을 반환합니다 — marimo에서는 `mo.ui.slider`, Jupyter와 Colab에서는 `ipywidgets.FloatSlider`를 줍니다. 둘 다 필수 종속성이 아니기 때문에, 노트북 환경이 아니라면 에러가 발생합니다.

### 11.3 여러 개의 슬라이더

한 매개변수가 애니메이션으로 변하는 동안 다른 매개변수들은 현재 값을 유지합니다. Plotly의 프레임은 1차원적인 시퀀스이므로, 여러 개의 슬라이더를 그리드로 구성하면 수천 개의 곡선을 미리 계산해야 하는 꼴이 됩니다. `animate(..., over=b)`는 어떤 슬라이더가 동작할지 지정하며, 보고서는 어떤 매개변수들이 고정되었는지 알려줍니다.

```python
c = slider(1, 2, name="c")
print([n for n in plot(a*c*sin(x), verbose=False).notes if "holds" in n])
```

### 11.4 `show_python()`은 움직이는 그림을 출력합니다

멈춰 있는 프레임이라면 "동일한 결과"가 아닐 것입니다(§9). 출력된 코드는 모든 프레임을 포함하고, 똑같은 Plotly 컨트롤을 생성합니다:

```python
f = plot(a*sin(x), verbose=False)
namespace = {}
exec(f.python(), namespace)
print(len(namespace["fig"].frames) == len(f.plotly.frames))
```

### 11.5 슬라이더는 전체 세션 동안 지속됩니다

슬라이더를 만들면 누군가 명시적으로 해제할 때까지 해당 기호가 매개변수로 바인딩됩니다. 이 수명은 고의적입니다 — 매번 `plot(a*sin(x))`를 호출할 때마다 `a`가 매개변수라는 것을 계속 알려주지 않기 위함입니다 — 하지만 이 바인딩이 슬라이더를 만든 셀(cell)의 수명보다 더 오래 지속된다는 것을 의미하기도 합니다.

그래서, 나중에 그 위에서 플롯을 그리려고 했던 변수와 같은 이름의 슬라이더를 만들면 충돌이 발생합니다:

```python raises=UnsupportedInputError
from mathslate.ui import release_all

release_all()
time = slider(0, 10, default=3, name="t")
plot((cos(t), sin(t)))
```

이제 `t`는 어디서나 매개변수이므로, 원은 점 하나로 얼어붙을 것입니다. MathSlate는 이 점 하나를 그리는 대신, 거부하고 빠져나갈 세 가지 방법을 제시합니다:

```python
from mathslate.ui import release_all

# 1. 그래도 강제로 그리기 — 명시적인 범위 지정은 슬라이더를 덮어씁니다
print(plot((cos(t), sin(t)), (t, 0, 6.283), verbose=False).plan.kind)

# 2. 슬라이더 해제하기
release_all()
print(plot((cos(t), sin(t)), verbose=False).plan.kind)
```

세 번째 방법은 슬라이더 자체에 고유한 이름을 부여하는 것으로, 보통은 이것이 원래 의도였을 가능성이 높습니다: `slider(0, 10, name="time")`.

`release(a_slider)`는 한 개만 해제하고, `release_all()`은 세션 내의 모든 슬라이더를 해제합니다. 두 함수 모두 `mathslate.ui`에 있습니다. 두 개의 기호 중 하나만 바인딩하는 것은 충돌이 아닙니다 — 이는 아주 평범한 일이며, 단순하게 곡면(surface)이 곡선(curve)으로 변할 뿐입니다:

```python
from mathslate.ui import release_all

release_all()
height = slider(0, 3, default=2, name="y")
print(plot(x*y, verbose=False).plan.kind)
release_all()
```

---

## 12. `table()` 및 HTML 내보내기

```text
table(obj, *ranges, rows=11, label=None, parameters=()) -> Table
result.table(rows=11)                                   -> Table
result.to_html(path=None, *, standalone=True)           -> str
```

### 12.1 똑같은 함수를 숫자로 읽어내기

그래프는 형태를 보여주고, 표는 값을 보여줍니다. 손으로 푼 답을 검증할 때 필요한 것이 바로 이 두 번째입니다.

```python
print(table(sin(x), (x, 0, 1), rows=3).text())
```

리스트를 전달하면 여러 개의 열(column)이 생성되며, 축은 다른 모든 곳에서와 동일한 규칙으로 선택됩니다(§5):

```python
print(table([sin(x), cos(x)], (x, 0, 1), rows=3).headers())
```

플롯에서 호출할 경우, 현재 보고 있는 화면(window)의 범위를 표로 만듭니다:

```python
print(plot(sin(x), (x, 0, 2), verbose=False).table(rows=3).inputs)
```

식이 실수 범위를 벗어나는 부분은 0이 아니라 **빈칸**으로 처리됩니다 — 거기에 임의로 숫자를 찍는다면 그것은 수학적으로 거짓말을 하는 셈이 될 것입니다:

```python
f = table(sqrt(x), (x, -1, 1), rows=5)
print("—" in f.text(), f.notes[0][:20])
```

이스케이프 해치는 이미 알고 계신 것과 동일하며, `.python()`은 동일한 행을 출력하는 순수한 NumPy 코드를 생성합니다:

```python
inputs, values = table([sin(x), cos(x)], (x, 0, 1), rows=4).numpy
print(inputs.shape, values.shape)
```

### 12.2 수업에 배포할 수 있는 단일 파일

`to_html()`은 플롯을 **완전히 독립적인 단일 페이지**로 작성합니다: Plotly 자체가 파일에 내장되므로, 네트워크 연결이나 아무 설치 없이 바로 열립니다.

```python
page = plot(sin(x), verbose=False).to_html()
print(len(page) > 1_000_000, "<script src=" not in page)
```

`to_html(path)`는 이 파일을 디스크에 저장합니다. `standalone=False`로 설정하면 Plotly를 CDN에서 링크합니다 — 파일 크기는 훨씬 작아지지만 네트워크 연결이 필요해집니다.

이것이 바로 §11의 슬라이더가 노트북 위젯 대신 그림 내부 프레임에 들어가 있는 이유입니다: 위젯은 살아있는 커널이 필요하지만, 학생들에게 배포되는 파일에는 그런 것이 없기 때문입니다. 프레임을 이용한 인터랙티브 플롯은 내보내기 환경에서도 완벽히 살아남습니다.

```python
b = slider(-2, 2, default=0, name="handout")
print("addFrames" in plot(b*sin(x), verbose=False).to_html())
```

---

## 13. `dataset()`, 통계 및 선형 대수

```text
dataset(source, columns=None, encoding=None) -> Dataset
Dataset.fit(model, x=None, y=None, symbol=None, guess=None) -> FitResult
```

### 13.1 기호와 데이터를 잇는 다리

이전까지의 모든 과정은 수식에서 출발했습니다. 하지만 실제 작업은 주로 측정값에서 출발하며, `fit()`은 그 두 세계가 만나는 지점입니다.

```python
readings = dataset({"x": [0, 1, 2, 3, 4], "y": [1.0, 3.1, 4.9, 7.2, 8.9]})
c, d = symbols("c d", real=True)
found = readings.fit(c*x + d)
print(found.describe())
```

핵심은 돌려받는 결과물입니다. `found.expr`은 매개변수가 숫자로 채워진 **일반적인 SymPy 식(expression)**입니다 — 따라서 별도의 변환 없이도 MathSlate의 다른 모든 기능이 똑같이 작동합니다:

```python
print(diff(found.expr, x))
print(analyze(found.expr).roots.describe())
```

매개변수에 대해 **선형(linear)인 모델**은 최소제곱법을 통해 정확하게 해결됩니다 — `x`에 대해 선형인지가 중요한 것이 아니므로 `c*x**2 + d`도 포함됩니다. 선형이 아닌 모델은 `guess`(또는 데이터에서 뽑은 소수의 초기값)에서부터 반복적으로 값을 개선해나갑니다:

```python
growth = dataset({"x": [0, 1, 2, 3], "y": [2.0, 4.0, 8.0, 16.0]})
print(growth.fit(c*exp(d*x)).describe())
```

이 반복 계산은 비용을 낮추는 방향(damped least squares, Levenberg-Marquardt 알고리즘)으로만 작동합니다. 이는 모델 내에 극점(pole)이 있을 때 매우 중요한데, 감쇠(damping)가 없다면 `c/(x + d)`를 계산할 때 단 한 번의 시도 만에 `d`가 극점 밖으로 넘어가버려서 다시는 돌아오지 못할 수 있기 때문입니다. 또한 감쇠만으로는 다른 분지(basin)로 넘어가지 못하므로(예를 들어 `c*exp(d*x)`에서 `d = -3`으로 시작하면 평평한 영역으로 빠져서 계속 그 값만 반환함), 여러 위치에서 탐색을 시작합니다.

```python
print(dataset({"x": [1, 2, 3, 4], "y": [1.0, 0.5, 0.3333, 0.25]}).fit(c/(x + d)).describe())
```

신뢰하기보다는 직접 적합도를 판단할 수 있도록 `.residuals`와 `.r_squared`가 함께 반환됩니다. 상수 데이터에 대해 R²는 1이나 0이 아니라 `None`을 반환합니다. R²는 "평균값으로 예측하는 것"과 비교하는 지표인데, 애초에 비교할 대상이 없기 때문입니다.

다음의 두 가지 경우에는 엉망인 답을 내놓기보다는 명시적으로 거부합니다: 매개변수를 도저히 분리할 수 없는 모델(`c*d*x` — 오직 둘의 곱만이 곡선을 움직임), 독립변수 열과 종속변수 열이 동일한 경우(이 경우 항상 R² = 1로 '성공'하며 아무 의미도 없음).

### 13.2 데이터셋 구축하기

매핑(딕셔너리), 2차원 배열, 또는 첫 번째 행이 열 이름인 CSV에서 생성합니다.

```python
print(dataset({"t": [0, 1], "v": [2.0, 4.0]}).names)
```

숫자가 아닌 셀은 에러 대신 `NaN`이 됩니다 — 내보낸 시트에는 종종 메모가 포함되어 있는데, 단 하나의 셀 때문에 전체 파일을 거부하는 것은 아무에게도 도움이 되지 않기 때문입니다.

```python
print(readings.describe())
```

#### CSV 인코딩

Excel에서 내보낸 CSV는 보통 UTF-8이 아닙니다. 한국어 Windows에서는 cp949, 유럽 상당 지역에서는 cp1252이고, UTF-8인 경우에도 BOM(byte-order mark)이 붙습니다. 이 모두는 따로 알려주지 않아도 읽힙니다 — UTF-8을 먼저 시도하고, 그다음 스프레드시트가 쓰는 인코딩들, 마지막으로 `latin-1`을 시도합니다:

```python
from pathlib import Path
from tempfile import mkdtemp

sheet = Path(mkdtemp()) / "측정값.csv"
sheet.write_bytes("시간,측정값\n0,1\n1,3\n2,5\n".encode("cp949"))

korean = dataset(str(sheet))          # 인코딩에 대해 아무것도 말하지 않음
print(korean.names, korean["측정값"].tolist())
```

인코딩을 알고 있거나 추측이 잘못됐을 때는 직접 지정합니다. 지정하면 추측이 꺼지므로, 맞지 않을 경우 표준 라이브러리 깊은 곳에서 나온 맨몸의 `UnicodeDecodeError` 대신 파일 이름과 이 인자가 담긴 메시지로 실패합니다:

```text
dataset("readings.csv", encoding="cp949")
dataset("exported.csv", encoding="utf-16")
```

UTF-16과 UTF-32는 반드시 지정해야 합니다. 이들은 NUL 바이트로 가득한데, MathSlate는 바로 그 NUL 바이트로 텍스트 파일과 바이너리 파일을 구분합니다. 이 검사가 없으면 `latin-1`이 *어떤* 바이트 열이든 받아들이므로, PNG를 넘겼을 때 `\x89PNG`라는 이름의 1열 데이터셋이 만들어지고 에러는 어디에도 나지 않았습니다:

```python raises=UnsupportedInputError
picture = Path(mkdtemp()) / "chart.csv"
picture.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
dataset(str(picture))
```

### 13.3 플로팅하기

`Dataset`은 스프레드시트처럼 읽힙니다: 첫 번째 열이 가로축, 나머지는 세로축이며 축의 제목은 열의 이름에서 가져옵니다.

```python
print(plot(readings, verbose=False).plan.kind)
```

`kind="hist"`와 `kind="box"`는 각 열의 모양과 분포를 묻는 완전히 다른 성격의 질문입니다:

```python
print(plot(readings, kind="hist", verbose=False).plan.kind)
print(plot([1.0, 2.0, 2.0, 3.0, 9.0], kind="box", verbose=False).plan.kind)
```

히스토그램은 쌓아 올리지 않고(stacked) 서로 겹쳐서(overlaid) 그립니다. 두 히스토그램을 쌓아 올리는 것은 두 값의 합에 대해 묻는 것인데, 이것은 여러분이 묻고자 했던 내용이 아닐 것입니다.

### 13.4 수행하는 변환 자체로서의 행렬

페이지에 나열된 숫자 행렬은 어떠한 그림도 아닙니다. 이 행렬이 의미를 가지는 순간은 그것이 공간을 어떻게 보내는지를 관찰할 때입니다. 2×2 `Matrix`는 단위 정사각형(unit square), 그것의 변환 결과(image), 그리고 고유벡터(eigenvectors, 늘어나기만 하는 방향)로 그려집니다.

```python
f = plot(Matrix([[2, 1], [1, 3]]), verbose=False)
print(f.plan.kind, len(f.plan.eigen))
print([n for n in f.notes if "determinant" in n])
```

회전 행렬은 아예 아무것도 그리지 않는 대신 이 상황을 글로 명확히 알려줍니다:

```python
print([n for n in plot(Matrix([[0, -1], [1, 0]]), verbose=False).notes if "turns" in n])
```

종횡비(aspect ratio)는 고정되어 있습니다. 그렇지 않으면 눈에 보이는 왜곡이 행렬의 변환이 아니라 플롯의 왜곡일 것이기 때문입니다.

### 13.5 선택적 어시스턴트

```python
from mathslate.ai import ask, PROVIDERS
print([p.name for p in PROVIDERS])
```

노트북에서는 안내 패널로 시작하는 것이 가장 간단합니다.

```python requires=ipywidgets
from mathslate.ai import assistant
assistant("탄젠트 함수를 한 주기 동안 그려줘")
```

패널은 provider 선택, API 키 발급 페이지 링크, 마스킹된 키 입력, 요청
중/성공/인증/할당량/시간 초과/빈 응답 상태를 제공합니다. 생성 코드는 별도의
**Validate & Run** 조작 전에는 실행하지 않습니다. 로컬 컴퓨터에서 **이 기기에
저장**을 선택하면 운영체제 자격 증명 저장소를 사용하며, JupyterLite에서는
영구 키 저장을 비활성화합니다.

`mathslate.ai`는 질문을 MathSlate 코드로 바꿉니다. 구조적으로 세 가지 사실이 보장됩니다:

- **코어는 이 모듈을 절대 가져오지(import) 않습니다.** PRD §8은 학교나 회사의 네트워크가 AI 엔드포인트를 차단할 수 있으므로 MathSlate가 완전히 오프라인 상태에서도 동작해야 한다고 명시합니다. `import mathslate`는 이 모듈에 도달하지 않으며 제공자 SDK들은 필요한 호출 내부에서 임포트됩니다. 테스트 스위트는 서브프로세스에서 이 사실을 검증합니다.
- **코드를 반환할 뿐, 실행하지 않습니다.** `ask(...)`는 읽을 수 있는 `Suggestion`을 반환합니다. 명시적으로 `Suggestion.run()`을 호출하면 허용 목록 기반의 수식 중심 문법을 검사하고 제한된 builtin으로 실행합니다. import, 파일·네트워크 접근, dunder 조회, 반복문과 간접 호출은 거부됩니다. SymPy가 코드로 평가할 문자열도 리터럴이든 실행 중에 만들어진 값이든 거부되며, 격리 프로세스에는 자격 증명 환경변수가 전달되지 않습니다. 신뢰하는 코드에 한해서만 `run(unsafe=True)`로 제한 없는 Python 실행을 선택할 수 있습니다.
- **제공자는 포함되어 있지 않습니다.** Claude, OpenAI, Gemini가 지원됩니다; 이 중 하나를 설치하고 키를 설정하세요. 키는 제공자의 환경 변수나 `ask()`/`configure()`의 `api_key=` 인자로 설정할 수 있습니다. 설정된 것이 없다면 에러 메시지가 이 상황을 정확히 알려줍니다.

```text
ask("plot sine", provider="openai", api_key="sk-...")
configure(provider="openai", api_key="sk-...")
configure(provider="gemini", api_key="...", remember=True)
```

**명시적인 키는 명시적인 제공자를 필요로 합니다.** 키는 특정 회사의 자격 증명인데, 문자열 그 자체만으로는 어떤 회사의 키인지 확실히 알 수 없습니다; 엉뚱한 회사에 키를 보내면 알지도 못하는 회사에 비밀을 넘겨주는 꼴이 됩니다. 그래서 두 개 이상의 제공자 SDK가 설치된 상황에서 `provider=` 없이 `api_key=`만 주어지면, 추측하는 대신 이를 거부합니다. 오직 하나의 SDK만 설치되어 있다면 모호함이 없으므로 해당 키를 사용합니다.

`configure()`에서 `None`은 "원래 상태로 두기"를 의미하며, 키를 삭제할 수 없습니다. 대신 세션에서 키를 지우려면 `forget()`을 사용합니다. `forget(persistent=True)`는 운영체제 자격 증명 저장소에 기억된 키도 제거합니다. `configured()`는 키를 인쇄하지 않고 현재 보관 중인 설정을 보고합니다.

명시적인 세션 키는 해당 키가 설정된 제공자에 계속 남습니다. 적절한 키를 함께 제공하지 않고 제공자만 변경할 경우, 이전 키를 새로운 제공자로 넘겨주지 않습니다: 새 제공자는 이미 환경 변수가 설정되어 있어야 하며, 그렇지 않으면 변경이 거부됩니다. 마찬가지로 각 호출에 사용되는 `api_key=`는 다른 세션 제공자를 조용히 상속하지 않습니다; 여러 SDK가 설치되어 있다면 그 키에 매칭되는 `provider=`를 함께 명시해야 합니다.

```python
from mathslate.ai import configure, configured, forget

configure(model="some-model")
print(configured()["model"])
forget()
print(configured())
```

```python
from mathslate.ai import system_prompt
print(len(system_prompt()) < 4000)
```

프롬프트는 실제 디스패치 규칙에서 생성되므로, 어시스턴트가 플롯의 새로운 종류를 모른 채로 놔두는 커밋은 불가능합니다. 프롬프트가 짧은 이유는 API가 작기 때문입니다 — 그리고 이것이 API를 작게 유지하려는(PRD 목표 5) 가장 강력한 근거이기도 합니다.

#### 13.5.1 코드 말고 다른 것을 묻기

`ask()`는 여러분이 읽을 코드를 씁니다. 모듈의 나머지는 한 가지 기준으로 나뉩니다. 모델에게 **만들어 달라**고 해서 여러분이 검토해야 하는 것인지, 아니면 MathSlate가 **이미 계산해 둔 것을 읽어 달라**고 하는 것인지입니다. 후자가 더 안전한 일이며, 아래 대부분이 그쪽에 속합니다.

**`explain()` — 무엇이 잘못됐고, 어떻게 고치는지.** 초보자의 첫 실수는 대개 모양(shape) 실수이고, MathSlate는 이미 그것을 정확히 지적합니다. 빠진 것은 그 문단을 다시 코드로 옮기는 단계뿐입니다.

```text
plot(sin(x), 0, 6.28)
# TypeError: a range must be written (symbol, lo, hi); got 0

from mathslate.ai import explain
explain()
```

인자 없이 부르면 파이썬이 방금 보고한 예외를 읽으므로, 실패한 셀 다음에는 그냥 `explain()`이면 됩니다. 잡아 둔 예외를 넘기거나, `code=`로 실행하지 않고 코드만 검토하게 할 수도 있습니다.

근거가 되는 것은 모델이 기억하는 MathSlate가 아니라 **에러 메시지 자체**입니다. MathSlate가 낸 에러라면 그 메시지가 권위를 가지므로 모델은 그것을 고친 코드로 옮기기만 하면 됩니다. 그렇지 않은 경우 — `solve(x**2 - 4 = 0)`의 `SyntaxError`, SymPy 내부의 `TypeError` — 모델은 일반적인 파이썬 지식으로 추론하는 것이므로 그렇다고 말하도록 요구됩니다. 어느 쪽인지는 예외의 클래스가 아니라 **어느 프레임이 던졌는지**로 판단합니다. 모양이 잘못된 범위는 의도적으로 평범한 `TypeError`를 던지며(§4.1), 그것이 바로 이 기능이 존재하는 이유이기 때문입니다.

**`describe()` — 그 답이 무슨 뜻인지.** 근은 이미 풀렸고 불연속점도 이미 찾았습니다. 빠진 것은 그것이 결국 무엇을 뜻하는지 말하는 문장뿐이고, 거기에는 계산이 필요 없습니다.

```text
from mathslate.ai import describe
describe(analyze(x**3 - 3*x))
describe(plot(tan(x)), "선이 왜 끊겨 있나요?")
```

모델은 완성된 숫자를 받을 뿐 계산하지 않습니다. `facts()`가 바로 그 전달 내용이며, 그 자체로 읽을 가치가 있습니다 — "무엇을 보냈는가"에 대한 정직한 답이기 때문입니다.

```python
from mathslate.ai import facts

report = facts(analyze(x**3 - 3*x))
print(report["roots"]["exact"], report["roots"]["approximate"])
```

모든 속성이 `approximate`와 그 근거인 `method`를 함께 싣고 있어서, 풀어낸 답과 샘플링한 답이 똑같이 제시되는 일이 없습니다. 이곳은 이 모듈에서 모델의 답변을 초안이 아니라 **답으로** 보여주는 유일한 자리입니다. 그리고 `Analysis.rows()`, `Table.text()`, `PlotResult.summary()`는 같은 사실을 손으로 쓴 형태로, 네트워크 없이 그대로 남아 있습니다.

**`Suggestion.repair()` — 에러를 근거로 한 번 더.** 반쯤 기억하는 API에 대고 코드를 쓰는 모델은, 실제로 무슨 일이 일어났는지 보여주면 두 번째에 훨씬 가까워집니다.

```text
draft = ask("plot the tangent")
try:
    draft.run()
except MathSlateError as failure:
    better = draft.repair(failure)
```

루프는 의도적으로 닫지 않았습니다. `repair()`는 실행하는 대신 새 `Suggestion`을 돌려줍니다. 자동화할 가치가 있는 것은 재시도이지, 확인 단계를 건너뛰는 것이 아니기 때문입니다. 원래 질문이 에러와 함께 전달되므로, 에러만 없애려고 더 쉬운 문제를 슬쩍 푸는 수정은 프롬프트가 거부합니다.

**`ask(..., about=result)` — 지시 대상이 있는 후속 질문.**

```text
drawn = plot(sin(x)/x)
ask("같은 것을 로그 스케일로 보여줘", about=drawn)
```

전달되는 것은 `facts()`가 만드는 계산된 요약입니다. 맥락은 수집되는 것이 아니라 **지목되는** 것입니다. 옆에서 질문했다는 이유만으로 세션에 관한 정보가 기기를 떠나지 않습니다. `assistant(question, about=...)`은 같은 맥락을 패널로 이어 줍니다.

**`suggest_model()` — 어떤 곡선을 맞출 것인가.** 모델 형태를 고르는 일은 수학 이전의 단계이고, 초보자에게 근거가 가장 없는 단계입니다. 그 모양은 숫자 안에 있으므로, 모델에게 추측시키는 대신 MathSlate가 **측정**합니다. 모델 계열이란 결국 데이터를 직선으로 펴 주는 변환입니다.

```python
import numpy as np
from mathslate.ai import fit_evidence

xs = np.linspace(1.0, 5.0, 20)
readings = dataset({"x": xs, "y": 2 * np.exp(0.7 * xs)})

straightness = fit_evidence(readings)["straightness"]
print(max((name for name, r in straightness.items() if r),
          key=lambda name: straightness[name]["r"]))
```

지수 데이터에서 `log(y) ~ x`는 1.000이고 나머지는 0.94 근처에 머무르므로, 계열은 의견이 아니라 측정 결과입니다. 각 수치는 `rows_used`를 `of`와 함께 보고합니다. `log`는 0 이하의 값을 전부 버리므로, 0을 지나는 데이터에서 지수 상관계수는 양수 꼬리만을 설명하며 꼬리는 전체보다 곧기 쉽기 때문입니다. `suggest_model(readings)`은 이 숫자들을 모델에게 넘기고, 모델은 계열의 이름을 붙여 `.fit(...)` 호출을 씁니다.

#### 13.5.2 외부 에이전트를 위한 도구로서의 MathSlate

위의 모든 것은 바깥을 향합니다 — MathSlate가 모델에게 묻습니다. `mathslate.ai.tools`는 안쪽을 향합니다. 에이전트가 MathSlate에 식을 건네면, 그럴듯한 기억이 아니라 **계산된 답**을 받습니다.

```python
from mathslate.ai import call, tool_names

print(tool_names())
print(call("mathslate_analyze", {"expression": "x**2 - 2"})["roots"]["exact"])
```

`tool_schemas("anthropic")`과 `tool_schemas("openai")`가 하나의 정의에서 각 제공자 형식을 생성합니다. `mathslate_plot`은 이미지를 반환하지 않습니다 — 요약, MathSlate가 붙인 노트, 그리고 동등한 평범한 프로그램을 돌려줍니다.

인자는 모델에서 와서 SymPy에 도달하며, 그곳에서 문자열은 코드로 평가됩니다. 그래서 도구 호출에는 생성된 제안과 정확히 같은 신뢰 — 즉 없음 — 만 부여됩니다. 모든 요청은 MathSlate 소스로 조립되어 제안과 동일한 허용목록을 통과하고, 동일한 격리 프로세스에서 동일한 시간 예산 아래 실행됩니다. 기호 이름은 식별자여야 하고 경계값은 숫자로 다시 출력되므로, 어느 쪽도 소스를 실어 나를 수 없습니다. 거부는 예외가 아니라 `{"error": ...}`로 돌아옵니다. 호출자가 그것을 읽고 스스로 고칠 수 있는 에이전트이기 때문입니다.

### 13.6 워크시트

```python
from mathslate.classroom import worksheet

page = worksheet([
    "Where does sin(x)/x go at zero?",
    ("The graph", plot(sin(x)/x, verbose=False)),
    ("The numbers", table(sin(x)/x, (x, -1, 1), rows=5)),
], title="Limits")
print(len(page))
```

작성한 순서대로 그래프, 표, 분석 결과와 텍스트 단락을 하나의 페이지로 묶습니다. `page.save("handout.html")`은 이를 파일로 저장합니다. Plotly는 페이지 내 그림이 몇 개든 상관없이 **단 한 번만** 내장되므로, 그림이 10개라고 해서 파일이 40MB로 불어나지 않습니다. 또한 네트워크 연결이나 추가 설치 없이도 완벽하게 열리며, 여기에는 슬라이더도 포함됩니다 — 이것이 바로 §11.8에서 노트북 위젯 대신 내부 프레임으로 슬라이더를 구성했던 이유입니다.

네트워크가 연결된 환경에서 파일을 작게 만들려면 `PlotResult.to_html()`과 같은 `standalone=False`를 사용합니다:

```python
small = worksheet([plot(sin(x), verbose=False)], standalone=False)
print("cdn.plot.ly" in small.html())         # 버전이 고정된 Plotly CDN 링크
page.preview(standalone=False)              # 작은 노트북 출력
```

기본값은 계속 `standalone=True`이며 CDN 페이지는 네트워크 연결이 필요합니다.

텍스트는 해석되는 것이 아니라 이스케이프(escape) 처리됩니다: 선생님이 입력한 `<`는 HTML 태그가 아니라 언제나 "작다"를 의미하는 부등호입니다.

**워크시트를 보는 세 가지 방법 (서로 교환할 수 없습니다).** `html()`은 문서 타입, `<head>`, 스타일시트, Plotly 사본 등을 모두 포함한 완전한 하나의 문서입니다 — 이는 파일로는 적합하지만 브라우저가 외부 태그를 떼어내고 스타일시트만 남겨 전체 노트북의 모양을 망가뜨릴 수 있으므로 노트북의 셀에는 부적합합니다.

| 호출 | 결과물 |
| --- | --- |
| `page.save(path)` | 학생들에게 나눠줄 파일 저장 |
| `page.preview()` | 노트북 안에서 안전한 `<iframe>` 내부의 페이지 |
| `page` 출력하기 | 페이지에 무엇이 들어 있는지 보여주는 짧은 요약 카드 |

```python
from mathslate.classroom import worksheet

page = worksheet([("A graph", plot(sin(x), verbose=False))], title="One plot")
print(page.preview(height=400).startswith("<iframe"))
print("<!DOCTYPE" not in page._repr_html_())
```

---

## 14. 오류 (Errors)

모든 오류는 `mathslate.errors.MathSlateError`를 상속합니다.

| 에러 | 발생 조건 | 전형적인 해결 방법 |
| --- | --- | --- |
| `AmbiguousAxisError` | 축 개수보다 자유 기호가 많고 이를 결정할 기준이 없을 때 | 명시적 범위를 주거나, `parameters=`로 값을 고정하세요 |
| `UnsupportedInputError` | 어느 디스패치 행에도 맞지 않는 입력, 잘못된 옵션 값, 혹은 아무 의미 없는 범위나 매개변수 | §3, §4, §5를 확인하세요 |
| `NotYetImplementedError` | 다음 마일스톤에 예정된, 아직 미구현된 문서화된 API를 호출했을 때 | 더 이상 남아있지 않습니다 — §16 |
| `SamplingError` | 화면에 그릴 만한 유한한 점이 전혀 없을 때 | 지정한 범위와 함수의 실수 정의역을 확인하세요 |

`AmbiguousAxisError`는 추가로 `.question`과 `.candidates` 속성을 가집니다.

```python
from mathslate.errors import MathSlateError, SamplingError

try:
    plot(sqrt(-1 - x**2), (x, -1, 1))
except SamplingError as error:
    print(isinstance(error, MathSlateError), str(error)[:30])
```

```python raises=UnsupportedInputError
plot({"not": "plottable"})
```

```python raises=UnsupportedInputError
plot(sin(x), points=0)
```

---

## 15. 프론트엔드 어댑터

특정 노트북 환경과 결합하게 되면 반응형 위젯을 얻을 수 있지만, 별도의 설치를 요구하지 않는 다른 환경들을 잃게 됩니다. 그리고 학생들이 가장 많이 접하는 환경은 보통 후자입니다. 그래서 프론트엔드는 런타임에 감지되며, MathSlate를 설치한다고 해서 프론트엔드가 함께 설치되지는 않습니다.

| 환경 | `slider()` 구현 방식 (v0.5) |
| --- | --- |
| marimo | `mo.ui.slider`, 반응형 |
| Jupyter / Colab | `ipywidgets` |
| 그 외 | Plotly 애니메이션 프레임, 독립적인 단일 HTML |

```python
from mathslate.ui import Frontend, detect_frontend

print(detect_frontend() in set(Frontend))
print(frontend_report())
```

이 감지 과정은 어떤 것도 임포트하지 않습니다: `sys.modules`와 IPython 쉘 클래스만을 검사합니다. 수치 해석 과정의 어떤 것도 이를 참조하지 않기 때문에, 모든 환경에서 동일한 결과를 보장합니다 — CI(지속적 통합) 과정에서 일반 Python, Jupyter 커널, marimo 환경 각각에 대해 동일한 핑거프린트(fingerprint)를 확인하여 이를 검증합니다.

---

## 16. 마일스톤

문서화된 모든 기능이 존재합니다. 아직 구축되지 않은 기능들은 지연된 마일스톤의 이름을 명시하며 `NotYetImplementedError`를 발생시키므로, API 리스트를 따라가다 원인을 알 수 없는 `AttributeError`를 마주하게 되는 일은 결코 없습니다.

| API | 마일스톤 |
| --- | --- |
| `analyze()` | **v0.5 — 출시됨**, §10 참조 |
| `slider()`, `animate()` | **v0.5 — 출시됨**, §11 참조 |
| 곡면, 등고선, 음함수 곡선, 3D 공간 곡선 | **v0.5 — 출시됨**, §4.11 참조 |
| `table()`, 단일 파일 HTML 내보내기 | **v0.5 — 출시됨**, §12 참조 |
| `dataset()`, 통계, 선형 대수 | **v1.0 — 출시됨**, §13 참조 |
| 선택적 AI 어시스턴트 | **v1.0 — 출시됨**, `mathslate.ai` |
| 교실 워크시트 | **v1.0 — 출시됨**, `mathslate.classroom` |

**미뤄진 기능은 없습니다.** §2에 명시된 모든 이름은 설명된 대로 동작합니다.

### 영구적으로 범위에서 제외됨

어떤 형태로도 `explain()`이나 "풀이 단계 보기", 또는 자동 생성된 풀이 과정은 존재하지 않습니다. SymPy는 범용 단계별 문제 풀이 엔진을 제공하지 않으며, 이를 어설프게 구현하는 것은 없는 것보다 훨씬 더 해롭습니다. `analyze()`는 도출 과정이 아니라 *객체의 속성*(근, 극값, 변곡점, 대칭성, 주기성, 점근선, 단조 증가/감소 구간)만을 보고합니다.

MathSlate는 또한 CAS(컴퓨터 대수 시스템)가 아니며, 노트북도 아니고, Wolfram과 호환되지 않으며, 채점 도구나 고성능 수치 해석 라이브러리, 또는 GUI 수식 편집기도 아닙니다.

---

## 17. 내부 구조 맵

참고할 수 있을 만큼 충분히 안정적이지만; 공개 API 규칙의 일부는 아닙니다.

| 모듈 | 책임 |
| --- | --- |
| `mathslate/api.py` | 공개 API 표면 |
| `mathslate/core/dispatch.py` | 입력 → 플롯 종류 추론, 완성된 `PlotPlan` |
| `mathslate/core/binding.py` | 기호 → 축, 범위, 관례적인 순서 배정 |
| `mathslate/core/sampling.py` | §6의 6단계 알고리즘 |
| `mathslate/core/_sets.py` | SymPy의 `Set` 대수 → 일반 float 범위 구간 |
| `mathslate/render/plotly_backend.py` | 계획(plan) → `go.Figure`; **유일한 Plotly 임포트** |
| `mathslate/render/options.py` | `codegen`과 공유되는 그림 수준의 결정, 이 둘이 엇갈리는 일 방지 |
| `mathslate/render/axes.py` | π 틱, 로그 스케일 제안 |
| `mathslate/codegen.py` | `show_python()` |
| `mathslate/result.py` | `PlotResult` 및 이스케이프 해치 |
| `mathslate/ui/adapters.py` | 런타임 프론트엔드 감지 |
| `mathslate/_text.py` | 인코딩으로부터 안전한 콘솔 출력 |
| `mathslate/errors.py` | 에러 계층 구조 |

`core`는 오직 SymPy와 NumPy에만 의존하며 — 절대 Plotly나 프론트엔드에 의존하지 않습니다. 향후 백엔드 교체 비용을 낮게 유지하기 위해 Plotly는 단 하나의 파일에 격리되어 있습니다.

---

## 18. 보장 사항 및 보장하지 않는 사항

### 보장 사항

- **§3의 디스패치 규칙은 안정적입니다.** 메이저 버전 업데이트 전까지는 변경되지 않습니다.
- `show_python()`**의 결과는 모든 플롯 종류에 대해 문자 그대로 완벽히 실행됩니다.**
- **불연속점 너머로 선이 그려지는 일은 결코 없습니다.** 200개의 코퍼스와 30-case의 직접 검증된 리뷰를 통해 스위트 환경에서 확인되었습니다.
- **MathSlate 설치 시 어떤 프론트엔드 패키지도 설치되지 않습니다.** CI(지속적 통합)에서의 깨끗한 설치 작업을 통해 확인되었습니다.
- **모든 결과는 적용 가능한 경우** `.sympy`**,** `.plotly`**,** `.numpy`**를 노출합니다.**
- **어떤 노트북 환경에서 실행하든 완벽히 동일한 결과를 생성합니다.**
- **UTF-8이 아닌 터미널에서 콘솔 출력이 중단되는 일은 절대 없습니다.** 스트림이 인코딩할 수 없는 장식용 유니코드는 ASCII로 자동 변환됩니다.

### 보장하지 않는 사항

- **MathSlate와** `show_python()` **간의 완벽한 배열 단위 일치** — §9를 참조하세요.
- **정확한 샘플 개수.** 적응형 샘플러에 의해 결정되므로 버전마다 달라질 수 있습니다; 안정적인 것은 *그려진 그림 자체*입니다.
- **교육용 목적을 뛰어넘는 고성능.** 목표는 인지할 수 있는 지연 없이 ≤10⁶ 포인트를 처리하는 것입니다; MathSlate는 고성능 수치 해석 라이브러리가 아닙니다.
- **임의의 난해한 식에 대한 기호적 분석.** SymPy가 정의역을 판별할 수 없을 경우, MathSlate는 수치적 탐지로 전환하며 이 사실을 주의사항(notes)에 남깁니다. 주의사항을 꼭 확인하세요.

---

## 함께 보기

- [튜토리얼](tutorial_ko.md) — 가이드가 포함된 단계별 소개입니다.
- [PRD](../mathslate_prd_0.3.md) — 프로젝트가 왜 이런 형태로 설계되었는지, 달성하지 않기로 한 목표는 무엇인지, 그리고 주요 인수 기준(acceptance criteria)에 대한 근거와 구현 상태.
- `README` — 짧은 요약본.
