# MathSlate 튜토리얼

**첫 번째 그래프 그리기부터 실제 Python 코드를 읽는 것까지, 한 번에 끝내는 가이드입니다.**

이 튜토리얼을 진행하려면 Python과 약 30분의 시간이 필요합니다. NumPy나 Matplotlib, 혹은 `lambda`가 무엇인지 몰라도 괜찮습니다. 여기서 다루는 모든 내용은 Jupyter, Colab, marimo 또는 일반 터미널에서 실행할 수 있습니다.

모든 옵션에 대한 자세한 설명이 필요하다면 [레퍼런스 매뉴얼](manual_ko.md)을 참조하세요. 이 페이지는 둘러보기용입니다.

---

## 0. 설치

```bash
pip install mathslate
```

노트북 환경과 관련된 패키지는 함께 제공되지 않습니다. Jupyter나 Colab에서 사용하고 싶다면 다음을 실행하세요:

```bash
pip install "mathslate[starter]"
```

이 한 줄로 Jupyter Lab, 대화형 그래프 컨트롤에 필요한 `ipywidgets`와
`anywidget`, Gemini assistant의 `google-genai`, API key를 기기에 기억하는
`keyring`까지 설치됩니다. 개별 패키지를 따로 설치하거나 import할 필요가
없습니다. 저장소에서 시작한다면 Windows는 `scripts/quickstart.ps1`,
macOS/Linux는 `scripts/quickstart.sh`를 실행하면 같은 설정을 자동으로
만들고 Jupyter Lab을 엽니다.

---

## 1. 첫 번째 그래프

```python
from mathslate import *

plot(sin(x)/x)
```

이게 전부입니다. `import numpy`나 `symbols`, `linspace`, `figure`, `show`는 필요하지 않습니다.

> **marimo에서는 import 문을 직접 작성하세요.** marimo는 셀이 실행되기 전에 별표(*)를 사용한 import를 거부하며, *marimo에서는 허용되지 않는다*고 알린 뒤 셀을 실행하지 않습니다. 이는 선택 사항이 아니며 MathSlate의 문제도 아닙니다(§1.1 참조). 대신 첫 번째 셀에 다음과 같이 작성하세요:
>
> ```python
> from mathslate import (
>     plot, polar, analyze, show_python, set_verbose, get_verbose,
>     slider, animate, table, dataset, frontend_report,
>     sin, cos, tan, atan, exp, log, sqrt, Abs, floor, pi,
>     diff, integrate, limit, solve, simplify, symbols,
>     Eq, Integer, Matrix,
>     x, y, z, t, n, k, theta,
> )
> ```
>
> 그런 다음 이 튜토리얼의 나머지 내용을 그대로 진행하면 됩니다.

여기서 주목할 만한 두 가지가 있습니다.

`x`**는 이미 존재합니다.** MathSlate는 수학자들이 실제로 쓰는 기호인 `x, y, z, t, n, k`, `theta`를 미리 정의해 둡니다. 기호를 따로 선언할 필요가 없습니다.

`sin`**은 MathSlate에서 만든 것이 아니라 SymPy의** `sin`**입니다.** MathSlate는 SymPy를 래핑(wrapping)하는 대신 그대로 다시 내보내므로(re-export), 여기서 배운 모든 것은 SymPy에 대한 지식이 됩니다. 즉, 10년 뒤에도 여전히 사용할 수 있는 라이브러리 지식을 얻는 것입니다.

그리고 더 흥미로운 세 번째 사실이 있습니다: `sin(x)/x`**는** `x = 0`**에서 정의되지 않으며**, 그래프도 이를 알고 있습니다. 존재하지 않는 점을 향해 직선을 그리는 대신 그 위치에서 곡선이 끊어집니다. 이것이 왜 중요한지에 대해서는 §5에서 다시 다루겠습니다.

### 1.1 marimo에서 `import *`가 금지된 이유

여기에는 흥미로운 이유가 있으며, MathSlate의 편의성이 노트북의 설계와 충돌하는 유일한 부분이기 때문에 2분 정도 읽어볼 가치가 있습니다.

marimo는 *반응형(reactive)*입니다: 한 셀을 수정하면 그에 종속된 모든 셀이 자동으로 다시 실행됩니다. 이를 위해서는 셀을 실행하기 전에 각 셀이 어떤 이름을 **정의**하고 **사용**하는지 알아야 합니다. 이 정보를 바탕으로 셀 간의 의존성 그래프를 그리기 때문입니다. marimo는 코드를 실행하는 대신 코드를 읽어서 이를 파악합니다.

`from mathslate import *`는 이러한 방식을 완전히 무력화합니다. 이 방식이 정의하는 이름의 집합은 모듈에 포함된 항목에 따라 달라지며, 모듈을 임포트하지 않고는 알 수 없습니다. 그래서 marimo는 파싱(parse) 단계에서 이를 거부하며, 그래서 "셀이 실행되지 않음(Cell not run)"이라는 메시지가 나타나는 것입니다. 셀의 첫 번째 줄조차 실행되지 않은 상태에서 거부된 것이죠.

이 규칙은 예외 없이 엄격하게 적용되며, 이 라이브러리뿐만 아니라 모든 라이브러리에 적용됩니다. Jupyter, Colab, 일반 Python은 의존성 그래프를 그리지 않기 때문에 이러한 제약이 없습니다.

해결책은 위에서 언급한 명시적 임포트(explicit import)를 사용하는 것입니다. 그리고 이것이 원래 MathSlate가 권장하는 방향이기도 합니다. 임포트할 대상을 직접 지정하는 것은 실제 Python 코드의 기본입니다. 초보자가 작성할 첫 코드를 짧게 만들기 위해 `import *`가 존재했을 뿐, marimo는 단지 여러분을 조금 더 일찍 성장하게 만드는 것뿐입니다.

---

## 2. 출력된 메시지 읽기

`plot()`은 호출될 때마다 한 줄을 출력합니다:

```text
curve | x ∈ [-10, 10] | 411 samples | 1 discontinuity handled
  · singularities at x = 0
```

이렇게 읽으면 됩니다: *"이것을 단일 2D 곡선으로 판단했습니다. 범위를* `-10`*부터* `10`*까지로 선택했습니다. 등간격이 아닌 411개의 샘플 포인트를 사용했습니다. 함수가 끊어지는 곳을 한 군데 발견해서 선을 끊어 그렸습니다."*

이 줄은 단순한 로그 기록이 아닙니다. 이것이야말로 MathSlate의 핵심 교육 전략을 단 한 줄로 요약한 것입니다: **사용자가 스스로 결정을 내릴 수 있도록, 대신 내려진 결정을 알려줍니다.** 이 메시지에 있는 모든 숫자는 사용자가 직접 제어할 수 있으며, 이 튜토리얼의 나머지 내용 대부분은 이를 다루는 방법에 대한 것입니다.

출력 메시지가 거슬린다면 다음과 같이 끌 수 있습니다:

```python
set_verbose(False)
plot(sin(x))          # 이제 메시지가 출력되지 않습니다.

set_verbose(True)
```

또는 콘솔 대신 결과 객체에서 이 내용을 읽어올 수도 있습니다:

```python
f = plot(sin(x)/x)
print(f.summary())
```

---

## 3. 여러 곡선 그리기와 단 하나의 규칙

여러 곡선을 그리고 싶을 때 기억해야 할 유일한 규칙이 있으며, 이는 Python에서 이미 배운 규칙과 같습니다:

> **리스트(list)**는 *독립적인 여러 개의 항목*을 의미합니다.
> **튜플(tuple)**은 *여러 구성 요소로 이루어진 하나의 항목*을 의미합니다.

따라서 리스트는 여러 곡선을 겹쳐서 그립니다:

```python
plot([sin(x), cos(x), sin(x) + cos(x)])
```

반면 튜플은 하나의 매개변수 곡선으로, 두 식이 이동하는 한 점의 `x`와 `y`를 나타냅니다:

```python
plot((cos(t), sin(t)))
```

첫 번째는 세 개의 곡선을 그립니다. 두 번째는 하나의 원을 그립니다. 같은 두 개의 식이 양쪽 괄호 모두에 들어갈 수 있으며, 어느 것을 의도했는지는 괄호의 종류로 MathSlate에 알려줍니다.

이 규칙 하나로 대부분의 디스패치 규칙을 커버할 수 있습니다. 전체 표는 [매뉴얼](manual_ko.md#3-the-dispatch-contract)에서 확인할 수 있습니다.

---

## 4. 원하는 범위 지정하기

MathSlate는 기본적으로 `-10`부터 `10`까지를 선택했습니다. 다른 범위를 원한다면 직접 지정하세요. 범위는 `(symbol, from, to)` 형태의 튜플로 작성합니다:

```python
plot(sin(x), (x, 0, 6.28))
```

여기서 출력 메시지의 가치가 처음으로 드러납니다. 여러분은 `x ∈ [-10, 10]`을 보았고, 그 범위가 자동으로 결정된 것임을 알게 되었습니다. 이제 여러분이 직접 범위를 지정하고 있습니다. 이 패턴은 이후의 모든 기능에도 동일하게 적용됩니다.

이 외에도 몇 가지를 더 설정할 수 있습니다:

```python
plot(exp(-x**2), (x, -3, 3), title="A bell curve", label="Gaussian")
```

```python
plot(sin(x), points=2000)
```

`points`는 샘플링을 시작할 점의 개수를 설정합니다. 하지만 이 설정이 필요한 경우는 거의 없을 것입니다. 그 이유는 다음 섹션에서 설명합니다.

---

## 5. 그리기 어려운 함수들

이 부분이 정말 중요하며, MathSlate가 플로팅 라이브러리를 감싸는 5줄짜리 래퍼가 아니라 독자적으로 존재하는 이유입니다.

인수 없이 다음 명령들을 실행해 보세요:

```python
plot(tan(x))
```

```python
plot(1/x)
```

```python
plot(floor(x))
```

```python
plot(sqrt(x))
```

```python
plot(x/Abs(x))
```

어떤 일이 **일어나지 않았는지** 살펴보세요.

`tan(x)`는 한쪽 가지의 맨 위에서 다음 가지의 맨 아래로 이어지는 수직선에 가까운 선을 그리지 않았습니다. 이 선은 탄젠트 그래프의 일부가 아니라, 함수가 그 사이에서 연속적인지 묻지 않고 연속된 샘플 점들을 이어버리는 플로터의 부작용입니다. 대부분의 도구가 이 선을 그립니다.

`1/x`는 0에서 선이 연결되지 않고 끊어집니다. `floor(x)`는 지그재그 모양이 아니라 약속된 계단 모양을 볼 수 있도록 모든 정수에서 끊어집니다. `sqrt(x)`는 0보다 작은 범위에서는 실수가 아니므로 아예 그려지지 않습니다. 0 근처에서 희미해지거나 0으로 그려지는 것이 아니라, 멈춥니다. 그리고 `x/Abs(x)`는 중앙을 가로지르는 선 없이 `-1`에서 `+1`로 뛰어오릅니다.

MathSlate가 무엇을 찾아냈는지 확인할 수 있습니다:

```python
f = plot(tan(x))
print(f.plan.series[0].sample.breakpoints)
```

```python
f = plot(sqrt(x))
print(f.plan.series[0].sample.domain_intervals)
```

첫 번째는 현재 화면(window) 안에서 발견된 탄젠트의 6개 극점을 출력합니다. 두 번째는 `((0.0, 10.0),)`을 출력하는데, 이는 화면 안에서 `sqrt(x)`가 실수인 유일한 구간을 의미합니다.

**어떻게 알아냈을까요.** 간단히 말하면: MathSlate는 샘플링을 시작하기 전에 SymPy에게 극점과 실수 정의역을 물어보므로, 추측하는 대신 정확한 값을 얻어냅니다. 그런 다음 적응형(adaptive)으로 샘플링하여 곡선이 구부러지는 곳에는 점을 많이 찍고 직선인 곳에는 적게 찍으며, 모든 끊김 지점에서 선을 자릅니다. SymPy가 볼 수 없는 점프(`floor`, `sign`, `Piecewise`)의 경우 기발한 방법을 사용합니다: 의심되는 점프 주변의 구간을 좁혀가며 점프가 어떻게 되는지 지켜봅니다. 진짜 불연속점은 화면을 아무리 확대해도 크기를 유지하지만, 단지 가파른 경사일 뿐이라면 결국 평평해집니다. MathSlate는 "floor"라는 단어를 전혀 알지 못하고도 이 단 하나의 테스트로 모든 계단 함수를 처리합니다.

원한다면 [매뉴얼](manual_ko.md#6-sampling-the-technical-heart)에서 전체 알고리즘을 볼 수 있습니다.

조용히 일어난 또 한 가지 일이 있습니다:

```python
f = plot(tan(x))
print(f.plan.y_range)
```

y-범위를 직접 지정하지 않았을 때, 10⁷로 솟아오르는 단일 극점이 존재한다면 그래프의 나머지 부분이 모두 0 근처의 평평한 선으로 찌그러질 것입니다. MathSlate는 데이터의 중간에서 범위를 골라내어 탄젠트 곡선을 가로로 길게 번진 얼룩이 아니라 원래 모습으로 보여줍니다.

---

## 6. 축에 표시되는 π

여기 틱(tick) 라벨을 살펴보세요:

```python
plot(sin(x))
```

그리고 이것도 보세요:

```python
plot(exp(x))
```

첫 번째 그래프는 `-2π`, `-3π/2`, `-π`와 같이 π의 배수로 표시되어 있습니다. 삼각 함수 그래프를 읽는 사람은 모두 이렇게 생각하기 때문입니다. 두 번째 그래프는 일반적인 숫자가 나옵니다. π는 `exp`와 아무런 관련이 없기 때문입니다.

아무도 이렇게 표시해달라고 요청하지 않았습니다. MathSlate가 표현식 안에 *무엇이 있는지*를 살펴본 것입니다.

단, 절대 **자동으로** 로그 스케일로 전환하지는 않습니다. 다음을 시도해 보세요:

```python
f = plot(exp(x))
print(f.notes)
```

MathSlate는 `exp(x)`가 여러 자릿수(order of magnitude)에 걸쳐 있음을 감지하고, 로그 스케일을 권장하는 *메시지를 글로 남깁니다*. 그러나 스케일을 자동으로 적용하지는 않습니다. 로그 스케일로 된 플롯을 그것이 로그 스케일인지 모르고 보는 학습자는 어색한 선형 플롯을 보는 학습자보다 더 나쁜 상황에 빠지게 됩니다. 원할 때는 직접 지정해서 요청하세요:

```python
plot(exp(x), yscale="log")
```

---

## 7. 원, 나선, 그리고 꽃 모양

매개변수 곡선은 앞서 이미 만나보았습니다:

```python
plot((cos(t), sin(t)))
```

기본 범위를 주목하세요: 삼각 함수 성분이 포함된 매개변수 곡선의 경우 MathSlate는 `-10`부터 `10` 대신 `0`에서 `2π` — 즉, 한 바퀴 — 를 사용합니다.

반지름이 점점 커지면 나선형이 됩니다:

```python
plot((t*cos(t), t*sin(t)), (t, 0, 20))
```

그 다음은 `r = f(θ)` 형태인 극좌표(polar)입니다. 이 경우 MathSlate는 고의로 추측을 거부합니다. `1 + cos(t)`라고 쓰여진 식은 그 자체로 단일 변수에 대한 아주 평범한 함수입니다. 그 식 안에 파동(wave)을 뜻하는지 심장형(cardioid)을 뜻하는지 알려주는 정보는 없습니다. 따라서 사용자가 직접 지정해야 합니다:

```python
polar(1 + cos(t))
```

```python
polar(sin(3*t))
```

심장형과 세 잎 장미입니다. 추론할 수 없는 것에 대한 솔직함은 설계 원칙이지 실수가 아닙니다 — 매뉴얼에서는 이 원칙이 적용되는 두 가지 경우를 구체적으로 언급합니다.

---

## 8. 자신만의 데이터

모든 것이 식으로 표현되는 것은 아닙니다. 숫자로 된 리스트도 사용할 수 있습니다:

```python
plot([2.0, 4.0, 8.0, 16.0, 32.0])
```

리스트 두 개를 묶은 쌍은 x대 y 그래프로 동작합니다:

```python
plot(([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0]))
```

그리고 일반적인 Python 함수도 사용할 수 있습니다:

```python
import numpy as np

plot(np.tanh, (x, -5, 5))
```

이 모든 것이 §3에서 배운 단 하나의 규칙을 따른다는 점에 주목하세요. 숫자로 된 리스트는 여러 숫자의 집합이고, 리스트 두 개의 튜플은 하나의 x-y 데이터셋입니다.

---

## 9. 이 모든 것의 목적

이것이 전체 프로젝트가 구축된 핵심 기능입니다.

```python
plot(sin(x)/x).show_python()
```

위 명령을 실행하면 똑같은 그림을 생성하는 순수 NumPy + SymPy + Plotly 프로그램이 반환됩니다:

```text
# Equivalent code — this runs exactly as printed.
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from sympy import sin

x = sp.symbols('x', real=True)

expr = sin(x)/x
fn = sp.lambdify(x, expr, 'numpy')
...
```

첫 번째 줄의 주석을 글자 그대로 읽으세요. 단순한 예시도, 의사 코드(pseudocode)도, "대략 이런 식"도 아닙니다. 이 코드를 복사해서 파일에 붙여넣으면 곧바로 실행됩니다. 이 튜토리얼과 매뉴얼에 있는 모든 예제, 그리고 모든 플롯 종류에 대한 `show_python()`의 결과는 테스트 스위트에 의해 실행됩니다. 만약 코드가 실행되지 않는다면 빌드는 실패할 것입니다.

이것은 여러분의 첫 `lambdify`이자 첫 `linspace`입니다. 불필요한 정보(noise)가 될 수 있었던 튜토리얼의 첫 줄에 노출되는 대신, 질문할 준비가 된 바로 그 순간에 나타납니다.

이 코드에 *무엇이 포함되어 있는지* 확인해 보세요. 단순히 `linspace(-10, 10, 1000)`을 넣고 잘 되길 바라는 코드가 아닙니다. 정의역의 각 조각, 끊어진 선들, y축 윈도우까지, MathSlate가 여러분을 대신해 내렸던 모든 결정이 코드로 작성되어 있습니다. 이제 여러분이 이 코드를 직접 수정할 수 있습니다. 어려운 함수로 한번 시도해 보세요:

```python
plot(tan(x)).show_python()
```

출력 화면 대신 코드를 문자열로 얻고 싶다면 이렇게 하세요:

```python
code = plot(sin(x)).python()
print(len(code))
```

---

## 10. 외부에서 작업 계속하기

여기의 모든 것은 언제든 빠져나갈 수 있습니다. 반환된 모든 결과 객체는 기반이 되는 진짜 객체들을 포함하고 있습니다:

```python
f = plot(sin(x)/x)

f.plotly      # 수정 가능한 Plotly Figure. 마음대로 조작하세요.
f.sympy       # SymPy 표현식
f.numpy       # 샘플링된 (x, y) 배열
```

따라서 MathSlate가 잘하는 영역에서는 이 툴을 사용하다가, 기능의 한계에 부딪히는 순간 진짜 라이브러리로 자유롭게 넘어가면 됩니다:

```python
f = plot(sin(x)/x)
f.plotly.update_layout(title="Now it is a Plotly problem")
```

```python
f = plot(sin(x))
xs, ys = f.numpy
print(xs.shape, ys.max().round(3))
```

```python
f = plot(sin(x)/x)
print(integrate(f.sympy, (x, 1, 2)).evalf(6))
```

마지막 코드는 잠시 주목할 만합니다: `f.sympy`는 일반적인 SymPy 표현식이므로 SymPy의 모든 기능이 그대로 적용됩니다. `solve`, `diff`, `integrate`, `limit`, `series`, `simplify`와 같은 함수들은 MathSlate에서 다시 내보내어 제공되며 모두 순수 SymPy 그 자체입니다:

```python
print(diff(sin(x)*exp(x), x))
print(solve(x**2 - 5*x + 6, x))
print(limit(sin(x)/x, x, 0))
```

MathSlate는 이 기능들을 새로 작성하지 않았습니다. 이에 대한 의견도 없습니다. 이 점이 중요합니다 — 여러분이 나중에 잊어야(unlearn) 할 것은 없습니다.

---

## 11. 판단하기 모호할 때, MathSlate는 묻습니다.

정말로 모호한 상황을 주었을 때 어떤 일이 일어나는지 확인해 보세요:

```python raises=AmbiguousAxisError
a, b, c = symbols("a b c", real=True)
plot(a*b*c)
```

자유 기호 3개 중 어느 것도 축 이름과 유사하지 않고, 어느 쪽이 가로축인지 알 수 있는 지표가 없습니다. 어느 하나를 골라 사용자가 요구하지 않은 것을 그리는 대신, MathSlate는 메시지 자체가 질문인 에러를 발생시키며 문제를 해결할 수 있는 정확한 호출을 보여줍니다.

```python
a, b, c = symbols("a b c", real=True)
plot(a*b*c, (a, -5, 5), parameters={b: 2, c: 3})
```

`parameters`는 축이 *아닌* 기호의 값을 고정합니다. `slider()`도 사용자의 명시적인 지시 없이 동일한 기능을 수행합니다.

모호하지 않을 경우 MathSlate는 일반적인 관례를 따릅니다: `x, y, z`가 `t, u, v`보다 먼저, `t, u, v`가 `r, theta`보다 먼저 적용되며, 그 다음에는 알파벳순입니다. 따라서 `a*sin(x)`에서는 여러분의 의도대로 `x`가 축이 되고 `a`는 축이 아닙니다.

하지만 자유 기호가 두 개인 경우는 에러가 아닙니다. 곡면(surface)이 됩니다:

```python
plot(x*y)
```

그리고 `kind="contour"`는 동일한 객체를 평면으로 보여줍니다. 등식은 양변이 일치하는 곳에 곡선을 그리고, 3개의 요소를 가진 튜플은 공간 곡선(space curve)을 그립니다:

```python
plot(Eq(x**2 + y**2, 4))
```

```python
plot((cos(t), sin(t), t), (t, 0, 12))
```

---

## 12. 자신의 데이터 도입하기

지금까지의 모든 내용은 식(expression)에서 시작되었습니다. 실제 작업은 보통 측정 데이터에서 시작되며, 이것이 그 두 가지가 만나는 지점입니다:

```python
readings = dataset({"x": [0, 1, 2, 3, 4], "y": [1.0, 3.1, 4.9, 7.2, 8.9]})
c, d = symbols("c d", real=True)
found = readings.fit(c*x + d)
print(found.describe())
```

종이에 쓰듯이 모델을 작성하면 **숫자가 채워진 동일한 수식**을 얻을 수 있습니다. 이는 여전히 SymPy 형식이므로, 이전 9개 섹션에서 배운 모든 내용이 여기에 그대로 적용됩니다:

```python
print(diff(found.expr, x))
```

`plot(readings)`은 각 열을 표시하고, `kind="hist"`는 열의 형태에 대해 묻습니다. 2×2 크기의 `Matrix`는 행렬이 수행하는 변환을 나타내는 모양으로 그려지며, 오직 늘어나기만 하는 방향인 고유벡터(eigenvector)가 함께 표시됩니다.

```python
print(plot(Matrix([[2, 1], [1, 3]]), verbose=False).plan.kind)
```

## 13. 다른 사람에게 전달하기

```python
from mathslate.classroom import worksheet

page = worksheet([
    "Where does sin(x)/x go at zero?",
    ("The graph", plot(sin(x)/x, verbose=False)),
    ("The numbers", table(sin(x)/x, (x, -1, 1), rows=5)),
], title="Limits")
print(len(page), "sections")
```

`page.save("handout.html")`는 네트워크나 추가 설치 없이 (슬라이더를 포함하여) 바로 열리는 하나의 파일을 작성합니다.

`mathslate.ai`라는 선택적 어시스턴트도 사용할 수 있습니다. 질문을 MathSlate 코드로 바꿔줍니다. 이를 사용하려면 API 키와 세 개의 제공자(provider) 패키지 중 하나가 필요하지만, MathSlate 자체에는 아무것도 필요하지 않고 이를 사용하려 들지도 않습니다.

---

## 14. Mathematica에서 넘어오셨나요?

만약 "이걸 어떻게 그리지?"라는 직감이 Wolfram 언어에 맞춰져 있다면, 여기 두 도구에서 같은 작업을 처리하는 방법이 있습니다. 기본 아이디어는 동일합니다. 사용자가 수식을 쓰면 도구가 그림을 추론하므로 필요한 것은 구문(syntax)이 아니라 직감입니다.

<!--octarine-table-cols:255,0,0-->
| 작업 | Mathematica | MathSlate |
| --- | --- | --- |
| 단일 곡선 | `Plot[Sin[x], {x, 0, 2 Pi}]` | `plot(sin(x))` — 범위는 선택 사항이며 자동 추론됨 |
| 여러 곡선 중첩 | `Plot[{Sin[x], Cos[x]}, {x, a, b}]` | `plot([sin(x), cos(x)])` |
| 매개변수 곡선 | `ParametricPlot[{Cos[t], Sin[t]}, {t, 0, 2 Pi}]` | `plot((cos(t), sin(t)))` |
| 극좌표 | `PolarPlot[1 + Cos[t], {t, 0, 2 Pi}]` | `polar(1 + cos(t))` |
| 곡면 (Surface) | `Plot3D[x*y, {x, a, b}, {y, c, d}]` | `plot(x*y)` — 두 개의 자유 기호로부터 자동 추론 |
| 음함수 곡선 (Implicit) | `ContourPlot[x^2 + y^2 == 4, {x, a, b}, {y, c, d}]` | `plot(Eq(x**2 + y**2, 4))` |
| 기호 선언 | 필요 없음 — `x`는 기본적으로 기호로 처리 | `x, y, z, t, n, k, theta`는 미리 선언됨; 그 외에는 `symbols("a")` 사용 |
| 방정식 풀이 | `Solve[x^2 - 4 == 0, x]` | `solve(x**2 - 4, x)` — 동일한 SymPy 함수 사용 |
| 미분 | `D[Sin[x], x]` | `diff(sin(x), x)` |
| 적분 | `Integrate[Sin[x], x]` | `integrate(sin(x), x)` |
| 극한 | `Limit[Sin[x]/x, x -> 0]` | `limit(sin(x)/x, x, 0)` |
| 함수의 속성 | 그래프를 보고 판단하거나 여러 개의 개별 호출을 사용 (`Solve`, `D`, `Limit`, …) | `analyze(sin(x)/x)` — 근, 극값, 점근선, 대칭성을 한 번에 확인 |
| 대화형 매개변수 | `Manipulate[Plot[a*Sin[x], {x, 0, 2Pi}], {a, 1, 3}]` | `a = slider(1, 3); plot(a*sin(x))` — 축과 매개변수가 선언되지 않고 자동 추론됨 |
| 데이터 적합 (Fitting) | `FindFit[data, model, pars, x]` | `dataset(data).fit(model)` — 모델을 SymPy 수식으로 반환하며, 매개변수가 채워진 상태로 출력 |
| 함수 호출 구문 | 대괄호 사용, 대문자 시작: `Sin[x]` | 괄호 사용, 소문자 시작 — 일반 Python 구문: `sin(x)` |
| 인덱싱 | 1부터 시작, `list[[1]]` | 0부터 시작, `list[0]` — 일반 Python 구문 |
| 작업물 공유 | `.nb` 노트북 파일 공유, 또는 읽기 전용 뷰어를 위한 CDF/Player 사용 | `worksheet(...).save("handout.html")` — 뷰어 설치 없이 모든 브라우저에서 열리는 단일 파일 생성 |
| "실제로 어떻게 계산되었는지" 보기 | 플롯 옵션이 그림을 변경하지만 코드는 표시되지 않음 | `show_python()`을 호출하면 그림을 생성한 정확한 NumPy/SymPy/Plotly 코드가 출력됨 |
| 비용 및 개방성 | 상용 라이선스 필요 | 무료; MIT 라이선스 |
| 학습 내용이 적용되는 범위 | Wolfram Language에 국한됨 | Python, NumPy, SymPy, Plotly 등 — 어디서든, 영구적으로 사용 가능 |

위 항목 중에서도 다음 세 가지 차이점은 특히 더 깊게 이해할 필요가 있습니다.

**MathSlate만의 독자적인 기술은 없습니다.** 위의 `sin`, `solve`, `diff`, `integrate`, `limit`은 Mathematica와 비슷하게 이름 붙여진 MathSlate 함수가 아닙니다 — 이들은 변형 없이 그대로 가져온 *SymPy 자체*입니다. 여기서 한 번 배우면 10년 뒤 어떤 Python 프로젝트에서도 별도의 변환 없이 그대로 쓸 수 있습니다.

**MathSlate는 의도적으로 추론을 최소화합니다.** Mathematica의 `Plot`은 처리할 수 없는 가지(branch)를 조용히 삭제하며, 진짜 어려운 함수에 대해서는 사용자가 직접 `Exclusions -> None` 또는 `Exclusions -> Automatic`을 세밀하게 설정해 주어야 합니다. 반면 `plot(tan(x))`는 어떤 옵션도 주지 않아도 모든 극점(pole)을 올바르게 잘라냅니다(§5). 그러나 축처럼 보이는 기호가 똑같이 두 개 있을 때처럼 추론만으로 해결할 수 없는 요청이 들어오면, MathSlate는 섣불리 추측하여 잘못된 그림을 그리는 대신 어떤 기호가 축인지 정확히 지정하는 코드를 보여주는 `AmbiguousAxisError`를 발생시킵니다(§11).

**모든 탈출구(escape hatch)는 진짜입니다.** `f.plotly`는 진짜 `plotly.graph_objects`의 figure이고, `f.numpy`는 진짜 배열(array)이며, `f.sympy`는 진짜 SymPy 식입니다 — 아무것도 래핑(wrapping)되지 않았으며, 계속 사용하기 위해 MathSlate를 고집할 필요도 없습니다. MathSlate가 원하는 기능을 제공하지 못한다면, 여러분의 손에는 이미 그 작업을 수행할 진짜 도구가 쥐어져 있습니다. 변환 없이 바로 사용하세요.

---

## 다음으로 볼 것

- [MathSlate 이해하기](guide_ko.md) — 방금 따라 한 동작들이 *왜* 그렇게 되는지, 원리 중심으로 정리한 안내서.
- [레퍼런스 매뉴얼](manual_ko.md) — 모든 옵션, 전체 디스패치 규칙, 샘플링 알고리즘 및 정확한 보장 항목.
- [PRD](design/mathslate_prd_0.3.md) — 이 프로젝트가 왜 이런 형태를 갖추게 되었는지, 고의로 지원하지 않는 기능은 무엇인지.
- `examples/quickstart.py` — 이 튜토리얼의 모든 내용을 한 번에 실행 가능한 파일로 모아둔 것.

절대로 추가되지 않을 한 가지 기능이 있습니다: **단계별 문제 풀이 제공.** MathSlate는 함수가 무엇인지, 그 성질이 어떠한지 보여줍니다. 하지만 대수학적 풀이 과정(나레이션)은 보여주지 않습니다. 임의의 식에 대해 솔직하게 풀이 과정을 보여줄 방법이 없으며, 부정직한 풀이 과정을 보여주는 것은 아예 보여주지 않는 것보다 못하기 때문입니다.
