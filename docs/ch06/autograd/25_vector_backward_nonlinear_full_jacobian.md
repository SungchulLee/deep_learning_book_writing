# 비선형 함수의 전체 야코비안 — 자리마다 달라진다

「선형 사상의 VJP」에서 야코비가 $A$ 자신이었다. **상수**였다는 뜻이다. 비선형
함수에서는 야코비가 **$x$에 따라 달라진다.**

$$
f(x_1, x_2) = \begin{pmatrix} x_1^2 x_2 \\ \sin x_1 + x_2^3 \end{pmatrix},
\qquad
J = \begin{pmatrix} 2x_1 x_2 & x_1^2 \\ \cos x_1 & 3x_2^2 \end{pmatrix}
$$

이 야코비를 **세 가지로** 구해 서로 견준다 — VJP를 모아서, `torch.func`으로,
그리고 수치 미분으로. 셋이 맞으면 어느 하나도 의심할 까닭이 없다.

## 1. 코드

```python
"""비선형 함수의 전체 야코비안 — x 에 따라 달라진다.

「선형 사상의 VJP」에서 야코비가 $A$ 자신이었다. **상수**였다는 뜻이다.
비선형 함수에서는 야코비가 **자리마다 달라진다.**

$$
f(x_1, x_2) = \\begin{pmatrix} x_1^2 x_2 \\\\ \\sin x_1 + x_2^3 \\end{pmatrix}
$$

의 야코비를 세 가지로 구해 서로 견준다 — VJP 를 모아서, `torch.func`
으로, 그리고 수치 미분으로.

실행: 이 쪽의 코드를 파일로 갈무리해 돌린다
"""
import math

import torch
from torch.func import jacrev


def header(title):
    print(f"\n=== {title} ===")


def f(t):
    return torch.stack([t[0] ** 2 * t[1],
                        torch.sin(t[0]) + t[1] ** 3])


def hand_jacobian(a, b):
    """손으로 미분한 야코비."""
    return [[2 * a * b, a ** 2],
            [math.cos(a), 3 * b ** 2]]


# ========================================================================
# 1. 야코비가 자리마다 달라진다
# ========================================================================

def varies():
    header("1. 자리마다 다르다")
    for pt in ([1.0, 2.0], [2.0, 1.0]):
        J = jacrev(f)(torch.tensor(pt))
        print(f"  x={pt}")
        for r, h in zip(J.tolist(), hand_jacobian(*pt)):
            print(f"    자동 {[round(v, 6) for v in r]}"
                  f"   손으로 {[round(v, 6) for v in h]}")
    print("  선형에서는 A 하나였는데 여기서는 x 를 넣어야 정해진다")


# ========================================================================
# 2. VJP 를 모아서
# ========================================================================

def by_vjp():
    header("2. VJP 를 모아서")
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = f(x)
    rows = []
    for i in range(2):
        e = torch.zeros(2)
        e[i] = 1.0
        x.grad = None
        y.backward(e, retain_graph=True)
        rows.append(x.grad.clone())
    J = torch.stack(rows)
    print("  모은 J:", [[round(v, 6) for v in r] for r in J.tolist()])
    print("  jacrev 와 같은가:",
          torch.allclose(J, jacrev(f)(torch.tensor([1.0, 2.0]))))
    print("  출력이 2 개라 backward 를 2 번 불렀다")


# ========================================================================
# 3. 수치 미분으로 검산
# ========================================================================

def numeric_check():
    header("3. 수치 미분")

    def f64(t):
        return torch.stack([t[0] ** 2 * t[1],
                            torch.sin(t[0]) + t[1] ** 3])

    def numeric(fn, x, h=1e-4):
        cols = []
        for j in range(len(x)):
            xp, xm = x.clone(), x.clone()
            xp[j] += h
            xm[j] -= h
            cols.append((fn(xp) - fn(xm)) / (2 * h))
        return torch.stack(cols, dim=1)

    x = torch.tensor([1.0, 2.0], dtype=torch.float64)
    Jn = numeric(f64, x)
    Ja = jacrev(f64)(x)
    print("  수치:", [[round(v, 6) for v in r] for r in Jn.tolist()])
    print("  자동:", [[round(v, 6) for v in r] for r in Ja.tolist()])
    print(f"  가장 큰 차: {(Jn - Ja).abs().max():.2e}")
    print("  float64 로 해야 이만큼 맞는다. float32 면 한참 거칠다")


# ========================================================================
# 4. 세 길의 값
# ========================================================================

def costs():
    header("4. 셋 중 무엇을 쓰는가")
    print("  VJP 모으기  : 출력 수만큼 backward. retain_graph 와 grad 비우기를 챙겨야")
    print("  jacrev      : 같은 일을 라이브러리가 한다. 짧고 안전하다")
    print("  수치 미분    : 입력 수의 두 배만큼 순전파. 느리고 거칠다")
    print("  수치 미분의 쓸모는 속도가 아니라 **검산**이다")
    print("  손으로 짠 backward 가 맞는지 보려면 이것 말고 길이 없다")


# ========================================================================
# 5. 그런데 보통은 야코비가 필요 없다
# ========================================================================

def usually_not_needed():
    header("5. 보통은 안 쓴다")
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = f(x).pow(2).sum()          # 홑값으로 줄인다
    loss.backward()
    print("  로스를 홑값으로 줄이면 backward 한 번")
    print("  x.grad:", [round(v, 6) for v in x.grad.tolist()])
    J = jacrev(f)(torch.tensor([1.0, 2.0]))
    y = f(torch.tensor([1.0, 2.0]))
    print("  야코비로 손계산 (2y)^T J:",
          [round(v, 6) for v in (J.t() @ (2 * y)).tolist()])
    print("  같은 값이다. 야코비를 거치지 않고 바로 얻은 것뿐이다")


def main():
    varies()
    by_vjp()
    numeric_check()
    costs()
    usually_not_needed()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 자리마다 다르다 ===
  x=[1.0, 2.0]
    자동 [4.0, 1.0]   손으로 [4.0, 1.0]
    자동 [0.540302, 12.0]   손으로 [0.540302, 12.0]
  x=[2.0, 1.0]
    자동 [4.0, 4.0]   손으로 [4.0, 4.0]
    자동 [-0.416147, 3.0]   손으로 [-0.416147, 3.0]
  선형에서는 A 하나였는데 여기서는 x 를 넣어야 정해진다

=== 2. VJP 를 모아서 ===
  모은 J: [[4.0, 1.0], [0.540302, 12.0]]
  jacrev 와 같은가: True
  출력이 2 개라 backward 를 2 번 불렀다

=== 3. 수치 미분 ===
  수치: [[4.0, 1.0], [0.540302, 12.0]]
  자동: [[4.0, 1.0], [0.540302, 12.0]]
  가장 큰 차: 1.00e-08
  float64 로 해야 이만큼 맞는다. float32 면 한참 거칠다

=== 4. 셋 중 무엇을 쓰는가 ===
  VJP 모으기  : 출력 수만큼 backward. retain_graph 와 grad 비우기를 챙겨야
  jacrev      : 같은 일을 라이브러리가 한다. 짧고 안전하다
  수치 미분    : 입력 수의 두 배만큼 순전파. 느리고 거칠다
  수치 미분의 쓸모는 속도가 아니라 **검산**이다
  손으로 짠 backward 가 맞는지 보려면 이것 말고 길이 없다

=== 5. 보통은 안 쓴다 ===
  로스를 홑값으로 줄이면 backward 한 번
  x.grad: [25.554134, 216.195297]
  야코비로 손계산 (2y)^T J: [25.554134, 216.195297]
  같은 값이다. 야코비를 거치지 않고 바로 얻은 것뿐이다
```

## 2. 자리마다 달라진다

| $x$ | $J$ |
|---|---|
| $(1, 2)$ | $\begin{pmatrix} 4 & 1 \\ 0.5403 & 12 \end{pmatrix}$ |
| $(2, 1)$ | $\begin{pmatrix} 4 & 4 \\ -0.4161 & 3 \end{pmatrix}$ |

손으로 미분한 값과 소수점까지 같다. 그리고 **두 자리의 야코비가 전혀 다르다.**

선형에서는 "$J = A$"라고 적으면 끝이었는데, 여기서는 $x$를 넣어야 정해진다.
이것이 딥러닝에서 야코비를 들고 다니지 않는 또 하나의 까닭이다 — 파라미터가
바뀔 때마다 다시 구해야 한다.

## 3. VJP를 모아서

```
모은 J: [[4.0, 1.0], [0.540302, 12.0]]
jacrev 와 같은가: True
```

「벡터 출력의 역전파」에서 한 그대로다. 출력이 2개이므로 `backward`를 2번
부르고, `retain_graph=True`와 `x.grad = None`을 챙긴다.

`torch.func.jacrev`가 같은 일을 한다. 손으로 할 까닭이 없지만, **무엇을 하는지
알고 쓰는 것**과 모르고 쓰는 것은 다르다.

## 4. 수치 미분으로 검산

```
수치: [[4.0, 1.0], [0.540302, 12.0]]
자동: [[4.0, 1.0], [0.540302, 12.0]]
가장 큰 차: 1.00e-08
```

중심 차분

$$
\frac{\partial f_i}{\partial x_j} \approx \frac{f_i(x + he_j) - f_i(x - he_j)}{2h}
$$

으로 구한 것이 자동 미분과 $10^{-8}$까지 맞는다.

**float64라서 이만큼 맞는다.** float32에서는 한참 거칠고, 연습문제 1에서 그
한계를 잰다.

## 5. 세 길의 쓸모

| | 값 | 쓸 자리 |
|---|---|---|
| VJP 모으기 | `backward` × 출력 수 | 무엇을 하는지 알기 위해 |
| `jacrev` | 같은 일 | 야코비가 정말 필요할 때 |
| 수치 미분 | 순전파 × 입력 수 × 2 | **검산** |

수치 미분의 쓸모는 속도가 아니다. **손으로 짠 `backward`가 맞는지 보는 데는
이것 말고 길이 없다.** 「커스텀 함수」에서 `gradcheck`을 쓴 것이 바로 이
방법이고, 연습문제 2에서 다시 본다.

## 6. 그런데 보통은 야코비가 필요 없다

```
로스를 홑값으로 줄이면 backward 한 번
x.grad: [25.554134, 216.195297]
야코비로 손계산 (2y)^T J: [25.554134, 216.195297]
```

$L = \sum f_i^2$의 경사를 두 가지로 구했다. 야코비를 만들어
$(2y)^\top J$를 계산한 것과, 그냥 `loss.backward()`를 부른 것이 같다.

**뒤쪽이 야코비를 거치지 않고 바로 얻는다.** 그것이 역전파가 하는 일이고,
그래서 야코비를 명시적으로 만들 일이 거의 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
수치 미분의 $h$를 작게 할수록 정확해지는가? float32에서 $h$를 $10^{-1}$부터
$10^{-7}$까지 바꿔 가며 오차를 재어라.

</div>

??? success "풀이"
    ```python
    x32 = torch.tensor([1.0, 2.0])
    Ja = jacrev(f)(x32)
    for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        err = (numeric(f, x32, h) - Ja).abs().max().item()
        print(f"  h={h:.0e}  가장 큰 차 {err:.3e}")
    ```

    ```
      h=1e-01  가장 큰 차 9.993e-03
      h=1e-02  가장 큰 차 8.392e-05
      h=1e-03  가장 큰 차 3.939e-04
      h=1e-04  가장 큰 차 7.545e-03
      h=1e-05  가장 큰 차 1.630e-02
      h=1e-06  가장 큰 차 5.559e-01
      h=1e-07  가장 큰 차 7.232e+00
    ```

    **아니다.** $h = 10^{-2}$에서 가장 좋고, 더 줄이면 **나빠진다.** $10^{-7}$에서는
    오차가 7.2로, 아예 쓸 수 없다.

    두 가지 오차가 반대로 움직이기 때문이다.

    - **절단 오차** — 중심 차분은 $O(h^2)$만큼 틀린다. $h$가 **클수록** 나쁘다
    - **반올림 오차** — $f(x+h) - f(x-h)$가 거의 같은 두 수의 뺄셈이라 유효숫자가
      깎인다. 그리고 $2h$로 나누므로 그 오차가 **$1/h$배 확대**된다.
      $h$가 **작을수록** 나쁘다

    합이 가장 작아지는 자리가 대략 $h \approx \sqrt{\epsilon}$이고, float32의
    $\epsilon \approx 10^{-7}$이니 $h \approx 3 \times 10^{-4}$쯤이어야 한다.
    잰 값이 $10^{-2}$로 조금 큰 것은 이 함수의 2계 도함수가 커서 절단 오차 쪽이
    무겁기 때문이다.

    **수치 미분으로 검산할 때 float64를 쓰는 까닭**이 이것이다. $\epsilon$이
    $10^{-16}$이면 좋은 $h$가 $10^{-8}$ 언저리로 내려가고, 거기서 얻는 정확도가
    훨씬 높다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`torch.autograd.gradcheck`이 바로 이 검산을 해 준다. float64와 float32에서
각각 돌려 보라.

</div>

??? success "풀이"
    ```python
    def g(t):
        return (t[0] ** 2 * t[1] + torch.sin(t[0])).sum()

    x64 = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    print("  float64 gradcheck:", torch.autograd.gradcheck(g, (x64,)))

    x32 = torch.tensor([1.0, 2.0], requires_grad=True)
    try:
        print("  float32 gradcheck:", torch.autograd.gradcheck(g, (x32,)))
    except Exception as e:
        print("  float32 gradcheck:", str(e).splitlines()[0][:66])
    ```

    ```
      float64 gradcheck: True
      float32 gradcheck: Jacobian mismatch for output 0 with respect to input 0,
    ```

    **float32에서는 실패한다.** 게다가 돌리기 전에 경고까지 띄운다 —
    "입력이 double이 아니면 이 검사는 실패할 가능성이 높다".

    자동 미분이 틀려서가 아니다. **수치 미분 쪽이 못 따라오는 것**이다.
    연습문제 1에서 본 그 한계다.

    그래서 `gradcheck`을 쓸 때는 **반드시 float64로** 바꿔서 쓴다. 실제
    모델은 float32로 돌리더라도, 검사할 때만 올린다.

    이것이 「커스텀 함수」에서 손으로 짠 `backward`를 확인하던 방법이고,
    새 연산을 만들 때 가장 먼저 할 일이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
야코비의 야코비가 **헤세 행렬**이다. $s(x) = x_1^2 x_2 + \sin x_1$의 헤세를
구하고 손계산과 견주어라. 대칭인가?

</div>

??? success "풀이"
    손으로 미분하면

    $$
    \nabla s = \begin{pmatrix} 2x_1 x_2 + \cos x_1 \\ x_1^2 \end{pmatrix},
    \qquad
    H = \begin{pmatrix} 2x_2 - \sin x_1 & 2x_1 \\ 2x_1 & 0 \end{pmatrix}
    $$

    ```python
    from torch.func import hessian

    def s(t):
        return (t[0] ** 2 * t[1] + torch.sin(t[0])).sum()

    xh = torch.tensor([1.0, 2.0], dtype=torch.float64)
    H = hessian(s)(xh)
    print("  자동:", [[round(v, 6) for v in r] for r in H.tolist()])
    a, b = 1.0, 2.0
    print("  손으로:", [[round(2 * b - math.sin(a), 6), round(2 * a, 6)],
                       [round(2 * a, 6), 0.0]])
    print("  대칭인가:", torch.allclose(H, H.t()))
    ```

    ```
      자동: [[3.158529, 2.0], [2.0, 0.0]]
      손으로: [[3.158529, 2.0], [2.0, 0.0]]
      대칭인가: True
    ```

    맞는다. 그리고 **대칭이다** — $\partial^2 s/\partial x_1 \partial x_2$와
    $\partial^2 s/\partial x_2 \partial x_1$이 같기 때문이다(슈바르츠 정리).

    대칭성은 **검산의 도구**로 쓸모가 있다. 헤세를 구했는데 대칭이 아니면
    어딘가 틀린 것이다 — 함수가 매끄럽지 않거나, 중간에 제자리 연산이 끼었거나,
    수치 오차가 너무 크거나.

    헤세는 값이 비싸다. $n \times n$ 행렬이므로 파라미터가 백만 개면 $10^{12}$개의
    수다. 「완전한 선형 회귀」에서 안정 경계를 구할 때 헤세의 최대 고윳값이
    필요했는데, 실제 모델에서는 **헤세를 만들지 않고 벡터와의 곱만** 구한다
    (헤세-벡터 곱). 그것은 경사의 경사를 한 번 더 미분하는 일이라
    `create_graph=True`로 할 수 있다 — 「계산 그래프 재사용」에서 본 그 길이다.
