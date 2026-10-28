# Linear 웨이트 관례 — 왜 `(out, in)`인가

`nn.Linear(5, 3)`을 만들면 `weight.shape`가 `(3, 5)`다. 입력이 5인데 5가 뒤에
온다. 그리고 순전파는 `x @ W.t() + b`로, 쓸 때마다 전치가 붙는다.

처음 보면 거꾸로 저장해 둔 것처럼 보인다. 그렇게 둔 까닭이 있고, 이 쪽은 그것을
재어서 가린다. "전치를 매번 하니 느릴 것"이라는 짐작이 그럴듯하므로 **그것부터
재어 본다.**

## 1. 코드

```python
"""nn.Linear 의 웨이트는 왜 (out, in) 인가.

`nn.Linear(5, 3)` 의 `weight.shape` 는 `(3, 5)` 다. 입력이 5 인데 5 가
뒤에 온다. 수학책의 $y = Wx$ 와 같은 방향이면서, 코드는
$Y = XW^\top$ 로 적는다.

이 쪽은 그 방향이 무엇을 쉽게 만드는지 본다. 빠르기 때문이라는 설명이
그럴듯해 보이므로 그것부터 재어 본다 — 재어 보면 아니다.

- 모양 (out, in) 과 한 행의 뜻
- 뒤집어 저장하면 느려지는가 (재어 보니 그렇지 않다)
- 수학식 y = Wx 와 코드 Y = XW^T 가 같은 것
- 경사 셋의 값: dW = g^T x, db = sum g, dx = g W
- 앞쪽 축은 몇 개여도 된다

실행: python 05_linear_weight_convention.py
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


def spread(fn, n=9):
    """한 번 잰 시간은 측정이 아니다. 가장 빠른 값과 가장 느린 값."""
    got = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        got.append(time.perf_counter() - t0)
    return min(got) * 1e3, max(got) * 1e3


# ========================================================================
# 1. 모양 — 한 행이 출력 뉴런 하나
# ========================================================================

def shapes():
    header("1. 모양과 한 행의 뜻")
    torch.manual_seed(0)
    lin = nn.Linear(5, 3)
    W, b = lin.weight, lin.bias
    print("  nn.Linear(5, 3)")
    print("    weight.shape :", tuple(W.shape), "= (out, in)")
    print("    bias.shape   :", tuple(b.shape), "= (out,)")
    print("    파라미터 수   :", sum(p.numel() for p in lin.parameters()),
          "= 3*5 + 3")

    print("  W[1] = 출력 뉴런 1 로 들어오는 웨이트 다섯 개")
    print("    연속인가 :", W[1].is_contiguous(), " 보폭", W[1].stride())
    print("    메모리 간격(바이트):",
          [W[1][i].data_ptr() - W[1][0].data_ptr() for i in range(5)])
    print("  W[:,1] = 입력 1 이 나가는 웨이트 세 개")
    print("    연속인가 :", W[:, 1].is_contiguous(), " 보폭", W[:, 1].stride())
    print("    메모리 간격(바이트):",
          [W[:, 1][i].data_ptr() - W[:, 1][0].data_ptr() for i in range(3)])


# ========================================================================
# 2. 빠르기 때문인가 — 재어 본다
# ========================================================================

def is_it_speed():
    header("2. 뒤집어 저장하면 느려지는가")
    torch.manual_seed(0)
    print("  W 를 (out,in) 로 두고 x @ W.t() 하는 것과")
    print("  V 를 (in,out) 로 두고 x @ V    하는 것을 견준다")
    # 어느 쪽이 빠른지는 돌릴 때마다 뒤바뀐다. 그래서 '겹치는가' 같은
    # 참·거짓을 찍어 두면 그 줄이 쪽마다 달라진다. 범위만 적고, 판단은
    # 본문에서 한다.
    for B, din, dout in ((512, 2048, 2048), (4096, 512, 512), (64, 4096, 1024)):
        x = torch.randn(B, din)
        W = torch.randn(dout, din)
        V = W.t().contiguous()
        print(f"  B={B:5d} in={din:5d} out={dout:5d}")
        print("    두 결과가 같은가:", torch.allclose(x @ W.t(), x @ V, atol=1e-3))
        a = spread(lambda: x @ W.t())
        c = spread(lambda: x @ V)
        print(f"    x @ W.t() {a[0]:5.2f} ~ {a[1]:5.2f} ms")
        print(f"    x @ V     {c[0]:5.2f} ~ {c[1]:5.2f} ms")


# ========================================================================
# 3. y = Wx 와 Y = XW^T 는 같은 것
# ========================================================================

def same_math():
    header("3. 수학식과 코드")
    torch.manual_seed(0)
    lin = nn.Linear(5, 3)
    W, b = lin.weight, lin.bias
    x = torch.randn(4, 5)

    print("  배치로:  lin(x).shape =", tuple(lin(x).shape),
          " x @ W.t() + b 와 같은가:",
          torch.allclose(lin(x), x @ W.t() + b, atol=1e-6))

    # 한 표본을 열벡터로 보면 W x 다
    one = x[0]
    print("  한 표본:  W @ x[0] + b 와 같은가:",
          torch.allclose(lin(one), W @ one + b, atol=1e-6))
    print("  둘 다 맞다. 표본을 행으로 쌓으면 전치가 붙는 것뿐이다")

    # 한 뉴런은 내적 하나
    k = 1
    print(f"  뉴런 {k} 하나:  lin(x)[0,{k}] =",
          round(lin(x)[0, k].item(), 6))
    print(f"              x[0] @ W[{k}] + b[{k}] =",
          round((one @ W[k] + b[k]).item(), 6))
    print("  W[k] 가 연속이라 이 내적은 메모리를 한 줄로 읽는다")


# ========================================================================
# 4. 경사 셋
# ========================================================================

def gradients():
    header("4. 경사 셋의 모양과 값")
    torch.manual_seed(0)
    lin = nn.Linear(5, 3)
    W, b = lin.weight, lin.bias
    x = torch.randn(4, 5, requires_grad=True)
    g = torch.randn(4, 3)        # 위에서 내려온 경사
    lin.zero_grad(set_to_none=True)
    F.linear(x, W, b).backward(g)

    print("  W.grad == g.T @ x  :",
          torch.allclose(W.grad, g.t() @ x.detach(), atol=1e-6),
          " 모양", tuple(W.grad.shape))
    print("  b.grad == g.sum(0) :",
          torch.allclose(b.grad, g.sum(0), atol=1e-6),
          " 모양", tuple(b.grad.shape))
    print("  x.grad == g @ W    :",
          torch.allclose(x.grad, g @ W.detach(), atol=1e-6),
          " 모양", tuple(x.grad.shape))
    print("  경사의 모양은 언제나 그 파라미터의 모양과 같다")


# ========================================================================
# 5. 앞쪽 축은 몇 개여도 된다
# ========================================================================

def leading_axes():
    header("5. 앞쪽 축")
    torch.manual_seed(0)
    lin = nn.Linear(5, 3)
    for shape in ((5,), (4, 5), (2, 7, 5), (2, 3, 7, 5)):
        print(f"  입력 {str(shape):14s} -> 출력 {tuple(lin(torch.randn(shape)).shape)}")
    x3 = torch.randn(2, 7, 5)
    flat = lin(x3.reshape(-1, 5)).reshape(2, 7, 3)
    print("  (2,7,5) 를 (14,5) 로 펴서 넣은 것과 같은가:",
          torch.allclose(lin(x3), flat, atol=1e-6))
    print("  마지막 축만 in_features 와 맞으면 된다")


# ========================================================================
# 6. 전치는 복사하지 않는다
# ========================================================================

def transpose_is_free():
    header("6. W.t() 는 복사가 아니다")
    torch.manual_seed(0)
    W = nn.Linear(5, 3).weight
    print("  W.is_contiguous()     :", W.is_contiguous(), " 보폭", W.stride())
    print("  W.t().is_contiguous() :", W.t().is_contiguous(),
          " 보폭", W.t().stride())
    print("  같은 저장소인가        :",
          W.untyped_storage().data_ptr() == W.t().untyped_storage().data_ptr())
    print("  그래서 x @ W.t() 는 전치하느라 메모리를 쓰지 않는다")


def main():
    shapes()
    is_it_speed()
    same_math()
    gradients()
    leading_axes()
    transpose_is_free()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 모양과 한 행의 뜻 ===
  nn.Linear(5, 3)
    weight.shape : (3, 5) = (out, in)
    bias.shape   : (3,) = (out,)
    파라미터 수   : 18 = 3*5 + 3
  W[1] = 출력 뉴런 1 로 들어오는 웨이트 다섯 개
    연속인가 : True  보폭 (1,)
    메모리 간격(바이트): [0, 4, 8, 12, 16]
  W[:,1] = 입력 1 이 나가는 웨이트 세 개
    연속인가 : False  보폭 (5,)
    메모리 간격(바이트): [0, 20, 40]

=== 2. 뒤집어 저장하면 느려지는가 ===
  W 를 (out,in) 로 두고 x @ W.t() 하는 것과
  V 를 (in,out) 로 두고 x @ V    하는 것을 견준다
  B=  512 in= 2048 out= 2048
    두 결과가 같은가: True
    x @ W.t()  3.92 ~  5.45 ms
    x @ V      5.45 ~  7.68 ms
  B= 4096 in=  512 out=  512
    두 결과가 같은가: True
    x @ W.t()  2.12 ~  2.95 ms
    x @ V      2.50 ~  2.93 ms
  B=   64 in= 4096 out= 1024
    두 결과가 같은가: True
    x @ W.t()  0.99 ~  1.33 ms
    x @ V      1.06 ~  1.12 ms

=== 3. 수학식과 코드 ===
  배치로:  lin(x).shape = (4, 3)  x @ W.t() + b 와 같은가: True
  한 표본:  W @ x[0] + b 와 같은가: True
  둘 다 맞다. 표본을 행으로 쌓으면 전치가 붙는 것뿐이다
  뉴런 1 하나:  lin(x)[0,1] = 0.587614
              x[0] @ W[1] + b[1] = 0.587614
  W[k] 가 연속이라 이 내적은 메모리를 한 줄로 읽는다

=== 4. 경사 셋의 모양과 값 ===
  W.grad == g.T @ x  : True  모양 (3, 5)
  b.grad == g.sum(0) : True  모양 (3,)
  x.grad == g @ W    : True  모양 (4, 5)
  경사의 모양은 언제나 그 파라미터의 모양과 같다

=== 5. 앞쪽 축 ===
  입력 (5,)           -> 출력 (3,)
  입력 (4, 5)         -> 출력 (4, 3)
  입력 (2, 7, 5)      -> 출력 (2, 7, 3)
  입력 (2, 3, 7, 5)   -> 출력 (2, 3, 7, 3)
  (2,7,5) 를 (14,5) 로 펴서 넣은 것과 같은가: True
  마지막 축만 in_features 와 맞으면 된다

=== 6. W.t() 는 복사가 아니다 ===
  W.is_contiguous()     : True  보폭 (5, 1)
  W.t().is_contiguous() : False  보폭 (1, 5)
  같은 저장소인가        : True
  그래서 x @ W.t() 는 전치하느라 메모리를 쓰지 않는다
```

## 2. 한 행이 출력 뉴런 하나다

`(out, in)`이 뜻하는 것은 이것이다. **`W[k]`는 출력 뉴런 $k$로 들어오는 웨이트
전부**다. 그리고 행 우선 저장이므로 그 다섯 개가 메모리에 잇달아 누워 있다.

| | 뜻 | 연속인가 | 메모리 간격 |
|---|---|---|---|
| `W[1]` | 출력 1로 **들어오는** 웨이트 | True | 0, 4, 8, 12, 16 |
| `W[:,1]` | 입력 1이 **나가는** 웨이트 | False | 0, 20, 40 |

`W[1]`은 4바이트씩 붙어 있고 `W[:,1]`은 20바이트씩 뛴다. 둘 중 하나는 연속일
수밖에 없고, 파이토치는 **출력 쪽을 연속으로** 골랐다.

그 선택이 쓸 때 어떻게 나타나는가. 뉴런 하나의 값은 내적 하나다.

```
lin(x)[0,1]          = 0.587614
x[0] @ W[1] + b[1]   = 0.587614
```

전치가 없다. `W[k]`가 이미 벡터이므로 `x[0]`과 그냥 내적하면 된다. 그리고
편향도 `b[k]`로 같은 축을 쓴다 — 출력 뉴런 하나를 다루는 일이 **두 텐서에서
모두 `[k]`** 하나로 끝난다.

## 3. 속도 때문은 아니다

"매번 전치하니 느릴 것"이라는 짐작을 재어 보았다. `W`를 `(out,in)`으로 두고
`x @ W.t()`하는 쪽과, 미리 뒤집어 `V = (in,out)`으로 두고 `x @ V`하는 쪽이다.
값은 같다.

| 모양 | `x @ W.t()` (관례) | `x @ V` (뒤집어 저장) |
|---|---|---|
| B=512, 2048→2048 | **3.92 ~ 5.45 ms** | 5.45 ~ 7.68 ms |
| B=4096, 512→512 | 2.12 ~ 2.95 ms | 2.50 ~ 2.93 ms |
| B=64, 4096→1024 | 0.99 ~ 1.33 ms | 1.06 ~ 1.12 ms |

**관례 쪽이 느린 모양이 하나도 없다.** 네 번 돌려 네 번, 세 모양 모두에서 관례
쪽의 가장 빠른 값이 더 작았다. 다만 범위가 겹치는 일이 잦으므로 — 큰 정사각에서도
돌림에 따라 겹친다 — "몇 배 빠르다"고 적을 수는 없다. 말할 수 있는 것은 이만큼이다.
**전치를 매번 하는 쪽이 손해를 보지 않는다.**

까닭은 6절에 있다. `W.t()`는 **복사가 아니다.**

```
W.is_contiguous()     : True   보폭 (5, 1)
W.t().is_contiguous() : False  보폭 (1, 5)
같은 저장소인가        : True
```

보폭만 바꿔 쓴 뷰이고, 행렬곱 커널은 "두 번째 인자가 전치되어 있다"는 표시를
받아 그대로 처리한다. 전치 때문에 메모리를 옮기는 일은 아예 일어나지 않는다.
그러니 전치는 비용이 아니고, 이 관례를 고른 까닭도 속도가 아니다. **한 행이
무엇을 뜻하느냐가 까닭이다.**

## 4. 수학식과 코드가 어긋난 것이 아니다

수학책은 $y = Wx + b$로 적고 코드는 $Y = XW^\top + b$로 적는다. 서로 다른 식처럼
보이지만 같은 것이다. 갈리는 것은 **표본을 열로 세우는가 행으로 눕히는가**뿐이다.

$$
\begin{aligned}
\text{수학식} &: y = Wx + b, \quad x \in \mathbb{R}^{d_{in} \times 1} \\
\text{코드} &: Y = XW^\top + b, \quad X \in \mathbb{R}^{B \times d_{in}}
\end{aligned}
$$

코드가 표본을 행으로 눕히는 것은 배치를 쌓기 때문이다. $B$개를 세로로 쌓으려면
표본이 행이어야 하고, 그러면 곱셈의 방향이 뒤집히므로 $W^\top$가 붙는다. $W$를
거꾸로 저장한 것이 아니라, $X$를 거꾸로 눕힌 것이다.

한 표본만 넣어 확인해 보면 둘이 같다.

```
배치로:  x @ W.t() + b 와 같은가: True
한 표본:  W @ x[0] + b 와 같은가: True
```

## 5. 경사 셋

역전파에서 나오는 세 경사는 모양만 보아도 맞춰진다. 위에서 내려온 경사를
$g$($B \times d_{out}$)라 하면

$$
\frac{\partial L}{\partial W} = g^\top X, \qquad
\frac{\partial L}{\partial b} = \sum_{i} g_i, \qquad
\frac{\partial L}{\partial X} = g W
$$

코드가 셋 다 확인한다.

```
W.grad == g.T @ x  : True  모양 (3, 5)
b.grad == g.sum(0) : True  모양 (3,)
x.grad == g @ W    : True  모양 (4, 5)
```

**경사의 모양은 언제나 그 파라미터의 모양과 같다.** 그래서 $\partial L/\partial W$가
$(3,5)$이고, $g^\top X$가 $(3,4) \times (4,5) = (3,5)$로 맞아떨어진다. 전치를
어디에 붙일지 헷갈릴 때는 모양을 맞춰 보면 한 가지 답만 남는다.

$b$의 경사가 배치 **합**이라는 점은 따로 볼 값이 있다. 편향 하나가 배치의 모든
표본에 쓰였으니 — 「메모리와 저장소」의 보폭 0과 같은 이야기다 — 경사가 모두
더해져 돌아온다. 연습문제 3에서 이것이 러닝레이트와 어떻게 맞물리는지 본다.

## 6. 앞쪽 축은 몇 개여도 된다

`nn.Linear`는 **마지막 축만** `in_features`와 맞으면 된다.

| 입력 | 출력 |
|---|---|
| `(5,)` | `(3,)` |
| `(4, 5)` | `(4, 3)` |
| `(2, 7, 5)` | `(2, 7, 3)` |
| `(2, 3, 7, 5)` | `(2, 3, 7, 3)` |

그리고 `(2,7,5)`를 `(14,5)`로 펴서 넣은 결과와 같다. 앞쪽 축이 몇 개든 그냥
"표본이 그만큼 많다"로 취급한다.

이것이 트랜스포머에서 `(배치, 길이, 차원)`을 그대로 `Linear`에 넣을 수 있는
까닭이다. 토큰마다 같은 선형층을 따로 적용하는 것이 수식으로는 복잡해 보이지만,
코드로는 `reshape` 없이 한 번 부르면 된다.

## 7. 정리

| 묻는 것 | 답 |
|---|---|
| `weight.shape` | `(out_features, in_features)` |
| `W[k]`는 무엇인가 | 출력 뉴런 $k$로 들어오는 웨이트, **연속** |
| 왜 이 방향인가 | 한 행이 뉴런 하나가 되고, 편향도 같은 축을 쓴다 |
| 전치가 비용인가 | 아니다. 뷰이고, 재어 보면 관례 쪽이 느리지 않다 |
| 파라미터 수 | $d_{out} d_{in} + d_{out}$ |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`nn.Linear(784, 128)`과 `nn.Linear(128, 10)`의 파라미터 수를 손으로 세고 코드로
확인하라. `bias=False`로 하면 몇 개가 줄어드는가?

</div>

??? success "풀이"
    ```python
    for din, dout, bias in ((5, 3, True), (5, 3, False),
                            (784, 128, True), (128, 10, True)):
        lin = nn.Linear(din, dout, bias=bias)
        n = sum(p.numel() for p in lin.parameters())
        hand = dout * din + (dout if bias else 0)
        print(f"  Linear({din:3d}, {dout:3d}, bias={bias!s:5s})  {n:>7,}개   "
              f"손으로 {dout}*{din}{f'+{dout}' if bias else ''} = {hand:,}")
    ```

    ```
      Linear(  5,   3, bias=True )       18개   손으로 3*5+3 = 18
      Linear(  5,   3, bias=False)       15개   손으로 3*5 = 15
      Linear(784, 128, bias=True )  100,480개   손으로 128*784+128 = 100,480
      Linear(128,  10, bias=True )    1,290개   손으로 10*128+10 = 1,290
    ```

    편향은 출력마다 하나이므로 `bias=False`는 `out_features`개를 줄인다. 3개가
    줄어 18에서 15가 되었다.

    눈에 둘 것은 비율이다. `Linear(784, 128)`에서 편향 128개는 전체 100,480개의
    **0.13%**다. 「실전 예제」에서 `fc1` 하나가 모델 전체의 91%를 차지한 것과
    같은 이야기로, 파라미터는 거의 전부 행렬에 있고 편향은 세는 의미가 없을
    만큼 적다.

    그런데도 편향을 빼면 모델이 달라진다 — 원점을 지나는 함수만 표현하게 된다.
    개수와 중요도는 따로다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
출력 뉴런 하나를 끄고 싶다. `(out, in)` 관례 덕분에 `W[k] = 0`과 `b[k] = 0`
두 줄로 된다. 해 보고 다른 출력이 멀쩡한지 확인하라. 그리고 **그 뉴런이 계속
꺼져 있는지** 확인하라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    lin = nn.Linear(5, 3)
    x = torch.randn(2, 5)
    print("  끄기 전     :", [round(v, 4) for v in lin(x)[0].tolist()])
    with torch.no_grad():
        lin.weight[1] = 0.0
        lin.bias[1] = 0.0
    print("  뉴런 1 끈 뒤 :", [round(v, 4) for v in lin(x)[0].tolist()])

    lin.zero_grad(set_to_none=True)
    lin(x).sum().backward()
    print("  W.grad[1] =", [round(v, 4) for v in lin.weight.grad[1].tolist()])
    print("  W.grad[0] =", [round(v, 4) for v in lin.weight.grad[0].tolist()])
    ```

    ```
      끄기 전     : [-0.0137, 0.1524, -0.0641]
      뉴런 1 끈 뒤 : [-0.0137, 0.0, -0.0641]
      W.grad[1] = [-1.0563, 0.1139, 0.7876, 1.2483, 0.5949]
      W.grad[0] = [-1.0563, 0.1139, 0.7876, 1.2483, 0.5949]
    ```

    끄는 것은 두 줄로 되고, **둘째 출력만 0이 되고 나머지는 그대로**다.
    `(in, out)`으로 저장했다면 `W[:, 1] = 0`이 되어 비연속 열에 쓰는 일이
    되었을 것이다.

    그런데 두 번째 물음의 답이 중요하다. **계속 꺼져 있지 않다.** `W.grad[1]`이
    0이 아니다. 그러니 다음 `opt.step()`에서 그 뉴런이 되살아난다.

    `W.grad[1]`과 `W.grad[0]`이 똑같은 것도 눈에 걸린다. 5절의 식이 답이다 —
    $\partial L/\partial W = g^\top X$이고 여기서는 `sum()`을 역전파했으니 $g$가
    전부 1이다. 그러면 모든 행이 같은 값($X$의 배치 합)을 받는다. 웨이트가 0이든
    아니든 경사는 입력만으로 정해진다.

    그래서 뉴런을 정말 죽이려면 값을 0으로 두는 것으로는 안 되고 **경사를 막아야**
    한다. 가지치기(pruning) 구현이 마스크를 따로 들고 다니며 매 걸음 다시
    씌우는 까닭이 이것이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
5절에서 $\partial L/\partial b$가 배치 **합**이라고 했다. 확인하고, 로스를 배치
크기로 나누면 어떻게 달라지는지 재어라. 그리고 이것이 배치 크기를 바꿀 때
러닝레이트에 무엇을 뜻하는지 적어라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    B, din, dout = 4, 5, 3
    W = torch.randn(dout, din, requires_grad=True)
    b = torch.randn(dout, requires_grad=True)
    x = torch.randn(B, din, requires_grad=True)
    g = torch.randn(B, dout)

    F.linear(x, W, b).backward(g)
    print("  dW == g.T @ x  :", torch.allclose(W.grad, g.t() @ x.detach(), atol=1e-6))
    print("  db == g.sum(0) :", torch.allclose(b.grad, g.sum(0), atol=1e-6))
    print("  dx == g @ W    :", torch.allclose(x.grad, g @ W.detach(), atol=1e-6))
    print("  b.grad   =", [round(v, 4) for v in b.grad.tolist()])
    print("  g.sum(0) =", [round(v, 4) for v in g.sum(0).tolist()])

    W2 = W.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    F.linear(x.detach(), W2, b2).mul(g).sum().div(B).backward()
    print(f"  로스를 {B} 로 나누면 b.grad =",
          [round(v, 4) for v in b2.grad.tolist()])
    ```

    ```
      dW == g.T @ x  : True
      db == g.sum(0) : True
      dx == g @ W    : True
      b.grad   = [-2.2219, -1.2663, -0.7461]
      g.sum(0) = [-2.2219, -1.2663, -0.7461]
      로스를 4 로 나누면 b.grad = [-0.5555, -0.3166, -0.1865]
    ```

    세 식이 다 맞고, `b.grad`는 정확히 `g.sum(0)`이다. 배치 4개를 **더한** 값이다.
    나누면 정확히 $1/4$이 된다.

    여기서 따라오는 것이 러닝레이트와 배치 크기의 관계다.

    - **합으로 로스를 잡으면** 경사의 크기가 배치 크기에 비례한다. 배치를 32에서
      256으로 늘리면 걸음이 8배 커지므로, 같은 러닝레이트를 쓰면 발산한다.
    - **평균으로 잡으면** 경사의 크기가 배치 크기와 (거의) 무관해진다. 그래서
      배치를 바꾸어도 러닝레이트를 그대로 둘 수 있다.

    파이토치의 로스 함수들이 `reduction="mean"`을 기본값으로 둔 까닭이 이것이다.
    「최적화 기초」에서 러닝레이트가 다른 어떤 선택보다 크게 작용한 것을 재었는데,
    그 러닝레이트가 뜻을 가지려면 경사의 크기가 배치 크기에 휘둘리지 않아야 한다.

    다만 평균이 경사의 **크기**를 고정해 줄 뿐 **잡음**까지 고정하지는 않는다.
    배치가 커지면 경사의 추정이 정확해지므로 더 큰 걸음을 걸어도 되고, 거기서
    "배치를 $k$배 하면 러닝레이트를 $\sqrt{k}$배 또는 $k$배" 같은 규칙이 나온다.
    그것은 이 절의 이야기가 아니라 최적화 쪽의 이야기다.
