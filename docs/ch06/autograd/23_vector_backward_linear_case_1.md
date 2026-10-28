# 선형 사상의 VJP — 왜 $A$가 아니라 $A^\top v$인가

$y = Ax$에서 `y.backward(v)`를 부르면 `x.grad`에 무엇이 들어오는가.

답은 $A^\top v$다. **$A$를 곱하는 것이 아니라 $A^\top$을 곱한다.** 전치가 어디서
오는지는 두 가지로 설명된다 — 셈으로 유도하는 길과, **모양만 보고 아는** 길.

## 1. 코드

```python
"""선형 사상의 VJP — 왜 A 가 아니라 A^T v 인가.

$y = Ax$ 에서 `y.backward(v)` 를 부르면 `x.grad` 에 무엇이 들어오는가.
손으로 유도해 보면 $A^\\top v$ 이고, 그것은 **$A$ 를 곱하는 것이 아니라
$A^\\top$ 을 곱하는 것**이다.

전치가 붙는 까닭을 모양만으로도 알 수 있다. $A$ 가 $(m, n)$ 이고
$v$ 가 $(m,)$ 이면, 결과는 $x$ 와 같은 $(n,)$ 이어야 한다.
$(m, n)$ 과 $(m,)$ 으로 $(n,)$ 을 만드는 길은 $A^\\top v$ 뿐이다.

실행: python 23_vector_backward_linear_case_1.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


M, N = 4, 3


def setup(seed=0):
    torch.manual_seed(seed)
    return torch.randn(M, N), torch.randn(N, requires_grad=True)


# ========================================================================
# 1. autograd 와 손계산
# ========================================================================

def compare():
    header("1. A^T v 와 맞는가")
    A, x = setup()
    v = torch.tensor([1.0, 0.5, -1.0, 2.0])
    y = A @ x
    print("  A 모양", tuple(A.shape), "  x 모양", tuple(x.shape),
          "  y 모양", tuple(y.shape))
    y.backward(v)
    hand = A.t() @ v
    print("  x.grad :", [round(t, 6) for t in x.grad.tolist()])
    print("  A^T v  :", [round(t, 6) for t in hand.tolist()])
    print("  같은가 :", torch.allclose(x.grad, hand))


# ========================================================================
# 2. 모양만 보아도 전치가 나온다
# ========================================================================

def shapes_force_it():
    header("2. 모양이 전치를 강요한다")
    A, x = setup()
    v = torch.randn(M)
    print(f"  A ({M},{N})   v ({M},)   결과는 x 와 같은 ({N},) 여야 한다")
    for name, expr in (("A @ v  ", lambda: A @ v),
                       ("A.t() @ v", lambda: A.t() @ v),
                       ("v @ A  ", lambda: v @ A)):
        try:
            r = expr()
            print(f"  {name} -> 모양 {tuple(r.shape)}")
        except RuntimeError as e:
            print(f"  {name} -> {str(e).splitlines()[0][:50]}")
    print("  A.t() @ v 와 v @ A 가 둘 다 (3,) 을 준다")
    print("  그리고 둘은 같은 값이다:", torch.allclose(A.t() @ v, v @ A))


# ========================================================================
# 3. 유도
# ========================================================================

def derivation():
    header("3. 유도")
    A, x = setup()
    v = torch.randn(M)
    print("  y_i = sum_j A_ij x_j 이므로  dy_i/dx_j = A_ij")
    print("  스칼라 s = sum_i v_i y_i 를 만들면")
    print("    ds/dx_j = sum_i v_i * dy_i/dx_j = sum_i v_i A_ij = (A^T v)_j")
    s = (v * (A @ x)).sum()
    s.backward()
    print("  s = (v*y).sum() 을 역전파한 x.grad:",
          [round(t, 6) for t in x.grad.tolist()])
    print("  A^T v                            :",
          [round(t, 6) for t in (A.t() @ v).tolist()])
    print("  backward(v) 는 이 s 를 만들어 역전파한 것과 같다")


# ========================================================================
# 4. 야코비는 A 자신이다
# ========================================================================

def jacobian_is_A():
    header("4. 야코비 = A")
    A, x = setup()
    y = A @ x
    rows = []
    for i in range(M):
        e = torch.zeros(M)
        e[i] = 1.0
        x.grad = None
        y.backward(e, retain_graph=True)
        rows.append(x.grad.clone())
    J = torch.stack(rows)
    print("  단위벡터로 모은 J 가 A 와 같은가:", torch.allclose(J, A))
    print("  그래서 v^T J = v^T A = A^T v 다 (벡터이므로 같다)")


# ========================================================================
# 5. A 에 대한 경사도 같이 본다
# ========================================================================

def grad_wrt_A():
    header("5. A 에 대한 경사")
    torch.manual_seed(0)
    A = torch.randn(M, N, requires_grad=True)
    x = torch.randn(N, requires_grad=True)
    v = torch.randn(M)
    (A @ x).backward(v)
    outer = torch.outer(v, x.detach())
    print("  A.grad 모양:", tuple(A.grad.shape), " = A 와 같다")
    print("  A.grad == outer(v, x) 인가:", torch.allclose(A.grad, outer))
    print("  y_i = sum_j A_ij x_j 이므로 dy_i/dA_ij = x_j")
    print("  ds/dA_ij = v_i x_j — 바깥곱이다")


def main():
    compare()
    shapes_force_it()
    derivation()
    jacobian_is_A()
    grad_wrt_A()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. A^T v 와 맞는가 ===
  A 모양 (4, 3)   x 모양 (3,)   y 모양 (4,)
  x.grad : [0.615178, -2.866987, -1.794756]
  A^T v  : [0.615178, -2.866987, -1.794756]
  같은가 : True

=== 2. 모양이 전치를 강요한다 ===
  A (4,3)   v (4,)   결과는 x 와 같은 (3,) 여야 한다
  A @ v   -> size mismatch, got input (4), mat (4x3), vec (4)
  A.t() @ v -> 모양 (3,)
  v @ A   -> 모양 (3,)
  A.t() @ v 와 v @ A 가 둘 다 (3,) 을 준다
  그리고 둘은 같은 값이다: True

=== 3. 유도 ===
  y_i = sum_j A_ij x_j 이므로  dy_i/dx_j = A_ij
  스칼라 s = sum_i v_i y_i 를 만들면
    ds/dx_j = sum_i v_i * dy_i/dx_j = sum_i v_i A_ij = (A^T v)_j
  s = (v*y).sum() 을 역전파한 x.grad: [0.377445, 1.423056, 0.093968]
  A^T v                            : [0.377445, 1.423056, 0.093968]
  backward(v) 는 이 s 를 만들어 역전파한 것과 같다

=== 4. 야코비 = A ===
  단위벡터로 모은 J 가 A 와 같은가: True
  그래서 v^T J = v^T A = A^T v 다 (벡터이므로 같다)

=== 5. A 에 대한 경사 ===
  A.grad 모양: (4, 3)  = A 와 같다
  A.grad == outer(v, x) 인가: True
  y_i = sum_j A_ij x_j 이므로 dy_i/dA_ij = x_j
  ds/dA_ij = v_i x_j — 바깥곱이다
```

## 2. 모양이 전치를 강요한다

$A$가 $(4, 3)$, $v$가 $(4,)$이고 결과는 $x$와 같은 $(3,)$이어야 한다.

| | 결과 |
|---|---|
| `A @ v` | `size mismatch, got input (4), mat (4x3), vec (4)` |
| `A.t() @ v` | `(3,)` |
| `v @ A` | `(3,)` |

**$(4,3)$과 $(4,)$으로 $(3,)$을 만드는 길이 전치뿐이다.** 그래서 식을 유도하기
전에도 전치가 붙는다는 것을 알 수 있다.

`A.t() @ v`와 `v @ A`는 같은 값이다. 벡터에는 행·열 구별이 없으므로
$v^\top A = A^\top v$다.

## 3. 유도

$y_i = \sum_j A_{ij} x_j$이므로 $\dfrac{\partial y_i}{\partial x_j} = A_{ij}$다.

`backward(v)`가 하는 일은 홑값 $s = \sum_i v_i y_i$를 만들어 역전파하는 것과
같으므로

$$
\frac{\partial s}{\partial x_j}
= \sum_i v_i \frac{\partial y_i}{\partial x_j}
= \sum_i v_i A_{ij}
= (A^\top v)_j
$$

코드가 그것을 확인한다 — `(v * y).sum().backward()`로 얻은 `x.grad`가
$A^\top v$와 소수점까지 같다.

## 4. 야코비는 $A$ 자신이다

단위벡터로 모은 $J$가 $A$와 같다. 그러니 $v^\top J = v^\top A$이고, 벡터이므로
$A^\top v$와 같은 것을 다르게 적은 것뿐이다.

**선형 사상에서는 야코비가 상수**다. $x$가 무엇이든 $A$다. 비선형 함수에서는
$x$에 따라 달라지고, 그래서 「비선형 함수의 전체 야코비안」에서 다시 본다.

## 5. $A$에 대한 경사는 바깥곱이다

```
A.grad 모양: (4, 3)  = A 와 같다
A.grad == outer(v, x) 인가: True
```

$y_i = \sum_j A_{ij} x_j$이므로 $\dfrac{\partial y_i}{\partial A_{ij}} = x_j$이고,
따라서

$$
\frac{\partial s}{\partial A_{ij}} = v_i x_j
$$

**바깥곱 $v x^\top$**이다. 「Linear 웨이트 관례」에서 본
$\partial L/\partial W = g^\top X$가 배치가 있는 꼴이고, 여기서는 표본 하나라
바깥곱이 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$v$를 단위벡터로 주면 `x.grad`가 무엇이 되는가? 그리고 전부 1로 주면?
**먼저 적고** 확인하라.

</div>

??? success "풀이"
    $A^\top v$에서 $v$가 $i$번째 단위벡터면 $A^\top$의 $i$번째 **열**, 곧 $A$의
    $i$번째 **줄**이다. 전부 1이면 $A$의 줄들을 다 더한 것이다.

    ```python
    A = torch.randn(4, 3)
    for v in ([1., 0, 0, 0], [0, 0, 0, 1.], [1., 1, 1, 1]):
        x = torch.randn(3, requires_grad=True)
        (A @ x).backward(torch.tensor(v))
        print(f"  v={str(v):18s} x.grad={[round(t, 4) for t in x.grad.tolist()]}")
    ```

    ```
      v=[1.0, 0, 0, 0]     x.grad=[1.541, -0.2934, -2.1788]   A[0] = [1.541, -0.2934, -2.1788]
      v=[0, 0, 0, 1.0]     x.grad=[-0.4033, -0.5966, 0.182]
      v=[1.0, 1, 1, 1]     x.grad=[2.1094, -1.1366, -4.1146]  A.sum(0) = [2.1094, -1.1366, -4.1146]
    ```

    짐작대로다. 첫 줄이 `A[0]`과 같고, 셋째 줄이 `A.sum(0)`과 같다.

    **$v$는 "$A$의 어느 줄을 얼마씩 섞을 것인가"**를 정한다. 「벡터 출력의
    역전파」에서 $v$를 "출력을 얼마씩 중히 볼 것인가"라고 한 것이 선형 사상에서는
    이렇게 구체화된다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
선형 사상 둘을 이으면 $y = A_2 A_1 x$다. VJP가 $A_1^\top (A_2^\top v)$인지
확인하고, **먼저 $A_2 A_1$을 곱해 두는 것**과 시간을 견주어라.

</div>

??? success "풀이"
    ```python
    A1, A2 = torch.randn(5, 4), torch.randn(3, 5)
    x = torch.randn(4, requires_grad=True)
    v = torch.randn(3)
    (A2 @ (A1 @ x)).backward(v)
    print("  x.grad        :", [round(t, 6) for t in x.grad.tolist()])
    print("  A1^T (A2^T v) :", [round(t, 6) for t in (A1.t() @ (A2.t() @ v)).tolist()])
    print("  (A2 A1)^T v   :", [round(t, 6) for t in ((A2 @ A1).t() @ v).tolist()])

    n = 2048
    B1, B2, vv = torch.randn(n, n), torch.randn(n, n), torch.randn(n)
    print(f"  순서대로  B1^T (B2^T v)  {timeit(lambda: B1.t() @ (B2.t() @ vv)):.3f} ms")
    print(f"  먼저 곱하고 (B2 B1)^T v  {timeit(lambda: (B2 @ B1).t() @ vv):.3f} ms")
    ```

    ```
      x.grad        : [-0.665507, -3.905338, -2.741568, 2.191602]
      A1^T (A2^T v) : [-0.665507, -3.905338, -2.741568, 2.191602]
      (A2 A1)^T v   : [-0.665507, -3.905338, -2.741568, 2.191602]

      순서대로  B1^T (B2^T v)   0.699 ms
      먼저 곱하고 (B2 B1)^T v  12.717 ms
    ```

    셋이 같은 값이고, **순서대로 미는 쪽이 18배 빠르다.**

    까닭은 셈의 크기다. $n = 2048$에서

    - **순서대로**: 행렬-벡터 곱 두 번 — $2n^2$번 곱셈
    - **먼저 곱하고**: 행렬-행렬 곱 한 번 — $n^3$번 곱셈

    $n$배 차이가 난다. 실제로 잰 18배는 BLAS가 행렬-행렬 곱을 훨씬 잘 최적화하기
    때문에 $n = 2048$보다 훨씬 작게 나온 것이다.

    **이것이 역전파가 야코비를 안 만드는 까닭**이다. 층이 $L$개인 그물에서
    야코비를 곱해 나가면 행렬-행렬 곱이 $L$번이지만, 벡터를 밀면 행렬-벡터
    곱이 $L$번이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
$v$가 $i$번째 단위벡터일 때 `A.grad`는 어떻게 생겼는가? 5절의 바깥곱으로
**먼저 예측하고** 확인하라.

</div>

??? success "풀이"
    $A.\text{grad} = v x^\top$이고 $v$가 $i$번째 단위벡터이면, **$i$번째 줄만
    $x^\top$이고 나머지는 0**일 것이다.

    ```python
    A = torch.randn(4, 3, requires_grad=True)
    x = torch.randn(3, requires_grad=True)
    for v in ([1., 0, 0, 0], [0, 1., 0, 0]):
        A.grad = None
        (A @ x).backward(torch.tensor(v))
        nz = (A.grad.abs() > 0).sum(1).tolist()
        print(f"  v={str(v):16s} A.grad 의 줄마다 0 이 아닌 원소 수: {nz}")
    ```

    ```
      v=[1.0, 0, 0, 0]   A.grad 의 줄마다 0 이 아닌 원소 수: [3, 0, 0, 0]
      v=[0, 1.0, 0, 0]   A.grad 의 줄마다 0 이 아닌 원소 수: [0, 3, 0, 0]
    ```

    예측대로다. **$i$번째 줄만 채워진다.**

    뜻을 읽으면 이렇다. $y_i$만 중히 본다고 했으니, $y_i$를 만드는 데 쓰인
    파라미터 — $A$의 $i$번째 줄 — 만 경사를 받는다. 나머지 줄은 $y_i$에 아무
    기여도 안 했으므로 할 말이 없다.

    「Linear 웨이트 관례」의 $(out, in)$ 배치가 여기서 다시 쓸모를 보인다.
    출력 하나에 해당하는 파라미터가 **한 줄에 모여 있으므로**, 그 출력만
    건드리는 일이 한 줄을 건드리는 일이 된다.
