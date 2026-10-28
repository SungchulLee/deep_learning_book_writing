# 열벡터로 적으면 — `v`의 모양은 출력과 꼭 맞아야 한다

「선형 사상의 VJP」에서 $x$를 $(3,)$으로 두었다. 수학책처럼 $(3,1)$ 열벡터로
두면 어떻게 되는가. **답은 같고 모양만 달라진다.**

그런데 `backward(v)`의 `v`는 출력과 **꼭 같은 모양**이어야 한다. 브로드캐스팅으로
넘어가지 않는다 — 이것이 반가운 소식이다. 틀리면 터져 준다.

반가운 소식에는 짝이 있다. 모양이 **조용히** 틀리는 자리는 따로 있고, 그쪽이
훨씬 자주 문다. 5절이 그 자리다.

## 1. 코드

```python
"""같은 수학, 다른 모양 — v 의 모양은 출력과 꼭 맞아야 한다.

「선형 사상의 VJP」에서 $x$ 를 $(3,)$ 으로 두었다. 수학책처럼
$(3,1)$ 열벡터로 두면 어떻게 되는가. **답은 같고 모양만 달라진다.**

그런데 `backward(v)` 의 `v` 는 출력과 **꼭 같은 모양**이어야 한다.
브로드캐스팅으로 넘어가지 않는다 — 이것이 반가운 소식이다.

반가운 소식에는 짝이 있다. 모양이 **조용히** 틀리는 자리는 따로
있고, 그쪽이 훨씬 자주 문다. 5절이 그 자리다.

실행: python 24_vector_backward_linear_case_2.py
"""
import warnings

import torch
import torch.nn.functional as F


def header(title):
    print(f"\n=== {title} ===")


# ========================================================================
# 1. 같은 수학, 다른 모양
# ========================================================================

def same_math():
    header("1. (3,) 와 (3,1)")
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    base = torch.randn(3)
    x1 = base.clone().requires_grad_(True)              # (3,)
    x2 = base.view(3, 1).clone().requires_grad_(True)   # (3,1)
    print("  x (3,)  -> y", tuple((A @ x1).shape))
    print("  x (3,1) -> y", tuple((A @ x2).shape))

    v = torch.tensor([1.0, 0.5, -1.0, 2.0])
    (A @ x1).backward(v)
    (A @ x2).backward(v.view(4, 1))
    print("  x1.grad 모양", tuple(x1.grad.shape),
          " x2.grad 모양", tuple(x2.grad.shape))
    print("  x1.grad", [round(t, 6) for t in x1.grad.tolist()])
    print("  x2.grad", [round(t, 6) for t in x2.grad.flatten().tolist()])
    print("  값이 같은가:", torch.allclose(x1.grad, x2.grad.flatten()))
    print("  경사는 언제나 그 텐서와 같은 모양으로 온다")


# ========================================================================
# 2. v 의 모양은 엄격하다
# ========================================================================

def strict_v():
    header("2. v 의 모양 검사")
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    for shape in ((4, 1), (4,), (1, 1), (1,)):
        x = torch.randn(3, 1, requires_grad=True)
        try:
            (A @ x).backward(torch.ones(shape))
            print(f"  v 모양 {str(shape):8s} -> 통과. x.grad "
                  f"{[round(t, 4) for t in x.grad.flatten().tolist()]}")
        except RuntimeError as e:
            print(f"  v 모양 {str(shape):8s} -> {str(e).splitlines()[0][:52]}")
    print("  출력이 (4,1) 이므로 v 도 (4,1) 이라야 한다")
    print("  브로드캐스팅으로 넘어가지 않는다 — 터져 주는 쪽이다")


# ========================================================================
# 3. 열벡터로 적으면 식이 수학책과 같아진다
# ========================================================================

def column_form():
    header("3. 열벡터 꼴")
    torch.manual_seed(0)
    A = torch.randn(4, 3)
    x = torch.randn(3, 1, requires_grad=True)
    v = torch.randn(4, 1)
    (A @ x).backward(v)
    print("  x.grad    :", [round(t, 6) for t in x.grad.flatten().tolist()])
    print("  A^T v     :", [round(t, 6) for t in (A.t() @ v).flatten().tolist()])
    print("  모양도 같은가:", x.grad.shape == (A.t() @ v).shape)
    print("  (3,1) 로 두면 A^T v 의 모양이 그대로 x.grad 다")
    print("  (3,) 로 두면 값은 같고 모양만 (3,) 로 눌린다")


# ========================================================================
# 4. 어느 쪽으로 적을 것인가
# ========================================================================

def which_form():
    header("4. 고르기")
    print("  (3,) 꼴")
    print("    장점: 코드가 짧다. 파이토치의 거의 모든 함수가 이쪽을 받는다")
    print("    단점: 수학책의 열벡터 식과 모양이 다르다")
    print("  (3,1) 꼴")
    print("    장점: A^T v 의 모양이 그대로 나온다")
    print("    단점: 어디선가 (3,) 와 섞이면 브로드캐스팅이 일어난다")
    print("  파이토치는 (배치, 피처) 를 쓰므로 표본 하나도 (1, 3) 이 보통이다")


# ========================================================================
# 5. 조용히 틀리는 자리
# ========================================================================

def the_silent_one():
    header("5. 모양이 조용히 무는 자리")
    torch.manual_seed(0)
    pred = torch.randn(8, 1)
    targ_col = torch.randn(8, 1)
    targ_flat = targ_col.flatten()
    print("  pred", tuple(pred.shape),
          " target 열", tuple(targ_col.shape),
          " target 평평", tuple(targ_flat.shape))

    a = F.mse_loss(pred, targ_col)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b = F.mse_loss(pred, targ_flat)
    print(f"  (8,1) vs (8,1) 로스 {a.item():.6f}")
    print(f"  (8,1) vs (8,)  로스 {b.item():.6f}   경고 {len(caught)}개")
    print("  (8,1) 과 (8,) 이 브로드캐스팅되어 (8,8) 이 된다")
    print("  표본 여덟 개를 서로 엇갈려 견준 셈이다")

    p1 = pred.clone().requires_grad_(True)
    F.mse_loss(p1, targ_col).backward()
    p2 = pred.clone().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F.mse_loss(p2, targ_flat).backward()
    print("  경사가 같은가:", torch.allclose(p1.grad, p2.grad))
    print("  바른 경사:", [round(t, 5) for t in p1.grad.flatten()[:3].tolist()])
    print("  틀린 경사:", [round(t, 5) for t in p2.grad.flatten()[:3].tolist()])
    print("  경고는 뜨지만 터지지는 않는다. 경고를 끄면 아무 표도 없다")


def main():
    same_math()
    strict_v()
    column_form()
    which_form()
    the_silent_one()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. (3,) 와 (3,1) ===
  x (3,)  -> y (4,)
  x (3,1) -> y (4, 1)
  x1.grad 모양 (3,)  x2.grad 모양 (3, 1)
  x1.grad [0.615178, -2.866987, -1.794756]
  x2.grad [0.615178, -2.866987, -1.794756]
  값이 같은가: True
  경사는 언제나 그 텐서와 같은 모양으로 온다

=== 2. v 의 모양 검사 ===
  v 모양 (4, 1)   -> 통과. x.grad [2.1094, -1.1366, -4.1146]
  v 모양 (4,)     -> Mismatch in shape: grad_output[0] has a shape of tor
  v 모양 (1, 1)   -> Mismatch in shape: grad_output[0] has a shape of tor
  v 모양 (1,)     -> Mismatch in shape: grad_output[0] has a shape of tor
  출력이 (4,1) 이므로 v 도 (4,1) 이라야 한다
  브로드캐스팅으로 넘어가지 않는다 — 터져 주는 쪽이다

=== 3. 열벡터 꼴 ===
  x.grad    : [0.377445, 1.423056, 0.093968]
  A^T v     : [0.377445, 1.423056, 0.093968]
  모양도 같은가: True
  (3,1) 로 두면 A^T v 의 모양이 그대로 x.grad 다
  (3,) 로 두면 값은 같고 모양만 (3,) 로 눌린다

=== 4. 고르기 ===
  (3,) 꼴
    장점: 코드가 짧다. 파이토치의 거의 모든 함수가 이쪽을 받는다
    단점: 수학책의 열벡터 식과 모양이 다르다
  (3,1) 꼴
    장점: A^T v 의 모양이 그대로 나온다
    단점: 어디선가 (3,) 와 섞이면 브로드캐스팅이 일어난다
  파이토치는 (배치, 피처) 를 쓰므로 표본 하나도 (1, 3) 이 보통이다

=== 5. 모양이 조용히 무는 자리 ===
  pred (8, 1)  target 열 (8, 1)  target 평평 (8,)
  (8,1) vs (8,1) 로스 2.094649
  (8,1) vs (8,)  로스 1.850731   경고 1개
  (8,1) 과 (8,) 이 브로드캐스팅되어 (8,8) 이 된다
  표본 여덟 개를 서로 엇갈려 견준 셈이다
  경사가 같은가: False
  바른 경사: [0.56506, 0.02748, -0.39554]
  틀린 경사: [0.4553, -0.0033, -0.47464]
  경고는 뜨지만 터지지는 않는다. 경고를 끄면 아무 표도 없다
```

## 2. 모양만 다르고 값은 같다

```
x (3,)  -> y (4,)
x (3,1) -> y (4, 1)

x1.grad 모양 (3,)    x2.grad 모양 (3, 1)
값이 같은가: True
```

**경사는 언제나 그 텐서와 같은 모양으로 온다.** 그래서 $x$를 어떻게 두든
`x.grad`의 모양이 따라온다.

달라지는 것은 `v`다. 출력이 $(4,1)$이면 `v`도 $(4,1)$이어야 한다.

## 3. `v`의 모양 검사는 엄격하다

| `v` 모양 | 결과 |
|---|---|
| `(4, 1)` | 통과 |
| `(4,)` | `Mismatch in shape` |
| `(1, 1)` | `Mismatch in shape` |
| `(1,)` | `Mismatch in shape` |

**브로드캐스팅되지 않는다.** $(4,)$는 $(4,1)$로 늘어날 수 있는 모양인데도
거절한다.

일부러 그렇게 만들어 둔 것으로 보인다. `v`는 "각 출력을 얼마씩 중히 보는가"라
출력과 일대일로 맞아야 뜻이 분명하고, 브로드캐스팅을 허용하면 어느 축이
늘어났는지 모르게 된다.

그래서 **여기서 틀리면 반드시 터진다.** 고마운 자리다.

## 4. 열벡터 꼴의 쓸모

```
x.grad    : [0.615178, -2.866987, -1.794756]
A^T v     : [0.615178, -2.866987, -1.794756]
모양도 같은가: True
```

$(3,1)$로 두면 $A^\top v$의 모양이 그대로 `x.grad`다. 수학책 식을 그대로 코드로
옮기기 좋다.

| | 장점 | 단점 |
|---|---|---|
| `(3,)` | 짧다. 파이토치 함수 대부분이 이쪽 | 수학책 식과 모양이 다르다 |
| `(3,1)` | $A^\top v$가 그대로 나온다 | $(3,)$와 섞이면 브로드캐스팅 |

실전에서는 파이토치가 `(배치, 피처)`를 쓰므로 표본 하나도 `(1, 3)`으로 두는
것이 보통이다. 열벡터 꼴은 손으로 유도한 식과 맞춰 볼 때 쓸모가 있다.

## 5. 모양이 조용히 무는 자리

`backward(v)`는 터져 주는데, **로스 안의 브로드캐스팅은 안 터진다.**

```
pred (8, 1)   target 열 (8, 1)   target 평평 (8,)

(8,1) vs (8,1) 로스 2.094649
(8,1) vs (8,)  로스 1.850731   경고 1개

경사가 같은가: False
바른 경사: [0.56506, 0.02748, -0.39554]
틀린 경사: [0.4553, -0.0033, -0.47464]
```

$(8,1)$과 $(8,)$이 브로드캐스팅되어 **$(8,8)$**이 된다. 표본 여덟 개를 서로
엇갈려 견준 셈이고, 로스도 경사도 전혀 다른 값이 나온다.

경고가 뜨기는 한다. 그런데

- 경고일 뿐 **터지지 않는다**
- 로그가 많으면 **묻힌다**
- `warnings.filterwarnings("ignore")`를 어디선가 걸어 두면 **아무 표도 없다**

모델 출력이 `(N, 1)`이고 라벨이 `(N,)`인 것은 흔한 꼴이다. `nn.Linear(d, 1)`이
`(N,1)`을 주고, CSV에서 읽은 라벨은 `(N,)`이다. 그래서 이 버그가 자주 생긴다.

> **고치는 법은 둘 중 아무거나.** `target.view(-1, 1)`로 라벨을 세우거나
> `pred.squeeze(-1)`로 예측을 눕힌다. 어느 쪽이든 로스를 계산하기 **전에**
> 모양을 맞춰 두는 것이 요점이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`nn.Linear(4, 1)`의 출력과 `(N,)` 라벨로 MSE를 계산하면 얼마나 틀리는가?
고치는 두 길이 같은 값을 주는지 확인하라.

</div>

??? success "풀이"
    ```python
    m = nn.Linear(4, 1)
    x = torch.randn(8, 4)
    y_col = torch.randn(8, 1)
    y_flat = y_col.flatten()
    print("  model(x) 모양:", tuple(m(x).shape))
    for name, t in (("(8,1)", y_col), ("(8,)", y_flat)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            l = F.mse_loss(m(x), t)
        print(f"  target {name:6s} 로스 {l.item():.6f}  경고 {len(w)}개")
    print("  squeeze 쪽:", round(F.mse_loss(m(x).squeeze(-1), y_flat).item(), 6))
    print("  view 쪽   :", round(F.mse_loss(m(x), y_flat.view(-1, 1)).item(), 6))
    ```

    ```
      model(x) 모양: (8, 1)
      target (8,1)  로스 2.663356  경고 0개
      target (8,)   로스 2.125755  경고 1개
      squeeze 쪽: 2.663356
      view 쪽   : 2.663356
    ```

    로스가 2.663에서 2.126으로 **20%** 틀린다. 그럴듯한 수라서 더 나쁘다 —
    `nan`이면 바로 알 텐데 이건 그냥 "로스가 좀 작네"로 보인다.

    고치는 두 길이 같은 값을 준다. 어느 쪽을 쓰든 되고, 모델 쪽을 건드리는
    `squeeze`가 더 흔하다 — 라벨은 데이터 적재 쪽에서 오므로 손대기 번거로울
    때가 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`cross_entropy`도 같은 문제가 있는가? `(N, C)` 로짓에 `(N,)` 라벨과 `(N, 1)`
라벨을 각각 주어 보라.

</div>

??? success "풀이"
    ```python
    logits = torch.randn(8, 3)
    targ = torch.randint(0, 3, (8,))
    print("  (N,C) 와 (N,):", round(F.cross_entropy(logits, targ).item(), 6))
    try:
        F.cross_entropy(logits, targ.view(-1, 1))
    except Exception as e:
        print("  (N,1) 을 주면:", str(e).splitlines()[0][:70])
    ```

    ```
      (N,C) 와 (N,): 1.332264
      (N,1) 을 주면: 0D or 1D target tensor expected, multi-target not supported
    ```

    **`cross_entropy`는 터진다.** MSE와 반대다.

    까닭은 기대하는 모양이 다르기 때문이다.

    | | 예측 | 목표 |
    |---|---|---|
    | `mse_loss` | `(N, 1)` | **같은 모양** `(N, 1)` |
    | `cross_entropy` | `(N, C)` | **다른 모양** `(N,)`, 클래스 번호 |

    `cross_entropy`의 목표는 값이 아니라 **번호**이므로 모양이 하나 적고,
    그래서 브로드캐스팅될 여지가 없다. 틀리면 바로 터진다.

    그러니 "`(N,)`과 `(N,1)` 중 무엇을 쓸까"에 일반적인 답이 없다. **쓰는 로스
    함수가 정한다.** 회귀에서 `(N,1)`로 맞추던 습관을 분류에 가져가면 여기서
    터진다 — 다행히 터진다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
3절에서 `v`의 모양이 엄격하다고 했다. **0차원 `v`**는 어떤가? 그리고 출력이
홑값일 때 `(1,)` 짜리 `v`를 주면?

</div>

??? success "풀이"
    ```python
    A = torch.randn(4, 3)
    x = torch.randn(3, requires_grad=True)
    y = A @ x
    for desc, v in (("정상 (4,)", torch.ones(4)),
                    ("0 차원", torch.tensor(1.0)),
                    ("(4,1)", torch.ones(4, 1))):
        x.grad = None
        try:
            y.backward(v, retain_graph=True)
            print(f"  {desc:10s} -> 통과")
        except RuntimeError as e:
            print(f"  {desc:10s} -> {str(e).splitlines()[0][:56]}")

    s = (A @ x).sum()
    x.grad = None
    try:
        s.backward(torch.ones(1))
        print("  홑값에 (1,) 짜리 v -> 통과")
    except RuntimeError as e:
        print("  홑값에 (1,) 짜리 v ->", str(e).splitlines()[0][:56])
    ```

    ```
      정상 (4,)    -> 통과  x.grad [2.1094, -1.1366, -4.1146]
      0 차원       -> Mismatch in shape: grad_output[0] has a shape of torch.S
      (4,1)      -> Mismatch in shape: grad_output[0] has a shape of torch.S
      홑값에 (1,) 짜리 v -> Mismatch in shape: grad_output[0] has a shape of torch.S
    ```

    **모두 거절한다.** 0차원도, $(4,1)$도, 홑값에 준 $(1,)$도.

    홑값 쪽이 재미있다. `s`가 0차원이므로 `v`도 0차원이어야 하고, $(1,)$은
    원소가 하나뿐이어도 **모양이 다르므로** 안 된다. 「형상, 자료형, 장치」에서
    본 0차원과 $(1,)$의 구별이 여기서도 걸린다.

    그래서 홑값에는 `backward()`를 인자 없이 부르거나
    `backward(torch.tensor(1.0))`을 준다.

    이 엄격함이 5절의 느슨함과 대비된다. **역전파의 입구는 모양을 꼭 맞추라고
    하는데, 순전파 안의 로스 계산은 브로드캐스팅을 허용한다.** 같은 라이브러리가
    같은 종류의 실수를 한쪽에서는 막고 한쪽에서는 통과시킨다.
