# Autograd 관련 속성 — 누구에게 `.grad` 가 생기는가

`requires_grad=True`로 만든 텐서라야 미분이 따라붙는다. 그런데 `requires_grad`가
`True`인 텐서라고 모두 `.grad`를 받지는 않는다. **잎(leaf)만 받는다.**

이 쪽은 그 경계를 재어서 보인다. 어느 텐서가 잎인지, 잎이 아닌 텐서의 `.grad`를
읽으면 무엇이 오는지, 그리고 그 경계를 몰라서 생기는 잘못 가운데 파이토치가
막아 주는 것과 **조용히 지나가는 것**을 가른다.

## 1. 코드

```python
"""자동 미분에 쓰이는 텐서 속성 — 누구에게 .grad 가 생기는가.

`requires_grad=True` 로 만들어도 `.grad` 가 채워지는 텐서는 그 가운데
일부뿐이다. 잎(leaf)만 받는다. 이 쪽은 그 경계를 재어서 보인다.

- requires_grad / is_leaf / grad_fn 의 조합 — 넷 가운데 셋만 있다
- 잎이 아닌 텐서의 .grad 는 None 이고, 파이토치가 경고를 띄운다
- 누적이 기본값이다. set_to_none=True 는 0 으로 채우기와 걸음이 다르다
- 홑값이 아닌 출력에는 v 를 주어야 한다(벡터-야코비 곱)
- backward 를 두 번 부르면 터지는 그래프와 안 터지는 그래프가 있다
- no_grad 없이 잎을 제자리에서 고치면 막힌다

실행: python 01_autograd_related.py
"""
import textwrap
import unicodedata
import warnings

import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


def pad(s, width):
    """한글은 두 칸을 차지하므로 글자 수로 세면 표가 어긋난다."""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))
    return str(s) + " " * max(0, width - w)


def fn_name(t):
    """grad_fn 을 이름으로만 적는다. 객체를 그대로 찍으면 메모리
    주소가 따라 나와 돌릴 때마다 달라진다."""
    return type(t.grad_fn).__name__ if t.grad_fn is not None else None


# ========================================================================
# 1. requires_grad 와 is_leaf — 넷 가운데 셋
# ========================================================================

def combinations():
    header("1. requires_grad x is_leaf")
    a = torch.tensor([1.0])
    b = torch.tensor([1.0], requires_grad=True)
    cases = (
        ("상수", a),
        ("파라미터", b),
        ("파라미터 x 2", b * 2),
        ("상수 x 2", a * 2),
        ("detach()", (b * 2).detach()),
    )
    print(pad("", 14) + f"{'requires_grad':>13s} {'is_leaf':>8s} {'grad_fn':>14s}")
    for name, t in cases:
        print(pad(name, 14) + f"{str(t.requires_grad):>13s} "
              f"{str(t.is_leaf):>8s} {str(fn_name(t)):>14s}")
    print("requires_grad=False 이면서 is_leaf=False 인 텐서는 없다")


# ========================================================================
# 2. .grad 는 잎에만 채워진다
# ========================================================================

def only_leaves_get_grad():
    header("2. 잎이 아니면 .grad 가 None 이다")
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    h = x * 3           # 잎이 아니다
    y = (h ** 2).sum()
    y.backward()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = h.grad
        print("h.grad :", g)
        # 경고문 뒤쪽에는 파이토치를 빌드한 기계의 경로가 붙어 나오므로
        # (돌리는 기계마다 다르다) 앞 두 문장만 적는다.
        msg = str(caught[0].message) if caught else "경고 없음"
        msg = msg[:msg.index("If you indeed")].strip()
        print("경고   :", textwrap.fill(msg, 66, subsequent_indent=" " * 9))
    print("x.grad :", x.grad)

    # retain_grad() 로 부탁하면 중간 텐서도 받는다
    x2 = torch.tensor([1.0, 2.0], requires_grad=True)
    h2 = x2 * 3
    h2.retain_grad()
    (h2 ** 2).sum().backward()
    print("retain_grad 뒤 h2.grad :", h2.grad)
    print("손으로  dy/dh = 2h = 2*[3,6] =", (2 * (x2 * 3)).tolist())


# ========================================================================
# 3. 누적이 기본값이다
# ========================================================================

def accumulation():
    header("3. 지우지 않으면 쌓인다")
    w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    for i in range(1, 4):
        (w ** 2).sum().backward()
        print(f"  {i}번째 backward 뒤 w.grad = {w.grad.tolist()}")
    w.grad.zero_()
    (w ** 2).sum().backward()
    print(f"  zero_() 하고 한 번 더  w.grad = {w.grad.tolist()}  (2w)")


def none_vs_zeros():
    header("4. set_to_none=True 는 0 으로 채우기와 다르다")
    got = []
    for set_to_none in (True, False):
        torch.manual_seed(0)
        m = nn.Linear(3, 1)
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
        x, y = torch.randn(5, 3), torch.randn(5, 1)
        nn.functional.mse_loss(m(x), y).backward()
        opt.step()                       # 모멘텀이 쌓인 한 걸음
        opt.zero_grad(set_to_none=set_to_none)
        opt.step()                       # 경사가 비었는데 또 걷는다
        got.append(m.weight.detach().flatten().clone())
    print("  set_to_none=True   :", "  ".join(f"{v:+.6f}" for v in got[0].tolist()))
    print("  set_to_none=False  :", "  ".join(f"{v:+.6f}" for v in got[1].tolist()))
    print(f"  가장 큰 차이        : {(got[0] - got[1]).abs().max():.6f}")
    print("  None 이면 step 이 그 파라미터를 건너뛴다.")
    print("  0 이면 모멘텀이 남아 계속 밀고 간다.")


# ========================================================================
# 5. 홑값이 아닌 출력 — 벡터-야코비 곱
# ========================================================================

def vjp():
    header("5. 홑값이 아니면 v 를 주어야 한다")
    torch.manual_seed(0)
    x = torch.randn(4, 3, requires_grad=True)
    A = torch.randn(2, 3)
    out = x @ A.t()                       # (4, 3) @ (3, 2) = (4, 2)
    v = torch.tensor([[1.0, 0.0], [0.5, -1.0], [0.0, 0.0], [2.0, 3.0]])

    try:
        out.backward()
    except RuntimeError as e:
        print("  v 없이 부르면:", str(e).splitlines()[0])

    out.backward(v)
    print("  x.grad.shape :", tuple(x.grad.shape))
    print("  손으로 v @ A 와 같은가:", torch.allclose(x.grad, v @ A))
    print("  x.grad[1]    :", [round(t, 4) for t in x.grad[1].tolist()])
    print("  (v @ A)[1]   :", [round(t, 4) for t in (v @ A)[1].tolist()])


# ========================================================================
# 6. backward 두 번 — 터지는 그래프와 안 터지는 그래프
# ========================================================================

def twice():
    header("6. 두 번 부르면 터지는가")
    builders = (
        ("x.sum()", lambda t: t.sum(), "저장할 것이 없다"),
        ("(x**2).sum()", lambda t: (t ** 2).sum(), "x 를 저장해 둔다"),
        ("x.exp().sum()", lambda t: t.exp().sum(), "exp(x) 를 저장해 둔다"),
    )
    for expr, build, why in builders:
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = build(x)
        y.backward()
        try:
            y.backward()
            print(f"  {expr:14s} 두 번째도 된다   grad={x.grad.tolist()}  ({why})")
        except RuntimeError as e:
            print(f"  {expr:14s} {str(e).splitlines()[0][:46]}...  ({why})")

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = (x ** 2).sum()
    y.backward(retain_graph=True)
    y.backward()
    print(f"  retain_graph=True 로는 두 번   grad={x.grad.tolist()}  (2*2x)")


# ========================================================================
# 7. no_grad 없이 잎을 고치면
# ========================================================================

def updating():
    header("7. 파라미터를 제자리에서 고치려면")
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    (p ** 2).sum().backward()
    try:
        p -= 0.1 * p.grad
    except RuntimeError as e:
        print("  no_grad 없이:", str(e).splitlines()[0])

    with torch.no_grad():
        p -= 0.1 * p.grad
    print("  no_grad 안에서:", [round(v, 4) for v in p.tolist()])
    print("  p.requires_grad 는 그대로:", p.requires_grad)
    print("  p.is_leaf 도 그대로:", p.is_leaf)


def main():
    combinations()
    only_leaves_get_grad()
    accumulation()
    none_vs_zeros()
    vjp()
    twice()
    updating()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. requires_grad x is_leaf ===
              requires_grad  is_leaf        grad_fn
상수                  False     True           None
파라미터               True     True           None
파라미터 x 2           True    False   MulBackward0
상수 x 2              False     True           None
detach()              False     True           None
requires_grad=False 이면서 is_leaf=False 인 텐서는 없다

=== 2. 잎이 아니면 .grad 가 None 이다 ===
h.grad : None
경고   : The .grad attribute of a Tensor that is not a leaf Tensor is being
         accessed. Its .grad attribute won't be populated during
         autograd.backward().
x.grad : tensor([18., 36.])
retain_grad 뒤 h2.grad : tensor([ 6., 12.])
손으로  dy/dh = 2h = 2*[3,6] = [6.0, 12.0]

=== 3. 지우지 않으면 쌓인다 ===
  1번째 backward 뒤 w.grad = [2.0, 4.0, 6.0]
  2번째 backward 뒤 w.grad = [4.0, 8.0, 12.0]
  3번째 backward 뒤 w.grad = [6.0, 12.0, 18.0]
  zero_() 하고 한 번 더  w.grad = [2.0, 4.0, 6.0]  (2w)

=== 4. set_to_none=True 는 0 으로 채우기와 다르다 ===
  set_to_none=True   : -0.021486  +0.282214  -0.422172
  set_to_none=False  : -0.036934  +0.257463  -0.374460
  가장 큰 차이        : 0.047712
  None 이면 step 이 그 파라미터를 건너뛴다.
  0 이면 모멘텀이 남아 계속 밀고 간다.

=== 5. 홑값이 아니면 v 를 주어야 한다 ===
  v 없이 부르면: grad can be implicitly created only for scalar outputs
  x.grad.shape : (4, 3)
  손으로 v @ A 와 같은가: True
  x.grad[1]    : [-0.551, 1.1166, -0.9087]
  (v @ A)[1]   : [-0.551, 1.1166, -0.9087]

=== 6. 두 번 부르면 터지는가 ===
  x.sum()        두 번째도 된다   grad=[2.0, 2.0]  (저장할 것이 없다)
  (x**2).sum()   Trying to backward through the graph a second ...  (x 를 저장해 둔다)
  x.exp().sum()  Trying to backward through the graph a second ...  (exp(x) 를 저장해 둔다)
  retain_graph=True 로는 두 번   grad=[4.0, 8.0]  (2*2x)

=== 7. 파라미터를 제자리에서 고치려면 ===
  no_grad 없이: a leaf Variable that requires grad is being used in an in-place operation.
  no_grad 안에서: [0.8, 1.6]
  p.requires_grad 는 그대로: True
  p.is_leaf 도 그대로: True
```

## 2. 잎이란 무엇인가

`requires_grad`와 `is_leaf`는 둘 다 참·거짓 하나씩이니 네 조합이 있어야 할 텐데,
실제로는 **셋만 있다.**

| | `requires_grad` | `is_leaf` | `grad_fn` |
|---|---|---|---|
| 상수 | False | True | None |
| 파라미터 | True | True | None |
| 파라미터 × 2 | True | **False** | `MulBackward0` |
| 상수 × 2 | False | True | None |
| `detach()` | False | True | None |

`requires_grad=False`이면서 `is_leaf=False`인 텐서는 없다. 미분을 따라가지 않는
텐서는 그래프 위에 있지도 않으니, 그래프 바깥에 있다는 말과 잎이라는 말이 같은
뜻이 된다. 그래서 `상수 × 2`도 잎이고, `detach()`로 떼어 낸 것도 잎이다.

잎의 뜻은 이렇게 적는 게 맞다. **그래프를 거꾸로 거슬러 올라가 더 갈 곳이 없는
텐서.** 그래서 잎은 두 갈래다 — 내가 손으로 만든 것(파라미터·입력), 그리고
미분을 따라가지 않아 애초에 그래프에 들지 않은 것(상수).

`grad_fn`은 그 텐서를 만든 연산을 가리킨다. 잎에는 만든 연산이 없으니 `None`이고,
`None`이 아니라면 그 텐서는 잎이 아니다. 둘은 같은 사실을 두 방향에서 말한다.

## 3. `.grad`는 잎에만 채워진다

`h = x * 3`은 `requires_grad=True`지만 잎이 아니다. `backward()`를 부른 뒤
`h.grad`를 읽으면 `None`이 온다.

조용히 `None`이 오는 것이 아니다. **파이토치가 경고를 띄운다.**

```
The .grad attribute of a Tensor that is not a leaf Tensor is being
accessed. Its .grad attribute won't be populated during autograd.backward().
```

이 경고는 두 번 읽을 값이 있다. 그냥 못 받는다는 말이 아니라, **받고 싶으면
부탁하라**는 말이기도 하다. `h.retain_grad()`를 미리 불러 두면 중간 텐서도
`.grad`를 받는다. 위 출력에서 $h = 3x = [3, 6]$이고
$\partial y/\partial h = 2h = [6, 12]$인데, 그 값이 그대로 들어왔다.

중간 그래디언트를 보려고 `retain_graph`를 켜는 사람이 흔한데, 그것은 다른
스위치다. **`retain_grad`는 값을 남기고, `retain_graph`는 그래프를 남긴다.**
이름이 한 글자 차이라 섞이기 쉽다.

## 4. 누적이 기본값이다

`.grad`는 덮어쓰는 것이 아니라 **더한다.** 같은 식으로 세 번 `backward()`를 부르면
`[2,4,6] → [4,8,12] → [6,12,18]`로 쌓인다.

기본값이 왜 누적인지는 「메모리 관리」에서 본 경사 누적이 답이다 — 배치를 쪼개
여러 번 `backward()`하고 한 번 `step()`하면 큰 배치와 같아진다. 그러니 누적은
버그가 아니라 기능이고, 매 걸음 지우는 쪽이 사용자의 몫이다.

### 0으로 채우는 것과 `None`으로 두는 것은 다르다

지우는 방법이 둘이다. `.grad.zero_()`로 0을 채우거나,
`opt.zero_grad(set_to_none=True)`로 `None`으로 두거나. 흔히 메모리를 아끼려는
차이로만 설명되는데, **걸음이 달라진다.**

모멘텀을 켠 SGD로 한 걸음 걷고 경사를 지운 다음 `step()`을 한 번 더 불러 보았다.

| | 가중치 |
|---|---|
| `set_to_none=True` | −0.021486  +0.282214  −0.422172 |
| `set_to_none=False` | −0.036934  +0.257463  −0.374460 |

0.047712만큼 다른 자리에 가 있다. `None`이면 `step()`이 그 파라미터를 아예
건너뛰는데, 0이면 경사가 0일 뿐이므로 **모멘텀이 남아 계속 밀고 간다.** 가중치
감쇠를 켜 두었다면 그것도 계속 작용한다.

파이토치 2.0부터 `zero_grad()`의 기본값이 `set_to_none=True`인 까닭이 여기 있다.
경사를 지웠다는 말은 "이번에는 이 파라미터에 할 말이 없다"는 뜻이어야 하고,
0은 "0만큼 움직여라"라는 다른 말이다.

## 5. 홑값이 아닌 출력에는 $v$를 주어야 한다

`out`이 홑값이면 `out.backward()`로 끝나지만, 모양이 있으면 거절한다.

```
grad can be implicitly created only for scalar outputs
```

미분의 결과가 야코비 행렬이어서 그렇다. `out`이 $(4,2)$이고 `x`가 $(4,3)$이면
야코비는 $4 \times 2 \times 4 \times 3$짜리이고, 그것을 `x.grad`에 담을 수 없다.
`.grad`는 언제나 텐서와 같은 모양이다.

그래서 `backward(v)`는 야코비를 만들지 않고 **$v^\top J$만 계산한다**(벡터-야코비
곱). $v$는 "출력 각 자리를 얼마씩 중히 보겠다"는 무게이고, 홑값일 때
`backward()`가 되는 것은 $v=1$을 넣어 주기 때문이다.

$\text{out} = xA^\top$이면 $x$에 대한 VJP는 $vA$다. 코드가 그 둘을 견주어
`True`를 받았고, 두 번째 줄의 값도 소수점까지 같다.

$$
x.\text{grad}[1] = (vA)[1] = [-0.551,\ 1.1166,\ -0.9087]
$$

## 6. 두 번 부르면 터지는 그래프와 안 터지는 그래프

`backward()`를 두 번 부르면 터진다고 흔히 말하지만, 그렇지 않은 그래프가 있다.

| 식 | 두 번째 `backward()` | 까닭 |
|---|---|---|
| `x.sum()` | **된다** | 역전파에 저장해 둘 값이 없다 |
| `(x**2).sum()` | `RuntimeError` | 미분에 `x`가 필요해 저장해 둔다 |
| `x.exp().sum()` | `RuntimeError` | 미분에 `exp(x)`가 필요해 저장해 둔다 |

터지는 것은 그래프가 사라져서가 아니라 **저장해 둔 중간값이 해제되어서**다.
$\frac{d}{dx}x^2 = 2x$를 계산하려면 $x$가 있어야 하고, $\frac{d}{dx}e^x = e^x$를
계산하려면 $e^x$가 있어야 한다. 첫 역전파가 끝나면 파이토치는 그 버퍼를 놓아
준다. 반면 `sum`의 미분은 1이라서 아무것도 필요 없고, 그래서 두 번째도 그냥
된다.

이것이 **같은 코드가 어떤 식에서는 터지고 어떤 식에서는 안 터지는** 까닭이다.
`retain_graph=True`를 켜면 버퍼를 남겨 두므로 두 번 걸을 수 있고, 그만큼 메모리를
더 오래 쥔다.

## 7. 파라미터를 제자리에서 고치려면

옵티마이저 없이 손으로 가중치를 고쳐 보자. `p -= 0.1 * p.grad`를 그냥 쓰면
막힌다.

```
a leaf Variable that requires grad is being used in an in-place operation.
```

막는 까닭은 분명하다. `p`는 잎이고 미분이 따라붙어 있는데 그 값을 제자리에서
바꾸면, 이미 쌓아 둔 그래프가 가리키는 값과 달라진다. `torch.no_grad()` 안에서는
이 검사를 넘기고, `p`는 그대로 잎이며 `requires_grad`도 `True`로 남는다.

여기까지는 파이토치가 지켜 준다. **지켜 주지 않는 길도 있다** — 연습문제 2가
그 자리다.

## 8. 정리

| 묻는 것 | 보는 속성 |
|---|---|
| 미분을 따라가는가 | `requires_grad` |
| `.grad`를 받는가 | `is_leaf` (그리고 `retain_grad()`로 예외를 만든다) |
| 무엇이 이 텐서를 만들었나 | `grad_fn` |

한 줄로 줄이면 이렇다. **`requires_grad`는 추적 여부이고, `is_leaf`는 수령
여부다.** 둘을 하나로 생각하면 중간 텐서의 `.grad`가 `None`인 것이 버그처럼
보이고, 경고문을 읽어도 무슨 말인지 모른다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$f(x) = x^3 - 2x^2 + x$에 대해 autograd로 $f'(3)$을 구하고, 손으로 미분한 값과
견주어라.

</div>

??? success "풀이"
    ```python
    x = torch.tensor(3.0, requires_grad=True)
    f = x ** 3 - 2 * x ** 2 + x
    f.backward()
    print("f(3)      :", f.item())
    print("x.grad    :", x.grad.item())
    print("손으로 3x^2-4x+1 =", 3 * 3 ** 2 - 4 * 3 + 1)
    ```

    ```
    f(3)      : 12.0
    x.grad    : 16.0
    손으로 3x^2-4x+1 = 16
    ```

    $f'(x) = 3x^2 - 4x + 1$이므로 $f'(3) = 27 - 12 + 1 = 16$이다.

    `x`는 손으로 만든 잎이므로 `.grad`를 받는다. 중간에 생긴 `f`는 잎이 아니어서
    `f.grad`는 `None`이고, 읽으려 하면 3절의 경고가 뜬다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`detach()`와 `torch.no_grad()`는 둘 다 미분 추적을 끊는다. 그러나 같지 않다.
두 가지로 만든 텐서가 원본과 **저장소를 공유하는지** 확인하라. 그리고 `detach()`로
받은 손잡이를 제자리에서 고친 뒤 `backward()`를 불러, 나온 경사가 맞는지 보라.

</div>

??? success "풀이"
    ```python
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    d = w.detach()
    with torch.no_grad():
        n = w * 1
    print(f"  detach()        requires_grad={d.requires_grad}  "
          f"저장소 공유={d.data_ptr() == w.data_ptr()}")
    print(f"  no_grad 안 w*1  requires_grad={n.requires_grad}  "
          f"저장소 공유={n.data_ptr() == w.data_ptr()}")

    # detach 로 받은 손잡이를 고치면 원본이 바뀐다
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    y = w * 3
    y.detach()[0] = 99.0
    (y ** 2).sum().backward()
    print("  detach 로 중간값을 99 로 고치고 backward")
    print("    w.grad     =", w.grad.tolist())
    print("    참값 18w   =", (18 * torch.tensor([1.0, 2.0])).tolist())
    print("    터지지도, 경고하지도 않는다")

    # 잎을 직접 고치려 하면 막는다 (7절)
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    try:
        p[0] = 99.0
    except RuntimeError as e:
        print("  잎을 직접:", str(e).splitlines()[0])
    ```

    ```
      detach()        requires_grad=False  저장소 공유=True
      no_grad 안 w*1  requires_grad=False  저장소 공유=False
      detach 로 중간값을 99 로 고치고 backward
        w.grad     = [594.0, 36.0]
        참값 18w   = [18.0, 36.0]
        터지지도, 경고하지도 않는다
      잎을 직접: a view of a leaf Variable that requires grad is being used in an in-place operation.
    ```

    **`detach()`는 저장소를 공유한다.** 미분만 떼어 낸 같은 데이터이고, 그래서
    고치면 원본이 함께 바뀐다. `no_grad()` 안의 `w * 1`은 연산이 새 텐서를 만드니
    공유하지 않는다. 둘 다 `requires_grad=False`인데 뒤가 다르다.

    그 다음이 이 문제의 핵심이다. $y = 3w$, 로스 $= \sum (3w)^2$이므로 참값은
    $18w = [18, 36]$인데 **`[594, 36]`이 나왔다.** 594는 $2 \times 99 \times 3$,
    곧 99로 고쳐진 값으로 계산한 경사다. 역전파가 저장해 둔 버퍼를 `detach()`
    손잡이로 덮어썼고, 엔진은 덮어쓴 것을 그대로 읽었다.

    **터지지도, 경고하지도 않는다.** 7절에서 같은 종류의 잘못을 파이토치가
    `RuntimeError`로 막았고, 잎의 뷰를 직접 고치려 해도 막는다. 그런데
    `detach()`를 한 번 거치면 그 검사를 지나가고, 남는 것은 조용히 틀린 숫자다.

    그래서 `detach()`는 "미분을 끊는다"로만 기억할 것이 아니다. **읽기 전용으로
    쓸 때만 안전하다.** 값을 들고 나가 고칠 작정이면 `detach().clone()`이라야
    한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
5절은 야코비를 만들지 않고 $v^\top J$만 구했다. 그렇다면 $v$를 단위벡터로 바꿔
가며 여러 번 불러 **야코비를 한 줄씩 모으면** 전체를 얻을 수 있다.
`out = x @ A.t()`에서 `x`가 $(1,3)$일 때 이렇게 야코비를 세우고, 그것이 $A$와
같은지 확인하라. 6절이 왜 여기 필요한지도 적어라.

</div>

??? success "풀이"
    ```python
    torch.manual_seed(0)
    A = torch.randn(2, 3)
    x = torch.randn(1, 3, requires_grad=True)
    out = x @ A.t()
    rows = []
    for j in range(2):
        v = torch.zeros(1, 2)
        v[0, j] = 1.0
        x.grad = None                      # 누적을 막는다 (4절)
        out.backward(v, retain_graph=True)  # 그래프를 남긴다 (6절)
        rows.append(x.grad[0].clone())
    J = torch.stack(rows)
    print("  J == A ?", torch.allclose(J, A))
    for r, a in zip(J.tolist(), A.tolist()):
        print("    J", [round(t, 4) for t in r], "  A", [round(t, 4) for t in a])
    ```

    ```
      J == A ? True
        J [1.541, -0.2934, -2.1788]   A [1.541, -0.2934, -2.1788]
        J [0.5684, -1.0845, -1.3986]   A [0.5684, -1.0845, -1.3986]
    ```

    $v$를 $j$번째 단위벡터로 두면 $v^\top J$는 $J$의 $j$번째 줄이다. 두 번
    불러 두 줄을 모으면 야코비가 완성되고, $\text{out} = xA^\top$의 야코비는
    바로 $A$이므로 둘이 일치한다.

    6절이 필요한 까닭은 **두 번째 호출 때문**이다. `retain_graph=True`가 없으면
    첫 호출이 버퍼를 놓아 주고 둘째 호출이 `RuntimeError`로 터진다. 그리고
    `x.grad = None`을 매번 넣지 않으면 4절의 누적이 일어나 둘째 줄이 첫째 줄과
    더해진 채 나온다 — **터지지 않고 틀린 값이 나오는 쪽**이라 더 위험하다.

    이 방법의 값은 출력이 몇 개인지에 달려 있다. 출력이 $m$개면 역전파를 $m$번
    불러야 하므로, 출력이 하나인 로스에서는 한 번으로 끝나고 출력이 큰 함수에서는
    비싸다. 로스를 홑값으로 만들어 쓰는 관행이 여기서 나온다. 야코비가 정말
    필요하면 `torch.autograd.functional.jacobian`이나 `torch.func.jacrev`를
    쓰는 쪽이 낫다.
