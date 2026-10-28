# 제자리 연산 — 막히는 자리와 막히지 않는 자리

밑줄로 끝나는 연산(`add_`, `mul_`, `zero_`)은 새 텐서를 만들지 않고 가리키는
메모리를 직접 고친다. 한 장을 덜 잡으니 메모리가 빠듯할 때 쓸모가 있다.

문제는 그 텐서에 미분이 걸려 있을 때다. 역전파가 쓰려고 적어 둔 값을 덮어쓰면
셈이 어긋난다. 파이토치가 그것을 막기는 하는데, **한 가지 방식으로 막지 않는다.**

| 어디에 쓰는가 | 어떻게 되는가 |
|---|---|
| 잎(파라미터) | **그 자리에서** 막는다 |
| 중간 텐서 | 통과시킨다. 역전파가 그 값을 쓰려 할 때만 터진다 |

그래서 같은 `add_` 한 줄이 어떤 식에서는 멀쩡하고 어떤 식에서는 터진다. 무엇이
가르는지가 이 쪽의 이야기다.

## 1. 코드

```python
"""제자리 연산 — 막히는 자리와 막히지 않는 자리.

밑줄로 끝나는 연산(`add_`, `mul_`, `zero_`)은 새 텐서를 만들지 않고
가리키는 메모리를 직접 고친다. 메모리를 아끼지만 미분이 걸린 텐서를
고치면 셈이 어긋날 수 있다.

파이토치가 그것을 어떻게 막는지가 이 쪽의 이야기인데, 한 가지로
막지 않는다.

- 잎에는 **그 자리에서** 막는다
- 중간 텐서에는 막지 않는다. 통과시켜 두었다가, 역전파가 그 값을
  쓰려 할 때만 터진다
- 그래서 같은 제자리 연산이 어떤 식에서는 멀쩡하고 어떤 식에서는
  터진다. 그 연산이 역전파에 무엇을 저장해 두었느냐가 가른다

실행: python 07_inplace_operations.py
"""
import time

import torch
import torch.nn as nn


def header(title):
    print(f"\n=== {title} ===")


def spread(fn, rep=7):
    """한 번 잰 시간은 측정이 아니다."""
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3, max(ts) * 1e3


# ========================================================================
# 1. 밑줄 관례와 별칭
# ========================================================================

def basics():
    header("1. 밑줄은 '제자리에서' 라는 표시다")
    x = torch.tensor([1.0, 2.0, 3.0])
    before = x.data_ptr()
    y = x.add(10)            # 밑줄 없음 — 새 텐서
    print("  x.add(10)  -> 새 텐서, x 는", x.tolist())
    print("    같은 메모리인가:", y.data_ptr() == before)
    x.add_(10)               # 밑줄 있음 — 제자리
    print("  x.add_(10) -> x 가", x.tolist())
    print("    같은 메모리인가:", x.data_ptr() == before)

    a = torch.zeros(3)
    b = a                    # 같은 텐서를 가리키는 다른 이름
    a.add_(5)
    print("  a 를 고치면 b 도:", b.tolist(), " 같은 객체인가:", a is b)


# ========================================================================
# 2. 잎에는 그 자리에서 막는다
# ========================================================================

def leaves_are_blocked():
    header("2. 잎에는 막는다")
    leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    try:
        leaf.add_(1)
    except RuntimeError as e:
        print("  leaf.add_(1):", str(e).splitlines()[0])

    # 길 1: 제자리가 아닌 연산을 쓴다
    ok = leaf + 1
    print("  leaf + 1 은 된다:", ok.tolist(), " grad_fn",
          type(ok.grad_fn).__name__)

    # 길 2: no_grad 안에서 고친다 (옵티마이저가 하는 일)
    with torch.no_grad():
        leaf.add_(1)
    print("  no_grad 안에서는 된다:", leaf.tolist())
    print("  requires_grad 는 그대로:", leaf.requires_grad)


# ========================================================================
# 3. 중간 텐서에는 막지 않는다
# ========================================================================

def non_leaves_pass():
    header("3. 중간 텐서에는 막지 않는다")
    print("  y 를 만들고 y.add_(1) 한 다음 역전파해 본다")
    cases = (
        ("y = x * 2", lambda t: t * 2, "아무것도"),
        ("y = x ** 2", lambda t: t ** 2, "입력 x"),
        ("y = x.exp()", lambda t: t.exp(), "출력 y"),
        ("y = x.relu()", lambda t: t.relu(), "출력 y"),
    )
    for label, build, saves in cases:
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = build(x)
        y.add_(1.0)                      # 넷 다 여기서는 통과한다
        try:
            y.sum().backward()
            result = f"역전파 통과  x.grad={[round(v, 3) for v in x.grad.tolist()]}"
        except RuntimeError:
            result = "역전파에서 RuntimeError"
        print(f"  {label:13s} 역전파가 쓰는 것 {saves:8s} -> {result}")
    print("  제자리 연산 자체는 넷 다 통과했다. 갈린 것은 역전파다")


# ========================================================================
# 4. 그러면 nn.ReLU(inplace=True) 는 왜 안전한가
# ========================================================================

def inplace_relu():
    header("4. nn.ReLU(inplace=True)")
    for flag in (False, True):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(inplace=flag),
                            nn.Linear(8, 1))
        x = torch.randn(3, 4)
        net.zero_grad()
        net(x).sum().backward()
        g = sum(p.grad.abs().sum().item() for p in net.parameters())
        print(f"  inplace={flag!s:5s} 역전파 통과, 경사 절댓값 합 {g:.6f}")
    print("  두 값이 같다. relu 를 제자리에서 해도 셈이 바뀌지 않는다")
    print("  파이토치가 제 연산의 제자리 꼴은 역전파까지 맞춰 두었기 때문이다")

    x = torch.tensor([-1.0, 2.0], requires_grad=True)
    y = torch.relu(x)
    y.add_(1.0)                          # 내가 손으로 끼워 넣은 제자리 연산
    try:
        y.sum().backward()
        print("  손으로 add_ 를 넣어도 통과")
    except RuntimeError as e:
        print("  손으로 add_ 를 넣으면:", str(e).splitlines()[0][:66])
    print("  파이토치가 아는 제자리와 내가 끼워 넣은 제자리는 다르다")


# ========================================================================
# 5. 무엇을 아끼는가
# ========================================================================

def savings():
    header("5. 아끼는 것")
    for shape in ((100, 100), (1000, 1000), (4000, 4000)):
        big = torch.randn(*shape)
        mb = big.numel() * 4 / 1e6
        o = spread(lambda: big + 1.0)
        i = spread(lambda: big.add_(1.0))
        print(f"  {str(shape):14s} 한 장 {mb:6.2f} MB")
        print(f"    새로 만들기 {o[0]:7.3f} ~ {o[1]:7.3f} ms")
        print(f"    제자리     {i[0]:7.3f} ~ {i[1]:7.3f} ms")
    print("  새로 만드는 쪽은 결과를 담을 한 장을 더 잡고 거기 적는다")


def main():
    basics()
    leaves_are_blocked()
    non_leaves_pass()
    inplace_relu()
    savings()


if __name__ == "__main__":
    main()
```

**출력:**

```

=== 1. 밑줄은 '제자리에서' 라는 표시다 ===
  x.add(10)  -> 새 텐서, x 는 [1.0, 2.0, 3.0]
    같은 메모리인가: False
  x.add_(10) -> x 가 [11.0, 12.0, 13.0]
    같은 메모리인가: True
  a 를 고치면 b 도: [5.0, 5.0, 5.0]  같은 객체인가: True

=== 2. 잎에는 막는다 ===
  leaf.add_(1): a leaf Variable that requires grad is being used in an in-place operation.
  leaf + 1 은 된다: [2.0, 3.0, 4.0]  grad_fn AddBackward0
  no_grad 안에서는 된다: [2.0, 3.0, 4.0]
  requires_grad 는 그대로: True

=== 3. 중간 텐서에는 막지 않는다 ===
  y 를 만들고 y.add_(1) 한 다음 역전파해 본다
  y = x * 2     역전파가 쓰는 것 아무것도     -> 역전파 통과  x.grad=[2.0, 2.0]
  y = x ** 2    역전파가 쓰는 것 입력 x     -> 역전파 통과  x.grad=[2.0, 4.0]
  y = x.exp()   역전파가 쓰는 것 출력 y     -> 역전파에서 RuntimeError
  y = x.relu()  역전파가 쓰는 것 출력 y     -> 역전파에서 RuntimeError
  제자리 연산 자체는 넷 다 통과했다. 갈린 것은 역전파다

=== 4. nn.ReLU(inplace=True) ===
  inplace=False 역전파 통과, 경사 절댓값 합 10.365578
  inplace=True  역전파 통과, 경사 절댓값 합 10.365578
  두 값이 같다. relu 를 제자리에서 해도 셈이 바뀌지 않는다
  파이토치가 제 연산의 제자리 꼴은 역전파까지 맞춰 두었기 때문이다
  손으로 add_ 를 넣으면: one of the variables needed for gradient computation has been modi
  파이토치가 아는 제자리와 내가 끼워 넣은 제자리는 다르다

=== 5. 아끼는 것 ===
  (100, 100)     한 장   0.04 MB
    새로 만들기   0.002 ~   0.006 ms
    제자리       0.002 ~   0.003 ms
  (1000, 1000)   한 장   4.00 MB
    새로 만들기   0.142 ~   0.323 ms
    제자리       0.038 ~   0.049 ms
  (4000, 4000)   한 장  64.00 MB
    새로 만들기   2.399 ~   3.915 ms
    제자리       1.390 ~   1.750 ms
  새로 만드는 쪽은 결과를 담을 한 장을 더 잡고 거기 적는다
```

## 2. 밑줄은 "제자리에서"라는 표시다

```
x.add(10)  -> 새 텐서, x 는 [1.0, 2.0, 3.0]    같은 메모리인가: False
x.add_(10) -> x 가 [11.0, 12.0, 13.0]          같은 메모리인가: True
```

밑줄이 없으면 결과를 새 텐서에 담아 돌려주고 `x`는 그대로다. 밑줄이 있으면 `x`가
가리키는 메모리를 고친다.

고치는 것은 **텐서이지 이름이 아니다.** 같은 텐서를 가리키는 이름이 둘이면 둘 다
바뀐 것을 본다.

```
a 를 고치면 b 도: [5.0, 5.0, 5.0]   같은 객체인가: True
```

## 3. 잎에는 그 자리에서 막는다

```
leaf.add_(1): a leaf Variable that requires grad is being used in an
in-place operation.
```

잎은 파라미터다. 역전파가 끝나고 옵티마이저가 쓸 값이므로, 그래프가 살아 있는
동안 고치지 못하게 막아 둔다. 빠져나가는 길이 둘이다.

- **제자리가 아닌 연산을 쓴다** — `leaf + 1`은 새 텐서를 만들고 `AddBackward0`을
  달아 준다. 그래프에 한 칸이 더 생기는 대신 아무것도 안 깨진다.
- **`no_grad` 안에서 고친다** — 옵티마이저가 하는 일이 이것이다. 기록을 끄고
  제자리에서 고치며, `requires_grad`는 그대로 `True`로 남는다.

## 4. 중간 텐서에는 막지 않는다

여기가 이 쪽의 핵심이다. `y = x * 2`처럼 계산으로 생긴 텐서에 `y.add_(1)`을 하면
**막지 않는다.** 네 가지를 나란히 돌려 보았다.

| 식 | 역전파가 쓰는 것 | `y.add_(1)` 뒤 역전파 |
|---|---|---|
| `y = x * 2` | 아무것도 | **통과** — `x.grad = [2, 2]` |
| `y = x ** 2` | 입력 `x` | **통과** — `x.grad = [2, 4]` |
| `y = x.exp()` | 출력 `y` | `RuntimeError` |
| `y = x.relu()` | 출력 `y` | `RuntimeError` |

**제자리 연산 자체는 넷 다 통과했다.** 갈린 것은 역전파다. 그리고 규칙이 보인다.

> 그 연산의 역전파가 **출력을 적어 두었으면** 터지고, 입력을 적어 두었거나
> 아무것도 안 적어 두었으면 통과한다.

$\frac{d}{dx}e^x = e^x$이므로 `exp`는 결과를 들고 있어야 하고, 그 결과를
고쳤으니 어긋난다. $\frac{d}{dx}x^2 = 2x$는 입력만 있으면 되고 입력은 건드리지
않았으므로 멀쩡하다. 터질 때 나오는 말이 정확히 그 뜻이다.

```
one of the variables needed for gradient computation has been modified
by an inplace operation
```

이것이 「데이터 관련 속성」에서 본 버전 카운터가 하는 일이다. 파이토치는 텐서마다
`_version`을 세어 두고, 역전파가 쓰려는 값의 버전이 적어 둘 때와 다르면 거절한다.
**막는 것이 아니라 들키면 거절하는 것**이고, 그래서 쓸 일이 없는 값은 고쳐도
아무 일이 없다.

## 5. 그러면 `nn.ReLU(inplace=True)`는 왜 안전한가

`relu`의 역전파는 출력을 쓴다. 위 표대로라면 제자리 `relu`는 위험해야 한다.
그런데 `nn.ReLU(inplace=True)`는 어디서나 쓰인다.

```
inplace=False 역전파 통과, 경사 절댓값 합 10.365578
inplace=True  역전파 통과, 경사 절댓값 합 10.365578
```

소수점까지 같다. 까닭은 **파이토치가 제 연산의 제자리 꼴을 역전파까지 맞춰
두었기** 때문이다. 제자리 `relu`의 출력이 곧 입력 자리에 있는 값이고, 역전파는
그 값으로 부호를 판별하면 되므로 어긋날 것이 없다. 라이브러리가 아는 제자리다.

반면 내가 손으로 끼워 넣은 제자리 연산은 다르다.

```
y = torch.relu(x)
y.add_(1.0)          # 파이토치가 모르는 제자리
-> one of the variables needed for gradient computation has been modified
```

**`inplace=True` 옵션을 믿는 것과 아무 데나 `add_`를 쓰는 것은 다른 일이다.**
앞은 그 연산을 만든 쪽이 책임지고 맞춰 둔 것이고, 뒤는 내가 그래프 중간에
손을 넣은 것이다.

## 6. 무엇을 아끼는가

| 크기 | 한 장 | 새로 만들기 | 제자리 |
|---|---|---|---|
| 100×100 | 0.04 MB | 0.002 ~ 0.006 ms | 0.002 ~ 0.003 ms |
| 1000×1000 | 4.00 MB | 0.142 ~ 0.323 ms | 0.038 ~ 0.049 ms |
| 4000×4000 | 64.00 MB | 2.399 ~ 3.915 ms | 1.390 ~ 1.750 ms |

작은 텐서에서는 범위가 겹쳐 차이를 말할 수 없다. 커지면 벌어져서, 1000×1000에서
제자리 쪽이 서너 배 빠르다. 그러나 **시간보다 메모리가 본론**이다. 4000×4000에서
새로 만드는 쪽은 64 MB를 더 잡아야 하고, 그 한 장이 없어서 학습이 멈추는 일은
흔하지만 0.1밀리초가 아쉬운 일은 드물다.

## 7. 정리

| | 써도 되는가 |
|---|---|
| `no_grad` 안에서 파라미터 갱신 | **된다** — 옵티마이저가 하는 일이다 |
| 텐서 초기화 (`zero_`, `fill_`, `normal_`) | **된다** — 그래프가 없다 |
| 라이브러리의 `inplace=True` 옵션 | **된다** — 역전파까지 맞춰져 있다 |
| 그래프 중간 텐서에 손으로 | 되기도 하고 안 되기도 한다 — 걸면 안 된다 |

마지막 줄이 이 쪽에서 가져갈 것이다. 중간 텐서의 제자리 연산은 **되는지 안 되는지
그 식을 보아야 알 수 있고**, 식을 조금 고치면 어제 되던 것이 오늘 터진다.
아끼는 것이 수십 MB가 아니라면 그냥 새 텐서를 만드는 편이 낫다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`zero_()`·`fill_(7)`·`normal_()`로 텐서를 채워 보고, 매번 `data_ptr()`이 그대로인지
확인하라. 이 셋은 왜 그래프를 걱정하지 않아도 되는가?

</div>

??? success "풀이"
    ```python
    t = torch.empty(4)
    base = t.data_ptr()
    torch.manual_seed(0)
    for name, fn in (("zero_()", lambda: t.zero_()),
                     ("fill_(7)", lambda: t.fill_(7.0)),
                     ("normal_()", lambda: t.normal_(0, 1))):
        fn()
        print(f"  {name:11s} 같은 메모리 {t.data_ptr() == base}   "
              f"{[round(v, 3) for v in t.tolist()]}")
    print("  requires_grad:", t.requires_grad)
    ```

    ```
      zero_()     같은 메모리 True   [0.0, 0.0, 0.0, 0.0]
      fill_(7)    같은 메모리 True   [7.0, 7.0, 7.0, 7.0]
      normal_()   같은 메모리 True   [1.541, -0.293, -2.179, 0.568]
      requires_grad: False
    ```

    `data_ptr()`이 셋 다 그대로다. 세 연산 모두 자리를 새로 잡지 않고 있던 칸에
    값을 적는다.

    그래프를 걱정하지 않아도 되는 까닭은 `requires_grad`가 `False`이기 때문이다.
    미분을 따라가지 않는 텐서는 역전파가 적어 둘 일도 없으니 덮어써도 어긋날
    것이 없다. 3절의 검사는 **그래프에 올라 있는 텐서에만** 걸린다.

    모델 초기화 코드가 `nn.init.normal_(m.weight)`처럼 거의 다 밑줄 꼴인 것도
    이것이다 — 초기화는 그래프를 만들기 전에 하는 일이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저가 `no_grad` 안에서 제자리 연산을 쓴다고 했다. 손으로 그렇게 짠
SGD 세 걸음이 `torch.optim.SGD` 세 걸음과 같은 값을 주는지 확인하라.

</div>

??? success "풀이"
    ```python
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        (w ** 2).sum().backward()
        opt.step()
    print("  torch SGD  세 걸음 뒤 w =", [round(v, 6) for v in w.tolist()])

    w2 = torch.tensor([1.0, 2.0], requires_grad=True)
    for _ in range(3):
        if w2.grad is not None:
            w2.grad.zero_()
        (w2 ** 2).sum().backward()
        with torch.no_grad():
            w2.add_(w2.grad, alpha=-0.1)
    print("  손으로 add_ 세 걸음 뒤 w =", [round(v, 6) for v in w2.tolist()])
    print("  같은가:", torch.allclose(w, w2))
    ```

    ```
      torch SGD  세 걸음 뒤 w = [0.512, 1.024]
      손으로 add_ 세 걸음 뒤 w = [0.512, 1.024]
      같은가: True
    ```

    같다. `opt.step()`이 하는 일이 바로 이 `no_grad` + `add_`다.

    여기서 제자리 연산을 쓰는 것이 선택이 아니라는 점이 중요하다. 파라미터가
    1억 개인 모델에서 `w = w - lr * g`로 적으면 **걸음마다 1억 개짜리 텐서를 새로
    잡는다.** 게다가 새 텐서는 더 이상 옵티마이저가 들고 있는 그 텐서가 아니므로
    다음 걸음에서 엉뚱한 것을 고치게 된다. 그래서 `add_`여야 한다.

    `alpha=-0.1`로 적은 것도 같은 생각이다. `w2.add_(-0.1 * w2.grad)`로 적으면
    곱셈 결과를 담을 텐서가 잠깐 생기는데, `alpha`로 넘기면 그것도 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
4절에서 규칙을 하나 얻었다 — **역전파가 출력을 적어 두는 연산이면 터진다.**
`sigmoid`·`tanh`·`sqrt`·`log`·`sin`·`+1`에 대해 **먼저 짐작하고** 확인하라.
짐작은 각 함수의 도함수를 보고 세운다.

</div>

??? success "풀이"
    도함수를 적어 보면 무엇이 있어야 하는지 바로 나온다.

    | 연산 | 도함수 | 무엇이 있어야 하는가 |
    |---|---|---|
    | $\sigma(x)$ | $\sigma(x)(1-\sigma(x))$ | **출력** |
    | $\tanh x$ | $1 - \tanh^2 x$ | **출력** |
    | $\sqrt{x}$ | $1 / (2\sqrt{x})$ | **출력** |
    | $\log x$ | $1/x$ | 입력 |
    | $\sin x$ | $\cos x$ | 입력 |
    | $x + 1$ | $1$ | 아무것도 |

    앞의 셋은 도함수를 **결과로** 적을 수 있다. 뒤의 셋은 그럴 수 없다.

    ```python
    guesses = (("x.sigmoid()", lambda t: t.sigmoid(), "출력"),
               ("x.tanh()", lambda t: t.tanh(), "출력"),
               ("x.sqrt()", lambda t: t.sqrt(), "출력"),
               ("x.log()", lambda t: t.log(), "입력"),
               ("x.sin()", lambda t: t.sin(), "입력"),
               ("x + 1", lambda t: t + 1, "아무것도"))
    for label, build, guess in guesses:
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = build(x)
        y.add_(1.0)
        try:
            y.sum().backward(); got = "통과"
        except RuntimeError:
            got = "터진다"
        agree = "맞음" if (got == "터진다") == (guess == "출력") else "틀림"
        print(f"  {label:13s} 짐작 '{guess}' -> y 고치면 {got:5s}  {agree}")
    ```

    ```
      x.sigmoid()   짐작 '출력' -> y 고치면 터진다    맞음
      x.tanh()      짐작 '출력' -> y 고치면 터진다    맞음
      x.sqrt()      짐작 '출력' -> y 고치면 터진다    맞음
      x.log()       짐작 '입력' -> y 고치면 통과     맞음
      x.sin()       짐작 '입력' -> y 고치면 통과     맞음
      x + 1         짐작 '아무것도' -> y 고치면 통과     맞음
    ```

    여섯 개 모두 짐작대로다. **도함수를 출력으로 적을 수 있는 연산은 그렇게 적어
    두고, 그래서 출력을 고치면 터진다.**

    이것이 구현 쪽의 선택이기도 하다. $\sigma'$를 $\sigma(x)(1-\sigma(x))$로
    적으면 이미 계산해 둔 출력을 쓰면 되므로 지수를 다시 셈하지 않아도 된다. 즉
    **빠르자고 출력을 들고 있는 것**이고, 그 대가로 출력을 고치지 못하게 된다.

    그러니 이 규칙은 외울 것이 아니라 **도함수를 적어 보면 매번 다시 알아낼 수
    있는 것**이다. 처음 보는 연산을 만나도 짐작할 수 있다.
