# `detach()`와 `detach_()` — 새 손잡이냐 제자리냐

둘 다 미분 추적을 끊는다. 다른 것은 **무엇에 대고 하느냐**다.

| | 하는 일 | 원본 |
|---|---|---|
| `t.detach()` | 추적 없는 **새 텐서**를 내준다 | 그대로 |
| `t.detach_()` | **`t` 자신**을 끊는다 | **바뀐다** |

이름이 한 글자 차이이고 둘 다 같은 것을 돌려주는 것처럼 보이므로, 바꿔 써도
될 것 같다. 그런데 `detach_()`에는 함정이 하나 있다 — **이미 역전파가 저장해 둔
텐서를 제자리에서 끊으면 뒤에 `backward()`가 터진다.**

## 1. 코드

```python
"""detach() 와 detach_() — 새 손잡이냐 제자리냐.

둘 다 미분 추적을 끊는다. 다른 것은 **무엇에 대고 하느냐**다.

- `t.detach()`  — 추적 없는 **새 텐서**를 내준다. `t` 는 그대로
- `t.detach_()` — **`t` 자신**을 끊는다. 되돌릴 수 없다

그리고 `detach_()` 에는 함정이 하나 있다. 이미 역전파가 저장해 둔
텐서를 제자리에서 끊어 버리면, 뒤에 `backward()` 가 터진다.

실행: python 09_detach_vs_detach_.py
"""
import torch


def header(title):
    print(f"\n=== {title} ===")


def describe(name, t):
    fn = type(t.grad_fn).__name__ if t.grad_fn is not None else None
    print(f"  {name:22s} requires_grad={t.requires_grad!s:5s} grad_fn={fn}")


# ========================================================================
# 1. 새 손잡이냐 제자리냐
# ========================================================================

def new_or_in_place():
    header("1. 무엇이 바뀌는가")
    x = torch.randn(3, requires_grad=True)

    y = x * 2
    d = y.detach()
    print("  y.detach()")
    describe("    돌려받은 d", d)
    describe("    원본 y", y)
    print("    같은 객체인가:", d is y,
          "  저장소 공유:", d.data_ptr() == y.data_ptr())

    y2 = x * 2
    r = y2.detach_()
    print("  y2.detach_()")
    describe("    돌려받은 r", r)
    describe("    원본 y2", y2)
    print("    같은 객체인가:", r is y2)
    print("  detach_ 는 y2 자신을 바꾸었다. 되돌릴 길이 없다")


# ========================================================================
# 2. 잎이 아닌 것에도 걸린다
# ========================================================================

def works_on_non_leaf():
    header("2. requires_grad_ 와 달리 잎이 아니어도 된다")
    x = torch.randn(3, requires_grad=True)
    h = x * 2
    try:
        h.requires_grad_(False)
    except RuntimeError as e:
        print("  h.requires_grad_(False):", str(e).splitlines()[0][:58])
    h.detach_()
    print("  h.detach_()            : 된다.", h.requires_grad)
    print("  오류 메시지가 가리키는 길이 바로 이것이다")


# ========================================================================
# 3. detach_ 의 함정 — 저장된 텐서를 끊으면
# ========================================================================

def the_trap():
    header("3. 저장된 텐서를 제자리에서 끊으면")
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    loss = (y ** 2).sum()        # 역전파가 y 를 저장해 둔다
    y.detach_()                  # 그 y 를 제자리에서 끊는다
    try:
        loss.backward()
    except RuntimeError as e:
        print("  backward:", str(e).splitlines()[0][:72])
    print("  detach() 였다면 새 텐서를 만들 뿐이라 아무 일도 없다")

    x2 = torch.randn(3, requires_grad=True)
    y2 = x2 * 2
    loss2 = (y2 ** 2).sum()
    _ = y2.detach()              # 새 손잡이만 하나 더 만든다
    loss2.backward()
    print("  detach() 뒤 backward:", torch.allclose(x2.grad, 8 * x2.detach()),
          " (dloss/dx = 8x)")


# ========================================================================
# 4. 어느 쪽을 쓰는가
# ========================================================================

def which_one():
    header("4. 쓰는 자리")
    x = torch.randn(4, requires_grad=True)

    # (가) 값만 꺼내 기록한다 — detach()
    loss = (x ** 2).sum()
    history = [loss.detach()]
    print("  (가) 기록용으로 꺼내기 -> detach()")
    print("       원본 그래프를 건드리지 않는다:",
          type(loss.grad_fn).__name__)

    # (나) 순환 신경망에서 숨은 상태를 끊는다 — 둘 다 쓰인다
    h = torch.zeros(2, requires_grad=True)
    for _ in range(3):
        h = (h + x[:2]).tanh()
    print("  (나) RNN 숨은 상태: 바퀴마다 h = h.detach() 로 끊는다")
    print("       안 끊으면 그래프가 바퀴 수만큼 길어진다")
    print("       h 를 다시 대입하므로 detach() 로 충분하다")
    print("  detach_() 가 꼭 필요한 자리는 드물다")
    print("  같은 이름을 계속 쓰면서 그래프만 끊고 싶을 때뿐이다")


def main():
    new_or_in_place()
    works_on_non_leaf()
    the_trap()
    which_one()


if __name__ == "__main__":
    main()
```

**출력:**

```
=== 1. 무엇이 바뀌는가 ===
  y.detach()
      돌려받은 d             requires_grad=False grad_fn=None
      원본 y               requires_grad=True  grad_fn=MulBackward0
    같은 객체인가: False   저장소 공유: True
  y2.detach_()
      돌려받은 r             requires_grad=False grad_fn=None
      원본 y2              requires_grad=False grad_fn=None
    같은 객체인가: True
  detach_ 는 y2 자신을 바꾸었다. 되돌릴 길이 없다

=== 2. requires_grad_ 와 달리 잎이 아니어도 된다 ===
  h.requires_grad_(False): you can only change requires_grad flags of leaf variables.
  h.detach_()            : 된다. False
  오류 메시지가 가리키는 길이 바로 이것이다

=== 3. 저장된 텐서를 제자리에서 끊으면 ===
  backward: Trying to use a saved tensor that has been detached in-place, i.e. with 
  detach() 였다면 새 텐서를 만들 뿐이라 아무 일도 없다
  detach() 뒤 backward: True  (dloss/dx = 8x)

=== 4. 쓰는 자리 ===
  (가) 기록용으로 꺼내기 -> detach()
       원본 그래프를 건드리지 않는다: SumBackward0
  (나) RNN 숨은 상태: 바퀴마다 h = h.detach() 로 끊는다
       안 끊으면 그래프가 바퀴 수만큼 길어진다
       h 를 다시 대입하므로 detach() 로 충분하다
  detach_() 가 꼭 필요한 자리는 드물다
  같은 이름을 계속 쓰면서 그래프만 끊고 싶을 때뿐이다
```

## 2. 무엇이 바뀌는가

```
y.detach()
    돌려받은 d : requires_grad=False  grad_fn=None
    원본 y     : requires_grad=True   grad_fn=MulBackward0
  같은 객체인가: False   저장소 공유: True

y2.detach_()
    돌려받은 r : requires_grad=False  grad_fn=None
    원본 y2    : requires_grad=False  grad_fn=None
  같은 객체인가: True
```

`detach()`는 **원본을 건드리지 않는다.** `y`는 여전히 `MulBackward0`을 달고 있고
그래프 위에 있다. 새로 받은 `d`만 그래프 바깥에 있다 — 다만 **저장소는
공유한다.**

`detach_()`는 `y2` 자신의 `grad_fn`을 떼어 낸다. 돌려받은 것과 원본이 같은
객체이고, **되돌릴 길이 없다.** `requires_grad_(True)`로 다시 켜도 그래프와의
연결은 돌아오지 않는다.

## 3. `requires_grad_`와 달리 잎이 아니어도 된다

```
h.requires_grad_(False): you can only change requires_grad flags of leaf variables.
h.detach_()            : 된다. False
```

「경사 추적 끄기」에서 본 오류다. `requires_grad_`는 잎에만 걸리는데, 그 오류
메시지가 가리키는 대안이 바로 `detach()`다.

그래서 둘의 쓰임이 갈린다. **파라미터를 얼리는 것은 `requires_grad_`**이고,
**중간 텐서에서 그래프를 끊는 것은 `detach`**다. 대상이 다르다.

## 4. `detach_()`의 함정

```python
y = x * 2
loss = (y ** 2).sum()    # 역전파가 y 를 저장해 둔다
y.detach_()              # 그 y 를 제자리에서 끊는다
loss.backward()
```

```
Trying to use a saved tensor that has been detached in-place,
i.e. with .detach_()
```

파이토치가 정확히 이 경우를 위한 메시지를 따로 두었다. 그만큼 흔한 실수라는
뜻이다.

까닭은 「제자리 연산」에서 본 것과 같은 구조다. $\frac{d}{dy}y^2 = 2y$를
계산하려면 `y`가 있어야 하는데, 그 `y`를 그래프에서 떼어 버렸으니 엔진이 쓸 수
없다.

`detach()`였다면 아무 일도 없다 — 새 손잡이를 하나 더 만들 뿐 `y`는 그대로다.

```
detach() 뒤 backward: True   (dloss/dx = 8x)
```

> **헷갈릴 때는 `detach()`를 쓴다.** 원본을 안 건드리므로 뒤에 무엇이 와도
> 안전하고, 쓰이는 자리의 거의 전부가 "값만 꺼내 간다"이므로 그것으로 충분하다.

## 5. 쓰는 자리

**(가) 기록용으로 값만 꺼낼 때** — `loss.detach()`. 원본 `loss`는 그래프를
그대로 달고 있으므로 뒤에 `backward()`를 부를 수 있다. 「경사 추적을 멈추는
시점」의 연습문제에서 이것을 빼먹으면 그래프가 백 개씩 쌓이는 것을 보았다.

**(나) 순환 신경망에서 숨은 상태를 끊을 때** — 바퀴마다 `h = h.detach()`.
안 끊으면 그래프가 시퀀스 길이만큼 길어져, 역전파가 처음까지 거슬러 올라간다
(이것을 잘라 내는 것을 truncated BPTT라고 부른다).

여기서도 `detach()`로 충분하다. `h`라는 이름에 **새 텐서를 다시 대입**하기
때문이다. `detach_()`가 꼭 필요한 자리는 **같은 객체를 계속 써야 하는데 그래프만
끊고 싶을 때**뿐이고, 실제 코드에서는 드물다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`d = y.detach()`로 받은 `d`를 고치면 `y`도 바뀌는가? 그리고 `d.clone()`은?

</div>

??? success "풀이"
    ```python
    torch.manual_seed(1)
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    d = y.detach()
    c = y.detach().clone()
    print("  고치기 전 y :", [round(v, 4) for v in y.detach().tolist()])
    d[0] = 99.0
    print("  d[0]=99 뒤 y:", [round(v, 4) for v in y.detach().tolist()])
    c[1] = -99.0
    print("  c[1]=-99 뒤 y:", [round(v, 4) for v in y.detach().tolist()])
    ```

    ```
      고치기 전 y : [1.3227, 0.5338, 0.1234]
      d[0]=99 뒤 y: [99.0, 0.5338, 0.1234]
      c[1]=-99 뒤 y: [99.0, 0.5338, 0.1234]
    ```

    **`d`를 고치면 `y`가 바뀐다.** `detach()`는 저장소를 공유하므로 같은
    메모리다. `clone()`을 붙인 `c`는 새 자리를 잡았으므로 고쳐도 `y`가 그대로다.

    「Autograd 관련 속성」에서 본 대로, 이렇게 고친 뒤 역전파하면 **경사가 조용히
    틀린다.** 터지지도 경고하지도 않는다.

    그래서 규칙이 짧다. **`detach()`는 읽기 전용, 고칠 작정이면
    `detach().clone()`.**

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
순환 신경망에서 `h = h.detach()`를 빼먹으면 그래프가 얼마나 길어지는가? 바퀴를
돌며 **그래프의 깊이**를 세어라.

</div>

??? success "풀이"
    ```python
    def depth(t):
        d, fn = 0, t.grad_fn
        while fn is not None:
            d += 1
            fn = fn.next_functions[0][0] if fn.next_functions else None
        return d

    x = torch.randn(2, requires_grad=True)
    for label, cut in (("detach 함  ", True), ("detach 안 함", False)):
        h = torch.zeros(2)
        for step in range(1, 6):
            h = (h + x).tanh()
            if cut:
                h = h.detach()
        print(f"  {label}  다섯 바퀴 뒤 그래프 깊이 {depth(h)}")
    ```

    ```
      detach 함    다섯 바퀴 뒤 그래프 깊이 0
      detach 안 함  다섯 바퀴 뒤 그래프 깊이 10
    ```

    끊지 않으면 **바퀴마다 깊이가 2씩 는다**(`add` 하나, `tanh` 하나). 다섯
    바퀴면 10이고, 천 바퀴면 2000이다.

    길어진 그래프는 두 가지 값을 치른다. **메모리** — 바퀴마다 중간값을 들고
    있어야 한다. 그리고 **시간** — 역전파가 처음까지 거슬러 올라간다.

    끊은 쪽의 깊이가 0인 것은 마지막에 `detach()`를 불렀기 때문이다. 실제
    코드에서는 바퀴 안에서 로스를 계산한 다음 끊으므로, 한 바퀴 치 그래프는
    남는다.

    이것이 **잘린 역전파(truncated BPTT)**다. 긴 시퀀스를 통째로 역전파하는 대신
    정해진 길이마다 끊어, 메모리를 상수로 묶는다. 대신 그보다 먼 과거의 의존성은
    배우지 못한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
4절의 함정을 **`detach()`로도** 일으킬 수 있는가? 그리고 고치는 **시점**을
바꾸면 어떻게 되는가?

</div>

??? success "풀이"
    ```python
    for when in ("순전파 전", "순전파 뒤"):
        w = torch.tensor([1.0, 2.0], requires_grad=True)
        y = w * 3
        if when == "순전파 전":
            y.detach()[0] = 99.0
            loss = (y ** 2).sum()
        else:
            loss = (y ** 2).sum()
            y.detach()[0] = 99.0
        try:
            loss.backward()
            print(f"  {when}: 통과. w.grad = "
                  f"{[round(v, 1) for v in w.grad.tolist()]}  참값 [18, 36]")
        except RuntimeError as e:
            print(f"  {when}: {str(e).splitlines()[0][:60]}")
    ```

    ```
      순전파 전: 통과. w.grad = [594.0, 36.0]  참값 [18, 36]
      순전파 뒤: one of the variables needed for gradient computation has bee
    ```

    **같은 한 줄인데 어디에 두느냐로 결과가 갈린다.**

    - **순전파 뒤에 고치면** 버전 카운터가 알아챈다. 역전파가 저장해 둔 값의
      버전이 0인데 지금은 1이므로 어긋난 것을 알고 거절한다
    - **순전파 전에 고치면** 저장되는 값이 **이미 고쳐진 값**이다. 버전이
      어긋나지 않으므로 검사를 통과하고, 99로 계산한 **594**라는 틀린 경사가
      조용히 나온다

    참값은 $\frac{d}{dw}\sum(3w)^2 = 18w = [18, 36]$이다. 594는
    $2 \times 99 \times 3$으로, 그래프가 만든 적 없는 값으로 센 것이다.

    그러니 "검사가 잡아 준다"를 믿을 수 없다. **검사는 저장한 뒤에 바뀐 것만
    잡는다.** 저장되기 전에 망가뜨리면 할 말이 없다.

    4절의 `detach_()`도 시점에 따라 갈린다. `(y ** 2).sum()`처럼 `y`를 저장하는
    식이면 터지지만, `y.sum()`처럼 저장할 것이 없는 식이면 **아무 일도 없이
    통과한다.** 「계산 그래프 재사용」에서 본 것과 같은 갈림이다.

    결론은 짧다. **꺼낸 것을 고칠 작정이면 `detach().clone()`이다.** 어느 쪽이
    잡히고 어느 쪽이 안 잡히는지를 외우는 것보다 싸다.
