# 별명, 뷰, clone의 구별

텐서 둘이 같은 숫자를 들여다보고 있는지는 겉으로 보이지 않는다. `b = a`와 `b = a[1:4]`와 `b = a.clone()`은 모두 비슷하게 생겼지만, 앞의 둘은 저장소를 함께 쓰고 마지막만 끊는다. 함께 쓰는 사이에서 제자리 연산을 하면 내가 보고 있지 않은 텐서까지 바뀌므로, 어느 쪽인지 아는 것이 이 쪽의 일이다.

## 1. 코드

```python
"""베끼기와 제자리 셈 견주기."""
import torch

# ========================================================================
# 메인
# ========================================================================


def header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def shares(a: torch.Tensor, b: torch.Tensor) -> bool:
    """두 텐서가 같은 밑 저장소를 나눠 쓰는지 돌려준다.

    • `t.untyped_storage().data_ptr()`는 저장소 버퍼의 첫머리 번지이다. 그 번지를
      그대로 찍으면 `4437640000` 같은 수가 나오는데, 이 수는 돌릴 때마다, 기계마다
      달라진다. 쪽에 실어 두어도 읽는 이가 다시 얻을 수 없는 수이므로 싣지 않는다.
    • 여기서 가르치려는 것은 번지 값이 아니라 **두 텐서가 같은 저장소를 쓰는가**이다.
      그 견줌은 어디서 돌리든 늘 같은 답이 나온다. 그래서 번지 대신 견줌을 찍는다.
    • 두 텐서가 *같은* 저장소를 나눠 쓰면 모양이나 걸음이 달라도 이 값이 True이다.
    """
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def offset(t: torch.Tensor) -> int:
    """저장소 첫머리에서 이 텐서의 첫 원소까지의 거리(원소 수).

    같은 저장소를 쓰는 보기들끼리 어디서부터 보는지를 가른다. 번지와 달리
    돌릴 때마다 같은 수가 나온다.
    """
    return t.storage_offset()


def main():
    torch.manual_seed(0)

    # ----------------------------------------------------------------------------
    # 0) 기준 텐서 준비
    # ----------------------------------------------------------------------------
    # requires_grad_(True)는 플래그를 제자리에서 켜서 autograd가 `base`의 연산을 추적하게 한다.
    base = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3).requires_grad_(True)
    header("Base tensor")
    print("base:\n", base)
    print("base.requires_grad:", base.requires_grad)
    print("base storage (elements):", base.untyped_storage().nbytes() // base.element_size())
    print("base storage_offset:", offset(base))

    # ----------------------------------------------------------------------------
    # 1) 평범한 파이썬 대입: 복사 없음(또 하나의 참조일 뿐)
    # ----------------------------------------------------------------------------
    header("1) Plain assignment: alias reference (NO COPY)")
    # `alias`와 `base`는 완전히 같은 파이썬 객체이다 → 같은 저장소, 같은 경사 플래그.
    alias = base
    print("alias is base?       ", alias is base)     # True (same object identity)
    print("alias shares storage with base?", shares(alias, base))

    # 한쪽 이름으로 제자리 변경을 하면 다른 쪽에서도 보인다(같은 객체이다).
    # base가 requires_grad=True인 잎이라 제자리 변경은 no_grad 안에서만
    # 할 수 있다. 이 페이지의 제자리 연산이 모두 no_grad로 감싸인 까닭이다
    with torch.no_grad():
        base.add_(100)
    print("\nAfter base.add_(100):")
    print("base:\n", base)
    print("alias (same object):\n", alias)

    # 다음 시연을 위해 되돌린다(제자리 뺄셈).
    with torch.no_grad():
        base.sub_(100)

    # ----------------------------------------------------------------------------
    # 2) 뷰(저장소 공유): 슬라이싱 / view / reshape
    # ----------------------------------------------------------------------------
    header("2) Views that SHARE storage (slicing / view / reshape)")
    # 많은 모양/스트라이드 변환은 같은 저장소를 가리키는 *뷰* 를 반환한다.
    view_slice = base[:, 1:]          # slice → shares when possible
    view_view  = base.view(2, 3)      # view with same shape → shares
    view_resh  = base.reshape(2, 3)   # may return a view; may allocate if needed

    # 번지 대신 "같은 저장소인가"와 "저장소 어디서부터 보는가"를 찍는다.
    print("view_slice shares storage with base?", shares(view_slice, base),
          " storage_offset:", offset(view_slice))
    print("view_view  shares storage with base?", shares(view_view, base),
          " storage_offset:", offset(view_view))
    print("view_resh  shares storage with base?", shares(view_resh, base),
          " storage_offset:", offset(view_resh))
    print("All share storage with base? ->",
          shares(view_slice, base) and shares(view_view, base))

    # 뷰를 통한 제자리 변경이 원본을 갱신한다(저장소를 공유한다).
    with torch.no_grad():
        view_slice.mul_(10)
    print("\nAfter view_slice.mul_(10):")
    print("base:\n", base)
    print("view_slice:\n", view_slice)

    # 다음 부분을 위해 되돌린다.
    with torch.no_grad():
        view_slice.div_(10)

    # ----------------------------------------------------------------------------
    # 3) clone(): 바탕 데이터의 깊은 복사(저장소 공유 없음)
    # ----------------------------------------------------------------------------
    header("3) .clone(): DEEP COPY (no storage sharing)")
    # clone()은 자체 저장소를 가진 새 텐서를 만든다. autograd 그래프는 보존된다.
    c = base.clone()
    # 값은 같지만 저장소는 다르다 — 이것이 깊은 복사다.
    print("Same values as base? ->", torch.equal(c.detach(), base.detach()))
    print("Shares storage? ->", shares(c, base))

    # 원본에 대한 제자리 변경은 복제본에 영향을 주지 않는다(버퍼가 독립적이다).
    with torch.no_grad():
        base.add_(1000)
    print("\nAfter base.add_(1000):")
    print("base:\n", base)
    print("clone (unchanged):\n", c)

    # 되돌리기
    with torch.no_grad():
        base.sub_(1000)

    # ----------------------------------------------------------------------------
    # 4) detach(): 저장소는 공유하지만 경사 추적을 끊는다
    # ----------------------------------------------------------------------------
    header("4) .detach(): shares storage, stops grad")
    # detach()는 같은 저장소를 가리키되 requires_grad=False인 텐서를 반환하며
    # 원래 그래프와의 grad_fn 관계도 없다.
    d = base.detach()
    print("d.requires_grad:", d.requires_grad)
    print("detach shares storage with base?", shares(d, base))

    # 원본에 대한 제자리 변경이 d에서도 보인다(저장소를 공유한다).
    with torch.no_grad():
        base.add_(5)
    print("\nAfter base.add_(5):")
    print("base:\n", base)
    print("detach (reflects change):\n", d)

    # 되돌리기
    with torch.no_grad():
        base.sub_(5)

    # ----------------------------------------------------------------------------
    # 5) detach().clone(): 경사를 끊고 저장소 공유도 없다
    # ----------------------------------------------------------------------------
    header("5) .detach().clone(): no grad + deep copy")
    # autograd와 연결되지 않고 메모리도 독립적인 "안전한 스냅숏" 패턴.
    dc = base.detach().clone()
    print("dc.requires_grad:", dc.requires_grad)
    print("detach().clone shares storage with base?", shares(dc, base))

    # 원본에 대한 제자리 변경은 dc에 영향을 주지 않는다(독립적이다).
    with torch.no_grad():
        base.mul_(2)
    print("\nAfter base.mul_(2):")
    print("base:\n", base)
    print("detach().clone (unchanged):\n", dc)

    # 되돌리기(2로 나누기)
    with torch.no_grad():
        base.div_(2)

    # ----------------------------------------------------------------------------
    # 6) Autograd 참고: clone은 경사 흐름을 유지하고 detach는 그렇지 않다
    # ----------------------------------------------------------------------------
    header("6) Autograd note: clone vs detach")
    x = torch.ones(3, requires_grad=True)

    # clone(): 계산 그래프를 보존한다. 경사가 `x`로 되돌아 흐를 수 있다.
    y_clone = x.clone() * 3.0  # grad_fn=MulBackward; clone keeps graph connectivity

    # detach(): 그래프를 끊는다. 이후 연산은 x.grad에 기여하지 않는다.
    y_detach = x.detach() * 3.0  # computed from a leaf with requires_grad=False

    y_clone.sum().backward()   # d/dx of (sum(3*x)) = 3
    print("x.grad from clone-path:", x.grad)  # tensor([3., 3., 3.])

    x.grad.zero_()
    try:
        y_detach.sum().backward()
    except RuntimeError as e:
        # x를 포함하지 않는 그래프로 역전파하면 → x에 대한 경사가 없다.
        print("backward on detach path raised:", e)

    # ----------------------------------------------------------------------------
    # 7) 제자리 연산과 공유 저장소: 주의할 것
    # ----------------------------------------------------------------------------
    header("7) In-place ops can silently affect ALL tensors sharing the storage")
    # 텐서에 대한 제자리 연산은 저장소를 공유하는 모든 뷰/별칭에 영향을 준다.
    a = torch.tensor([1., 2., 3.], requires_grad=True)
    v = a[1:]      # view: shares storage (elements a[1], a[2])
    c = a.clone()  # independent copy

    print("Before in-place on view:")
    print("a:", a)
    print("v:", v, " shares storage with a:", shares(v, a),
          " storage_offset:", offset(v))
    print("c:", c, " shares storage with a:", shares(c, a),
          " storage_offset:", offset(c))

    with torch.no_grad():
        v.add_(100)  # in-place on the view → updates shared positions in `a`
    print("\nAfter v.add_(100):")
    print("a (affected):", a)  # a[1], a[2] changed
    print("v (view):    ", v)
    print("c (clone):   ", c)  # unchanged (separate storage)

    # ----------------------------------------------------------------------------
    # 8) 간단 요약
    # ----------------------------------------------------------------------------
    header("8) Summary")
    print(
        "• alias = base           : NO COPY, same Python object & storage\n"
        "• view/slice/reshape     : SHARE storage (when possible)\n"
        "• clone()                : COPY, independent storage; keeps autograd link\n"
        "• detach()               : SHARE storage; breaks autograd link\n"
        "• detach().clone()       : COPY + no grad (safe snapshot)\n"
        "• In-place ops affect ALL tensors sharing the storage; use with care.\n"
    )


if __name__ == "__main__":
    main()
```

??? note "전체 출력 (112줄)"

    ```

    ================================================================================
    Base tensor
    ================================================================================
    base:
     tensor([[1., 2., 3.],
            [4., 5., 6.]], requires_grad=True)
    base.requires_grad: True
    base storage (elements): 6
    base storage_offset: 0

    ================================================================================
    1) Plain assignment: alias reference (NO COPY)
    ================================================================================
    alias is base?        True
    alias shares storage with base? True

    After base.add_(100):
    base:
     tensor([[101., 102., 103.],
            [104., 105., 106.]], requires_grad=True)
    alias (same object):
     tensor([[101., 102., 103.],
            [104., 105., 106.]], requires_grad=True)

    ================================================================================
    2) Views that SHARE storage (slicing / view / reshape)
    ================================================================================
    view_slice shares storage with base? True  storage_offset: 1
    view_view  shares storage with base? True  storage_offset: 0
    view_resh  shares storage with base? True  storage_offset: 0
    All share storage with base? -> True

    After view_slice.mul_(10):
    base:
     tensor([[ 1., 20., 30.],
            [ 4., 50., 60.]], requires_grad=True)
    view_slice:
     tensor([[20., 30.],
            [50., 60.]], grad_fn=<AsStridedBackward0>)

    ================================================================================
    3) .clone(): DEEP COPY (no storage sharing)
    ================================================================================
    Same values as base? -> True
    Shares storage? -> False

    After base.add_(1000):
    base:
     tensor([[1001., 1002., 1003.],
            [1004., 1005., 1006.]], requires_grad=True)
    clone (unchanged):
     tensor([[1., 2., 3.],
            [4., 5., 6.]], grad_fn=<CloneBackward0>)

    ================================================================================
    4) .detach(): shares storage, stops grad
    ================================================================================
    d.requires_grad: False
    detach shares storage with base? True

    After base.add_(5):
    base:
     tensor([[ 6.,  7.,  8.],
            [ 9., 10., 11.]], requires_grad=True)
    detach (reflects change):
     tensor([[ 6.,  7.,  8.],
            [ 9., 10., 11.]])

    ================================================================================
    5) .detach().clone(): no grad + deep copy
    ================================================================================
    dc.requires_grad: False
    detach().clone shares storage with base? False

    After base.mul_(2):
    base:
     tensor([[ 2.,  4.,  6.],
            [ 8., 10., 12.]], requires_grad=True)
    detach().clone (unchanged):
     tensor([[1., 2., 3.],
            [4., 5., 6.]])

    ================================================================================
    6) Autograd note: clone vs detach
    ================================================================================
    x.grad from clone-path: tensor([3., 3., 3.])
    backward on detach path raised: element 0 of tensors does not require grad and does not have a grad_fn

    ================================================================================
    7) In-place ops can silently affect ALL tensors sharing the storage
    ================================================================================
    Before in-place on view:
    a: tensor([1., 2., 3.], requires_grad=True)
    v: tensor([2., 3.], grad_fn=<SliceBackward0>)  shares storage with a: True  storage_offset: 1
    c: tensor([1., 2., 3.], grad_fn=<CloneBackward0>)  shares storage with a: False  storage_offset: 0

    After v.add_(100):
    a (affected): tensor([  1., 102., 103.], requires_grad=True)
    v (view):     tensor([102., 103.], grad_fn=<AsStridedBackward0>)
    c (clone):    tensor([1., 2., 3.], grad_fn=<CloneBackward0>)

    ================================================================================
    8) Summary
    ================================================================================
    • alias = base           : NO COPY, same Python object & storage
    • view/slice/reshape     : SHARE storage (when possible)
    • clone()                : COPY, independent storage; keeps autograd link
    • detach()               : SHARE storage; breaks autograd link
    • detach().clone()       : COPY + no grad (safe snapshot)
    • In-place ops affect ALL tensors sharing the storage; use with care.

    ```


## 2. 논의

네 가지가 저마다 다른 일을 한다. 가르는 축이 둘인데 — **저장소를 함께 쓰는가**와 **경사를 좇는가** — 둘이 따로 움직인다.

| | 저장소 | 경사 | 쓰는 자리 |
|---|---|---|---|
| `b = a` | 함께 쓴다 | 그대로 좇는다 | 이름만 하나 더 붙인 것. 텐서가 생기지 않는다 |
| `a[1:4]`, `a.view(…)` | 함께 쓴다 | 그대로 좇는다 | 같은 숫자를 다른 모양으로 본다 |
| `a.clone()` | **끊는다** | 그대로 좇는다 | 값만 따로 두고 학습은 이어 가고 싶을 때 |
| `a.detach()` | 함께 쓴다 | **끊는다** | 값을 기록하거나 NumPy로 넘길 때 |

`clone`과 `detach`가 서로 반대라는 것이 요점이다. `clone`은 메모리를 끊고 그래프는 이어 두므로 경사가 `clone`을 지나 원래 텐서까지 흐른다. `detach`는 그래프를 끊고 메모리는 함께 쓰므로, 값이 바뀌면 양쪽이 함께 바뀌는데 경사는 흐르지 않는다. 둘 다 끊으려면 `a.detach().clone()`처럼 둘을 겹쳐 쓴다.

**제자리 연산은 이 그림 위에서 위험해진다.** 밑줄로 끝나는 이름(`add_`, `mul_`, `clamp_`)은 새 텐서를 만들지 않고 저장소를 직접 고친다. 그래서 그 저장소를 함께 쓰는 **모든** 텐서가 함께 바뀐다. 위 7번이 그것을 보인다.

경사까지 얽히면 결과가 셋으로 갈린다.

- 역전파에 출력값이 필요 없는 연산(`a * 2`)의 출력을 고치면 **오류가 나지 않는다.** 대신 미분하는 함수가 조용히 달라진다.
- 출력값을 저장해 두는 연산(`a.exp()`)의 출력을 고치면 `RuntimeError`다. 그 값 없이는 경사를 셀 수 없기 때문이다.
- 잎 텐서를 직접 고치면 `RuntimeError`다. 가중치를 갱신할 때 `torch.no_grad()`가 필요한 까닭이다.

막아 주는 쪽이 오히려 다행이다. 넘어가는 첫째 경우에는 경사가 달라졌다는 표시가 아무 데도 남지 않는다.

출력에 번지가 하나도 없다는 점을 눈여겨보자. `t.untyped_storage().data_ptr()`를 그대로 찍으면 `4437640000` 같은 수가 나오지만, 그 수는 돌릴 때마다, 기계마다 달라서 이 쪽에 실어 두어도 읽는 이가 다시 얻을 수 없다. 게다가 그 수 자체는 아무것도 가르쳐 주지 않는다 — 가르치는 것은 **두 텐서의 번지가 같은가**이다. 그래서 `shares(a, b)`가 그 견줌을, `offset(t)`이 저장소 첫머리에서 몇 번째 원소부터 보는지를 돌려주도록 두었다. 둘 다 어느 기계에서 돌리든 같은 답이 나오며, 슬라이스와 뷰는 저장소를 함께 쓰고(`True`) clone은 그렇지 않다(`False`)는 사실은 그대로 읽힌다. 실제로 `view_slice`의 `storage_offset`이 `1`인 것은, 같은 저장소를 두 번째 원소부터 본다는 뜻이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`base = torch.arange(6.)`에 대해 아래 넷을 만들고, 각각이 `base`와 저장소를 함께 쓰는지 위 코드의 `shares()`로 판정하라.

```python
alias = base          # 이름만 하나 더 붙인다
view  = base[1:4]     # 슬라이스
flat  = base.view(6)  # 같은 모양의 뷰
copy  = base.clone()
```

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    def shares(a, b):
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

    base = torch.arange(6.)
    for name, t in [("alias", base), ("view", base[1:4]),
                    ("flat", base.view(6)), ("copy", base.clone())]:
        print(f"{name:6} shares={shares(base, t)}  offset={t.storage_offset()}")
    ```

    ```
    alias  shares=True  offset=0
    view   shares=True  offset=1
    flat   shares=True  offset=0
    copy   shares=False  offset=0
    ```

    `clone`만 끊는다. 나머지 셋은 모두 같은 숫자를 들여다본다.

    `alias`는 텐서를 만들지도 않았다 — 파이썬 이름 하나가 같은 객체를 가리킬 뿐이다.
    `view`는 새 텐서지만 저장소가 같고, `offset=1`이 "둘째 원소부터 본다"는 뜻이다.
    모양이 달라도 저장소는 하나라는 것이 요점이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`base = torch.arange(6.)`에서 `v = base[1:4]`를 잘라 낸 뒤 `v.add_(100)`을 하라. `base`를 찍어 보고 무슨 일이 일어났는지 설명하라. `v`만 바꾸고 `base`는 지키려면 어떻게 적어야 하는가?

</div>

??? success "연습문제 2 풀이"
    ```python
    import torch

    base = torch.arange(6.)
    v = base[1:4]
    v.add_(100)            # 밑줄은 "제자리에서"라는 뜻이다
    print("base:", base)
    ```

    ```
    base: tensor([  0., 101., 102., 103.,   4.,   5.])
    ```

    `base`를 건드리지 않았는데 가운데 셋이 바뀌었다. `v`는 베낀 것이 아니라 `base`의
    저장소를 함께 보는 뷰이므로, 제자리 연산이 그 저장소를 고치면 `base`가 함께 바뀐다.

    고치는 길은 둘 중 하나다. 제자리 연산을 쓰지 않거나, 미리 끊어 두는 것이다.

    ```python
    base = torch.arange(6.)
    v = base[1:4] + 100            # 제자리가 아니다 — 새 텐서가 나온다
    print(base)

    base = torch.arange(6.)
    v = base[1:4].clone()          # 미리 끊는다
    v.add_(100)
    print(base)
    ```

    ```
    tensor([0., 1., 2., 3., 4., 5.])
    tensor([0., 1., 2., 3., 4., 5.])
    ```

    밑줄로 끝나는 이름(`add_`, `mul_`, `clamp_`)은 모두 제자리 연산이다. 저장소를 함께
    쓰는 텐서가 하나라도 있으면, 제자리 연산은 **내가 보고 있지 않은 텐서까지** 바꾼다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 세 가지를 모두 돌려 보라. 하나는 오류 없이 넘어가고 둘은 `RuntimeError`를 낸다.

```python
# (가) 곱셈의 출력을 고친다
a = torch.tensor([1., 2., 3.], requires_grad=True); b = a * 2
b[0] = 99.; b.sum().backward()

# (나) exp의 출력을 고친다
a = torch.tensor([1., 2., 3.], requires_grad=True); b = a.exp()
b[0] = 99.; b.sum().backward()

# (다) 잎 텐서를 고친다
w = torch.tensor([1., 2.], requires_grad=True); w[0] = 5.
```

(가)에서 `a.grad`가 무엇이 되는지 확인하고, **오류가 나지 않는 쪽이 왜 더 조심할 자리인지** 설명하라. 그리고 (가)와 (나)가 갈리는 까닭을 밝혀라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch

    a = torch.tensor([1., 2., 3.], requires_grad=True)
    b = a * 2
    b[0] = 99.
    b.sum().backward()
    print("(가) a.grad:", a.grad)

    a = torch.tensor([1., 2., 3.], requires_grad=True)
    b = a.exp()
    try:
        b[0] = 99.
        b.sum().backward()
    except RuntimeError as e:
        print("(나)", str(e)[:96])

    w = torch.tensor([1., 2.], requires_grad=True)
    try:
        w[0] = 5.
    except RuntimeError as e:
        print("(다)", str(e)[:96])
    ```

    ```
    (가) a.grad: tensor([0., 2., 2.])
    (나) one of the variables needed for gradient computation has been modified by an inpl
    (다) a view of a leaf Variable that requires grad is being used in an in-place operatio
    ```

    **(가)와 (나)가 갈리는 까닭은 역전파에 무엇이 필요한가다.** $y = 2x$의 도함수는
    2라는 상수이므로, 역전파에 출력값이 필요하지 않다. 그래서 출력을 고쳐도 PyTorch가
    막을 이유가 없다. 반면 $y = e^x$의 도함수는 $e^x$, 곧 **출력 그 자체**다. 그래서
    `exp`는 출력을 저장해 두고 역전파에서 다시 꺼내 쓰는데, 그 값이 99로 바뀌어 있으면
    경사를 계산할 수가 없다. 그러므로 막는다.

    **조심할 자리는 (가)다.** `a.grad`가 `[0., 2., 2.]`인데, 첫 원소가 2가 아니라 0이다.
    이것은 PyTorch가 틀린 것이 아니다. `b[0] = 99`로 덮어쓴 뒤의 합은 `a[0]`에 더 이상
    의존하지 않으므로 0이 **맞다.** 틀린 것은 내가 미분하려던 함수가 바뀌었다는 사실을
    모르고 있다는 점이다.

    곧 제자리 연산은 계산 그래프를 **조용히 다른 것으로 바꾼다.** 오류가 나는 경우는
    PyTorch가 알려 주니 오히려 다행이고, 넘어가는 경우에는 경사가 달라졌다는 표시가
    어디에도 남지 않는다.

    (다)는 가장 흔한 실수다. 모델의 가중치는 잎 텐서이므로 `w[0] = 5.`처럼 직접 고칠
    수 없다. 갱신은 `with torch.no_grad():` 안에서 하거나 최적화기에 맡긴다.

## 정리하며

두 텐서가 같은 숫자를 들여다보고 있는지는 겉으로 보이지 않는다. 가르는 축이 둘이고, 둘이 따로 움직인다.

- **저장소를 끊는 것은 `clone`뿐이다.** `b = a`는 이름만 하나 더 붙인 것이고, 슬라이스와 `view`는 모양만 다른 같은 저장소다.
- **그래프를 끊는 것은 `detach`다.** 저장소는 여전히 함께 쓴다. 둘 다 끊으려면 `a.detach().clone()`.

`clone`과 `detach`는 서로 반대다. 하나는 메모리를 끊고 경사를 잇고, 하나는 경사를 끊고 메모리를 잇는다.

제자리 연산(밑줄로 끝나는 이름)은 저장소를 직접 고치므로, 그 저장소를 함께 쓰는 텐서가 전부 바뀐다. 경사가 얽히면 셋으로 갈리는데 — 조용히 넘어가거나, 저장된 출력이 망가졌다고 막거나, 잎 텐서라서 막는다 — **조용히 넘어가는 경우가 가장 위험하다.** 미분하는 함수가 달라졌는데 표시가 남지 않는다.

그래서 어림 규칙은 하나다. **제자리 연산을 쓰기 전에 이 저장소를 누가 함께 보고 있는지 센다.** 셀 수 없으면 쓰지 않는다.
