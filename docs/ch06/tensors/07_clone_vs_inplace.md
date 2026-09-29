# clone과 제자리 연산

이 스크립트는 clone과 제자리 연산의 차이를 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

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

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

출력에 번지가 하나도 없다는 점을 눈여겨보자. `t.untyped_storage().data_ptr()`를 그대로 찍으면 `4437640000` 같은 수가 나오지만, 그 수는 돌릴 때마다, 기계마다 달라서 이 쪽에 실어 두어도 읽는 이가 다시 얻을 수 없다. 게다가 그 수 자체는 아무것도 가르쳐 주지 않는다 — 가르치는 것은 **두 텐서의 번지가 같은가**이다. 그래서 `shares(a, b)`가 그 견줌을, `offset(t)`이 저장소 첫머리에서 몇 번째 원소부터 보는지를 돌려주도록 두었다. 둘 다 어느 기계에서 돌리든 같은 답이 나오며, 슬라이스와 뷰는 저장소를 함께 쓰고(`True`) clone은 그렇지 않다(`False`)는 사실은 그대로 읽힌다. 실제로 `view_slice`의 `storage_offset`이 `1`인 것은, 같은 저장소를 두 번째 원소부터 본다는 뜻이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`retain_graph=True` 없이 같은 계산 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

</div>

??? success "연습문제 2 풀이"
    기본적으로 PyTorch는 메모리를 아끼기 위해 `.backward()` 후에 계산 그래프를 해제한다. `.backward()`를 두 번째로 호출하면 더 이상 존재하지 않는 그래프를 훑으려 하므로 `RuntimeError`가 발생한다. `retain_graph=True`로 두면 그래프가 메모리에 남아 재사용할 수 있지만, 모든 중간 텐서가 할당된 채로 남으므로 메모리 소비가 늘어난다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고 손실을 계산한 뒤, 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하는 코드를 작성하라. 관찰된 값을 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    import torch

    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f'After backward {i+1}: w.grad = {w.grad}')
    # 출력: 4.0, 8.0, 12.0
    # 경사가 누적된다. 매 backward가 기존 경사에 2*w = 4.0을 더한다.
    ```

## 정리하며

**다룬 것** — clone과 제자리 연산

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다.

앞의 연습문제 3개로 직접 확인할 수 있다.
