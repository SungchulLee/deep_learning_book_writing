# clone과 제자리 연산

이 스크립트는 clone과 제자리 연산의 차이을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

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


def ptr(t: torch.Tensor) -> int:
    """밑 저장소의 데이터 가리개를 (십진수로) 돌려준다.

    • 밑 저장소 버퍼의 첫머리를 가리킨다(텐서가 자리 옮김이나 걸음을 지닌 보기라면
      논리상 첫 원소를 가리키지는 않는다).
    • 두 텐서가 *같은* 저장소를 나눠 쓰면 모양이나 걸음이 달라도
      data_ptr() 값이 같다.
    """
    return t.storage().data_ptr()


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
    print("ptr(base):", ptr(base))

    # ----------------------------------------------------------------------------
    # 1) 평범한 파이썬 대입: 복사 없음(또 하나의 참조일 뿐)
    # ----------------------------------------------------------------------------
    header("1) Plain assignment: alias reference (NO COPY)")
    # `alias`와 `base`는 완전히 같은 파이썬 객체이다 → 같은 저장소, 같은 경사 플래그.
    alias = base
    print("alias is base?       ", alias is base)     # True (same object identity)
    print("ptr(alias) == ptr(base)?", ptr(alias) == ptr(base))

    # 한쪽 이름으로 제자리 변경을 하면 다른 쪽에서도 보인다(같은 객체이다).
    base.add_(100)
    print("\nAfter base.add_(100):")
    print("base:\n", base)
    print("alias (same object):\n", alias)

    # 다음 시연을 위해 되돌린다(제자리 뺄셈).
    base.sub_(100)

    # ----------------------------------------------------------------------------
    # 2) 뷰(저장소 공유): 슬라이싱 / view / reshape
    # ----------------------------------------------------------------------------
    header("2) Views that SHARE storage (slicing / view / reshape)")
    # 많은 모양/스트라이드 변환은 같은 저장소를 가리키는 *뷰* 를 반환한다.
    view_slice = base[:, 1:]          # slice → shares when possible
    view_view  = base.view(2, 3)      # view with same shape → shares
    view_resh  = base.reshape(2, 3)   # may return a view; may allocate if needed

    print("ptr(view_slice):", ptr(view_slice))
    print("ptr(view_view) :", ptr(view_view))
    print("ptr(view_resh) :", ptr(view_resh))
    print("All share storage with base? ->",
          ptr(view_slice) == ptr(base) and ptr(view_view) == ptr(base))

    # 뷰를 통한 제자리 변경이 원본을 갱신한다(저장소를 공유한다).
    view_slice.mul_(10)
    print("\nAfter view_slice.mul_(10):")
    print("base:\n", base)
    print("view_slice:\n", view_slice)

    # 다음 부분을 위해 되돌린다.
    view_slice.div_(10)

    # ----------------------------------------------------------------------------
    # 3) clone(): 바탕 데이터의 깊은 복사(저장소 공유 없음)
    # ----------------------------------------------------------------------------
    header("3) .clone(): DEEP COPY (no storage sharing)")
    # clone()은 자체 저장소를 가진 새 텐서를 만든다. autograd 그래프는 보존된다.
    c = base.clone()
    print("ptr(clone):", ptr(c), "  ptr(base):", ptr(base))
    print("Shares storage? ->", ptr(c) == ptr(base))

    # 원본에 대한 제자리 변경은 복제본에 영향을 주지 않는다(버퍼가 독립적이다).
    base.add_(1000)
    print("\nAfter base.add_(1000):")
    print("base:\n", base)
    print("clone (unchanged):\n", c)

    # 되돌리기
    base.sub_(1000)

    # ----------------------------------------------------------------------------
    # 4) detach(): 저장소는 공유하지만 경사 추적을 끊는다
    # ----------------------------------------------------------------------------
    header("4) .detach(): shares storage, stops grad")
    # detach()는 같은 저장소를 가리키되 requires_grad=False인 텐서를 반환하며
    # 원래 그래프와의 grad_fn 관계도 없다.
    d = base.detach()
    print("d.requires_grad:", d.requires_grad)
    print("ptr(detach) == ptr(base)?", ptr(d) == ptr(base))

    # 원본에 대한 제자리 변경이 d에서도 보인다(저장소를 공유한다).
    base.add_(5)
    print("\nAfter base.add_(5):")
    print("base:\n", base)
    print("detach (reflects change):\n", d)

    # 되돌리기
    base.sub_(5)

    # ----------------------------------------------------------------------------
    # 5) detach().clone(): 경사를 끊고 저장소 공유도 없다
    # ----------------------------------------------------------------------------
    header("5) .detach().clone(): no grad + deep copy")
    # autograd와 연결되지 않고 메모리도 독립적인 "안전한 스냅숏" 패턴.
    dc = base.detach().clone()
    print("dc.requires_grad:", dc.requires_grad)
    print("ptr(detach().clone) == ptr(base)?", ptr(dc) == ptr(base))

    # 원본에 대한 제자리 변경은 dc에 영향을 주지 않는다(독립적이다).
    base.mul_(2)
    print("\nAfter base.mul_(2):")
    print("base:\n", base)
    print("detach().clone (unchanged):\n", dc)

    # 되돌리기(2로 나누기)
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
    print("a:", a, " ptr:", ptr(a))
    print("v:", v, " ptr:", ptr(v))
    print("c:", c, " ptr:", ptr(c))

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

## 2. 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

## 연습문제

**연습문제 1.**
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


**연습문제 2.**
`retain_graph=True` 없이 같은 계산 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    기본적으로 PyTorch는 메모리를 아끼기 위해 `.backward()` 후에 계산 그래프를 해제한다. `.backward()`를 두 번째로 호출하면 더 이상 존재하지 않는 그래프를 훑으려 하므로 `RuntimeError`가 발생한다. `retain_graph=True`로 두면 그래프가 메모리에 남아 재사용할 수 있지만, 모든 중간 텐서가 할당된 채로 남으므로 메모리 소비가 늘어난다.

---


**연습문제 3.**
잎 텐서 `w`를 만들고 손실을 계산한 뒤, 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하는 코드를 작성하라. 관찰된 값을 설명하라.

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
