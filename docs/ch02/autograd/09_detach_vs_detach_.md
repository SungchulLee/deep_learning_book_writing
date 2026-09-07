# detach와 detach_

이 스크립트는 `detach()`와 `detach_()`의 차이을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""detach와 detach 견주기."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():
    # -----------------------------------------------------------------------------
    # Autograd와 분리 — 간단한 어림 규칙
    # -----------------------------------------------------------------------------
    # • 잎 텐서: 사용자가 만든 텐서(autograd 연산의 결과가 아니다).
    #   - grad_fn=None이다
    #   - .requires_grad_(True/False)로 requires_grad를 바꿀 수 있다
    #
    # • 비잎 텐서: 경사가 필요한 텐서에 대한 연산의 결과.
    #   - <AddBackward0>, <SinBackward> 같은 grad_fn을 가진다
    #   - requires_grad_를 바꿀 수 없지만(오류가 난다) detach는 할 수 있다.
    #
    # • detach의 여러 형태:
    #   - t.detach()  → 저장소는 공유하지만 추적되지 않는 새 텐서를 반환한다
    #                   (requires_grad=False, grad_fn=None).
    #   - t.detach_() → 제자리: t 자체를 바꾸어 추적을 멈춘다
    #                   (requires_grad=False, grad_fn=None). 비잎 텐서에도 허용된다.
    #
    # • 주어진 값에서 추적을 "새로 시작"하려면:
    #       t = t.detach().requires_grad_(True)   # 여기서부터 새 잎 텐서
    # -----------------------------------------------------------------------------

    a = torch.linspace(-1, 1, steps=5, requires_grad=True)
    # a는 사용자가 직접 만들었으므로 잎 텐서이다
    # requires_grad=True로 만든 것이다. 잎 텐서는 언제나 grad_fn=None이며,
    # autograd가 역전파 중에 그 경사를 계산하더라도 그렇다.
    print("a.requires_grad:", a.requires_grad, "| a.grad_fn:", a.grad_fn)  # grad_fn=None

    b = a * a + 1.0
    # b는 a로부터 계산되므로 b.requires_grad=True이다.
    # 그러나 b는 잎이 아니다:
    #   - b.grad_fn은 b의 순전파 값을 계산한 함수가 아니다.
    #   - 대신 역전파 함수 객체이다(예: <AddBackward0>).
    #     역전파할 때 ∂b/∂a를 계산할 줄 아는 함수이다.
    print("b.requires_grad:", b.requires_grad, "| b.grad_fn:", b.grad_fn)

    # detach(): b와 저장소를 공유하는 새 텐서를 만든다
    # 계산 그래프에는 연결되어 있지 않다:
    #   - requires_grad=False
    #   - grad_fn=None
    # b를 바꾸지 않으면서 autograd 관점에서 읽기 전용 뷰가 필요할 때 쓴다.
    b_det = b.detach()
    print("\n--- Detach example ---")
    print("b_det.requires_grad (expect False):", b_det.requires_grad)
    print("b_det.grad_fn (expect None):", b_det.grad_fn)
    print("b_det is b? (expect False):", b_det is b)  # new object

    # detach_(): detach()의 제자리 버전
    # 이는 텐서 자체를 바꾸어 autograd가 더 이상 추적하지 않게 한다.
    # 이후 이 텐서는 requires_grad=False, grad_fn=None이 된다.
    # 잎이 아닌 텐서에도 안전하고 유효하다.
    b2 = (a + 3).sin()
    print("\n--- In-place detach_ example ---")
    print("Before detach_(): b2.requires_grad:", b2.requires_grad, "| b2.grad_fn:", b2.grad_fn)
    b2.detach_()  # in-place: removes gradient tracking (now behaves like a leaf going forward)
    print("After  detach_(): b2.requires_grad:", b2.requires_grad, "| b2.grad_fn:", b2.grad_fn)

    # 이 값에서부터 추적을 다시 켜고 싶다면 새 잎으로 만든다:
    # b2 = b2.detach().requires_grad_(True)

if __name__ == "__main__":
    main()```

## 논의

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

여기서 보여준 패턴들은 실무적인 PyTorch 개발의 토대이다. 각 개념은 데이터 표현, 자동 미분, 하드웨어 가속을 하나의 일관된 API로 통합하는 텐서 추상화 위에 세워진다.

## 연습문제

**연습문제 1.**
텐서를 하나 만들고 `clone()` 복사본과 `detach()` 복사본을 각각 만들어라. 원본을 제자리에서 수정한 뒤 어느 복사본이 영향을 받는지 보여라.

??? success "연습문제 1 풀이"
    ```python
    original = torch.tensor([1., 2., 3.], requires_grad=True)
    cloned = original.clone()
    detached = original.detach()
    original.add_(10)  # 제자리(no_grad 안에서)
    print(cloned)    # 변하지 않음(독립적인 저장소)
    print(detached)  # 변함(저장소를 공유한다)
    ```

---


**연습문제 2.**
`clone()`만 쓰거나 `detach()`만 쓰는 경우와 비교하여 `detach().clone()`을 언제 쓰는지 설명하라.

??? success "연습문제 2 풀이"
    `clone()`만 쓰면 데이터는 복사하지만 autograd 그래프 연결은 유지된다. `detach()`만 쓰면 저장소는 공유하되 그래프에서 떨어져 나온다. `detach().clone()`은 그래프 연결이 없는 독립적인 복사본을 준다. 학습에 영향을 주지 않으면서 기록, 비교, 직렬화를 위한 스냅숏을 만들 때 가장 안전한 선택이다.

---


**연습문제 3.**
슬라이스를 수정한 뒤 원본을 확인하여, 텐서를 슬라이싱하면 뷰(저장소 공유)가 만들어짐을 보여라.

??? success "연습문제 3 풀이"
    ```python
    x = torch.tensor([1., 2., 3., 4., 5.])
    s = x[1:4]  # 뷰
    s[0] = 99.
    print(x)  # tensor([ 1., 99.,  3.,  4.,  5.])
    # 슬라이스가 저장소를 공유하므로 원본이 바뀌었다.
    ```
