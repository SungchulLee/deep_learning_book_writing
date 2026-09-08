# 경사 추적 끄기

이 스크립트는 경사 추적을 끄는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""기울기 좇기를 끈다."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    # 기본적으로 텐서는 requires_grad=False로 만들어진다
    # (대부분의 텐서는 매개변수가 아니라 그냥 데이터이기 때문이다).
    # 최적화하려는 매개변수에는 requires_grad=True를 명시적으로 설정해야 한다.
    w = torch.randn(4, requires_grad=True)  # Equivalent: w = torch.randn(4).requires_grad_()
    print("Before: w.requires_grad =", w.requires_grad)  # True

    # --- requires_grad_와 torch.no_grad 비교(논의) -----------------
    # • w.requires_grad_(False)
    #     - 메타데이터 플래그만 제자리에서 바꾼다. 텐서 값은 바뀌지 않는다.
    #     - 이 동작은 autograd가 추적하지 않으므로 torch.no_grad()로 감싸는 것은
    #       불필요하며 아무 이득이 없다.
    #
    # • with torch.no_grad():
    #     - 값을 바꾸는 연산을 autograd 그래프에
    #       기록하고 싶지 않을 때 쓴다(예: w.add_(...), w.copy_(...), 수동 갱신).
    #     - 예:
    #         with torch.no_grad():
    #             w.add_(update)   # 추적 없이 값을 바꾼다
    # ---------------------------------------------------------------------------
    print("Disabling gradient tracking for w ...")
    w.requires_grad_(False)  # metadata toggle; no value change; no need for torch.no_grad()
    print("After:  w.requires_grad =", w.requires_grad)  # False

    # autograd의 규칙:
    #   어떤 연산의 모든 입력이 requires_grad=False이면
    #   결과도 requires_grad=False이다(autograd 이력이 없다).
    loss = (w ** 2).sum()
    print("loss.requires_grad (expect False):", loss.requires_grad)

    try:
        loss.backward()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()```

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

**다룬 것** — 경사 추적 끄기

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다.

앞의 연습문제 3개로 직접 확인할 수 있다.
