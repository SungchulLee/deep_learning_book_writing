# 작은 학습 단계

이 스크립트는 작은 학습 단계 하나을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""작은 학습 걸음."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    torch.manual_seed(0)

    # 아주 작은 데이터셋: 선형 회귀 장난감 예제
    # 입출력 쌍: x ∈ {-1, 0, 1}, 목표 y ∈ {1, 0, -1}
    x = torch.tensor([[-1.0], [0.0], [1.0]])  # shape (3,1)
    y = torch.tensor([[1.0], [0.0], [-1.0]])  # shape (3,1)

    # 학습 가능한 매개변수(잎 텐서).
    # 잎 텐서는 사용자가 requires_grad=True로 만든 텐서이다.
    # autograd가 이들에 대한 경사를 직접 계산한다.
    w = torch.randn(1, requires_grad=True)  # dtype is torch.float32 by default
    b = torch.zeros(1, requires_grad=True)  # dtype is torch.float32 by default

    lr = 0.1
    for step in range(3):
        # ---------------- 순전파 ----------------
        y_hat = x * w + b                     # linear model, shape (3,1)
        loss = torch.mean((y_hat - y) ** 2)   # mean squared error (scalar)
        # 참고:
        #   - loss는 0차원 텐서(스칼라 텐서)이다.
        #   - loss.grad_fn = <MeanBackward0>이다. 그 이유는
        #     계산 그래프의 평균 연산으로 만들어졌기 때문이다.
        #   - grad_fn은 다음을 계산할 줄 아는 함수를 가리킨다.
        #     이 텐서의 부모에 대한 경사를 계산하는 함수를 가리킨다.
        #   - (w, b 같은) 잎 텐서는 grad_fn=None이다.

        # ---------------- 역전파 ----------------
        # 경사는 기본적으로 누적되므로 먼저 예전 경사를 지운다.
        # 경사를 0이 아니라 None으로 되돌리려면 (제자리인) .zero_()를 쓴다.
        # 경사를 0이 아니라 None으로 만드는 것이 zero_()의 기본 동작이다.
        # PyTorch에는 .zero()가 없다(.zero_()만 있다).
        if w.grad is not None: w.grad.zero_()
        if b.grad is not None: b.grad.zero_()
        print(f"{w.grad = }")

        # 계산 그래프를 훑으며 d(loss)/dw와 d(loss)/db를 계산한다
        # loss.backward() → grad_fn → 부모 텐서 → 연쇄 법칙의 순서로 이루어진다.
        loss.backward()

        # ---------------- 매개변수 갱신 ----------------
        # 중요: autograd가 기록하지 않도록 torch.no_grad()로 감싼다
        # 이 제자리 갱신들을 계산 그래프에 기록하지 않도록 한다.
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        # ---------------- 기록 ----------------
        # 왜 .item()인가? 이유는:
        #   - loss, w, b는 텐서이다(모양이 []인 스칼라).
        #   - 텐서를 그대로 출력하면 "tensor(..., grad_fn=...)"이 보인다.
        #   - .item()은 순수한 파이썬 실수를 뽑아내므로 다음과 같은 형식 지정이
        #     :.6f나 :.4f가 오류 없이 동작한다.
        print(f"step {step}: loss={loss.item():.6f} | w={w.item():.4f} | b={b.item():.4f}")

    # 최종 학습된 매개변수
    print("Final params:", {"w": w.item(), "b": b.item()})
    # - loss.item()에 관하여:
    #   * loss.item()은 **CPU의 파이썬 실수를 반환한다**. `loss`가 GPU/MPS에 있으면
    #     .item()을 호출하면 **장치→호스트 복사와 동기화 장벽** 이 발생한다
    #     (`loss`를 만들어 내는 앞선 GPU 연산이 모두 끝날 때까지 호스트가 기다린다).
    #   * 이는 GPU 파이프라인을 멈추게 하므로, 빡빡한 학습 루프에서는 **
    #     매 단계마다 .item()을 호출하는 것을 피하는** 편이 낫다. 다음을 권한다:
    #       - 기록 빈도를 줄인다(예: k 단계마다).
    #       - 텐서 값을 누적해 두었다가 성능이 중요한 구간 밖에서 `.detach().cpu()`를 호출한다.
    #       - 평균을 낼 때는 이따금 동기화한 **뒤** 누적합을 CPU 실수로 유지한다.
    #   * 파이썬 실수가 정말로 필요할 때에만 .item()을 쓴다(출력, 조기 종료 판정 등).
    #     다만 그것이 만드는 동기화 지점에 유의해야 한다.

if __name__ == "__main__":
    main()```

## 논의

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
