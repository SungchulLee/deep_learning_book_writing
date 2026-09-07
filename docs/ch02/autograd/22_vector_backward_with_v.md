# v를 이용한 벡터 역전파

이 스크립트는 벡터 출력에 대해 $v$를 넘기는 역전파을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""v를 주는 벡터 뒤로 걸음."""
import torch

# ========================================================================
# 메인
# ========================================================================

def main():

    torch.manual_seed(0)

    # 순전파: x ∈ R^3 → y ∈ R^3 (스칼라가 아닌 벡터 출력).
    x = torch.randn(3, requires_grad=True)
    y = torch.sin(x)  # elementwise sine, shape [3]
    print("x:", x)
    print("y = sin(x):", y, "| shape:", tuple(y.shape))

    # 인수 없이 .backward()를 호출하려면 autograd는 스칼라 손실을 요구한다.
    # y가 스칼라가 아니면 PyTorch는 v^T J를 만들기 위해 명시적인 "벡터" v를 필요로 한다.
    try:
        y.backward()
    except RuntimeError as e:
        print("As expected, calling y.backward() fails for non-scalar y:")
        print("  ", e)

    # y와 같은 모양의 v를 넘긴다. autograd는 다음을 계산한다:
    #   x.grad = J^T v이며 J = ∂y/∂x이다.
    # 이것이 벡터-야코비 곱(VJP)이다.
    v = torch.tensor([0.1, 1.0, 0.01], dtype=torch.float32)
    y.backward(v)
    print("Chosen v:", v)
    print("x.grad (v^T * J):", x.grad)

if __name__ == "__main__":
    main()```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

벡터-야코비 곱(VJP)은 후진 모드 자동 미분의 핵심 기본 연산이다. 야코비안이 $J$인 함수 $f: \mathbb{R}^n \to \mathbb{R}^m$에 대해 VJP는 주어진 벡터 $v$에 대한 $v^\top J$를 계산한다. 출력이 스칼라가 아닐 때 PyTorch는 어떤 출력들의 선형 결합을 미분할지 지정하도록 `.backward()`에 경사 인수 $v$를 명시할 것을 요구한다.

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
