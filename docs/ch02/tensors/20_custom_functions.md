# 사용자 정의 Autograd 함수 - PyTorch 미분 확장하기

이 스크립트는 사용자 정의 autograd 함수로 PyTorch의 미분을 확장하는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""Tutorial 20: Custom Autograd Functions - Extending PyTorch's differentiation"""
import torch

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 * input

class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

def main():
    header("1. Custom Function: Square")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    square_fn = SquareFunction.apply
    y = square_fn(x)
    print(f"x = {x}")
    print(f"y = x^2 = {y}")
    y.sum().backward()
    print(f"x.grad = {x.grad}")
    
    header("2. Custom Function: ReLU")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    relu_fn = ReLUFunction.apply
    y = relu_fn(x)
    print(f"x = {x}")
    print(f"ReLU(x) = {y}")
    y.sum().backward()
    print(f"x.grad = {x.grad}")
    
    header("3. When to Use Custom Functions")
    print("""
    Use custom autograd functions when:
    1. Implementing new operations not in PyTorch
    2. Need custom gradient behavior
    3. Optimizing performance of specific operations
    4. Interfacing with external libraries
    5. Research: testing new activation functions
    """)
    
    header("4. Built-in vs Custom")
    x = torch.randn(5, requires_grad=True)
    built_in = torch.relu(x)
    custom = ReLUFunction.apply(x)
    print(f"Built-in ReLU: {built_in}")
    print(f"Custom ReLU: {custom}")
    print(f"Match? {torch.allclose(built_in, custom)}")

if __name__ == "__main__":
    main()```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

여기서 보여준 패턴들은 실무적인 PyTorch 개발의 토대이다. 각 개념은 데이터 표현, 자동 미분, 하드웨어 가속을 하나의 일관된 API로 통합하는 텐서 추상화 위에 세워진다.

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
