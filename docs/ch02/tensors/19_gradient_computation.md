# 경사 계산 - 고급 경사 연산

이 스크립트는 경사 계산과 고급 경사 연산을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""튜토리얼 19: 기울기 셈 - 앞선 기울기 셈"""
import torch

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Higher-Order Gradients")
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3  # y = x^3
    print(f"y = x^3 where x = {x}")
    grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"dy/dx = 3x^2 = {grad_y}")
    grad2_y = torch.autograd.grad(grad_y, x)[0]
    print(f"d²y/dx² = 6x = {grad2_y}")
    
    header("2. Gradient of Multiple Outputs")
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y1 = x[0] ** 2
    y2 = x[1] ** 3
    print(f"x = {x}")
    print(f"y1 = x[0]^2 = {y1}")
    print(f"y2 = x[1]^3 = {y2}")
    grad_x = torch.autograd.grad([y1, y2], x, grad_outputs=[torch.tensor(1.0), torch.tensor(1.0)])[0]
    print(f"Gradient: {grad_x}")
    
    header("3. Jacobian Matrix")
    def f(x):
        return torch.stack([x[0]**2, x[1]**2, x[0]*x[1]])
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = f(x)
    print(f"x = {x}")
    print(f"f(x) = {y}")
    jacobian = torch.autograd.functional.jacobian(f, x)
    print(f"Jacobian:\n{jacobian}")
    
    header("4. Gradient Checking")
    def numerical_gradient(f, x, eps=1e-5):
        grad = torch.zeros_like(x)
        for i in range(x.numel()):
            x_plus = x.clone()
            x_plus.view(-1)[i] += eps
            x_minus = x.clone()
            x_minus.view(-1)[i] -= eps
            grad.view(-1)[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    def f(x): return (x**2).sum()
    y = f(x)
    y.backward()
    auto_grad = x.grad.clone()
    x.grad.zero_()
    num_grad = numerical_gradient(f, x)
    print(f"Autograd: {auto_grad}")
    print(f"Numerical: {num_grad}")
    print(f"Close? {torch.allclose(auto_grad, num_grad)}")
    
    header("5. Gradient Accumulation Pattern")
    model_output = torch.tensor(0.0, requires_grad=True)
    accumulated_loss = 0
    for i in range(3):
        loss = (model_output - i) ** 2
        loss.backward()
        accumulated_loss += loss.item()
        print(f"Step {i+1}: grad = {model_output.grad}")
    print(f"Total accumulated loss: {accumulated_loss}")
    
    header("6. Gradient Masking")
    x = torch.randn(5, requires_grad=True)
    y = x ** 2
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    y.backward(mask)
    print(f"x = {x}")
    print(f"Gradient (masked): {x.grad}")
    print("Only positions with mask=1.0 get gradients!")
    
    header("7. Practical: L2 Regularization")
    weights = torch.randn(10, requires_grad=True)
    predictions = weights.sum()
    target = torch.tensor(5.0)
    loss = (predictions - target) ** 2
    reg_lambda = 0.01
    regularization = reg_lambda * (weights ** 2).sum()
    total_loss = loss + regularization
    print(f"Loss: {loss.item():.4f}")
    print(f"Regularization: {regularization.item():.4f}")
    print(f"Total: {total_loss.item():.4f}")
    total_loss.backward()
    print(f"Gradient includes regularization term!")

if __name__ == "__main__":
    main()```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

PyTorch는 (저장소를 공유하는 뷰를 반환하는) 기본 슬라이싱과 (복사본을 반환하는) 불리언 마스크나 정수 배열을 이용한 고급 인덱싱을 모두 지원한다. 이 구분을 이해하는 것은 메모리 효율을 위해서도, 인덱싱한 결과를 수정할 때 의도치 않은 부작용을 피하기 위해서도 중요하다.

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
