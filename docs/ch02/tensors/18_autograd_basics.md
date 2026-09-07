# Autograd 기초 - 자동 미분의 기본

이 스크립트는 Autograd의 기초, 즉 자동 미분의 기본을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""튜토리얼 18: 자동 미분 기초 - 자동 미분의 바탕"""
import torch

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. requires_grad - Tracking Computations")
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    print(f"x = {x}")
    print(f"x.requires_grad: {x.requires_grad}")
    y = torch.tensor([5.0, 6.0])
    print(f"y = {y}")
    print(f"y.requires_grad: {y.requires_grad}")
    z = x + y  # z inherits requires_grad from x
    print(f"\nz = x + y = {z}")
    print(f"z.requires_grad: {z.requires_grad}")
    
    header("2. Backward Pass - Computing Gradients")
    x = torch.tensor(3.0, requires_grad=True)
    print(f"x = {x}")
    y = x ** 2  # y = x^2
    print(f"y = x^2 = {y}")
    y.backward()  # Compute dy/dx
    print(f"x.grad (dy/dx = 2x = 6): {x.grad}")
    
    header("3. Gradient Accumulation")
    x = torch.tensor(5.0, requires_grad=True)
    for i in range(3):
        y = x ** 2
        y.backward()
        print(f"Iteration {i+1}: x.grad = {x.grad}")
    print("\nNote: Gradients ACCUMULATE! Use zero_grad() to reset.")
    
    header("4. Zeroing Gradients")
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3
    y.backward()
    print(f"First backward: x.grad = {x.grad}")
    x.grad.zero_()  # Reset gradient
    print(f"After zero_grad(): x.grad = {x.grad}")
    y = x ** 2
    y.backward()
    print(f"Second backward: x.grad = {x.grad}")
    
    header("5. Multiple Variables")
    x = torch.tensor(3.0, requires_grad=True)
    y = torch.tensor(4.0, requires_grad=True)
    z = x**2 + y**3  # z = x^2 + y^3
    print(f"x = {x}, y = {y}")
    print(f"z = x^2 + y^3 = {z}")
    z.backward()
    print(f"x.grad (dz/dx = 2x = 6): {x.grad}")
    print(f"y.grad (dz/dy = 3y^2 = 48): {y.grad}")
    
    header("6. Vector-Jacobian Product")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2  # Element-wise
    print(f"x = {x}")
    print(f"y = x^2 = {y}")
    gradient = torch.tensor([1.0, 1.0, 1.0])
    y.backward(gradient)  # Need gradient for non-scalar output
    print(f"x.grad = {x.grad}")  # [2, 4, 6]
    
    header("7. Detaching from Graph")
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2
    print(f"y requires_grad: {y.requires_grad}")
    z = y.detach()  # Detach from computation graph
    print(f"z requires_grad: {z.requires_grad}")
    w = z * 3
    print(f"w requires_grad: {w.requires_grad}")
    
    header("8. No-Grad Context")
    x = torch.tensor(2.0, requires_grad=True)
    print(f"x requires_grad: {x.requires_grad}")
    with torch.no_grad():
        y = x ** 2
        print(f"Inside no_grad, y requires_grad: {y.requires_grad}")
    print("Use no_grad() during inference to save memory!")
    
    header("9. Practical: Simple Loss Function")
    prediction = torch.tensor(2.5, requires_grad=True)
    target = torch.tensor(3.0)
    loss = (prediction - target) ** 2
    print(f"Prediction: {prediction}")
    print(f"Target: {target}")
    print(f"Loss (MSE): {loss}")
    loss.backward()
    print(f"Gradient: {prediction.grad}")
    print("Gradient tells us to increase prediction!")

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
