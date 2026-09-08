# 간단한 신경망 - 바닥부터 만들기

이 스크립트는 간단한 신경망을 바닥부터 만드는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""튜토리얼 21: 단순한 신경망 - 밑바닥부터 짓기"""
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Simple Linear Layer")
    input_size, output_size = 3, 2
    layer = nn.Linear(input_size, output_size)
    print(f"Layer: {layer}")
    print(f"Weight shape: {layer.weight.shape}")  # (2, 3)
    print(f"Bias shape: {layer.bias.shape}")  # (2,)
    x = torch.randn(1, 3)  # Batch of 1 sample
    output = layer(x)
    print(f"\nInput: {x.shape}")
    print(f"Output: {output.shape}")
    
    header("2. Multi-Layer Network")
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SimpleNet()
    print(model)
    x = torch.randn(5, 10)  # 5 samples, 10 features
    output = model(x)
    print(f"\nInput: {x.shape}")
    print(f"Output: {output.shape}")
    
    header("3. Model Parameters")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    
    header("4. Forward Pass")
    x = torch.randn(3, 10)
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")
    
    header("5. Loss Function")
    predictions = torch.tensor([[2.5], [3.0], [1.5]], requires_grad=True)
    targets = torch.tensor([[2.0], [3.0], [2.0]])
    mse_loss = nn.MSELoss()
    loss = mse_loss(predictions, targets)
    print(f"Predictions:\n{predictions}")
    print(f"Targets:\n{targets}")
    print(f"MSE Loss: {loss.item():.4f}")
    
    header("6. Backward Pass")
    loss.backward()
    print(f"Gradient of predictions:\n{predictions.grad}")
    
    header("7. Simple Training Loop")
    model = SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X = torch.randn(100, 10)  # 100 samples
    y = torch.randn(100, 1)   # 100 targets
    
    for epoch in range(3):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    header("8. Model Evaluation")
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        test_input = torch.randn(5, 10)
        test_output = model(test_input)
    print(f"Test output shape: {test_output.shape}")
    print("Model in eval mode - no gradients computed!")

if __name__ == "__main__":
    main()```

## 2. 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사 초기화, 역전파, 매개변수 갱신이다. 각 구성 요소가 결정적인 역할을 한다. 최적화기는 갱신 규칙(SGD, Adam 등)을 캡슐화하고 학습률과 모멘텀 상태를 내부에서 관리한다.

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

**다룬 것** — 간단한 신경망 - 바닥부터 만들기

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다.

핵심 클래스는 `SimpleNet`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
