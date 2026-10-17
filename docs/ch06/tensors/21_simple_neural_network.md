# 간단한 뉴럴 넷트웍 - 바닥부터 만들기

텐서 연산과 자동 미분을 뉴럴 넷트웍의 학습 단계로 연결한다. 수동으로 관리하는 웨이트와 `nn.Module`로 등록한 파라메터가 같은 학습 흐름에 어떻게 들어가는지 살펴본다.

## 1. 코드

```python
"""튜토리얼 21: 단순한 뉴럴 넷트웍 - 밑바닥부터 짓기"""
import torch
import torch.nn as nn
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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
    main()
```

**출력:**

```

======================================================================
1. Simple Linear Layer
======================================================================
Layer: Linear(in_features=3, out_features=2, bias=True)
Weight shape: torch.Size([2, 3])
Bias shape: torch.Size([2])

Input: torch.Size([1, 3])
Output: torch.Size([1, 2])

======================================================================
2. Multi-Layer Network
======================================================================
SimpleNet(
  (fc1): Linear(in_features=10, out_features=20, bias=True)
  (fc2): Linear(in_features=20, out_features=10, bias=True)
  (fc3): Linear(in_features=10, out_features=1, bias=True)
)

Input: torch.Size([5, 10])
Output: torch.Size([5, 1])

======================================================================
3. Model Parameters
======================================================================
Total parameters: 441
Trainable parameters: 441
fc1.weight: torch.Size([20, 10])
fc1.bias: torch.Size([20])
fc2.weight: torch.Size([10, 20])
fc2.bias: torch.Size([10])
fc3.weight: torch.Size([1, 10])
fc3.bias: torch.Size([1])

======================================================================
4. Forward Pass
======================================================================
Input shape: torch.Size([3, 10])
Output shape: torch.Size([3, 1])
Output:
tensor([[0.0619],
        [0.1475],
        [0.0994]])

======================================================================
5. Loss Function
======================================================================
Predictions:
tensor([[2.5000],
        [3.0000],
        [1.5000]], requires_grad=True)
Targets:
tensor([[2.],
        [3.],
        [2.]])
MSE Loss: 0.1667

======================================================================
6. Backward Pass
======================================================================
Gradient of predictions:
tensor([[ 0.3333],
        [ 0.0000],
        [-0.3333]])

======================================================================
7. Simple Training Loop
======================================================================
Epoch 1, Loss: 1.1780
Epoch 2, Loss: 1.1735
Epoch 3, Loss: 1.1694

======================================================================
8. Model Evaluation
======================================================================
Test output shape: torch.Size([5, 1])
Model in eval mode - no gradients computed!
```

## 2. 논의

순전파에서 예측과 로스를 계산하고, 역전파로 파라메터의 그래디언트를 구한 뒤 갱신한다. 이전 단계의 그래디언트를 지우는 일과 새 그래디언트를 계산하는 일은 별개다. 수동 갱신은 `no_grad` 안에서 수행하고, 옵티마이저를 쓰면 `step()`에 맡긴다.

`nn.Module`은 하위 레이어와 `Parameter`를 등록하여 `parameters()`·`state_dict()`·장치 이동에 포함시킨다. 텐서를 클래스 속성에 저장했다는 이유만으로 모두 학습 파라메터가 되는 것은 아니다. 추론에서는 평가 모드와 그래디언트 추적 중단의 역할을 구별한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

</div>

??? success "풀이"
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
제곱합처럼 저장된 중간값이 필요한 로스에서 `retain_graph=True` 없이 같은 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

</div>

??? success "풀이"
    기본 역전파는 사용을 마친 저장 텐서를 해제한다. 같은 그래프의 두 번째 역전파에 이 값들이 필요하면 `RuntimeError`가 발생한다. 저장 텐서가 필요 없는 단순 연산은 반복 호출이 가능할 수도 있으므로 모든 그래프에서 반드시 오류가 나는 것은 아니다. 첫 호출의 `retain_graph=True`는 필요한 중간 결과를 보존해 재사용을 가능하게 하지만 그만큼 메모리를 더 오래 사용한다. `.grad` 초기화는 이 보존 여부와 별개의 작업이다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잎 텐서 `w`를 만들고, 반복마다 같은 식으로 로스를 새로 계산하라. 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하라. 관찰된 값을 설명하라.

</div>

??? success "풀이"
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

텐서 연산과 자동 미분을 뉴럴 넷트웍의 학습 단계로 연결한다. 수동으로 관리하는 웨이트와 `nn.Module`로 등록한 파라메터가 같은 학습 흐름에 어떻게 들어가는지 살펴본다.
