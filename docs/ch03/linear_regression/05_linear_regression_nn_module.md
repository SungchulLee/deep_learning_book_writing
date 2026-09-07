# nn 모듈로 만드는 선형 회귀

`nn.Module` 클래스는 모델을 정의하기 위한 PyTorch의 표준 추상화이며, `nn.Linear`는 완전 연결 층의 내장 구현을 제공한다. 여기에 `torch.optim`의 최적화기를 결합하면 PyTorch 생태계 전반의 관례를 따르는 깔끔하고 확장 가능한 코드가 된다. 이 튜토리얼은 임의로 복잡한 구조에도 그대로 적용되는 PyTorch의 관용적 학습 루프를 보여준다.

## 코드

```python
"""
==============================================================================
05_linear_regression_nn_module.py
==============================================================================
어려움: ⭐⭐ (가운데)

DESCRIPTION:
    PyTorch의 nn.Module과 nn.Linear을 쓰는 선형 회귀.
    이것이 PyTorch에서 모델을 짓는 "제대로 된" 길이다.

다루는 것:
    - 모델을 위한 nn.Module 클래스
    - nn.Linear 층
    - 최적화기(torch.optim.SGD)
    - 더 깔끔하고 잘 늘어나는 코드

PREREQUISITES:
    - 튜토리얼 04(자동 미분)

학습 목표:
    - nn.Module으로 맞춤 모델을 만든다
    - 붙박이 층(nn.Linear)을 쓴다
    - 매개변수 고치기에 최적화기를 쓴다
    - PyTorch의 좋은 버릇을 따른다

걸리는 때: 20분쯤
==============================================================================
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("=" * 70)
print("LINEAR REGRESSION WITH NN.MODULE")
print("=" * 70)

# ============================================================================
# 1부: 데이터 생성
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE DATA")
print("=" * 70)

torch.manual_seed(42)

TRUE_W = 2.5
TRUE_B = 3.0
n_samples = 100

X = torch.rand(n_samples, 1) * 20 - 10  # Shape: (100, 1)
noise = torch.randn(n_samples, 1) * 2
y = TRUE_W * X + TRUE_B + noise

print(f"Data shapes: X={X.shape}, y={y.shape}")
print(f"True parameters: w={TRUE_W}, b={TRUE_B}")

# ============================================================================
# 2부: NN.MODULE로 모델 정의하기
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: DEFINE MODEL CLASS")
print("=" * 70)

class LinearRegressionModel(nn.Module):
    """
    nn.Module을 쓰는 선형 회귀 모델
    
    이것이 PyTorch에서 모델을 매기는 여느 길이다.
    모든 모델은 nn.Module을 물려받아야 한다.
    """
    
    def __init__(self, input_dim=1, output_dim=1):
        """
        모델의 초기화한다
        
        Args:
            input_dim: 입력 특징의 수
            output_dim: 출력 특징의 수
        """
        # 언제나 부모 생성자를 먼저 호출한다
        super(LinearRegressionModel, self).__init__()
        
        # 층을 정의한다
        # nn.Linear(in_features, out_features)
        # 이렇게 하면 y = X @ W.T + b가 된다
        # 여기서 W의 모양은 (out_features, in_features)이다
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        """
        순전파: 데이터가 모델을 어떻게 흐르는지 매긴다
        
        Args:
            x: 입력 텐서
        
        Returns:
            내놓는 예측
        """
        return self.linear(x)

# 모델 인스턴스 생성
model = LinearRegressionModel(input_dim=1, output_dim=1)

print("Model created:")
print(model)
print(f"\nModel parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# ============================================================================
# 3부: 손실과 최적화기 정의
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: DEFINE LOSS AND OPTIMIZER")
print("=" * 70)

# 손실 함수
criterion = nn.MSELoss()  # Mean Squared Error
print(f"Loss function: {criterion}")

# 최적화기 - 매개변수 갱신을 자동으로 처리한다!
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(f"Optimizer: {optimizer}")
print(f"Learning rate: {learning_rate}")

print("""
최적화기를 쓰는 핵심 이점:
1. 모델의 매개변수를 모두 절로 고친다
2. 직접 매개변수를 고칠 일이 없다
3. 최적화기를 갈아 끼우기 쉽다(SGD, Adam, RMSprop 따위)
4. optimizer.zero_grad()으로 기울기 0으로 만들기를 다룬다
""")

# ============================================================================
# 4부: 학습 루프 - PyTorch 방식
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: TRAINING LOOP")
print("=" * 70)

n_epochs = 100
loss_history = []
w_history = []
b_history = []

print(f"Training for {n_epochs} epochs...")
print(f"\n{'Epoch':<8} {'Loss':<12} {'w':<12} {'b':<12}")
print("-" * 50)

for epoch in range(n_epochs):
    # 1. 순전파
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # 2. 역전파
    optimizer.zero_grad()  # Zero gradients (replaces w.grad.zero_())
    loss.backward()        # Compute gradients
    optimizer.step()       # Update parameters (replaces manual update)
    
    # 이력 추적
    loss_history.append(loss.item())
    w_history.append(model.linear.weight.item())
    b_history.append(model.linear.bias.item())
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {loss.item():<12.4f} "
              f"{model.linear.weight.item():<12.4f} "
              f"{model.linear.bias.item():<12.4f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

final_w = model.linear.weight.item()
final_b = model.linear.bias.item()

print(f"\nFinal Results:")
print(f"  Learned w: {final_w:.4f} (True: {TRUE_W}, Error: {abs(final_w-TRUE_W):.4f})")
print(f"  Learned b: {final_b:.4f} (True: {TRUE_B}, Error: {abs(final_b-TRUE_B):.4f})")
print(f"  Final loss: {loss_history[-1]:.6f}")

# ============================================================================
# 5부: 모델 평가 모드
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: MODEL EVALUATION MODE")
print("=" * 70)

print("""
모델에는 두 결이 있다.
1. 학습 결(model.train()): 기본이며 드롭아웃, 배치 정규화 따위를 켠다
2. 따짐 결(model.eval()): 드롭아웃, 배치 정규화 따위를 끈다

선형 회귀에서는 걸리지 않지만 좋은 버릇이다!
""")

# 모델을 평가 모드로 바꾼다
model.eval()
print("Model set to evaluation mode")

# 예측한다 (경사 추적이 필요 없다)
with torch.no_grad():
    X_test = torch.tensor([[5.0], [-3.0], [0.0]])
    y_pred_test = model(X_test)
    
print(f"\nTest predictions:")
for i in range(len(X_test)):
    x_val = X_test[i].item()
    y_true_val = TRUE_W * x_val + TRUE_B
    y_pred_val = y_pred_test[i].item()
    print(f"  X={x_val:6.1f} -> Pred: {y_pred_val:7.2f}, True: {y_true_val:7.2f}")

# ============================================================================
# 6부: 모델 저장하고 불러오기
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: SAVING AND LOADING MODELS")
print("=" * 70)

# 모델을 저장한다
model_path = '/home/claude/pytorch_linear_regression_tutorial/linear_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# 모델을 불러온다
new_model = LinearRegressionModel(input_dim=1, output_dim=1)
new_model.load_state_dict(torch.load(model_path))
new_model.eval()
print("Model loaded successfully")

# 불러온 모델을 확인한다
with torch.no_grad():
    y_pred_loaded = new_model(X_test)
    print("\nVerifying loaded model (should match above):")
    for i in range(len(X_test)):
        print(f"  X={X_test[i].item():6.1f} -> Pred: {y_pred_loaded[i].item():7.2f}")

# ============================================================================
# 7부: 결과 시각화
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 손실 곡선
axes[0, 0].plot(loss_history, linewidth=2, color='blue')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss (nn.Module)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 매개변수의 변화
axes[0, 1].plot(w_history, label='w', linewidth=2)
axes[0, 1].axhline(y=TRUE_W, color='r', linestyle='--', label=f'True w={TRUE_W}')
axes[0, 1].plot(b_history, label='b', linewidth=2)
axes[0, 1].axhline(y=TRUE_B, color='g', linestyle='--', label=f'True b={TRUE_B}')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Value')
axes[0, 1].set_title('Parameter Evolution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 최종 적합
model.eval()
with torch.no_grad():
    X_sorted, _ = torch.sort(X, dim=0)
    y_pred_sorted = model(X_sorted)

axes[1, 0].scatter(X.numpy(), y.numpy(), alpha=0.5, s=20)
axes[1, 0].plot(X_sorted.numpy(), y_pred_sorted.numpy(), 'r-', linewidth=2)
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('Final Model Fit')
axes[1, 0].grid(True, alpha=0.3)

# 코드 비교
comparison = """
학습 루프가 나아온 길:

튜토리얼 02(넘파이):
  - 직접 쓰는 기울기 식
  - 직접 하는 매개변수 고치기
  - 코드 40줄쯤

튜토리얼 03(직접 하는 PyTorch):
  - 텐서 셈
  - 직접 하는 기울기
  - 직접 하는 고치기

튜토리얼 04(자동 미분):
  - loss.backward()
  - 직접 하는 고치기
  - grad.zero_()

튜토리얼 05(nn.Module):
  - model(X)
  - optimizer.zero_grad()
  - loss.backward()
  - optimizer.step()
  
훨씬 깔끔하고 손보기 좋다!
"""
axes[1, 1].text(0.05, 0.95, comparison, transform=axes[1, 1].transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/05_nn_module_results.png', dpi=100)
print("\nSaved visualization to: 05_nn_module_results.png")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
여느 PyTorch 학습 루프:

model = MyModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(n_epochs):
    # 순전파
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

핵심 조각:
1. nn.Module: 모든 모델의 밑 클래스
2. nn.Linear: 붙박이 선형 층
3. optimizer: 매개변수 고치기를 다룬다
4. criterion: 손실 함수

Advantages:
✓ 깔끔하고 읽기 좋은 코드
✓ 복잡한 모델으로 넓히기 쉽다
✓ 매개변수를 절로 다룬다
✓ 모델을 저장하고 불러오기 쉽다
✓ GPU을 받친다(.to('cuda')만 더하면 된다)

다음: 튜토리얼 06 - 여러 입력 특징!
""")


if __name__ == "__main__":
    pass
```

## 논의

모델을 `nn.Module`의 하위 클래스로 정의하면 코드 정리를 넘어서는 여러 이점이 생긴다. `parameters()` 메서드는 등록된 모든 층에서 학습 가능한 가중치를 자동으로 모아 주므로, 최적화기가 따로 장부를 관리하지 않고도 갱신할 수 있다. `train()`과 `eval()` 메서드는 드롭아웃이나 배치 정규화 같은 동작을 전환한다. 그리고 `state_dict()` / `load_state_dict()`로 함수 호출 한 번에 모델 가중치를 저장하고 불러올 수 있다.

학습 루프는 네 줄짜리 패턴이 된다. `y_pred = model(X)`, `loss = criterion(y_pred, y)`, `optimizer.zero_grad(); loss.backward()`, `optimizer.step()`이다. 각 줄은 순전파, 손실 계산, 경사 계산, 매개변수 갱신이라는 분명한 역할을 맡으며, 이 패턴은 모델이 아무리 복잡해져도 바뀌지 않는다. 이 패턴을 익히는 것이 PyTorch를 능숙하게 쓰기 위한 가장 중요한 한 걸음이다.

최적화기를 바꾸는 일은 간단하다. `torch.optim.SGD`를 `torch.optim.Adam`이나 다른 최적화기로 바꾸기만 하면 나머지 코드는 그대로다. 이런 모듈성은 모델(계산을 정의), 기준(목적을 정의), 최적화기(갱신 규칙을 정의) 사이의 깔끔한 분리에서 곧바로 따라 나온다. 같은 분리 덕분에 여러 손실 함수, 학습률 일정, 정칙화 전략을 손쉽게 실험할 수 있다.

## 연습문제

**익힘 1.**
`model.named_parameters()`로 모델의 이름 붙은 매개변수를 모두 출력하고, 가중치와 편향의 모양이 입력 1개·출력 1개인 선형 층에서 기대하는 것과 일치하는지 확인하라.

??? success "익힘 1 풀이"
    ```python
    import torch.nn as nn
    
    model = nn.Linear(1, 1)
    for name, param in model.named_parameters():
        print(f'{name}: shape={param.shape}, requires_grad={param.requires_grad}')
    # 예상 출력:
    # weight: shape=torch.Size([1, 1]), requires_grad=True
    # bias: shape=torch.Size([1]), requires_grad=True
    ```

---

**익힘 2.**
`nn.MSELoss()`를 평균절대오차(MAE)를 계산하는 사용자 정의 손실 함수로 대체하라. 모델을 학습시키고 MSE 버전과 수렴을 비교하라.

??? success "익힘 2 풀이"
    ```python
    import torch
    import torch.nn as nn
    
    def mae_loss(y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))
    
    torch.manual_seed(42)
    X = torch.rand(100, 1) * 20 - 10
    y = 2.5 * X + 3.0 + torch.randn(100, 1) * 2
    
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        y_pred = model(X)
        loss = mae_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}: MAE={loss.item():.4f}')
    
    print(f'w={model.weight.item():.4f}, b={model.bias.item():.4f}')
    ```

---

**익힘 3.**
`torch.save(model.state_dict(), path)`로 학습된 모델을 저장한 뒤 새 `nn.Linear` 인스턴스에 불러오고, 시험 데이터에서 예측이 원래 모델과 일치하는지 확인하라.

??? success "익힘 3 풀이"
    ```python
    import torch
    import torch.nn as nn
    
    torch.manual_seed(42)
    X = torch.rand(100, 1) * 20 - 10
    y = 2.5 * X + 3.0 + torch.randn(100, 1) * 2
    
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(100):
        loss = nn.MSELoss()(model(X), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    torch.save(model.state_dict(), '/tmp/model.pth')
    
    model2 = nn.Linear(1, 1)
    model2.load_state_dict(torch.load('/tmp/model.pth'))
    model2.eval()
    
    with torch.no_grad():
        pred1 = model(X[:5])
        pred2 = model2(X[:5])
        print(f'Original: {pred1.flatten().tolist()}')
        print(f'Loaded:   {pred2.flatten().tolist()}')
        print(f'Match: {torch.allclose(pred1, pred2)}')
    ```
