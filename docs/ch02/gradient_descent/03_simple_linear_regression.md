# PyTorch로 만드는 간단한 선형 회귀

이 스크립트는 PyTorch로 간단한 선형 회귀를 만드는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""
================================================================================
1단계 - 보기 3: PyTorch로 하는 단순 선형 회귀
================================================================================

학습 목표:
- 경사 하강법을 참 회귀 문제에 쓴다
- 코드를 더 잘 짜려고 PyTorch의 nn.Module을 쓴다
- 학습 루프와 따짐 루프를 짠다
- 모델의 예측을 그림으로 본다

어려움: ⭐ 첫걸음

걸리는 때: 30~40분

PREREQUISITES:
- 보기 01과 02을 마쳤을 것
- 선형 회귀를 기본으로 이해하고 있을 것

================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("LINEAR REGRESSION WITH PYTORCH")
print("="*80)

# ============================================================================
# 1부: 현실적인 데이터셋 만들기
# ============================================================================
print("\n" + "="*80)
print("PART 1: DATASET CREATION")
print("="*80)

# 재현성을 위한 난수 시드 설정
torch.manual_seed(42)
np.random.seed(42)

# 합성 데이터 생성: y = 3x + 2 + 잡음
n_samples = 100
X_numpy = np.random.rand(n_samples, 1) * 10  # Random x values from 0 to 10
y_numpy = 3 * X_numpy + 2 + np.random.randn(n_samples, 1) * 2  # Add noise

# PyTorch 텐서로 변환
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"Dataset size: {n_samples} samples")
print(f"X shape: {X.shape}  (n_samples, n_features)")
print(f"y shape: {y.shape}  (n_samples, 1)")
print(f"\nTrue relationship: y = 3*x + 2")
print("Goal: Learn this relationship from noisy data")

# ============================================================================
# 2부: nn.Module로 모델 정의하기(실무적인 방법)
# ============================================================================
print("\n" + "="*80)
print("PART 2: MODEL DEFINITION")
print("="*80)

class LinearRegressionModel(nn.Module):
    """
    선형 회귀 모델: y = wx + b
    
    이것이 PyTorch에서 모델을 매기는 방식이다.
    Benefits:
    - 매개변수를 절로 다룬다
    - 넓히기 쉽다
    - PyTorch 최적화기와 맞물린다
    - 깔끔하고 짜임새 있는 코드
    """
    
    def __init__(self, input_dim, output_dim):
        """
        모델 매개변수의 초기화한다
        
        Args:
            input_dim: 입력 특징의 수
            output_dim: 출력 값의 수
        """
        super(LinearRegressionModel, self).__init__()
        
        # 선형 층 정의: y = wx + b
        # 이것이 requires_grad=True인 가중치(w)와 편향(b)을 만든다
        self.linear = nn.Linear(input_dim, output_dim)
        
        print(f"Created linear layer: {input_dim} inputs → {output_dim} outputs")
    
    def forward(self, x):
        """
        순전파: 예측을 계산한다
        
        Args:
            x: 모양이 (batch_size, input_dim)인 입력 텐서
        
        Returns:
            predictions: 모양이 (batch_size, output_dim)인 출력 텐서
        """
        return self.linear(x)


# 모델 인스턴스 생성
input_dim = 1   # Single feature (x)
output_dim = 1  # Single output (y)
model = LinearRegressionModel(input_dim, output_dim)

print(f"\nModel architecture:")
print(model)

# 초기 매개변수 출력
print(f"\nInitial parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.data.numpy().flatten()}")

# ============================================================================
# 3부: 손실 함수와 최적화기 정의
# ============================================================================
print("\n" + "="*80)
print("PART 3: LOSS FUNCTION & OPTIMIZER")
print("="*80)

# 손실 함수: 평균제곱오차
criterion = nn.MSELoss()
print("Loss function: MSE (Mean Squared Error)")
print("  L = (1/N) * Σ(y_pred - y_true)²")

# 최적화기: 확률적 경사 하강법
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(f"\nOptimizer: SGD (Stochastic Gradient Descent)")
print(f"  Learning rate: {learning_rate}")
print(f"  Optimizing: {sum(p.numel() for p in model.parameters())} parameters")

# ============================================================================
# 4부: 학습 루프
# ============================================================================
print("\n" + "="*80)
print("PART 4: TRAINING")
print("="*80)

n_epochs = 100
loss_history = []

print("Epoch |   Loss    | Weight  | Bias")
print("-" * 45)

for epoch in range(n_epochs):
    # -------------------------
    # 표준적인 학습 단계:
    # -------------------------
    
    # 1. 순전파
    y_pred = model(X)
    
    # 2. 손실 계산
    loss = criterion(y_pred, y)
    
    # 3. 경사 초기화(중요!)
    optimizer.zero_grad()
    
    # 4. 역전파(경사 계산)
    loss.backward()
    
    # 5. 매개변수 갱신
    optimizer.step()
    
    # 그림을 그리기 위해 손실을 저장
    loss_history.append(loss.item())
    
    # 진행 상황 출력
    if (epoch + 1) % 10 == 0:
        weight = model.linear.weight.item()
        bias = model.linear.bias.item()
        print(f"{epoch+1:4d}  | {loss.item():9.4f} | {weight:7.4f} | {bias:6.4f}")

print("-" * 45)

# ============================================================================
# 5부: 학습된 모델 평가
# ============================================================================
print("\n" + "="*80)
print("PART 5: EVALUATION")
print("="*80)

# 최종 매개변수 얻기
final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()

print(f"Learned equation: y = {final_weight:.4f}*x + {final_bias:.4f}")
print(f"True equation:    y = 3.0000*x + 2.0000")
print(f"\nParameter errors:")
print(f"  Weight error: {abs(final_weight - 3.0):.4f}")
print(f"  Bias error:   {abs(final_bias - 2.0):.4f}")

# 테스트 데이터에 대한 예측
model.eval()  # Set model to evaluation mode (important for some layers)
with torch.no_grad():  # Don't track gradients during evaluation
    # 몇 개의 점에서 시험
    x_test = torch.tensor([[2.0], [5.0], [8.0]])
    y_test_pred = model(x_test)
    
    print("\nTest predictions:")
    for i, x_val in enumerate(x_test):
        pred = y_test_pred[i].item()
        true = 3 * x_val.item() + 2
        print(f"  x = {x_val.item():.1f} → predicted: {pred:.2f}, true: {true:.2f}")

# ============================================================================
# 6부: 시각화
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 그림 1: 학습 손실
axes[0].plot(loss_history, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 그림 2: 데이터와 적합된 직선
axes[1].scatter(X.numpy(), y.numpy(), alpha=0.5, label='Training data')
x_line = torch.linspace(0, 10, 100).reshape(-1, 1)
with torch.no_grad():
    y_line = model(x_line)
axes[1].plot(x_line.numpy(), y_line.numpy(), 'r-', linewidth=2, 
             label=f'Fitted line (y={final_weight:.2f}x+{final_bias:.2f})')
axes[1].plot(x_line.numpy(), 3*x_line.numpy()+2, 'g--', linewidth=2, 
             label='True line (y=3x+2)', alpha=0.7)
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('y', fontsize=12)
axes[1].set_title('Linear Regression Fit', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 그림 3: 잔차(오차)
with torch.no_grad():
    y_pred_all = model(X)
    residuals = (y - y_pred_all).numpy()
axes[2].scatter(X.numpy(), residuals, alpha=0.5)
axes[2].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[2].set_xlabel('x', fontsize=12)
axes[2].set_ylabel('Residual (y_true - y_pred)', fontsize=12)
axes[2].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_1_basics/linear_regression.png', dpi=150)
print("\n✓ Plot saved as 'linear_regression.png'")
print("\nClose the plot window to continue...")
plt.show()

# ============================================================================
# 7부: 지표 계산
# ============================================================================
print("\n" + "="*80)
print("PART 7: PERFORMANCE METRICS")
print("="*80)

with torch.no_grad():
    y_pred_all = model(X)
    
    # 평균제곱오차(MSE)
    mse = criterion(y_pred_all, y).item()
    
    # 평균제곱근오차(RMSE)
    rmse = np.sqrt(mse)
    
    # 평균절대오차(MAE)
    mae = torch.mean(torch.abs(y_pred_all - y)).item()
    
    # 결정계수(R²)
    ss_res = torch.sum((y - y_pred_all) ** 2).item()
    ss_tot = torch.sum((y - torch.mean(y)) ** 2).item()
    r2_score = 1 - (ss_res / ss_tot)

print(f"MSE (Mean Squared Error):      {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error):     {mae:.4f}")
print(f"R² Score:                      {r2_score:.4f}")
print("\nInterpretation:")
print(f"  - RMSE of {rmse:.2f} means predictions are off by ±{rmse:.2f} on average")
print(f"  - R² of {r2_score:.4f} means model explains {r2_score*100:.2f}% of variance")

# ============================================================================
# 8부: 핵심 요점
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. nn.Module이 모델을 매기는 여느 방식이다
   - nn.Module을 물려받는다
   - __init__()에서 층을 매긴다
   - forward() 방법을 짠다

2. 학습 루프에는 여느 걸음 다섯이 있다.
   1. 순전파:      y_pred = model(X)
   2. 손실 계산:      loss = criterion(y_pred, y)
   3. 기울기 0으로:    optimizer.zero_grad()
   4. 역전파:     loss.backward()
   5. 매개변수 고치기: optimizer.step()

3. 따질 때는 model.eval()과 torch.no_grad()을 써라
   - 어떤 층(드롭아웃, 배치 정규화)을 끈다
   - 기울기를 좇지 않아 기억 자리를 아낀다

4. 여러 자를 쓰면 더 잘 따질 수 있다
   - MSE: 큰 오차을 더 크게 벌한다
   - MAE: 모든 오차을 똑같이 다룬다
   - R²: 설명된 분산의 몫을 잰다

5. 잔차 그림이 탈을 짚어내는 데 도움이 된다
   - 마구잡이로 흩어져 있으면 잘 맞은 것이다
   - 무늬가 보이면 비선형 모델이 필요할 수 있다
""")

# ============================================================================
# 9부: 해 볼 만한 실험
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENTS TO TRY")
print("="*80)
print("""
1. 여러 학습률:
   - Try lr = 0.001, 0.1, 0.5
   - 모여드는 속도와 든든함을 살펴보아라

2. 학습 데이터 늘리기:
   - n_samples을 1000으로 키운다
   - 모델이 더 잘 맞는가?

3. 잡음 늘리기:
   - 데이터를 만들 때 잡음을 키운다
   - R² 점수에 어떤 영향을 주는가?

4. 여러 최적화기:
   - SGD을 Adam으로 갈음한다: torch.optim.Adam(...)
   - 모여드는 모습을 견준다

5. 모델 저장하고 불러오기:
   - torch.save(model.state_dict(), 'model.pth')
   - model.load_state_dict(torch.load('model.pth'))
""")
print("="*80)


if __name__ == "__main__":
    pass
```

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

**다룬 것** — PyTorch로 만드는 간단한 선형 회귀

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다.

핵심 클래스는 `LinearRegressionModel`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
