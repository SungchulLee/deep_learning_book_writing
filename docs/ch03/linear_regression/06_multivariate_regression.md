# 다변량 회귀

현실의 회귀 문제는 거의 언제나 여러 개의 입력 특징을 포함한다. 이 튜토리얼은 California Housing 데이터셋을 사용해 선형 회귀를 다변량 경우로 확장하며, 특징 표준화와 학습/시험 분할 같은 필수 전처리 단계를 다룬다. `nn.Linear(n_features, 1)` 층은 여러 입력을 자연스럽게 처리하고, 특징들의 규모가 다를 때는 Adam 최적화기가 SGD보다 나은 경우가 많다.

## 코드

```python
"""
==============================================================================
06_multivariate_regression.py
==============================================================================
DIFFICULTY: ⭐⭐⭐ (Intermediate)

DESCRIPTION:
    Linear regression with multiple input features (multivariate).
    Uses California housing dataset for real-world example.

TOPICS COVERED:
    - Multiple input features
    - Real-world dataset
    - Feature scaling/normalization
    - Train/test split
    - Model evaluation metrics

PREREQUISITES:
    - Tutorial 05 (nn.Module)

TIME: ~25 minutes
==============================================================================
"""

import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("MULTIVARIATE LINEAR REGRESSION")
print("=" * 70)

# ============================================================================
# 1부: 실제 데이터셋 불러오기
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: LOAD CALIFORNIA HOUSING DATASET")
print("=" * 70)

# 데이터셋을 불러온다
housing = fetch_california_housing()
X_numpy = housing.data
y_numpy = housing.target

print(f"Dataset loaded:")
print(f"  Samples: {X_numpy.shape[0]}")
print(f"  Features: {X_numpy.shape[1]}")
print(f"\nFeature names:")
for i, name in enumerate(housing.feature_names):
    print(f"  {i}: {name}")

print(f"\nTarget: Median house value ($100k)")
print(f"  Min: ${y_numpy.min()*100:.1f}k")
print(f"  Max: ${y_numpy.max()*100:.1f}k")
print(f"  Mean: ${y_numpy.mean()*100:.1f}k")

# ============================================================================
# 2부: 데이터 전처리
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: DATA PREPROCESSING")
print("=" * 70)

# 학습 집합과 시험 집합으로 나눈다
X_train, X_test, y_train, y_test = train_test_split(
    X_numpy, y_numpy, test_size=0.2, random_state=42
)

print(f"Data split:")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")

# 특징 스케일링 (표준화)
# 중요: 스케일러는 학습 데이터에만 적합시킨다!
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"\nFeature scaling applied (StandardScaler)")
print(f"  Train X: mean≈0, std≈1")
print(f"  Train y: mean≈0, std≈1")

# PyTorch 텐서로 변환
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled).reshape(-1, 1)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test_scaled).reshape(-1, 1)

print(f"\nTensor shapes:")
print(f"  X_train: {X_train_t.shape}")
print(f"  y_train: {y_train_t.shape}")

# ============================================================================
# 3부: 모델 정의
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: DEFINE MULTIVARIATE MODEL")
print("=" * 70)

class MultiLinearRegression(nn.Module):
    def __init__(self, n_features):
        super(MultiLinearRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        
    def forward(self, x):
        return self.linear(x)

n_features = X_train_t.shape[1]
model = MultiLinearRegression(n_features)

print(f"Model created with {n_features} input features")
print(model)
print(f"\nParameter shapes:")
print(f"  Weight: {model.linear.weight.shape}")
print(f"  Bias: {model.linear.bias.shape}")

# ============================================================================
# 4부: 학습
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: TRAINING")
print("=" * 70)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

n_epochs = 200
train_losses = []
test_losses = []

print(f"Training for {n_epochs} epochs with Adam optimizer...")
print(f"\n{'Epoch':<8} {'Train Loss':<15} {'Test Loss':<15}")
print("-" * 45)

for epoch in range(n_epochs):
    # 학습
    model.train()
    y_pred_train = model(X_train_t)
    loss_train = criterion(y_pred_train, y_train_t)
    
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    
    # 평가
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t)
        loss_test = criterion(y_pred_test, y_test_t)
    
    train_losses.append(loss_train.item())
    test_losses.append(loss_test.item())
    
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {loss_train.item():<15.6f} {loss_test.item():<15.6f}")

print(f"\nTraining completed!")

# ============================================================================
# 5부: 평가
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: EVALUATION")
print("=" * 70)

model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_t)
    y_pred_test = model(X_test_t)

# 원래 규모로 되돌린다
y_pred_train_orig = scaler_y.inverse_transform(y_pred_train.numpy())
y_pred_test_orig = scaler_y.inverse_transform(y_pred_test.numpy())

# R² 점수를 계산한다
from sklearn.metrics import r2_score, mean_absolute_error

r2_train = r2_score(y_train, y_pred_train_orig)
r2_test = r2_score(y_test, y_pred_test_orig)
mae_train = mean_absolute_error(y_train, y_pred_train_orig)
mae_test = mean_absolute_error(y_test, y_pred_test_orig)

print(f"Model Performance:")
print(f"  Train R²: {r2_train:.4f}")
print(f"  Test R²: {r2_test:.4f}")
print(f"  Train MAE: ${mae_train*100:.2f}k")
print(f"  Test MAE: ${mae_test*100:.2f}k")

print(f"\nFeature Importance (absolute weights):")
weights = model.linear.weight.detach().numpy().flatten()
for i, (name, weight) in enumerate(zip(housing.feature_names, weights)):
    print(f"  {name:20s}: {abs(weight):8.4f}")

# ============================================================================
# 6부: 시각화
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 손실 곡선들
axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0, 0].plot(test_losses, label='Test Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Training History')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 예측 대 실제 (시험 집합)
axes[0, 1].scatter(y_test, y_pred_test_orig, alpha=0.5, s=10)
axes[0, 1].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Price ($100k)')
axes[0, 1].set_ylabel('Predicted Price ($100k)')
axes[0, 1].set_title(f'Predictions vs Actual (Test R²={r2_test:.4f})')
axes[0, 1].grid(True, alpha=0.3)

# 잔차
residuals = y_test - y_pred_test_orig.flatten()
axes[1, 0].scatter(y_pred_test_orig, residuals, alpha=0.5, s=10)
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Price ($100k)')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residual Plot')
axes[1, 0].grid(True, alpha=0.3)

# 특징 중요도
axes[1, 1].barh(housing.feature_names, np.abs(weights))
axes[1, 1].set_xlabel('Absolute Weight')
axes[1, 1].set_title('Feature Importance')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/06_multivariate_results.png', dpi=100)
print("\nSaved visualization")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key Points for Multivariate Regression:

1. Multiple Features: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

2. Feature Scaling is Critical:
   - Features on different scales can cause training issues
   - StandardScaler: (x - mean) / std
   - Always fit on training data only!

3. Train/Test Split:
   - Evaluate on unseen data
   - Prevents overfitting assessment

4. Adam Optimizer:
   - Often works better than SGD for multivariate
   - Adaptive learning rates per parameter

5. Evaluation Metrics:
   - R²: Proportion of variance explained (1.0 is perfect)
   - MAE: Mean Absolute Error (interpretable)

Next: Tutorial 07 - Polynomial Regression!
""")


if __name__ == "__main__":
    pass
```

## 논의

다변량 선형 모델 $y = w_1 x_1 + w_2 x_2 + \cdots + w_p x_p + b$는 `nn.Linear(p, 1)` 층 하나로 간결하게 표현되며, 이 층은 모양이 $(1, p)$인 가중치 행렬과 스칼라 편향을 저장한다. 순전파는 행렬 곱 한 번으로 $\hat{y} = X W^T + b$를 계산하므로 특징이 수백 개여도 효율적이다. 단변량 경우와 비교해 손실 함수나 학습 루프를 바꿀 필요가 없다.

평균을 빼고 표준편차로 나누는 특징 표준화는 다변량 회귀에서 결정적으로 중요하다. 표준화하지 않으면 수치 범위가 큰 특징이 경사 신호를 지배하여 수렴이 느려지거나 불안정해진다. `StandardScaler`는 학습 데이터에만 적합시켜야 한다. 학습 통계량을 써서 시험 데이터에 적용하면 정보 누출을 막고 시험 시점에도 모델이 같은 분포를 보게 된다.

MSE 외의 평가 지표는 추가적인 통찰을 준다. $R^2$ 점수는 설명된 분산의 비율을 재며 0(평균을 예측하는 것보다 낫지 않음)에서 1(완벽한 예측) 사이의 값을 갖는다. 평균절대오차(MAE)는 목표값과 단위가 같아서 MSE보다 해석하기 쉽다. 학습된 가중치의 크기를 살펴보면 모델이 어떤 특징을 가장 중요하게 여기는지 알 수 있으며, 이는 사실상 간단한 형태의 특징 중요도 분석이 된다.

## 연습문제

**Exercise 1.**
특징 표준화 단계를 없애고 모델을 다시 학습시켜라. 최종 시험 $R^2$과 수렴에 필요한 에폭 수를 표준화한 버전과 비교하라.

??? success "Solution to Exercise 1"
    표준화하지 않으면 중위 소득(범위 약 0.5-15)과 인구(범위 약 3-35000) 같은 특징의 규모가 크게 달라진다. 규모가 큰 특징의 경사 기여가 지배하기 때문에 모델은 훨씬 느리게 수렴하거나 아예 발산할 수 있다. 같은 에폭 수를 돌린 뒤의 $R^2$은 눈에 띄게 나쁠 것이다. 이는 경사 기반 최적화에서 표준화가 왜 필수인지를 보여준다.

---

**Exercise 2.**
학습된 가중치 벡터를 살펴보고 8개의 특징을 가중치의 절댓값 크기로 순위 매겨라. 모델은 집값을 예측하는 데 어떤 특징을 가장 중요하게 여기는가?

??? success "Solution to Exercise 2"
    ```python
    import numpy as np
    weights = model.linear.weight.detach().numpy().flatten()
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    ranked = sorted(zip(housing.feature_names, np.abs(weights)), key=lambda x: -x[1])
    for name, w in ranked:
        print(f'{name:20s}: {w:.4f}')
    # 대체로 MedInc(중위 소득)의 가중치가 가장 크다.
    ```

---

**Exercise 3.**
Adam 최적화기를 SGD로 바꾸고, 200 에폭 안에 비슷한 $R^2$을 얻도록 학습률을 조정하라. 어떤 학습률이 통하며, 손실 곡선은 Adam과 어떻게 다른가?

??? success "Solution to Exercise 3"
    ```python
    # 전형적인 결과: lr=0.01인 SGD는 매끄럽게 수렴하지만 Adam보다 느리다.
    # lr=0.1인 SGD는 진동할 수 있다. Adam은 매개변수마다 학습률을 조절하므로
    # 전역 lr을 어떻게 고르는지에 덜 민감하다.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # 200 에폭 동안 학습하고 손실 곡선을 비교한다.
    # SGD의 손실 곡선은 더 매끄럽지만 느리다. Adam은 적응적 이동 폭 덕분에
    # 초반 에폭에서 더 빠르게 수렴한다.
    ```
