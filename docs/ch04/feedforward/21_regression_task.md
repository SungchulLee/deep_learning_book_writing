# 회귀 과제

13_regression_task.py - 연속적인 값 예측하기. 다음을 사용하여 집값을 예측하는 회귀 모델을 만든다

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 코드

```python
"""
13_regression_task.py - 연속값 예측하기

California Housing 데이터셋을 써서 집값을 예측하는
회귀 모델을 만든다.

회귀와 분류:
- 회귀: 연속값을 예측한다 (가격, 온도, 나이)
- 분류: 이산적인 범주를 예측한다 (고양이/개, 예/아니오)

소요 시간: 30~35분 | 난이도: ⭐⭐⭐☆☆
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# ========================================================================
# 메인
# ========================================================================

print("="*80)
print("Regression Task: California Housing Prices")
print("="*80)

# California Housing 데이터셋 불러오기
housing = fetch_california_housing()
X, y = housing.data, housing.target

print(f"Features: {housing.feature_names}")
print(f"Samples: {X.shape[0]}")
print(f"Features per sample: {X.shape[1]}")
print(f"Target: Median house value (in $100,000s)")

# 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 특징 정규화 (회귀에서 중요하다!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PyTorch 텐서로 바꾸기
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

print(f"\nTrain samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

print("\n" + "="*80)
print("Regression Model")
print("="*80)

class RegressionNet(nn.Module):
    """회귀를 위한 신경망."""
    
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1)  # 회귀를 위한 단일 출력
        )
    
    def forward(self, x):
        return self.network(x)

model = RegressionNet(X_train.shape[1])

# 회귀를 위한 MSE 손실
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model created!")
print("Loss: MSE (Mean Squared Error)")

# 학습
epochs = 200
batch_size = 64
losses = []

print("\n" + "="*80)
print("Training...")
print("="*80)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    # 미니배치 학습
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    losses.append(epoch_loss / (len(X_train) // batch_size))
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] Loss: {losses[-1]:.4f}")

# 평가
model.eval()
with torch.no_grad():
    train_pred = model(X_train)
    test_pred = model(X_test)
    
    train_mse = criterion(train_pred, y_train).item()
    test_mse = criterion(test_pred, y_test).item()
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

print(f"\nFinal Results:")
print(f"Train RMSE: ${train_rmse*100000:.2f}")
print(f"Test RMSE: ${test_rmse*100000:.2f}")

# 시각화
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 학습 손실
ax1.plot(losses, linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

# 예측값과 실제값 비교 (학습)
ax2.scatter(y_train, train_pred, alpha=0.5)
ax2.plot([y_train.min(), y_train.max()], 
         [y_train.min(), y_train.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Price')
ax2.set_ylabel('Predicted Price')
ax2.set_title('Training Set Predictions')
ax2.grid(True, alpha=0.3)

# 예측값과 실제값 비교 (시험)
ax3.scatter(y_test, test_pred, alpha=0.5, color='green')
ax3.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 'r--', lw=2)
ax3.set_xlabel('Actual Price')
ax3.set_ylabel('Predicted Price')
ax3.set_title('Test Set Predictions')
ax3.grid(True, alpha=0.3)

# 잔차
residuals = (test_pred - y_test).numpy()
ax4.hist(residuals, bins=50, edgecolor='black')
ax4.set_xlabel('Prediction Error')
ax4.set_ylabel('Frequency')
ax4.set_title('Residuals Distribution')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('13_regression_results.png', dpi=150)
print("\nPlots saved!")

print("\n" + "="*80)
print("KEY POINTS FOR REGRESSION")
print("="*80)
print("""
REGRESSION SPECIFICS:
✓ Use MSELoss or L1Loss (not CrossEntropy!)
✓ Single output neuron (no activation function)
✓ Normalize/standardize input features
✓ Evaluate with RMSE, MAE, R² score

LOSS FUNCTIONS:
- MSELoss: Penalizes large errors more
- L1Loss: More robust to outliers
- SmoothL1Loss: Hybrid approach

METRICS:
- MSE: Mean Squared Error
- RMSE: Root MSE (same units as target)
- MAE: Mean Absolute Error
- R²: Coefficient of determination (0-1)

TIPS:
- Always normalize features for regression
- Check for outliers in data
- Visualize predictions vs actual
- Analyze residuals distribution
""")
plt.show()


if __name__ == "__main__":
    pass```

## 논의

`RegressionNet` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `RegressionNet`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `RegressionNet`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = RegressionNet(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
