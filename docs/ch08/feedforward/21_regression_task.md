# 회귀 과제

California Housing 데이터의 연속 목표값을 MLP로 예측한다. 학습 데이터로만 입력 표준화 통계량을 구하고, 항등 출력과 MSE를 사용한다.

## 1. 코드

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
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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

# 피처 정규화 (회귀에서 중요하다!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PyTorch 텐서로 바꾸기
X_train = torch.FloatTensor(X_train)
# reshape(-1, 1)이 반드시 필요하다. 모델의 출력이 (N, 1)인데 목표가
# (N,)이면 MSELoss가 오류를 내지 않고 방송을 해 버려, (N, N) 크기의
# 차이 행렬을 만들고 그 평균을 로스로 내놓는다. 값이 조용히 틀리므로
# 알아채기 어렵다. 분류에서 쓰는 LongTensor 인덱스와 달리 회귀의
# 목표는 실수이고 모양까지 맞추어 주어야 한다.
# 목표를 표준화하지 않은 점도 눈여겨보라. 집값이 10만 달러 단위라
# 이미 0.15~5 정도의 얌전한 범위이기 때문이며, 그래서 아래에서
# RMSE에 100000을 곱해 달러로 되돌릴 수 있다
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

print(f"\nTrain samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

print("\n" + "="*80)
print("Regression Model")
print("="*80)

class RegressionNet(nn.Module):
    """회귀를 위한 뉴럴 넷트웍."""
    
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 이 레이어 뒤에는 드롭아웃이 없다. 폭이 16까지 좁아진 데다
            # 출력이 바로 뒤라, 여기서 뉴런을 끄면 예측이 크게 흔들린다
            nn.Linear(32, 16),
            nn.ReLU(),
            
            # 활성화가 붙지 않는다. 회귀의 출력은 어떤 실수든 될 수
            # 있어야 하는데, ReLU를 걸면 음수를 못 내고 시그모이드를
            # 걸면 [0, 1]에 갇힌다
            nn.Linear(16, 1)  # 회귀를 위한 단일 출력
        )
    
    def forward(self, x):
        return self.network(x)

model = RegressionNet(X_train.shape[1])

# 회귀를 위한 MSE 로스.
# 교차 엔트로피는 확률 분포를 견주는 로스라 여기에 쓸 수 없다.
# MSE는 오차를 제곱하므로 크게 빗나간 표본에 훨씬 큰 벌점을 준다.
# 이상치가 많은 데이터라면 그 성질이 독이 되므로, 오차에 비례하는
# L1Loss나 둘을 섞은 SmoothL1Loss를 쓰기도 한다
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
    
    # 제곱근을 취하면 단위가 목표와 같아진다. MSE는 "달러의 제곱"이라
    # 크기를 가늠할 수 없지만, RMSE는 달러라 아래에서 100000을 곱해
    # 곧바로 "평균 몇 달러쯤 빗나가는가"로 읽을 수 있다
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

print(f"\nFinal Results:")
print(f"Train RMSE: ${train_rmse*100000:.2f}")
print(f"Test RMSE: ${test_rmse*100000:.2f}")

# 시각화
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 학습 로스
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
회귀의 피처:
✓ MSELoss나 L1Loss를 쓴다(교차 엔트로피가 아니다!)
✓ 출력 뉴런 하나(엑티베이션 함수 없음)
✓ 입력 피처를 정규화하거나 표준화하라
✓ RMSE, MAE, R² 점수로 평가하라

로스 함수:
- MSELoss: 큰 오차에 더 큰 벌점을 준다
- L1Loss: 이상치에 더 강건하다
- SmoothL1Loss: 혼합형 접근

METRICS:
- MSE: 평균 제곱 오차
- RMSE: MSE의 제곱근(목표와 단위가 같다)
- MAE: 평균 절대 오차
- R²: Coefficient of determination (0-1)

TIPS:
- 회귀에서는 늘 피처를 정규화하라
- 데이터에 이상치가 있는지 확인하라
- 예측과 실제를 견주어 시각화하라
- 잔차의 분포를 분석하라
""")
plt.show()


if __name__ == "__main__":
    pass
```

**출력:**

```
================================================================================
Regression Task: California Housing Prices
================================================================================
Features: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
Samples: 20640
Features per sample: 8
Target: Median house value (in $100,000s)

Train samples: 16512
Test samples: 4128

================================================================================
Regression Model
================================================================================
Model created!
Loss: MSE (Mean Squared Error)

================================================================================
Training...
================================================================================
Epoch [ 20/200] Loss: 0.3538
Epoch [ 40/200] Loss: 0.3166
Epoch [ 60/200] Loss: 0.2956
Epoch [ 80/200] Loss: 0.2806
Epoch [100/200] Loss: 0.2801
Epoch [120/200] Loss: 0.2753
Epoch [140/200] Loss: 0.2700
Epoch [160/200] Loss: 0.2672
Epoch [180/200] Loss: 0.2684
Epoch [200/200] Loss: 0.2685

Final Results:
Train RMSE: $47604.32
Test RMSE: $51187.30

Plots saved!

================================================================================
KEY POINTS FOR REGRESSION
================================================================================

회귀의 피처:
✓ MSELoss나 L1Loss를 쓴다(교차 엔트로피가 아니다!)
✓ 출력 뉴런 하나(엑티베이션 함수 없음)
✓ 입력 피처를 정규화하거나 표준화하라
✓ RMSE, MAE, R² 점수로 평가하라

로스 함수:
- MSELoss: 큰 오차에 더 큰 벌점을 준다
- L1Loss: 이상치에 더 강건하다
- SmoothL1Loss: 혼합형 접근

METRICS:
- MSE: 평균 제곱 오차
- RMSE: MSE의 제곱근(목표와 단위가 같다)
- MAE: 평균 절대 오차
- R²: Coefficient of determination (0-1)

TIPS:
- 회귀에서는 늘 피처를 정규화하라
- 데이터에 이상치가 있는지 확인하라
- 예측과 실제를 견주어 시각화하라
- 잔차의 분포를 분석하라

```

## 2. 논의

RegressionNet은 8→64→32→16→1 구조이며 마지막 레이어 뒤에 시그모이드가 없다. 목표값의 가능한 범위를 모델 출력에서 강제로 제한하지 않는 회귀 설계이다.

StandardScaler는 학습 입력에 fit하고 테스트 입력에는 transform만 수행한다. 복원한 모델을 사용할 때도 같은 scaler가 필요하다. 모델 웨이트만으로 원래 입력의 전처리가 복원되지는 않는다.

MSE는 목표 단위의 제곱이고 RMSE와 MAE는 목표와 같은 단위이다. 이 데이터의 금액 환산은 데이터셋 목표의 단위를 반영한 것이다. 무작위 분할의 성능을 다른 시기나 지역에 대한 예측 성능으로 바로 일반화하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
입력 피처가 8개일 때 RegressionNet의 파라메터 수를 구하라.

</div>

??? success "연습문제 1 풀이"
    각 레이어는 576, 2080, 528, 17개이므로 총 3201개이다. 드롭아웃과 ReLU는 학습 가능한 계수를 추가하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
scaler를 전체 데이터에 fit하면 왜 문제가 되는가?

</div>

??? success "연습문제 2 풀이"
    테스트 입력의 평균과 분산이 학습 변환에 반영된다. 목표값을 쓰지 않더라도 평가 데이터의 분포가 새어 들어간다. 훈련 분할 안에서 fit하고 검증·테스트에는 같은 변환만 적용한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
RMSE와 R²를 함께 볼 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"
    RMSE는 목표 단위의 오차 크기를 나타낸다. R²는 해당 평가셋의 평균 예측 기준에 대한 상대적 제곱 오차를 나타내며 음수도 가능하다. 어느 지표도 그 자체로 다른 지역·시기의 성능을 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

회귀에서는 출력과 목표의 단위를 일치시키고 전처리를 학습 데이터에만 맞춘다. MSE, RMSE와 R²는 다른 의미의 지표이며 모델 복원에는 전처리 상태도 필요하다.
