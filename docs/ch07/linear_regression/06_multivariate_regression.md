# 여러 입력을 사용하는 선형 회귀

여러 입력 피처로 하나의 연속값을 예측하는 다중 선형 회귀를 California Housing 데이터로 살펴본다. 피처 표준화와 데이터 분할, 회귀 지표의 해석을 함께 다룬다. 여러 목표변수를 예측하는 다중 출력 회귀와 구분한다.

## 1. 코드

```python
"""
==============================================================================
06_multivariate_regression.py
==============================================================================
어려움: ⭐⭐⭐ (가운데)

DESCRIPTION:
    입력 피처가 여럿인 선형 회귀(다변량).
    참 세상 보기로 캘리포니아 집값 데이터셋을 쓴다.

다루는 것:
    - 여러 입력 피처
    - 참 세상 데이터셋
    - 피처 척도 잡기/정규화
    - 학습/시험 나누기
    - 모델 따짐 자

PREREQUISITES:
    - 튜토리얼 05(nn.Module)

걸리는 때: 25분쯤
==============================================================================
"""

import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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

# 피처 스케일링 (표준화)
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

# 로스 곡선들
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

# 피처 중요도
axes[1, 1].barh(housing.feature_names, np.abs(weights))
axes[1, 1].set_xlabel('Absolute Weight')
axes[1, 1].set_title('Feature Importance')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('06_multivariate_results.png', dpi=100)
print("\nSaved visualization")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
다변량 회귀의 핵심:

1. 여러 피처: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

2. 피처 척도 잡기가 종요롭다.
   - 잣대가 다른 피처는 학습에 탈을 낼 수 있다
   - StandardScaler: (x - mean) / std
   - 늘 학습 데이터에만 맞춰라!

3. 학습/시험 나누기:
   - 본 적 없는 데이터로 평가한다
   - 지나치게 맞춰졌는지 가늠할 수 있다

4. Adam 최적화기:
   - 다변량에서는 SGD보다 잘 듣는 일이 잦다
   - 파라메터마다 맞추어 가는 러닝레이트

5. 따짐 자:
   - R²: 설명된 분산의 몫(1.0이면 완벽하다)
   - MAE: 평균 절대 오차(뜻을 읽기 쉽다)

다음: 튜토리얼 07 - 다항 회귀!
""")


if __name__ == "__main__":
    pass
```

??? note "전체 출력 (122줄)"

    ```
    ======================================================================
    MULTIVARIATE LINEAR REGRESSION
    ======================================================================

    ======================================================================
    PART 1: LOAD CALIFORNIA HOUSING DATASET
    ======================================================================
    Dataset loaded:
      Samples: 20640
      Features: 8

    Feature names:
      0: MedInc
      1: HouseAge
      2: AveRooms
      3: AveBedrms
      4: Population
      5: AveOccup
      6: Latitude
      7: Longitude

    Target: Median house value ($100k)
      Min: $15.0k
      Max: $500.0k
      Mean: $206.9k

    ======================================================================
    PART 2: DATA PREPROCESSING
    ======================================================================
    Data split:
      Training samples: 16512
      Test samples: 4128

    Feature scaling applied (StandardScaler)
      Train X: mean≈0, std≈1
      Train y: mean≈0, std≈1

    Tensor shapes:
      X_train: torch.Size([16512, 8])
      y_train: torch.Size([16512, 1])

    ======================================================================
    PART 3: DEFINE MULTIVARIATE MODEL
    ======================================================================
    Model created with 8 input features
    MultiLinearRegression(
      (linear): Linear(in_features=8, out_features=1, bias=True)
    )

    Parameter shapes:
      Weight: torch.Size([1, 8])
      Bias: torch.Size([1])

    ======================================================================
    PART 4: TRAINING
    ======================================================================
    Training for 200 epochs with Adam optimizer...

    Epoch    Train Loss      Test Loss      
    ---------------------------------------------
    1        1.491943        1.123750       
    20       0.441741        0.437953       
    40       0.391823        0.416824       
    60       0.388718        0.414640       
    80       0.387528        0.415757       
    100      0.387456        0.416443       
    120      0.387450        0.415813       
    140      0.387449        0.415788       
    160      0.387449        0.415847       
    180      0.387449        0.415848       
    200      0.387449        0.415845       

    Training completed!

    ======================================================================
    PART 5: EVALUATION
    ======================================================================
    Model Performance:
      Train R²: 0.6126
      Test R²: 0.5758
      Train MAE: $52.86k
      Test MAE: $53.32k

    Feature Importance (absolute weights):
      MedInc              :   0.7390
      HouseAge            :   0.1060
      AveRooms            :   0.2546
      AveBedrms           :   0.2934
      Population          :   0.0020
      AveOccup            :   0.0353
      Latitude            :   0.7758
      Longitude           :   0.7523

    Saved visualization

    ======================================================================
    SUMMARY
    ======================================================================

    다변량 회귀의 핵심:

    1. 여러 피처: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

    2. 피처 척도 잡기가 종요롭다.
       - 잣대가 다른 피처는 학습에 탈을 낼 수 있다
       - StandardScaler: (x - mean) / std
       - 늘 학습 데이터에만 맞춰라!

    3. 학습/시험 나누기:
       - 본 적 없는 데이터로 평가한다
       - 지나치게 맞춰졌는지 가늠할 수 있다

    4. Adam 최적화기:
       - 다변량에서는 SGD보다 잘 듣는 일이 잦다
       - 파라메터마다 맞추어 가는 러닝레이트

    5. 따짐 자:
       - R²: 설명된 분산의 몫(1.0이면 완벽하다)
       - MAE: 평균 절대 오차(뜻을 읽기 쉽다)

    다음: 튜토리얼 07 - 다항 회귀!

    ```


## 2. 논의

다변량 선형 모델 $y = w_1 x_1 + w_2 x_2 + \cdots + w_p x_p + b$는 `nn.Linear(p, 1)` 레이어 하나로 간결하게 표현되며, 이 레이어는 모양이 $(1, p)$인 웨이트 행렬과 길이 1의 바이어스 텐서를 저장한다. 순전파는 행렬 곱 한 번으로 $\hat{y} = X W^T + b$를 계산하므로 피처가 수백 개여도 효율적이다. 단변량 경우와 비교해 로스 함수나 학습 루프를 바꿀 필요가 없다.

평균을 빼고 표준편차로 나누는 피처 표준화는 다변량 회귀에서 결정적으로 중요하다. 표준화하지 않으면 수치 범위가 큰 피처가 경사 신호를 지배하여 수렴이 느려지거나 불안정해진다. `StandardScaler`는 학습 데이터에만 적합시켜야 한다. 학습 통계량을 써서 시험 데이터에 적용하면 정보 누출을 막고 시험 데이터에 학습 때와 같은 변환을 적용한다. 변환 규칙이 같다고 데이터 분포 자체가 같아지는 것은 아니다.

$R^2=1-\text{SSE}/\text{SST}$는 정답이 상수가 아닐 때 평균 예측을 기준으로 잔차제곱합을 비교하는 지표다. 1이 최댓값이며 평균 예측보다 나쁘면 음수가 될 수 있다. MAE와 RMSE는 정답과 같은 단위를 가진다. 표준화한 피처의 계수 크기는 선형 모델의 조건부 관계를 살펴보는 단서지만, 상관된 피처와 파라미터 불안정성 때문에 중요도나 인과효과로 단정할 수는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
피처 표준화가 경사 기반 선형 회귀에 도움이 되는 이유와, 표준화 없이도 학습이 가능할 수 있는 조건을 설명하라.

</div>

??? success "연습문제 1 풀이"
    피처의 척도 차이가 크면 로스의 조건수가 나빠져 단일 러닝레이트로 수렴시키기 어려울 수 있다. 표준화는 이를 완화할 수 있지만 필수 수학 조건은 아니다. 다른 러닝레이트, 적절한 전처리나 해법으로도 학습할 수 있으며 실제 비교에서는 러닝레이트까지 조정해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
학습된 웨이트 벡터를 살펴보고 8개의 피처를 웨이트의 절댓값 크기로 순위 매겨라. 모델은 집값을 예측하는 데 어떤 피처를 가장 중요하게 여기는가?

</div>

??? success "연습문제 2 풀이"
    ```python
    import numpy as np
    weights = model.linear.weight.detach().numpy().flatten()
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    ranked = sorted(zip(housing.feature_names, np.abs(weights)), key=lambda x: -x[1])
    for name, w in ranked:
        print(f'{name:20s}: {w:.4f}')
    # 대체로 MedInc(중위 소득)의 웨이트가 가장 크다.
    ```

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Adam 옵티마이저를 SGD로 바꾸고, 200 에포크 안에 비슷한 $R^2$을 얻도록 러닝레이트를 조정하라. 어떤 러닝레이트가 통하며, 로스 곡선은 Adam과 어떻게 다른가?

</div>

??? success "연습문제 3 풀이"
    ```python
    # 전형적인 결과: lr=0.01인 SGD는 매끄럽게 수렴하지만 Adam보다 느리다.
    # lr=0.1인 SGD는 진동할 수 있다. Adam은 파라메터마다 러닝레이트를 조절하므로
    # 전역 lr을 어떻게 고르는지에 덜 민감하다.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # 200 에폭 동안 학습하고 로스 곡선을 비교한다.
    # SGD의 로스 곡선은 더 매끄럽지만 느리다. Adam은 적응적 이동 폭 덕분에
    # 초반 에폭에서 더 빠르게 수렴한다.
    ```

## 정리하며

피처의 척도와 상관이 최적화와 계수 해석에 영향을 준다. 전처리는 학습 데이터에만 적합하고 여러 지표로 새 데이터 성능을 평가한다.
