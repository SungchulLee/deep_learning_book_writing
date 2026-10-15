# 회귀 손실의 비교

MSE, MAE와 SmoothL1을 같은 잔차에 적용해 비교한다. 손실값뿐 아니라 예측에 대한 경사, 목표 단위와 전환점을 함께 살핀다.

## 1. 코드

```python
"""
================================================================================
입문 02: 흔한 회귀 손실 함수
================================================================================

배울 내용:
- 회귀 과제를 위한 여러 손실 함수
- MSE, MAE, 후버 손실 중 무엇을 언제 쓸까
- 손실마다 이상점을 다루는 방식
- 손실 함수의 차이를 눈으로 보기

선수 지식:
- 01_intro_to_loss_functions.py를 마친다
- 기본적인 회귀 개념을 이해한다

소요 시간: 약 15분
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("COMMON REGRESSION LOSS FUNCTIONS")
print("=" * 80)

# ============================================================================
# 1절: 이상점이 있는 예시 데이터
# ============================================================================
print("\n" + "-" * 80)
print("SAMPLE DATA: Predicting Test Scores")
print("-" * 80)

# 실제 시험 점수
actual_scores = torch.tensor([85.0, 90.0, 88.0, 92.0, 15.0])  # 참고: 15는 이상점이다!
print(f"Actual scores: {actual_scores}")

# 모델의 예측 (이상점만 빼면 꽤 가깝다)
predicted_scores = torch.tensor([84.0, 89.0, 87.0, 91.0, 87.0])
print(f"Predicted scores: {predicted_scores}")

# 오차 보이기
errors = actual_scores - predicted_scores
print(f"Errors: {errors}")
print("\nNote: The 5th student has an error of -72 (a huge outlier!)")
print("Let's see how different loss functions handle this...")

# ============================================================================
# 2절: 평균제곱오차(MSE) - L2 손실
# ============================================================================
print("\n" + "-" * 80)
print("1. MEAN SQUARED ERROR (MSE) - L2 Loss")
print("-" * 80)

mse_criterion = nn.MSELoss()
mse_loss = mse_criterion(predicted_scores, actual_scores)

print(f"MSE Loss: {mse_loss.item():.4f}")
print(f"RMSE (Root MSE): {torch.sqrt(mse_loss).item():.4f}")

print("\nCHARACTERISTICS:")
print("✓ Most commonly used for regression")
print("✓ Differentiable everywhere (smooth gradients)")
print("✓ Sensitive to outliers (squares the error!)")
print(f"✓ Formula: (1/n) × Σ(predicted - actual)²")

# 개별 제곱 오차를 살펴보자
squared_errors = (predicted_scores - actual_scores) ** 2
print(f"\nSquared errors: {squared_errors}")
print(f"Notice how the outlier error {squared_errors[4].item():.0f} dominates!")
print(f"It's {(squared_errors[4] / squared_errors[:4].sum()).item():.1f}x larger than all others combined!")

# ============================================================================
# 3절: 평균절대오차(MAE) - L1 손실
# ============================================================================
print("\n" + "-" * 80)
print("2. MEAN ABSOLUTE ERROR (MAE) - L1 Loss")
print("-" * 80)

mae_criterion = nn.L1Loss()  # PyTorch에서 L1Loss가 MAE이다
mae_loss = mae_criterion(predicted_scores, actual_scores)

print(f"MAE Loss: {mae_loss.item():.4f}")

print("\nCHARACTERISTICS:")
print("✓ More robust to outliers than MSE")
print("✓ Less sensitive to large errors (doesn't square them)")
print("✗ Not differentiable at zero (can cause optimization issues)")
print(f"✓ Formula: (1/n) × Σ|predicted - actual|")

# 개별 절대 오차를 살펴보자
absolute_errors = torch.abs(predicted_scores - actual_scores)
print(f"\nAbsolute errors: {absolute_errors}")
print(f"The outlier contributes {absolute_errors[4].item():.0f}, but not as dramatically as MSE")

# ============================================================================
# 4절: 이상점에서 MSE와 MAE 견주기
# ============================================================================
print("\n" + "-" * 80)
print("COMPARISON: MSE vs MAE with Outlier")
print("-" * 80)

print(f"MSE Loss: {mse_loss.item():.4f}")
print(f"MAE Loss: {mae_loss.item():.4f}")

print("\nWithout the outlier:")
# 마지막 데이터 점(이상점)을 빼고 계산
mse_no_outlier = mse_criterion(predicted_scores[:4], actual_scores[:4])
mae_no_outlier = mae_criterion(predicted_scores[:4], actual_scores[:4])

print(f"MSE Loss (no outlier): {mse_no_outlier.item():.4f}")
print(f"MAE Loss (no outlier): {mae_no_outlier.item():.4f}")

print("\nImpact of the outlier:")
print(f"MSE increased by: {((mse_loss - mse_no_outlier) / mse_no_outlier * 100).item():.1f}%")
print(f"MAE increased by: {((mae_loss - mae_no_outlier) / mae_no_outlier * 100).item():.1f}%")
print("\n→ MSE is MUCH more sensitive to outliers!")

# ============================================================================
# 5절: 매끄러운 L1 손실 (후버 손실)
# ============================================================================
print("\n" + "-" * 80)
print("3. SMOOTH L1 LOSS (Huber Loss) - Best of Both Worlds")
print("-" * 80)

smooth_l1_criterion = nn.SmoothL1Loss()
smooth_l1_loss = smooth_l1_criterion(predicted_scores, actual_scores)

print(f"Smooth L1 Loss: {smooth_l1_loss.item():.4f}")

print("\nCHARACTERISTICS:")
print("✓ Combines benefits of MSE and MAE")
print("✓ Quadratic for small errors (like MSE)")
print("✓ Linear for large errors (like MAE)")
print("✓ More robust to outliers than MSE")
print("✓ Smoother gradients than MAE")

print("\nHOW IT WORKS:")
print("If |error| < 1: loss = 0.5 × error²  (MSE behavior)")
print("If |error| ≥ 1: loss = |error| - 0.5  (MAE behavior)")

# 각 오차가 어느 구간에 들어가는지 보이기
for i, error in enumerate(errors):
    abs_error = abs(error.item())
    regime = "MSE regime" if abs_error < 1 else "MAE regime"
    print(f"Error {i+1}: {error.item():6.1f} → {regime}")

# ============================================================================
# 6절: 알맞은 손실 함수 고르기
# ============================================================================
print("\n" + "-" * 80)
print("DECISION GUIDE: Which Loss Should You Use?")
print("-" * 80)

print("""
📊 평균 제곱 오차(MSE)를 쓸 때:
   ✓ 데이터가 깨끗하고 이상치가 적다
   ✓ 큰 오차에 큰 벌점을 주어야 한다
   ✓ 최적화를 위해 매끄러운 기울기를 원한다
   ✓ 예: 안정된 시장에서 집값 예측

📏 평균 절대 오차(MAE)를 쓸 때:
   ✓ 데이터에 이상치가 있다
   ✓ 모든 오차를 더 고르게 다루어야 한다
   ✓ 오차를 데이터와 같은 단위로 보고 싶다
   ✓ 예: 배달 시간 예측(교통 이상치가 흔하다)

🎯 Smooth L1 손실(후버)을 쓸 때:
   ✓ 이상치에 강건하기를 바란다
   ✓ 그러면서도 매끄러운 최적화를 원한다
   ✓ 이따금 이상값이 섞인 실제 데이터에 가장 알맞다
   ✓ 예: 객체 탐지의 경계 상자 회귀
""")

# ============================================================================
# 7절: 실전 예제 - 학습에 미치는 영향
# ============================================================================
print("\n" + "-" * 80)
print("PRACTICAL IMPACT: How Loss Choice Affects Training")
print("-" * 80)

# 기울기의 크기 모의실험 (모델이 얼마나 갱신될지)
# 간략하게 만들었지만 개념은 드러난다

print("When we have an outlier with error = 72:")
outlier_error = torch.tensor(72.0, requires_grad=True)

# MSE의 기울기
mse_loss_example = outlier_error ** 2 / 2  # 간략화함
mse_loss_example.backward()
print(f"MSE gradient magnitude: {abs(outlier_error.grad.item()):.1f}")

# MAE의 기울기
outlier_error.grad = None  # 기울기 초기화
mae_loss_example = torch.abs(outlier_error)
mae_loss_example.backward()
print(f"MAE gradient magnitude: {abs(outlier_error.grad.item()):.1f}")

print("\n→ MSE produces a gradient 72x larger for this outlier!")
print("→ This means the model will update much more aggressively")
print("→ Outliers can dominate training with MSE")

# ============================================================================
# 8절: 여러 상황으로 시험하기
# ============================================================================
print("\n" + "-" * 80)
print("EXPERIMENT: Different Data Scenarios")
print("-" * 80)

# 상황 1: 깨끗한 데이터 (이상점 없음)
clean_actual = torch.tensor([85.0, 90.0, 88.0, 92.0, 87.0])
clean_pred = torch.tensor([84.0, 89.0, 87.0, 91.0, 86.0])

# 상황 2: 오차가 보통인 데이터
moderate_actual = torch.tensor([85.0, 90.0, 88.0, 92.0, 87.0])
moderate_pred = torch.tensor([80.0, 85.0, 83.0, 87.0, 82.0])

# 상황 3: 이상점이 있는 데이터
outlier_actual = torch.tensor([85.0, 90.0, 88.0, 92.0, 15.0])
outlier_pred = torch.tensor([84.0, 89.0, 87.0, 91.0, 87.0])

scenarios = [
    ("Clean data (small errors)", clean_pred, clean_actual),
    ("Moderate errors", moderate_pred, moderate_actual),
    ("With outlier", outlier_pred, outlier_actual)
]

for name, pred, actual in scenarios:
    mse = F.mse_loss(pred, actual)
    mae = F.l1_loss(pred, actual)
    smooth_l1 = F.smooth_l1_loss(pred, actual)
    
    print(f"\n{name}:")
    print(f"  MSE:       {mse.item():6.2f}")
    print(f"  MAE:       {mae.item():6.2f}")
    print(f"  Smooth L1: {smooth_l1.item():6.2f}")

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. 손실 함수마다 오차를 다르게 다룬다.
   • MSE: 오차를 제곱한다 → 이상치에 매우 민감하다
   • MAE: 절대 오차 → 이상치에 강건하다
   • Smooth L1: 혼합형 → 두 방식의 장점을 함께 지닌다

2. 손실 선택은 학습에 영향을 준다.
   • MSE는 모델이 이상치에 무리하게 맞추도록 밀어붙인다
   • MAE는 모든 오차를 더 고르게 다룬다
   • Smooth L1은 둘 사이에서 균형을 잡는다

3. 데이터에 맞추어 고른다.
   • 깨끗한 데이터 → MSE
   • 잡음이나 이상치 → MAE 또는 Smooth L1
   • 범용 → Smooth L1

4. 세 손실 모두 미분 가능하며 PyTorch 자동 미분과 함께 쓸 수 있다

다음 단계:
→ 자신의 데이터로 시도해 보라
→ 이상치의 크기를 달리하여 실험해 보라
→ 분류 손실(교차 엔트로피 등)을 배워 보라
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

??? note "전체 출력 (161줄)"

    ```
    ================================================================================
    COMMON REGRESSION LOSS FUNCTIONS
    ================================================================================

    --------------------------------------------------------------------------------
    SAMPLE DATA: Predicting Test Scores
    --------------------------------------------------------------------------------
    Actual scores: tensor([85., 90., 88., 92., 15.])
    Predicted scores: tensor([84., 89., 87., 91., 87.])
    Errors: tensor([  1.,   1.,   1.,   1., -72.])

    Note: The 5th student has an error of -72 (a huge outlier!)
    Let's see how different loss functions handle this...

    --------------------------------------------------------------------------------
    1. MEAN SQUARED ERROR (MSE) - L2 Loss
    --------------------------------------------------------------------------------
    MSE Loss: 1037.6000
    RMSE (Root MSE): 32.2118

    CHARACTERISTICS:
    ✓ Most commonly used for regression
    ✓ Differentiable everywhere (smooth gradients)
    ✓ Sensitive to outliers (squares the error!)
    ✓ Formula: (1/n) × Σ(predicted - actual)²

    Squared errors: tensor([1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 5.1840e+03])
    Notice how the outlier error 5184 dominates!
    It's 1296.0x larger than all others combined!

    --------------------------------------------------------------------------------
    2. MEAN ABSOLUTE ERROR (MAE) - L1 Loss
    --------------------------------------------------------------------------------
    MAE Loss: 15.2000

    CHARACTERISTICS:
    ✓ More robust to outliers than MSE
    ✓ Less sensitive to large errors (doesn't square them)
    ✗ Not differentiable at zero (can cause optimization issues)
    ✓ Formula: (1/n) × Σ|predicted - actual|

    Absolute errors: tensor([ 1.,  1.,  1.,  1., 72.])
    The outlier contributes 72, but not as dramatically as MSE

    --------------------------------------------------------------------------------
    COMPARISON: MSE vs MAE with Outlier
    --------------------------------------------------------------------------------
    MSE Loss: 1037.6000
    MAE Loss: 15.2000

    Without the outlier:
    MSE Loss (no outlier): 1.0000
    MAE Loss (no outlier): 1.0000

    Impact of the outlier:
    MSE increased by: 103660.0%
    MAE increased by: 1420.0%

    → MSE is MUCH more sensitive to outliers!

    --------------------------------------------------------------------------------
    3. SMOOTH L1 LOSS (Huber Loss) - Best of Both Worlds
    --------------------------------------------------------------------------------
    Smooth L1 Loss: 14.7000

    CHARACTERISTICS:
    ✓ Combines benefits of MSE and MAE
    ✓ Quadratic for small errors (like MSE)
    ✓ Linear for large errors (like MAE)
    ✓ More robust to outliers than MSE
    ✓ Smoother gradients than MAE

    HOW IT WORKS:
    If |error| < 1: loss = 0.5 × error²  (MSE behavior)
    If |error| ≥ 1: loss = |error| - 0.5  (MAE behavior)
    Error 1:    1.0 → MAE regime
    Error 2:    1.0 → MAE regime
    Error 3:    1.0 → MAE regime
    Error 4:    1.0 → MAE regime
    Error 5:  -72.0 → MAE regime

    --------------------------------------------------------------------------------
    DECISION GUIDE: Which Loss Should You Use?
    --------------------------------------------------------------------------------

    📊 평균 제곱 오차(MSE)를 쓸 때:
       ✓ 데이터가 깨끗하고 이상치가 적다
       ✓ 큰 오차에 큰 벌점을 주어야 한다
       ✓ 최적화를 위해 매끄러운 기울기를 원한다
       ✓ 예: 안정된 시장에서 집값 예측

    📏 평균 절대 오차(MAE)를 쓸 때:
       ✓ 데이터에 이상치가 있다
       ✓ 모든 오차를 더 고르게 다루어야 한다
       ✓ 오차를 데이터와 같은 단위로 보고 싶다
       ✓ 예: 배달 시간 예측(교통 이상치가 흔하다)

    🎯 Smooth L1 손실(후버)을 쓸 때:
       ✓ 이상치에 강건하기를 바란다
       ✓ 그러면서도 매끄러운 최적화를 원한다
       ✓ 이따금 이상값이 섞인 실제 데이터에 가장 알맞다
       ✓ 예: 객체 탐지의 경계 상자 회귀


    --------------------------------------------------------------------------------
    PRACTICAL IMPACT: How Loss Choice Affects Training
    --------------------------------------------------------------------------------
    When we have an outlier with error = 72:
    MSE gradient magnitude: 72.0
    MAE gradient magnitude: 1.0

    → MSE produces a gradient 72x larger for this outlier!
    → This means the model will update much more aggressively
    → Outliers can dominate training with MSE

    --------------------------------------------------------------------------------
    EXPERIMENT: Different Data Scenarios
    --------------------------------------------------------------------------------

    Clean data (small errors):
      MSE:         1.00
      MAE:         1.00
      Smooth L1:   0.50

    Moderate errors:
      MSE:        25.00
      MAE:         5.00
      Smooth L1:   4.50

    With outlier:
      MSE:       1037.60
      MAE:        15.20
      Smooth L1:  14.70

    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. 손실 함수마다 오차를 다르게 다룬다.
       • MSE: 오차를 제곱한다 → 이상치에 매우 민감하다
       • MAE: 절대 오차 → 이상치에 강건하다
       • Smooth L1: 혼합형 → 두 방식의 장점을 함께 지닌다

    2. 손실 선택은 학습에 영향을 준다.
       • MSE는 모델이 이상치에 무리하게 맞추도록 밀어붙인다
       • MAE는 모든 오차를 더 고르게 다룬다
       • Smooth L1은 둘 사이에서 균형을 잡는다

    3. 데이터에 맞추어 고른다.
       • 깨끗한 데이터 → MSE
       • 잡음이나 이상치 → MAE 또는 Smooth L1
       • 범용 → Smooth L1

    4. 세 손실 모두 미분 가능하며 PyTorch 자동 미분과 함께 쓸 수 있다

    다음 단계:
    → 자신의 데이터로 시도해 보라
    → 이상치의 크기를 달리하여 실험해 보라
    → 분류 손실(교차 엔트로피 등)을 배워 보라

    ================================================================================
    ```


## 2. 논의

잔차 72의 제곱은 5184이고 절댓값은 72이다. 이것은 두 손실의 비용 정의와 단위가 다르다는 뜻이며 숫자가 더 작다는 이유로 손실을 고를 수는 없다.

MAE는 0이 아닌 잔차에서 예측에 대한 도함수 크기가 일정하지만, 매개변수 경사는 모델의 야코비안이 곱해진다. 따라서 모든 표본이 가중치를 같은 크기로 조정한다고 말할 수는 없다.

SmoothL1은 beta보다 작은 잔차에 이차, 큰 잔차에 선형 항을 쓴다. 양의 beta에서 Huber(delta=beta)를 beta로 나눈 값이므로 전환점과 전체 배율을 구분한다. 강건한 손실이 큰 입력 레버리지나 모든 학습 불안정을 해결하지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
잔차가 두 배일 때 세 손실의 큰 잔차 비용은 어떻게 변하는가?

</div>

??? success "연습문제 1 풀이"
    MSE는 네 배이다. MAE는 두 배이고 SmoothL1의 선형 구간에서는 기울기가 일정하지만 상수항이 있어 값이 정확히 두 배는 아닐 수 있다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
SmoothL1 beta가 목표 단위와 관계있는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    잔차가 beta와 같은 단위로 비교되므로 목표를 배율 변환하면 전환점도 고려해야 한다. 다른 손실과 조합할 때 전체 배율 차이도 반영한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
이상치가 있을 때 MAE를 선택하면 모든 영향을 제거하는가?

</div>

??? success "연습문제 3 풀이"
    아니다. 예측에 대한 도함수는 유계지만 입력이 크면 매개변수 경사가 커질 수 있다. 데이터 오류·목표 비용과 모델 구조를 함께 살펴야 한다.

## 정리하며

손실 비교는 잔차 비용과 도함수의 차이이다. 값의 단위, 전환점과 집계를 맞추고 목적에 맞는 검증 지표로 비교한다.
