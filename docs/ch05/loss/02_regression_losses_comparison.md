# 회귀 손실의 비교

회귀 손실 함수마다 오차를 다루는 방식이 다르다. MSE는 오차를 제곱하여 이상점에 민감하고, MAE는 절댓값을 써서 더 튼튼하며, 매끄러운 L1(후버) 손실은 둘의 좋은 점을 결합한다. 어떤 손실을 고르느냐가 학습 중에 모델이 이상점에 얼마나 세게 반응할지를 곧바로 좌우한다.

## 코드

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
📊 USE MEAN SQUARED ERROR (MSE) when:
   ✓ You have clean data with few outliers
   ✓ Large errors should be penalized heavily
   ✓ You want smooth gradients for optimization
   ✓ Example: Predicting house prices in a stable market

📏 USE MEAN ABSOLUTE ERROR (MAE) when:
   ✓ Your data has outliers
   ✓ All errors should be treated more equally
   ✓ You want the error in the same units as your data
   ✓ Example: Predicting delivery times (traffic outliers common)

🎯 USE SMOOTH L1 LOSS (Huber) when:
   ✓ You want robustness to outliers
   ✓ But still want smooth optimization
   ✓ Best for real-world data with occasional anomalies
   ✓ Example: Object detection bounding box regression
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
1. Different loss functions handle errors differently:
   • MSE: Squares errors → very sensitive to outliers
   • MAE: Absolute errors → robust to outliers
   • Smooth L1: Hybrid → best of both worlds

2. Loss choice impacts training:
   • MSE pushes model to fit outliers aggressively
   • MAE treats all errors more equally
   • Smooth L1 balances between the two

3. Choose based on your data:
   • Clean data → MSE
   • Noisy/outliers → MAE or Smooth L1
   • General purpose → Smooth L1

4. All three losses are differentiable and work with PyTorch autograd

NEXT STEPS:
→ Try with your own data
→ Experiment with different outlier magnitudes
→ Learn about classification losses (CrossEntropy, etc.)
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

MSE 손실은 오차를 제곱하므로 큰 오차를 크게 부풀린다. 이상점의 오차가 72인 예에서 제곱 오차는 5,184가 되어 전체 손실을 압도하며, 모델이 나머지 데이터를 잘 맞추기보다 이 이상점 하나를 줄이는 데 매달리게 만든다. 모든 데이터 점을 믿을 수 있을 때에는 바람직하지만 이상점이 잡음일 때에는 문제가 된다.

MAE 손실은 절댓값을 써서 크기와 상관없이 모든 오차를 같은 비율로 다룬다. MAE의 기울기는 상수($\pm 1$)이므로 오차가 1이든 72이든 모델이 같은 만큼 조정된다. 덕분에 학습이 이상점에 더 튼튼해지지만, 기울기가 정의되지 않는 0 근처에서 최적화가 어려워질 수 있다.

매끄러운 L1(후버) 손실은 둘의 좋은 점을 결합한다. 작은 오차에서는 MSE처럼(0 근처에서 매끄러운 기울기) 움직이고 큰 오차에서는 MAE처럼(선형으로 늘고 기울기가 상수) 움직인다. 전환점(기본값은 오차 = 1)은 `beta` 매개변수로 조정하여 이상점에 대한 민감도를 조절할 수 있다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

