# 로스 함수 소개

MSE를 텐서 연산, 함수형 API와 로스 모듈로 계산해 비교한다. 같은 식과 집계 조건이면 값과 경사가 일치해야 한다.

## 1. 코드

```python
"""
================================================================================
입문 01: PyTorch의 로스 함수 소개
================================================================================

배울 내용:
- 로스 함수란 무엇이며 왜 필요한가
- PyTorch에서 로스를 계산하는 세 가지 방법
- 기본적인 회귀 로스 (평균제곱오차)
- 로스값을 해석하는 법

선수 지식:
- 기본적인 파이썬 지식
- PyTorch 텐서에 대한 기본 이해

소요 시간: 약 10분
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("INTRODUCTION TO LOSS FUNCTIONS")
print("=" * 80)

# ============================================================================
# 1절: 로스 함수란 무엇인가?
# ============================================================================
"""
로스 함수(비용 함수 또는 목적 함수라고도 한다)는 모델의 예측이 실제값에
견주어 얼마나 틀렸는지를 잰다.

이렇게 생각해 보라:
- 모델이 예측한다: "가격이 $100일 것 같다"
- 실제 가격은 $150이다
- 로스 함수가 말한다: "$50만큼 틀렸다!"

학습의 목표는 이 로스를 최소화하는 것이다.
"""

# ============================================================================
# 2절: 예시 데이터 - 집값 예측
# ============================================================================
print("\n" + "-" * 80)
print("SAMPLE DATA: House Size vs Price")
print("-" * 80)

# 실제 집값 (참값 / 목푯값)
# 실제 가격이 있는 집 5채가 있다고 하자
actual_prices = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])
print(f"Actual prices (in $1000s): {actual_prices}")

# 모델의 예측 (처음에는 틀린다!)
# 학습하지 않은 모델이 생각하는 가격이다
predicted_prices = torch.tensor([140.0, 210.0, 245.0, 310.0, 360.0])
print(f"Predicted prices (in $1000s): {predicted_prices}")

# 차이를 살펴보자
differences = actual_prices - predicted_prices
print(f"Differences (actual - predicted): {differences}")
print("Negative = model predicted too high, Positive = model predicted too low")

# ============================================================================
# 3절: 방법 1 - 로스를 직접 계산하기
# ============================================================================
print("\n" + "-" * 80)
print("METHOD 1: Manual Loss Calculation")
print("-" * 80)
print("We'll compute Mean Squared Error (MSE) step by step")

# 1단계: 집마다 차이(오차)를 계산한다
errors = actual_prices - predicted_prices
print(f"\nStep 1 - Errors: {errors}")

# 2단계: 오차를 제곱한다 (모든 오차가 양수가 되고 큰 오차에 더 큰 벌점이 간다)
squared_errors = errors ** 2
print(f"Step 2 - Squared errors: {squared_errors}")

# 3단계: 제곱 오차의 평균을 낸다
mse_manual = torch.mean(squared_errors)
print(f"Step 3 - Mean Squared Error: {mse_manual.item():.4f}")

print("\nWHAT THIS MEANS:")
print(f"On average, our predictions are off by about ${torch.sqrt(mse_manual).item():.2f}k")
print("(Square root of MSE gives us the Root Mean Squared Error)")

# ============================================================================
# 4절: 방법 2 - PyTorch 함수형 API 쓰기
# ============================================================================
print("\n" + "-" * 80)
print("METHOD 2: Using torch.nn.functional")
print("-" * 80)

# PyTorch는 MSE를 계산하는 내장 함수를 제공한다
# 더 편리하고 최적화되어 있다
mse_functional = F.mse_loss(predicted_prices, actual_prices)
print(f"MSE using F.mse_loss: {mse_functional.item():.4f}")

# 서로 같은지 확인
print(f"\nManual MSE == Functional MSE? {torch.allclose(mse_manual, mse_functional)}")

# ============================================================================
# 5절: 방법 3 - PyTorch 로스 클래스 쓰기 (학습에서 가장 흔하다)
# ============================================================================
print("\n" + "-" * 80)
print("METHOD 3: Using nn.MSELoss Class")
print("-" * 80)

# 로스 함수 객체 만들기
# 학습 루프에서 가장 흔히 쓰는 방식이다
criterion = nn.MSELoss()

# 이것으로 로스를 계산한다
mse_class = criterion(predicted_prices, actual_prices)
print(f"MSE using nn.MSELoss: {mse_class.item():.4f}")

print("\nWHY USE A CLASS?")
print("- You can configure it once (e.g., different reduction methods)")
print("- Cleaner code in training loops")
print("- Can easily swap different loss functions")

# ============================================================================
# 6절: 여러 줄이기 방식 이해하기
# ============================================================================
print("\n" + "-" * 80)
print("BONUS: Understanding 'reduction' Parameter")
print("-" * 80)

# reduction='mean': 모든 오차의 평균 (기본값)
criterion_mean = nn.MSELoss(reduction='mean')
loss_mean = criterion_mean(predicted_prices, actual_prices)
print(f"Reduction='mean': {loss_mean.item():.4f}")

# reduction='sum': 모든 오차의 합
criterion_sum = nn.MSELoss(reduction='sum')
loss_sum = criterion_sum(predicted_prices, actual_prices)
print(f"Reduction='sum': {loss_sum.item():.4f}")

# reduction='none': 표본마다의 개별 오차
criterion_none = nn.MSELoss(reduction='none')
loss_none = criterion_none(predicted_prices, actual_prices)
print(f"Reduction='none': {loss_none}")

print("\nNote: sum = mean × number_of_samples")
print(f"Verification: {loss_mean.item():.4f} × {len(actual_prices)} = {loss_sum.item():.4f}")

# ============================================================================
# 7절: 좋은 로스와 나쁜 로스의 차이
# ============================================================================
print("\n" + "-" * 80)
print("INTERPRETING LOSS VALUES")
print("-" * 80)

# 완벽한 예측 (로스가 0이어야 한다)
perfect_predictions = actual_prices.clone()
loss_perfect = criterion(perfect_predictions, actual_prices)
print(f"Perfect predictions → Loss: {loss_perfect.item():.4f}")

# 조금 나은 예측
better_predictions = torch.tensor([148.0, 202.0, 249.0, 301.0, 351.0])
loss_better = criterion(better_predictions, actual_prices)
print(f"Better predictions → Loss: {loss_better.item():.4f}")

# 더 나쁜 예측
worse_predictions = torch.tensor([120.0, 230.0, 220.0, 330.0, 380.0])
loss_worse = criterion(worse_predictions, actual_prices)
print(f"Worse predictions → Loss: {loss_worse.item():.4f}")

print("\nKEY INSIGHT: Lower loss = Better predictions!")

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. 로스 함수는 예측이 얼마나 틀렸는지 잰다
2. 로스가 낮을수록 예측이 좋다(로스 0이면 완벽하다)
3. PyTorch에서 로스를 계산하는 세 가지 방법:
   - 직접 계산(학습용이나 맞춤 로스용)
   - F.mse_loss()(함수형 API, 빠르고 단순하다)
   - nn.MSELoss()(클래스 API, 학습 루프에 가장 알맞다)
4. MSE는 회귀 문제(연속값 예측)에 알맞다
5. 'reduction' 파라메터는 오차를 모으는 방식을 정한다

다음 단계:
→ predicted_prices를 바꾸어 로스가 어떻게 달라지는지 보라
→ 다른 로스 함수(MAE, 후버 등)를 배워 보라
→ 최적화기가 로스를 써서 모델을 어떻게 개선하는지 이해하라
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

**출력:**

```
================================================================================
INTRODUCTION TO LOSS FUNCTIONS
================================================================================

--------------------------------------------------------------------------------
SAMPLE DATA: House Size vs Price
--------------------------------------------------------------------------------
Actual prices (in $1000s): tensor([150., 200., 250., 300., 350.])
Predicted prices (in $1000s): tensor([140., 210., 245., 310., 360.])
Differences (actual - predicted): tensor([ 10., -10.,   5., -10., -10.])
Negative = model predicted too high, Positive = model predicted too low

--------------------------------------------------------------------------------
METHOD 1: Manual Loss Calculation
--------------------------------------------------------------------------------
We'll compute Mean Squared Error (MSE) step by step

Step 1 - Errors: tensor([ 10., -10.,   5., -10., -10.])
Step 2 - Squared errors: tensor([100., 100.,  25., 100., 100.])
Step 3 - Mean Squared Error: 85.0000

WHAT THIS MEANS:
On average, our predictions are off by about $9.22k
(Square root of MSE gives us the Root Mean Squared Error)

--------------------------------------------------------------------------------
METHOD 2: Using torch.nn.functional
--------------------------------------------------------------------------------
MSE using F.mse_loss: 85.0000

Manual MSE == Functional MSE? True

--------------------------------------------------------------------------------
METHOD 3: Using nn.MSELoss Class
--------------------------------------------------------------------------------
MSE using nn.MSELoss: 85.0000

WHY USE A CLASS?
- You can configure it once (e.g., different reduction methods)
- Cleaner code in training loops
- Can easily swap different loss functions

--------------------------------------------------------------------------------
BONUS: Understanding 'reduction' Parameter
--------------------------------------------------------------------------------
Reduction='mean': 85.0000
Reduction='sum': 425.0000
Reduction='none': tensor([100., 100.,  25., 100., 100.])

Note: sum = mean × number_of_samples
Verification: 85.0000 × 5 = 425.0000

--------------------------------------------------------------------------------
INTERPRETING LOSS VALUES
--------------------------------------------------------------------------------
Perfect predictions → Loss: 0.0000
Better predictions → Loss: 2.2000
Worse predictions → Loss: 900.0000

KEY INSIGHT: Lower loss = Better predictions!

================================================================================
KEY TAKEAWAYS
================================================================================

1. 로스 함수는 예측이 얼마나 틀렸는지 잰다
2. 로스가 낮을수록 예측이 좋다(로스 0이면 완벽하다)
3. PyTorch에서 로스를 계산하는 세 가지 방법:
   - 직접 계산(학습용이나 맞춤 로스용)
   - F.mse_loss()(함수형 API, 빠르고 단순하다)
   - nn.MSELoss()(클래스 API, 학습 루프에 가장 알맞다)
4. MSE는 회귀 문제(연속값 예측)에 알맞다
5. 'reduction' 파라메터는 오차를 모으는 방식을 정한다

다음 단계:
→ predicted_prices를 바꾸어 로스가 어떻게 달라지는지 보라
→ 다른 로스 함수(MAE, 후버 등)를 배워 보라
→ 최적화기가 로스를 써서 모델을 어떻게 개선하는지 이해하라

================================================================================
```

## 2. 논의

MSE는 예측과 목표의 차이를 제곱해 음이 아닌 값으로 만든다. 원소별 오차의 양·음 상쇄를 피하고 큰 잔차를 더 강조한다. 큰 잔차가 항상 오류나 제거할 이상치라는 뜻은 아니다.

F.mse_loss와 nn.MSELoss는 같은 계산을 함수나 재사용 모듈로 표현한다. 둘 중 하나가 통계적으로 더 정확한 방식은 아니다. 설정과 코드 구성에 맞춰 고른다.

MSE의 reduction=none은 입력과 목표의 브로드캐스팅 결과에 따른 원소별 로스며, 벡터 출력에서는 표본당 하나가 아닐 수 있다. mean은 전체 원소로 나누고 sum은 더하므로 로스의 최소점과 갱신 배율을 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
오차가 -2와 2이면 평균 오차와 MSE는 얼마인가?

</div>

??? success "연습문제 1 풀이"
    평균 오차는 0이고 MSE는 4이다. 바이어스의 상쇄와 예측 오차의 크기는 다른 양이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
출력 `(B, D)`의 원소별 MSE와 표본별 MSE를 구분하라.

</div>

??? success "연습문제 2 풀이"
    none은 B×D개의 제곱 오차를 반환한다. 표본별 로스가 필요하면 D축을 먼저 집계하고 배치 B축의 집계를 별도로 정한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
로스가 벡터일 때 인수 없는 backward를 바로 호출할 수 있는가?

</div>

??? success "연습문제 3 풀이"
    일반적으로 스칼라로 집계하거나 출력과 같은 모양의 외부 경사 벡터를 전달해야 한다. 원하는 mean과 sum에 따라 배율을 명시한다.

## 정리하며

MSE 계산 방식은 같아도 집계 축과 입력 모양이 다르면 의미가 달라진다. 원소별·표본별 집계와 경사 스케일을 구분한다.
