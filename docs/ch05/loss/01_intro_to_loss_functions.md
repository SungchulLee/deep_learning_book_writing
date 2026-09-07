# 손실 함수 소개

손실 함수는 모델의 예측이 참값에서 얼마나 벗어났는지를 재어 학습을 이끄는 신호를 준다. PyTorch는 손실을 계산하는 세 가지 방법을 제공한다. 직접 계산하기, 함수형 API(`F.mse_loss`), 손실 클래스(`nn.MSELoss`)이다. 손실 함수를 이해하는 것이 어떤 신경망을 학습시키든 그 바탕이 된다.

## 코드

```python
"""
================================================================================
입문 01: PyTorch의 손실 함수 소개
================================================================================

배울 내용:
- 손실 함수란 무엇이며 왜 필요한가
- PyTorch에서 손실을 계산하는 세 가지 방법
- 기본적인 회귀 손실 (평균제곱오차)
- 손실값을 해석하는 법

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
# 1절: 손실 함수란 무엇인가?
# ============================================================================
"""
손실 함수(비용 함수 또는 목적 함수라고도 한다)는 모델의 예측이 실제값에
견주어 얼마나 틀렸는지를 잰다.

이렇게 생각해 보라:
- 모델이 예측한다: "가격이 $100일 것 같다"
- 실제 가격은 $150이다
- 손실 함수가 말한다: "$50만큼 틀렸다!"

학습의 목표는 이 손실을 최소화하는 것이다.
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
# 3절: 방법 1 - 손실을 직접 계산하기
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
# 5절: 방법 3 - PyTorch 손실 클래스 쓰기 (학습에서 가장 흔하다)
# ============================================================================
print("\n" + "-" * 80)
print("METHOD 3: Using nn.MSELoss Class")
print("-" * 80)

# 손실 함수 객체 만들기
# 학습 루프에서 가장 흔히 쓰는 방식이다
criterion = nn.MSELoss()

# 이것으로 손실을 계산한다
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
# 7절: 좋은 손실과 나쁜 손실의 차이
# ============================================================================
print("\n" + "-" * 80)
print("INTERPRETING LOSS VALUES")
print("-" * 80)

# 완벽한 예측 (손실이 0이어야 한다)
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
1. 손실 함수는 예측이 얼마나 틀렸는지 잰다
2. 손실이 낮을수록 예측이 좋다(손실 0이면 완벽하다)
3. PyTorch에서 손실을 계산하는 세 가지 방법:
   - 직접 계산(학습용이나 맞춤 손실용)
   - F.mse_loss()(함수형 API, 빠르고 단순하다)
   - nn.MSELoss()(클래스 API, 학습 루프에 가장 알맞다)
4. MSE는 회귀 문제(연속값 예측)에 알맞다
5. 'reduction' 매개변수는 오차를 모으는 방식을 정한다

다음 단계:
→ predicted_prices를 바꾸어 손실이 어떻게 달라지는지 보라
→ 다른 손실 함수(MAE, 후버 등)를 배워 보라
→ 최적화기가 손실을 써서 모델을 어떻게 개선하는지 이해하라
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

평균제곱오차는 예측과 목푯값의 차이를 제곱하여 평균한다. 제곱은 두 가지 구실을 한다. 모든 오차를 양수로 만들어(그냥 차이를 쓰면 서로 상쇄될 수 있다) 주고, 큰 오차에 불균형하게 벌점을 주어 모델이 가장 나쁜 예측을 줄이는 데 집중하게 한다.

PyTorch는 MSE를 계산하는 동등한 세 가지 방법을 제공한다. 텐서 연산으로 직접 계산하기, 함수형 API `F.mse_loss()`, 클래스 기반 `nn.MSELoss()`이다. 학습 루프에서는 클래스 방식을 선호한다. 한 번 설정해 두고(줄이는 방식을 고른다) 계속 쓸 수 있고, 다른 손실 함수로 쉽게 바꿀 수 있기 때문이다.

`reduction` 매개변수는 표본별 손실을 어떻게 합칠지 정한다. `mean`은 평균을 내고(기본값), `sum`은 더하며, `none`은 표본별 손실을 그대로 돌려준다. `none`은 가중 손실을 쓰거나 어떤 표본이 전체 손실에 가장 많이 기여하는지 살펴볼 때 쓸모 있다.

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

