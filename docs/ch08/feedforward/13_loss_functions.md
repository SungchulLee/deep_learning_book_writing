# 손실 함수

회귀와 분류의 대표 손실을 작은 텐서에 적용한다. 손실을 선택할 때는 과제뿐 아니라 입력이 로짓인지 확률인지, 목표의 모양과 자료형이 무엇인지 확인한다.

## 1. 코드

```python
"""
07_loss_functions.py - 알맞은 손실 함수 고르기

여러 손실 함수와 그것을 언제 쓰는지 배운다.
손실 함수를 이해하는 것이 학습 성공의 열쇠다!

소요 시간: 25~30분 | 난이도: ⭐⭐⭐☆☆
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

# ========================================================================
# 메인
# ========================================================================

print("="*70)
print("Loss Functions Guide")
print("="*70)

# 예시 예측값과 목푯값 만들기
y_true = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
y_pred = torch.tensor([0.1, 1.5, 1.8, 3.2, 3.9])

print("\nREGRESSION LOSSES")
print("-"*70)

# MSE 손실 (L2)
mse = nn.MSELoss()
print(f"MSE Loss: {mse(y_pred, y_true):.4f}")
print("  Formula: mean((pred - true)^2)")
print("  Use: Regression, penalizes large errors heavily")

# MAE 손실 (L1)
mae = nn.L1Loss()
print(f"\nMAE Loss: {mae(y_pred, y_true):.4f}")
print("  Formula: mean(|pred - true|)")
print("  Use: Regression with outliers, more robust than MSE")

# 매끄러운 L1 손실 (후버)
smooth_l1 = nn.SmoothL1Loss()
print(f"\nSmooth L1 Loss: {smooth_l1(y_pred, y_true):.4f}")
print("  Formula: Combines L1 and L2")
print("  Use: Robust to outliers, used in object detection")

print("\n" + "="*70)
print("CLASSIFICATION LOSSES")
print("-"*70)

# 이진 분류
logits_binary = torch.tensor([[0.8], [-0.5], [1.2], [-2.0]])
targets_binary = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

# 로짓을 쓰는 BCE (이진 분류에 권장)
bce_with_logits = nn.BCEWithLogitsLoss()
print(f"BCE with Logits: {bce_with_logits(logits_binary, targets_binary):.4f}")
print("  Use: Binary classification (includes sigmoid)")
print("  More numerically stable than BCE alone")

# 다중 클래스 분류
logits_multi = torch.randn(4, 3)  # 표본 4개, 클래스 3개
targets_multi = torch.tensor([0, 1, 2, 1])  # 클래스 인덱스

# 교차 엔트로피 (다중 클래스에 권장)
ce = nn.CrossEntropyLoss()
print(f"\nCross Entropy: {ce(logits_multi, targets_multi):.4f}")
print("  Use: Multi-class classification (includes log_softmax)")
print("  Most common for classification tasks")

print("\n" + "="*70)
print("LOSS FUNCTION SELECTION GUIDE")
print("="*70)
print("""
문제 유형             | 권장 손실             | 출력 활성화
----------------------|----------------------|-------------------
회귀                  | MSELoss              | 없음(선형)
회귀(이상치 있음)    | L1Loss / SmoothL1    | 없음
이진 분류             | BCEWithLogitsLoss    | 없음(로짓)
다중 클래스           | CrossEntropyLoss     | 없음(로짓)
다중 레이블           | BCEWithLogitsLoss    | 없음(로짓)

핵심:
✓ *WithLogits 판을 쓰라. 더 안정적이다
✓ 이 손실 앞에 활성화를 걸지 마라
✓ CrossEntropyLoss는 원-핫이 아니라 클래스 인덱스를 받는다
✓ MSE는 회귀에, 교차 엔트로피는 분류에 알맞다
✓ L1은 MSE보다 이상치에 강건하다

흔한 실수:
✗ BCEWithLogitsLoss 앞에 시그모이드를 걸기
✗ CrossEntropyLoss 앞에 소프트맥스를 걸기
✗ 분류에 MSE 쓰기
✗ 회귀에 교차 엔트로피 쓰기
""")

# 손실의 거동 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# L1과 L2의 거동 비교
errors = np.linspace(-3, 3, 100)
l1_loss = np.abs(errors)
l2_loss = errors**2

ax1.plot(errors, l1_loss, label='L1 (MAE)', linewidth=2)
ax1.plot(errors, l2_loss, label='L2 (MSE)', linewidth=2)
ax1.set_xlabel('Prediction Error', fontsize=12)
ax1.set_ylabel('Loss Value', fontsize=12)
ax1.set_title('L1 vs L2 Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 교차 엔트로피 시각화
probs = np.linspace(0.01, 0.99, 100)
ce_loss = -np.log(probs)

ax2.plot(probs, ce_loss, linewidth=2, color='red')
ax2.set_xlabel('Predicted Probability (for true class)', fontsize=12)
ax2.set_ylabel('Cross Entropy Loss', fontsize=12)
ax2.set_title('Cross Entropy Loss Behavior', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('07_loss_functions.png', dpi=150)
print("\nLoss visualizations saved!")

print("\nEXERCISES:")
print("1. Implement custom loss function for imbalanced data")
print("2. Compare MSE vs MAE on dataset with outliers")
print("3. Create weighted cross entropy for class imbalance")
print("4. Visualize gradient magnitude for different losses")
plt.show()


if __name__ == "__main__":
    pass
```

**출력:**

```
======================================================================
Loss Functions Guide
======================================================================

REGRESSION LOSSES
----------------------------------------------------------------------
MSE Loss: 0.0700
  Formula: mean((pred - true)^2)
  Use: Regression, penalizes large errors heavily

MAE Loss: 0.2200
  Formula: mean(|pred - true|)
  Use: Regression with outliers, more robust than MSE

Smooth L1 Loss: 0.0350
  Formula: Combines L1 and L2
  Use: Robust to outliers, used in object detection

======================================================================
CLASSIFICATION LOSSES
----------------------------------------------------------------------
BCE with Logits: 0.3088
  Use: Binary classification (includes sigmoid)
  More numerically stable than BCE alone

Cross Entropy: 1.4412
  Use: Multi-class classification (includes log_softmax)
  Most common for classification tasks

======================================================================
LOSS FUNCTION SELECTION GUIDE
======================================================================

문제 유형             | 권장 손실             | 출력 활성화
----------------------|----------------------|-------------------
회귀                  | MSELoss              | 없음(선형)
회귀(이상치 있음)    | L1Loss / SmoothL1    | 없음
이진 분류             | BCEWithLogitsLoss    | 없음(로짓)
다중 클래스           | CrossEntropyLoss     | 없음(로짓)
다중 레이블           | BCEWithLogitsLoss    | 없음(로짓)

핵심:
✓ *WithLogits 판을 쓰라. 더 안정적이다
✓ 이 손실 앞에 활성화를 걸지 마라
✓ CrossEntropyLoss는 원-핫이 아니라 클래스 인덱스를 받는다
✓ MSE는 회귀에, 교차 엔트로피는 분류에 알맞다
✓ L1은 MSE보다 이상치에 강건하다

흔한 실수:
✗ BCEWithLogitsLoss 앞에 시그모이드를 걸기
✗ CrossEntropyLoss 앞에 소프트맥스를 걸기
✗ 분류에 MSE 쓰기
✗ 회귀에 교차 엔트로피 쓰기


Loss visualizations saved!

EXERCISES:
1. Implement custom loss function for imbalanced data
2. Compare MSE vs MAE on dataset with outliers
3. Create weighted cross entropy for class imbalance
4. Visualize gradient magnitude for different losses
```

## 2. 논의

MSE는 큰 잔차를 제곱하여 강조하고, MAE는 절댓값을 사용한다. SmoothL1은 작은 오차와 큰 오차를 다르게 처리한다. SmoothL1과 Huber는 매개변수에 따라 배율이 다르므로 이름만으로 같은 수치를 기대하지 않는다.

이진 분류의 `BCEWithLogitsLoss`는 로짓을, `BCELoss`는 확률을 받는다. 다중 클래스 `CrossEntropyLoss`는 로짓을 받으며, 클래스 인덱스 목표는 보통 정수 long 텐서이다. 확률 목표를 쓰는 경우에는 로짓과 같은 모양의 부동소수점 텐서를 전달한다.

손실의 절댓값은 단위와 집계 방식에도 달려 있다. 서로 다른 손실값이 작다는 이유만으로 같은 의미의 예측이 더 좋다고 결론 내리지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이진 분류에서 BCEWithLogitsLoss 앞에 시그모이드를 다시 적용하면 왜 문제가 되는가?

</div>

??? success "연습문제 1 풀이"
    이 손실은 입력을 로짓으로 해석하여 내부에서 시그모이드와 로그 항을 안정적으로 계산한다. 확률을 넣으면 다시 변환되어 의도한 확률과 손실이 달라진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
CrossEntropyLoss에 클래스 인덱스와 확률 목표를 줄 때 필요한 모양과 자료형을 설명하라.

</div>

??? success "연습문제 2 풀이"
    분류 로짓이 `(B, K)`이면 클래스 인덱스 목표는 `(B,)`의 long 텐서이다. 확률 목표는 `(B, K)`의 부동소수점 텐서이며 각 행은 유효한 분포여야 한다. 정수 원핫 텐서를 그대로 전달하는 방식과 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
같은 예측에서 MSE와 MAE 값을 직접 비교해 더 좋은 손실을 고를 수 있는가?

</div>

??? success "연습문제 3 풀이"
    두 값은 잔차에 적용하는 함수와 단위가 다르다. 예측 목적, 큰 오차에 부여할 비용, 데이터 오염 가능성과 검증 지표를 기준으로 선택한다. 수치 크기만으로 선택하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_loss functions():
        model = Loss Functions(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 손실이나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

손실과 출력의 입력 규약을 맞추는 것이 우선이다. 회귀에서는 잔차의 비용과 단위를, 분류에서는 로짓·확률과 목표 표현을 확인한다. 손실 선택은 검증 목적에 맞춰 판단한다.
