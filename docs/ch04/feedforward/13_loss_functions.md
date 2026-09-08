# 손실 함수

07_loss_functions.py - 알맞은 손실 함수 고르기. 여러 손실 함수와 각각을 언제 쓰는지 배운다.

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

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

## 2. 논의

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심적인 설계 결정을 찾아내라. 구체적인 구현 선택 세 가지를 나열하고, 각각이 순방향 신경망에 왜 적절한지 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
손실 함수 구현을 검증하는 종합적인 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)을 가진 입력 등 경계 사례를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
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
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 손실 함수

손실 계산은 모델의 출력을 최적화 목표와 이어 준다.

앞의 연습문제 4개로 직접 확인할 수 있다.
