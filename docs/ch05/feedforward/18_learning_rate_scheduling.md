# 학습률 일정 조절

10_learning_rate_scheduling.py - 동적인 학습률. 더 나은 수렴을 위해 학습 중에 학습률을 조절하는 법을 배운다.

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 1. 코드

```python
"""
10_learning_rate_scheduling.py - 동적 학습률

더 나은 수렴을 위해 학습 중에 학습률을 조정하는 법을 배운다.
처음에는 높은 학습률로 빠르게 나아가고, 나중에는 줄여 세밀하게 다듬는다.

소요 시간: 30~40분 | 난이도: ⭐⭐⭐⭐☆
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================

print("="*70)
print("Learning Rate Scheduling")
print("="*70)

model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

optimizer = optim.SGD(model.parameters(), lr=0.1)

print("LEARNING RATE SCHEDULERS IN PyTorch:")
print("-"*70)

# 1. StepLR: N 에포크마다 학습률 감쇠
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
print("StepLR: Reduces LR by 0.1x every 30 epochs")

# 2. ExponentialLR: 지수 감쇠
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
print("ExponentialLR: Multiply LR by 0.9 each epoch")

# 3. CosineAnnealingLR: 코사인 감쇠
scheduler3 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
print("CosineAnnealingLR: Cosine curve over 100 epochs")

# 4. ReduceLROnPlateau: 지표가 정체되면 줄인다
scheduler4 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
print("ReduceLROnPlateau: Reduce when val loss doesn't improve for 5 epochs")

# 여러 일정을 시각화한다
epochs = 100
lrs_step = []
lrs_exp = []
lrs_cos = []

# StepLR 모의실험
lr = 0.1
for epoch in range(epochs):
    lrs_step.append(lr)
    if (epoch + 1) % 30 == 0:
        lr *= 0.1

# ExponentialLR 모의실험
lr = 0.1
for epoch in range(epochs):
    lrs_exp.append(lr)
    lr *= 0.9

# CosineAnnealingLR 모의실험
import math
for epoch in range(epochs):
    lr = 0.1 * (1 + math.cos(math.pi * epoch / 100)) / 2
    lrs_cos.append(lr)

# 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(lrs_step, label='StepLR', linewidth=2)
plt.plot(lrs_exp, label='ExponentialLR', linewidth=2)
plt.plot(lrs_cos, label='CosineAnnealingLR', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedules Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.savefig('10_lr_schedules.png', dpi=150)
print("\nSchedules visualization saved!")

print("\n" + "="*70)
print("USAGE PATTERN")
print("="*70)
print("""
# 준비
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

# 학습 루프
for epoch in range(num_epochs):
    for batch in train_loader:
        # 학습 단계
        loss.backward()
        optimizer.step()
    
    # 학습률을 갱신한다
    scheduler.step()
    
    # 현재 학습률 확인
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}, LR: {current_lr}")

RECOMMENDATIONS:
- 기준선을 잡으려면 고정 학습률로 시작하라
- 단순한 감쇠에는 StepLR을 쓰라
- 적응적 조정에는 ReduceLROnPlateau를 쓰라
- 순환 학습에는 CosineAnnealing을 쓰라
""")
plt.show()


if __name__ == "__main__":
    pass
```

**출력:**

```
======================================================================
Learning Rate Scheduling
======================================================================
LEARNING RATE SCHEDULERS IN PyTorch:
----------------------------------------------------------------------
StepLR: Reduces LR by 0.1x every 30 epochs
ExponentialLR: Multiply LR by 0.9 each epoch
CosineAnnealingLR: Cosine curve over 100 epochs
ReduceLROnPlateau: Reduce when val loss doesn't improve for 5 epochs

Schedules visualization saved!

======================================================================
USAGE PATTERN
======================================================================

# 준비
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

# 학습 루프
for epoch in range(num_epochs):
    for batch in train_loader:
        # 학습 단계
        loss.backward()
        optimizer.step()
    
    # 학습률을 갱신한다
    scheduler.step()
    
    # 현재 학습률 확인
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}, LR: {current_lr}")

RECOMMENDATIONS:
- 기준선을 잡으려면 고정 학습률로 시작하라
- 단순한 감쇠에는 StepLR을 쓰라
- 적응적 조정에는 ReduceLROnPlateau를 쓰라
- 순환 학습에는 CosineAnnealing을 쓰라
```

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

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
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.

## 정리하며

**다룬 것** — 학습률 일정 조절

학습 루프는 표준적인 PyTorch 패턴을 따른다.

앞의 연습문제 4개로 직접 확인할 수 있다.
