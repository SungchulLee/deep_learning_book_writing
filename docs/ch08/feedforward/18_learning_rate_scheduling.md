# 학습률 일정 조절

학습률 스케줄러의 종류와 일정을 비교한다. 본문 코드는 학습 성능 실험 대신 학습률 곡선을 수동 계산해 그리는 시연이다.

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

코드는 StepLR, ExponentialLR, CosineAnnealingLR와 ReduceLROnPlateau 객체를 생성한다. 그래프에 쓰는 값은 각 스케줄러를 실제 학습에서 step한 결과가 아니라 식과 조건문으로 만든 값이다. 따라서 곡선만으로 수렴 속도나 최종 정확도를 비교할 수 없다.

StepLR 등은 정한 호출 주기에 따라 갱신한다. ReduceLROnPlateau는 검증 손실 같은 지표를 전달해야 한다. 여러 스케줄러가 하나의 옵티마이저를 공유하는 이 시연을 그대로 복사해 모두 동시에 적용하지 않는다.

학습률 스케줄과 옵티마이저의 갱신 순서를 확인한다. 일반적인 에포크 단위 스케줄은 해당 에포크의 옵티마이저 갱신 뒤 호출하지만, 배치 단위 스케줄이나 지표 기반 스케줄에는 각 규약을 적용한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
그래프가 실제 스케줄러의 동작을 검증하는가?

</div>

??? success "연습문제 1 풀이"
    아니다. 그래프는 수동 계산한 일정의 모양이다. 실제 동작을 확인하려면 독립 옵티마이저와 스케줄러를 구성해 호출 횟수별 학습률을 기록하고 수동 식과 비교해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
ReduceLROnPlateau가 StepLR와 다른 입력을 요구하는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    StepLR는 호출 횟수를 기준으로 줄이고 ReduceLROnPlateau는 지정한 지표의 개선 여부를 기준으로 줄인다. 검증 손실을 줄이는 설정이라면 매 평가 후 그 손실을 전달한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
스케줄러를 추가하면 최종 성능이 항상 좋아지는가?

</div>

??? success "연습문제 3 풀이"
    아니다. 초기 학습률, 감쇠 시점, 전체 예산과 데이터에 따라 달라진다. 고정 학습률 기준 모델과 같은 선택 예산에서 비교한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
조기 종료를 구현하라. 매 에포크 후 검증 손실을 추적하고, 10 에포크 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

</div>

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
    위 조각은 `evaluate`와 `val_loader` 등의 정의가 필요한 개략 코드이다. `state_dict().copy()`는 텐서를 독립 복제하지 않는 얕은 복사이므로 최선 상태가 바뀔 수 있다. 실제 구현에서는 깊은 복사나 파일 저장으로 상태를 보존하고, 그 상태를 복원한 뒤 별도 테스트셋에서 평가한다.

## 정리하며

스케줄러의 호출 주기와 입력 규약을 확인한다. 이 페이지의 수동 학습률 곡선과 실제 학습 결과는 구분해야 한다.
