# 학습률 스케줄러

학습 내내 고정된 학습률이 최적인 경우는 드물다. 학습률 스케줄러는 시간에 따라 학습률을 조정한다. StepLR은 일정 간격마다 줄이고, ExponentialLR은 매끄럽게 감쇠시키며, CosineAnnealingLR은 코사인 곡선을 따르고, ReduceLROnPlateau은 검증 지표에 따라 맞춘다. 알맞은 스케줄링은 수렴 속도와 최종 성능을 모두 높인다.

## 코드

```python
"""
================================================================================
중급 02: 학습률 스케줄러
================================================================================

배울 내용:
- 학습률 스케줄링이 중요한 이유
- 여러 스케줄러의 종류 (Step, Exponential, Cosine, ReduceLROnPlateau)
- 각 스케줄러를 언제 어떻게 쓸까
- 스케줄러와 최적화기를 함께 쓰기

선수 지식:
- 입문자용 튜토리얼을 모두 마친다
- 최적화기의 기본을 이해한다

소요 시간: 약 20분
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 80)
print("LEARNING RATE SCHEDULERS")
print("=" * 80)

# ============================================================================
# 1절: 왜 학습률 스케줄링을 쓰는가?
# ============================================================================
print("\n" + "-" * 80)
print("WHY LEARNING RATE SCHEDULING?")
print("-" * 80)

print("""
고정 학습률의 문제:
  
  학습 초기:
  • 빠르게 나아가려면 큰 학습률이 필요하다
  • 안장점에서 벗어난다
  • 손실 지형을 탐색한다
  
  학습 말기:
  • 학습률이 크면 최적점 둘레에서 진동한다
  • 해를 정밀하게 다듬을 수 없다
  • 가장 좋은 해로 수렴하지 못할 수 있다
  
  해법:
  큰 학습률로 시작 → 차츰 줄이기 → 끝에서 정밀 조정
  
  이를 "학습률 어닐링" 또는 "학습률 감쇠"라 한다
""")

# ============================================================================
# 2절: 흔한 스케줄러의 종류
# ============================================================================
print("\n" + "-" * 80)
print("COMMON SCHEDULER TYPES")
print("-" * 80)

# 시연을 위한 임시 최적화기 만들기
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("""
1. STEP LR:
   • N 에폭마다 학습률을 일정 배수로 줄인다
   • 단순하고 예측하기 쉽다
   • 예: 30 에폭마다 lr × 0.1

2. EXPONENTIAL LR:
   • 에폭마다 일정 배수로 학습률을 줄인다
   • 매끄러운 지수 감쇠
   • 예: 에폭마다 lr × 0.95

3. 코사인 어닐링:
   • 코사인 곡선을 따른다
   • 매끄럽게 줄어들며 주기적 재시작을 고를 수 있다
   • 깊은 신경망에서 널리 쓴다

4. 정체 시 학습률 감소:
   • 지표가 더 나아지지 않으면 학습률을 줄인다
   • 학습 진행에 맞추어 적응한다
   • 예: 검증 손실이 5 에폭 동안 나아지지 않으면 0.1배로 줄인다
""")

# ============================================================================
# 3절: StepLR - 일정 간격마다 줄이기
# ============================================================================
print("\n" + "-" * 80)
print("1. STEP LR SCHEDULER")
print("-" * 80)

optimizer_step = optim.SGD(model.parameters(), lr=0.1)
# step_size 에포크마다 학습률에 gamma를 곱한다
scheduler_step = StepLR(optimizer_step, step_size=10, gamma=0.5)

print(f"Initial LR: {optimizer_step.param_groups[0]['lr']:.6f}")
print("\nLearning rate over 50 epochs:")

lrs_step = []
for epoch in range(50):
    lrs_step.append(optimizer_step.param_groups[0]['lr'])
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}: LR = {optimizer_step.param_groups[0]['lr']:.6f}")
    
    # 학습 단계 모의실험
    optimizer_step.zero_grad()
    loss = torch.tensor(1.0, requires_grad=True)
    loss.backward()
    optimizer_step.step()
    
    # 학습률을 갱신한다
    scheduler_step.step()

print("\nEXPLANATION:")
print("  step_size=10: LR changes every 10 epochs")
print("  gamma=0.5: LR is multiplied by 0.5 at each step")
print("  Result: 0.1 → 0.05 → 0.025 → 0.0125 → ...")

# ============================================================================
# 4절: ExponentialLR - 매끄러운 감쇠
# ============================================================================
print("\n" + "-" * 80)
print("2. EXPONENTIAL LR SCHEDULER")
print("-" * 80)

optimizer_exp = optim.SGD(model.parameters(), lr=0.1)
# 에포크마다 학습률에 gamma를 곱한다
scheduler_exp = ExponentialLR(optimizer_exp, gamma=0.95)

print(f"Initial LR: {optimizer_exp.param_groups[0]['lr']:.6f}")
print("\nLearning rate over 50 epochs:")

lrs_exp = []
for epoch in range(50):
    lrs_exp.append(optimizer_exp.param_groups[0]['lr'])
    
    if epoch % 10 == 0:
        print(f"  Epoch {epoch+1}: LR = {optimizer_exp.param_groups[0]['lr']:.6f}")
    
    # 학습 모의실험
    optimizer_exp.zero_grad()
    loss = torch.tensor(1.0, requires_grad=True)
    loss.backward()
    optimizer_exp.step()
    scheduler_exp.step()

print("\nEXPLANATION:")
print("  gamma=0.95: LR is multiplied by 0.95 every epoch")
print("  Smooth exponential decay: lr(t) = lr(0) × gamma^t")
print("  More gradual than StepLR")

# ============================================================================
# 5절: CosineAnnealingLR - 매끄러운 코사인 곡선
# ============================================================================
print("\n" + "-" * 80)
print("3. COSINE ANNEALING LR SCHEDULER")
print("-" * 80)

optimizer_cos = optim.SGD(model.parameters(), lr=0.1)
# 코사인 곡선을 따라 학습률을 줄인다
scheduler_cos = CosineAnnealingLR(optimizer_cos, T_max=50, eta_min=0.001)

print(f"Initial LR: {optimizer_cos.param_groups[0]['lr']:.6f}")
print("\nLearning rate over 50 epochs:")

lrs_cos = []
for epoch in range(50):
    lrs_cos.append(optimizer_cos.param_groups[0]['lr'])
    
    if epoch % 10 == 0:
        print(f"  Epoch {epoch+1}: LR = {optimizer_cos.param_groups[0]['lr']:.6f}")
    
    # 학습 모의실험
    optimizer_cos.zero_grad()
    loss = torch.tensor(1.0, requires_grad=True)
    loss.backward()
    optimizer_cos.step()
    scheduler_cos.step()

print("\nEXPLANATION:")
print("  T_max=50: Complete cosine cycle over 50 epochs")
print("  eta_min=0.001: Minimum learning rate")
print("  Smooth decrease with faster drop at beginning")
print("  Very popular for training vision models")

# ============================================================================
# 6절: ReduceLROnPlateau - 적응형 스케줄링
# ============================================================================
print("\n" + "-" * 80)
print("4. REDUCE LR ON PLATEAU SCHEDULER")
print("-" * 80)

optimizer_plateau = optim.SGD(model.parameters(), lr=0.1)
# 지표가 정체되면 학습률을 줄인다
scheduler_plateau = ReduceLROnPlateau(
    optimizer_plateau, 
    mode='min',           # 지표를 최소화
    factor=0.5,           # 학습률에 0.5를 곱한다
    patience=5,           # 줄이기 전에 5 에포크 기다린다
    verbose=True,         # 학습률이 바뀌면 출력
    min_lr=0.001          # 이보다 낮추지 않는다
)

print(f"Initial LR: {optimizer_plateau.param_groups[0]['lr']:.6f}")
print("\nSimulating training with plateaus:")

# 검증 손실 모의실험
# 손실이 줄다가 정체된 뒤 다시 준다
simulated_losses = (
    [2.0, 1.8, 1.6, 1.4, 1.2] +  # 나아지는 중
    [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2] +  # 정체 (7 에포크)
    [1.0, 0.9, 0.8] +  # 다시 나아지는 중
    [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]  # 다시 정체
)

lrs_plateau = []
for epoch, val_loss in enumerate(simulated_losses):
    current_lr = optimizer_plateau.param_groups[0]['lr']
    lrs_plateau.append(current_lr)
    
    print(f"  Epoch {epoch+1}: Val Loss = {val_loss:.2f}, LR = {current_lr:.6f}")
    
    # 이 스케줄러에는 검증 지표가 필요하다
    scheduler_plateau.step(val_loss)

print("\nEXPLANATION:")
print("  Monitors validation loss (or any metric)")
print("  Reduces LR when no improvement for 'patience' epochs")
print("  More adaptive than time-based schedulers")
print("  Good when you don't know optimal schedule in advance")

# ============================================================================
# 7절: 모든 스케줄러 시각화
# ============================================================================
print("\n" + "-" * 80)
print("VISUALIZATION")
print("-" * 80)

plt.figure(figsize=(14, 5))

# 그림 1: 시간 기반 스케줄러 비교
plt.subplot(1, 2, 1)
plt.plot(range(len(lrs_step)), lrs_step, 'b-', label='StepLR', linewidth=2)
plt.plot(range(len(lrs_exp)), lrs_exp, 'r-', label='ExponentialLR', linewidth=2)
plt.plot(range(len(lrs_cos)), lrs_cos, 'g-', label='CosineAnnealingLR', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Time-Based Schedulers', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')

# 그림 2: ReduceLROnPlateau
plt.subplot(1, 2, 2)
plt.plot(range(len(lrs_plateau)), lrs_plateau, 'purple', linewidth=2, label='ReduceLROnPlateau')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Adaptive Scheduler (ReduceLROnPlateau)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plot_path = '/home/claude/scheduler_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

# ============================================================================
# 8절: 학습 루프에서의 실제 사용
# ============================================================================
print("\n" + "-" * 80)
print("PRACTICAL USAGE IN TRAINING LOOP")
print("-" * 80)

print("""
스케줄러를 곁들인 기본 학습 루프:

# 준비
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()

# 학습 루프
for epoch in range(num_epochs):
    # 학습 단계
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # 검증 단계 (선택 사항이지만 권장한다)
    model.eval()
    val_loss = validate(model, val_loader, criterion)
    
    # 학습률 갱신
    scheduler.step()  # 시간 기반 스케줄러용
    # OR
    scheduler.step(val_loss)  # ReduceLROnPlateau용
    
    # 현재 학습률 기록
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}, LR: {current_lr:.6f}')
""")

# ============================================================================
# 9절: 알맞은 스케줄러 고르기
# ============================================================================
print("\n" + "-" * 80)
print("DECISION GUIDE: Which Scheduler to Use?")
print("-" * 80)

print("""
📅 StepLR을 쓸 때:
   ✓ 단순하고 예측 가능한 스케줄링을 원한다
   ✓ 정해진 에폭 수만큼 학습한다
   ✓ 전통적인 비전 과제에서 흔하다
   ✓ Example: StepLR(step_size=30, gamma=0.1) for 100 epochs
   
📉 지수 학습률을 쓸 때:
   ✓ 매끄럽고 연속적인 감쇠를 원한다
   ✓ 많은 에폭 동안 학습한다
   ✓ Example: ExponentialLR(gamma=0.95)
   
🌊 코사인 어닐링을 쓸 때:
   ✓ 최신 심층 신경망을 학습한다
   ✓ 매끄럽고 점진적인 감소를 원한다
   ✓ ImageNet과 트랜스포머에서 널리 쓴다
   ✓ 주기적 재시작에는 CosineAnnealingWarmRestarts를 쓸 수 있다
   ✓ Example: CosineAnnealingLR(T_max=epochs, eta_min=1e-6)
   
🎯 ReduceLROnPlateau를 쓸 때:
   ✓ 최적의 스케줄을 미리 알 수 없다
   ✓ 적응적인 동작을 원한다
   ✓ 살펴볼 검증 지표가 있다
   ✓ 학습 길이가 달라질 수 있다
   ✓ Example: ReduceLROnPlateau(patience=10, factor=0.5)

💡 실전 요령:
   가장 좋은 결과를 얻으려면 스케줄러에 워밍업을 곁들여라!
   낮은 학습률로 시작하여 기준 학습률까지 올린 뒤 감쇠시킨다
""")

# ============================================================================
# 10절: 심화 - 워밍업 + 코사인 어닐링
# ============================================================================
print("\n" + "-" * 80)
print("ADVANCED: Learning Rate Warmup")
print("-" * 80)

print("""
워밍업이란 무엇인가?
  • 아주 낮은 학습률로 시작한다
  • 기준 학습률까지 차츰 올린다
  • 그다음 일반적인 스케줄링을 적용한다
  
왜 워밍업을 쓰는가?
  • 학습 초기의 불안정을 막는다
  • 배치 크기가 클 때 중요하다
  • 트랜스포머에서는 매우 중요하다
  • 배치 정규화에 도움이 된다

예: 워밍업 + 코사인 어닐링
  에폭 1~10: 1e-6에서 1e-3까지 선형 증가
  에폭 11~100: 1e-3에서 1e-6까지 코사인 어닐링
  
PyTorch에는 워밍업이 내장되어 있지 않지만 다음을 할 수 있다.
  1. 맞춤 스케줄러를 쓴다
  2. 처음 N 에폭 동안 학습률을 직접 조정한다
  3. transformers 라이브러리의 get_linear_schedule_with_warmup을 쓴다
""")

# 간단한 워밍업 구현
def get_lr_with_warmup(epoch, warmup_epochs, base_lr, min_lr, total_epochs):
    """워밍업 뒤에 코사인 어닐링을 적용한 학습률을 계산한다"""
    if epoch < warmup_epochs:
        # 선형 워밍업
        return min_lr + (base_lr - min_lr) * epoch / warmup_epochs
    else:
        # 코사인 어닐링
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

# 워밍업 시연
warmup_epochs = 10
total_epochs = 100
base_lr = 0.1
min_lr = 0.001

lrs_warmup = [get_lr_with_warmup(e, warmup_epochs, base_lr, min_lr, total_epochs) 
              for e in range(total_epochs)]

print(f"\nLR with warmup (first 20 epochs):")
for epoch in range(0, 20, 2):
    print(f"  Epoch {epoch+1}: {lrs_warmup[epoch]:.6f}")

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. 학습률 스케줄링은 학습을 개선한다.
   • 더 빠른 수렴
   • 더 나은 최종 성능
   • 익힘이 더 든든하다

2. 필요에 따라 스케줄러를 달리 쓴다.
   • StepLR: 단순하고 예측 가능한 감소
   • ExponentialLR: 매끄러운 지수 감쇠
   • CosineAnnealingLR: 최신 신경망에서 널리 쓴다
   • ReduceLROnPlateau: 지표에 따라 적응한다

3. 일반적인 전략:
   • Start with high LR (explore)
   • Gradually decrease (fine-tune)
   • 필요하면 처음에 워밍업을 쓴다

4. 구현은 단순하다.
   • 최적화기를 만든 뒤 스케줄러를 만든다
   • Call scheduler.step() after each epoch
   • ReduceLROnPlateau needs metric: scheduler.step(val_loss)

5. 학습 중에 학습률을 살핀다.
   • 스케줄이 잘 도는지 보려면 기록하라
   • 학습이 불안정하거나 너무 느리면 조정하라

다음 단계:
→ 문제에 여러 스케줄러를 써 보라
→ 워밍업 전략을 실험해 보라
→ 순환 학습률을 배워 보라
→ 조기 종료와 함께 써 보라
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

StepLR은 가장 단순한 스케줄러로, 일정한 간격마다 학습률에 정해진 인수를 곱한다. 예를 들어 10 에포크마다 $\gamma=0.5$을 곱하면 0.1, 0.05, 0.025, 0.0125 같은 열이 된다. 이렇게 급격히 바뀌면 학습이 잠시 불안정해질 수 있다.

CosineAnnealingLR은 처음 학습률에서 최솟값까지 반코사인 곡선을 따라 매끄럽게 단조 감소하는 일정을 준다. 처음에 천천히 줄고(최고점 근처에서 완만하게) 끝에도 천천히 다가가는(최솟값에 완만하게) 모양은, 초반에는 큰 걸음이 필요하고 후반에는 정밀함이 필요하다는 직관과 잘 맞는다.

ReduceLROnPlateau은 미리 정한 일정을 따르지 않고 학습의 흐름에 반응한다는 점에서 다른 스케줄러와 다르다. 감시하는 지표(보통 검증 손실)가 정해진 인내 기간 동안 나아지지 않으면 학습률을 줄인다. 이러한 적응성 덕분에 최적의 일정을 모를 때에도 튼튼하다.

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

