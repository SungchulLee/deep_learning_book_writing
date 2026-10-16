# 러닝레이트 스케줄러

러닝레이트 스케줄러는 정해진 일정이나 관찰한 지표에 따라 옵티마이저의 러닝레이트를 바꾼다. 이 예제는 네 종류의 러닝레이트 곡선을 만들고 실제 학습 루프에 연결할 때의 호출 위치를 설명한다.

## 1. 코드

??? note "코드 (452줄)"

    ```python
    """
    ================================================================================
    중급 02: 러닝레이트 스케줄러
    ================================================================================

    배울 내용:
    - 러닝레이트 스케줄링이 중요한 이유
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
    # 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
    # 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
    torch.manual_seed(0)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 80)
    print("LEARNING RATE SCHEDULERS")
    print("=" * 80)

    # ============================================================================
    # 1절: 왜 러닝레이트 스케줄링을 쓰는가?
    # ============================================================================
    print("\n" + "-" * 80)
    print("WHY LEARNING RATE SCHEDULING?")
    print("-" * 80)

    print("""
    고정 러닝레이트의 문제:

      학습 초기:
      • 빠르게 나아가려면 큰 러닝레이트가 필요하다
      • 안장점에서 벗어난다
      • 로스 지형을 탐색한다

      학습 말기:
      • 러닝레이트가 크면 최적점 둘레에서 진동한다
      • 해를 정밀하게 다듬을 수 없다
      • 가장 좋은 해로 수렴하지 못할 수 있다

      해법:
      큰 러닝레이트로 시작 → 차츰 줄이기 → 끝에서 정밀 조정

      이를 "러닝레이트 어닐링" 또는 "러닝레이트 감쇠"라 한다
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
       • N 에폭마다 러닝레이트를 일정 배수로 줄인다
       • 단순하고 예측하기 쉽다
       • 예: 30 에폭마다 lr × 0.1

    2. EXPONENTIAL LR:
       • 에폭마다 일정 배수로 러닝레이트를 줄인다
       • 매끄러운 지수 감쇠
       • 예: 에폭마다 lr × 0.95

    3. 코사인 어닐링:
       • 코사인 곡선을 따른다
       • 매끄럽게 줄어들며 주기적 재시작을 고를 수 있다
       • 깊은 뉴럴 넷트웍에서 널리 쓴다

    4. 정체 시 러닝레이트 감소:
       • 지표가 더 나아지지 않으면 러닝레이트를 줄인다
       • 학습 진행에 맞추어 적응한다
       • 예: 검증 로스가 5 에폭 동안 나아지지 않으면 0.1배로 줄인다
    """)

    # ============================================================================
    # 3절: StepLR - 일정 간격마다 줄이기
    # ============================================================================
    print("\n" + "-" * 80)
    print("1. STEP LR SCHEDULER")
    print("-" * 80)

    optimizer_step = optim.SGD(model.parameters(), lr=0.1)
    # step_size 에포크마다 러닝레이트에 gamma를 곱한다
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

        # 러닝레이트를 갱신한다
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
    # 에포크마다 러닝레이트에 gamma를 곱한다
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
    # 코사인 곡선을 따라 러닝레이트를 줄인다
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
    # 지표가 정체되면 러닝레이트를 줄인다
    scheduler_plateau = ReduceLROnPlateau(
        optimizer_plateau, 
        mode='min',           # 지표를 최소화
        factor=0.5,           # 러닝레이트에 0.5를 곱한다
        patience=5,           # 줄이기 전에 5 에포크 기다린다
        verbose=True,         # 러닝레이트가 바뀌면 출력
        min_lr=0.001          # 이보다 낮추지 않는다
    )

    print(f"Initial LR: {optimizer_plateau.param_groups[0]['lr']:.6f}")
    print("\nSimulating training with plateaus:")

    # 검증 로스 모의실험
    # 로스가 줄다가 정체된 뒤 다시 준다
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
    plot_path = 'scheduler_comparison.png'
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

        # 러닝레이트 갱신
        scheduler.step()  # 시간 기반 스케줄러용
        # OR
        scheduler.step(val_loss)  # ReduceLROnPlateau용

        # 현재 러닝레이트 기록
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
       ✓ 예: 100 에폭에 StepLR(step_size=30, gamma=0.1)

    📉 지수 러닝레이트를 쓸 때:
       ✓ 매끄럽고 연속적인 감쇠를 원한다
       ✓ 많은 에폭 동안 학습한다
       ✓ 예: ExponentialLR(gamma=0.95)

    🌊 코사인 어닐링을 쓸 때:
       ✓ 최신 심층 뉴럴 넷트웍을 학습한다
       ✓ 매끄럽고 점진적인 감소를 원한다
       ✓ ImageNet과 트랜스포머에서 널리 쓴다
       ✓ 주기적 재시작에는 CosineAnnealingWarmRestarts를 쓸 수 있다
       ✓ 예: CosineAnnealingLR(T_max=epochs, eta_min=1e-6)

    🎯 ReduceLROnPlateau를 쓸 때:
       ✓ 최적의 스케줄을 미리 알 수 없다
       ✓ 적응적인 동작을 원한다
       ✓ 살펴볼 검증 지표가 있다
       ✓ 학습 길이가 달라질 수 있다
       ✓ 예: ReduceLROnPlateau(patience=10, factor=0.5)

    💡 실전 요령:
       가장 좋은 결과를 얻으려면 스케줄러에 워밍업을 곁들여라!
       낮은 러닝레이트로 시작하여 기준 러닝레이트까지 올린 뒤 감쇠시킨다
    """)

    # ============================================================================
    # 10절: 심화 - 워밍업 + 코사인 어닐링
    # ============================================================================
    print("\n" + "-" * 80)
    print("ADVANCED: Learning Rate Warmup")
    print("-" * 80)

    print("""
    워밍업이란 무엇인가?
      • 아주 낮은 러닝레이트로 시작한다
      • 기준 러닝레이트까지 차츰 올린다
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
      2. 처음 N 에폭 동안 러닝레이트를 직접 조정한다
      3. transformers 라이브러리의 get_linear_schedule_with_warmup을 쓴다
    """)

    # 간단한 워밍업 구현
    def get_lr_with_warmup(epoch, warmup_epochs, base_lr, min_lr, total_epochs):
        """워밍업 뒤에 코사인 어닐링을 적용한 러닝레이트를 계산한다"""
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
    1. 러닝레이트 스케줄링은 학습을 개선한다.
       • 더 빠른 수렴
       • 더 나은 최종 성능
       • 익힘이 더 든든하다

    2. 필요에 따라 스케줄러를 달리 쓴다.
       • StepLR: 단순하고 예측 가능한 감소
       • ExponentialLR: 매끄러운 지수 감쇠
       • CosineAnnealingLR: 최신 뉴럴 넷트웍에서 널리 쓴다
       • ReduceLROnPlateau: 지표에 따라 적응한다

    3. 일반적인 전략:
       • 큰 러닝레이트로 시작한다(탐색)
       • 차츰 줄인다(정밀 조정)
       • 필요하면 처음에 워밍업을 쓴다

    4. 구현은 단순하다.
       • 최적화기를 만든 뒤 스케줄러를 만든다
       • 에폭마다 scheduler.step()을 부른다
       • ReduceLROnPlateau에는 지표가 필요하다: scheduler.step(val_loss)

    5. 학습 중에 러닝레이트를 살핀다.
       • 스케줄이 잘 도는지 보려면 기록하라
       • 학습이 불안정하거나 너무 느리면 조정하라

    다음 단계:
    → 문제에 여러 스케줄러를 써 보라
    → 워밍업 전략을 실험해 보라
    → 순환 러닝레이트를 배워 보라
    → 조기 종료와 함께 써 보라
    """)
    print("=" * 80)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (286줄)"

    ```
    ================================================================================
    LEARNING RATE SCHEDULERS
    ================================================================================

    --------------------------------------------------------------------------------
    WHY LEARNING RATE SCHEDULING?
    --------------------------------------------------------------------------------

    고정 러닝레이트의 문제:
      
      학습 초기:
      • 빠르게 나아가려면 큰 러닝레이트가 필요하다
      • 안장점에서 벗어난다
      • 로스 지형을 탐색한다
      
      학습 말기:
      • 러닝레이트가 크면 최적점 둘레에서 진동한다
      • 해를 정밀하게 다듬을 수 없다
      • 가장 좋은 해로 수렴하지 못할 수 있다
      
      해법:
      큰 러닝레이트로 시작 → 차츰 줄이기 → 끝에서 정밀 조정
      
      이를 "러닝레이트 어닐링" 또는 "러닝레이트 감쇠"라 한다


    --------------------------------------------------------------------------------
    COMMON SCHEDULER TYPES
    --------------------------------------------------------------------------------

    1. STEP LR:
       • N 에폭마다 러닝레이트를 일정 배수로 줄인다
       • 단순하고 예측하기 쉽다
       • 예: 30 에폭마다 lr × 0.1

    2. EXPONENTIAL LR:
       • 에폭마다 일정 배수로 러닝레이트를 줄인다
       • 매끄러운 지수 감쇠
       • 예: 에폭마다 lr × 0.95

    3. 코사인 어닐링:
       • 코사인 곡선을 따른다
       • 매끄럽게 줄어들며 주기적 재시작을 고를 수 있다
       • 깊은 뉴럴 넷트웍에서 널리 쓴다

    4. 정체 시 러닝레이트 감소:
       • 지표가 더 나아지지 않으면 러닝레이트를 줄인다
       • 학습 진행에 맞추어 적응한다
       • 예: 검증 로스가 5 에폭 동안 나아지지 않으면 0.1배로 줄인다


    --------------------------------------------------------------------------------
    1. STEP LR SCHEDULER
    --------------------------------------------------------------------------------
    Initial LR: 0.100000

    Learning rate over 50 epochs:
      Epoch 10: LR = 0.100000
      Epoch 20: LR = 0.050000
      Epoch 30: LR = 0.025000
      Epoch 40: LR = 0.012500
      Epoch 50: LR = 0.006250

    EXPLANATION:
      step_size=10: LR changes every 10 epochs
      gamma=0.5: LR is multiplied by 0.5 at each step
      Result: 0.1 → 0.05 → 0.025 → 0.0125 → ...

    --------------------------------------------------------------------------------
    2. EXPONENTIAL LR SCHEDULER
    --------------------------------------------------------------------------------
    Initial LR: 0.100000

    Learning rate over 50 epochs:
      Epoch 1: LR = 0.100000
      Epoch 11: LR = 0.059874
      Epoch 21: LR = 0.035849
      Epoch 31: LR = 0.021464
      Epoch 41: LR = 0.012851

    EXPLANATION:
      gamma=0.95: LR is multiplied by 0.95 every epoch
      Smooth exponential decay: lr(t) = lr(0) × gamma^t
      More gradual than StepLR

    --------------------------------------------------------------------------------
    3. COSINE ANNEALING LR SCHEDULER
    --------------------------------------------------------------------------------
    Initial LR: 0.100000

    Learning rate over 50 epochs:
      Epoch 1: LR = 0.100000
      Epoch 11: LR = 0.090546
      Epoch 21: LR = 0.065796
      Epoch 31: LR = 0.035204
      Epoch 41: LR = 0.010454

    EXPLANATION:
      T_max=50: Complete cosine cycle over 50 epochs
      eta_min=0.001: Minimum learning rate
      Smooth decrease with faster drop at beginning
      Very popular for training vision models

    --------------------------------------------------------------------------------
    4. REDUCE LR ON PLATEAU SCHEDULER
    --------------------------------------------------------------------------------
    Initial LR: 0.100000

    Simulating training with plateaus:
      Epoch 1: Val Loss = 2.00, LR = 0.100000
      Epoch 2: Val Loss = 1.80, LR = 0.100000
      Epoch 3: Val Loss = 1.60, LR = 0.100000
      Epoch 4: Val Loss = 1.40, LR = 0.100000
      Epoch 5: Val Loss = 1.20, LR = 0.100000
      Epoch 6: Val Loss = 1.20, LR = 0.100000
      Epoch 7: Val Loss = 1.20, LR = 0.100000
      Epoch 8: Val Loss = 1.20, LR = 0.100000
      Epoch 9: Val Loss = 1.20, LR = 0.100000
      Epoch 10: Val Loss = 1.20, LR = 0.100000
      Epoch 11: Val Loss = 1.20, LR = 0.100000
      Epoch 12: Val Loss = 1.20, LR = 0.050000
      Epoch 13: Val Loss = 1.00, LR = 0.050000
      Epoch 14: Val Loss = 0.90, LR = 0.050000
      Epoch 15: Val Loss = 0.80, LR = 0.050000
      Epoch 16: Val Loss = 0.80, LR = 0.050000
      Epoch 17: Val Loss = 0.80, LR = 0.050000
      Epoch 18: Val Loss = 0.80, LR = 0.050000
      Epoch 19: Val Loss = 0.80, LR = 0.050000
      Epoch 20: Val Loss = 0.80, LR = 0.050000
      Epoch 21: Val Loss = 0.80, LR = 0.050000

    EXPLANATION:
      Monitors validation loss (or any metric)
      Reduces LR when no improvement for 'patience' epochs
      More adaptive than time-based schedulers
      Good when you don't know optimal schedule in advance

    --------------------------------------------------------------------------------
    VISUALIZATION
    --------------------------------------------------------------------------------
    Plot saved to: scheduler_comparison.png

    --------------------------------------------------------------------------------
    PRACTICAL USAGE IN TRAINING LOOP
    --------------------------------------------------------------------------------

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
        
        # 러닝레이트 갱신
        scheduler.step()  # 시간 기반 스케줄러용
        # OR
        scheduler.step(val_loss)  # ReduceLROnPlateau용
        
        # 현재 러닝레이트 기록
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}, LR: {current_lr:.6f}')


    --------------------------------------------------------------------------------
    DECISION GUIDE: Which Scheduler to Use?
    --------------------------------------------------------------------------------

    📅 StepLR을 쓸 때:
       ✓ 단순하고 예측 가능한 스케줄링을 원한다
       ✓ 정해진 에폭 수만큼 학습한다
       ✓ 전통적인 비전 과제에서 흔하다
       ✓ 예: 100 에폭에 StepLR(step_size=30, gamma=0.1)
       
    📉 지수 러닝레이트를 쓸 때:
       ✓ 매끄럽고 연속적인 감쇠를 원한다
       ✓ 많은 에폭 동안 학습한다
       ✓ 예: ExponentialLR(gamma=0.95)
       
    🌊 코사인 어닐링을 쓸 때:
       ✓ 최신 심층 뉴럴 넷트웍을 학습한다
       ✓ 매끄럽고 점진적인 감소를 원한다
       ✓ ImageNet과 트랜스포머에서 널리 쓴다
       ✓ 주기적 재시작에는 CosineAnnealingWarmRestarts를 쓸 수 있다
       ✓ 예: CosineAnnealingLR(T_max=epochs, eta_min=1e-6)
       
    🎯 ReduceLROnPlateau를 쓸 때:
       ✓ 최적의 스케줄을 미리 알 수 없다
       ✓ 적응적인 동작을 원한다
       ✓ 살펴볼 검증 지표가 있다
       ✓ 학습 길이가 달라질 수 있다
       ✓ 예: ReduceLROnPlateau(patience=10, factor=0.5)

    💡 실전 요령:
       가장 좋은 결과를 얻으려면 스케줄러에 워밍업을 곁들여라!
       낮은 러닝레이트로 시작하여 기준 러닝레이트까지 올린 뒤 감쇠시킨다


    --------------------------------------------------------------------------------
    ADVANCED: Learning Rate Warmup
    --------------------------------------------------------------------------------

    워밍업이란 무엇인가?
      • 아주 낮은 러닝레이트로 시작한다
      • 기준 러닝레이트까지 차츰 올린다
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
      2. 처음 N 에폭 동안 러닝레이트를 직접 조정한다
      3. transformers 라이브러리의 get_linear_schedule_with_warmup을 쓴다


    LR with warmup (first 20 epochs):
      Epoch 1: 0.001000
      Epoch 3: 0.020800
      Epoch 5: 0.040600
      Epoch 7: 0.060400
      Epoch 9: 0.080200
      Epoch 11: 0.100000
      Epoch 13: 0.099879
      Epoch 15: 0.099518
      Epoch 17: 0.098918
      Epoch 19: 0.098082

    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. 러닝레이트 스케줄링은 학습을 개선한다.
       • 더 빠른 수렴
       • 더 나은 최종 성능
       • 익힘이 더 든든하다

    2. 필요에 따라 스케줄러를 달리 쓴다.
       • StepLR: 단순하고 예측 가능한 감소
       • ExponentialLR: 매끄러운 지수 감쇠
       • CosineAnnealingLR: 최신 뉴럴 넷트웍에서 널리 쓴다
       • ReduceLROnPlateau: 지표에 따라 적응한다

    3. 일반적인 전략:
       • 큰 러닝레이트로 시작한다(탐색)
       • 차츰 줄인다(정밀 조정)
       • 필요하면 처음에 워밍업을 쓴다

    4. 구현은 단순하다.
       • 최적화기를 만든 뒤 스케줄러를 만든다
       • 에폭마다 scheduler.step()을 부른다
       • ReduceLROnPlateau에는 지표가 필요하다: scheduler.step(val_loss)

    5. 학습 중에 러닝레이트를 살핀다.
       • 스케줄이 잘 도는지 보려면 기록하라
       • 학습이 불안정하거나 너무 느리면 조정하라

    다음 단계:
    → 문제에 여러 스케줄러를 써 보라
    → 워밍업 전략을 실험해 보라
    → 순환 러닝레이트를 배워 보라
    → 조기 종료와 함께 써 보라

    ================================================================================
    ```


## 2. 논의

앞부분은 실제 모델의 수렴 성능을 비교하는 실험이 아니다. 일정 기반 예제의 로스는 모델 출력과 연결되지 않은 상수 텐서이고, ReduceLROnPlateau에는 미리 만든 검증 로스를 전달한다. 그래프는 설정에 따른 러닝레이트 변화를 보여 준다.

StepLR은 지정한 호출 간격마다 러닝레이트에 gamma를 곱하고 ExponentialLR은 매 호출마다 곱한다. CosineAnnealingLR은 처음부터 T_max까지 최솟값을 향해 감소하지만, T_max를 넘겨 계속 호출할 때도 단조 감소한다고 가정하면 안 된다. 이 클래스는 주기적인 재시작을 구현하는 CosineAnnealingWarmRestarts와 구분한다.

호출 단위는 학습 루프가 정한다. 에포크마다 호출하면 T_max도 에포크 수로, 갱신마다 호출하면 갱신 수로 해석한다. 일반적인 일정 기반 스케줄러는 optimizer.step() 뒤에 호출한다. ReduceLROnPlateau에는 검증 후 해당 지표를 전달한다. 후반부의 학습 루프 조각은 미정의 데이터 로더·검증 함수를 포함한 사용 예시이며, 서로 다른 스케줄러의 호출 방식을 한 객체에 동시에 적용하는 완성 코드가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
StepLR의 초기 러닝레이트가 0.1, step_size=10, gamma=0.5라면 20회 스케줄 갱신 후 값은 얼마인가?

</div>

??? success "연습문제 1 풀이"
    통상적인 초기 상태에서 매 에포크 말 한 번 호출하면 10회 후 0.05, 20회 후 0.025이다. 학습에 실제 사용한 값과 갱신 직후 기록한 값의 시점은 구분한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
ReduceLROnPlateau를 사용할 때 훈련 로스 대신 검증 로스를 전달하는 이유는 무엇인가?

</div>

??? success "연습문제 2 풀이"
    모델 선택의 관심 대상인 검증 성능의 정체에 반응하도록 하기 위해서다. mode, threshold, patience를 지표 정의와 함께 설정하며, 훈련 로스를 감시하는 것도 가능하지만 다른 폴리시가 된다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
에포크마다 호출하던 코사인 스케줄러를 배치마다 호출하면 무엇을 바꿔야 하는가?

</div>

??? success "연습문제 3 풀이"
    T_max를 의도한 총 옵티마이저 갱신 수에 맞춰 다시 정한다. 그래디언트 누적을 사용하면 미니배치 수와 실제 갱신 수가 다르므로 optimizer.step()의 횟수를 기준으로 일정을 정한다.

## 정리하며

러닝레이트 곡선을 해석하려면 호출 단위와 기록 시점을 먼저 확인한다. 스케줄러의 수렴 효과는 곡선 그림만으로 판단할 수 없으며, 실제 학습과 검증을 통해 비교해야 한다.
