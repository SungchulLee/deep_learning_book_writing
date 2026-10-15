# 신경망 학습에서 옵티마이저 비교

합성 회귀 데이터에 같은 구조의 신경망을 적용하고 SGD, 모멘텀 SGD, RMSprop, Adam, AdamW의 학습 손실을 비교한다. 여기서 관찰하는 것은 지정된 초기값과 설정에서의 최적화 과정이다.

## 1. 코드

??? note "코드 (438줄)"

    ```python
    """
    ================================================================================
    중급 01: 널리 쓰이는 최적화기 견주기 (SGD, Adam, RMSprop, AdamW)
    ================================================================================

    배울 내용:
    - 최적화기마다 작동하는 방식
    - 각 최적화기를 언제 쓸까
    - Adam과 AdamW
    - 모멘텀과 적응형 학습률
    - 실제 문제에서의 실용적인 비교

    선수 지식:
    - 입문 튜토리얼을 모두 마친다
    - 기본적인 최적화 개념을 이해한다

    소요 시간: 약 25분
    ================================================================================
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib
    matplotlib.use('Agg')  # 비대화형 백엔드용
    import matplotlib.pyplot as plt

    print("=" * 80)
    print("COMPARING POPULAR OPTIMIZERS")
    print("=" * 80)

    # ============================================================================
    # 1절: 최적화 알고리즘 개관
    # ============================================================================
    print("\n" + "-" * 80)
    print("OPTIMIZATION ALGORITHMS OVERVIEW")
    print("-" * 80)

    print("""
    확률적 경사 하강법(SGD):
      • 가장 단순한 최적화기: param -= learning_rate × gradient
      • 장점: 단순하며 모멘텀과 잘 어울린다
      • 단점: 느릴 수 있고 학습률에 민감하다
      • 알맞은 곳: 잘 알려진 문제, 조율할 시간이 있을 때

    모멘텀을 쓰는 SGD:
      • "속도"를 더해 국소 최솟값을 뚫고 나간다
      • 기울기를 시간에 걸쳐 쌓는다: v = momentum × v + gradient
      • 잡음 섞인 기울기를 매끄럽게 한다
      • 알맞은 곳: 깊은 신경망, 잡음 섞인 기울기

    Adam(적응적 모멘트 추정):
      • 매개변수마다 배움 빠르기를 맞춘다
      • 모멘텀과 RMSprop을 합친다
      • 장점: 기본 설정으로도 잘 돌고 빠르게 수렴한다
      • 단점: SGD보다 일반화가 나쁠 수 있고 메모리를 더 쓴다
      • 알맞은 곳: 빠른 시제품 제작, 대부분의 딥러닝 과제

    AdamW(가중치 감쇠를 갖춘 Adam):
      • 가중치 감쇠를 올바로 구현한 Adam
      • Adam보다 일반화가 낫다
      • Adam의 가중치 감쇠 결함을 고친다
      • 알맞은 곳: 트랜스포머, 최신 구조, 가중치 감쇠를 쓸 때

    RMSprop(제곱평균제곱근 전파):
      • 제곱 기울기의 이동평균으로 학습률을 조절한다
      • 비정상 문제에 알맞다
      • 알맞은 곳: RNN, 온라인 학습
    """)

    # ============================================================================
    # 2절: 만만치 않은 최적화 문제 만들기
    # ============================================================================
    print("\n" + "-" * 80)
    print("SETUP: Non-Convex Optimization Problem")
    print("-" * 80)

    # 재현성을 위해 씨앗 고정
    torch.manual_seed(42)

    # 비선형 양상을 갖는 합성 데이터 생성
    n_samples = 200
    X = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
    # 비선형 함수: y = sin(x) + 0.5*x + 잡음
    y = torch.sin(X) + 0.5 * X + torch.randn(n_samples, 1) * 0.1

    print(f"Dataset: {n_samples} samples")
    print(f"Task: Learn the non-linear function y = sin(x) + 0.5*x")

    # ============================================================================
    # 3절: 간단한 신경망 정의
    # ============================================================================
    print("\n" + "-" * 80)
    print("NEURAL NETWORK ARCHITECTURE")
    print("-" * 80)

    class SimpleNet(nn.Module):
        """
        은닉층이 2개인 간단한 순방향 신경망
        입력 → 뉴런 20개 → 뉴런 20개 → 출력
        """
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(1, 20)
            self.fc2 = nn.Linear(20, 20)
            self.fc3 = nn.Linear(20, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # 구조 출력
    model = SimpleNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture:")
    print(model)
    print(f"\nTotal parameters: {total_params}")

    # ============================================================================
    # 4절: 학습 함수
    # ============================================================================

    def train_model(model, optimizer_name, optimizer, X, y, epochs=100):
        """모델을 학습시키고 손실 이력을 돌려준다"""
        criterion = nn.MSELoss()
        loss_history = []

        for epoch in range(epochs):
            # 순전파
            y_pred = model(X)
            loss = criterion(y_pred, y)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 손실 기록
            loss_history.append(loss.item())

            # 진행 상황 출력
            if (epoch + 1) % 20 == 0:
                print(f"  [{optimizer_name}] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        return loss_history

    # ============================================================================
    # 5절: 최적화기 비교
    # ============================================================================
    print("\n" + "-" * 80)
    print("TRAINING WITH DIFFERENT OPTIMIZERS")
    print("-" * 80)

    epochs = 100
    learning_rate = 0.01

    # 비교를 위해 결과 저장
    results = {}
    optimizers_to_test = {}

    # 1. SGD (기본)
    print("\n1. SGD (Vanilla - No Momentum):")
    model_sgd = SimpleNet()
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=learning_rate)
    optimizers_to_test['SGD'] = (model_sgd, opt_sgd)

    # 2. 모멘텀을 쓰는 SGD
    print("\n2. SGD with Momentum (0.9):")
    model_sgd_momentum = SimpleNet()
    opt_sgd_momentum = optim.SGD(model_sgd_momentum.parameters(), 
                                   lr=learning_rate, momentum=0.9)
    optimizers_to_test['SGD+Momentum'] = (model_sgd_momentum, opt_sgd_momentum)

    # 3. RMSprop
    print("\n3. RMSprop:")
    model_rmsprop = SimpleNet()
    opt_rmsprop = optim.RMSprop(model_rmsprop.parameters(), lr=learning_rate)
    optimizers_to_test['RMSprop'] = (model_rmsprop, opt_rmsprop)

    # 4. Adam
    print("\n4. Adam:")
    model_adam = SimpleNet()
    opt_adam = optim.Adam(model_adam.parameters(), lr=learning_rate)
    optimizers_to_test['Adam'] = (model_adam, opt_adam)

    # 5. AdamW
    print("\n5. AdamW (Adam with Weight Decay):")
    model_adamw = SimpleNet()
    opt_adamw = optim.AdamW(model_adamw.parameters(), lr=learning_rate, 
                             weight_decay=0.01)
    optimizers_to_test['AdamW'] = (model_adamw, opt_adamw)

    # 모든 모델 학습
    for name, (model, optimizer) in optimizers_to_test.items():
        loss_history = train_model(model, name, optimizer, X, y, epochs)
        results[name] = loss_history

    # ============================================================================
    # 6절: 결과 분석
    # ============================================================================
    print("\n" + "-" * 80)
    print("RESULTS ANALYSIS")
    print("-" * 80)

    print("\nFinal Loss (lower is better):")
    for name, loss_history in results.items():
        final_loss = loss_history[-1]
        print(f"  {name:15s}: {final_loss:.6f}")

    print("\nConvergence Speed (loss after 20 epochs):")
    for name, loss_history in results.items():
        early_loss = loss_history[19]  # 20번째 에포크 (0부터 셈)
        print(f"  {name:15s}: {early_loss:.6f}")

    # ============================================================================
    # 7절: 수렴 시각화
    # ============================================================================
    print("\n" + "-" * 80)
    print("VISUALIZATION")
    print("-" * 80)

    plt.figure(figsize=(12, 5))

    # 그림 1: 손실 곡선
    plt.subplot(1, 2, 1)
    for name, loss_history in results.items():
        plt.plot(loss_history, label=name, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 차이를 잘 보려고 로그 척도

    # 그림 2: 최종 예측
    plt.subplot(1, 2, 2)
    # 참 데이터 그리기
    plt.scatter(X.numpy(), y.numpy(), alpha=0.3, s=10, label='True Data', color='black')

    # 최적화기별 예측 그리기
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    with torch.no_grad():
        for (name, (model, _)), color in zip(optimizers_to_test.items(), colors):
            y_pred = model(X)
            plt.plot(X.numpy(), y_pred.numpy(), label=name, linewidth=2, 
                    color=color, alpha=0.7)

    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Final Predictions', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = 'optimizer_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # ============================================================================
    # 8절: 최적화기 자세히 설명하기
    # ============================================================================
    print("\n" + "-" * 80)
    print("DETAILED OPTIMIZER MECHANICS")
    print("-" * 80)

    print("""
    1. SGD(확률적 경사 하강법):
       Update: θ = θ - lr × ∇L

       • 가장 단순한 알고리즘
       • 매 단계가 곧장 내리막으로 간다
       • 좁은 골짜기에서 진동할 수 있다
       • 학습률 선택에 매우 민감하다

    2. 모멘텀을 쓰는 SGD:
       Velocity: v = β × v + ∇L
       Update: θ = θ - lr × v

       • 지난 기울기를 쌓는다(β는 보통 0.9)
       • "관성"을 쌓아 작은 굴곡을 뚫고 나간다
       • 잡음 섞인 기울기를 매끄럽게 한다
       • 최적값을 지나칠 수 있다

    3. RMSPROP:
       RMS: s = β × s + (1-β) × ∇L²
       Update: θ = θ - (lr / √s) × ∇L

       • 매개변수마다 학습률을 조절한다
       • 기울기의 제곱평균제곱근으로 나눈다
       • 학습률이 너무 작아지는 것을 막는다
       • 비정상 목적함수에 알맞다

    4. Adam(적응적 모멘트 추정):
       Momentum: m = β₁ × m + (1-β₁) × ∇L
       Velocity: v = β₂ × v + (1-β₂) × ∇L²
       Update: θ = θ - lr × (m / √v)

       • 모멘텀과 RMSprop을 합친다
       • 매개변수마다 배움 빠르기를 맞춘다
       • 편향 보정을 포함한다
       • Default β₁=0.9, β₂=0.999
       • 딥러닝에서 가장 널리 쓰는 최적화기

    5. ADAMW:
       Adam과 같으나 가중치 감쇠를 바로잡았다.
       Update: θ = θ - lr × (m / √v) - lr × λ × θ

       • Adam의 가중치 감쇠 구현을 고친다
       • 정칙화가 더 낫다
       • 트랜스포머와 큰 모델에 선호된다
    """)

    # ============================================================================
    # 9절: 실용적인 지침
    # ============================================================================
    print("\n" + "-" * 80)
    print("PRACTICAL GUIDELINES: Which Optimizer to Use?")
    print("-" * 80)

    print("""
    🎯 Adam을 쓸 때:
       ✓ 새 프로젝트를 시작할 때(좋은 기본값)
       ✓ 빠른 수렴이 필요하다
       ✓ 초매개변수를 조율할 시간이 적다
       ✓ RNN이나 트랜스포머를 다룬다
       ✓ 예: 빠른 시제품 제작, 연구 실험

    📊 SGD + 모멘텀을 쓸 때:
       ✓ 최종 모델 학습(일반화가 더 나을 때가 많다)
       ✓ 학습률을 조율할 시간이 있다
       ✓ CNN 학습(특히 ResNet, VGG)
       ✓ 가장 안정적인 장기 성능을 원한다
       ✓ 예: 실서비스 모델, ImageNet 학습

    🔬 AdamW를 쓸 때:
       ✓ 트랜스포머 학습(BERT, GPT 등)
       ✓ 가중치 감쇠 정칙화를 쓴다
       ✓ Adam보다 나은 일반화가 필요하다
       ✓ 최신 모범 사례를 따른다
       ✓ 예: 자연어 처리 모델, 대규모 학습

    ⚡ RMSprop을 쓸 때:
       ✓ RNN 학습(예전부터 널리 썼다)
       ✓ 비정상 문제
       ✓ 온라인 학습 상황
       ✓ 예: 시계열, 온라인 추천

    학습률 권장값:
    • SGD: 0.01 - 0.1
    • SGD + 모멘텀: 0.01~0.1
    • Adam: 0.001 (1e-3)
    • AdamW: 0.0001 - 0.001 (1e-4 to 1e-3)
    • RMSprop: 0.001

    언제나 이 기본값으로 시작하여 손실 곡선을 보며 조정하라!
    """)

    # ============================================================================
    # 10절: 흔히 쓰는 초매개변수
    # ============================================================================
    print("\n" + "-" * 80)
    print("HYPERPARAMETER REFERENCE")
    print("-" * 80)

    print("""
    LEARNING RATE (lr):
      • 가장 중요한 초매개변수
      • 너무 크면: 학습이 불안정하고 손실이 폭발한다
      • 너무 작으면: 학습이 너무 느리다
      • 학습 중 조정하려면 학습률 스케줄러를 쓴다

    MOMENTUM (SGD):
      • Typical: 0.9 or 0.95
      • 클수록 더 매끄럽지만 지나칠 수 있다

    베타(Adam과 AdamW):
      • β₁ (momentum): typically 0.9
      • β₂ (RMS): typically 0.999
      • 이 값을 바꿀 일은 거의 없다

    가중치 감쇠:
      • 정칙화 세기
      • Typical: 0.0001 to 0.01
      • 클수록 정칙화가 세다
      • 올바른 구현을 원하면 AdamW를 쓴다

    엡실론(Adam과 RMSprop):
      • 수치 안정을 위한 상수
      • Default: 1e-8
      • 바꿀 일이 거의 없다
    """)

    # ============================================================================
    # 요약
    # ============================================================================
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. 최적화기마다 서로 다른 절충을 한다.
       • 속도 대 안정성
       • 쓰기 쉬움 대 최종 성능
       • 메모리 효율

    2. Adam과 AdamW는 훌륭한 기본값이다.
       • 빠른 수렴
       • 기본 설정으로도 잘 돈다
       • 최신 구조에는 AdamW가 선호된다

    3. SGD + 모멘텀이 최종 성능에서 가장 나을 때가 많다.
       • 조율이 더 필요하다
       • 일반화가 더 낫다
       • 실서비스 모델에 선호된다

    4. 언제나 손실 곡선을 살펴라.
       • 매끄럽게 줄어들면 좋다
       • 진동하면 학습률이 너무 크다
       • 평평하면 학습률이 너무 작거나 이미 수렴했다

    5. 하나뿐인 "최고" 최적화기는 없다.
       • 문제, 구조, 데이터에 달렸다
       • 실험하고 견주어라
       • 더 나은 결과를 얻으려면 학습률 스케줄러를 쓰라

    다음 단계:
    → 학습률을 달리하여 실험해 보라
    → 학습률 스케줄러를 배워 보라
    → 최적화기와 정칙화를 함께 써 보라
    → 고급 최적화기(RAdam, Lookahead 등)를 살펴보라
    """)
    print("=" * 80)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (278줄)"

    ```
    ================================================================================
    COMPARING POPULAR OPTIMIZERS
    ================================================================================

    --------------------------------------------------------------------------------
    OPTIMIZATION ALGORITHMS OVERVIEW
    --------------------------------------------------------------------------------

    확률적 경사 하강법(SGD):
      • 가장 단순한 최적화기: param -= learning_rate × gradient
      • 장점: 단순하며 모멘텀과 잘 어울린다
      • 단점: 느릴 수 있고 학습률에 민감하다
      • 알맞은 곳: 잘 알려진 문제, 조율할 시간이 있을 때

    모멘텀을 쓰는 SGD:
      • "속도"를 더해 국소 최솟값을 뚫고 나간다
      • 기울기를 시간에 걸쳐 쌓는다: v = momentum × v + gradient
      • 잡음 섞인 기울기를 매끄럽게 한다
      • 알맞은 곳: 깊은 신경망, 잡음 섞인 기울기

    Adam(적응적 모멘트 추정):
      • 매개변수마다 배움 빠르기를 맞춘다
      • 모멘텀과 RMSprop을 합친다
      • 장점: 기본 설정으로도 잘 돌고 빠르게 수렴한다
      • 단점: SGD보다 일반화가 나쁠 수 있고 메모리를 더 쓴다
      • 알맞은 곳: 빠른 시제품 제작, 대부분의 딥러닝 과제

    AdamW(가중치 감쇠를 갖춘 Adam):
      • 가중치 감쇠를 올바로 구현한 Adam
      • Adam보다 일반화가 낫다
      • Adam의 가중치 감쇠 결함을 고친다
      • 알맞은 곳: 트랜스포머, 최신 구조, 가중치 감쇠를 쓸 때

    RMSprop(제곱평균제곱근 전파):
      • 제곱 기울기의 이동평균으로 학습률을 조절한다
      • 비정상 문제에 알맞다
      • 알맞은 곳: RNN, 온라인 학습


    --------------------------------------------------------------------------------
    SETUP: Non-Convex Optimization Problem
    --------------------------------------------------------------------------------
    Dataset: 200 samples
    Task: Learn the non-linear function y = sin(x) + 0.5*x

    --------------------------------------------------------------------------------
    NEURAL NETWORK ARCHITECTURE
    --------------------------------------------------------------------------------
    Architecture:
    SimpleNet(
      (fc1): Linear(in_features=1, out_features=20, bias=True)
      (fc2): Linear(in_features=20, out_features=20, bias=True)
      (fc3): Linear(in_features=20, out_features=1, bias=True)
      (relu): ReLU()
    )

    Total parameters: 481

    --------------------------------------------------------------------------------
    TRAINING WITH DIFFERENT OPTIMIZERS
    --------------------------------------------------------------------------------

    1. SGD (Vanilla - No Momentum):

    2. SGD with Momentum (0.9):

    3. RMSprop:

    4. Adam:

    5. AdamW (Adam with Weight Decay):
      [SGD] Epoch 20/100, Loss: 0.236019
      [SGD] Epoch 40/100, Loss: 0.105891
      [SGD] Epoch 60/100, Loss: 0.092962
      [SGD] Epoch 80/100, Loss: 0.084975
      [SGD] Epoch 100/100, Loss: 0.079076
      [SGD+Momentum] Epoch 20/100, Loss: 0.229655
      [SGD+Momentum] Epoch 40/100, Loss: 0.078116
      [SGD+Momentum] Epoch 60/100, Loss: 0.038757
      [SGD+Momentum] Epoch 80/100, Loss: 0.029874
      [SGD+Momentum] Epoch 100/100, Loss: 0.024268
      [RMSprop] Epoch 20/100, Loss: 0.073624
      [RMSprop] Epoch 40/100, Loss: 0.087447
      [RMSprop] Epoch 60/100, Loss: 0.061306
      [RMSprop] Epoch 80/100, Loss: 0.043914
      [RMSprop] Epoch 100/100, Loss: 0.042899
      [Adam] Epoch 20/100, Loss: 0.132862
      [Adam] Epoch 40/100, Loss: 0.088054
      [Adam] Epoch 60/100, Loss: 0.051410
      [Adam] Epoch 80/100, Loss: 0.033878
      [Adam] Epoch 100/100, Loss: 0.022868
      [AdamW] Epoch 20/100, Loss: 0.239450
      [AdamW] Epoch 40/100, Loss: 0.090572
      [AdamW] Epoch 60/100, Loss: 0.056621
      [AdamW] Epoch 80/100, Loss: 0.037305
      [AdamW] Epoch 100/100, Loss: 0.024713

    --------------------------------------------------------------------------------
    RESULTS ANALYSIS
    --------------------------------------------------------------------------------

    Final Loss (lower is better):
      SGD            : 0.079076
      SGD+Momentum   : 0.024268
      RMSprop        : 0.042899
      Adam           : 0.022868
      AdamW          : 0.024713

    Convergence Speed (loss after 20 epochs):
      SGD            : 0.236019
      SGD+Momentum   : 0.229655
      RMSprop        : 0.073624
      Adam           : 0.132862
      AdamW          : 0.239450

    --------------------------------------------------------------------------------
    VISUALIZATION
    --------------------------------------------------------------------------------
    Plot saved to: optimizer_comparison.png

    --------------------------------------------------------------------------------
    DETAILED OPTIMIZER MECHANICS
    --------------------------------------------------------------------------------

    1. SGD(확률적 경사 하강법):
       Update: θ = θ - lr × ∇L
       
       • 가장 단순한 알고리즘
       • 매 단계가 곧장 내리막으로 간다
       • 좁은 골짜기에서 진동할 수 있다
       • 학습률 선택에 매우 민감하다

    2. 모멘텀을 쓰는 SGD:
       Velocity: v = β × v + ∇L
       Update: θ = θ - lr × v
       
       • 지난 기울기를 쌓는다(β는 보통 0.9)
       • "관성"을 쌓아 작은 굴곡을 뚫고 나간다
       • 잡음 섞인 기울기를 매끄럽게 한다
       • 최적값을 지나칠 수 있다

    3. RMSPROP:
       RMS: s = β × s + (1-β) × ∇L²
       Update: θ = θ - (lr / √s) × ∇L
       
       • 매개변수마다 학습률을 조절한다
       • 기울기의 제곱평균제곱근으로 나눈다
       • 학습률이 너무 작아지는 것을 막는다
       • 비정상 목적함수에 알맞다

    4. Adam(적응적 모멘트 추정):
       Momentum: m = β₁ × m + (1-β₁) × ∇L
       Velocity: v = β₂ × v + (1-β₂) × ∇L²
       Update: θ = θ - lr × (m / √v)
       
       • 모멘텀과 RMSprop을 합친다
       • 매개변수마다 배움 빠르기를 맞춘다
       • 편향 보정을 포함한다
       • Default β₁=0.9, β₂=0.999
       • 딥러닝에서 가장 널리 쓰는 최적화기

    5. ADAMW:
       Adam과 같으나 가중치 감쇠를 바로잡았다.
       Update: θ = θ - lr × (m / √v) - lr × λ × θ
       
       • Adam의 가중치 감쇠 구현을 고친다
       • 정칙화가 더 낫다
       • 트랜스포머와 큰 모델에 선호된다


    --------------------------------------------------------------------------------
    PRACTICAL GUIDELINES: Which Optimizer to Use?
    --------------------------------------------------------------------------------

    🎯 Adam을 쓸 때:
       ✓ 새 프로젝트를 시작할 때(좋은 기본값)
       ✓ 빠른 수렴이 필요하다
       ✓ 초매개변수를 조율할 시간이 적다
       ✓ RNN이나 트랜스포머를 다룬다
       ✓ 예: 빠른 시제품 제작, 연구 실험

    📊 SGD + 모멘텀을 쓸 때:
       ✓ 최종 모델 학습(일반화가 더 나을 때가 많다)
       ✓ 학습률을 조율할 시간이 있다
       ✓ CNN 학습(특히 ResNet, VGG)
       ✓ 가장 안정적인 장기 성능을 원한다
       ✓ 예: 실서비스 모델, ImageNet 학습

    🔬 AdamW를 쓸 때:
       ✓ 트랜스포머 학습(BERT, GPT 등)
       ✓ 가중치 감쇠 정칙화를 쓴다
       ✓ Adam보다 나은 일반화가 필요하다
       ✓ 최신 모범 사례를 따른다
       ✓ 예: 자연어 처리 모델, 대규모 학습

    ⚡ RMSprop을 쓸 때:
       ✓ RNN 학습(예전부터 널리 썼다)
       ✓ 비정상 문제
       ✓ 온라인 학습 상황
       ✓ 예: 시계열, 온라인 추천

    학습률 권장값:
    • SGD: 0.01 - 0.1
    • SGD + 모멘텀: 0.01~0.1
    • Adam: 0.001 (1e-3)
    • AdamW: 0.0001 - 0.001 (1e-4 to 1e-3)
    • RMSprop: 0.001

    언제나 이 기본값으로 시작하여 손실 곡선을 보며 조정하라!


    --------------------------------------------------------------------------------
    HYPERPARAMETER REFERENCE
    --------------------------------------------------------------------------------

    LEARNING RATE (lr):
      • 가장 중요한 초매개변수
      • 너무 크면: 학습이 불안정하고 손실이 폭발한다
      • 너무 작으면: 학습이 너무 느리다
      • 학습 중 조정하려면 학습률 스케줄러를 쓴다

    MOMENTUM (SGD):
      • Typical: 0.9 or 0.95
      • 클수록 더 매끄럽지만 지나칠 수 있다

    베타(Adam과 AdamW):
      • β₁ (momentum): typically 0.9
      • β₂ (RMS): typically 0.999
      • 이 값을 바꿀 일은 거의 없다

    가중치 감쇠:
      • 정칙화 세기
      • Typical: 0.0001 to 0.01
      • 클수록 정칙화가 세다
      • 올바른 구현을 원하면 AdamW를 쓴다

    엡실론(Adam과 RMSprop):
      • 수치 안정을 위한 상수
      • Default: 1e-8
      • 바꿀 일이 거의 없다


    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. 최적화기마다 서로 다른 절충을 한다.
       • 속도 대 안정성
       • 쓰기 쉬움 대 최종 성능
       • 메모리 효율

    2. Adam과 AdamW는 훌륭한 기본값이다.
       • 빠른 수렴
       • 기본 설정으로도 잘 돈다
       • 최신 구조에는 AdamW가 선호된다

    3. SGD + 모멘텀이 최종 성능에서 가장 나을 때가 많다.
       • 조율이 더 필요하다
       • 일반화가 더 낫다
       • 실서비스 모델에 선호된다

    4. 언제나 손실 곡선을 살펴라.
       • 매끄럽게 줄어들면 좋다
       • 진동하면 학습률이 너무 크다
       • 평평하면 학습률이 너무 작거나 이미 수렴했다

    5. 하나뿐인 "최고" 최적화기는 없다.
       • 문제, 구조, 데이터에 달렸다
       • 실험하고 견주어라
       • 더 나은 결과를 얻으려면 학습률 스케줄러를 쓰라

    다음 단계:
    → 학습률을 달리하여 실험해 보라
    → 학습률 스케줄러를 배워 보라
    → 최적화기와 정칙화를 함께 써 보라
    → 고급 최적화기(RAdam, Lookahead 등)를 살펴보라

    ================================================================================
    ```


## 2. 논의

각 모델은 1→20→20→1 구조이며 전체 데이터를 한 번에 사용해 MSE를 줄인다. 따라서 이 실험에는 미니배치 추출에 따른 잡음이 없다. 별도의 검증·테스트 분할도 없어 학습 손실의 순위를 일반화 성능의 순위로 해석할 수 없다.

난수 시드를 한 번 고정한 뒤 모델을 차례로 생성하므로 구조가 같아도 초기 가중치는 서로 다르다. 모든 옵티마이저에 같은 학습률을 주는 설정 역시 각 방법을 동일하게 조율했다는 뜻은 아니다. 공정한 비교에는 같은 초기 상태에서 시작하는 반복 실험과 방법별 학습률 탐색이 필요하다.

이 코드에서 Adam은 가중치 감쇠를 지정하지 않고, AdamW에는 0.01을 지정한다. 관찰되는 차이에는 감쇠의 유무도 포함된다. Adam의 두 번째 모멘트는 기울기 제곱의 이동평균이며, 그 자체를 중심화된 분산이라고 부르면 부정확하다. 코드와 출력에 있는 특정 모델 계열에 대한 추천이나 성능 우위는 보편적인 보장이 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
SimpleNet의 학습 매개변수 수를 구하라.

</div>

??? success "연습문제 1 풀이"
    첫 층은 1×20+20=40개, 둘째 층은 20×20+20=420개, 출력층은 20×1+1=21개로 총 481개이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
난수 시드를 한 번 고정하는 것만으로 다섯 모델의 초기 가중치가 같아지는가?

</div>

??? success "연습문제 2 풀이"
    아니다. 모델을 생성할 때마다 난수 상태가 진행된다. 같은 기준 모델의 state_dict를 각 모델에 복사하면 동일한 초기 가중치에서 비교할 수 있다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
최종 학습 손실이 가장 작은 옵티마이저의 일반화 성능이 가장 좋다고 할 수 있는가?

</div>

??? success "연습문제 3 풀이"
    그럴 수 없다. 모델 선택에는 검증셋이 필요하고, 선택 후 독립적인 테스트셋에서 평가해야 한다. 같은 초기 상태, 계산 예산과 방법별 조율 범위를 정하고 여러 시드의 변동도 함께 비교한다.

## 정리하며

학습 손실 곡선은 해당 설정에서의 최적화 경로를 보여 준다. 초기 가중치, 학습률과 가중치 감쇠가 다르면 그 영향을 알고리즘 자체의 효과와 분리해서 해석해야 한다.
