# 간단한 이진 분류

합성 이진 데이터를 단일 선형 계층과 시그모이드로 분류한다. 확률 출력에 BCE를 적용하고 임곗값에 따라 예측 라벨을 만든다.

## 1. 코드

??? note "코드 (549줄)"

    ```python
    """
    ================================================================================
    02_simple_binary_classification.py - 첫 로지스틱 회귀 모델
    ================================================================================

    학습 목표:
    - 둘 분류 문제를 이해한다
    - PyTorch로 로지스틱 회귀를 밑바닥부터 짠다
    - 시그모이드 함수와 그 결을 배운다
    - 경사 하강법으로 모델을 익힌다
    - 모델의 성능을 평가한다

    PREREQUISITES:
    - 01_introduction.py을 마쳤을 것
    - 선형 모델(y = mx + b) 이해
    - 기본 확률 개념

    소요 시간: 45분쯤

    어려움: ⭐⭐☆☆☆ (쉬움)
    ================================================================================
    """

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("="*80)
    print("PART 1: UNDERSTANDING THE PROBLEM")
    print("="*80)

    # ============================================================================
    # 1.1: 이진 분류
    # ============================================================================
    print("\n1.1: What is Binary Classification?")
    print("-" * 40)

    print("""
    둘 분류: 두 클래스(0 또는 1) 가운데 하나를 예측한다

    Examples:
      - 전자우편: 광고(1)인가 아닌가(0)
      - 의료: 병(1)인가 건강(0)인가
      - 손님: 살 것(1)인가 안 살 것(0)인가
      - 그림: 고양이(1)인가 개(0)인가

    이 학습에서는
      - 피처 2개(x1, x2)를 지닌 인공 데이터를 만든다
      - 표본마다 클래스 0이나 클래스 1에 든다
      - 목표: 피처에서 클래스를 예측하는 법을 배운다
    """)

    # ============================================================================
    # 1.2: 합성 데이터 생성
    # ============================================================================
    print("\n1.2: Generating Dataset")
    print("-" * 40)

    # 재현성을 위한 난수 시드 설정
    torch.manual_seed(42)
    np.random.seed(42)

    # 간단한 2차원 이진 분류 데이터셋을 생성한다
    # n_samples: 데이터 점의 개수
    # n_features: 입력 피처의 개수 (시각화하기 쉽게 2로 잡는다)
    # n_classes: 2 (이진 분류)
    # n_clusters_per_class: 클래스들이 얼마나 "떨어져" 있는지
    X, y = make_classification(
        n_samples=200,           # Total number of examples
        n_features=2,            # 2D data (x1, x2) for easy plotting
        n_redundant=0,           # No redundant features
        n_informative=2,         # Both features are informative
        n_clusters_per_class=1,  # Single cluster per class
        random_state=42,
        flip_y=0.1              # Add 10% noise (some mislabeled examples)
    )

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"X (features): {X.shape[0]} samples × {X.shape[1]} features")
    print(f"y (labels): {y.shape[0]} labels")
    print(f"Class distribution: Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")

    # 학습 집합과 시험 집합으로 나눈다
    # 학습: 80%, 시험: 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nAfter split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # PyTorch 텐서로 변환
    # 중요: float32(PyTorch의 기본값)로 바꾼다
    X_train = torch.FloatTensor(X_train)  # Shape: (160, 2)
    X_test = torch.FloatTensor(X_test)    # Shape: (40, 2)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)  # Shape: (160, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)    # Shape: (40, 1)

    print(f"\nTensor shapes:")
    print(f"X_train: {X_train.shape} (160 samples, 2 features)")
    print(f"y_train: {y_train.shape} (160 labels, reshaped to column vector)")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")


    print("\n" + "="*80)
    print("PART 2: THE LOGISTIC REGRESSION MODEL")
    print("="*80)

    # ============================================================================
    # 2.1: 모델 이해하기
    # ============================================================================
    print("\n2.1: Model Architecture")
    print("-" * 40)

    print("""
    로지스틱 회귀 모델:

    스텝 1: 선형 아우르기
        z = w1*x1 + w2*x2 + b
        여기서 w1, w2은 웨이트이고 b은 바이어스다

    스텝 2: 시그모이드 살림
        확률 = sigmoid(z) = 1 / (1 + e^(-z))

    시그모이드의 결:
        - 어떤 값이든 (0, 1) 범위로 옮긴다
        - sigmoid(0) = 0.5(분류 테두리)
        - sigmoid(큰 양수) ≈ 1
        - sigmoid(큰 음수) ≈ 0

    스텝 3: 분류
        확률 >= 0.5이면 클래스 1으로 예측한다
        확률 < 0.5이면 클래스 0으로 예측한다
    """)

    # ============================================================================
    # 2.2: 모델 구현하기
    # ============================================================================
    print("\n2.2: Implementing in PyTorch")
    print("-" * 40)

    class LogisticRegression(nn.Module):
        """
        단순한 로지스틱 회귀 모델

        Architecture:
            입력(n_features) → 선형 레이어 → 시그모이드 → 출력(확률)

        Parameters:
            n_features (int): 입력 피처의 수
        """

        def __init__(self, n_features):
            super(LogisticRegression, self).__init__()

            # 선형 레이어: y = xW^T + b
            # in_features: 입력 피처의 개수 (여기서는 2)
            # out_features: 출력의 개수 (이진 분류에서는 1)
            self.linear = nn.Linear(n_features, 1)

            # 선형 레이어가 만드는 것:
            # - self.linear.weight: 모양 (1, n_features) - 웨이트
            # - self.linear.bias: 모양 (1,) - 바이어스 항

        def forward(self, x):
            """
            망을 지나는 순전파

            Args:
                x: 모양이 (batch_size, n_features)인 입력 텐서

            Returns:
                probability: 모양이 (batch_size, 1)인 출력 텐서
                            값은 (0, 1) 범위에 있다
            """
            # 1단계: 선형 변환
            # x 꼴: (batch_size, n_features)
            # 출력 꼴: (batch_size, 1)
            z = self.linear(x)  # z = w*x + b

            # 2단계: 시그모이드 활성화 적용
            # 시그모이드는 z를 (0, 1) 범위의 확률로 보낸다
            probability = torch.sigmoid(z)

            return probability

    # 모델 인스턴스 생성
    n_features = 2  # We have 2 features (x1, x2)
    model = LogisticRegression(n_features)

    print("Model created!")
    print(f"Model structure:\n{model}")
    print(f"\nInitial weights: {model.linear.weight.data}")
    print(f"Initial bias: {model.linear.bias.data}")


    print("\n" + "="*80)
    print("PART 3: LOSS FUNCTION AND OPTIMIZER")
    print("="*80)

    # ============================================================================
    # 3.1: 이진 교차 엔트로피 로스
    # ============================================================================
    print("\n3.1: Understanding the Loss Function")
    print("-" * 40)

    print("""
    둘 교차 엔트로피(BCE) 로스:

    보기 하나에 대해
        참 레이블가 y = 1이면
            로스 = -log(예측 확률)
            → 클래스 1의 확률을 낮게 예측하면 모델이 벌을 받는다

        참 레이블가 y = 0이면
            로스 = -log(1 - 예측 확률)
            → 클래스 1의 확률을 높게 예측하면 모델이 벌을 받는다

    온 식:
        로스 = -[y*log(p) + (1-y)*log(1-p)]

    Properties:
        - 늘 양수다
        - 작을수록 좋다
        - 자신 있게 틀린 예측을 크게 벌한다
    """)

    # 로스 함수를 만든다
    # BCELoss: 이진 교차 엔트로피 로스
    # 예상: 예측과 목표가 모두 [0, 1] 범위의 확률
    criterion = nn.BCELoss()

    print("Loss function: Binary Cross-Entropy (BCE)")

    # ============================================================================
    # 3.2: 최적화기
    # ============================================================================
    print("\n3.2: Choosing an Optimizer")
    print("-" * 40)

    print("""
    최적화기: 모델 파라메터(웨이트와 바이어스)를 고치는 알고리즘

    흔한 최적화기:
        - SGD(확률 경사 하강법): 기본이지만 믿을 만하다
        - Adam: 맞추어 가는 러닝레이트. 대개 더 빨리 모여든다
        - RMSprop: 되도는 뉴럴 넷트웍에 좋다

    여기서는 단순하고 이해하기 쉽도록 SGD을 쓴다.
    """)

    learning_rate = 0.1  # How big each update step is
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(f"Optimizer: SGD with learning rate = {learning_rate}")


    print("\n" + "="*80)
    print("PART 4: TRAINING THE MODEL")
    print("="*80)

    # ============================================================================
    # 4.1: 학습 루프
    # ============================================================================
    print("\n4.1: Training Loop")
    print("-" * 40)

    num_epochs = 1000  # Number of times to see the entire dataset
    print_every = 100  # Print progress every N epochs

    # 학습 기록을 담을 리스트들
    train_losses = []
    train_accuracies = []

    print(f"Starting training for {num_epochs} epochs...")
    print("-" * 40)

    for epoch in range(num_epochs):
        # ====================
        # 학습 단계
        # ====================

        # 1. 순전파: 예측을 계산한다
        # X_train 꼴: (160, 2)
        # predictions 꼴: (160, 1)
        predictions = model(X_train)  # Get model's predicted probabilities

        # 2. 로스 계산
        # 예측을 참 레이블와 비교한다
        loss = criterion(predictions, y_train)

        # 3. 역전파: 경사를 계산한다
        optimizer.zero_grad()  # Clear old gradients (important!)
        loss.backward()        # Compute new gradients

        # 4. 파라메터 갱신
        optimizer.step()       # Update weights and bias using gradients

        # ====================
        # 진행 상황 추적
        # ====================

        # 학습 정확도를 계산한다
        with torch.no_grad():  # Don't compute gradients for evaluation
            # 확률을 클래스 예측(0 또는 1)으로 바꾼다
            # 확률이 0.5 이상이면 1로, 아니면 0으로 예측한다
            predicted_classes = (predictions >= 0.5).float()

            # 정확도를 계산한다: 맞힌 예측의 비율
            correct = (predicted_classes == y_train).sum()
            accuracy = (correct / y_train.shape[0]).item()

        # 이력 저장
        train_losses.append(loss.item())
        train_accuracies.append(accuracy)

        # 진행 상황 출력
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Loss: {loss.item():.4f} "
                  f"Accuracy: {accuracy:.4f}")

    print("\nTraining completed!")
    print(f"Final loss: {train_losses[-1]:.4f}")
    print(f"Final training accuracy: {train_accuracies[-1]:.4f}")


    print("\n" + "="*80)
    print("PART 5: EVALUATING THE MODEL")
    print("="*80)

    # ============================================================================
    # 5.1: 시험 집합 평가
    # ============================================================================
    print("\n5.1: Performance on Test Set")
    print("-" * 40)

    # 시험 집합(보지 않은 데이터)에서 평가한다
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Don't compute gradients during evaluation
        # 시험 집합의 예측을 얻는다
        test_predictions = model(X_test)

        # 확률을 클래스 예측으로 바꾼다
        test_predicted_classes = (test_predictions >= 0.5).float()

        # 시험 정확도를 계산한다
        test_correct = (test_predicted_classes == y_test).sum()
        test_accuracy = (test_correct / y_test.shape[0]).item()

        # 시험 로스를 계산한다
        test_loss = criterion(test_predictions, y_test)

    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Correct predictions: {int(test_correct)}/{len(y_test)}")


    print("\n" + "="*80)
    print("PART 6: VISUALIZATION")
    print("="*80)

    # ============================================================================
    # 6.1: 학습 기록
    # ============================================================================
    print("\n6.1: Creating Visualizations...")

    fig = plt.figure(figsize=(15, 10))

    # 그림 1: 로스 곡선
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 그림 2: 정확도 곡선
    plt.subplot(2, 3, 2)
    plt.plot(train_accuracies, 'g-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])

    # 그림 3: 데이터 분포
    plt.subplot(2, 3, 3)
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy().flatten()
    plt.scatter(X_train_np[y_train_np==0, 0], X_train_np[y_train_np==0, 1], 
               c='blue', label='Class 0', alpha=0.6, edgecolors='k')
    plt.scatter(X_train_np[y_train_np==1, 0], X_train_np[y_train_np==1, 1], 
               c='red', label='Class 1', alpha=0.6, edgecolors='k')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Training Data Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 그림 4: 결정 경계
    plt.subplot(2, 3, 4)
    # 결정 경계를 그리기 위한 격자를 만든다
    x_min, x_max = X_train_np[:, 0].min() - 1, X_train_np[:, 0].max() + 1
    y_min, y_max = X_train_np[:, 1].min() - 1, X_train_np[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # 격자에서 예측한다
    with torch.no_grad():
        Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape).numpy()

    plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
    plt.colorbar(label='Probability')
    plt.scatter(X_train_np[y_train_np==0, 0], X_train_np[y_train_np==0, 1], 
               c='blue', label='Class 0', edgecolors='k', s=50)
    plt.scatter(X_train_np[y_train_np==1, 0], X_train_np[y_train_np==1, 1], 
               c='red', label='Class 1', edgecolors='k', s=50)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Decision Boundary', fontsize=14, fontweight='bold')
    plt.legend()

    # 그림 5: 모델 파라메터
    plt.subplot(2, 3, 5)
    weights = model.linear.weight.data.numpy().flatten()
    bias = model.linear.bias.data.numpy()[0]
    params_text = f"Learned Parameters:\n\n"
    params_text += f"Weight 1 (w1): {weights[0]:.3f}\n"
    params_text += f"Weight 2 (w2): {weights[1]:.3f}\n"
    params_text += f"Bias (b): {bias:.3f}\n\n"
    params_text += f"Decision boundary:\n"
    params_text += f"{weights[0]:.3f}*x1 + {weights[1]:.3f}*x2 + {bias:.3f} = 0"
    plt.text(0.1, 0.5, params_text, fontsize=12, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')
    plt.title('Model Parameters', fontsize=14, fontweight='bold')

    # 그림 6: 성능 요약
    plt.subplot(2, 3, 6)
    summary_text = f"Performance Summary\n\n"
    summary_text += f"Training:\n"
    summary_text += f"  Loss: {train_losses[-1]:.4f}\n"
    summary_text += f"  Accuracy: {train_accuracies[-1]:.4f}\n\n"
    summary_text += f"Testing:\n"
    summary_text += f"  Loss: {test_loss.item():.4f}\n"
    summary_text += f"  Accuracy: {test_accuracy:.4f}\n\n"
    summary_text += f"Dataset:\n"
    summary_text += f"  Training samples: {len(X_train)}\n"
    summary_text += f"  Test samples: {len(X_test)}\n"
    summary_text += f"  Features: {n_features}\n\n"
    summary_text += f"Training:\n"
    summary_text += f"  Epochs: {num_epochs}\n"
    summary_text += f"  Learning rate: {learning_rate}"

    plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    plt.axis('off')
    plt.title('Summary', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('simple_classification_results.png',
                dpi=150, bbox_inches='tight')
    print("Visualization saved as: simple_classification_results.png")


    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)

    print("""
    1. 로지스틱 회귀
       - 선형 모델 + 시그모이드 살림
       - 0과 1 사이의 확률을 내놓는다
       - 둘 분류에서는 0.5을 임계로 삼는다

    2. 학습 과정
       - 순전파: 예측을 계산한다
       - 로스 계산: 오차를 잰다
       - 역전파: 그래디언트를 계산한다
       - 파라메터 고치기: 모델을 낫게 한다

    3. 종요로운 개념
       - backward() 앞에는 늘 optimizer.zero_grad()을 불러라
       - 따질 때는 model.eval()을 써라
       - 익히지 않을 때는 torch.no_grad()을 써라

    4. EVALUATION
       - 학습 배치: 모델이 배운 데이터
       - 시험 배치: 두루 얼마나 잘 미치는지
       - 정확도: 옳게 예측한 비율
    """)


    print("\n" + "="*80)
    print("EXERCISES")
    print("="*80)

    print("""
    1. 쉬움: learning_rate을 0.01과 1.0으로 바꾸어라
       - 모여듦에 어떤 영향을 주는가?
       - 어느 쪽이 더 빨리 배우는가?

    2. 보통: num_epochs을 100과 5000으로 바꾸어라
       - 더 익히면 늘 나아지는가?
       - 지나치게 맞춰진 확률을 살펴라

    3. 보통: 학습/시험 나누기를 바꾸어 보아라
       - test_size을 0.1과 0.5으로 바꾸어라
       - 결과에 어떤 영향을 주는가?

    4. 어려움: 새 데이터를 예측하는 함수를 짜라.
       def predict(model, x1, x2):
           # 여기에 코드를 작성한다
           pass

       이렇게 시험하라: predict(model, 0.5, 0.5)

    5. 어려움: 데이터셋에 피처를 더하여라.
       - n_features=4이나 10을 쓴다
       - 모델도 그에 맞게 고친다
       - 성능을 견준다
    """)

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
    잘했다! 첫 로지스틱 회귀 모델을 지었다!

    다음 튜토리얼: 03_with_sklearn_data.py
    - 참 세상 데이터셋을 다룬다
    - 데이터를 미리 다듬는 기법을 배운다
    - 여러 데이터 클래스를 다룬다

    채비가 되었는가? 다음을 돌려라: python 03_with_sklearn_data.py
    """)
    print("="*80)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (227줄)"

    ```
    ================================================================================
    PART 1: UNDERSTANDING THE PROBLEM
    ================================================================================

    1.1: What is Binary Classification?
    ----------------------------------------

    둘 분류: 두 클래스(0 또는 1) 가운데 하나를 예측한다

    Examples:
      - 전자우편: 광고(1)인가 아닌가(0)
      - 의료: 병(1)인가 건강(0)인가
      - 손님: 살 것(1)인가 안 살 것(0)인가
      - 그림: 고양이(1)인가 개(0)인가

    이 학습에서는
      - 피처 2개(x1, x2)를 지닌 인공 데이터를 만든다
      - 표본마다 클래스 0이나 클래스 1에 든다
      - 목표: 피처에서 클래스를 예측하는 법을 배운다


    1.2: Generating Dataset
    ----------------------------------------
    Dataset shape: X=(200, 2), y=(200,)
    X (features): 200 samples × 2 features
    y (labels): 200 labels
    Class distribution: Class 0: 99, Class 1: 101

    After split:
    Training set: 160 samples
    Test set: 40 samples

    Tensor shapes:
    X_train: torch.Size([160, 2]) (160 samples, 2 features)
    y_train: torch.Size([160, 1]) (160 labels, reshaped to column vector)
    X_test: torch.Size([40, 2])
    y_test: torch.Size([40, 1])

    ================================================================================
    PART 2: THE LOGISTIC REGRESSION MODEL
    ================================================================================

    2.1: Model Architecture
    ----------------------------------------

    로지스틱 회귀 모델:

    스텝 1: 선형 아우르기
        z = w1*x1 + w2*x2 + b
        여기서 w1, w2은 웨이트이고 b은 바이어스다

    스텝 2: 시그모이드 살림
        확률 = sigmoid(z) = 1 / (1 + e^(-z))
        
    시그모이드의 결:
        - 어떤 값이든 (0, 1) 범위로 옮긴다
        - sigmoid(0) = 0.5(분류 테두리)
        - sigmoid(큰 양수) ≈ 1
        - sigmoid(큰 음수) ≈ 0
        
    스텝 3: 분류
        확률 >= 0.5이면 클래스 1으로 예측한다
        확률 < 0.5이면 클래스 0으로 예측한다


    2.2: Implementing in PyTorch
    ----------------------------------------
    Model created!
    Model structure:
    LogisticRegression(
      (linear): Linear(in_features=2, out_features=1, bias=True)
    )

    Initial weights: tensor([[0.5406, 0.5869]])
    Initial bias: tensor([-0.1657])

    ================================================================================
    PART 3: LOSS FUNCTION AND OPTIMIZER
    ================================================================================

    3.1: Understanding the Loss Function
    ----------------------------------------

    둘 교차 엔트로피(BCE) 로스:

    보기 하나에 대해
        참 레이블가 y = 1이면
            로스 = -log(예측 확률)
            → 클래스 1의 확률을 낮게 예측하면 모델이 벌을 받는다
        
        참 레이블가 y = 0이면
            로스 = -log(1 - 예측 확률)
            → 클래스 1의 확률을 높게 예측하면 모델이 벌을 받는다

    온 식:
        로스 = -[y*log(p) + (1-y)*log(1-p)]

    Properties:
        - 늘 양수다
        - 작을수록 좋다
        - 자신 있게 틀린 예측을 크게 벌한다

    Loss function: Binary Cross-Entropy (BCE)

    3.2: Choosing an Optimizer
    ----------------------------------------

    최적화기: 모델 파라메터(웨이트와 바이어스)를 고치는 알고리즘

    흔한 최적화기:
        - SGD(확률 경사 하강법): 기본이지만 믿을 만하다
        - Adam: 맞추어 가는 러닝레이트. 대개 더 빨리 모여든다
        - RMSprop: 되도는 뉴럴 넷트웍에 좋다
        
    여기서는 단순하고 이해하기 쉽도록 SGD을 쓴다.

    Optimizer: SGD with learning rate = 0.1

    ================================================================================
    PART 4: TRAINING THE MODEL
    ================================================================================

    4.1: Training Loop
    ----------------------------------------
    Starting training for 1000 epochs...
    ----------------------------------------
    Epoch [100/1000] Loss: 0.5013 Accuracy: 0.7875
    Epoch [200/1000] Loss: 0.4955 Accuracy: 0.7750
    Epoch [300/1000] Loss: 0.4940 Accuracy: 0.7875
    Epoch [400/1000] Loss: 0.4936 Accuracy: 0.7875
    Epoch [500/1000] Loss: 0.4935 Accuracy: 0.7937
    Epoch [600/1000] Loss: 0.4934 Accuracy: 0.7937
    Epoch [700/1000] Loss: 0.4934 Accuracy: 0.7937
    Epoch [800/1000] Loss: 0.4934 Accuracy: 0.7937
    Epoch [900/1000] Loss: 0.4934 Accuracy: 0.8000
    Epoch [1000/1000] Loss: 0.4934 Accuracy: 0.7937

    Training completed!
    Final loss: 0.4934
    Final training accuracy: 0.7937

    ================================================================================
    PART 5: EVALUATING THE MODEL
    ================================================================================

    5.1: Performance on Test Set
    ----------------------------------------
    Test Loss: 0.4758
    Test Accuracy: 0.8500
    Correct predictions: 34/40

    ================================================================================
    PART 6: VISUALIZATION
    ================================================================================

    6.1: Creating Visualizations...
    Visualization saved as: simple_classification_results.png

    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. 로지스틱 회귀
       - 선형 모델 + 시그모이드 살림
       - 0과 1 사이의 확률을 내놓는다
       - 둘 분류에서는 0.5을 임계로 삼는다

    2. 학습 과정
       - 순전파: 예측을 계산한다
       - 로스 계산: 오차를 잰다
       - 역전파: 그래디언트를 계산한다
       - 파라메터 고치기: 모델을 낫게 한다

    3. 종요로운 개념
       - backward() 앞에는 늘 optimizer.zero_grad()을 불러라
       - 따질 때는 model.eval()을 써라
       - 익히지 않을 때는 torch.no_grad()을 써라

    4. EVALUATION
       - 학습 배치: 모델이 배운 데이터
       - 시험 배치: 두루 얼마나 잘 미치는지
       - 정확도: 옳게 예측한 비율


    ================================================================================
    EXERCISES
    ================================================================================

    1. 쉬움: learning_rate을 0.01과 1.0으로 바꾸어라
       - 모여듦에 어떤 영향을 주는가?
       - 어느 쪽이 더 빨리 배우는가?

    2. 보통: num_epochs을 100과 5000으로 바꾸어라
       - 더 익히면 늘 나아지는가?
       - 지나치게 맞춰진 확률을 살펴라

    3. 보통: 학습/시험 나누기를 바꾸어 보아라
       - test_size을 0.1과 0.5으로 바꾸어라
       - 결과에 어떤 영향을 주는가?

    4. 어려움: 새 데이터를 예측하는 함수를 짜라.
       def predict(model, x1, x2):
           # 여기에 코드를 작성한다
           pass
       
       이렇게 시험하라: predict(model, 0.5, 0.5)

    5. 어려움: 데이터셋에 피처를 더하여라.
       - n_features=4이나 10을 쓴다
       - 모델도 그에 맞게 고친다
       - 성능을 견준다


    ================================================================================
    NEXT STEPS
    ================================================================================

    잘했다! 첫 로지스틱 회귀 모델을 지었다!

    다음 튜토리얼: 03_with_sklearn_data.py
    - 참 세상 데이터셋을 다룬다
    - 데이터를 미리 다듬는 기법을 배운다
    - 여러 데이터 클래스를 다룬다

    채비가 되었는가? 다음을 돌려라: python 03_with_sklearn_data.py

    ================================================================================
    ```


## 2. 논의

### 확률 모델

모델은 $p(y=1\mid x)=\sigma(w^\top x+b)$를 계산한다. 피처가 $d$개이면 학습 파라미터는 $d+1$개다. `BCELoss`는 로짓이 아니라 확률을 입력으로 받고 정답은 출력과 같은 모양의 0·1 실수 텐서다.

### 로스와 결정의 차이

0.5 임곗값은 로짓의 부호를 기준으로 결정한다. 같은 라벨을 예측해도 확률의 확신 정도에 따라 BCE는 달라진다. 정확도와 로스, 클래스별 오차를 함께 본다. 불균형 데이터의 적절한 임곗값은 검증 목적과 오류 비용에 따라 달라질 수 있다.

### 시각화와 해석

웨이트는 선형 경계의 법선 방향을 정한다. 그림에서 분리되어 보이는 것과 독립 표본의 성능은 별개다. 러닝레이트와 반복 수를 바꿔도 볼록성만으로 모든 설정의 수렴이나 유한한 MLE가 보장되지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
입력 피처 수가 $d$인 단일 선형 이진 분류기의 파라미터 수와 출력 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"
    웨이트 $d$개와 바이어스 1개로 총 $d+1$개다. 배치 입력 $(B,d)$에 대한 출력은 $(B,1)$이다. 시그모이드에는 학습 파라미터가 없다. 유방암 데이터의 피처 30개를 쓰는 경우에는 31개다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 옵티마이저와 학습 수렴을 비교하라. 두 로스 곡선을 같은 그래프에 그려라.

</div>

??? success "연습문제 2 풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 경사와 경사 제곱의 이동 평균으로 성분별 업데이트를 조정한다. 어느 옵티마이저가 더 빨리 수렴하는지는 문제와 러닝레이트에 달려 있다. 같은 초기 웨이트와 데이터 순서, 평가 기준을 사용하고 각 러닝레이트를 조정해 비교한다. 여러 시드의 변동과 검증 성능도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
정확도는 같지만 BCE가 다른 두 모델을 어떻게 해석하는가?

</div>

??? success "연습문제 3 풀이"
    임곗값에 따른 예측 클래스는 같아도 확률의 크기와 오분류에 대한 확신이 다를 수 있다. 로스와 클래스별 오차, 확률 보정을 함께 평가하며 같은 정확도만으로 동등한 모델이라고 판단하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 일반적인 계층 확장 코드가 이 페이지의 원래 클래스에 그대로 적용되는지 검토하라. 모델과 데이터셋의 역할, 생성자 인수, 입력·출력 모양과 비선형성에서 필요한 수정 사항을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 일반적인 ModuleList 구성 패턴의 일부이며 이 페이지 클래스의 완성된 대체 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    적용하려면 생성자와 `forward`를 함께 정의하고 마지막 출력 모양과 로스의 입력 조건을 유지해야 한다. 선형 계층만 여러 개 합성하면 여전히 하나의 선형 사상이므로 표현력을 늘리려면 적절한 비선형성이 필요하다. Dataset은 표본 접근 인터페이스이므로 계층을 등록할 대상이 아니며 별도 모델 클래스에 이 구성을 적용한다.

## 정리하며

모델 출력과 로스의 입력 조건을 일치시키고 데이터 분할, 전처리와 지표의 의미를 함께 확인한다. 기록된 실행 결과는 해당 설정의 예시이며 모든 환경의 수렴이나 일반화를 보장하지 않는다.
