# 다중 클래스 분류

여러 클래스 중 하나를 예측하는 선형 분류기를 구성한다. 학습용 로짓과 해석용 확률, 클래스 인덱스 정답의 입력 조건을 구분한다.

## 1. 코드

??? note "코드 (472줄)"

    ```python
    """
    ==============================================================================
    02_multiclass_classification.py - 소프트맥스와 여러 클래스 문제
    ================================================================================

    학습 목표:
    - 로지스틱 회귀를 여러 클래스로 넓힌다
    - 소프트맥스 살림을 이해한다
    - CrossEntropyLoss을 쓴다
    - 원핫 부호를 다룬다
    - 여러 클래스 모델을 평가한다

    소요 시간: 1시간 반쯤
    어려움: ⭐⭐⭐⭐☆ (앞선)
    ================================================================================
    """

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("="*80)
    print("MULTI-CLASS CLASSIFICATION")
    print("="*80)

    # =============================================================================
    # 1부: 이진 대 다중 클래스
    # =============================================================================

    print("\n" + "="*80)
    print("PART 1: UNDERSTANDING MULTI-CLASS CLASSIFICATION")
    print("="*80)

    print("""
    둘 분류:
      2 classes: 0 or 1
      출력: 확률 하나
      살림: 시그모이드
      로스: 둘 교차 엔트로피(BCE)

    여러 클래스 분류:
      K classes: 0, 1, 2, ..., K-1
      출력: 확률 K개(합이 1이다)
      살림: 소프트맥스
      로스: 교차 엔트로피

    소프트맥스 함수:
      클래스가 K개일 때 소프트맥스는 로짓을 확률로 바꾼다.

      P(class=k) = exp(logit_k) / sum(exp(logit_j) for all j)

      Properties:
      ✓ 확률이 모두 양수다
      ✓ 확률의 합이 1이다
      ✓ Differentiable
    """)

    # =============================================================================
    # 2부: 다중 클래스 데이터 생성
    # =============================================================================

    print("\n" + "="*80)
    print("PART 2: PREPARING MULTI-CLASS DATA")
    print("="*80)

    # 3개 클래스 데이터셋을 생성한다
    n_classes = 3
    n_samples = 1500

    torch.manual_seed(42)
    np.random.seed(42)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )

    print(f"Dataset: {n_samples} samples, {n_classes} classes")
    print(f"Class distribution:")
    for i in range(n_classes):
        count = (y == i).sum()
        pct = 100 * count / len(y)
        print(f"  Class {i}: {count:4d} samples ({pct:.1f}%)")

    # 나누고 표준화한다
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = torch.FloatTensor(scaler.fit_transform(X_train))
    X_test = torch.FloatTensor(scaler.transform(X_test))

    # 중요: CrossEntropyLoss에서 목표는 클래스 인덱스(Long)여야 한다
    # 원-핫 부호가 아니다!
    y_train = torch.LongTensor(y_train)  # Shape: (N,) with values 0, 1, 2
    y_test = torch.LongTensor(y_test)

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape} (class indices)")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")

    # DataLoader들을 만든다
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=64,
        shuffle=False
    )

    # =============================================================================
    # 3부: 다중 클래스 모델
    # =============================================================================

    print("\n" + "="*80)
    print("PART 3: MULTI-CLASS LOGISTIC REGRESSION MODEL")
    print("="*80)

    class MultiClassLogisticRegression(nn.Module):
        """
        여러 클래스 로지스틱 회귀(소프트맥스 회귀)

        Args:
            input_dim: 입력 피처의 수
            num_classes: 출력 클래스의 수
        """

        def __init__(self, input_dim: int, num_classes: int):
            super().__init__()
            # 선형 레이어: (input_dim) -> (num_classes)
            # 각 클래스가 자기만의 피처 선형결합을 갖는다
            self.linear = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            """
            순전파

            Args:
                x: 모양이 (batch_size, input_dim)인 입력 텐서

            Returns:
                logits: 모양이 (batch_size, num_classes)인 날것 점수
                       눈여겨볼 것: 확률이 아니라 로짓을 돌려준다!
                       CrossEntropyLoss이 안에서 소프트맥스를 건다
            """
            logits = self.linear(x)  # Shape: (batch_size, num_classes)
            return logits

        def predict_proba(self, x):
            """클래스 확률을 얻는다 (추론용)"""
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities


    # 모델 생성
    input_dim = X_train.shape[1]
    model = MultiClassLogisticRegression(input_dim, n_classes)

    print(f"Model created:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of classes: {n_classes}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"\nModel architecture:")
    print(model)

    # =============================================================================
    # 4부: 로스 함수와 학습
    # =============================================================================

    print("\n" + "="*80)
    print("PART 4: TRAINING MULTI-CLASS MODEL")
    print("="*80)

    print("""
    CrossEntropyLoss:
      - LogSoftmax과 NLLLoss을 아우른다
      - Expects:
        * 예측: (batch_size, num_classes) - 날것 로짓
        * 타깃: (batch_size,) - 클래스 번호(Long 텐서)
      - 소프트맥스를 절로 건다
      - 직접 하는 소프트맥스 + 로그보다 수치가 든든하다
    """)

    # 준비
    criterion = nn.CrossEntropyLoss()  # Combines softmax + log + NLL
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 100

    # 학습 기록
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 60)

    for epoch in range(num_epochs):
        # 학습
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            # 순전파
            logits = model(batch_X)  # (batch_size, num_classes)
            loss = criterion(logits, batch_y)  # Expects logits and class indices

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 정확도 계산
            _, predicted = torch.max(logits, 1)  # Get class with highest logit
            train_loss += loss.item() * len(batch_X)
            correct += (predicted == batch_y).sum().item()
            total += len(batch_X)

        avg_train_loss = train_loss / total
        train_acc = correct / total

        # 검증
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                logits = model(batch_X)
                loss = criterion(logits, batch_y)

                _, predicted = torch.max(logits, 1)
                test_loss += loss.item() * len(batch_X)
                correct += (predicted == batch_y).sum().item()
                total += len(batch_X)

        avg_test_loss = test_loss / total
        test_acc = correct / total

        # 이력 저장
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)

        # 진행 상황 출력
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {avg_test_loss:.4f} Acc: {test_acc:.4f}")

    print("\n✓ Training completed!")

    # =============================================================================
    # 5부: 평가
    # =============================================================================

    print("\n" + "="*80)
    print("PART 5: COMPREHENSIVE EVALUATION")
    print("="*80)

    # 예측을 얻는다
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            logits = model(batch_X)
            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)

    # 분류 보고서
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions,
                              target_names=[f'Class {i}' for i in range(n_classes)]))

    # 혼동 행렬
    cm = confusion_matrix(all_targets, all_predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # =============================================================================
    # 6부: 시각화
    # =============================================================================

    print("\n" + "="*80)
    print("PART 6: VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(16, 10))

    # 그림 1: 학습 곡선
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 그림 2: 정확도 곡선
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(history['test_acc'], label='Test Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 그림 3: 혼동 행렬
    ax3 = plt.subplot(2, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=[f'C{i}' for i in range(n_classes)],
                yticklabels=[f'C{i}' for i in range(n_classes)])
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    ax3.set_title('Confusion Matrix', fontweight='bold')

    # 그림 4: 클래스별 정확도
    ax4 = plt.subplot(2, 3, 4)
    per_class_acc = []
    for i in range(n_classes):
        mask = all_targets == i
        acc = (all_predictions[mask] == all_targets[mask]).mean()
        per_class_acc.append(acc)

    bars = ax4.bar(range(n_classes), per_class_acc, color='steelblue', alpha=0.7)
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Per-Class Accuracy', fontweight='bold')
    ax4.set_xticks(range(n_classes))
    ax4.set_xticklabels([f'Class {i}' for i in range(n_classes)])
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')

    # 그림 5: 확률 분포
    ax5 = plt.subplot(2, 3, 5)
    for i in range(n_classes):
        mask = all_targets == i
        probs = all_probabilities[mask, i]  # Probability of correct class
        ax5.hist(probs, bins=30, alpha=0.5, label=f'Class {i}')

    ax5.set_xlabel('Predicted Probability (for true class)')
    ax5.set_ylabel('Count')
    ax5.set_title('Confidence Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 그림 6: 모델 웨이트 시각화
    ax6 = plt.subplot(2, 3, 6)
    weights = model.linear.weight.data.cpu().numpy()  # Shape: (n_classes, input_dim)
    # imshow에는 center 인자가 없다(그것은 seaborn.heatmap의 것이다).
    # 발산형 색지도를 0에 맞추려면 범위를 0 둘레로 대칭이 되게 준다.
    vmax = float(abs(weights).max())
    im = ax6.imshow(weights, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax6.set_xlabel('Feature Index')
    ax6.set_ylabel('Class')
    ax6.set_title('Model Weights', fontweight='bold')
    ax6.set_yticks(range(n_classes))
    ax6.set_yticklabels([f'Class {i}' for i in range(n_classes)])
    plt.colorbar(im, ax=ax6)

    plt.tight_layout()
    plt.show()

    print("Visualizations created!")

    # =============================================================================
    # 핵심 요점
    # =============================================================================

    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
    1. 여러 클래스와 둘 분류 견주기
       둘 분류: 출력 1개 → 시그모이드 → BCE
       여러 클래스: 출력 K개 → 소프트맥스 → 교차 엔트로피

    2. 종요로운 다름
       ✓ 출력 레이어: Linear(input_dim, num_classes)
       ✓ 로짓을 돌려준다(확률이 아니다)
       ✓ CrossEntropyLoss을 쓴다
       ✓ 타깃은 클래스 번호다(Long)
       ✓ 예측: 로짓의 argmax

    3. CROSSENTROPYLOSS
       ✓ 소프트맥스 + 로그 + NLL을 아우른다
       ✓ 수치가 더 든든하다
       ✓ 입력으로 로짓을 바란다
       ✓ 타깃으로 클래스 번호를 바란다

    4. EVALUATION
       ✓ 클래스마다의 정확도
       ✓ 혼동 행렬
       ✓ 클래스마다의 정밀도/재현율
       ✓ 클래스마다의 F1 점수

    5. 언제 쓸까
       ✓ 클래스가 둘을 넘을 때
       ✓ 클래스가 서로 겹치지 않을 때
       ✓ 레이블가 하나뿐인 분류
    """)

    print("\n" + "="*80)
    print("EXERCISES")
    print("="*80)
    print("""
    1. 쉬움: 클래스 수를 바꾸어 보아라(5, 10)

    2. 보통: 위 k개 정확도를 짜라.
       - 참 클래스가 위 2개나 위 3개 예측에 드는지 살핀다

    3. 보통: 바이어스를 다루도록 클래스 웨이트를 더하여라.
       - CrossEntropyLoss의 class_weight 파라메터를 쓴다

    4. 어려움: 하나 대 나머지 길을 짜라.
       - 둘 분류기 K개를 익힌다
       - 곧바로 하는 여러 클래스 분류과 견준다

    5. 어려움: 레이블 스무딩을 더하여라.
       - 딱딱한 0/1 대신 부드러운 타깃
       - 일반화이 나아진다
    """)

    print("\n" + "="*80)
    print("NEXT: 03_regularization.py - Preventing overfitting")
    print("="*80)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (170줄)"

    ```
    ================================================================================
    MULTI-CLASS CLASSIFICATION
    ================================================================================

    ================================================================================
    PART 1: UNDERSTANDING MULTI-CLASS CLASSIFICATION
    ================================================================================

    둘 분류:
      2 classes: 0 or 1
      출력: 확률 하나
      살림: 시그모이드
      로스: 둘 교차 엔트로피(BCE)

    여러 클래스 분류:
      K classes: 0, 1, 2, ..., K-1
      출력: 확률 K개(합이 1이다)
      살림: 소프트맥스
      로스: 교차 엔트로피

    소프트맥스 함수:
      클래스가 K개일 때 소프트맥스는 로짓을 확률로 바꾼다.
      
      P(class=k) = exp(logit_k) / sum(exp(logit_j) for all j)
      
      Properties:
      ✓ 확률이 모두 양수다
      ✓ 확률의 합이 1이다
      ✓ Differentiable


    ================================================================================
    PART 2: PREPARING MULTI-CLASS DATA
    ================================================================================
    Dataset: 1500 samples, 3 classes
    Class distribution:
      Class 0:  493 samples (32.9%)
      Class 1:  505 samples (33.7%)
      Class 2:  502 samples (33.5%)

    Data shapes:
      X_train: torch.Size([1200, 20])
      y_train: torch.Size([1200]) (class indices)
      X_test: torch.Size([300, 20])
      y_test: torch.Size([300])

    ================================================================================
    PART 3: MULTI-CLASS LOGISTIC REGRESSION MODEL
    ================================================================================
    Model created:
      Input dimension: 20
      Number of classes: 3
      Parameters: 63

    Model architecture:
    MultiClassLogisticRegression(
      (linear): Linear(in_features=20, out_features=3, bias=True)
    )

    ================================================================================
    PART 4: TRAINING MULTI-CLASS MODEL
    ================================================================================

    CrossEntropyLoss:
      - LogSoftmax과 NLLLoss을 아우른다
      - Expects:
        * 예측: (batch_size, num_classes) - 날것 로짓
        * 타깃: (batch_size,) - 클래스 번호(Long 텐서)
      - 소프트맥스를 절로 건다
      - 직접 하는 소프트맥스 + 로그보다 수치가 든든하다


    Training for 100 epochs...
    ------------------------------------------------------------
    Epoch [ 10/100] Train Loss: 0.2763 Acc: 0.9217 | Test Loss: 0.2468 Acc: 0.9200
    Epoch [ 20/100] Train Loss: 0.2663 Acc: 0.9208 | Test Loss: 0.2397 Acc: 0.9233
    Epoch [ 30/100] Train Loss: 0.2650 Acc: 0.9217 | Test Loss: 0.2404 Acc: 0.9233
    Epoch [ 40/100] Train Loss: 0.2646 Acc: 0.9250 | Test Loss: 0.2393 Acc: 0.9233
    Epoch [ 50/100] Train Loss: 0.2646 Acc: 0.9200 | Test Loss: 0.2381 Acc: 0.9233
    Epoch [ 60/100] Train Loss: 0.2642 Acc: 0.9250 | Test Loss: 0.2384 Acc: 0.9233
    Epoch [ 70/100] Train Loss: 0.2650 Acc: 0.9250 | Test Loss: 0.2400 Acc: 0.9233
    Epoch [ 80/100] Train Loss: 0.2651 Acc: 0.9225 | Test Loss: 0.2398 Acc: 0.9233
    Epoch [ 90/100] Train Loss: 0.2651 Acc: 0.9258 | Test Loss: 0.2369 Acc: 0.9233
    Epoch [100/100] Train Loss: 0.2662 Acc: 0.9233 | Test Loss: 0.2371 Acc: 0.9233

    ✓ Training completed!

    ================================================================================
    PART 5: COMPREHENSIVE EVALUATION
    ================================================================================

    Classification Report:
                  precision    recall  f1-score   support

         Class 0       0.92      0.89      0.90        99
         Class 1       0.97      0.96      0.97       101
         Class 2       0.88      0.92      0.90       100

        accuracy                           0.92       300
       macro avg       0.92      0.92      0.92       300
    weighted avg       0.92      0.92      0.92       300


    Confusion Matrix:
    [[88  0 11]
     [ 3 97  1]
     [ 5  3 92]]

    ================================================================================
    PART 6: VISUALIZATIONS
    ================================================================================
    Visualizations created!

    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. 여러 클래스와 둘 분류 견주기
       둘 분류: 출력 1개 → 시그모이드 → BCE
       여러 클래스: 출력 K개 → 소프트맥스 → 교차 엔트로피

    2. 종요로운 다름
       ✓ 출력 레이어: Linear(input_dim, num_classes)
       ✓ 로짓을 돌려준다(확률이 아니다)
       ✓ CrossEntropyLoss을 쓴다
       ✓ 타깃은 클래스 번호다(Long)
       ✓ 예측: 로짓의 argmax

    3. CROSSENTROPYLOSS
       ✓ 소프트맥스 + 로그 + NLL을 아우른다
       ✓ 수치가 더 든든하다
       ✓ 입력으로 로짓을 바란다
       ✓ 타깃으로 클래스 번호를 바란다

    4. EVALUATION
       ✓ 클래스마다의 정확도
       ✓ 혼동 행렬
       ✓ 클래스마다의 정밀도/재현율
       ✓ 클래스마다의 F1 점수

    5. 언제 쓸까
       ✓ 클래스가 둘을 넘을 때
       ✓ 클래스가 서로 겹치지 않을 때
       ✓ 레이블가 하나뿐인 분류


    ================================================================================
    EXERCISES
    ================================================================================

    1. 쉬움: 클래스 수를 바꾸어 보아라(5, 10)

    2. 보통: 위 k개 정확도를 짜라.
       - 참 클래스가 위 2개나 위 3개 예측에 드는지 살핀다

    3. 보통: 바이어스를 다루도록 클래스 웨이트를 더하여라.
       - CrossEntropyLoss의 class_weight 파라메터를 쓴다

    4. 어려움: 하나 대 나머지 길을 짜라.
       - 둘 분류기 K개를 익힌다
       - 곧바로 하는 여러 클래스 분류과 견준다

    5. 어려움: 레이블 스무딩을 더하여라.
       - 딱딱한 0/1 대신 부드러운 타깃
       - 일반화이 나아진다


    ================================================================================
    NEXT: 03_regularization.py - Preventing overfitting
    ================================================================================
    ```


## 2. 논의

### 모델 출력과 로스

`nn.Linear(input_dim, num_classes)`는 표본마다 클래스 수만큼 로짓을 만든다. `CrossEntropyLoss`에 원핫이 아닌 정수 클래스 인덱스를 줄 때 정답 모양은 $(B,)$, 자료형은 `long`이다. 학습 경로에 Softmax를 추가할 필요가 없으며 확률을 볼 때만 `predict_proba()`를 사용한다.

### 클래스별 평가

최대 로짓의 인덱스가 예측 클래스이며 정확도는 표본별 정답 일치를 집계한다. 혼동 행렬과 클래스별 정밀도·재현율을 보면 전체 정확도가 가리는 오차를 찾을 수 있다. 예측 확률의 순위가 같아도 교차 엔트로피와 보정 상태는 다를 수 있다.

### 선형 분류기의 범위

단일 선형 계층은 피처 공간에서 선형인 클래스 간 경계를 만든다. 비선형 은닉 레이어를 추가하면 표현력이 달라지며 원래의 소프트맥스 회귀와 다른 모델이 된다. 같은 클래스 순서와 전처리를 유지해 지표를 비교한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 $d$개, 클래스 $K$개의 단일 선형 분류기에는 파라미터가 몇 개 있는가?

</div>

??? success "풀이"
    웨이트가 $(K,d)$로 $Kd$개, 바이어스가 $(K,)$로 $K$개이므로 총 $K(d+1)$개다. Softmax는 별도 학습 파라미터가 없으며 학습 로스는 로짓을 직접 받는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 옵티마이저와 학습 수렴을 비교하라. 두 로스 곡선을 같은 그래프에 그려라.

</div>

??? success "풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 경사와 경사 제곱의 이동 평균으로 성분별 업데이트를 조정한다. 어느 옵티마이저가 더 빨리 수렴하는지는 문제와 러닝레이트에 달려 있다. 같은 초기 웨이트와 데이터 순서, 평가 기준을 사용하고 각 러닝레이트를 조정해 비교한다. 여러 시드의 변동과 검증 성능도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
클래스 인덱스와 모델 출력 모양이 잘못되었을 때 무엇을 확인해야 하는가?

</div>

??? success "풀이"
    로짓은 $(B,K)$, 정답 인덱스는 $(B,)$이고 0부터 $K-1$ 범위의 정수여야 한다. 확률 목표 분포를 사용하는 다른 모드와 혼동하지 않는다. 학습 전에 Softmax를 추가하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 일반적인 계층 확장 코드가 이 페이지의 원래 클래스에 그대로 적용되는지 검토하라. 모델과 데이터셋의 역할, 생성자 인수, 입력·출력 모양과 비선형성에서 필요한 수정 사항을 설명하라.

</div>

??? success "풀이"
    아래 코드는 일반적인 ModuleList 구성 패턴의 일부이며 이 페이지 클래스의 완성된 대체 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    적용하려면 생성자와 `forward`를 함께 정의하고 마지막 출력 모양과 로스의 입력 조건을 유지해야 한다. 선형 계층만 여러 개 합성하면 여전히 하나의 선형 사상이므로 표현력을 늘리려면 적절한 비선형성이 필요하다. Dataset은 표본 접근 인터페이스이므로 계층을 등록할 대상이 아니며 별도 모델 클래스에 이 구성을 적용한다.

## 정리하며

모델 출력과 로스의 입력 조건을 일치시키고 데이터 분할, 전처리와 지표의 의미를 함께 확인한다. 기록된 실행 결과는 해당 설정의 예시이며 모든 환경의 수렴이나 일반화를 보장하지 않는다.
