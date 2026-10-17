# 정규화 기법 자세히 보기

MNIST를 학습·검증·테스트로 나누고, 기본 MLP와 드롭아웃·웨이트 감쇠를 적용한 MLP를 비교한다. 검증 지표를 이용한 설정 선택과 마지막 테스트 평가를 구분한다.

## 1. 코드

??? note "코드 (460줄)"

    ```python
    """
    ==============================================================================
    튜토리얼 07: 정칙화 기법
    ==============================================================================
    난이도: ⭐⭐⭐ 중급~고급

    배울 내용:
    - 과적합과 과소적합
    - 드롭아웃 정칙화
    - L2 정칙화 (웨이트 감쇠)
    - 검증 집합 쓰기
    - 조기 종료

    선수 지식:
    - 튜토리얼 06 (MNIST 분류)

    핵심 개념:
    - 과적합 예방
    - nn.Dropout
    - 웨이트 감쇠
    - 모델 검증
    - 학습 과정 감시
    ==============================================================================
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, random_split
    import numpy as np

    torch.manual_seed(42)

    # ==============================================================================
    # 들어가며: 과적합 문제
    # ==============================================================================
    print("=" * 70)
    print("Understanding Overfitting and Regularization")
    print("=" * 70)
    print("""
    과적합이란 무엇인가?
      - 모델이 학습 데이터를 지나치게 잘 외운다
      - 학습 집합에서는 아주 잘한다
      - 처음 보는 데이터(시험 집합)에서 성능이 나쁘다
      - 일반화하지 않고 암기한다

    과적합을 어떻게 막는가?
      1. 학습 데이터를 더 모은다
      2. 더 단순한 모델(파라메터가 적다)
      3. 정칙화 기법:
         - Dropout
         - L2 정칙화 (웨이트 감쇠)
         - 데이터 어그멘테이션
      4. 조기 종료
    """)

    # ==============================================================================
    # 1단계: 검증 분할과 함께 데이터 준비
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Creating Train/Validation/Test Split")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # MNIST 불러오기
    transform = transforms.Compose([transforms.ToTensor()])

    train_val_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )

    # 학습 데이터를 학습용과 검증용으로 나눈다 (80/20)
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print("Dataset split:")
    print(f"  Training:   {len(train_dataset):,} samples (80%)")
    print(f"  Validation: {len(val_dataset):,} samples (20%)")
    print(f"  Test:       {len(test_dataset):,} samples")

    print("\nWhy use validation set?")
    print("  - Monitor overfitting during training")
    print("  - Tune hyperparameters")
    print("  - Test set remains untouched until final evaluation")

    # 데이터 로더 만들기
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ==============================================================================
    # 2단계: 모델 정의 (정칙화 있는 것과 없는 것)
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Defining Models")
    print("=" * 70)

    class SimpleNet(nn.Module):
        """
        정칙화가 없는 간단한 뉴럴 넷트웍
        과적합하기 쉽다!
        """
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(28 * 28, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.network(x)

    class RegularizedNet(nn.Module):
        """
        정칙화 기법을 쓰는 뉴럴 넷트웍

        쓰인 정칙화 방법:
          1. 드롭아웃: 학습 중에 일부 뉴런을 무작위로 0으로 만든다
          2. 웨이트 감쇠: 최적화기를 통해 더한다 (L2 정칙화)
        """
        def __init__(self, dropout_rate=0.5):
            super(RegularizedNet, self).__init__()

            self.network = nn.Sequential(
                # 첫 번째 레이어
                nn.Linear(28 * 28, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),  # 활성화 뒤의 드롭아웃

                # 두 번째 레이어
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),

                # 출력 레이어 (여기에는 드롭아웃을 두지 않는다)
                nn.Linear(128, 10)
            )

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.network(x)

    print("Two models defined:")
    print("  1. SimpleNet: No regularization")
    print("  2. RegularizedNet: Dropout + Weight decay")
    print(f"\nDropout explanation:")
    print("  - Randomly sets neurons to 0 during training")
    print("  - Forces network to learn robust features")
    print("  - Prevents co-adaptation of neurons")
    print("  - Automatically disabled during eval mode")

    # ==============================================================================
    # 3단계: 학습 함수
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Defining Training and Evaluation Functions")
    print("=" * 70)

    def train_epoch(model, train_loader, criterion, optimizer, device):
        """한 에포크 동안 학습한다"""
        model.train()  # 드롭아웃 켜기
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # 여기서 세는 정확도는 드롭아웃이 켜진 채, 그것도 갱신 전
            # 웨이트로 계산한 outputs에서 나온 값이다. 학습이 도는 동안
            # 대강의 진행을 보기에는 쓸 만하지만 모델끼리 견주는 데에는
            # 쓸 수 없다. 드롭아웃을 쓰는 모델에서만 낮게 잡히기 때문이다.
            # 그래서 아래 4단계는 이 값을 버리고 evaluate()로 다시 잰다
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def evaluate(model, data_loader, criterion, device):
        """데이터셋에서 모델을 평가한다"""
        model.eval()  # 드롭아웃 끄기
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(data_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    print("Functions defined:")
    print("  - train_epoch(): Trains for one epoch")
    print("  - evaluate(): Evaluates model (no gradient computation)")

    # ==============================================================================
    # 4단계: 두 모델 모두 학습
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Training Both Models")
    print("=" * 70)

    n_epochs = 15
    learning_rate = 0.001

    # 모델 1: 정칙화 없음
    print("\n" + "-" * 70)
    print("Training Model 1: SimpleNet (No Regularization)")
    print("-" * 70)

    # 주의: 씨앗을 심지 않아 두 모델의 초기 웨이트가 다르다. 차이를
    # 정칙화 탓으로만 돌리려면 두 모델을 만들기 직전마다
    # torch.manual_seed(42)를 불러야 한다
    model1 = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
    # 모델 1에는 웨이트 감쇠를 쓰지 않는다

    history1 = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(n_epochs):
        train_loss, _ = train_epoch(model1, train_loader, criterion, optimizer1, device)
        # 학습 정확도를 eval 모드에서 다시 잰다. train_epoch가 돌려주는 값은
        # 드롭아웃이 켜진 채, 그것도 갱신 전 출력으로 계산한 것이라 정칙화를
        # 쓰는 모델에서만 낮게 잡힌다. 그대로 쓰면 아래 "학습-검증 차이"가
        # 모델 2에서만 작아 보여, 정칙화의 효과를 실제보다 부풀린다
        _, train_acc = evaluate(model1, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model1, val_loader, criterion, device)

        history1['train_loss'].append(train_loss)
        history1['train_acc'].append(train_acc)
        history1['val_loss'].append(val_loss)
        history1['val_acc'].append(val_acc)

        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # 모델 2: 정칙화 있음
    print("\n" + "-" * 70)
    print("Training Model 2: RegularizedNet (With Dropout + Weight Decay)")
    print("-" * 70)

    # 기본값 0.5가 아니라 0.3을 넘긴다. 폭이 256과 128인 레이어라
    # 절반을 끄면 남는 뉴런이 적어 학습이 더디기 때문이다
    model2 = RegularizedNet(dropout_rate=0.3).to(device)
    # 웨이트 감쇠 추가 (L2 정칙화).
    # 모델 1에는 이 인자가 없으므로, 위와 달리 여기 기준선은 정말로
    # 정칙화가 없는 모델이다.
    # 다만 한 번에 두 가지(드롭아웃과 웨이트 감쇠)를 바꾸었으므로,
    # 아래 결과에서 어느 쪽이 일했는지는 갈라낼 수 없다.
    # 그것까지 보려면 셋째 모델로 하나씩만 켜서 견주어야 한다
    optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=1e-4)

    print(f"Regularization parameters:")
    print(f"  - Dropout rate: 0.3 (30% of neurons dropped)")
    print(f"  - Weight decay: 1e-4 (L2 penalty)\n")

    history2 = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(n_epochs):
        train_loss, _ = train_epoch(model2, train_loader, criterion, optimizer2, device)
        # 학습 정확도를 eval 모드에서 다시 잰다. train_epoch가 돌려주는 값은
        # 드롭아웃이 켜진 채, 그것도 갱신 전 출력으로 계산한 것이라 정칙화를
        # 쓰는 모델에서만 낮게 잡힌다. 그대로 쓰면 아래 "학습-검증 차이"가
        # 모델 2에서만 작아 보여, 정칙화의 효과를 실제보다 부풀린다
        _, train_acc = evaluate(model2, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model2, val_loader, criterion, device)

        history2['train_loss'].append(train_loss)
        history2['train_acc'].append(train_acc)
        history2['val_loss'].append(val_loss)
        history2['val_acc'].append(val_acc)

        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # ==============================================================================
    # 5단계: 결과 비교
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Comparing Models on Test Set")
    print("=" * 70)

    # 두 모델을 시험 집합에서 평가
    test_loss1, test_acc1 = evaluate(model1, test_loader, criterion, device)
    test_loss2, test_acc2 = evaluate(model2, test_loader, criterion, device)

    print("\nFinal Test Results:")
    print("-" * 70)
    print(f"Model 1 (No Regularization):")
    print(f"  Test Accuracy: {test_acc1:.2f}%")
    print(f"  Test Loss: {test_loss1:.4f}")
    print(f"\nModel 2 (With Regularization):")
    print(f"  Test Accuracy: {test_acc2:.2f}%")
    print(f"  Test Loss: {test_loss2:.4f}")

    # 과적합 지표 계산 (학습-검증 차이)
    # 과적합을 "학습 정확도 빼기 검증 정확도"로 잰다. 학습 데이터에서만
    # 잘하고 처음 보는 데이터에서 못하는 정도를 뜻하므로 타당한 잣대다.
    # 두 모델의 학습 정확도를 모두 eval() 상태에서 다시 재었으므로, 이
    # 차이는 정칙화의 효과만 담는다. 드롭아웃이 켜진 채로 잰 값을 그대로
    # 썼다면 model2의 차이가 실제보다 작게 나와 정칙화를 과대평가하게 된다
    overfit1 = history1['train_acc'][-1] - history1['val_acc'][-1]
    overfit2 = history2['train_acc'][-1] - history2['val_acc'][-1]
    print(f"\nOverfitting Analysis (Train-Val Accuracy Gap):")
    print(f"  Model 1: {overfit1:.2f}% gap")
    print(f"  Model 2: {overfit2:.2f}% gap")
    print(f"  {'Model 2 has less overfitting! ✓' if overfit2 < overfit1 else 'Unexpected result'}")

    # ==============================================================================
    # 6단계: 시각화
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Visualizing Training Progress")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, n_epochs + 1)

    # 모델 1: 로스
    axes[0, 0].plot(epochs, history1['train_loss'], 'b-o', label='Train Loss', alpha=0.7)
    axes[0, 0].plot(epochs, history1['val_loss'], 'r-s', label='Val Loss', alpha=0.7)
    axes[0, 0].set_title('Model 1 (No Regularization): Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 모델 1: 정확도
    axes[0, 1].plot(epochs, history1['train_acc'], 'b-o', label='Train Acc', alpha=0.7)
    axes[0, 1].plot(epochs, history1['val_acc'], 'r-s', label='Val Acc', alpha=0.7)
    axes[0, 1].set_title('Model 1 (No Regularization): Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 모델 2: 로스
    axes[1, 0].plot(epochs, history2['train_loss'], 'b-o', label='Train Loss', alpha=0.7)
    axes[1, 0].plot(epochs, history2['val_loss'], 'r-s', label='Val Loss', alpha=0.7)
    axes[1, 0].set_title('Model 2 (With Regularization): Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 모델 2: 정확도
    axes[1, 1].plot(epochs, history2['train_acc'], 'b-o', label='Train Acc', alpha=0.7)
    axes[1, 1].plot(epochs, history2['val_acc'], 'r-s', label='Val Acc', alpha=0.7)
    axes[1, 1].set_title('Model 2 (With Regularization): Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('07_regularization_comparison.png', dpi=100)
    print("Comparison saved as '07_regularization_comparison.png'")

    # ==============================================================================
    # 핵심 정리:
    # ==============================================================================
    print("\n" + "=" * 70)
    print("핵심 정리")
    print("=" * 70)
    print("""
    1. 과적합의 징후:
       - 학습 정확도와 검증 정확도의 간격이 크다
       - 학습 정확도는 계속 오르는데 검증은 정체된다
       - 처음 보는 데이터에서 성능이 나쁘다

    2. Dropout (nn.Dropout):
       - 학습 중 뉴런을 무작위로 0으로 만든다
       - dropout_rate: 떨어뜨릴 확률(보통 0.2~0.5)
       - model.eval()을 부르면 자동으로 꺼진다
       - 뉴런들이 함께 적응하는 것을 막는다

    3. 웨이트 감쇠(L2 정칙화):
       - 큰 웨이트에 벌점을 더한다: 로스 = 데이터 로스 + λ * Σ(웨이트²)
       - 최적화기에서 설정한다: weight_decay=1e-4
       - 웨이트를 작게 유지하도록 북돋운다
       - 두루 미침이 나아진다

    4. 검증 집합:
       - 과적합을 살피는 데 꼭 필요하다
       - 초파라메터 조율에 쓴다
       - 그래디언트 갱신에는 쓰지 않는다
       - 시험 집합과 따로 둔다

    5. 그 밖의 정칙화 기법(여기서는 다루지 않는다):
       - 데이터 어그멘테이션
       - 배치 정규화
       - 조기 종료
       - L1 regularization

    6. 모범 사례:
       - 늘 검증 집합을 쓰라
       - 학습과 검증의 간격을 살피라
       - 가벼운 정칙화로 시작하여 필요하면 늘리라
       - 드롭아웃은 ReLU 앞이 아니라 뒤에 두라

    다음 단계:
    - 튜토리얼 08: 배치 정규화
    - 튜토리얼 09: 러닝레이트 스케줄링
    - 튜토리얼 10: 고급 구조
    """)

    print("\nTraining completed successfully! ✓")
    # ==============================================================================


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (151줄)"

    ```
    ======================================================================
    Understanding Overfitting and Regularization
    ======================================================================

    과적합이란 무엇인가?
      - 모델이 학습 데이터를 지나치게 잘 외운다
      - 학습 집합에서는 아주 잘한다
      - 처음 보는 데이터(시험 집합)에서 성능이 나쁘다
      - 일반화하지 않고 암기한다

    과적합을 어떻게 막는가?
      1. 학습 데이터를 더 모은다
      2. 더 단순한 모델(파라메터가 적다)
      3. 정칙화 기법:
         - Dropout
         - L2 정칙화 (웨이트 감쇠)
         - 데이터 어그멘테이션
      4. 조기 종료


    ======================================================================
    STEP 1: Creating Train/Validation/Test Split
    ======================================================================
    Using device: cpu

    Dataset split:
      Training:   48,000 samples (80%)
      Validation: 12,000 samples (20%)
      Test:       10,000 samples

    Why use validation set?
      - Monitor overfitting during training
      - Tune hyperparameters
      - Test set remains untouched until final evaluation

    ======================================================================
    STEP 2: Defining Models
    ======================================================================
    Two models defined:
      1. SimpleNet: No regularization
      2. RegularizedNet: Dropout + Weight decay

    Dropout explanation:
      - Randomly sets neurons to 0 during training
      - Forces network to learn robust features
      - Prevents co-adaptation of neurons
      - Automatically disabled during eval mode

    ======================================================================
    STEP 3: Defining Training and Evaluation Functions
    ======================================================================
    Functions defined:
      - train_epoch(): Trains for one epoch
      - evaluate(): Evaluates model (no gradient computation)

    ======================================================================
    STEP 4: Training Both Models
    ======================================================================

    ----------------------------------------------------------------------
    Training Model 1: SimpleNet (No Regularization)
    ----------------------------------------------------------------------
    Epoch  3: Train Loss: 0.0785, Train Acc: 98.52% | Val Loss: 0.0915, Val Acc: 97.15%
    Epoch  6: Train Loss: 0.0328, Train Acc: 99.34% | Val Loss: 0.0823, Val Acc: 97.47%
    Epoch  9: Train Loss: 0.0163, Train Acc: 99.37% | Val Loss: 0.0977, Val Acc: 97.39%
    Epoch 12: Train Loss: 0.0136, Train Acc: 99.54% | Val Loss: 0.1183, Val Acc: 97.33%
    Epoch 15: Train Loss: 0.0101, Train Acc: 99.72% | Val Loss: 0.1134, Val Acc: 97.65%

    ----------------------------------------------------------------------
    Training Model 2: RegularizedNet (With Dropout + Weight Decay)
    ----------------------------------------------------------------------
    Regularization parameters:
      - Dropout rate: 0.3 (30% of neurons dropped)
      - Weight decay: 1e-4 (L2 penalty)

    Epoch  3: Train Loss: 0.1290, Train Acc: 97.77% | Val Loss: 0.1137, Val Acc: 96.43%
    Epoch  6: Train Loss: 0.0845, Train Acc: 98.89% | Val Loss: 0.0812, Val Acc: 97.58%
    Epoch  9: Train Loss: 0.0709, Train Acc: 99.19% | Val Loss: 0.0719, Val Acc: 97.88%
    Epoch 12: Train Loss: 0.0599, Train Acc: 99.33% | Val Loss: 0.0812, Val Acc: 97.59%
    Epoch 15: Train Loss: 0.0526, Train Acc: 99.42% | Val Loss: 0.0763, Val Acc: 97.83%

    ======================================================================
    STEP 5: Comparing Models on Test Set
    ======================================================================

    Final Test Results:
    ----------------------------------------------------------------------
    Model 1 (No Regularization):
      Test Accuracy: 97.93%
      Test Loss: 0.1022

    Model 2 (With Regularization):
      Test Accuracy: 98.20%
      Test Loss: 0.0685

    Overfitting Analysis (Train-Val Accuracy Gap):
      Model 1: 2.07% gap
      Model 2: 1.59% gap
      Model 2 has less overfitting! ✓

    ======================================================================
    STEP 6: Visualizing Training Progress
    ======================================================================
    Comparison saved as '07_regularization_comparison.png'

    ======================================================================
    핵심 정리
    ======================================================================

    1. 과적합의 징후:
       - 학습 정확도와 검증 정확도의 간격이 크다
       - 학습 정확도는 계속 오르는데 검증은 정체된다
       - 처음 보는 데이터에서 성능이 나쁘다

    2. Dropout (nn.Dropout):
       - 학습 중 뉴런을 무작위로 0으로 만든다
       - dropout_rate: 떨어뜨릴 확률(보통 0.2~0.5)
       - model.eval()을 부르면 자동으로 꺼진다
       - 뉴런들이 함께 적응하는 것을 막는다

    3. 웨이트 감쇠(L2 정칙화):
       - 큰 웨이트에 벌점을 더한다: 로스 = 데이터 로스 + λ * Σ(웨이트²)
       - 최적화기에서 설정한다: weight_decay=1e-4
       - 웨이트를 작게 유지하도록 북돋운다
       - 두루 미침이 나아진다

    4. 검증 집합:
       - 과적합을 살피는 데 꼭 필요하다
       - 초파라메터 조율에 쓴다
       - 그래디언트 갱신에는 쓰지 않는다
       - 시험 집합과 따로 둔다

    5. 그 밖의 정칙화 기법(여기서는 다루지 않는다):
       - 데이터 어그멘테이션
       - 배치 정규화
       - 조기 종료
       - L1 regularization

    6. 모범 사례:
       - 늘 검증 집합을 쓰라
       - 학습과 검증의 간격을 살피라
       - 가벼운 정칙화로 시작하여 필요하면 늘리라
       - 드롭아웃은 ReLU 앞이 아니라 뒤에 두라

    다음 단계:
    - 튜토리얼 08: 배치 정규화
    - 튜토리얼 09: 러닝레이트 스케줄링
    - 튜토리얼 10: 고급 구조


    Training completed successfully! ✓
    ```


## 2. 논의

`SimpleNet`과 `RegularizedNet`은 같은 선형 레이어를 사용하며, 후자는 은닉 레이어에 드롭아웃을 추가한다. 옵티마이저의 웨이트 감쇠도 함께 달라지므로 한 번의 비교로 두 기법의 개별 효과를 분리할 수 없다.

기록된 로스는 배치 평균 로스를 배치 수로 다시 나눈 값이다. 마지막 배치가 작으면 표본별 전체 평균과 약간 다르다. 정확한 전체 평균에는 배치 크기를 반영해야 한다.

학습 정확도는 드롭아웃이 켜진 모드에서, 검증 정확도는 꺼진 모드에서 계산된다. 두 값의 격차만으로 과적합 정도의 순위를 확정하지 말고, 같은 평가 모드의 성능과 여러 초기값의 변동도 살펴본다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
SimpleNet과 RegularizedNet의 선형 레이어 파라메터 수가 같은 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"
    둘 다 784→256→128→10이며 235146개이다. 드롭아웃은 파라메터를 추가하지 않고, 웨이트 감쇠는 목적이나 갱신을 바꾸므로 파라메터 수를 바꾸지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
드롭아웃과 웨이트 감쇠의 개별 효과를 비교하는 실험을 설계하라.

</div>

??? success "연습문제 2 풀이"
    기본, 드롭아웃만, 감쇠만, 둘 다의 네 조건을 같은 데이터 분할과 학습 예산에서 비교한다. 강도는 검증셋에서 선택하고 최종 테스트는 선택 후에 수행한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
학습·검증 정확도 격차가 더 작은 모델이 반드시 더 좋은가?

</div>

??? success "연습문제 3 풀이"
    과소적합 모델도 작은 격차를 보일 수 있다. 드롭아웃의 모드 차이와 데이터 난이도도 영향을 준다. 검증 성능 자체와 동일 평가 조건의 학습 성능을 함께 보아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

정규화 기법의 효과는 같은 평가 조건에서 판단한다. 이 예제처럼 여러 기법을 동시에 바꾸면 전체 조합의 결과를 얻으며, 각 기법의 효과를 알려면 조건을 나눠 비교해야 한다.
