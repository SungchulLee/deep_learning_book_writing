# 배치 정규화 자세히 보기

학습·검증·테스트 분할을 유지하며 배치 정규화 유무를 비교한다. 학습 가능한 아핀 계수와 이동 통계량의 역할, 학습·평가 모드 차이를 살펴본다.

## 1. 코드

??? note "코드 (476줄)"

    ```python
    """
    ==============================================================================
    튜토리얼 08: 배치 정규화
    ==============================================================================
    난이도: ⭐⭐⭐ 고급

    배울 내용:
    - 내부 공변량 변화 문제
    - 배치 정규화 레이어
    - 배치 정규화가 학습을 개선하는 방식
    - 배치 정규화를 언제 어디에 쓰는지

    선수 지식:
    - 튜토리얼 07 (정칙화 기법)

    핵심 개념:
    - nn.BatchNorm1d
    - 학습의 안정성
    - 더 빠른 수렴
    - 배치 정규화와 레이어 정규화
    ==============================================================================
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, random_split
    import time

    torch.manual_seed(42)

    # ==============================================================================
    # 들어가며: 내부 공변량 변화 문제
    # ==============================================================================
    print("=" * 70)
    print("Understanding Batch Normalization")
    print("=" * 70)
    print("""
    배치 정규화는 어떤 문제를 푸는가?

    내부 공변량 이동:
      - 학습이 진행되면 레이어 입력의 분포가 바뀐다
      - 레이어마다 이 변화에 적응해야 한다
      - 학습이 느려진다
      - 웨이트 초기화에 뉴럴 써야 한다
      - 러닝레이트에 민감하다

    배치 정규화라는 해법:
      - 미니배치마다 레이어 입력을 정규화한다
      - 안쪽 함께 바뀜의 옮겨감을 줄인다
      - 러닝 빠르기를 더 크게 쓸 수 있다
      - 초기화에 대한 민감도를 낮춘다
      - 정칙화 노릇을 한다(드롭아웃처럼)

    식(배치 안의 피처마다):
      1. μ = mean(batch)
      2. σ² = variance(batch)
      3. x̂ = (x - μ) / √(σ² + ε)     # 정규화
      4. y = γ * x̂ + β                # 배율 조정과 이동

    여기서 γ(감마)와 β(베타)는 학습되는 파라메터이다!
    """)

    # ==============================================================================
    # 1단계: 데이터 불러오기
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Loading MNIST Data")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    transform = transforms.Compose([transforms.ToTensor()])

    train_val_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )

    # 80/20 학습/검증 분할
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = 128  # 배치 정규화 통계를 안정시키려고 배치 크기를 키운다
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset loaded (batch size: {batch_size})")

    # ==============================================================================
    # 2단계: 모델 정의
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Defining Models (With and Without BatchNorm)")
    print("=" * 70)

    class NetWithoutBN(nn.Module):
        """
        배치 정규화가 없는 깊은 뉴럴 넷트웍
        특히 더 깊은 구조에서는 학습이 어려울 수 있다
        """
        def __init__(self):
            super(NetWithoutBN, self).__init__()

            self.network = nn.Sequential(
                nn.Linear(28*28, 256),
                nn.ReLU(),

                nn.Linear(256, 256),
                nn.ReLU(),

                nn.Linear(256, 128),
                nn.ReLU(),

                nn.Linear(128, 10)
            )

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.network(x)

    class NetWithBN(nn.Module):
        """
        배치 정규화가 있는 깊은 뉴럴 넷트웍

        배치 정규화의 위치:
        - 보통 선형 레이어 뒤, 활성화 앞
        - 활성화 뒤에 두어도 된다는 연구도 있다
        - 여기서는 표준을 따른다: 선형 -> 배치 정규화 -> 활성화
        """
        def __init__(self):
            super(NetWithBN, self).__init__()

            self.network = nn.Sequential(
                # 1레이어
                nn.Linear(28*28, 256),
                # 선형 레이어 뒤, ReLU 앞의 배치 정규화.
                # 이 자리에 두면 ReLU에 들어가는 값이 0 언저리에 모여
                # 절반쯤 살고 절반쯤 죽으므로 신호가 한쪽으로 쏠리지 않는다.
                # 바로 앞 Linear의 바이어스는 사실상 쓸모가 없다. 여기서 평균을
                # 빼 버리기 때문이며, 그래서 실무에서는 bias=False를 주기도 한다
                nn.BatchNorm1d(256),
                nn.ReLU(),

                # 2레이어
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                # 3레이어
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),

                # 출력 레이어 (여기에는 배치 정규화를 두지 않는다)
                nn.Linear(128, 10)
            )

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.network(x)

    print("Two models defined:")
    print("  1. NetWithoutBN: Standard network")
    print("  2. NetWithBN: Network with BatchNorm layers")
    print("\nBatchNorm1d:")
    print("  - For fully connected layers (1D data)")
    print("  - BatchNorm2d exists for CNNs (2D data)")
    print("  - BatchNorm3d for 3D data (videos, medical images)")

    # ==============================================================================
    # 3단계: 배치 정규화의 거동 이해하기
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Understanding BatchNorm Behavior")
    print("=" * 70)

    # 시연을 위해 간단한 배치 정규화 레이어 만들기.
    # 갓 만든 레이어라 아래 출력이 gamma=[1,1,1], beta=[0,0,0],
    # running_mean=[0,0,0], running_var=[1,1,1]로 찍힌다. 곧 학습을
    # 한 번도 하지 않은 배치 정규화는 eval() 상태에서 입력을 거의
    # 그대로 통과시킨다. (x - 0) / sqrt(1 + eps)이기 때문이다.
    # 이동 통계는 학습을 거치며 배치마다 조금씩 갱신되어 쌓이는 값이다
    demo_bn = nn.BatchNorm1d(3)
    print("\nBatchNorm1d layer parameters:")
    print(f"  gamma (weight): {demo_bn.weight.data}")
    print(f"  beta (bias):    {demo_bn.bias.data}")
    print(f"  running_mean:   {demo_bn.running_mean}")
    print(f"  running_var:    {demo_bn.running_var}")

    print("\nKey points:")
    print("  - gamma and beta are learnable (updated by optimizer)")
    print("  - running_mean and running_var are NOT learnable")
    print("  - During training: uses batch statistics")
    print("  - During evaluation: uses running statistics")
    print("  - This is why model.train() and model.eval() matter!")

    # ==============================================================================
    # 4단계: 학습 함수
    # ==============================================================================

    def train_and_evaluate(model, train_loader, val_loader, optimizer, 
                           criterion, n_epochs, device):
        """모델을 학습시키고 지표를 기록한다"""
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'epoch_time': []
        }

        for epoch in range(n_epochs):
            # 주의: GPU에서는 이 시간이 부정확하다. CUDA 연산이 비동기라
            # 파이썬이 먼저 앞서 나가므로, 제대로 재려면 앞뒤로
            # torch.cuda.synchronize()를 불러야 한다.
            # 배치 정규화는 레이어마다 계산이 더해지므로 에포크 시간이 늘지만,
            # 대신 수렴이 빨라 전체 학습 시간은 줄어드는 것이 보통이다
            start_time = time.time()

            # 학습
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # 검증.
            # 배치 정규화가 있는 모델에서 eval()은 선택이 아니다. 이 줄이
            # 없으면 평가에서도 배치 통계를 쓰게 되어, 같은 표본이라도
            # 어떤 표본들과 한 배치에 묶였느냐에 따라 예측이 달라진다.
            # 배치 크기가 1이면 분산이 정의되지 않아 오류까지 난다
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # 지표를 계산한다
            epoch_time = time.time() - start_time
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total

            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            history['epoch_time'].append(epoch_time)

            if (epoch + 1) % 3 == 0:
                print(f"Epoch {epoch+1:2d}/{n_epochs}: "
                      f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                      f"Time: {epoch_time:.2f}s")

        return history

    # ==============================================================================
    # 5단계: 두 모델 모두 학습
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Training Both Models")
    print("=" * 70)

    n_epochs = 15
    # 참고: 배치 정규화에서는 러닝레이트를 높게 잡는 편이 낫다!
    # 다만 아래에서 두 모델이 이 값을 함께 쓴다. 정규화의 강점 하나가
    # "더 큰 러닝레이트를 견딘다"는 것인데, 같은 값을 주면 그 강점은
    # 표에 드러나지 않는다. 정규화 없는 쪽에 0.01이 이미 큰 값이라
    # 학습이 불안정해 보인다면, 그것도 이 실험이 말해 주는 바의 하나다
    learning_rate = 0.01

    criterion = nn.CrossEntropyLoss()

    # 모델 1: 배치 정규화 없음
    print("\n" + "-" * 70)
    print("Training Model WITHOUT BatchNorm")
    print("-" * 70)

    model1 = NetWithoutBN().to(device)
    optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate, momentum=0.9)
    history1 = train_and_evaluate(model1, train_loader, val_loader, 
                                   optimizer1, criterion, n_epochs, device)

    # 모델 2: 배치 정규화 있음
    print("\n" + "-" * 70)
    print("Training Model WITH BatchNorm")
    print("-" * 70)

    model2 = NetWithBN().to(device)
    optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate, momentum=0.9)
    history2 = train_and_evaluate(model2, train_loader, val_loader, 
                                   optimizer2, criterion, n_epochs, device)

    # ==============================================================================
    # 6단계: 결과 비교
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Comparison")
    print("=" * 70)

    # 시험 집합 평가
    model1.eval()
    model2.eval()

    def test_accuracy(model, test_loader, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    test_acc1 = test_accuracy(model1, test_loader, device)
    test_acc2 = test_accuracy(model2, test_loader, device)

    print("\nFinal Results:")
    print("-" * 70)
    print(f"Without BatchNorm:")
    print(f"  Best Val Accuracy: {max(history1['val_acc']):.2f}%")
    print(f"  Test Accuracy: {test_acc1:.2f}%")
    print(f"  Avg Time/Epoch: {sum(history1['epoch_time'])/len(history1['epoch_time']):.2f}s")

    print(f"\nWith BatchNorm:")
    print(f"  Best Val Accuracy: {max(history2['val_acc']):.2f}%")
    print(f"  Test Accuracy: {test_acc2:.2f}%")
    print(f"  Avg Time/Epoch: {sum(history2['epoch_time'])/len(history2['epoch_time']):.2f}s")

    improvement = test_acc2 - test_acc1
    print(f"\nImprovement with BatchNorm: {improvement:+.2f}% {'✓' if improvement > 0 else ''}")

    # ==============================================================================
    # 7단계: 시각화
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Visualizing Training Progress")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, n_epochs + 1)

    # 학습 로스
    axes[0, 0].plot(epochs, history1['train_loss'], 'b-', label='Without BN', linewidth=2)
    axes[0, 0].plot(epochs, history2['train_loss'], 'r-', label='With BN', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 검증 로스
    axes[0, 1].plot(epochs, history1['val_loss'], 'b-', label='Without BN', linewidth=2)
    axes[0, 1].plot(epochs, history2['val_loss'], 'r-', label='With BN', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 학습 정확도
    axes[1, 0].plot(epochs, history1['train_acc'], 'b-', label='Without BN', linewidth=2)
    axes[1, 0].plot(epochs, history2['train_acc'], 'r-', label='With BN', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Training Accuracy Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 검증 정확도
    axes[1, 1].plot(epochs, history1['val_acc'], 'b-', label='Without BN', linewidth=2)
    axes[1, 1].plot(epochs, history2['val_acc'], 'r-', label='With BN', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Validation Accuracy Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('08_batchnorm_comparison.png', dpi=100)
    print("Comparison saved as '08_batchnorm_comparison.png'")

    # ==============================================================================
    # 핵심 정리:
    # ==============================================================================
    print("\n" + "=" * 70)
    print("핵심 정리")
    print("=" * 70)
    print("""
    1. 배치 정규화가 하는 일:
       - 미니배치별로 레이어 입력을 정규화한다
       - 최적의 크기(γ)와 이동(β)을 학습한다
       - 추론을 위해 이동 통계를 유지한다

    2. Benefits:
       ✓ 더 빨리 수렴한다(더 큰 러닝레이트를 쓸 수 있다)
       ✓ 초기화에 대한 민감도를 낮춘다
       ✓ 정칙화 노릇을 한다(일반화가 조금 나아진다)
       ✓ 학습이 더 안정적이다(특히 깊은 뉴럴 넷트웍에서)

    3. 배치 정규화를 어디에 둘까:
       - 보통: 선형 -> 배치 정규화 -> 활성화
       - 선형 레이어 뒤, 활성화 앞
       - 출력 레이어에는 필요 없다
       - 완전 연결 레이어에는 BatchNorm1d를 쓴다
       - 컨볼루션 레이어에는 BatchNorm2d를 쓴다

    4. 학습과 평가:
       - 학습: 배치 통계를 쓴다(현재 배치의 평균과 분산)
       - 평가: 이동 통계를 쓴다(학습 중에 쌓은 값)
       - model.train()과 model.eval()을 늘 알맞게 부르라!

    5. 배치 정규화의 변형:
       - BatchNorm1d: 완전 연결 레이어용
       - BatchNorm2d: 2차원 컨볼루션 레이어용
       - BatchNorm3d: 3차원 컨볼루션 레이어용
       - LayerNorm: 대안(배치가 아니라 피처 방향으로 정규화한다)
       - GroupNorm: 배치 크기가 작아도 잘 작동한다

    6. 배치 정규화를 쓰지 말아야 할 때:
       - 배치 크기가 아주 작을 때(통계를 믿기 어렵다)
       - 순환 뉴럴 넷트웍(대신 LayerNorm을 쓴다)
       - 배치 사이의 독립성이 중요할 때

    7. 흔한 함정:
       ✗ 추론할 때 model.eval() 부르기를 잊는다
       ✗ 배치 크기를 1로 쓴다(통계가 뜻이 없다)
       ✗ 학습 모드와 평가 모드를 헷갈리기

    다음 단계:
    - 튜토리얼 09: 러닝레이트 스케줄링
    - 튜토리얼 10: 고급 구조
    - 튜토리얼 11: CIFAR-10 도전
    """)

    print("\nTraining completed successfully! ✓")
    # ==============================================================================


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (161줄)"

    ```
    ======================================================================
    Understanding Batch Normalization
    ======================================================================

    배치 정규화는 어떤 문제를 푸는가?

    내부 공변량 이동:
      - 학습이 진행되면 레이어 입력의 분포가 바뀐다
      - 레이어마다 이 변화에 적응해야 한다
      - 학습이 느려진다
      - 웨이트 초기화에 뉴럴 써야 한다
      - 러닝레이트에 민감하다

    배치 정규화라는 해법:
      - 미니배치마다 레이어 입력을 정규화한다
      - 안쪽 함께 바뀜의 옮겨감을 줄인다
      - 러닝 빠르기를 더 크게 쓸 수 있다
      - 초기화에 대한 민감도를 낮춘다
      - 정칙화 노릇을 한다(드롭아웃처럼)

    식(배치 안의 피처마다):
      1. μ = mean(batch)
      2. σ² = variance(batch)
      3. x̂ = (x - μ) / √(σ² + ε)     # 정규화
      4. y = γ * x̂ + β                # 배율 조정과 이동

    여기서 γ(감마)와 β(베타)는 학습되는 파라메터이다!


    ======================================================================
    STEP 1: Loading MNIST Data
    ======================================================================
    Using device: cpu

    Dataset loaded (batch size: 128)

    ======================================================================
    STEP 2: Defining Models (With and Without BatchNorm)
    ======================================================================
    Two models defined:
      1. NetWithoutBN: Standard network
      2. NetWithBN: Network with BatchNorm layers

    BatchNorm1d:
      - For fully connected layers (1D data)
      - BatchNorm2d exists for CNNs (2D data)
      - BatchNorm3d for 3D data (videos, medical images)

    ======================================================================
    STEP 3: Understanding BatchNorm Behavior
    ======================================================================

    BatchNorm1d layer parameters:
      gamma (weight): tensor([1., 1., 1.])
      beta (bias):    tensor([0., 0., 0.])
      running_mean:   tensor([0., 0., 0.])
      running_var:    tensor([1., 1., 1.])

    Key points:
      - gamma and beta are learnable (updated by optimizer)
      - running_mean and running_var are NOT learnable
      - During training: uses batch statistics
      - During evaluation: uses running statistics
      - This is why model.train() and model.eval() matter!

    ======================================================================
    STEP 5: Training Both Models
    ======================================================================

    ----------------------------------------------------------------------
    Training Model WITHOUT BatchNorm
    ----------------------------------------------------------------------
    Epoch  3/15: Train Acc: 93.67%, Val Acc: 94.24%, Time: 46.38s
    Epoch  6/15: Train Acc: 96.97%, Val Acc: 96.15%, Time: 56.28s
    Epoch  9/15: Train Acc: 98.22%, Val Acc: 96.81%, Time: 59.30s
    Epoch 12/15: Train Acc: 98.91%, Val Acc: 97.08%, Time: 59.07s
    Epoch 15/15: Train Acc: 99.36%, Val Acc: 97.17%, Time: 56.09s

    ----------------------------------------------------------------------
    Training Model WITH BatchNorm
    ----------------------------------------------------------------------
    Epoch  3/15: Train Acc: 98.49%, Val Acc: 97.55%, Time: 62.20s
    Epoch  6/15: Train Acc: 99.62%, Val Acc: 97.68%, Time: 69.25s
    Epoch  9/15: Train Acc: 99.83%, Val Acc: 97.73%, Time: 69.09s
    Epoch 12/15: Train Acc: 99.96%, Val Acc: 98.09%, Time: 93.54s
    Epoch 15/15: Train Acc: 99.93%, Val Acc: 97.96%, Time: 91.31s

    ======================================================================
    STEP 6: Comparison
    ======================================================================

    Final Results:
    ----------------------------------------------------------------------
    Without BatchNorm:
      Best Val Accuracy: 97.21%
      Test Accuracy: 97.64%
      Avg Time/Epoch: 52.98s

    With BatchNorm:
      Best Val Accuracy: 98.29%
      Test Accuracy: 98.03%
      Avg Time/Epoch: 78.71s

    Improvement with BatchNorm: +0.39% ✓

    ======================================================================
    STEP 7: Visualizing Training Progress
    ======================================================================
    Comparison saved as '08_batchnorm_comparison.png'

    ======================================================================
    핵심 정리
    ======================================================================

    1. 배치 정규화가 하는 일:
       - 미니배치별로 레이어 입력을 정규화한다
       - 최적의 크기(γ)와 이동(β)을 학습한다
       - 추론을 위해 이동 통계를 유지한다

    2. Benefits:
       ✓ 더 빨리 수렴한다(더 큰 러닝레이트를 쓸 수 있다)
       ✓ 초기화에 대한 민감도를 낮춘다
       ✓ 정칙화 노릇을 한다(일반화가 조금 나아진다)
       ✓ 학습이 더 안정적이다(특히 깊은 뉴럴 넷트웍에서)

    3. 배치 정규화를 어디에 둘까:
       - 보통: 선형 -> 배치 정규화 -> 활성화
       - 선형 레이어 뒤, 활성화 앞
       - 출력 레이어에는 필요 없다
       - 완전 연결 레이어에는 BatchNorm1d를 쓴다
       - 컨볼루션 레이어에는 BatchNorm2d를 쓴다

    4. 학습과 평가:
       - 학습: 배치 통계를 쓴다(현재 배치의 평균과 분산)
       - 평가: 이동 통계를 쓴다(학습 중에 쌓은 값)
       - model.train()과 model.eval()을 늘 알맞게 부르라!

    5. 배치 정규화의 변형:
       - BatchNorm1d: 완전 연결 레이어용
       - BatchNorm2d: 2차원 컨볼루션 레이어용
       - BatchNorm3d: 3차원 컨볼루션 레이어용
       - LayerNorm: 대안(배치가 아니라 피처 방향으로 정규화한다)
       - GroupNorm: 배치 크기가 작아도 잘 작동한다

    6. 배치 정규화를 쓰지 말아야 할 때:
       - 배치 크기가 아주 작을 때(통계를 믿기 어렵다)
       - 순환 뉴럴 넷트웍(대신 LayerNorm을 쓴다)
       - 배치 사이의 독립성이 중요할 때

    7. 흔한 함정:
       ✗ 추론할 때 model.eval() 부르기를 잊는다
       ✗ 배치 크기를 1로 쓴다(통계가 뜻이 없다)
       ✗ 학습 모드와 평가 모드를 헷갈리기

    다음 단계:
    - 튜토리얼 09: 러닝레이트 스케줄링
    - 튜토리얼 10: 고급 구조
    - 튜토리얼 11: CIFAR-10 도전


    Training completed successfully! ✓
    ```


## 2. 논의

NetWithoutBN과 NetWithBN은 네 개의 선형 레이어를 갖는 대안 모델이다. 후자는 세 은닉 레이어에 BatchNorm1d를 추가한다. 두 모델을 이어서 하나의 뉴럴 넷트웍을 만드는 구조는 아니다.

배치 정규화의 이동 통계량은 옵티마이저가 학습하는 계수가 아니다. 기본 설정에서는 학습 중 배치로 갱신되고 평가 중 사용된다. 학습 가능한 weight와 bias와 구별해야 한다.

기록된 배치 로스의 단순 평균은 배치 크기가 다르면 전체 표본 평균과 다르다. 학습 시간 비교는 GPU 동기화 여부도 확인해야 한다. 최고 검증 정확도와 마지막 모델의 테스트 정확도는 같은 시점의 모델을 평가한 값이 아닐 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
NetWithoutBN과 배치 정규화를 추가한 모델의 학습 파라메터 수를 비교하라.

</div>

??? success "풀이"
    NetWithoutBN은 300938개이다. NetWithBN은 세 은닉 레이어의 배율·이동 1280개가 더해져 302218개이다. running_mean과 running_var는 버퍼이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
최고 검증 정확도를 보고한 뒤 마지막 모델을 테스트하면 무엇을 구분해야 하는가?

</div>

??? success "풀이"
    검증 최고 시점과 테스트에 사용한 웨이트가 같은지 확인한다. 최고 상태를 저장·복원하지 않았다면 마지막 모델의 테스트 성능이다. 최고 검증 숫자만으로 최선 체크포인트의 테스트 성능을 알 수는 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
평가에서 배치 정규화를 학습 모드로 두면 어떤 문제가 생길 수 있는가?

</div>

??? success "풀이"
    기본 설정에서 현재 배치 통계량을 사용하고 이동 통계량도 갱신하므로, 예측이 함께 들어간 표본과 배치 크기에 영향을 받을 수 있다. 평가 모드와 경사 기록 중지를 각각 설정한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

배치 정규화는 학습과 평가에서 사용하는 통계량을 구분해야 한다. 효과를 비교할 때는 계수와 버퍼, 모델 선택 시점과 최종 평가 조건을 함께 확인한다.
