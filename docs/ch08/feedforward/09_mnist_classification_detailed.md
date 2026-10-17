# MNIST 분류 자세히 보기

MNIST 이미지를 펼쳐 다층 퍼셉트론으로 분류하는 전체 예제이다. 데이터 로딩, 로짓과 교차 엔트로피, 학습, 테스트 평가와 웨이트 저장의 역할을 연결한다.

## 1. 코드

??? note "코드 (447줄)"

    ```python
    """
    ==============================================================================
    튜토리얼 06: MNIST 숫자 분류
    ==============================================================================
    난이도: ⭐⭐ 중급

    배울 내용:
    - 실제 데이터셋 다루기
    - DataLoader로 데이터 불러오기
    - 학습 분할과 검증 분할
    - 배치 처리
    - 모델 평가 지표

    선수 지식:
    - 튜토리얼 05 (nn.Module과 최적화기)

    핵심 개념:
    - torchvision.datasets
    - torch.utils.data.DataLoader
    - 배치 학습
    - 학습/시험 분할
    - 정확도 계산
    ==============================================================================
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, random_split

    torch.manual_seed(42)

    # ==============================================================================
    # 들어가며: 실제 데이터셋
    # ==============================================================================
    print("=" * 70)
    print("Welcome to MNIST Classification!")
    print("=" * 70)
    print("\nMNIST Dataset:")
    print("  - 70,000 handwritten digit images (0-9)")
    print("  - 28x28 grayscale images")
    print("  - Classic machine learning benchmark")
    print("  - Real-world computer vision task!")

    # ==============================================================================
    # 1단계: MNIST 데이터셋 불러오고 살펴보기
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Loading MNIST Dataset")
    print("=" * 70)

    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # 변환: PIL 이미지를 텐서로
    # ToTensor()는 [0, 1]로 자동 정규화한다
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 학습 데이터셋 내려받아 불러오기
    print("Downloading MNIST dataset (if not already present)...")
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    # 시험 데이터셋 불러오기
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )

    print(f"\nDataset loaded successfully!")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # 표본 하나 살펴보기
    sample_image, sample_label = train_dataset[0]
    print(f"\nSample exploration:")
    print(f"  Image shape: {sample_image.shape}")  # [채널, 높이, 너비]
    print(f"  Label: {sample_label}")
    print(f"  Image value range: [{sample_image.min():.2f}, {sample_image.max():.2f}]")

    # ==============================================================================
    # 2단계: 데이터 로더 만들기
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Creating Data Loaders")
    print("=" * 70)

    print("\nWhat is a DataLoader?")
    print("  - Batches data for efficient training")
    print("  - Shuffles data each epoch")
    print("  - Handles parallel data loading")
    print("  - Essential for large datasets!")

    batch_size = 64

    # 데이터 로더 만들기
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 더 나은 학습을 위해 섞기
        # num_workers 를 0으로 둔다. 1 이상이면 자료를 나르는 일을 딴 프로세스에
        # 맡기는데, 맥과 윈도에서는 그 프로세스를 **새로 띄우며**(spawn) 이 각본을
        # 처음부터 다시 읽는다. 로더를 만드는 이 줄이 맨 바깥에 있으므로 새 프로세스가
        # 또 로더를 만들고, 그것이 또 프로세스를 띄워 끝내 죽는다.
        # 1 이상을 쓰려면 이 아래 모든 것을 `if __name__ == "__main__":` 안으로
        # 넣어야 한다 — 그러면 다시 읽히는 쪽에서는 그 안이 돌지 않는다.
        num_workers=0
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False  # 시험 데이터는 섞을 필요가 없다
    )

    print(f"\nData loaders created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # 예: 배치 하나를 훑어보기
    images, labels = next(iter(train_loader))
    print(f"\nExample batch:")
    print(f"  Images shape: {images.shape}")  # [batch_size, 채널, 높이, 너비]
    print(f"  Labels shape: {labels.shape}")  # [batch_size]

    # ==============================================================================
    # 3단계: 표본 이미지 시각화
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Visualizing Sample Images")
    print("=" * 70)

    # 시각화를 위해 배치 하나 가져오기
    examples = iter(test_loader)
    example_images, example_labels = next(examples)

    # 표본 12개 그리기
    fig, axes = plt.subplots(2, 6, figsize=(12, 4))
    axes = axes.ravel()

    for i in range(12):
        axes[i].imshow(example_images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {example_labels[i].item()}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('06_mnist_samples.png', dpi=100)
    print("Sample images saved as '06_mnist_samples.png'")

    # ==============================================================================
    # 4단계: 뉴럴 넷트웍 정의
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Defining the Neural Network")
    print("=" * 70)

    class MNISTNet(nn.Module):
        """
        MNIST 분류를 위한 순방향 뉴럴 넷트웍

        구조:
          입력 (784) -> 은닉1 (128) -> 은닉2 (64) -> 출력 (10)

        왜 이런 구조인가?
          - 입력 784개: 28x28 화소를 펼친 것
          - 128, 64: 피처를 배우기 위한 은닉 레이어
          - 출력 10개: 숫자마다 하나 (0~9)
        """
        def __init__(self):
            super(MNISTNet, self).__init__()

            self.network = nn.Sequential(
                # 첫 번째 은닉 레이어
                nn.Linear(28 * 28, 128),
                nn.ReLU(),

                # 두 번째 은닉 레이어
                nn.Linear(128, 64),
                nn.ReLU(),

                # 출력 레이어 (활성화 없음 - CrossEntropyLoss를 쓸 것이다)
                nn.Linear(64, 10)
            )

        def forward(self, x):
            # 이미지 펼치기: [batch, 1, 28, 28] -> [batch, 784]
            x = x.view(x.size(0), -1)
            return self.network(x)

    # 모델 만들기
    model = MNISTNet().to(device)
    print("Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ==============================================================================
    # 5단계: 로스와 최적화기 정의
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Loss Function and Optimizer")
    print("=" * 70)

    # 다중 클래스 분류를 위한 CrossEntropyLoss
    # LogSoftmax와 NLLLoss를 합친다
    criterion = nn.CrossEntropyLoss()
    print("Loss function: CrossEntropyLoss")
    print("  - Perfect for multi-class classification")
    print("  - Expects raw logits (no softmax needed in model)")
    print("  - Numerically stable")

    # Adam 최적화기
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"\nOptimizer: Adam (lr={learning_rate})")

    # ==============================================================================
    # 6단계: 학습 루프
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Training the Model")
    print("=" * 70)

    n_epochs = 5
    train_losses = []
    train_accuracies = []

    print(f"Training for {n_epochs} epochs...\n")

    for epoch in range(n_epochs):
        model.train()  # 모델을 학습 모드로

        running_loss = 0.0
        correct = 0
        total = 0

        # 진행 상황 기록
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 데이터를 장치로 옮기기
            images = images.to(device)
            labels = labels.to(device)

            # 그래디언트 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 역전파와 최적화
            loss.backward()
            optimizer.step()

            # 통계
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 100 배치마다 진행 상황 출력
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        # 에포크 통계
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f'\nEpoch [{epoch+1}/{n_epochs}] Summary:')
        print(f'  Average Loss: {epoch_loss:.4f}')
        print(f'  Training Accuracy: {epoch_acc:.2f}%\n')

    # ==============================================================================
    # 7단계: 시험 집합에서 평가
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Testing the Model")
    print("=" * 70)

    model.eval()  # 모델을 평가 모드로

    test_correct = 0
    test_total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # 클래스별 정확도
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    test_accuracy = 100 * test_correct / test_total
    print(f'\nOverall Test Accuracy: {test_accuracy:.2f}%')

    print('\nPer-class accuracy:')
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i]
        print(f'  Digit {i}: {acc:.2f}%')

    # ==============================================================================
    # 8단계: 예측 시각화
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Visualizing Predictions")
    print("=" * 70)

    # 시험 이미지 몇 장 가져오기
    model.eval()
    with torch.no_grad():
        test_images, test_labels = next(iter(test_loader))
        test_images = test_images.to(device)
        outputs = model(test_images)
        _, predictions = torch.max(outputs, 1)

    # 예측 그리기
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()

    for i in range(12):
        img = test_images[i].cpu().squeeze()
        true_label = test_labels[i].item()
        pred_label = predictions[i].item()

        axes[i].imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('06_predictions.png', dpi=100)
    print("Predictions saved as '06_predictions.png'")

    # ==============================================================================
    # 9단계: 학습 과정 시각화
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 9: Training Progress Visualization")
    print("=" * 70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 학습 로스 그리기
    ax1.plot(range(1, n_epochs + 1), train_losses, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    # 학습 정확도 그리기
    ax2.plot(range(1, n_epochs + 1), train_accuracies, 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('06_training_progress.png', dpi=100)
    print("Training progress saved as '06_training_progress.png'")

    # ==============================================================================
    # 10단계: 모델 저장
    # ==============================================================================
    print("\n" + "=" * 70)
    print("STEP 10: Saving the Model")
    print("=" * 70)

    model_path = 'mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # ==============================================================================
    # 핵심 정리:
    # ==============================================================================
    print("\n" + "=" * 70)
    print("핵심 정리")
    print("=" * 70)
    print("""
    1. 실제 데이터 작업 흐름:
       a) 데이터셋을 불러온다(torchvision.datasets)
       b) 데이터 로더를 만든다(배치 구성, 섞기)
       c) 모델 구조를 정의한다
       d) 배치로 학습한다
       e) 시험 집합에서 평가한다

    2. DataLoader의 이점:
       - 자동 배치 구성
       - 데이터 섞기
       - 병렬 적재
       - 기억 자리 아끼기

    3. 분류에 쓰는 CrossEntropyLoss:
       - LogSoftmax와 NLLLoss를 합친다
       - 원본 로짓을 받는다(모델에 소프트맥스를 두지 않는다)
       - 수치적으로 안정적이다

    4. 학습 모범 사례:
       - 학습 앞에 model.train()
       - 평가 앞에 model.eval()
       - 추론 중에는 torch.no_grad()
       - 로스와 정확도를 함께 추적한다

    5. MNIST의 일반적인 정확도:
       - 단순 순전파 뉴럴 넷트웍: 95~97%
       - CNN(뒤에서 다룬다): 99% 이상
       - 우리 모델: 약 {test_accuracy:.1f}%

    다음 단계:
    - 튜토리얼 07: 검증 집합과 정칙화 더하기
    - 튜토리얼 08: 배치 정규화
    - 튜토리얼 09: 더 깊은 뉴럴 넷트웍
    - 튜토리얼 10: 고급 기법
    """)

    print("Training completed successfully! ✓")
    # ==============================================================================


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (225줄)"

    ```
    ======================================================================
    Welcome to MNIST Classification!
    ======================================================================

    MNIST Dataset:
      - 70,000 handwritten digit images (0-9)
      - 28x28 grayscale images
      - Classic machine learning benchmark
      - Real-world computer vision task!

    ======================================================================
    STEP 1: Loading MNIST Dataset
    ======================================================================
    Using device: cpu

    Downloading MNIST dataset (if not already present)...

    Dataset loaded successfully!
      Training samples: 60000
      Test samples: 10000

    Sample exploration:
      Image shape: torch.Size([1, 28, 28])
      Label: 5
      Image value range: [0.00, 1.00]

    ======================================================================
    STEP 2: Creating Data Loaders
    ======================================================================

    What is a DataLoader?
      - Batches data for efficient training
      - Shuffles data each epoch
      - Handles parallel data loading
      - Essential for large datasets!

    Data loaders created:
      Batch size: 64
      Training batches: 938
      Test batches: 157

    Example batch:
      Images shape: torch.Size([64, 1, 28, 28])
      Labels shape: torch.Size([64])

    ======================================================================
    STEP 3: Visualizing Sample Images
    ======================================================================
    Sample images saved as '06_mnist_samples.png'

    ======================================================================
    STEP 4: Defining the Neural Network
    ======================================================================
    Model architecture:
    MNISTNet(
      (network): Sequential(
        (0): Linear(in_features=784, out_features=128, bias=True)
        (1): ReLU()
        (2): Linear(in_features=128, out_features=64, bias=True)
        (3): ReLU()
        (4): Linear(in_features=64, out_features=10, bias=True)
      )
    )

    Total parameters: 109,386

    ======================================================================
    STEP 5: Loss Function and Optimizer
    ======================================================================
    Loss function: CrossEntropyLoss
      - Perfect for multi-class classification
      - Expects raw logits (no softmax needed in model)
      - Numerically stable

    Optimizer: Adam (lr=0.001)

    ======================================================================
    STEP 6: Training the Model
    ======================================================================
    Training for 5 epochs...

    Epoch [1/5], Step [100/938], Loss: 0.4545
    Epoch [1/5], Step [200/938], Loss: 0.2417
    Epoch [1/5], Step [300/938], Loss: 0.3676
    Epoch [1/5], Step [400/938], Loss: 0.3433
    Epoch [1/5], Step [500/938], Loss: 0.1827
    Epoch [1/5], Step [600/938], Loss: 0.2787
    Epoch [1/5], Step [700/938], Loss: 0.1157
    Epoch [1/5], Step [800/938], Loss: 0.3682
    Epoch [1/5], Step [900/938], Loss: 0.2025

    Epoch [1/5] Summary:
      Average Loss: 0.3366
      Training Accuracy: 90.35%

    Epoch [2/5], Step [100/938], Loss: 0.0667
    Epoch [2/5], Step [200/938], Loss: 0.1484
    Epoch [2/5], Step [300/938], Loss: 0.1382
    Epoch [2/5], Step [400/938], Loss: 0.1201
    Epoch [2/5], Step [500/938], Loss: 0.1329
    Epoch [2/5], Step [600/938], Loss: 0.0858
    Epoch [2/5], Step [700/938], Loss: 0.1096
    Epoch [2/5], Step [800/938], Loss: 0.2055
    Epoch [2/5], Step [900/938], Loss: 0.1312

    Epoch [2/5] Summary:
      Average Loss: 0.1333
      Training Accuracy: 96.05%

    Epoch [3/5], Step [100/938], Loss: 0.0865
    Epoch [3/5], Step [200/938], Loss: 0.0713
    Epoch [3/5], Step [300/938], Loss: 0.0296
    Epoch [3/5], Step [400/938], Loss: 0.1709
    Epoch [3/5], Step [500/938], Loss: 0.2423
    Epoch [3/5], Step [600/938], Loss: 0.0770
    Epoch [3/5], Step [700/938], Loss: 0.0820
    Epoch [3/5], Step [800/938], Loss: 0.0695
    Epoch [3/5], Step [900/938], Loss: 0.0555

    Epoch [3/5] Summary:
      Average Loss: 0.0914
      Training Accuracy: 97.15%

    Epoch [4/5], Step [100/938], Loss: 0.0684
    Epoch [4/5], Step [200/938], Loss: 0.0489
    Epoch [4/5], Step [300/938], Loss: 0.0804
    Epoch [4/5], Step [400/938], Loss: 0.0451
    Epoch [4/5], Step [500/938], Loss: 0.1064
    Epoch [4/5], Step [600/938], Loss: 0.0588
    Epoch [4/5], Step [700/938], Loss: 0.0642
    Epoch [4/5], Step [800/938], Loss: 0.1143
    Epoch [4/5], Step [900/938], Loss: 0.0189

    Epoch [4/5] Summary:
      Average Loss: 0.0680
      Training Accuracy: 97.94%

    Epoch [5/5], Step [100/938], Loss: 0.0096
    Epoch [5/5], Step [200/938], Loss: 0.0126
    Epoch [5/5], Step [300/938], Loss: 0.0561
    Epoch [5/5], Step [400/938], Loss: 0.0991
    Epoch [5/5], Step [500/938], Loss: 0.0387
    Epoch [5/5], Step [600/938], Loss: 0.0262
    Epoch [5/5], Step [700/938], Loss: 0.1680
    Epoch [5/5], Step [800/938], Loss: 0.0521
    Epoch [5/5], Step [900/938], Loss: 0.0338

    Epoch [5/5] Summary:
      Average Loss: 0.0534
      Training Accuracy: 98.33%


    ======================================================================
    STEP 7: Testing the Model
    ======================================================================

    Overall Test Accuracy: 97.46%

    Per-class accuracy:
      Digit 0: 98.67%
      Digit 1: 98.85%
      Digit 2: 96.80%
      Digit 3: 97.33%
      Digit 4: 95.21%
      Digit 5: 97.98%
      Digit 6: 96.56%
      Digit 7: 97.76%
      Digit 8: 97.74%
      Digit 9: 97.52%

    ======================================================================
    STEP 8: Visualizing Predictions
    ======================================================================
    Predictions saved as '06_predictions.png'

    ======================================================================
    STEP 9: Training Progress Visualization
    ======================================================================
    Training progress saved as '06_training_progress.png'

    ======================================================================
    STEP 10: Saving the Model
    ======================================================================
    Model saved to: mnist_model.pth

    ======================================================================
    핵심 정리
    ======================================================================

    1. 실제 데이터 작업 흐름:
       a) 데이터셋을 불러온다(torchvision.datasets)
       b) 데이터 로더를 만든다(배치 구성, 섞기)
       c) 모델 구조를 정의한다
       d) 배치로 학습한다
       e) 시험 집합에서 평가한다

    2. DataLoader의 이점:
       - 자동 배치 구성
       - 데이터 섞기
       - 병렬 적재
       - 기억 자리 아끼기

    3. 분류에 쓰는 CrossEntropyLoss:
       - LogSoftmax와 NLLLoss를 합친다
       - 원본 로짓을 받는다(모델에 소프트맥스를 두지 않는다)
       - 수치적으로 안정적이다

    4. 학습 모범 사례:
       - 학습 앞에 model.train()
       - 평가 앞에 model.eval()
       - 추론 중에는 torch.no_grad()
       - 로스와 정확도를 함께 추적한다

    5. MNIST의 일반적인 정확도:
       - 단순 순전파 뉴럴 넷트웍: 95~97%
       - CNN(뒤에서 다룬다): 99% 이상
       - 우리 모델: 약 {test_accuracy:.1f}%

    다음 단계:
    - 튜토리얼 07: 검증 집합과 정칙화 더하기
    - 튜토리얼 08: 배치 정규화
    - 튜토리얼 09: 더 깊은 뉴럴 넷트웍
    - 튜토리얼 10: 고급 기법

    Training completed successfully! ✓
    ```


## 2. 논의

`MNISTNet`은 784차원 입력을 128, 64차원 은닉 레이어로 변환하고 10개 로짓을 출력한다. `CrossEntropyLoss`가 로그 소프트맥스를 포함하므로 모델 출력에 소프트맥스를 별도로 적용하지 않는다.

학습에 쓰는 데이터와 테스트 데이터는 분리되어 있다. 코드의 테스트 정확도는 학습 후 정한 모델의 평가 결과로 읽는다. 이 수치를 보며 구조나 러닝레이트를 반복 선택하면 테스트셋도 선택 과정에 사용된 것이므로, 별도의 검증 분할이 필요하다.

저장하는 `state_dict`에는 파라메터와 영속 버퍼가 담긴다. 새 환경에서 복원하려면 같은 구조와 입력 전처리가 필요하며, 학습을 이어갈 목적이라면 옵티마이저 상태와 반복 위치도 함께 보관한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
MNISTNet의 레이어별 파라메터 개수를 계산하라.

</div>

??? success "풀이"
    $784\to128$은 $100480$개, $128\to64$는 $8256$개, $64\to10$은 $650$개로 총 $109386$개이다. ReLU와 평탄화는 학습 파라메터를 추가하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 코드는 이미 Adam을 사용한다. SGD와 비교하려면 어떤 조건을 통제해야 하는가?

</div>

??? success "풀이"
    같은 초기 웨이트와 데이터 분할·학습 예산을 사용하고, 두 옵티마이저의 러닝레이트를 각각 검증셋에서 선택한다. 한쪽만 충분히 조율한 결과를 알고리즘의 보편적인 우열로 해석하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
모델 복원 후 예측이 달라질 때 점검할 항목을 두 가지 이상 제시하라.

</div>

??? success "풀이"
    같은 구조와 웨이트를 불러왔는지, 이미지의 스케일·평탄화·클래스 순서가 같은지 확인한다. 드롭아웃 등이 있다면 평가 모드도 맞춘다. 장치와 자료형 차이로 인한 작은 수치 차이와 큰 구현 차이를 구분한다.

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

MNIST 분류의 입력 차원은 784이고 출력은 10개 로짓이다. 학습 로스와 테스트 정확도는 서로 다른 역할의 지표이다. 모델 선택에는 검증셋을 사용하고 복원에 필요한 구조와 전처리를 웨이트와 함께 기록한다.
