# 완전한 MNIST 학습

MNIST 분류의 데이터 로딩, 컨볼루션 모델, 학습·평가, 체크포인트 저장과 복원을 연결하는 예제이다. 각 구성 요소가 무엇을 보장하며 어떤 추가 설정이 필요한지 살펴본다.

## 1. 코드

??? note "코드 (519줄)"

    ```python
    """
    ================================================================================
    실전 예제: 완전한 MNIST 숫자 분류
    ================================================================================

    배울 내용:
    - 처음부터 끝까지의 완전한 학습 파이프라인
    - 데이터 적재와 전처리
    - 모델 정의
    - 로스와 최적화기를 쓰는 학습
    - 검증과 시험
    - 모델의 저장과 불러오기
    - 실전 코드의 좋은 관행

    선수 지식:
    - 입문자용 튜토리얼을 모두 마친다
    - CNN에 대한 기본 이해

    소요 시간: 약 30분
    ================================================================================
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import time

    print("=" * 80)
    print("COMPLETE MNIST DIGIT CLASSIFICATION")
    print("=" * 80)

    # ============================================================================
    # 1절: 설정과 준비
    # ============================================================================
    print("\n" + "-" * 80)
    print("CONFIGURATION")
    print("-" * 80)

    # 재현성을 위해 난수 씨앗 고정
    torch.manual_seed(42)

    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 초파라메터
    config = {
        'batch_size': 64,
        'test_batch_size': 1000,
        'epochs': 5,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'log_interval': 100,  # N 배치마다 출력
        'save_model': True,
        'model_path': 'mnist_model.pt'
    }

    print("\nHyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # ============================================================================
    # 2절: 데이터 적재와 전처리
    # ============================================================================
    print("\n" + "-" * 80)
    print("DATA LOADING")
    print("-" * 80)

    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.ToTensor(),  # PIL 이미지를 텐서로 바꾸기
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST의 평균과 표준편차로 정규화
    ])

    print("Downloading MNIST dataset...")

    # 학습 데이터 내려받아 불러오기
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 시험 데이터 내려받아 불러오기
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # 데이터 로더 만들기
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,  # 학습 데이터 섞기
        num_workers=0  # 데이터 적재에 쓸 프로세스의 수
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['test_batch_size'],
        shuffle=False,  # 시험 데이터는 섞지 않는다
        num_workers=0
    )

    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: 10 (digits 0-9)")
    print(f"  Image size: 28x28 pixels")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # ============================================================================
    # 3절: 모델 정의
    # ============================================================================
    print("\n" + "-" * 80)
    print("MODEL ARCHITECTURE")
    print("-" * 80)

    class ConvNet(nn.Module):
        """
        MNIST를 위한 컨볼루션 뉴럴 넷트웍

        구조:
        - 컨볼루션 레이어 1: 채널 1 → 32, 3x3 커널
        - 컨볼루션 레이어 2: 채널 32 → 64, 3x3 커널
        - 최댓값 풀링: 2x2
        - 완전 연결 1: 9216 → 128
        - 완전 연결 2: 128 → 10 (클래스)
        """
        def __init__(self):
            super(ConvNet, self).__init__()
            # 컨볼루션 레이어
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

            # 풀링 레이어
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # 정칙화를 위한 드롭아웃
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)

            # 완전 연결 레이어
            # conv1, conv2와 풀링 2번을 거치면: 28 → 14 → 7
            # 피처 맵의 크기: 7 * 7 * 64 = 3136
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            # 컨볼루션 블록 1
            x = F.relu(self.conv1(x))  # 28x28x32
            x = self.pool(x)            # 14x14x32

            # 컨볼루션 블록 2
            x = F.relu(self.conv2(x))  # 14x14x64
            x = self.pool(x)            # 7x7x64
            x = self.dropout1(x)

            # 펼치기
            x = x.view(-1, 64 * 7 * 7)  # 벡터로 펼치기

            # 완전 연결 레이어
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)  # 활성화 없음 (CrossEntropyLoss가 소프트맥스를 적용한다)

            return x

    # 모델을 만들어 장치로 옮기기
    model = ConvNet().to(device)

    # 파라메터 세기
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ============================================================================
    # 4절: 로스 함수와 최적화기
    # ============================================================================
    print("\n" + "-" * 80)
    print("LOSS FUNCTION AND OPTIMIZER")
    print("-" * 80)

    # 로스 함수
    criterion = nn.CrossEntropyLoss()
    print(f"Loss function: {criterion}")

    # 최적화기
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum']
    )
    print(f"Optimizer: {optimizer}")

    # 러닝레이트 스케줄러
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,  # 에포크마다 감쇠
        gamma=0.7     # 러닝레이트에 0.7을 곱한다
    )
    print(f"LR Scheduler: StepLR(step_size=1, gamma=0.7)")

    # ============================================================================
    # 5절: 학습 함수
    # ============================================================================

    def train(model, device, train_loader, optimizer, criterion, epoch):
        """
        모델을 한 에포크 동안 학습시킨다

        인수:
            model: 뉴럴 넷트웍 모델
            device: 학습에 쓸 장치 (CPU/GPU)
            train_loader: 학습 데이터의 DataLoader
            optimizer: 파라메터 갱신에 쓸 최적화기
            criterion: 로스 함수
            epoch: 현재 에포크 번호
        """
        model.train()  # 모델을 학습 모드로

        total_loss = 0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            # 데이터를 장치로 옮기기
            data, target = data.to(device), target.to(device)

            # 그래디언트 초기화
            optimizer.zero_grad()

            # 순전파
            output = model(data)

            # 로스 계산
            loss = criterion(output, target)

            # 역전파
            loss.backward()

            # 파라메터 갱신
            optimizer.step()

            # 통계
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 진행 상황 출력
            if batch_idx % config['log_interval'] == 0:
                print(f'  Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100. * correct / total:.2f}%')

        # 에포크 통계
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        print(f'\n  Epoch {epoch} Summary:')
        print(f'    Avg Loss: {avg_loss:.4f}')
        print(f'    Accuracy: {accuracy:.2f}%')
        print(f'    Time: {epoch_time:.2f}s')

        return avg_loss, accuracy

    # ============================================================================
    # 6절: 검증/시험 함수
    # ============================================================================

    def test(model, device, test_loader, criterion):
        """
        시험 데이터에서 모델을 평가한다

        인수:
            model: 뉴럴 넷트웍 모델
            device: 시험에 쓸 장치 (CPU/GPU)
            test_loader: 시험 데이터의 DataLoader
            criterion: 로스 함수

        반환값:
            평균 로스와 정확도
        """
        model.eval()  # 모델을 평가 모드로

        test_loss = 0
        correct = 0
        total = 0

        # 시험 중에는 그래디언트를 계산하지 않는다
        with torch.no_grad():
            for data, target in test_loader:
                # 데이터를 장치로 옮기기
                data, target = data.to(device), target.to(device)

                # 순전파
                output = model(data)

                # 로스 계산
                test_loss += criterion(output, target).item()

                # 예측을 얻는다
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        # 통계 계산
        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total

        print(f'\n  Test Results:')
        print(f'    Avg Loss: {avg_loss:.4f}')
        print(f'    Accuracy: {accuracy:.2f}% ({correct}/{total})')

        return avg_loss, accuracy

    # ============================================================================
    # 7절: 학습 루프
    # ============================================================================
    print("\n" + "-" * 80)
    print("TRAINING")
    print("-" * 80)

    # 이력 기록
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    print(f"\nTraining for {config['epochs']} epochs...\n")

    for epoch in range(1, config['epochs'] + 1):
        print(f"{'=' * 80}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'=' * 80}")

        # 학습
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 시험
        test_loss, test_acc = test(model, device, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # 러닝레이트를 갱신한다
        scheduler.step()

        print()

    # ============================================================================
    # 8절: 모델 저장
    # ============================================================================
    print("\n" + "-" * 80)
    print("SAVING MODEL")
    print("-" * 80)

    if config['save_model']:
        # 완전한 모델 저장
        torch.save({
            'epoch': config['epochs'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1],
            'test_loss': test_losses[-1],
            'test_accuracy': test_accuracies[-1]
        }, config['model_path'])

        print(f"Model saved to: {config['model_path']}")

        # 웨이트만 따로 저장하기도 한다 (파일이 더 작다)
        weights_path = config['model_path'].replace('.pt', '_weights.pt')
        torch.save(model.state_dict(), weights_path)
        print(f"Model weights saved to: {weights_path}")

    # ============================================================================
    # 9절: 모델 불러와 시험하기
    # ============================================================================
    print("\n" + "-" * 80)
    print("LOADING MODEL")
    print("-" * 80)

    # 새 모델 인스턴스 만들기
    loaded_model = ConvNet().to(device)

    # 체크포인트를 불러온다
    checkpoint = torch.load(config['model_path'])
    loaded_model.load_state_dict(checkpoint['model_state_dict'])

    print("Model loaded successfully!")
    print(f"  Trained for: {checkpoint['epoch']} epochs")
    print(f"  Final test accuracy: {checkpoint['test_accuracy']:.2f}%")

    # 불러온 모델 시험
    print("\nVerifying loaded model:")
    test_loss, test_acc = test(loaded_model, device, test_loader, criterion)

    # ============================================================================
    # 10절: 추론 예제
    # ============================================================================
    print("\n" + "-" * 80)
    print("INFERENCE EXAMPLE")
    print("-" * 80)

    # 시험 이미지 배치 하나 가져오기
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # 처음 이미지 5장 가져오기
    images = images[:5].to(device)
    labels = labels[:5]

    # 예측한다
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)

    print("Sample Predictions:")
    print(f"{'True':^6} {'Pred':^6} {'Confidence':^12}")
    print("-" * 26)

    for i in range(5):
        true_label = labels[i].item()
        pred_label = predictions[i].item()
        confidence = probabilities[i, pred_label].item()
        status = "✓" if true_label == pred_label else "✗"

        print(f"  {true_label}      {pred_label}      {confidence*100:5.1f}%    {status}")

    # ============================================================================
    # 11절: 학습 요약
    # ============================================================================
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    print(f"\nFinal Results:")
    print(f"  Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"  Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"  Training Loss: {train_losses[-1]:.4f}")
    print(f"  Test Loss: {test_losses[-1]:.4f}")

    print(f"\nTraining Progress:")
    for epoch in range(config['epochs']):
        print(f"  Epoch {epoch+1}: "
              f"Train Acc={train_accuracies[epoch]:.2f}%, "
              f"Test Acc={test_accuracies[epoch]:.2f}%, "
              f"Test Loss={test_losses[epoch]:.4f}")

    # ============================================================================
    # 요약
    # ============================================================================
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. 완전한 파이프라인:
       ✓ 자료 불러오기와 미리 다듬기
       ✓ 모형 매기기
       ✓ 로스 함수와 최적화기 설정
       ✓ 검증을 곁들인 학습 루프
       ✓ 모델 저장과 불러오기
       ✓ Inference

    2. 보여 준 모범 사례:
       ✓ 효율적인 배치 처리를 위해 DataLoader를 쓴다
       ✓ model.train()과 model.eval()을 알맞게 설정한다
       ✓ 추론에는 torch.no_grad()를 쓴다
       ✓ 학습 중 지표를 추적한다
       ✓ 메타데이터와 함께 체크포인트를 저장한다
       ✓ 러닝레이트 스케줄링을 쓴다
       ✓ 정칙화를 위해 드롭아웃을 더한다

    3. 실서비스에서 살필 점:
       ✓ 설정 다루기
       ✓ 되풀이할 수 있음(마구잡이 씨앗)
       ✓ 장치 처리(CPU와 GPU)
       ✓ 진행 상황 기록
       ✓ 오류 처리(여기 나오지 않지만 중요하다)
       ✓ 모델 버전 관리

    4. 최적화 선택:
       • 모멘텀을 쓰는 SGD(믿을 만하고 검증되었다)
       • CrossEntropyLoss(분류의 표준)
       • StepLR 스케줄러(점진적인 러닝레이트 감쇠)
       • 드롭아웃(과적합을 막는다)

    다음 단계:
    → 여러 구조로 실험해 보라
    → 다른 최적화기(Adam, AdamW)를 써 보라
    → 데이터 증강을 더해 보라
    → 조기 종료를 구현해 보라
    → 시각화에는 텐서보드를 쓰라
    → 자신의 데이터셋에 써 보라!
    """)
    print("=" * 80)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (286줄)"

    ```
    ================================================================================
    COMPLETE MNIST DIGIT CLASSIFICATION
    ================================================================================

    --------------------------------------------------------------------------------
    CONFIGURATION
    --------------------------------------------------------------------------------
    Using device: cpu

    Hyperparameters:
      batch_size: 64
      test_batch_size: 1000
      epochs: 5
      learning_rate: 0.01
      momentum: 0.9
      log_interval: 100
      save_model: True
      model_path: mnist_model.pt

    --------------------------------------------------------------------------------
    DATA LOADING
    --------------------------------------------------------------------------------
    Downloading MNIST dataset...

    Dataset Statistics:
      Training samples: 60000
      Test samples: 10000
      Number of classes: 10 (digits 0-9)
      Image size: 28x28 pixels
      Training batches: 938
      Test batches: 10

    --------------------------------------------------------------------------------
    MODEL ARCHITECTURE
    --------------------------------------------------------------------------------
    Model Architecture:
    ConvNet(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (dropout1): Dropout(p=0.25, inplace=False)
      (dropout2): Dropout(p=0.5, inplace=False)
      (fc1): Linear(in_features=3136, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=10, bias=True)
    )

    Total parameters: 421,642
    Trainable parameters: 421,642

    --------------------------------------------------------------------------------
    LOSS FUNCTION AND OPTIMIZER
    --------------------------------------------------------------------------------
    Loss function: CrossEntropyLoss()
    Optimizer: SGD (
    Parameter Group 0
        dampening: 0
        differentiable: False
        foreach: None
        fused: None
        lr: 0.01
        maximize: False
        momentum: 0.9
        nesterov: False
        weight_decay: 0
    )
    LR Scheduler: StepLR(step_size=1, gamma=0.7)

    --------------------------------------------------------------------------------
    TRAINING
    --------------------------------------------------------------------------------

    Training for 5 epochs...

    ================================================================================
    Epoch 1/5
    Current LR: 0.010000
    ================================================================================
      Batch [0/938] Loss: 2.2902 Acc: 10.94%
      Batch [100/938] Loss: 0.4061 Acc: 60.26%
      Batch [200/938] Loss: 0.3511 Acc: 73.86%
      Batch [300/938] Loss: 0.5002 Acc: 80.00%
      Batch [400/938] Loss: 0.1813 Acc: 83.40%
      Batch [500/938] Loss: 0.2966 Acc: 85.52%
      Batch [600/938] Loss: 0.1673 Acc: 87.18%
      Batch [700/938] Loss: 0.0808 Acc: 88.34%
      Batch [800/938] Loss: 0.0449 Acc: 89.28%
      Batch [900/938] Loss: 0.1546 Acc: 90.03%

      Epoch 1 Summary:
        Avg Loss: 0.3058
        Accuracy: 90.29%
        Time: 778.68s

      Test Results:
        Avg Loss: 0.0623
        Accuracy: 98.12% (9812/10000)

    ================================================================================
    Epoch 2/5
    Current LR: 0.007000
    ================================================================================
      Batch [0/938] Loss: 0.1803 Acc: 93.75%
      Batch [100/938] Loss: 0.1642 Acc: 97.05%
      Batch [200/938] Loss: 0.0898 Acc: 97.06%
      Batch [300/938] Loss: 0.0551 Acc: 96.97%
      Batch [400/938] Loss: 0.0978 Acc: 96.97%
      Batch [500/938] Loss: 0.0393 Acc: 97.00%
      Batch [600/938] Loss: 0.0454 Acc: 97.10%
      Batch [700/938] Loss: 0.0188 Acc: 97.14%
      Batch [800/938] Loss: 0.0453 Acc: 97.20%
      Batch [900/938] Loss: 0.0598 Acc: 97.21%

      Epoch 2 Summary:
        Avg Loss: 0.0924
        Accuracy: 97.20%
        Time: 179.71s

      Test Results:
        Avg Loss: 0.0402
        Accuracy: 98.66% (9866/10000)

    ================================================================================
    Epoch 3/5
    Current LR: 0.004900
    ================================================================================
      Batch [0/938] Loss: 0.1237 Acc: 93.75%
      Batch [100/938] Loss: 0.1517 Acc: 97.60%
      Batch [200/938] Loss: 0.0569 Acc: 97.73%
      Batch [300/938] Loss: 0.0392 Acc: 97.72%
      Batch [400/938] Loss: 0.0488 Acc: 97.76%
      Batch [500/938] Loss: 0.0501 Acc: 97.84%
      Batch [600/938] Loss: 0.0817 Acc: 97.90%
      Batch [700/938] Loss: 0.0998 Acc: 97.88%
      Batch [800/938] Loss: 0.0882 Acc: 97.91%
      Batch [900/938] Loss: 0.0136 Acc: 97.92%

      Epoch 3 Summary:
        Avg Loss: 0.0707
        Accuracy: 97.93%
        Time: 171.69s

      Test Results:
        Avg Loss: 0.0337
        Accuracy: 98.85% (9885/10000)

    ================================================================================
    Epoch 4/5
    Current LR: 0.003430
    ================================================================================
      Batch [0/938] Loss: 0.1114 Acc: 96.88%
      Batch [100/938] Loss: 0.0523 Acc: 98.30%
      Batch [200/938] Loss: 0.0349 Acc: 98.21%
      Batch [300/938] Loss: 0.0366 Acc: 98.19%
      Batch [400/938] Loss: 0.0285 Acc: 98.24%
      Batch [500/938] Loss: 0.0640 Acc: 98.23%
      Batch [600/938] Loss: 0.1009 Acc: 98.22%
      Batch [700/938] Loss: 0.0998 Acc: 98.22%
      Batch [800/938] Loss: 0.0661 Acc: 98.24%
      Batch [900/938] Loss: 0.1045 Acc: 98.26%

      Epoch 4 Summary:
        Avg Loss: 0.0599
        Accuracy: 98.25%
        Time: 126.60s

      Test Results:
        Avg Loss: 0.0310
        Accuracy: 99.00% (9900/10000)

    ================================================================================
    Epoch 5/5
    Current LR: 0.002401
    ================================================================================
      Batch [0/938] Loss: 0.0123 Acc: 100.00%
      Batch [100/938] Loss: 0.2034 Acc: 98.25%
      Batch [200/938] Loss: 0.2677 Acc: 98.22%
      Batch [300/938] Loss: 0.0197 Acc: 98.21%
      Batch [400/938] Loss: 0.0532 Acc: 98.27%
      Batch [500/938] Loss: 0.0177 Acc: 98.27%
      Batch [600/938] Loss: 0.0260 Acc: 98.32%
      Batch [700/938] Loss: 0.0057 Acc: 98.36%
      Batch [800/938] Loss: 0.0669 Acc: 98.35%
      Batch [900/938] Loss: 0.0441 Acc: 98.35%

      Epoch 5 Summary:
        Avg Loss: 0.0551
        Accuracy: 98.36%
        Time: 90.69s

      Test Results:
        Avg Loss: 0.0274
        Accuracy: 99.02% (9902/10000)


    --------------------------------------------------------------------------------
    SAVING MODEL
    --------------------------------------------------------------------------------
    Model saved to: mnist_model.pt
    Model weights saved to: mnist_model_weights.pt

    --------------------------------------------------------------------------------
    LOADING MODEL
    --------------------------------------------------------------------------------
    Model loaded successfully!
      Trained for: 5 epochs
      Final test accuracy: 99.02%

    Verifying loaded model:

      Test Results:
        Avg Loss: 0.0274
        Accuracy: 99.02% (9902/10000)

    --------------------------------------------------------------------------------
    INFERENCE EXAMPLE
    --------------------------------------------------------------------------------
    Sample Predictions:
     True   Pred   Confidence 
    --------------------------
      7      7      100.0%    ✓
      2      2      100.0%    ✓
      1      1      100.0%    ✓
      0      0      100.0%    ✓
      4      4      100.0%    ✓

    ================================================================================
    TRAINING SUMMARY
    ================================================================================

    Final Results:
      Training Accuracy: 98.36%
      Test Accuracy: 99.02%
      Training Loss: 0.0551
      Test Loss: 0.0274

    Training Progress:
      Epoch 1: Train Acc=90.29%, Test Acc=98.12%, Test Loss=0.0623
      Epoch 2: Train Acc=97.20%, Test Acc=98.66%, Test Loss=0.0402
      Epoch 3: Train Acc=97.93%, Test Acc=98.85%, Test Loss=0.0337
      Epoch 4: Train Acc=98.25%, Test Acc=99.00%, Test Loss=0.0310
      Epoch 5: Train Acc=98.36%, Test Acc=99.02%, Test Loss=0.0274

    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. 완전한 파이프라인:
       ✓ 자료 불러오기와 미리 다듬기
       ✓ 모형 매기기
       ✓ 로스 함수와 최적화기 설정
       ✓ 검증을 곁들인 학습 루프
       ✓ 모델 저장과 불러오기
       ✓ Inference

    2. 보여 준 모범 사례:
       ✓ 효율적인 배치 처리를 위해 DataLoader를 쓴다
       ✓ model.train()과 model.eval()을 알맞게 설정한다
       ✓ 추론에는 torch.no_grad()를 쓴다
       ✓ 학습 중 지표를 추적한다
       ✓ 메타데이터와 함께 체크포인트를 저장한다
       ✓ 러닝레이트 스케줄링을 쓴다
       ✓ 정칙화를 위해 드롭아웃을 더한다

    3. 실서비스에서 살필 점:
       ✓ 설정 다루기
       ✓ 되풀이할 수 있음(마구잡이 씨앗)
       ✓ 장치 처리(CPU와 GPU)
       ✓ 진행 상황 기록
       ✓ 오류 처리(여기 나오지 않지만 중요하다)
       ✓ 모델 버전 관리

    4. 최적화 선택:
       • 모멘텀을 쓰는 SGD(믿을 만하고 검증되었다)
       • CrossEntropyLoss(분류의 표준)
       • StepLR 스케줄러(점진적인 러닝레이트 감쇠)
       • 드롭아웃(과적합을 막는다)

    다음 단계:
    → 여러 구조로 실험해 보라
    → 다른 최적화기(Adam, AdamW)를 써 보라
    → 데이터 증강을 더해 보라
    → 조기 종료를 구현해 보라
    → 시각화에는 텐서보드를 쓰라
    → 자신의 데이터셋에 써 보라!

    ================================================================================
    ```


## 2. 논의

학습은 train() 모드에서 로짓과 교차 엔트로피를 계산하고, 그래디언트 초기화·역전파·옵티마이저 갱신을 반복한다. 평가는 eval()과 no_grad()를 함께 사용한다. 전자는 드롭아웃 같은 모듈의 동작을 바꾸고 후자는 자동 미분 그래프 생성을 막으므로 역할이 다르다. 이 ConvNet에는 드롭아웃이 있고 BatchNorm은 없다.

코드는 별도 검증셋을 만들지 않고 매 에포크 MNIST 테스트셋을 평가한다. 결과를 관찰하는 것과 그 결과로 에포크 수나 구조를 선택하는 것을 구분해야 한다. 선택에 사용했다면 그 분할은 검증셋 역할을 한 것이며 최종 보고용 테스트셋이 추가로 필요하다. 로스 기록은 배치 평균들의 단순 평균이라 마지막 배치의 크기가 다르면 전체 표본 평균과 차이가 난다.

체크포인트에는 모델·옵티마이저 상태와 일부 메타데이터가 담기지만 스케줄러 상태와 난수 상태까지 모두 저장하는 것은 아니다. 같은 경로로 정확히 학습을 재개하려면 스케줄러, 난수 생성기, 데이터 순서와 반복 위치도 복원해야 한다. 웨이트 파일만으로는 모델 구조나 전처리가 자동 복원되지 않는다. 코드의 시드 고정 또한 모든 장치와 환경에서 비트 단위 재현성을 보장하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
ConvNet에서 두 번의 풀링 후 완전 연결 레이어로 들어가는 피처 수를 구하라.

</div>

??? success "연습문제 1 풀이"
    패딩한 3×3 컨볼루션은 공간 크기를 유지하고 두 번의 2×2 풀링이 28→14→7로 줄인다. 채널이 64이므로 표본당 피처는 64×7×7=3136개이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
eval()과 no_grad() 중 하나만 사용해도 충분한가?

</div>

??? success "연습문제 2 풀이"
    역할이 달라 대체할 수 없다. eval()만으로 그래프 생성이 꺼지지 않고, no_grad()만으로 드롭아웃이 평가 동작으로 바뀌지 않는다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
저장된 모델·옵티마이저 상태만으로 동일한 학습 경로를 이어갈 수 있는가?

</div>

??? success "연습문제 3 풀이"
    항상 그렇지는 않다. 스케줄러, 난수 상태, 데이터 로더의 순서, 반복 위치와 실행 환경도 영향을 준다. 재개 목적의 체크포인트와 추론용 웨이트 저장을 구분한다.

## 정리하며

전체 학습 예제를 읽을 때는 학습·평가 모드, 분할의 역할과 저장 범위를 함께 확인한다. 지표를 기록하고 웨이트를 복원하는 기능만으로 독립적인 평가나 정확한 재개가 완성되는 것은 아니다.
