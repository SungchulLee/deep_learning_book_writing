# DataLoader로 배치 처리하기

분류 데이터를 TensorDataset과 DataLoader로 묶어 미니배치 학습과 평가를 구성한다. 배치별 경사와 데이터 전체의 지표를 구분한다.

## 1. 코드

??? note "코드 (450줄)"

    ```python
    """
    ================================================================================
    02_dataloader_batching.py - DataLoader으로 하는 효율적인 배치 다루기
    ================================================================================

    학습 목표:
    - 작은 배치 경사 하강법을 이해한다
    - 효율적인 배치 만들기에 PyTorch DataLoader를 쓴다
    - Dataset과 DataLoader 클래스를 배운다
    - 섞기와 배치 다루기를 익힌다
    - 더 큰 데이터셋으로 넓힌다

    PREREQUISITES:
    - 01_proper_training_loop.py을 마쳤을 것
    - 경사 하강법을 이해하고 있을 것

    소요 시간: 1시간 반쯤

    어려움: ⭐⭐⭐☆☆ (가운데)
    ================================================================================
    """

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    from typing import Tuple

    print("="*80)
    print("EFFICIENT BATCH PROCESSING WITH DATALOADER")
    print("="*80)

    # =============================================================================
    # 1부: 배치 처리 이해하기
    # =============================================================================

    print("\n" + "="*80)
    print("PART 1: WHY USE BATCHES?")
    print("="*80)

    print("""
    작은 배치 경사 하강법:

    데이터를 한꺼번에 다루거나(온 배치) 한 번에 하나씩 다루는(확률) 대신
    작은 배치로 다룬다.

    배치 크기 = 1(SGD):
      ✓ 고치기가 빠르다
      ✗ 기울기에 잡음이 많다
      ✗ 셈이 느리다(벡터로 묶지 못한다)

    배치 크기 = 모두(온 배치 경사 하강법):
      ✓ 기울기가 든든하다
      ✗ 기억 자리를 많이 쓴다
      ✗ 데이터가 크면 느리다

    배치 크기 = 32~256(작은 배치):
      ✓ 속도와 든든함이 고루 좋다
      ✓ GPU을 잘 쓴다
      ✓ 잡음에서 오는 정칙화 효과
      ✓ 기억 자리보다 큰 데이터셋도 다룰 수 있다

    흔한 배치 크기: 16, 32, 64, 128, 256
    """)

    # =============================================================================
    # 2부: 데이터 준비
    # =============================================================================

    print("\n" + "="*80)
    print("PART 2: PREPARING DATA")
    print("="*80)

    # 더 큰 데이터셋을 생성한다
    torch.manual_seed(42)
    np.random.seed(42)

    X, y = make_classification(
        n_samples=5000,  # Larger dataset
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    print(f"Dataset size: {X.shape[0]} samples")

    # 나누고 표준화한다
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 텐서로 바꾼다
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)

    print(f"Training: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")

    # =============================================================================
    # 3부: 데이터로더 만들기
    # =============================================================================

    print("\n" + "="*80)
    print("PART 3: CREATING DATALOADERS")
    print("="*80)

    # 데이터셋들을 만든다
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    print("\nDataset created using TensorDataset")
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Each item shape: features={train_dataset[0][0].shape}, label={train_dataset[0][1].shape}")

    # 데이터로더들을 만든다
    batch_size = 64

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # Shuffle training data each epoch
        num_workers=0,     # 0 for Windows, 2-4 for Linux/Mac
        drop_last=False    # Keep last incomplete batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,     # Don't shuffle test data
        num_workers=0
    )

    print(f"\nDataLoader created:")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches (train): {len(train_loader)}")
    print(f"Number of batches (test): {len(test_loader)}")
    print(f"Samples per batch: {batch_size}")
    print(f"Last batch size (train): {len(X_train) % batch_size if len(X_train) % batch_size != 0 else batch_size}")

    # =============================================================================
    # 4부: 모델 정의
    # =============================================================================

    class LogisticRegression(nn.Module):
        def __init__(self, input_dim):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    model = LogisticRegression(X_train.shape[1])

    # =============================================================================
    # 5부: 배치를 쓰는 학습
    # =============================================================================

    print("\n" + "="*80)
    print("PART 5: TRAINING WITH MINI-BATCHES")
    print("="*80)

    def train_epoch_with_batches(model, dataloader, criterion, optimizer):
        """배치를 써서 한 에폭을 학습한다"""
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        # 배치들을 순회한다
        for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
            # batch_X 꼴: (batch_size, n_features)
            # batch_y 꼴: (batch_size, 1)

            # 순전파
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 지표를 추적한다
            total_loss += loss.item() * len(batch_X)  # Accumulate loss
            predicted_classes = (predictions >= 0.5).float()
            correct += (predicted_classes == batch_y).sum().item()
            total += len(batch_X)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy


    def validate_with_batches(model, dataloader, criterion):
        """배치를 써서 검증한다"""
        model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)

                total_loss += loss.item() * len(batch_X)
                predicted_classes = (predictions >= 0.5).float()
                correct += (predicted_classes == batch_y).sum().item()
                total += len(batch_X)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy


    # 학습 준비
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    print(f"Training for {num_epochs} epochs with batch_size={batch_size}")
    print("-" * 60)

    for epoch in range(num_epochs):
        # 학습
        train_loss, train_acc = train_epoch_with_batches(
            model, train_loader, criterion, optimizer
        )

        # 검증
        test_loss, test_acc = validate_with_batches(
            model, test_loader, criterion
        )

        # 이력 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # 진행 상황 출력
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    print("\nTraining completed!")

    # =============================================================================
    # 6부: 배치 크기 비교
    # =============================================================================

    print("\n" + "="*80)
    print("PART 6: COMPARING DIFFERENT BATCH SIZES")
    print("="*80)

    def train_with_batch_size(batch_size, num_epochs=30):
        """특정 배치 크기로 모델을 학습시킨다"""
        # 새 모델을 만든다
        model = LogisticRegression(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 데이터로더를 만든다
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        losses = []

        for epoch in range(num_epochs):
            train_loss, _ = train_epoch_with_batches(
                model, train_loader, criterion, optimizer
            )
            losses.append(train_loss)

        final_test_loss, final_test_acc = validate_with_batches(
            model, test_loader, criterion
        )

        return losses, final_test_acc

    # 여러 배치 크기를 비교한다
    batch_sizes = [16, 32, 64, 128, 256]
    results = {}

    print("Training with different batch sizes...")
    for bs in batch_sizes:
        print(f"  Batch size {bs:3d}... ", end="", flush=True)
        losses, acc = train_with_batch_size(bs, num_epochs=30)
        results[bs] = {'losses': losses, 'accuracy': acc}
        print(f"Final accuracy: {acc:.4f}")

    # =============================================================================
    # 7부: 시각화
    # =============================================================================

    print("\n" + "="*80)
    print("PART 7: VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(15, 10))

    # 그림 1: 학습 곡선
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves (Batch Size = 64)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 그림 2: 정확도 곡선
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(history['test_acc'], label='Test Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 그림 3: 배치 크기 비교 (손실 곡선)
    ax3 = plt.subplot(2, 2, 3)
    for bs in batch_sizes:
        ax3.plot(results[bs]['losses'], label=f'BS={bs}', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training Loss')
    ax3.set_title('Effect of Batch Size on Training', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 그림 4: 배치 크기별 최종 정확도
    ax4 = plt.subplot(2, 2, 4)
    accuracies = [results[bs]['accuracy'] for bs in batch_sizes]
    bars = ax4.bar(range(len(batch_sizes)), accuracies, color='steelblue', alpha=0.7)
    ax4.set_xticks(range(len(batch_sizes)))
    ax4.set_xticklabels(batch_sizes)
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Final Test Accuracy')
    ax4.set_title('Final Accuracy vs Batch Size', fontweight='bold')
    ax4.set_ylim([min(accuracies)-0.01, max(accuracies)+0.01])
    ax4.grid(True, alpha=0.3, axis='y')

    # 막대에 값 레이블을 추가한다
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')

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
    1. DataLoader의 좋은 점
       ✓ 절로 묶어 준다
       ✓ 기억 자리를 잘 쓴다
       ✓ 붙박이 섞기
       ✓ 나란히 데이터 불러오기(num_workers)
       ✓ 기억 자리보다 큰 데이터셋도 다룬다

    2. 배치 크기 고르기
       ✓ 작은 배치: 잡음이 많고 정칙화가 잘 된다
       ✓ 큰 배치: 더 든든하고 학습이 빠르다
       ✓ 흔히 고르는 값: 32, 64, 128
       ✓ GPU 기억 자리에 매인다

    3. 좋은 버릇
       ✓ 학습 데이터는 늘 섞어라
       ✓ 시험/검증 데이터는 섞지 마라
       ✓ 단순한 자리에는 TensorDataset을 써라
       ✓ 복잡한 데이터에는 맞춤 Dataset을 만들어라
       ✓ 윈도에서는 num_workers=0으로 두어라

    4. 언제 배치를 쓸까
       ✓ 표본이 10,000개를 넘는 데이터셋
       ✓ 기억 자리가 넉넉하지 않을 때
       ✓ GPU으로 익힐 때
       ✓ 정칙화 효과를 바랄 때
    """)

    print("\n" + "="*80)
    print("EXERCISES")
    print("="*80)
    print("""
    1. 쉬움: batch_size=1으로 해 보아라. 작은 배치와 견주면 어떠한가?

    2. 보통: 맞춤 Dataset 클래스를 짜라.
       class CustomDataset(Dataset):
           def __init__(self, X, y):
               # 여기에 코드를 작성한다

           def __len__(self):
               # 여기에 코드를 작성한다

           def __getitem__(self, idx):
               # 여기에 코드를 작성한다

    3. 보통: 배치를 불러오는 동안 데이터 불리기를 더하여라

    4. 어려움: 배치 크기를 그때그때 바꾸도록 짜라.
       - 작은 배치에서 비롯한다
       - 익히는 동안 차츰 키운다

    5. 어려움: 배치 크기마다 학습 때를 견주어라
       재려면 time.time()을 써라
    """)

    print("\n" + "="*80)
    print("NEXT: 03_model_checkpointing.py - Save and resume training")
    print("="*80)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (143줄)"

    ```
    ================================================================================
    EFFICIENT BATCH PROCESSING WITH DATALOADER
    ================================================================================

    ================================================================================
    PART 1: WHY USE BATCHES?
    ================================================================================

    작은 배치 경사 하강법:

    데이터를 한꺼번에 다루거나(온 배치) 한 번에 하나씩 다루는(확률) 대신
    작은 배치로 다룬다.

    배치 크기 = 1(SGD):
      ✓ 고치기가 빠르다
      ✗ 기울기에 잡음이 많다
      ✗ 셈이 느리다(벡터로 묶지 못한다)

    배치 크기 = 모두(온 배치 경사 하강법):
      ✓ 기울기가 든든하다
      ✗ 기억 자리를 많이 쓴다
      ✗ 데이터가 크면 느리다

    배치 크기 = 32~256(작은 배치):
      ✓ 속도와 든든함이 고루 좋다
      ✓ GPU을 잘 쓴다
      ✓ 잡음에서 오는 정칙화 효과
      ✓ 기억 자리보다 큰 데이터셋도 다룰 수 있다

    흔한 배치 크기: 16, 32, 64, 128, 256


    ================================================================================
    PART 2: PREPARING DATA
    ================================================================================
    Dataset size: 5000 samples
    Training: 4000 samples
    Test: 1000 samples

    ================================================================================
    PART 3: CREATING DATALOADERS
    ================================================================================

    Dataset created using TensorDataset
    Train dataset length: 4000
    Each item shape: features=torch.Size([20]), label=torch.Size([1])

    DataLoader created:
    Batch size: 64
    Number of batches (train): 63
    Number of batches (test): 16
    Samples per batch: 64
    Last batch size (train): 32

    ================================================================================
    PART 5: TRAINING WITH MINI-BATCHES
    ================================================================================
    Training for 50 epochs with batch_size=64
    ------------------------------------------------------------
    Epoch [10/50] Train Loss: 0.4548 Acc: 0.8067 | Test Loss: 0.4249 Acc: 0.8340
    Epoch [20/50] Train Loss: 0.4162 Acc: 0.8197 | Test Loss: 0.3845 Acc: 0.8440
    Epoch [30/50] Train Loss: 0.4044 Acc: 0.8180 | Test Loss: 0.3705 Acc: 0.8450
    Epoch [40/50] Train Loss: 0.4001 Acc: 0.8190 | Test Loss: 0.3651 Acc: 0.8480
    Epoch [50/50] Train Loss: 0.3987 Acc: 0.8223 | Test Loss: 0.3628 Acc: 0.8510

    Training completed!

    ================================================================================
    PART 6: COMPARING DIFFERENT BATCH SIZES
    ================================================================================
    Training with different batch sizes...
      Batch size  16... Final accuracy: 0.8500
      Batch size  32... Final accuracy: 0.8510
      Batch size  64... Final accuracy: 0.8390
      Batch size 128... Final accuracy: 0.8390
      Batch size 256... Final accuracy: 0.8190

    ================================================================================
    PART 7: VISUALIZATIONS
    ================================================================================
    Visualizations created!

    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. DataLoader의 좋은 점
       ✓ 절로 묶어 준다
       ✓ 기억 자리를 잘 쓴다
       ✓ 붙박이 섞기
       ✓ 나란히 데이터 불러오기(num_workers)
       ✓ 기억 자리보다 큰 데이터셋도 다룬다

    2. 배치 크기 고르기
       ✓ 작은 배치: 잡음이 많고 정칙화가 잘 된다
       ✓ 큰 배치: 더 든든하고 학습이 빠르다
       ✓ 흔히 고르는 값: 32, 64, 128
       ✓ GPU 기억 자리에 매인다

    3. 좋은 버릇
       ✓ 학습 데이터는 늘 섞어라
       ✓ 시험/검증 데이터는 섞지 마라
       ✓ 단순한 자리에는 TensorDataset을 써라
       ✓ 복잡한 데이터에는 맞춤 Dataset을 만들어라
       ✓ 윈도에서는 num_workers=0으로 두어라

    4. 언제 배치를 쓸까
       ✓ 표본이 10,000개를 넘는 데이터셋
       ✓ 기억 자리가 넉넉하지 않을 때
       ✓ GPU으로 익힐 때
       ✓ 정칙화 효과를 바랄 때


    ================================================================================
    EXERCISES
    ================================================================================

    1. 쉬움: batch_size=1으로 해 보아라. 작은 배치와 견주면 어떠한가?

    2. 보통: 맞춤 Dataset 클래스를 짜라.
       class CustomDataset(Dataset):
           def __init__(self, X, y):
               # 여기에 코드를 작성한다
           
           def __len__(self):
               # 여기에 코드를 작성한다
           
           def __getitem__(self, idx):
               # 여기에 코드를 작성한다

    3. 보통: 배치를 불러오는 동안 데이터 불리기를 더하여라

    4. 어려움: 배치 크기를 그때그때 바꾸도록 짜라.
       - 작은 배치에서 비롯한다
       - 익히는 동안 차츰 키운다

    5. 어려움: 배치 크기마다 학습 때를 견주어라
       재려면 time.time()을 써라


    ================================================================================
    NEXT: 03_model_checkpointing.py - Save and resume training
    ================================================================================
    ```


## 2. 논의

### 분할과 전처리

먼저 데이터를 나누고 학습셋에 적합한 StandardScaler를 다른 분할에 적용한다. `stratify`는 분할의 클래스 비율을 맞추는 기능이며 개별 배치의 비율까지 같게 만들지는 않는다.

### 배치와 손실 집계

이진 분류 모델은 $(B,1)$의 확률을 출력하고 BCE는 같은 모양의 실수형 정답을 받는다. 배치마다 초기화·역전파·업데이트를 수행한다. 전체 손실을 보고할 때는 마지막 배치의 실제 표본 수를 반영해야 하며 배치 평균의 단순 평균과 표본 평균은 다를 수 있다.

### 배치 크기 비교

같은 에포크라도 배치 크기에 따라 업데이트 횟수와 처리 시간이 달라진다. 각 후보의 초기값, 학습률과 평가 방식을 맞추어 비교한다. 검증이라고 부르는 분할을 반복적으로 튜닝에 사용하면 독립적인 최종 테스트셋이 따로 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
입력 특징 수가 $d$인 단일 선형 이진 분류기의 파라미터 수와 출력 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"
    가중치 $d$개와 편향 1개로 총 $d+1$개다. 배치 입력 $(B,d)$에 대한 출력은 $(B,1)$이다. 시그모이드에는 학습 파라미터가 없다. 유방암 데이터의 특징 30개를 쓰는 경우에는 31개다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 옵티마이저와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

</div>

??? success "연습문제 2 풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 경사와 경사 제곱의 이동 평균으로 성분별 업데이트를 조정한다. 어느 옵티마이저가 더 빨리 수렴하는지는 문제와 학습률에 달려 있다. 같은 초기 가중치와 데이터 순서, 평가 기준을 사용하고 각 학습률을 조정해 비교한다. 여러 시드의 변동과 검증 성능도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
마지막 배치가 짧을 때 에포크 평균 BCE를 어떻게 계산하는가?

</div>

??? success "연습문제 3 풀이"
    평균 배치 손실에 실제 배치 표본 수를 곱해 더한 뒤 전체 처리 표본 수로 나눈다. 배치 평균들을 같은 가중치로 더하면 짧은 마지막 배치가 과대 반영될 수 있다.

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
    적용하려면 생성자와 `forward`를 함께 정의하고 마지막 출력 모양과 손실의 입력 조건을 유지해야 한다. 선형 계층만 여러 개 합성하면 여전히 하나의 선형 사상이므로 표현력을 늘리려면 적절한 비선형성이 필요하다. Dataset은 표본 접근 인터페이스이므로 계층을 등록할 대상이 아니며 별도 모델 클래스에 이 구성을 적용한다.

## 정리하며

모델 출력과 손실의 입력 조건을 일치시키고 데이터 분할, 전처리와 지표의 의미를 함께 확인한다. 기록된 실행 결과는 해당 설정의 예시이며 모든 환경의 수렴이나 일반화를 보장하지 않는다.
