# DataLoader로 배치 처리하기

분류 데이터를 TensorDataset과 DataLoader로 묶어 미니배치 트레인과 평가를 구성한다 — 미니배치 트레인은 데이터를 작은 묶음(배치)으로 잘라 묶음 하나마다 파라메터를 한 스텝씩 업데이트하는 것이다. 배치별 그래디언트와 데이터 전체의 지표를 구분한다.

## 1. 코드

??? note "코드 (450줄)"

    ```python
    """
    ================================================================================
    DataLoader로 하는 효율적인 배치 처리
    ================================================================================

    학습 목표:
    - 미니배치 경사 하강법을 이해한다
    - 효율적인 배치 만들기에 PyTorch DataLoader를 쓴다
    - Dataset과 DataLoader 클래스를 배운다
    - 셔플과 배치 처리를 익힌다
    - 더 큰 데이터셋으로 넓힌다

    PREREQUISITES:
    - 「BCE를 쓰는 경사 하강법」 쪽의 트레인 루프를 읽었을 것
    - 경사 하강법을 이해하고 있을 것

    소요 시간: 1시간 반쯤

    난이도: ⭐⭐⭐☆☆ (중급)
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
    미니배치 경사 하강법:

    데이터를 한꺼번에 다루거나(풀 배치) 한 번에 하나씩 다루는(확률적) 대신
    미니배치로 다룬다.

    배치 크기 = 1(SGD):
      ✓ 업데이트가 빠르다
      ✗ 그래디언트에 잡음이 많다
      ✗ 계산이 느리다(벡터화하지 못한다)

    배치 크기 = 전체(풀 배치 경사 하강법):
      ✓ 그래디언트가 안정적이다
      ✗ 메모리를 많이 쓴다
      ✗ 데이터가 크면 느리다

    배치 크기 = 32~256(미니배치):
      ✓ 속도와 안정성의 균형이 좋다
      ✓ GPU를 잘 쓴다
      ✓ 잡음에서 오는 정칙화 효과
      ✓ 메모리보다 큰 데이터셋도 다룰 수 있다

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
    # 5부: 배치를 쓰는 트레인
    # =============================================================================

    print("\n" + "="*80)
    print("PART 5: TRAINING WITH MINI-BATCHES")
    print("="*80)

    def train_epoch_with_batches(model, dataloader, criterion, optimizer):
        """배치를 써서 한 에포크를 트레인한다"""
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        # 배치들을 순회한다
        for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
            # batch_X 형태: (batch_size, n_features)
            # batch_y 형태: (batch_size, 1)

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


    # 트레인 준비
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
        # 트레인
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
        """특정 배치 크기로 모델을 트레인시킨다"""
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

    # 그림 1: 트레인 곡선
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

    # 그림 3: 배치 크기 비교 (로스 곡선)
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

    # 막대에 값 라벨을 추가한다
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
       ✓ 자동으로 배치를 만든다
       ✓ 메모리를 잘 쓴다
       ✓ 내장 셔플
       ✓ 병렬 데이터 로딩(num_workers)
       ✓ 메모리보다 큰 데이터셋도 다룬다

    2. 배치 크기 고르기
       ✓ 작은 배치: 잡음이 많고 정칙화가 잘 된다
       ✓ 큰 배치: 더 안정적이고 트레인이 빠르다
       ✓ 흔히 고르는 값: 32, 64, 128
       ✓ GPU 메모리에 제약된다

    3. 모범 사례
       ✓ 트레인 데이터는 늘 섞어라
       ✓ 시험/검증 데이터는 섞지 마라
       ✓ 단순한 경우에는 TensorDataset을 써라
       ✓ 복잡한 데이터에는 사용자 정의 Dataset을 만들어라
       ✓ 윈도에서는 num_workers=0으로 두어라

    4. 언제 배치를 쓸까
       ✓ 표본이 10,000개를 넘는 데이터셋
       ✓ 메모리가 넉넉하지 않을 때
       ✓ GPU로 트레인할 때
       ✓ 정칙화 효과를 바랄 때
    """)

    print("\n" + "="*80)
    print("EXERCISES")
    print("="*80)
    print("""
    1. 쉬움: batch_size=1로 해 보아라. 미니배치와 비교하면 어떠한가?

    2. 보통: 사용자 정의 Dataset 클래스를 구현하라.
       class CustomDataset(Dataset):
           def __init__(self, X, y):
               # 여기에 코드를 작성한다

           def __len__(self):
               # 여기에 코드를 작성한다

           def __getitem__(self, idx):
               # 여기에 코드를 작성한다

    3. 보통: 배치를 불러오는 동안 데이터 어그멘테이션을 더하여라

    4. 어려움: 배치 크기를 동적으로 바꾸도록 구현하라.
       - 작은 배치에서 시작한다
       - 트레인하는 동안 점차 키운다

    5. 어려움: 배치 크기마다 트레인 시간을 비교하라
       재려면 time.time()을 써라
    """)

    print("\n" + "="*80)
    print("다음: 「모델 체크포인트」 쪽 - 트레인을 저장하고 이어 가기")
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

    미니배치 경사 하강법:

    데이터를 한꺼번에 다루거나(풀 배치) 한 번에 하나씩 다루는(확률적) 대신
    미니배치로 다룬다.

    배치 크기 = 1(SGD):
      ✓ 업데이트가 빠르다
      ✗ 그래디언트에 잡음이 많다
      ✗ 계산이 느리다(벡터화하지 못한다)

    배치 크기 = 전체(풀 배치 경사 하강법):
      ✓ 그래디언트가 안정적이다
      ✗ 메모리를 많이 쓴다
      ✗ 데이터가 크면 느리다

    배치 크기 = 32~256(미니배치):
      ✓ 속도와 안정성의 균형이 좋다
      ✓ GPU를 잘 쓴다
      ✓ 잡음에서 오는 정칙화 효과
      ✓ 메모리보다 큰 데이터셋도 다룰 수 있다

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
       ✓ 자동으로 배치를 만든다
       ✓ 메모리를 잘 쓴다
       ✓ 내장 셔플
       ✓ 병렬 데이터 로딩(num_workers)
       ✓ 메모리보다 큰 데이터셋도 다룬다

    2. 배치 크기 고르기
       ✓ 작은 배치: 잡음이 많고 정칙화가 잘 된다
       ✓ 큰 배치: 더 안정적이고 트레인이 빠르다
       ✓ 흔히 고르는 값: 32, 64, 128
       ✓ GPU 메모리에 제약된다

    3. 모범 사례
       ✓ 트레인 데이터는 늘 섞어라
       ✓ 시험/검증 데이터는 섞지 마라
       ✓ 단순한 경우에는 TensorDataset을 써라
       ✓ 복잡한 데이터에는 사용자 정의 Dataset을 만들어라
       ✓ 윈도에서는 num_workers=0으로 두어라

    4. 언제 배치를 쓸까
       ✓ 표본이 10,000개를 넘는 데이터셋
       ✓ 메모리가 넉넉하지 않을 때
       ✓ GPU로 트레인할 때
       ✓ 정칙화 효과를 바랄 때


    ================================================================================
    EXERCISES
    ================================================================================

    1. 쉬움: batch_size=1로 해 보아라. 미니배치와 비교하면 어떠한가?

    2. 보통: 사용자 정의 Dataset 클래스를 구현하라.
       class CustomDataset(Dataset):
           def __init__(self, X, y):
               # 여기에 코드를 작성한다
           
           def __len__(self):
               # 여기에 코드를 작성한다
           
           def __getitem__(self, idx):
               # 여기에 코드를 작성한다

    3. 보통: 배치를 불러오는 동안 데이터 어그멘테이션을 더하여라

    4. 어려움: 배치 크기를 동적으로 바꾸도록 구현하라.
       - 작은 배치에서 시작한다
       - 트레인하는 동안 점차 키운다

    5. 어려움: 배치 크기마다 트레인 시간을 비교하라
       재려면 time.time()을 써라


    ================================================================================
    다음: 「모델 체크포인트」 쪽 - 트레인을 저장하고 이어 가기
    ================================================================================
    ```


## 2. 논의

### 분할과 전처리

먼저 데이터를 나누고, 트레인 집합에만 맞춘(fit) StandardScaler를 다른 분할에 적용한다. 시험 집합의 평균과 표준편차가 전처리에 새어 들지 않게 하려는 것이다. `stratify`는 분할의 클래스 비율을 맞추는 기능이며 개별 배치의 비율까지 같게 만들지는 않는다.

### 배치와 로스 집계

이진 분류 모델은 $(B,1)$ 형태의 확률을 출력하고($B$는 배치 크기다) BCE(이진 교차 엔트로피)는 같은 형태의 실수형 정답을 받는다. 배치마다 그래디언트 초기화·역전파·업데이트를 수행한다. 전체 로스를 보고할 때는 마지막 배치의 실제 표본 수를 반영해야 한다. 이 쪽의 트레인 집합 4000개는 64개씩 62배치와 32개짜리 마지막 배치로 나뉘므로, 배치 평균의 단순 평균과 표본 전체의 평균은 다를 수 있다.

### 배치 크기 비교

같은 에포크라도 배치 크기에 따라 업데이트 횟수와 처리 시간이 달라진다. 각 후보의 초기값, 러닝레이트와 평가 방식을 맞추어 비교한다. 이 쪽의 6부는 후보마다 시드를 다시 고정하지 않고 한 번씩만 돌렸으므로 출력의 0.8190~0.8510 차이를 배치 크기의 효과로 단정할 수 없다. 시드에 따른 흔들림이 얼마나 되는지는 출력에서 바로 보인다 — 배치 크기 64로 30에포크를 돈 시험 정확도가 5부에서는 0.8450, 6부에서는 0.8390이다. 설정은 같고 초기값과 섞는 순서만 다르다. 같은 30에포크라도 배치 크기 256은 에포크당 업데이트가 16번이고 16은 250번이라는 점도 함께 본다. 검증이라고 부르는 분할을 반복적으로 튜닝에 사용하면 독립적인 최종 시험 집합이 따로 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
입력 피처 수가 $d$인 단일 선형 이진 분류기의 파라메터 수와 출력 형태를 설명하라.

</div>

??? success "풀이"
    웨이트 $d$개와 바이어스 1개로 총 $d+1$개다. 배치 입력 $(B,d)$에 대한 출력은 $(B,1)$이다. 시그모이드에는 트레인 가능한 파라메터가 없다. 이 쪽의 `make_classification` 데이터는 피처가 20개이므로 21개다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 쪽은 `torch.optim.Adam`에 `lr=0.001`을 쓴다. 옵티마이저를 `torch.optim.SGD`(`lr=0.1`)로 바꾸고 원래 옵티마이저와 트레인 수렴을 비교하라. 두 로스 곡선을 같은 그래프에 그려라.

</div>

??? success "풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.SGD(model.parameters(), lr=0.1)`로 바꾼다. Adam은 그래디언트와 그래디언트 제곱의 이동 평균으로 성분별 업데이트를 조정한다. 어느 옵티마이저가 더 빨리 수렴하는지는 문제와 러닝레이트에 달려 있다. 같은 초기 웨이트와 데이터 순서, 평가 기준을 사용하고 각 러닝레이트를 조정해 비교한다. 여러 시드의 변동과 검증 성능도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
마지막 배치가 짧을 때 에포크 평균 BCE를 어떻게 계산하는가?

</div>

??? success "풀이"
    배치마다 평균 로스에 그 배치의 실제 표본 수를 곱해 더한 뒤 전체 처리 표본 수로 나눈다. 배치 평균들을 같은 가중치로 더하면 짧은 마지막 배치가 과대 반영될 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
6부의 `train_with_batch_size`는 배치 크기 후보를 공정하게 견주지 못한다. 후보마다 초기 웨이트와 섞는 순서가 다르고, 같은 30에포크라도 업데이트 횟수가 다르기 때문이다. (가) 후보마다 같은 초기 웨이트와 같은 난수 흐름에서 출발하도록 함수에 `seed` 인자를 더하라. (나) 배치 크기 16, 32, 64, 128, 256에서 에포크당 업데이트 횟수를 구하고, 배치 크기 16의 30에포크와 업데이트 횟수를 맞추려면 나머지 후보는 몇 에포크를 돌아야 하는지 계산하라.

</div>

??? success "풀이"
    (가) 모델을 만들기 직전에 `torch.manual_seed(seed)`를 부르면 `nn.Linear`의 초기 웨이트가 후보마다 같아진다. 섞는 순서는 `DataLoader`에 `generator`를 따로 넘겨 고정한다. 그래도 배치 크기가 다르면 한 스텝에 들어가는 표본이 다르므로 트레인 경로까지 같아지지는 않는다 — 같아지는 것은 출발점과 표본을 훑는 순서다. 시드 하나로 견준 차이는 여전히 한 번 잰 값이므로, `seed`를 여러 값으로 바꿔 돌려 그 흩어짐보다 큰 차이만 배치 크기의 효과로 읽는다.

    ```python
    def train_with_batch_size(batch_size, num_epochs=30, seed=0):
        """특정 배치 크기로 모델을 트레인시킨다 (출발점을 seed로 고정)"""
        torch.manual_seed(seed)                       # 초기 웨이트를 고정한다
        model = LogisticRegression(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed)   # 섞는 순서를 고정한다
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
    ```

    (나) `drop_last=False`이므로 에포크당 업데이트 횟수는 `len(train_loader)` $=\lceil 4000/B\rceil$이다. 기준은 배치 크기 16의 $250\times 30=7500$번이다.

    ```python
    import math

    target_steps = math.ceil(len(train_dataset) / 16) * 30      # 7500
    for bs in batch_sizes:
        steps = len(DataLoader(train_dataset, batch_size=bs))   # 에포크당 업데이트 횟수
        print(bs, steps, steps * 30, math.ceil(target_steps / steps))
    ```

    | 배치 크기 | 에포크당 업데이트 | 30에포크의 업데이트 | 7500번에 필요한 에포크 |
    |---|---|---|---|
    | 16 | 250 | 7500 | 30 |
    | 32 | 125 | 3750 | 60 |
    | 64 | 63 | 1890 | 120 |
    | 128 | 32 | 960 | 235 |
    | 256 | 16 | 480 | 469 |

    6부에서 배치 크기 256은 16이 받은 7500번의 16분의 1쯤인 480번만 받았다. 출력에서 256의 정확도가 가장 낮은 것이 배치 크기 탓인지 업데이트가 모자란 탓인지는 6부의 실험만으로는 가를 수 없다.

## 정리하며

모델 출력과 로스의 입력 조건을 일치시키고 데이터 분할, 전처리와 지표의 의미를 함께 확인한다. 기록된 실행 결과는 해당 설정의 예시이며 모든 환경의 수렴이나 일반화를 보장하지 않는다.
