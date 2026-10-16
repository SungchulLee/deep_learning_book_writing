# 데이터 적재 - DataLoader로 만드는 효율적인 데이터 파이프라인

학습 루프와 데이터 읽기를 분리하면 배치 구성과 전처리를 관리하기 쉽다. `Dataset`은 표본을 제공하고 `DataLoader`는 표본들을 모아 배치로 전달한다.

## 1. 코드

```python
"""튜토리얼 26: 데이터 불러오기 - DataLoader으로 효율적인 데이터 흐름 만들기"""
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

class CustomDataset(Dataset):
    """맞춤 데이터셋 보기."""
    def __init__(self, size=100):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def main():
    header("1. Basic Dataset and DataLoader")
    dataset = CustomDataset(size=100)
    print(f"Dataset size: {len(dataset)}")
    print(f"First sample: {dataset[0]}")
    
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(f"\nDataLoader created with batch_size=10")
    print(f"Number of batches: {len(dataloader)}")
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: data shape={data.shape}, labels shape={labels.shape}")
        if batch_idx == 2:
            break
    
    header("2. DataLoader Parameters")
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,      # Shuffle data each epoch
        num_workers=0,     # Number of parallel workers (0 = main process)
        drop_last=False,   # Drop incomplete last batch?
        pin_memory=False   # Pin memory for faster GPU transfer
    )
    print("Key DataLoader parameters:")
    print(f"  batch_size: {dataloader.batch_size}")
    print(f"  shuffle: True")
    print(f"  num_workers: {dataloader.num_workers}")
    print(f"  drop_last: {dataloader.drop_last}")
    
    header("3. TensorDataset - Quick Dataset")
    X = torch.randn(100, 5)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=20)
    print("TensorDataset: Simple way to create dataset from tensors")
    print(f"Dataset size: {len(dataset)}")
    for data, labels in dataloader:
        print(f"Batch: {data.shape}, {labels.shape}")
        break
    
    header("4. Training Loop with DataLoader")
    import torch.nn as nn
    import torch.optim as optim
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    dataset = CustomDataset(size=200)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("Training for 2 epochs:")
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (data, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
    
    header("5. Data Augmentation Example")
    class AugmentedDataset(Dataset):
        def __init__(self, size=100):
            self.data = torch.randn(size, 3, 32, 32)  # Images
            self.labels = torch.randint(0, 10, (size,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            img = self.data[idx]
            # 간단한 증강: 무작위 뒤집기
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[2])  # Horizontal flip
            return img, self.labels[idx]
    
    aug_dataset = AugmentedDataset(size=50)
    aug_dataloader = DataLoader(aug_dataset, batch_size=10)
    print("Dataset with random horizontal flip augmentation")
    for img, label in aug_dataloader:
        print(f"Batch: images={img.shape}, labels={label.shape}")
        break
    
    header("6. Splitting Dataset")
    from torch.utils.data import random_split
    
    dataset = CustomDataset(size=100)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Total dataset size: {len(dataset)}")
    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print("\nTrain and validation loaders created!")
    
    header("7. Collate Function - Custom Batching")
    def custom_collate(batch):
        """표본을 묶는 맞춤 함수."""
        data = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        # 여기에 사용자 정의 처리를 추가한다
        return data, labels
    
    dataloader = DataLoader(dataset, batch_size=10, collate_fn=custom_collate)
    print("Using custom collate function")
    for data, labels in dataloader:
        print(f"Batch: {data.shape}, {labels.shape}")
        break
    
    header("8. Best Practices")
    print("""
    DataLoader를 잘 쓰는 버릇:
    
    1. 더 빨리 불러오려면 일꾼을 여럿 써라(num_workers > 0)
    2. GPU을 쓸 때는 pin_memory=True를 켜라
    3. 학습 데이터를 섞어라(shuffle=True)
    4. 검증/시험 데이터는 섞지 마라
    5. 알맞은 배치 크기를 써라(2의 거듭제곱이 잘 듣는 일이 잦다)
    6. persistent_workers=True로 데이터를 미리 가져와라
    7. 배치 크기가 종요로우면 drop_last=True를 써라
    8. 맞춤 데이터셋에서는 __getitem__을 잘 들게 짜라
    9. 될 수 있으면 미리 다듬은 데이터를 저장해 두어라
    10. 데이터 불러오는 때와 익히는 때를 견주어 살펴라
    """)

if __name__ == "__main__":
    main()
```

**출력:**

```

======================================================================
1. Basic Dataset and DataLoader
======================================================================
Dataset size: 100
First sample: (tensor([-1.1258, -1.1524, -0.2506, -0.4339,  0.8487,  0.6920, -0.3160, -2.1152,
         0.3223, -1.2633]), tensor(1))

DataLoader created with batch_size=10
Number of batches: 10
Batch 0: data shape=torch.Size([10, 10]), labels shape=torch.Size([10])
Batch 1: data shape=torch.Size([10, 10]), labels shape=torch.Size([10])
Batch 2: data shape=torch.Size([10, 10]), labels shape=torch.Size([10])

======================================================================
2. DataLoader Parameters
======================================================================
Key DataLoader parameters:
  batch_size: 32
  shuffle: True
  num_workers: 0
  drop_last: False

======================================================================
3. TensorDataset - Quick Dataset
======================================================================
TensorDataset: Simple way to create dataset from tensors
Dataset size: 100
Batch: torch.Size([20, 5]), torch.Size([20])

======================================================================
4. Training Loop with DataLoader
======================================================================
Training for 2 epochs:
Epoch 1: Avg Loss = 0.7115
Epoch 2: Avg Loss = 0.7190

======================================================================
5. Data Augmentation Example
======================================================================
Dataset with random horizontal flip augmentation
Batch: images=torch.Size([10, 3, 32, 32]), labels=torch.Size([10])

======================================================================
6. Splitting Dataset
======================================================================
Total dataset size: 100
Train set size: 80
Val set size: 20

Train and validation loaders created!

======================================================================
7. Collate Function - Custom Batching
======================================================================
Using custom collate function
Batch: torch.Size([10, 10]), torch.Size([10])

======================================================================
8. Best Practices
======================================================================

    DataLoader를 잘 쓰는 버릇:
    
    1. 더 빨리 불러오려면 일꾼을 여럿 써라(num_workers > 0)
    2. GPU을 쓸 때는 pin_memory=True를 켜라
    3. 학습 데이터를 섞어라(shuffle=True)
    4. 검증/시험 데이터는 섞지 마라
    5. 알맞은 배치 크기를 써라(2의 거듭제곱이 잘 듣는 일이 잦다)
    6. persistent_workers=True로 데이터를 미리 가져와라
    7. 배치 크기가 종요로우면 drop_last=True를 써라
    8. 맞춤 데이터셋에서는 __getitem__을 잘 들게 짜라
    9. 될 수 있으면 미리 다듬은 데이터를 저장해 두어라
    10. 데이터 불러오는 때와 익히는 때를 견주어 살펴라
    
```

## 2. 논의

`Dataset`의 `__len__`은 표본 수, `__getitem__`은 인덱스에 대응하는 표본을 정의한다. `DataLoader`는 배치 크기와 순서를 관리하며, 같은 모양의 텐서는 기본 collate 동작으로 쌓는다. 길이가 다른 표본은 별도 패딩이나 collate 함수가 필요할 수 있다.

`num_workers`는 데이터를 읽는 프로세스 수다. 크게 잡는다고 항상 빨라지지 않으며 메모리·프로세스 생성·저장장치 비용이 함께 늘 수 있다. CUDA 전송에서는 pinned CPU 메모리와 비동기 전송이 도움이 될 수 있지만, 실제 데이터 경로에서 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
SGD 대신 Adam 옵티마이저를 쓰도록 코드를 수정하라. 100 에포크에 걸친 수렴 속도를 비교하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Adam은 적응적 러닝레이트와 모멘텀 덕분에 보통 SGD보다
    # 빠르게 수렴한다. 다만 Adam의 최적 러닝레이트는
    # 보통 SGD보다 작다.
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
학습 루프에서 `optimizer.zero_grad()`를 없애면 어떤 일이 생기는가? 실험해 보고 학습 로스에 미치는 영향을 설명하라.

</div>

??? success "연습문제 2 풀이"
    업데이트 사이에 경사를 초기화하지 않으면 서로 다른 파라미터에서 계산한 이전 경사까지 누적된다. 따라서 현재 로스의 경사만 사용하는 학습 규칙과 달라지고, 진동이나 발산을 일으킬 수 있다. 경사끼리 상쇄될 수도 있으므로 크기가 항상 증가하거나 반드시 발산하는 것은 아니다. 의도적인 경사 누적에서는 정해진 여러 마이크로배치를 처리한 뒤 한 번 업데이트하고 경사를 초기화한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
옵티마이저에 L2 정규화(웨이트 감쇠)를 추가하고 그것이 최종 파라메터 값에 어떤 영향을 주는지 관찰하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    # weight_decay는 로스에 L2 벌점항 lambda * ||w||^2을 더한다.
    # 이는 웨이트를 작게 유도하여 과적합을 막을 수 있다.
    # 최종 웨이트의 크기가 조금 더 작아진다.
    ```

## 정리하며

학습 루프와 데이터 읽기를 분리하면 배치 구성과 전처리를 관리하기 쉽다. `Dataset`은 표본을 제공하고 `DataLoader`는 표본들을 모아 배치로 전달한다.
