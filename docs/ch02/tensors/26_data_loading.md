# 데이터 적재 - DataLoader로 만드는 효율적인 데이터 파이프라인

이 스크립트는 DataLoader로 효율적인 데이터 파이프라인을 만드는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""튜토리얼 26: 데이터 불러오기 - DataLoader으로 효율적인 데이터 흐름 만들기"""
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

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
    main()```

## 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사 초기화, 역전파, 매개변수 갱신이다. 각 구성 요소가 결정적인 역할을 한다. 최적화기는 갱신 규칙(SGD, Adam 등)을 캡슐화하고 학습률과 모멘텀 상태를 내부에서 관리한다.

PyTorch의 `DataLoader`는 `Dataset`을 감싸 배치 구성, 섞기, 병렬 데이터 적재를 제공한다. `num_workers`, `pin_memory`, `batch_size`를 적절히 설정하면 GPU가 데이터를 기다리는 일이 없도록 하여 학습 처리량을 크게 개선할 수 있다.

## 연습문제

**연습문제 1.**
SGD 대신 Adam 최적화기를 쓰도록 코드를 수정하라. 100 에폭에 걸친 수렴 속도를 비교하라.

??? success "연습문제 1 풀이"
    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Adam은 적응적 학습률과 모멘텀 덕분에 보통 SGD보다
    # 빠르게 수렴한다. 다만 Adam의 최적 학습률은
    # 보통 SGD보다 작다.
    ```

---


**연습문제 2.**
학습 루프에서 `optimizer.zero_grad()`를 없애면 어떤 일이 생기는가? 실험해 보고 학습 손실에 미치는 영향을 설명하라.

??? success "연습문제 2 풀이"
    `optimizer.zero_grad()`가 없으면 경사가 반복에 걸쳐 누적된다. 실효 경사가 매 단계 커져서 매개변수 갱신이 점점 커진다. 학습이 불안정해지고 손실은 대개 발산한다. PyTorch가 경사 누적 패턴을 지원하기 위해 기본적으로 경사를 누적하기 때문이다.

---


**연습문제 3.**
최적화기에 L2 정칙화(가중치 감쇠)를 추가하고 그것이 최종 매개변수 값에 어떤 영향을 주는지 관찰하라.

??? success "연습문제 3 풀이"
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    # weight_decay는 손실에 L2 벌점항 lambda * ||w||^2을 더한다.
    # 이는 가중치를 작게 유도하여 과적합을 막을 수 있다.
    # 최종 가중치의 크기가 조금 더 작아진다.
    ```
