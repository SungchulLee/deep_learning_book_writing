# DataLoader로 배치 처리하기

02_dataloader_batching.py - DataLoader를 이용한 효율적인 배치 처리

이 튜토리얼은 PyTorch에서 로지스틱 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
"""
================================================================================
02_dataloader_batching.py - Efficient Batch Processing with DataLoader
================================================================================

LEARNING OBJECTIVES:
- Understand mini-batch gradient descent
- Use PyTorch DataLoader for efficient batching
- Learn about Dataset and DataLoader classes
- Handle shuffling and batch processing
- Scale to larger datasets

PREREQUISITES:
- Completed 01_proper_training_loop.py
- Understanding of gradient descent

TIME TO COMPLETE: ~1.5 hours

DIFFICULTY: ⭐⭐⭐☆☆ (Intermediate)
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
Mini-Batch Gradient Descent:

Instead of processing all data at once (full batch) or one sample at a time
(stochastic), we process small batches:

Batch Size = 1 (SGD):
  ✓ Fast updates
  ✗ Noisy gradients
  ✗ Slow computation (no vectorization)

Batch Size = ALL (Full Batch GD):
  ✓ Stable gradients
  ✗ Memory intensive
  ✗ Slow for large datasets

Batch Size = 32-256 (Mini-Batch):
  ✓ Balance between speed and stability
  ✓ Efficient GPU utilization
  ✓ Regularization effect from noise
  ✓ Can process datasets larger than memory

Common batch sizes: 16, 32, 64, 128, 256
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
        # batch_X shape: (batch_size, n_features)
        # batch_y shape: (batch_size, 1)
        
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

# 막대에 값 이름표를 추가한다
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
1. DATALOADER BENEFITS
   ✓ Automatic batching
   ✓ Efficient memory usage
   ✓ Built-in shuffling
   ✓ Parallel data loading (num_workers)
   ✓ Can handle datasets larger than RAM

2. BATCH SIZE SELECTION
   ✓ Smaller batches: More noise, better regularization
   ✓ Larger batches: More stable, faster training
   ✓ Common choices: 32, 64, 128
   ✓ Limited by GPU memory

3. BEST PRACTICES
   ✓ Always shuffle training data
   ✓ Don't shuffle test/validation data
   ✓ Use TensorDataset for simple cases
   ✓ Create custom Dataset for complex data
   ✓ Set num_workers=0 on Windows

4. WHEN TO USE BATCHES
   ✓ Datasets > 10,000 samples
   ✓ Limited memory
   ✓ GPU training
   ✓ Want regularization effect
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. EASY: Try batch_size=1. How does it compare to mini-batch?

2. MEDIUM: Implement custom Dataset class:
   class CustomDataset(Dataset):
       def __init__(self, X, y):
           # 여기에 코드를 작성한다
       
       def __len__(self):
           # 여기에 코드를 작성한다
       
       def __getitem__(self, idx):
           # 여기에 코드를 작성한다

3. MEDIUM: Add data augmentation during batch loading

4. HARD: Implement dynamic batch sizing:
   - Start with small batches
   - Gradually increase during training

5. HARD: Compare training time for different batch sizes
   Use time.time() to measure
""")

print("\n" + "="*80)
print("NEXT: 03_model_checkpointing.py - Save and resume training")
print("="*80)


if __name__ == "__main__":
    pass
```

## 논의

`LogisticRegression` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `LogisticRegression`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `LogisticRegression`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = LogisticRegression(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
