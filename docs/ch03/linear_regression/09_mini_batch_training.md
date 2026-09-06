# 미니배치 학습

가용 메모리를 넘어서는 큰 데이터셋에서는 전체 데이터셋을 한 번의 순전파로 처리하는 것이 비현실적이다. 미니배치 경사 하강법은 데이터를 작은 배치로 나누어 차례로 처리하며 배치마다 매개변수를 갱신한다. PyTorch의 `DataLoader` 클래스가 배치 묶기, 뒤섞기, 병렬 적재를 자동으로 처리한다. 이 튜토리얼은 여러 배치 크기를 비교하고 DataLoader 기반의 표준 학습 루프를 보여준다.

## 코드

```python
"""
==============================================================================
09_mini_batch_training.py
==============================================================================
DIFFICULTY: ⭐⭐⭐⭐ (Advanced)

DESCRIPTION:
    Mini-batch gradient descent using PyTorch DataLoader.
    Efficient training with batches instead of full dataset.

TOPICS COVERED:
    - Dataset and DataLoader classes
    - Mini-batch gradient descent
    - Batch size effects
    - Shuffling and sampling
    - Training efficiency

PREREQUISITES:
    - Tutorial 05 (nn.Module)

TIME: ~25 minutes
==============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time

print("=" * 70)
print("MINI-BATCH TRAINING WITH DATALOADER")
print("=" * 70)

# ============================================================================
# 1부: 큰 데이터셋 생성
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE LARGE DATASET")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)

# 큰 데이터셋을 생성한다
n_samples = 10000
n_features = 50

X = torch.randn(n_samples, n_features)
true_weights = torch.randn(n_features, 1)
y = X @ true_weights + torch.randn(n_samples, 1) * 0.5

print(f"Dataset created:")
print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# ============================================================================
# 2부: 사용자 정의 Dataset 클래스 만들기
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: CUSTOM DATASET CLASS")
print("=" * 70)

class RegressionDataset(Dataset):
    """회귀 데이터를 위한 사용자 정의 Dataset 클래스"""
    
    def __init__(self, X, y):
        """
        Args:
            X: Feature tensor
            y: Target tensor
        """
        self.X = X
        self.y = y
        
    def __len__(self):
        """표본 개수를 반환한다"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Return a single sample
        
        Args:
            idx: Sample index
        
        Returns:
            tuple: (features, target)
        """
        return self.X[idx], self.y[idx]

# 데이터셋 생성
dataset = RegressionDataset(X, y)
print(f"Dataset created with {len(dataset)} samples")

# 시험 데이터셋
sample_x, sample_y = dataset[0]
print(f"\nFirst sample:")
print(f"  X shape: {sample_x.shape}")
print(f"  y value: {sample_y.item():.4f}")

# 대안: TensorDataset 사용 (기본적인 경우에는 더 간단하다)
tensor_dataset = TensorDataset(X, y)
print(f"\nTensorDataset created (alternative approach)")

# ============================================================================
# 3부: DATALOADER 만들기
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: CREATE DATALOADER")
print("=" * 70)

batch_size = 128  # Number of samples per batch

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,      # Shuffle data each epoch
    num_workers=0,     # Number of subprocesses for data loading
    drop_last=False    # Keep last incomplete batch
)

print(f"DataLoader created:")
print(f"  Batch size: {batch_size}")
print(f"  Number of batches: {len(dataloader)}")
print(f"  Shuffle: True")

# 배치 묶기를 보여준다
print(f"\nIterating through first batch:")
for batch_X, batch_y in dataloader:
    print(f"  Batch X shape: {batch_X.shape}")
    print(f"  Batch y shape: {batch_y.shape}")
    break

# ============================================================================
# 4부: 학습 방법 비교
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: COMPARE BATCH SIZES")
print("=" * 70)

class SimpleModel(nn.Module):
    def __init__(self, n_features):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return self.linear(x)

def train_with_batch_size(batch_size, n_epochs=20):
    """지정한 배치 크기로 모델을 학습시킨다"""
    # DataLoader를 만든다
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 모델과 최적화기
    model = SimpleModel(n_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    losses = []
    epoch_times = []
    
    for epoch in range(n_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_X, batch_y in loader:
            # 순전파
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        epoch_times.append(time.time() - start_time)
    
    return losses, epoch_times, model

# 여러 배치 크기로 학습한다
batch_sizes = [32, 128, 512, len(dataset)]  # Last one is full batch
results = {}

print(f"Training with different batch sizes...")
for bs in batch_sizes:
    print(f"\n  Batch size: {bs}")
    losses, times, model = train_with_batch_size(bs)
    results[bs] = {
        'losses': losses,
        'times': times,
        'final_loss': losses[-1],
        'avg_time': np.mean(times)
    }
    print(f"    Final loss: {losses[-1]:.6f}")
    print(f"    Avg time/epoch: {np.mean(times):.4f}s")

# ============================================================================
# 5부: 결과 시각화
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: VISUALIZE BATCH SIZE EFFECTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 그림 1: 손실 곡선
ax = axes[0, 0]
for bs in batch_sizes:
    label = f'Batch size: {bs}' if bs != len(dataset) else 'Full batch (GD)'
    ax.plot(results[bs]['losses'], label=label, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss vs Batch Size')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# 그림 2: 학습 시간
ax = axes[0, 1]
batch_labels = [f'{bs}' if bs != len(dataset) else 'Full' for bs in batch_sizes]
times = [results[bs]['avg_time'] for bs in batch_sizes]
ax.bar(batch_labels, times)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Avg Time per Epoch (s)')
ax.set_title('Training Speed vs Batch Size')
ax.grid(True, alpha=0.3, axis='y')

# 그림 3: 최종 손실 비교
ax = axes[1, 0]
final_losses = [results[bs]['final_loss'] for bs in batch_sizes]
ax.bar(batch_labels, final_losses)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Final Loss')
ax.set_title('Final Loss vs Batch Size')
ax.grid(True, alpha=0.3, axis='y')

# 그림 4: 요약 문구
ax = axes[1, 1]
summary_text = """
BATCH SIZE EFFECTS:

Small Batches (32, 128):
✓ Faster iteration
✓ More gradient updates
✓ Noisier gradients
✓ Better generalization
✗ Less computational efficiency

Medium Batches (512):
✓ Good balance
✓ Stable training
✓ Efficient

Full Batch (GD):
✓ Most stable gradients
✓ Deterministic
✗ Slow updates
✗ Memory intensive
✗ May overfit

Recommendation:
- Start with 32-256
- Increase if memory allows
- Monitor validation loss
"""
ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/09_batch_size_comparison.png', dpi=100)
print("Saved visualization")
plt.show()

# ============================================================================
# 6부: 효율적인 학습 루프 본보기
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: EFFICIENT TRAINING LOOP TEMPLATE")
print("=" * 70)

print("""
STANDARD PYTORCH TRAINING LOOP WITH DATALOADER:

# 준비
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
model = MyModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# 학습 루프
for epoch in range(n_epochs):
    model.train()  # Set to training mode
    
    for batch_X, batch_y in train_loader:
        # 순전파
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 평가 (선택)
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)

Key Points:
1. DataLoader handles batching and shuffling
2. Each epoch iterates through all batches
3. One epoch = one pass through entire dataset
4. Gradients computed per batch, not entire dataset
""")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Mini-Batch Training:

1. DATASET CLASS:
   - Holds data (X, y)
   - Implements __len__ and __getitem__
   - Can add data augmentation, transforms

2. DATALOADER:
   - Batches data automatically
   - Shuffles each epoch
   - Parallel data loading (num_workers)
   - Handles last incomplete batch

3. BATCH SIZE TRADE-OFFS:
   - Smaller: Noisy but fast
   - Larger: Stable but slow
   - Sweet spot: 32-512
   - Powers of 2 recommended

4. BENEFITS:
   ✓ Memory efficient (don't load all data)
   ✓ Faster convergence (more updates)
   ✓ Better generalization
   ✓ Enables GPU parallelization

Next: Tutorial 10 - Complete production pipeline!
""")


if __name__ == "__main__":
    pass
```

## 논의

`Dataset` 클래스는 개별 표본에 어떻게 접근하는지를 정의하고, `DataLoader`는 데이터셋을 감싸서 배치 단위로 뒤섞인 반복을 제공한다. 사용자 정의 `Dataset` 하위 클래스는 `__len__`(전체 표본 수)과 `__getitem__`(인덱스로 표본 하나 반환)을 구현해야 한다. 단순한 텐서 데이터라면 `TensorDataset`이 이 인터페이스를 자동으로 제공한다. 그러면 `DataLoader`가 지정한 크기의 배치를 만들고, 매 에폭마다 순서를 뒤섞고, 선택적으로 여러 작업자로 데이터를 병렬 적재한다.

배치 크기는 경사의 품질과 갱신 빈도 사이의 절충을 만든다. 전체 배치 경사 하강법은 모든 데이터에 대해 정확한 경사를 계산하지만 에폭당 한 번만 매개변수를 갱신한다. 확률적 경사 하강법(배치 크기 1)은 표본마다 갱신하지만 경사에 잡음이 많다. 32에서 512 사이의 미니배치 크기가 실용적인 중간 지점이다. 경사 추정이 안정적으로 진행할 만큼 믿을 만하면서도 에폭당 여러 번 갱신하여 수렴을 앞당긴다. 2의 거듭제곱을 쓰는 것이 관례인데, GPU 메모리 구조와 잘 맞기 때문이다.

DataLoader를 쓰는 안쪽 학습 루프는 전체 데이터셋이 아니라 배치들을 순회한다. 각 반복은 첫 차원이 배치 크기와 같은(마지막 배치는 다를 수 있다) 텐서 튜플 `(batch_X, batch_y)`를 내놓는다. 손실은 배치마다 계산되고 `optimizer.step()`이 배치마다 매개변수를 갱신한다. 모든 배치를 한 번씩 다 도는 것이 한 에폭이다. 이 패턴은 모든 PyTorch 학습 코드에 공통이다.

## 연습문제

**Exercise 1.**
두 텐서에서 데이터를 읽어 `(features, target)` 튜플을 반환하는 사용자 정의 `Dataset` 클래스를 구현하라. 한 에폭을 순회하며 배치 모양을 출력하여 `DataLoader`와 잘 동작하는지 확인하라.

??? success "Solution to Exercise 1"
    ```python
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    class SimpleDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    X = torch.randn(100, 5)
    y = torch.randn(100, 1)
    ds = SimpleDataset(X, y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    
    for batch_X, batch_y in loader:
        print(f'Batch X: {batch_X.shape}, Batch y: {batch_y.shape}')
    ```

---

**Exercise 2.**
학습 데이터는 매 에폭마다 뒤섞어야 하지만 시험 데이터는 그럴 필요가 없는 이유를 설명하라.

??? success "Solution to Exercise 2"
    학습 데이터를 매 에폭마다 뒤섞으면 (예를 들어 표본이 클래스 이름순으로 정렬되어 있을 때) 모델이 표본의 순서에서 허위 패턴을 배우지 않게 된다. 또한 최적화기가 순환에 갇히는 것을 막는 데도 도움이 된다. 반면 시험 데이터는 매개변수를 갱신하는 데가 아니라 평가에만 쓰이므로 뒤섞을 필요가 없고, 매번 예측 순서가 바뀌어 디버깅이 어려워질 수도 있다.

---

**Exercise 3.**
같은 모델 구조를 배치 크기 1, 32, 128, 그리고 전체 데이터셋(10000)으로 학습시켜라. 손실 곡선을 그리고 각각의 에폭당 실제 소요 시간을 측정하라. 관찰한 절충을 논하라.

??? success "Solution to Exercise 3"
    ```python
    import torch, time
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    
    torch.manual_seed(42)
    X = torch.randn(10000, 50)
    w_true = torch.randn(50, 1)
    y = X @ w_true + torch.randn(10000, 1) * 0.5
    ds = TensorDataset(X, y)
    
    for bs in [1, 32, 128, len(X)]:
        model = nn.Linear(50, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loader = DataLoader(ds, batch_size=bs, shuffle=True)
        start = time.time()
        for epoch in range(5):
            for bx, by in loader:
                loss = nn.MSELoss()(model(bx), by)
                opt.zero_grad(); loss.backward(); opt.step()
        elapsed = time.time() - start
        print(f'BS={bs:5d}: time={elapsed:.2f}s, final_loss={loss.item():.6f}')
    # BS=1은 에폭당 가장 느리고, 전체 배치는 에폭당 가장 빠르지만 매개변수 갱신
    # 횟수가 적어 수렴이 더디다. BS=32-128이 실용적으로 가장 좋은 지점이다.
    ```
