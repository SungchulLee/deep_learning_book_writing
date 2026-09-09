# 미니배치 학습

가용 메모리를 넘어서는 큰 데이터셋에서는 전체 데이터셋을 한 번의 순전파로 처리하는 것이 비현실적이다. 미니배치 경사 하강법은 데이터를 작은 배치로 나누어 차례로 처리하며 배치마다 매개변수를 갱신한다. PyTorch의 `DataLoader` 클래스가 배치 묶기, 뒤섞기, 병렬 적재를 자동으로 처리한다. 이 튜토리얼은 여러 배치 크기를 비교하고 DataLoader 기반의 표준 학습 루프를 보여준다.

## 1. 코드

```python
"""
==============================================================================
09_mini_batch_training.py
==============================================================================
어려움: ⭐⭐⭐⭐ (앞선)

DESCRIPTION:
    PyTorch DataLoader를 쓰는 작은 배치 경사 하강법.
    온 데이터셋 대신 배치으로 잘 들게 익힌다.

다루는 것:
    - Dataset과 DataLoader 클래스
    - 작은 배치 경사 하강법
    - 배치 크기가 미치는 영향
    - 섞기와 뽑기
    - 학습이 잘 듦

PREREQUISITES:
    - 튜토리얼 05(nn.Module)

걸리는 때: 25분쯤
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
            X: 특징 텐서
            y: 과녁 텐서
        """
        self.X = X
        self.y = y
        
    def __len__(self):
        """표본 개수를 반환한다"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        표본 하나를 돌려준다
        
        Args:
            idx: 표본 번호
        
        Returns:
            튜플: (특징, 과녁)
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
배치 크기가 미치는 영향:

작은 배치(32, 128):
✓ 루프가 빠르다
✓ 기울기 고침이 더 잦다
✓ 기울기에 잡음이 더 많다
✓ 두루 더 잘 미친다
✗ 셈이 덜 효율적이다

가운데 배치(512):
✓ 고루 좋다
✓ 학습이 든든하다
✓ Efficient

온 배치(경사 하강법):
✓ 기울기가 가장 든든하다
✓ Deterministic
✗ 고치기가 더디다
✗ 기억 자리를 많이 쓴다
✗ 지나치게 맞춰질 수 있다

Recommendation:
- 32~256에서 비롯하라
- 기억 자리가 넉넉하면 키워라
- 검증 손실을 지켜보아라
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
DataLoader를 쓰는 여느 PyTorch 학습 루프:

# 준비
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
model = MyModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# 학습 루프
for epoch in range(n_epochs):
    model.train()  # 학습 결로 둔다
    
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

고갱이:
1. DataLoader이 배치 만들기와 섞기를 다룬다
2. 에폭마다 모든 배치을 훑는다
3. 한 에폭 = 온 데이터셋을 한 번 훑기
4. 기울기는 온 데이터가 아니라 배치마다 계산한다
""")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
작은 배치 익히기:

1. Dataset 클래스:
   - 데이터(X, y)를 지닌다
   - __len__과 __getitem__을 짠다
   - 데이터 불리기와 바꾸기를 더할 수 있다

2. DATALOADER:
   - 데이터를 절로 묶는다
   - 에폭마다 섞는다
   - 나란히 데이터 불러오기(num_workers)
   - 마지막의 모자란 배치을 다룬다

3. 배치 크기의 맞바꿈:
   - 작으면 잡음이 많지만 빠르다
   - 크면 든든하지만 느리다
   - 알맞은 자리: 32~512
   - 2의 거듭제곱을 권한다

4. BENEFITS:
   ✓ 기억 자리를 아낀다(데이터를 모두 올리지 않는다)
   ✓ 더 빨리 모여든다(고침이 잦다)
   ✓ 두루 더 잘 미친다
   ✓ GPU의 나란한 셈을 쓸 수 있다

다음: 튜토리얼 10 - 온전한 실전 흐름!
""")


if __name__ == "__main__":
    pass
```

## 2. 논의

`Dataset` 클래스는 개별 표본에 어떻게 접근하는지를 정의하고, `DataLoader`는 데이터셋을 감싸서 배치 단위로 뒤섞인 반복을 제공한다. 사용자 정의 `Dataset` 하위 클래스는 `__len__`(전체 표본 수)과 `__getitem__`(인덱스로 표본 하나 반환)을 구현해야 한다. 단순한 텐서 데이터라면 `TensorDataset`이 이 인터페이스를 자동으로 제공한다. 그러면 `DataLoader`가 지정한 크기의 배치를 만들고, 매 에폭마다 순서를 뒤섞고, 선택적으로 여러 작업자로 데이터를 병렬 적재한다.

배치 크기는 경사의 품질과 갱신 빈도 사이의 절충을 만든다. 전체 배치 경사 하강법은 모든 데이터에 대해 정확한 경사를 계산하지만 에폭당 한 번만 매개변수를 갱신한다. 확률적 경사 하강법(배치 크기 1)은 표본마다 갱신하지만 경사에 잡음이 많다. 32에서 512 사이의 미니배치 크기가 실용적인 중간 지점이다. 경사 추정이 안정적으로 진행할 만큼 믿을 만하면서도 에폭당 여러 번 갱신하여 수렴을 앞당긴다. 2의 거듭제곱을 쓰는 것이 관례인데, GPU 메모리 구조와 잘 맞기 때문이다.

DataLoader를 쓰는 안쪽 학습 루프는 전체 데이터셋이 아니라 배치들을 순회한다. 각 반복은 첫 차원이 배치 크기와 같은(마지막 배치는 다를 수 있다) 텐서 튜플 `(batch_X, batch_y)`를 내놓는다. 손실은 배치마다 계산되고 `optimizer.step()`이 배치마다 매개변수를 갱신한다. 모든 배치를 한 번씩 다 도는 것이 한 에폭이다. 이 패턴은 모든 PyTorch 학습 코드에 공통이다.

## 연습문제

**연습문제 1.**
두 텐서에서 데이터를 읽어 `(features, target)` 튜플을 반환하는 사용자 정의 `Dataset` 클래스를 구현하라. 한 에폭을 순회하며 배치 모양을 출력하여 `DataLoader`와 잘 동작하는지 확인하라.

??? success "연습문제 1 풀이"
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

**연습문제 2.**
학습 데이터는 매 에폭마다 뒤섞어야 하지만 시험 데이터는 그럴 필요가 없는 이유를 설명하라.

??? success "연습문제 2 풀이"
    학습 데이터를 매 에폭마다 뒤섞으면 (예를 들어 표본이 클래스 이름순으로 정렬되어 있을 때) 모델이 표본의 순서에서 허위 패턴을 배우지 않게 된다. 또한 최적화기가 순환에 갇히는 것을 막는 데도 도움이 된다. 반면 시험 데이터는 매개변수를 갱신하는 데가 아니라 평가에만 쓰이므로 뒤섞을 필요가 없고, 매번 예측 순서가 바뀌어 디버깅이 어려워질 수도 있다.

---

**연습문제 3.**
같은 모델 구조를 배치 크기 1, 32, 128, 그리고 전체 데이터셋(10000)으로 학습시켜라. 손실 곡선을 그리고 각각의 에폭당 실제 소요 시간을 측정하라. 관찰한 절충을 논하라.

??? success "연습문제 3 풀이"
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

## 정리하며

**다룬 것** — 미니배치 학습

`Dataset` 클래스는 개별 표본에 어떻게 접근하는지를 정의하고, `DataLoader`는 데이터셋을 감싸서 배치 단위로 뒤섞인 반복을 제공한다.

핵심 클래스는 `RegressionDataset`, `SimpleModel`, `SimpleDataset`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
