# 배치, 미니배치, 확률적 경사 하강법

이 스크립트는 배치, 미니배치, 확률적 경사 하강법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""
================================================================================
Level 2 - Example 1: Batch, Mini-batch, and Stochastic Gradient Descent
================================================================================

LEARNING OBJECTIVES:
- Understand different variants of gradient descent
- Compare batch GD, mini-batch GD, and SGD
- Learn about DataLoader and batching in PyTorch
- Understand trade-offs between variants

DIFFICULTY: ⭐⭐ Intermediate

TIME: 35-45 minutes

PREREQUISITES:
- Completed Level 1 examples
- Understanding of basic gradient descent

================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

print("="*80)
print("BATCH vs MINI-BATCH vs STOCHASTIC GRADIENT DESCENT")
print("="*80)

# ============================================================================
# 1부: 변형들 이해하기
# ============================================================================
print("\n" + "="*80)
print("PART 1: WHAT ARE THE VARIANTS?")
print("="*80)

print("""
GRADIENT DESCENT VARIANTS:
--------------------------

1. BATCH GRADIENT DESCENT (Batch GD)
   - Uses ALL training data in each iteration
   - Computes gradient over entire dataset
   - Most accurate gradient, but slow for large datasets
   
   Pseudocode:
   for epoch in epochs:
       gradient = compute_gradient(all_data)
       parameters -= lr * gradient

2. STOCHASTIC GRADIENT DESCENT (SGD)
   - Uses ONE random sample per iteration
   - Much faster per iteration
   - Noisy gradients, but can escape local minima
   
   Pseudocode:
   for epoch in epochs:
       for sample in shuffle(data):
           gradient = compute_gradient(sample)
           parameters -= lr * gradient

3. MINI-BATCH GRADIENT DESCENT (Mini-batch GD)
   - Uses SMALL BATCHES of data
   - Balance between Batch GD and SGD
   - Most common in practice (e.g., batch_size=32, 64, 128)
   
   Pseudocode:
   for epoch in epochs:
       for batch in create_batches(data, batch_size):
           gradient = compute_gradient(batch)
           parameters -= lr * gradient
""")

# ============================================================================
# 2부: 데이터셋 만들기
# ============================================================================
print("\n" + "="*80)
print("PART 2: DATASET PREPARATION")
print("="*80)

torch.manual_seed(42)
np.random.seed(42)

# 차이를 보기 위한 더 큰 데이터셋
n_samples = 1000
X_numpy = np.random.randn(n_samples, 1) * 2
y_numpy = 3 * X_numpy + 2 + np.random.randn(n_samples, 1) * 0.5

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"Dataset: {n_samples} samples")
print(f"True relationship: y = 3x + 2")

# ============================================================================
# 3부: 모델 정의
# ============================================================================

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# ============================================================================
# 4부: 각 변형을 위한 학습 함수
# ============================================================================

def train_batch_gd(X, y, n_epochs, lr):
    """Batch Gradient Descent - Use all data at once"""
    model = SimpleLinearModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # 한 번의 순전파에 데이터 전체를 쓴다
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
    
    training_time = time.time() - start_time
    return model, loss_history, training_time


def train_sgd(X, y, n_epochs, lr):
    """Stochastic Gradient Descent - One sample at a time"""
    model = SimpleLinearModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    loss_history = []
    start_time = time.time()
    n_samples = X.shape[0]
    
    for epoch in range(n_epochs):
        # 에폭마다 데이터 섞기
        indices = torch.randperm(n_samples)
        
        # 한 번에 표본 하나씩 처리
        epoch_losses = []
        for i in indices:
            X_sample = X[i:i+1]  # Keep dimension (1, 1)
            y_sample = y[i:i+1]
            
            y_pred = model(X_sample)
            loss = criterion(y_pred, y_sample)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # 이 에폭의 평균 손실 기록
        loss_history.append(np.mean(epoch_losses))
    
    training_time = time.time() - start_time
    return model, loss_history, training_time


def train_minibatch_gd(X, y, n_epochs, lr, batch_size):
    """Mini-batch Gradient Descent - Use small batches"""
    model = SimpleLinearModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # 자동 배치 구성을 위한 DataLoader 생성
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # 미니배치를 순회한다
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # 이 에폭의 평균 손실 기록
        loss_history.append(np.mean(epoch_losses))
    
    training_time = time.time() - start_time
    return model, loss_history, training_time

# ============================================================================
# 5부: 세 가지 변형 모두로 학습하기
# ============================================================================
print("\n" + "="*80)
print("PART 5: COMPARING ALL VARIANTS")
print("="*80)

n_epochs = 50
learning_rate = 0.01
batch_size = 32

print("Training with different variants...")
print(f"Epochs: {n_epochs}, Learning rate: {learning_rate}\n")

# 배치 경사 하강법 학습
print("1. Training with Batch GD (all 1000 samples at once)...")
model_batch, loss_batch, time_batch = train_batch_gd(X, y, n_epochs, learning_rate)
print(f"   ✓ Completed in {time_batch:.3f} seconds")

# SGD 학습
print("2. Training with SGD (1 sample at a time)...")
model_sgd, loss_sgd, time_sgd = train_sgd(X, y, n_epochs, learning_rate)
print(f"   ✓ Completed in {time_sgd:.3f} seconds")

# 미니배치 경사 하강법 학습
print(f"3. Training with Mini-batch GD (batch_size={batch_size})...")
model_minibatch, loss_minibatch, time_minibatch = train_minibatch_gd(
    X, y, n_epochs, learning_rate, batch_size
)
print(f"   ✓ Completed in {time_minibatch:.3f} seconds")

# ============================================================================
# 6부: 결과 비교
# ============================================================================
print("\n" + "="*80)
print("PART 6: COMPARISON RESULTS")
print("="*80)

print("\nTraining Time Comparison:")
print(f"  Batch GD:      {time_batch:.3f}s")
print(f"  SGD:           {time_sgd:.3f}s")
print(f"  Mini-batch GD: {time_minibatch:.3f}s")

print("\nFinal Loss:")
print(f"  Batch GD:      {loss_batch[-1]:.6f}")
print(f"  SGD:           {loss_sgd[-1]:.6f}")
print(f"  Mini-batch GD: {loss_minibatch[-1]:.6f}")

print("\nLearned Parameters:")
for name, model in [("Batch", model_batch), ("SGD", model_sgd), ("Mini-batch", model_minibatch)]:
    w = model.linear.weight.item()
    b = model.linear.bias.item()
    print(f"  {name:11s}: w={w:.4f}, b={b:.4f}")
print(f"  {'True':11s}: w=3.0000, b=2.0000")

# ============================================================================
# 7부: 시각화
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 그림 1: 손실 곡선(선형 축)
axes[0, 0].plot(loss_batch, label='Batch GD', linewidth=2)
axes[0, 0].plot(loss_sgd, label='SGD', linewidth=2, alpha=0.7)
axes[0, 0].plot(loss_minibatch, label='Mini-batch GD', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Convergence (Linear Scale)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 그림 2: 손실 곡선(로그 축)
axes[0, 1].plot(loss_batch, label='Batch GD', linewidth=2)
axes[0, 1].plot(loss_sgd, label='SGD', linewidth=2, alpha=0.7)
axes[0, 1].plot(loss_minibatch, label='Mini-batch GD', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Loss Convergence (Log Scale)')
axes[0, 1].set_yscale('log')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 그림 3: 학습 시간 비교
methods = ['Batch GD', 'SGD', 'Mini-batch']
times = [time_batch, time_sgd, time_minibatch]
colors = ['blue', 'orange', 'green']
axes[1, 0].bar(methods, times, color=colors, alpha=0.7)
axes[1, 0].set_ylabel('Training Time (seconds)')
axes[1, 0].set_title('Training Time Comparison')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 그림 4: 손실의 분산(매끄러움)
axes[1, 1].plot(loss_sgd[-20:], 'o-', label='SGD (last 20 epochs)', linewidth=2)
axes[1, 1].plot(loss_minibatch[-20:], 's-', label='Mini-batch (last 20 epochs)', linewidth=2)
axes[1, 1].plot(loss_batch[-20:], '^-', label='Batch (last 20 epochs)', linewidth=2)
axes[1, 1].set_xlabel('Last 20 Epochs')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Loss Stability (Zoomed In)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_2_intermediate/batch_comparison.png', dpi=150)
print("\n✓ Plot saved as 'batch_comparison.png'")
print("\nClose the plot window to continue...")
plt.show()

# ============================================================================
# 8부: 핵심 요점
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. BATCH GD:
   ✓ Most stable, smooth convergence
   ✗ Slow for large datasets
   ✗ High memory usage
   ✗ Can get stuck in local minima
   
2. STOCHASTIC GD:
   ✓ Fast per-epoch training
   ✓ Can escape shallow local minima
   ✗ Noisy gradients, erratic convergence
   ✗ May not reach exact minimum
   
3. MINI-BATCH GD: (BEST OF BOTH WORLDS!)
   ✓ Good balance of speed and stability
   ✓ Can leverage GPU parallelization
   ✓ Generalizes well
   → Most commonly used in practice!

BATCH SIZE SELECTION:
- Too small (e.g., 1-8): Noisy, slow to converge
- Too large (e.g., >512): Memory issues, poor generalization
- Sweet spot: 16, 32, 64, 128 (power of 2 for GPU efficiency)

PRACTICAL RECOMMENDATIONS:
• Use mini-batch GD with batch_size=32 or 64 as default
• Increase batch size if you have memory and need speed
• Decrease batch size if overfitting or for better generalization
• Try different batch sizes - it's a hyperparameter!
""")

print("="*80)
print("EXPERIMENTS TO TRY")
print("="*80)
print("""
1. Different batch sizes: 8, 16, 64, 128, 256
   - How does it affect convergence?
   - What about training time?

2. Larger dataset (10,000 samples)
   - Which variant is fastest now?

3. Different learning rates for each variant
   - SGD might need smaller lr
   - Batch GD might work with larger lr

4. Add momentum (next example!)
   - optimizer = torch.optim.SGD(..., momentum=0.9)
""")
print("="*80)


if __name__ == "__main__":
    pass
```

## 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사 초기화, 역전파, 매개변수 갱신이다. 각 구성 요소가 결정적인 역할을 한다. 최적화기는 갱신 규칙(SGD, Adam 등)을 캡슐화하고 학습률과 모멘텀 상태를 내부에서 관리한다.

PyTorch의 `DataLoader`는 `Dataset`을 감싸 배치 구성, 섞기, 병렬 데이터 적재를 제공한다. `num_workers`, `pin_memory`, `batch_size`를 적절히 설정하면 GPU가 데이터를 기다리는 일이 없도록 하여 학습 처리량을 크게 개선할 수 있다.

모델 체크포인팅은 학습 진행 상황을 디스크에 저장하여 중단으로부터의 복구와 모델 배포를 가능하게 한다. 모델 전체를 피클로 저장하는 것보다 (매개변수만 담은) `state_dict`를 저장하는 편이 낫다. 이식성이 좋고 정확한 클래스 정의 경로에 의존하지 않기 때문이다.

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
