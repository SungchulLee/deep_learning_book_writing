# 배치, 미니배치, 확률적 경사 하강법

이 스크립트는 배치, 미니배치, 확률적 경사 하강법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""
================================================================================
2단계 - 보기 1: 배치, 작은 배치, 확률 경사 하강법
================================================================================

학습 목표:
- 경사 하강법의 여러 클래스를 이해한다
- 배치 경사 하강법, 작은 배치 경사 하강법, SGD을 견준다
- PyTorch의 DataLoader과 배치 만들기를 배운다
- 클래스 사이의 맞바꿈을 이해한다

어려움: ⭐⭐ 가운데 수준

걸리는 때: 35~45분

PREREQUISITES:
- 1단계 보기를 마쳤을 것
- 기본 경사 하강법을 이해하고 있을 것

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
경사 하강법의 여러 클래스:
--------------------------

1. 배치 경사 하강법(배치 GD)
   - 루프마다 학습 데이터를 모두 쓴다
   - 온 데이터셋에 대해 기울기를 계산한다
   - 기울기가 가장 맞지만 데이터가 크면 느리다
   
   Pseudocode:
   for epoch in epochs:
       gradient = compute_gradient(all_data)
       parameters -= lr * gradient

2. 확률 경사 하강법(SGD)
   - 루프마다 마구잡이 표본 하나를 쓴다
   - 루프마다 훨씬 빠르다
   - 기울기가 잡음 섞이지만 그 자리 골짜기를 벗어날 수 있다
   
   Pseudocode:
   for epoch in epochs:
       for sample in shuffle(data):
           gradient = compute_gradient(sample)
           parameters -= lr * gradient

3. 작은 배치 경사 하강법(작은 배치 GD)
   - 데이터를 작은 배치으로 나누어 쓴다
   - 배치 GD과 SGD 사이의 고른 자리
   - 참으로 가장 흔하다(보기: batch_size=32, 64, 128)
   
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
    """배치 경사 하강법 - 데이터를 한꺼번에 모두 쓴다"""
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
    """확률 경사 하강법 - 한 번에 표본 하나"""
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
    """작은 배치 경사 하강법 - 작은 배치을 쓴다"""
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
   ✓ 가장 든든하고 매끄럽게 모여든다
   ✗ 데이터가 크면 느리다
   ✗ 기억 자리를 많이 쓴다
   ✗ 그 자리 골짜기에 갇힐 수 있다
   
2. STOCHASTIC GD:
   ✓ 에폭마다 학습이 빠르다
   ✓ 얕은 그 자리 골짜기를 벗어날 수 있다
   ✗ 기울기가 잡음 섞이고 모여듦이 들쭉날쭉하다
   ✗ 딱 맞는 골짜기에 닿지 못할 수 있다
   
3. 작은 배치 GD: (둘의 좋은 점을 다 가진다!)
   ✓ 빠르기와 든든함이 고루 좋다
   ✓ GPU의 나란한 셈을 쓸 수 있다
   ✓ 두루 잘 미친다
   → 참으로 가장 흔히 쓴다!

배치 크기 고르기:
- 너무 작으면(보기: 1~8) 잡음이 많고 더디게 모여든다
- 너무 크면(보기: 512 넘게) 기억 자리 탈이 생기고 일반화이 나빠진다
- 알맞은 자리: 16, 32, 64, 128(GPU이 잘 들도록 2의 거듭제곱)

손에 잡히는 권함:
• 기본으로 batch_size=32이나 64인 작은 배치 GD을 써라
• 기억 자리가 넉넉하고 빠르기가 필요하면 배치 크기를 키워라
• 지나치게 맞춰지거나 일반화을 높이려면 배치 크기를 줄여라
• 여러 배치 크기를 써 보아라. 이것도 초매개변수다!
""")

print("="*80)
print("EXPERIMENTS TO TRY")
print("="*80)
print("""
1. 여러 배치 크기: 8, 16, 64, 128, 256
   - 모여듦에 어떤 영향을 주는가?
   - 학습 때는 어떠한가?

2. 더 큰 데이터셋(표본 10,000개)
   - 이제는 어느 클래스가 가장 빠른가?

3. 클래스마다 다른 학습률
   - SGD에는 더 작은 학습률가 필요할 수 있다
   - 배치 GD은 더 큰 학습률로도 될 수 있다

4. 여세를 더해 보아라(다음 보기다!)
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
