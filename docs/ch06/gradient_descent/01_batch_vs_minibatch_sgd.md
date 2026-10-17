# 배치 크기에 따른 경사 하강법 비교

같은 선형 회귀 데이터를 전체 배치, 표본 하나, 미니배치로 처리한다. 에포크당 업데이트 수와 기록된 로스의 차이를 중심으로 비교한다.

## 1. 코드

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
   - 온 데이터셋에 대해 그래디언트를 계산한다
   - 그래디언트가 가장 맞지만 데이터가 크면 느리다
   
   Pseudocode:
   for epoch in epochs:
       gradient = compute_gradient(all_data)
       parameters -= lr * gradient

2. 확률 경사 하강법(SGD)
   - 루프마다 마구잡이 표본 하나를 쓴다
   - 루프마다 훨씬 빠르다
   - 그래디언트가 잡음 섞이지만 그 자리 골짜기를 벗어날 수 있다
   
   Pseudocode:
   for epoch in epochs:
       for sample in shuffle(data):
           gradient = compute_gradient(sample)
           parameters -= lr * gradient

3. 작은 배치 경사 하강법(작은 배치 GD)
   - 데이터를 작은 배치로 나누어 쓴다
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
        
        # 이 에폭의 평균 로스 기록
        loss_history.append(np.mean(epoch_losses))
    
    training_time = time.time() - start_time
    return model, loss_history, training_time


def train_minibatch_gd(X, y, n_epochs, lr, batch_size):
    """작은 배치 경사 하강법 - 작은 배치를 쓴다"""
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
        
        # 이 에폭의 평균 로스 기록
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

# 그림 1: 로스 곡선(선형 축)
axes[0, 0].plot(loss_batch, label='Batch GD', linewidth=2)
axes[0, 0].plot(loss_sgd, label='SGD', linewidth=2, alpha=0.7)
axes[0, 0].plot(loss_minibatch, label='Mini-batch GD', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Convergence (Linear Scale)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 그림 2: 로스 곡선(로그 축)
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

# 그림 4: 로스의 분산(매끄러움)
axes[1, 1].plot(loss_sgd[-20:], 'o-', label='SGD (last 20 epochs)', linewidth=2)
axes[1, 1].plot(loss_minibatch[-20:], 's-', label='Mini-batch (last 20 epochs)', linewidth=2)
axes[1, 1].plot(loss_batch[-20:], '^-', label='Batch (last 20 epochs)', linewidth=2)
axes[1, 1].set_xlabel('Last 20 Epochs')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Loss Stability (Zoomed In)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('batch_comparison.png', dpi=150)
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
   ✗ 그래디언트가 잡음 섞이고 모여듦이 들쭉날쭉하다
   ✗ 딱 맞는 골짜기에 닿지 못할 수 있다
   
3. 작은 배치 GD: (둘의 좋은 점을 다 가진다!)
   ✓ 속도와 든든함이 고루 좋다
   ✓ GPU의 나란한 셈을 쓸 수 있다
   ✓ 두루 잘 미친다
   → 참으로 가장 흔히 쓴다!

배치 크기 고르기:
- 너무 작으면(보기: 1~8) 잡음이 많고 더디게 모여든다
- 너무 크면(보기: 512 넘게) 기억 자리 탈이 생기고 일반화이 나빠진다
- 알맞은 자리: 16, 32, 64, 128(GPU이 잘 들도록 2의 거듭제곱)

손에 잡히는 권함:
• 기본으로 batch_size=32이나 64인 작은 배치 GD을 써라
• 기억 자리가 넉넉하고 속도가 필요하면 배치 크기를 키워라
• 지나치게 맞춰지거나 일반화를 높이려면 배치 크기를 줄여라
• 여러 배치 크기를 써 보아라. 이것도 초파라메터다!
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

3. 클래스마다 다른 러닝레이트
   - SGD에는 더 작은 러닝레이트가 필요할 수 있다
   - 배치 GD은 더 큰 러닝레이트로도 될 수 있다

4. 여세를 더해 보아라(다음 보기다!)
   - optimizer = torch.optim.SGD(..., momentum=0.9)
""")
print("="*80)


if __name__ == "__main__":
    pass
```

??? note "전체 출력 (143줄)"

    ```
    ================================================================================
    BATCH vs MINI-BATCH vs STOCHASTIC GRADIENT DESCENT
    ================================================================================

    ================================================================================
    PART 1: WHAT ARE THE VARIANTS?
    ================================================================================

    경사 하강법의 여러 클래스:
    --------------------------

    1. 배치 경사 하강법(배치 GD)
       - 루프마다 학습 데이터를 모두 쓴다
       - 온 데이터셋에 대해 그래디언트를 계산한다
       - 그래디언트가 가장 맞지만 데이터가 크면 느리다
       
       Pseudocode:
       for epoch in epochs:
           gradient = compute_gradient(all_data)
           parameters -= lr * gradient

    2. 확률 경사 하강법(SGD)
       - 루프마다 마구잡이 표본 하나를 쓴다
       - 루프마다 훨씬 빠르다
       - 그래디언트가 잡음 섞이지만 그 자리 골짜기를 벗어날 수 있다
       
       Pseudocode:
       for epoch in epochs:
           for sample in shuffle(data):
               gradient = compute_gradient(sample)
               parameters -= lr * gradient

    3. 작은 배치 경사 하강법(작은 배치 GD)
       - 데이터를 작은 배치로 나누어 쓴다
       - 배치 GD과 SGD 사이의 고른 자리
       - 참으로 가장 흔하다(보기: batch_size=32, 64, 128)
       
       Pseudocode:
       for epoch in epochs:
           for batch in create_batches(data, batch_size):
               gradient = compute_gradient(batch)
               parameters -= lr * gradient


    ================================================================================
    PART 2: DATASET PREPARATION
    ================================================================================
    Dataset: 1000 samples
    True relationship: y = 3x + 2

    ================================================================================
    PART 5: COMPARING ALL VARIANTS
    ================================================================================
    Training with different variants...
    Epochs: 50, Learning rate: 0.01

    1. Training with Batch GD (all 1000 samples at once)...
       ✓ Completed in 0.030 seconds
    2. Training with SGD (1 sample at a time)...
       ✓ Completed in 26.771 seconds
    3. Training with Mini-batch GD (batch_size=32)...
       ✓ Completed in 3.210 seconds

    ================================================================================
    PART 6: COMPARISON RESULTS
    ================================================================================

    Training Time Comparison:
      Batch GD:      0.030s
      SGD:           26.771s
      Mini-batch GD: 3.210s

    Final Loss:
      Batch GD:      0.446686
      SGD:           0.262440
      Mini-batch GD: 0.247880

    Learned Parameters:
      Batch      : w=2.9540, b=1.6070
      SGD        : w=3.0021, b=2.1055
      Mini-batch : w=2.9841, b=2.0357
      True       : w=3.0000, b=2.0000

    ================================================================================
    VISUALIZATION
    ================================================================================

    ✓ Plot saved as 'batch_comparison.png'

    Close the plot window to continue...

    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. BATCH GD:
       ✓ 가장 든든하고 매끄럽게 모여든다
       ✗ 데이터가 크면 느리다
       ✗ 기억 자리를 많이 쓴다
       ✗ 그 자리 골짜기에 갇힐 수 있다
       
    2. STOCHASTIC GD:
       ✓ 에폭마다 학습이 빠르다
       ✓ 얕은 그 자리 골짜기를 벗어날 수 있다
       ✗ 그래디언트가 잡음 섞이고 모여듦이 들쭉날쭉하다
       ✗ 딱 맞는 골짜기에 닿지 못할 수 있다
       
    3. 작은 배치 GD: (둘의 좋은 점을 다 가진다!)
       ✓ 속도와 든든함이 고루 좋다
       ✓ GPU의 나란한 셈을 쓸 수 있다
       ✓ 두루 잘 미친다
       → 참으로 가장 흔히 쓴다!

    배치 크기 고르기:
    - 너무 작으면(보기: 1~8) 잡음이 많고 더디게 모여든다
    - 너무 크면(보기: 512 넘게) 기억 자리 탈이 생기고 일반화이 나빠진다
    - 알맞은 자리: 16, 32, 64, 128(GPU이 잘 들도록 2의 거듭제곱)

    손에 잡히는 권함:
    • 기본으로 batch_size=32이나 64인 작은 배치 GD을 써라
    • 기억 자리가 넉넉하고 속도가 필요하면 배치 크기를 키워라
    • 지나치게 맞춰지거나 일반화를 높이려면 배치 크기를 줄여라
    • 여러 배치 크기를 써 보아라. 이것도 초파라메터다!

    ================================================================================
    EXPERIMENTS TO TRY
    ================================================================================

    1. 여러 배치 크기: 8, 16, 64, 128, 256
       - 모여듦에 어떤 영향을 주는가?
       - 학습 때는 어떠한가?

    2. 더 큰 데이터셋(표본 10,000개)
       - 이제는 어느 클래스가 가장 빠른가?

    3. 클래스마다 다른 러닝레이트
       - SGD에는 더 작은 러닝레이트가 필요할 수 있다
       - 배치 GD은 더 큰 러닝레이트로도 될 수 있다

    4. 여세를 더해 보아라(다음 보기다!)
       - optimizer = torch.optim.SGD(..., momentum=0.9)

    ================================================================================
    ```


## 2. 논의

### 배치 크기가 바꾸는 것

전체 배치 학습은 한 에포크에 한 번, 표본 단위 SGD는 표본 수만큼, 미니배치는 배치 수만큼 업데이트한다. 모두 `torch.optim.SGD`를 사용할 수 있다. 옵티마이저 클래스 이름이 입력 배치 크기를 결정하는 것은 아니다.

### 로스 곡선과 시간 비교의 조건

전체 배치의 기록은 한 파라미터 상태에서 구한 전체 로스다. 표본·미니배치 방식의 에포크 평균에는 에포크 도중 서로 다른 파라미터에서 계산한 로스가 섞여 있다. 마지막 배치가 짧으면 배치 평균들의 단순 평균은 표본 평균과도 다르다. 공정한 최종 성능 비교에는 에포크 종료 후 같은 전체 데이터에서 로스를 다시 평가하는 기준이 유용하다.

각 함수는 모델을 새로 만들므로 초기값이 동일하게 맞춰진 비교는 아니다. 같은 에포크 수라도 업데이트 횟수가 다르고, 측정 시간도 하드웨어와 데이터 처리 비용에 영향을 받는다. 따라서 제공된 곡선과 시간은 해당 설정의 예시이며 특정 배치 크기의 보편적인 우열을 보여주지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
SGD 대신 Adam 옵티마이저를 쓰도록 코드를 수정하라. 100 에포크에 걸친 수렴 속도를 비교하라.

</div>

??? success "풀이"
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

??? success "풀이"
    업데이트 사이에 경사를 초기화하지 않으면 서로 다른 파라미터에서 계산한 이전 경사까지 누적된다. 따라서 현재 로스의 경사만 사용하는 학습 규칙과 달라지고, 진동이나 발산을 일으킬 수 있다. 경사끼리 상쇄될 수도 있으므로 크기가 항상 증가하거나 반드시 발산하는 것은 아니다. 의도적인 경사 누적에서는 정해진 여러 마이크로배치를 처리한 뒤 한 번 업데이트하고 경사를 초기화한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
옵티마이저에 L2 정규화(웨이트 감쇠)를 추가하고 그것이 최종 파라메터 값에 어떤 영향을 주는지 관찰하라.

</div>

??? success "풀이"
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    # weight_decay는 로스에 L2 벌점항 lambda * ||w||^2을 더한다.
    # 이는 웨이트를 작게 유도하여 과적합을 막을 수 있다.
    # 최종 웨이트의 크기가 조금 더 작아진다.
    ```

## 정리하며

배치 크기는 경사 추정뿐 아니라 업데이트 횟수와 실행 비용을 바꾼다. 비교할 때는 초기값, 평가 기준, 처리 표본 수와 업데이트 수를 함께 맞춘다.
