# 정칙화

정칙화(regularization)는 손실 함수에 큰 매개변수 값을 억제하는 벌점 항을 더하여, 특히 데이터 양에 비해 매개변수가 많은 모델의 과적합을 막는 데 도움을 준다. L2 정칙화(릿지)는 모든 특징을 유지하면서 가중치를 0 쪽으로 줄이는 반면, L1 정칙화(라쏘)는 일부 가중치를 정확히 0으로 만들어 사실상 특징 선택을 수행한다. 이 튜토리얼은 소수의 특징만 실제로 유용한 고차원 데이터셋에서 두 방법을 비교한다.

## 1. 코드

```python
"""
==============================================================================
08_regularization.py
==============================================================================
어려움: ⭐⭐⭐⭐ (앞선)

DESCRIPTION:
    과적합을 막는 L1과 L2 정칙화.
    PyTorch로 하는 능선 회귀와 라쏘 회귀.

다루는 것:
    - L1(라쏘)과 L2(능선) 정칙화
    - 최적화기의 가중치 줄이기
    - L1으로 하는 특징 고르기
    - 정칙화 매개변수 손보기

PREREQUISITES:
    - 튜토리얼 07(다항 회귀)

걸리는 때: 30분쯤
==============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("REGULARIZATION: L1 AND L2")
print("=" * 70)

# ============================================================================
# 1부: 특징이 많은 데이터 생성
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE HIGH-DIMENSIONAL DATA")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)

# 소수의 특징만 실제로 중요한 데이터를 생성한다
n_samples = 100
n_features = 20
n_informative = 5  # Only 5 features actually matter

X = np.random.randn(n_samples, n_features)
true_weights = np.zeros(n_features)
true_weights[:n_informative] = np.array([3, -2, 1.5, -1, 2])  # Important features
y = X @ true_weights + np.random.randn(n_samples) * 0.5

print(f"Dataset:")
print(f"  Samples: {n_samples}")
print(f"  Total features: {n_features}")
print(f"  Informative features: {n_informative}")
print(f"\nTrue important weights (first 5 features):")
print(f"  {true_weights[:n_informative]}")

# 텐서로 바꾼다
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# ============================================================================
# 2부: 정칙화 없음 (기준선)
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: BASELINE MODEL (NO REGULARIZATION)")
print("=" * 70)

model_baseline = nn.Linear(n_features, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model_baseline.parameters(), lr=0.01)

n_epochs = 500
for epoch in range(n_epochs):
    y_pred = model_baseline(X_tensor)
    loss = criterion(y_pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

baseline_weights = model_baseline.weight.detach().numpy().flatten()
print(f"Baseline model trained")
print(f"  Number of near-zero weights (|w| < 0.1): "
      f"{np.sum(np.abs(baseline_weights) < 0.1)}/{n_features}")

# ============================================================================
# 3부: L2 정칙화 (릿지 회귀)
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: L2 REGULARIZATION (RIDGE)")
print("=" * 70)

print("""
L2 정칙화(능선):
  손실 = MSE + λ * Σ(w²)
  
  - 큰 가중치에 벌을 준다
  - 가중치가 0 쪽으로 오그라든다
  - 특징을 모두 남긴다(고르지 않는다)
  - 최적화기의 'weight_decay'으로 짠다
""")

# 여러 L2 벌점으로 학습한다
l2_lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
l2_models = {}

for lambda_val in l2_lambdas:
    model = nn.Linear(n_features, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, 
                                weight_decay=lambda_val)  # ← L2 regularization
    
    for epoch in range(n_epochs):
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    l2_models[lambda_val] = model
    weights = model.weight.detach().numpy().flatten()
    print(f"λ={lambda_val:<6.3f}: Mean |weight|={np.mean(np.abs(weights)):.4f}, "
          f"Max |weight|={np.max(np.abs(weights)):.4f}")

# ============================================================================
# 4부: L1 정칙화 (라쏘 회귀)
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: L1 REGULARIZATION (LASSO)")
print("=" * 70)

print("""
L1 정칙화(라쏘):
  손실 = MSE + λ * Σ(|w|)
  
  - 가중치의 절댓값에 벌을 준다
  - 어떤 가중치는 딱 0으로 만든다
  - 특징을 골라 준다
  - PyTorch에서는 직접 짜야 한다
""")

def train_with_l1(lambda_l1, n_epochs=500):
    """L1 정칙화로 모델을 학습시킨다"""
    model = nn.Linear(n_features, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(n_epochs):
        y_pred = model(X_tensor)
        mse_loss = criterion(y_pred, y_tensor)
        
        # L1 벌점을 직접 더한다
        l1_penalty = lambda_l1 * torch.sum(torch.abs(model.weight))
        total_loss = mse_loss + l1_penalty
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    return model

# 여러 L1 벌점으로 학습한다
l1_lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
l1_models = {}

for lambda_val in l1_lambdas:
    model = train_with_l1(lambda_val)
    l1_models[lambda_val] = model
    weights = model.weight.detach().numpy().flatten()
    n_zero = np.sum(np.abs(weights) < 0.01)  # Nearly zero
    print(f"λ={lambda_val:<6.3f}: Nearly zero weights={n_zero}/{n_features}, "
          f"Mean |weight|={np.mean(np.abs(weights)):.4f}")

# ============================================================================
# 5부: 가중치 크기 시각화
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: VISUALIZE REGULARIZATION EFFECTS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 기준선
ax = axes[0, 0]
ax.bar(range(n_features), baseline_weights)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax.set_title('No Regularization')
ax.set_xlabel('Feature Index')
ax.set_ylabel('Weight')
ax.grid(True, alpha=0.3)

# L2 정칙화
for idx, lambda_val in enumerate([0.001, 0.01, 0.1, 1.0]):
    ax = axes[idx//2, idx%2 + 1] if idx < 2 else axes[1, idx-2]
    if idx < 2:
        # L2
        weights = l2_models[lambda_val].weight.detach().numpy().flatten()
        ax.bar(range(n_features), weights)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.set_title(f'L2 (λ={lambda_val})')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)

# L1 정칙화
for idx, lambda_val in enumerate([0.01, 0.1]):
    ax = axes[1, idx + 1]
    weights = l1_models[lambda_val].weight.detach().numpy().flatten()
    ax.bar(range(n_features), weights)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_title(f'L1 (λ={lambda_val})')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Weight')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/08_regularization_weights.png', dpi=100)
print("Saved weight visualization")

# ============================================================================
# 6부: 비교 그림
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# L2의 효과 그리기
ax = axes[0]
for lambda_val in l2_lambdas:
    weights = l2_models[lambda_val].weight.detach().numpy().flatten()
    ax.plot(sorted(np.abs(weights), reverse=True), 
            marker='o', label=f'λ={lambda_val}')
ax.set_xlabel('Feature (sorted by magnitude)')
ax.set_ylabel('|Weight|')
ax.set_title('L2 Regularization Effect')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# L1의 효과 그리기
ax = axes[1]
for lambda_val in l1_lambdas:
    weights = l1_models[lambda_val].weight.detach().numpy().flatten()
    ax.plot(sorted(np.abs(weights), reverse=True), 
            marker='o', label=f'λ={lambda_val}')
ax.set_xlabel('Feature (sorted by magnitude)')
ax.set_ylabel('|Weight|')
ax.set_title('L1 Regularization Effect (Feature Selection)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/08_regularization_comparison.png', dpi=100)
print("Saved comparison visualization")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
정칙화 간추림:

L2 정칙화(능선):
✓ 모든 가중치를 0 쪽으로 오그린다
✓ 특징을 모두 남긴다
✓ 짜기 쉽다(최적화기의 weight_decay)
✓ 서로 얽힌 특징에 좋다
✗ 특징을 골라 주지 않는다

L1 정칙화(라쏘):
✓ 어떤 가중치는 딱 0으로 만든다
✓ 특징을 절로 골라 준다
✓ 성긴 모델(살아 있는 특징이 적다)
✗ 직접 짜야 한다
✗ 서로 얽힌 특징에서는 흔들린다

언제 쓸까:
- L2: 모든 특징이 종요로울 수 있을 때
- L1: 특징을 골라내고 싶을 때
- 둘 다(엘라스틱넷): 이점을 아우른다

PyTorch에서 짜기:
  # L2: 내장 기능
  optimizer = torch.optim.SGD(params, lr=0.01, weight_decay=0.01)
  
  # L1: 직접 구현
  l1_loss = lambda_l1 * torch.sum(torch.abs(model.weight))
  total_loss = mse_loss + l1_loss

다음: 튜토리얼 09 - DataLoader으로 하는 작은 배치 익히기!
""")


if __name__ == "__main__":
    pass
```

## 2. 논의

L2 정칙화는 손실을 $L = \text{MSE} + \lambda \sum w_i^2$로 바꾸어 가중치 제곱합에 벌점을 준다. PyTorch에서는 최적화기의 `weight_decay` 매개변수를 설정하여 구현하는데, 이는 벌점의 경사를 매개변수 갱신에 곧바로 더한다. 그 효과는 모든 가중치를 비례해서 줄이는 것이며, 더 매끄럽고 일반화가 잘 되는 모델이 나온다. $\lambda$가 클수록 더 많이 줄어들지만 너무 크면 과소적합할 수 있다.

L1 정칙화는 손실 $L = \text{MSE} + \lambda \sum |w_i|$를 써서 가중치 절댓값의 합에 벌점을 준다. L2와 달리 L1의 경사는 가중치의 크기에 의존하지 않으므로, 작은 가중치도 큰 가중치와 똑같은 힘으로 0 쪽으로 밀린다. 그 결과 많은 가중치가 정확히 0인 희소 해가 나오며, 사실상 가장 중요한 특징만 선택하게 된다. PyTorch에서 `weight_decay`는 L2만 구현하므로 L1은 손실에 직접 더해 주어야 한다.

L1과 L2 중 무엇을 고를지는 문제의 구조에 달려 있다. 모든 특징이 잠재적으로 유의미할 때(예: 이미지의 화소값)는 모든 특징을 줄어든 크기로 남겨 두는 L2가 적절하다. 많은 특징이 무관할 때(예: 소수의 유전자만 중요한 유전자 발현 데이터)는 잡음 특징을 찾아 버리는 L1이 낫다. 엘라스틱 넷(Elastic Net)은 두 벌점을 결합해 각각의 이점을 취하며, PyTorch에서는 L1 항과 weight_decay를 함께 더해 구현할 수 있다.

## 연습문제

**연습문제 1.**
L2 정칙화로 $\lambda \in \{0, 0.001, 0.01, 0.1, 1.0\}$인 모델들을 학습시키고, 각각에 대해 학습된 가중치 크기의 분포를 그려라. $\lambda$를 키우면 가중치 분포는 어떻게 달라지는가?

??? success "연습문제 1 풀이"
    $\lambda$가 커지면 가중치의 크기가 고르게 줄어든다. $\lambda = 0$이면 가중치에 제약이 없어 커질 수 있다. $\lambda = 1.0$이면 가중치가 0 쪽으로 크게 줄어든다. 정칙화가 강해질수록 분포가 0 주변으로 좁아진다. 각 $\lambda$에 대해 가중치 절댓값의 막대그래프를 그려 시각화할 수 있다.

---

**연습문제 2.**
L1과 L2 벌점을 결합하여 엘라스틱 넷 정칙화를 구현하라. $L = \text{MSE} + \alpha \sum |w_i| + \beta \sum w_i^2$. 모델을 학습시키고 0에 가까운 가중치의 개수를 순수 L1 및 순수 L2와 비교하라.

??? success "연습문제 2 풀이"
    ```python
    import torch
    import torch.nn as nn
    import numpy as np
    
    np.random.seed(42)
    n, p = 100, 20
    X = torch.FloatTensor(np.random.randn(n, p))
    true_w = torch.zeros(p); true_w[:5] = torch.tensor([3., -2., 1.5, -1., 2.])
    y = (X @ true_w + torch.randn(n) * 0.5).reshape(-1, 1)
    
    model = nn.Linear(p, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)  # L2
    alpha_l1 = 0.01  # L1 coefficient
    
    for epoch in range(500):
        y_pred = model(X)
        mse = nn.MSELoss()(y_pred, y)
        l1 = alpha_l1 * torch.sum(torch.abs(model.weight))
        loss = mse + l1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    weights = model.weight.detach().numpy().flatten()
    n_zero = np.sum(np.abs(weights) < 0.01)
    print(f'Near-zero weights: {n_zero}/{p}')
    ```

---

**연습문제 3.**
20개 중 5개만 유용한 특징을 가진 데이터셋에서 $\lambda = 0.1$인 L1 정칙화로 학습한 가중치를 막대그래프로 그리고, 0이 아닌 가중치가 처음 5개 특징에 대응하는지 확인하라.

??? success "연습문제 3 풀이"
    ```python
    import torch, numpy as np
    import torch.nn as nn
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    n, p = 100, 20
    X = torch.FloatTensor(np.random.randn(n, p))
    true_w = np.zeros(p); true_w[:5] = [3, -2, 1.5, -1, 2]
    y = (X @ torch.FloatTensor(true_w) + torch.randn(n) * 0.5).reshape(-1, 1)
    
    model = nn.Linear(p, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(500):
        loss = nn.MSELoss()(model(X), y) + 0.1 * torch.sum(torch.abs(model.weight))
        opt.zero_grad(); loss.backward(); opt.step()
    
    w = model.weight.detach().numpy().flatten()
    plt.bar(range(p), w)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Feature Index')
    plt.ylabel('Weight')
    plt.title('L1 Regularized Weights')
    plt.show()
    # 특징 0-4는 가중치가 크고, 특징 5-19는 0에 가까워야 한다.
    ```

## 정리하며

**다룬 것** — 정칙화

L2 정칙화는 손실을 $L = \text{MSE} + \lambda \sum w_i^2$로 바꾸어 가중치 제곱합에 벌점을 준다.

앞의 연습문제 3개로 직접 확인할 수 있다.
