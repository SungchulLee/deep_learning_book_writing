# 정규화

정규화(regularization)는 로스 함수에 큰 파라메터 값을 억제하는 패널티 항을 더하여, 특히 데이터 양에 비해 파라메터가 많은 모델의 과적합을 막는 데 도움을 준다. L2 정규화(릿지)는 모든 피처를 유지하면서 웨이트를 0 쪽으로 줄이는 반면, L1 정규화(라쏘)는 일부 웨이트를 정확히 0으로 만들어 사실상 피처 선택을 수행한다. 이 튜토리얼은 소수의 피처만 실제로 유용한 고차원 데이터셋에서 두 방법을 비교한다.

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
    - 최적화기의 웨이트 줄이기
    - L1으로 하는 피처 고르기
    - 정칙화 파라메터 손보기

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
# 1부: 피처가 많은 데이터 생성
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE HIGH-DIMENSIONAL DATA")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)

# 소수의 피처만 실제로 중요한 데이터를 생성한다
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
  로스 = MSE + λ * Σ(w²)
  
  - 큰 웨이트에 벌을 준다
  - 웨이트가 0 쪽으로 오그라든다
  - 피처를 모두 남긴다(고르지 않는다)
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
  로스 = MSE + λ * Σ(|w|)
  
  - 웨이트의 절댓값에 벌을 준다
  - 어떤 웨이트는 딱 0으로 만든다
  - 피처를 골라 준다
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
# 5부: 웨이트 크기 시각화
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
plt.savefig('08_regularization_weights.png', dpi=100)
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
plt.savefig('08_regularization_comparison.png', dpi=100)
print("Saved comparison visualization")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
정칙화 간추림:

L2 정칙화(능선):
✓ 모든 웨이트를 0 쪽으로 오그린다
✓ 피처를 모두 남긴다
✓ 짜기 쉽다(최적화기의 weight_decay)
✓ 서로 얽힌 피처에 좋다
✗ 피처를 골라 주지 않는다

L1 정칙화(라쏘):
✓ 어떤 웨이트는 딱 0으로 만든다
✓ 피처를 절로 골라 준다
✓ 성긴 모델(살아 있는 피처가 적다)
✗ 직접 짜야 한다
✗ 서로 얽힌 피처에서는 흔들린다

언제 쓸까:
- L2: 모든 피처가 종요로울 수 있을 때
- L1: 피처를 골라내고 싶을 때
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

??? note "전체 출력 (98줄)"

    ```
    ======================================================================
    REGULARIZATION: L1 AND L2
    ======================================================================

    ======================================================================
    PART 1: GENERATE HIGH-DIMENSIONAL DATA
    ======================================================================
    Dataset:
      Samples: 100
      Total features: 20
      Informative features: 5

    True important weights (first 5 features):
      [ 3.  -2.   1.5 -1.   2. ]

    ======================================================================
    PART 2: BASELINE MODEL (NO REGULARIZATION)
    ======================================================================
    Baseline model trained
      Number of near-zero weights (|w| < 0.1): 13/20

    ======================================================================
    PART 3: L2 REGULARIZATION (RIDGE)
    ======================================================================

    L2 정칙화(능선):
      로스 = MSE + λ * Σ(w²)
      
      - 큰 웨이트에 벌을 준다
      - 웨이트가 0 쪽으로 오그라든다
      - 피처를 모두 남긴다(고르지 않는다)
      - 최적화기의 'weight_decay'으로 짠다

    λ=0.000 : Mean |weight|=0.4985, Max |weight|=2.9159
    λ=0.001 : Mean |weight|=0.4975, Max |weight|=2.9160
    λ=0.010 : Mean |weight|=0.4956, Max |weight|=2.8989
    λ=0.100 : Mean |weight|=0.4770, Max |weight|=2.7422
    λ=1.000 : Mean |weight|=0.3672, Max |weight|=1.8283

    ======================================================================
    PART 4: L1 REGULARIZATION (LASSO)
    ======================================================================

    L1 정칙화(라쏘):
      로스 = MSE + λ * Σ(|w|)
      
      - 웨이트의 절댓값에 벌을 준다
      - 어떤 웨이트는 딱 0으로 만든다
      - 피처를 골라 준다
      - PyTorch에서는 직접 짜야 한다

    λ=0.000 : Nearly zero weights=1/20, Mean |weight|=0.4981
    λ=0.001 : Nearly zero weights=1/20, Mean |weight|=0.4966
    λ=0.010 : Nearly zero weights=4/20, Mean |weight|=0.4901
    λ=0.100 : Nearly zero weights=13/20, Mean |weight|=0.4617
    λ=1.000 : Nearly zero weights=13/20, Mean |weight|=0.3490

    ======================================================================
    PART 5: VISUALIZE REGULARIZATION EFFECTS
    ======================================================================
    Saved weight visualization
    Saved comparison visualization

    ======================================================================
    SUMMARY
    ======================================================================

    정칙화 간추림:

    L2 정칙화(능선):
    ✓ 모든 웨이트를 0 쪽으로 오그린다
    ✓ 피처를 모두 남긴다
    ✓ 짜기 쉽다(최적화기의 weight_decay)
    ✓ 서로 얽힌 피처에 좋다
    ✗ 피처를 골라 주지 않는다

    L1 정칙화(라쏘):
    ✓ 어떤 웨이트는 딱 0으로 만든다
    ✓ 피처를 절로 골라 준다
    ✓ 성긴 모델(살아 있는 피처가 적다)
    ✗ 직접 짜야 한다
    ✗ 서로 얽힌 피처에서는 흔들린다

    언제 쓸까:
    - L2: 모든 피처가 종요로울 수 있을 때
    - L1: 피처를 골라내고 싶을 때
    - 둘 다(엘라스틱넷): 이점을 아우른다

    PyTorch에서 짜기:
      # L2: 내장 기능
      optimizer = torch.optim.SGD(params, lr=0.01, weight_decay=0.01)
      
      # L1: 직접 구현
      l1_loss = lambda_l1 * torch.sum(torch.abs(model.weight))
      total_loss = mse_loss + l1_loss

    다음: 튜토리얼 09 - DataLoader으로 하는 작은 배치 익히기!

    ```


## 2. 논의

### 목적함수의 계수 맞추기

L2 목적을 $\text{MSE}+\lambda\|w\|_2^2$로 쓰면 패널티 경사는 $2\lambda w$다. 반면 표준 SGD의 `weight_decay=\gamma`는 경사에 $\gamma w$를 더하므로 이 표기에서는 $\gamma=2\lambda$에 대응한다. 로스의 합·평균과 1/2 계수에 따라 대응값이 달라진다. AdamW의 분리된 감쇠까지 모두 같은 L2 목적이라고 해석하면 안 된다.

### 희소 목적과 수치 알고리즘

L1 목적의 최적해는 일부 계수를 정확히 0으로 만들 수 있다. 하지만 `abs()` 항을 더해 보통의 SGD나 Adam으로 유한 번 학습하면, 계수가 0 근처에서 진동하거나 작은 비영 값으로 남을 수 있다. 정확한 희소 해에는 좌표 하강법이나 근접 경사법 같은 알고리즘이 적합하다. 출력의 “0에 가까움” 기준과 수학적인 0을 구분한다.

### 정규화 강도 선택

피처의 척도를 맞추고 같은 검증 분할에서 패널티 강도를 선택한다. L1이 항상 유용한 피처만 고르거나 L2가 모든 계수를 같은 비율로 줄이는 것은 아니다. 피처 간 상관, 표본 수와 잡음이 결과에 영향을 준다. 엘라스틱 넷은 두 패널티를 결합하며 구현에서 실제 목적과 감쇠 규칙을 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
L2 강도를 높일 때 전체 웨이트 크기와 개별 계수가 어떻게 달라질 수 있는가?

</div>

??? success "연습문제 1 풀이"
    최적해에서 전체 L2 노름은 정규화가 강해질수록 줄어드는 경향을 보이며, 같은 목적의 정확한 해 경로에서는 이를 비교할 수 있다. 하지만 상관된 피처의 개별 계수가 모두 같은 비율로 줄거나 단조 감소하는 것은 아니다. 유한 반복의 수치 해에는 수렴 정도도 영향을 준다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
L1과 L2 패널티를 결합하여 엘라스틱 넷 정규화를 구현하라. $L = \text{MSE} + \alpha \sum |w_i| + \beta \sum w_i^2$. 모델을 학습시키고 0에 가까운 웨이트의 개수를 순수 L1 및 순수 L2와 비교하라.

</div>

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

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
20개 중 5개만 유용한 피처를 가진 데이터셋에서 $\lambda = 0.1$인 L1 정규화로 학습한 웨이트를 막대그래프로 그리고, 0이 아닌 웨이트가 처음 5개 피처에 대응하는지 확인하라.

</div>

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
    # 피처 0-4는 웨이트가 크고, 피처 5-19는 0에 가까워야 한다.
    ```

## 정리하며

L1·L2의 목적, 패널티 계수와 최적화 알고리즘을 구분한다. 희소성은 패널티의 성질뿐 아니라 실제로 사용한 수치 해법과 수렴 정도에도 달려 있다.
