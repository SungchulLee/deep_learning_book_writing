# 결정 경계
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 로지스틱 회귀의 결정 경계 방정식 유도하기
- 가중치 벡터와 절편의 기하학적 역할 이해하기
- BCE와 BCEWithLogitsLoss, 그리고 그 수치적 안정성 비교하기
- 결정 경계 시각화와 BCE 손실 함수를 바닥부터 구현하기

---

## 결정 경계

### 경계는 어디에 있는가

**결정 경계**는 $P(Y=1|\mathbf{x}) = 0.5$인 점들의 집합이며, 이는 로그 승산이 0일 때 일어난다.

$$
\sigma(\mathbf{x}^\top\boldsymbol{\beta}) = 0.5 \implies \mathbf{x}^\top\boldsymbol{\beta} = 0
$$

특징이 둘이면 이는 직선을 정의한다.

$$
\beta_0 + \beta_1 x_1 + \beta_2 x_2 = 0 \implies x_2 = -\frac{\beta_0}{\beta_2} - \frac{\beta_1}{\beta_2}x_1
$$

일반적으로 $d$차원에서 결정 경계는 $(d-1)$차원 **초평면**이다.

### 기하학적 해석

가중치 벡터 $\boldsymbol{\beta}_{1:d} = [\beta_1, \ldots, \beta_d]$과 절편 $\beta_0$이 결정 경계를 완전히 결정한다.

- 가중치 벡터 $\boldsymbol{\beta}_{1:d}$은 결정 경계에 **수직**(법선)이다
- 절편 $\beta_0$은 원점으로부터의 **치우침**을 조절한다
- 크기 $\|\boldsymbol{\beta}\|$은 확률 전이가 얼마나 **가파른지**를 조절한다

**$\boldsymbol{\beta}_{1:d}$은 왜 경계에 수직인가?** 경계 위의 두 점 $\mathbf{x}^{(a)}$과 $\mathbf{x}^{(b)}$을 생각하자. 둘 다 $\beta_0 + \boldsymbol{\beta}_{1:d}^\top \mathbf{x} = 0$을 만족하므로 다음이 성립한다.

$$
\boldsymbol{\beta}_{1:d}^\top (\mathbf{x}^{(a)} - \mathbf{x}^{(b)}) = 0
$$

이는 $\boldsymbol{\beta}_{1:d}$이 경계 안에 놓인 어떤 벡터와도 직교한다는 뜻이며, 따라서 법선 벡터임이 확인된다.

### 확률 등고선

문턱값 0.5에서의 결정 경계는 여러 등위집합 중 하나일 뿐이다. 더 일반적으로 임의의 목표 확률 $p^*$에 대해 다음이 성립한다.

$$
P(Y=1|\mathbf{x}) = p^* \iff \mathbf{x}^\top\boldsymbol{\beta} = \log\frac{p^*}{1-p^*}
$$

따라서 확률이 일정한 등고선들은 서로 평행한 초평면이며, 각각은 $\boldsymbol{\beta}_{1:d}$ 방향으로 결정 경계로부터 $\frac{1}{\|\boldsymbol{\beta}_{1:d}\|}\log\frac{p^*}{1-p^*}$만큼 떨어져 있다.

### 계수 크기의 효과

경계에 수직인 방향에서의 확률 전이의 가파름은 $\|\boldsymbol{\beta}_{1:d}\|$이 조절한다.

| $\|\boldsymbol{\beta}\|$ | 전이 폭 | 모델의 거동 |
|:---:|---|---|
| 작음 | 넓고 완만함 | 경계 근처에서 불확실한 예측 |
| 큼 | 좁고 날카로움 | 확신에 찬 예측, "단단한" 경계 |

이는 [정칙화된 로지스틱 회귀](regularized.md)에 중요한 함의를 갖는다. 거기서는 $\|\boldsymbol{\beta}\|$에 벌점을 주어 지나치게 확신에 찬 예측을 막는다.

---

## BCE 손실 함수

### 로그가능도에서 BCE로

[이진 분류](binary_classification.md)에서 보았듯이 음의 평균 로그가능도가 **이진 교차 엔트로피(BCE)** 손실을 준다.

$$
\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

여기서 $p_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta})$이다.

### 개별 표본에서 손실의 거동

참 이름표가 $y$이고 예측 확률이 $p$인 표본 하나에 대해 다음과 같다.

| $y=1$일 때 | 손실 $= -\log(p)$ |
|:---|:---|
| $p \to 1$ (맞고 확신함) | 손실 $\to 0$ |
| $p \to 0$ (틀렸는데 확신함) | 손실 $\to +\infty$ |

| $y=0$일 때 | 손실 $= -\log(1-p)$ |
|:---|:---|
| $p \to 0$ (맞고 확신함) | 손실 $\to 0$ |
| $p \to 1$ (틀렸는데 확신함) | 손실 $\to +\infty$ |

이 손실은 확신에 차 있으면서 틀린 예측에 불확실한 예측보다 **지수적으로** 더 큰 벌점을 준다. 학습에 바람직한 성질이다.

---

## BCE 대 BCEWithLogitsLoss

### 수치적 안정성 문제

시그모이드를 먼저 적용한 뒤 로그를 취해 BCE를 계산하면 수치적 문제가 생긴다.

| 상황 | 문제 |
|----------|---------|
| $z \gg 0$ | $\sigma(z) \approx 1$이므로 $\log(1-\sigma(z)) \to -\infty$ |
| $z \ll 0$ | $\sigma(z) \approx 0$이므로 $\log(\sigma(z)) \to -\infty$ |

float32 산술에서도 $|z| \gtrsim 90$이면 $\sigma(z)$이 정확히 0.0이나 1.0으로 포화하여 $\log(\sigma(z))$이 `-inf`를 낸다.

### 안정한 정식화

**BCEWithLogitsLoss**은 시그모이드와 BCE를 수치적으로 안정한 하나의 연산으로 합친다. 로짓이 $z$이고 이름표가 $y$인 표본 하나의 손실에서 출발한다.

$$
\ell = -[y \log \sigma(z) + (1-y) \log(1-\sigma(z))]
$$

$\log \sigma(z) = z - \log(1+e^z)$과 $\log(1-\sigma(z)) = -\log(1+e^z)$을 대입하면 다음과 같다.

$$
\ell = -yz + \log(1+e^z)
$$

수치적 안정성을 위해 항등식 $\log(1+e^z) = \max(z, 0) + \log(1+e^{-|z|})$을 써서 다시 쓴다.

$$
\boxed{\text{BCE}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})}
$$

이렇게 하면 $\sigma(z)$을 명시적으로 계산하지 않으며 모든 $z$에 대해 안정하다.

### 어떤 손실을 언제 쓸 것인가

| 손실 함수 | 모델 출력 | 쓰는 때 |
|---------------|-------------|-------------|
| `nn.BCELoss()` | 확률 (시그모이드 이후) | 다른 계산에 확률이 필요할 때 |
| `nn.BCEWithLogitsLoss()` | 로짓 (시그모이드 이전) | **학습에는 이쪽을 권장한다** |

### BCEWithLogitsLoss의 장점

1. **수치적 안정성**: 극단적인 로짓도 넘침이나 아랫넘침 없이 다룬다
2. **계산 효율**: 시그모이드와 BCE를 하나의 융합 연산으로 합친다
3. **경사의 흐름**: 아주 확신에 찬 예측에서도 더 나은 경사를 준다

---

## PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("DECISION BOUNDARY AND BCE LOSS ANALYSIS")
print("=" * 70)

# ============================================================================
# 1부: BCE 직접 구현과 안정성
# ============================================================================

print("\n1. Comparing BCE implementations")
print("-" * 50)

def bce_manual(predictions, targets, eps=1e-12):
    """직접 구현한 BCE: -[y * log(p) + (1-y) * log(1-p)]."""
    predictions = torch.clamp(predictions, eps, 1 - eps)
    loss = -(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions))
    return loss.mean()

def bce_with_logits_manual(logits, targets):
    """수치적으로 안정한 BCE: max(z, 0) - z*y + log(1 + exp(-|z|))."""
    max_val = torch.clamp(logits, min=0)
    loss = max_val - logits * targets + torch.log(1 + torch.exp(-torch.abs(logits)))
    return loss.mean()

logits = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
targets = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
probs = torch.sigmoid(logits)

manual_bce = bce_manual(probs, targets)
manual_bce_logits = bce_with_logits_manual(logits, targets)
pytorch_bce = F.binary_cross_entropy(probs, targets)
pytorch_bce_logits = F.binary_cross_entropy_with_logits(logits, targets)

print(f"Manual BCE:              {manual_bce.item():.6f}")
print(f"Manual BCEWithLogits:    {manual_bce_logits.item():.6f}")
print(f"PyTorch BCE:             {pytorch_bce.item():.6f}")
print(f"PyTorch BCEWithLogits:   {pytorch_bce_logits.item():.6f}")

# ============================================================================
# 2부: 수치적 안정성 분석
# ============================================================================

print("\n" + "=" * 70)
print("NUMERICAL STABILITY COMPARISON")
print("=" * 70)

extreme_logits = torch.tensor([-100.0, -50.0, -10.0, 0.0, 10.0, 50.0, 100.0])
targets_ones = torch.ones_like(extreme_logits)

print("\nExtreme logits (y=1):")
print("-" * 55)
print(f"{'Logit':>10} {'BCEWithLogits':>15} {'BCE+Sigmoid':>15}")
print("-" * 55)

for z in extreme_logits:
    z_t = z.unsqueeze(0)
    y_t = torch.ones(1)

    stable_loss = F.binary_cross_entropy_with_logits(z_t, y_t)

    p = torch.sigmoid(z_t)
    p_clipped = torch.clamp(p, 1e-7, 1 - 1e-7)
    unstable_loss = F.binary_cross_entropy(p_clipped, y_t)

    print(f"{z.item():>10.1f} {stable_loss.item():>15.6f} {unstable_loss.item():>15.6f}")

# ============================================================================
# 3부: 결정 경계 시각화
# ============================================================================

print("\n" + "=" * 70)
print("DECISION BOUNDARY VISUALIZATION")
print("=" * 70)

# 2차원 분류 데이터를 생성한다
X, y = make_classification(
    n_samples=500, n_features=2, n_redundant=0,
    n_informative=2, random_state=42, n_clusters_per_class=1,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegression(2)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(500):
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 학습된 매개변수를 꺼낸다
learned_weights = model.linear.weight.data.numpy().flatten()
learned_bias = model.linear.bias.data.numpy()[0]

print(f"\nLearned coefficients:")
print(f"  β₀ (intercept): {learned_bias:.3f}")
print(f"  β₁: {learned_weights[0]:.3f}")
print(f"  β₂: {learned_weights[1]:.3f}")
print(f"\nDecision boundary equation:")
print(
    f"  x₂ = {-learned_bias / learned_weights[1]:.3f} "
    f"+ {-learned_weights[0] / learned_weights[1]:.3f} x₁"
)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 그림 1: 예측의 함수로 본 BCE
ax1 = axes[0]
p_range = torch.linspace(0.01, 0.99, 99)
loss_y1 = -torch.log(p_range)
loss_y0 = -torch.log(1 - p_range)

ax1.plot(p_range.numpy(), loss_y1.numpy(), "b-", linewidth=2, label="y=1: -log(p)")
ax1.plot(p_range.numpy(), loss_y0.numpy(), "r-", linewidth=2, label="y=0: -log(1-p)")
ax1.set_xlabel("Predicted Probability p", fontsize=11)
ax1.set_ylabel("BCE Loss", fontsize=11)
ax1.set_title("BCE Loss vs Predicted Probability", fontsize=12, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 5])

# 그림 2: 확률 등고선과 함께 본 결정 경계
ax2 = axes[1]
X_train_np = X_train.numpy()
y_train_np = y_train.numpy().flatten()

xx, yy = np.meshgrid(
    np.linspace(X_train_np[:, 0].min() - 0.5, X_train_np[:, 0].max() + 0.5, 100),
    np.linspace(X_train_np[:, 1].min() - 0.5, X_train_np[:, 1].max() + 0.5, 100),
)
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

with torch.no_grad():
    Z = model(grid).reshape(xx.shape).numpy()

contour = ax2.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.8)
fig.colorbar(contour, ax=ax2, label="P(Y=1|x)")

# p=0.5에서의 결정 경계
ax2.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2, linestyles="--")

ax2.scatter(
    X_train_np[y_train_np == 0, 0],
    X_train_np[y_train_np == 0, 1],
    c="blue", marker="o", label="Class 0", alpha=0.6, edgecolors="w",
)
ax2.scatter(
    X_train_np[y_train_np == 1, 0],
    X_train_np[y_train_np == 1, 1],
    c="red", marker="o", label="Class 1", alpha=0.6, edgecolors="w",
)

# 가중치 벡터를 그린다 (결정 경계에 수직)
scale = 0.5
ax2.arrow(
    0, 0, learned_weights[0] * scale, learned_weights[1] * scale,
    head_width=0.1, head_length=0.05, fc="green", ec="green", linewidth=2,
)
ax2.text(
    learned_weights[0] * scale + 0.1,
    learned_weights[1] * scale + 0.1,
    "β", fontsize=12, fontweight="bold", color="green",
)

ax2.set_xlabel("Feature 1 (standardized)", fontsize=11)
ax2.set_ylabel("Feature 2 (standardized)", fontsize=11)
ax2.set_title("Decision Boundary and Probability Contours", fontsize=12, fontweight="bold")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

# 그림 3: 극단적인 값에서 안정한 BCE와 불안정한 BCE
ax3 = axes[2]
extreme_z = torch.linspace(-20, 20, 100)
stable_losses = []
unstable_losses = []

for z in extreme_z:
    z_t = z.unsqueeze(0)
    y_t = torch.ones(1)
    stable = F.binary_cross_entropy_with_logits(z_t, y_t).item()
    p = torch.sigmoid(z_t)
    p_clipped = torch.clamp(p, 1e-7, 1 - 1e-7)
    unstable = F.binary_cross_entropy(p_clipped, y_t).item()
    stable_losses.append(stable)
    unstable_losses.append(unstable)

ax3.plot(extreme_z.numpy(), stable_losses, "g-", linewidth=2, label="BCEWithLogitsLoss (stable)")
ax3.plot(extreme_z.numpy(), unstable_losses, "r--", linewidth=2, label="BCE+Sigmoid (clipped)")
ax3.set_xlabel("Logit z", fontsize=11)
ax3.set_ylabel("Loss (y=1)", fontsize=11)
ax3.set_title("Numerical Stability Comparison", fontsize=12, fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("decision_boundary_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# 4부: 학습 비교 — BCELoss 대 BCEWithLogitsLoss
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING: BCELoss vs BCEWithLogitsLoss")
print("=" * 70)

X_large, y_large = make_classification(n_samples=1000, n_features=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_large, y_large, test_size=0.2, random_state=42)
sc = StandardScaler()
X_tr = torch.FloatTensor(sc.fit_transform(X_tr))
y_tr = torch.FloatTensor(y_tr).reshape(-1, 1)

class ModelWithSigmoid(nn.Module):
    """확률을 반환한다 (BCELoss와 함께 쓴다)."""
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class ModelWithLogits(nn.Module):
    """로짓을 반환한다 (BCEWithLogitsLoss와 함께 쓴다)."""
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

def train_model_simple(model, criterion, X_train, y_train, num_epochs=100, lr=0.1):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = []
    for epoch in range(num_epochs):
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history

torch.manual_seed(42)
model1 = ModelWithSigmoid(20)
torch.manual_seed(42)
model2 = ModelWithLogits(20)

history1 = train_model_simple(model1, nn.BCELoss(), X_tr, y_tr)
history2 = train_model_simple(model2, nn.BCEWithLogitsLoss(), X_tr, y_tr)

print(f"\nFinal loss (BCELoss):           {history1[-1]:.6f}")
print(f"Final loss (BCEWithLogitsLoss): {history2[-1]:.6f}")

print("\n✓ Both loss functions converge to the same optimum.")
print("  BCEWithLogitsLoss is preferred for numerical stability.")
```

---

## 요약

| 개념 | 공식 | 핵심 |
|---------|---------|-----------|
| 결정 경계 | $\mathbf{x}^\top\boldsymbol{\beta} = 0$ | $P(Y=1) = 0.5$인 곳 |
| 가중치 벡터의 역할 | $\boldsymbol{\beta}_{1:d} \perp$ 경계 | 결정면의 법선 |
| 절편의 역할 | $\beta_0$ | 원점으로부터의 치우침 |
| 가파름 | $\|\boldsymbol{\beta}\|$ | 전이의 날카로움을 조절한다 |
| BCE | $-[y\log p + (1-y)\log(1-p)]$ | 음의 로그가능도 |
| 안정한 BCE | $\max(z,0) - yz + \log(1+e^{-\|z\|})$ | 수치적으로 안정한 형태 |
| PyTorch | `BCEWithLogitsLoss` | **학습에는 언제나 이쪽을 쓰라** |

결정 경계는 가중치 벡터와 절편이 그 기하를 온전히 결정하는 초평면이다. 이 기하를 이해하면 로지스틱 회귀가 특징 공간을 어떻게 나누는지, 그리고 $\|\boldsymbol{\beta}\|$을 제약하는 정칙화가 왜 더 매끄럽고 더 잘 일반화되는 경계를 만드는지에 대한 직관을 얻을 수 있다.

## 연습문제

**연습문제 1.**
원점에서 결정 경계 $\mathbf{x}^\top\boldsymbol{\beta} = 0$까지의 수직 거리가 $\frac{|\beta_0|}{\|\boldsymbol{\beta}_{1:d}\|}$임을 증명하라.

??? success "연습문제 1 풀이"
    결정 경계는 초평면 $\beta_0 + \beta_1 x_1 + \cdots + \beta_d x_d = 0$, 즉 $\boldsymbol{\beta}_{1:d}^\top \mathbf{x} = -\beta_0$이다. 원점에서 초평면 $\mathbf{w}^\top\mathbf{x} = b$까지의 거리는 $|b|/\|\mathbf{w}\|$이다.

    따라서 거리는 $|-\beta_0|/\|\boldsymbol{\beta}_{1:d}\| = |\beta_0|/\|\boldsymbol{\beta}_{1:d}\|$이다.

    L2 정칙화는 $\|\boldsymbol{\beta}\|^2$에 벌점을 주는데, 이는 이 거리를 늘리는 경향이 있고(경계를 데이터 쪽으로 밀며) 동시에 전이를 더 매끄럽게 만든다. $\square$

---

**연습문제 2.**
안정한 BCE 공식 $\max(z, 0) - zy + \log(1+e^{-|z|})$이 $-[y\log\sigma(z) + (1-y)\log(1-\sigma(z))]$과 같음을 보여라.

??? success "연습문제 2 풀이"
    $\text{BCE} = -y\log\sigma(z) - (1-y)\log(1-\sigma(z))$에서 출발한다.

    $\log\sigma(z) = -\log(1+e^{-z})$이고 $\log(1-\sigma(z)) = -z - \log(1+e^{-z})$임에 유의하라.

    따라서 $\text{BCE} = y\log(1+e^{-z}) + (1-y)(z + \log(1+e^{-z})) = z - zy + \log(1+e^{-z})$이다.

    수치적 안정성을 위해 $z = |z|\cdot\text{sign}(z)$으로 쓰고 $\log(1+e^{-z}) = \max(z,0) - z + \log(1+e^{-|z|})$을 쓴다($z \geq 0$인 경우와 $z < 0$인 경우로 나누어 확인할 수 있다).

    대입하면 $\text{BCE} = \max(z,0) - zy + \log(1+e^{-|z|})$이다. $\square$

---

**연습문제 3.**
개별 계수 $\beta_0, \beta_1, \beta_2$이 변할 때 결정 경계가 어떻게 회전하고 이동하는지 보여주는 시각화를 만들어라.

??? success "연습문제 3 풀이"
    ```python
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.linspace(-3, 3, 100)

    # beta_0을 바꾼다 (절편이 경계를 이동시킨다)
    for b0 in [-2, -1, 0, 1, 2]:
        axes[0].plot(x, -(1.0*x + b0)/1.0, label=f'b0={b0}')
    axes[0].set_title('Varying intercept')

    # beta_1을 바꾼다 (경계를 회전시킨다)
    for b1 in [0.5, 1.0, 2.0, 4.0]:
        axes[1].plot(x, -(b1*x)/1.0, label=f'b1={b1}')
    axes[1].set_title('Varying beta_1')

    # beta_2를 바꾼다 (기울기를 바꾼다)
    for b2 in [0.5, 1.0, 2.0, 4.0]:
        axes[2].plot(x, -(1.0*x)/b2, label=f'b2={b2}')
    axes[2].set_title('Varying beta_2')

    for ax in axes:
        ax.legend(); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    plt.tight_layout()
    ```

---

**연습문제 4.**
`pos_weight`을 쓰는 `BCEWithLogitsLoss`을 구현하고, 불균형 데이터(음성 90%, 양성 10%)에서 결정 경계에 미치는 효과를 보여라.

??? success "연습문제 4 풀이"
    ```python
    import torch
    import torch.nn as nn

    # 불균형 데이터셋: 클래스 0이 90%, 클래스 1이 10%
    torch.manual_seed(0)
    X = torch.randn(1000, 2)
    y = (torch.rand(1000) < 0.1).float()

    for pw in [1.0, 5.0, 9.0]:
        model = nn.Linear(2, 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(1000):
            opt.zero_grad()
            criterion(model(X).squeeze(), y).backward()
            opt.step()
        # pos_weight을 키우면 경계가 다수 클래스 쪽으로 이동하여
        # 정밀도를 잃는 대신 재현율이 높아진다
    ```
