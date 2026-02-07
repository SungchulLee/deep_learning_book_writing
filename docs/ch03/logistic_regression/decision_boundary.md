# Decision Boundary

## Learning Objectives

By the end of this section, you will be able to:

- Derive the decision boundary equation for logistic regression
- Understand the geometric role of the weight vector and intercept
- Compare BCE, BCEWithLogitsLoss, and their numerical stability properties
- Implement decision boundary visualization and BCE loss functions from scratch

---

## The Decision Boundary

### Where the Boundary Lies

The **decision boundary** is the set of points where $P(Y=1|\mathbf{x}) = 0.5$, which occurs when the log-odds equal zero:

$$
\sigma(\mathbf{x}^\top\boldsymbol{\beta}) = 0.5 \implies \mathbf{x}^\top\boldsymbol{\beta} = 0
$$

For two features, this defines a line:

$$
\beta_0 + \beta_1 x_1 + \beta_2 x_2 = 0 \implies x_2 = -\frac{\beta_0}{\beta_2} - \frac{\beta_1}{\beta_2}x_1
$$

In general $d$ dimensions, the decision boundary is a $(d-1)$-dimensional **hyperplane**.

### Geometric Interpretation

The weight vector $\boldsymbol{\beta}_{1:d} = [\beta_1, \ldots, \beta_d]$ and the intercept $\beta_0$ fully determine the decision boundary:

- The weight vector $\boldsymbol{\beta}_{1:d}$ is **perpendicular** (normal) to the decision boundary
- The intercept $\beta_0$ controls the **offset** from the origin
- The magnitude $\|\boldsymbol{\beta}\|$ controls how **steep** the probability transition is

**Why is $\boldsymbol{\beta}_{1:d}$ perpendicular to the boundary?** Consider two points $\mathbf{x}^{(a)}$ and $\mathbf{x}^{(b)}$ on the boundary. Both satisfy $\beta_0 + \boldsymbol{\beta}_{1:d}^\top \mathbf{x} = 0$, so:

$$
\boldsymbol{\beta}_{1:d}^\top (\mathbf{x}^{(a)} - \mathbf{x}^{(b)}) = 0
$$

This means $\boldsymbol{\beta}_{1:d}$ is orthogonal to any vector lying in the boundary, confirming it is the normal vector.

### Probability Contours

The decision boundary at threshold 0.5 is just one level set. More generally, for any target probability $p^*$:

$$
P(Y=1|\mathbf{x}) = p^* \iff \mathbf{x}^\top\boldsymbol{\beta} = \log\frac{p^*}{1-p^*}
$$

So contours of constant probability are parallel hyperplanes, each offset from the decision boundary by $\frac{1}{\|\boldsymbol{\beta}_{1:d}\|}\log\frac{p^*}{1-p^*}$ in the direction of $\boldsymbol{\beta}_{1:d}$.

### Effect of Coefficient Magnitude

The steepness of the probability transition perpendicular to the boundary is controlled by $\|\boldsymbol{\beta}_{1:d}\|$:

| $\|\boldsymbol{\beta}\|$ | Transition width | Model behavior |
|:---:|---|---|
| Small | Wide, gradual | Uncertain predictions near boundary |
| Large | Narrow, sharp | Confident predictions, "hard" boundary |

This has important implications for [Regularized Logistic Regression](regularized.md), where penalizing $\|\boldsymbol{\beta}\|$ prevents overly confident predictions.

---

## The BCE Loss Function

### From Log-Likelihood to BCE

Recall from [Binary Classification](binary_classification.md) that the negative average log-likelihood gives us the **Binary Cross-Entropy (BCE)** loss:

$$
\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

where $p_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta})$.

### Loss Behavior for Individual Samples

For a single sample with true label $y$ and predicted probability $p$:

| When $y=1$ | Loss $= -\log(p)$ |
|:---|:---|
| $p \to 1$ (correct, confident) | Loss $\to 0$ |
| $p \to 0$ (wrong, confident) | Loss $\to +\infty$ |

| When $y=0$ | Loss $= -\log(1-p)$ |
|:---|:---|
| $p \to 0$ (correct, confident) | Loss $\to 0$ |
| $p \to 1$ (wrong, confident) | Loss $\to +\infty$ |

The loss penalizes confident wrong predictions **exponentially** more than uncertain predictions — a desirable property for training.

---

## BCE vs BCEWithLogitsLoss

### The Numerical Stability Problem

Computing BCE by first applying sigmoid and then taking logarithms introduces numerical issues:

| Scenario | Problem |
|----------|---------|
| $z \gg 0$ | $\sigma(z) \approx 1$, so $\log(1-\sigma(z)) \to -\infty$ |
| $z \ll 0$ | $\sigma(z) \approx 0$, so $\log(\sigma(z)) \to -\infty$ |

Even with float32 arithmetic, $\sigma(z)$ can saturate to exactly 0.0 or 1.0 for $|z| \gtrsim 90$, making $\log(\sigma(z))$ produce `-inf`.

### The Stable Formulation

**BCEWithLogitsLoss** combines sigmoid and BCE into a single numerically stable operation. Starting from the loss for a single sample with logit $z$ and label $y$:

$$
\ell = -[y \log \sigma(z) + (1-y) \log(1-\sigma(z))]
$$

Substituting $\log \sigma(z) = z - \log(1+e^z)$ and $\log(1-\sigma(z)) = -\log(1+e^z)$:

$$
\ell = -yz + \log(1+e^z)
$$

For numerical stability, we rewrite using the identity $\log(1+e^z) = \max(z, 0) + \log(1+e^{-|z|})$:

$$
\boxed{\text{BCE}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})}
$$

This avoids computing $\sigma(z)$ explicitly and remains stable for all $z$.

### When to Use Each Loss

| Loss Function | Model Output | When to Use |
|---------------|-------------|-------------|
| `nn.BCELoss()` | Probabilities (after sigmoid) | When you need probabilities for other computations |
| `nn.BCEWithLogitsLoss()` | Logits (before sigmoid) | **Recommended for training** |

### BCEWithLogitsLoss Advantages

1. **Numerical Stability**: Handles extreme logits without overflow/underflow
2. **Computational Efficiency**: Combines sigmoid + BCE in one fused operation
3. **Gradient Flow**: Better gradients for very confident predictions

---

## PyTorch Implementation

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
# Part 1: Manual BCE Implementation and Stability
# ============================================================================

print("\n1. Comparing BCE implementations")
print("-" * 50)


def bce_manual(predictions, targets, eps=1e-12):
    """Manual BCE: -[y * log(p) + (1-y) * log(1-p)]."""
    predictions = torch.clamp(predictions, eps, 1 - eps)
    loss = -(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions))
    return loss.mean()


def bce_with_logits_manual(logits, targets):
    """Numerically stable BCE: max(z, 0) - z*y + log(1 + exp(-|z|))."""
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
# Part 2: Numerical Stability Analysis
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
# Part 3: Decision Boundary Visualization
# ============================================================================

print("\n" + "=" * 70)
print("DECISION BOUNDARY VISUALIZATION")
print("=" * 70)

# Generate 2D classification data
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

# Extract learned parameters
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

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: BCE as function of prediction
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

# Plot 2: Decision boundary with probability contours
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

# Decision boundary at p=0.5
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

# Draw weight vector (perpendicular to decision boundary)
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

# Plot 3: Stable vs unstable BCE for extreme values
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
# Part 4: Training Comparison — BCELoss vs BCEWithLogitsLoss
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
    """Returns probabilities (use with BCELoss)."""
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class ModelWithLogits(nn.Module):
    """Returns logits (use with BCEWithLogitsLoss)."""
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

## Exercises

### Conceptual

1. **Prove** that the perpendicular distance from the origin to the decision boundary is $\frac{|\beta_0|}{\|\boldsymbol{\beta}_{1:d}\|}$. How does regularization affect this distance?

2. **Explain** geometrically what happens to the decision boundary when the intercept $\beta_0 = 0$.

3. **Show** that the stable BCE formula $\max(z, 0) - zy + \log(1+e^{-|z|})$ is equivalent to $-[y\log\sigma(z) + (1-y)\log(1-\sigma(z))]$ for all $z$ and $y \in \{0, 1\}$.

### Computational

4. Implement BCEWithLogitsLoss with `pos_weight` for handling class imbalance. Verify that setting `pos_weight > 1` increases the loss for positive-class errors.

5. Create a visualization showing how the decision boundary rotates and shifts as you vary individual coefficients $\beta_0, \beta_1, \beta_2$ one at a time.

---

## Summary

| Concept | Formula | Key Point |
|---------|---------|-----------|
| Decision boundary | $\mathbf{x}^\top\boldsymbol{\beta} = 0$ | Where $P(Y=1) = 0.5$ |
| Weight vector role | $\boldsymbol{\beta}_{1:d} \perp$ boundary | Normal to decision surface |
| Intercept role | $\beta_0$ | Offset from origin |
| Steepness | $\|\boldsymbol{\beta}\|$ | Controls transition sharpness |
| BCE | $-[y\log p + (1-y)\log(1-p)]$ | Negative log-likelihood |
| Stable BCE | $\max(z,0) - yz + \log(1+e^{-\|z\|})$ | Numerically stable form |
| PyTorch | `BCEWithLogitsLoss` | **Always prefer this for training** |

The decision boundary is a hyperplane whose geometry is entirely determined by the weight vector and intercept. Understanding this geometry provides intuition for how logistic regression partitions the feature space and why regularization — which constrains $\|\boldsymbol{\beta}\|$ — produces smoother, more generalizable boundaries.
