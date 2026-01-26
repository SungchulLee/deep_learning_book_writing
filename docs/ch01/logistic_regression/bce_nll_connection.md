# Binary Cross-Entropy as Negative Log-Likelihood

## Learning Objectives

By the end of this section, you will be able to:

- Understand the deep connection between BCE and maximum likelihood estimation
- Explain why BCE is the "correct" loss function for binary classification
- Compare BCE, BCEWithLogitsLoss, and their numerical stability properties
- Implement both loss functions from scratch and verify against PyTorch implementations

---

## The Mathematical Connection

### From Log-Likelihood to Loss Function

Recall the log-likelihood for logistic regression:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

In optimization, we typically minimize rather than maximize. The **negative log-likelihood (NLL)** becomes our loss:

$$
\text{NLL} = -\ell(\boldsymbol{\beta}) = -\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

Dividing by $n$ to get the average:

$$
\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

This is **Binary Cross-Entropy (BCE)** — the standard loss function for binary classification.

### Information-Theoretic Interpretation

Cross-entropy between true distribution $q$ and predicted distribution $p$:

$$
H(q, p) = -\mathbb{E}_q[\log p] = -\sum_x q(x) \log p(x)
$$

For binary classification:

$$
H(y, \hat{y}) = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]
$$

**BCE measures how well our predictions encode the true distribution.**

---

## BCE vs BCEWithLogitsLoss

### The Numerical Stability Problem

When computing BCE with sigmoid:

| Scenario | Problem |
|----------|---------|
| $z \gg 0$ | $\sigma(z) \approx 1$, $\log(1-\sigma(z)) \to -\infty$ |
| $z \ll 0$ | $\sigma(z) \approx 0$, $\log(\sigma(z)) \to -\infty$ |

### The Stable Formulation

**BCEWithLogitsLoss** combines sigmoid and BCE:

$$
\text{BCE}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})
$$

This avoids computing $\sigma(z)$ explicitly and remains stable for all $z$.

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

print("="*70)
print("BINARY CROSS-ENTROPY: LOSS FUNCTION DEEP DIVE")
print("="*70)

# ============================================================================
# Part 1: Manual BCE Implementation
# ============================================================================

def bce_manual(predictions, targets, eps=1e-12):
    """
    Manual implementation of Binary Cross-Entropy.
    
    BCE = -[y * log(p) + (1-y) * log(1-p)]
    """
    predictions = torch.clamp(predictions, eps, 1 - eps)
    loss = -(targets * torch.log(predictions) + 
             (1 - targets) * torch.log(1 - predictions))
    return loss.mean()


def bce_with_logits_manual(logits, targets):
    """
    Numerically stable BCE directly from logits.
    
    BCE(z, y) = max(z, 0) - z*y + log(1 + exp(-|z|))
    """
    max_val = torch.clamp(logits, min=0)
    loss = max_val - logits * targets + torch.log(1 + torch.exp(-torch.abs(logits)))
    return loss.mean()


# Test implementations
print("\n1. Comparing BCE implementations")
print("-" * 50)

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

print("\n" + "="*70)
print("NUMERICAL STABILITY COMPARISON")
print("="*70)

extreme_logits = torch.tensor([-100., -50., -10., 0., 10., 50., 100.])
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
    p_clipped = torch.clamp(p, 1e-7, 1-1e-7)
    unstable_loss = F.binary_cross_entropy(p_clipped, y_t)
    
    print(f"{z.item():>10.1f} {stable_loss.item():>15.6f} {unstable_loss.item():>15.6f}")

# ============================================================================
# Part 3: Training Comparison
# ============================================================================

print("\n" + "="*70)
print("TRAINING: BCELoss vs BCEWithLogitsLoss")
print("="*70)

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)


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


def train_model(model, criterion, X_train, y_train, num_epochs=100, lr=0.1):
    """Train and return loss history."""
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


# Train both models with same initialization
torch.manual_seed(42)
model1 = ModelWithSigmoid(20)

torch.manual_seed(42)
model2 = ModelWithLogits(20)

history1 = train_model(model1, nn.BCELoss(), X_train, y_train)
history2 = train_model(model2, nn.BCEWithLogitsLoss(), X_train, y_train)

print(f"\nFinal loss (BCELoss):           {history1[-1]:.6f}")
print(f"Final loss (BCEWithLogitsLoss): {history2[-1]:.6f}")

# ============================================================================
# Part 4: Visualization
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: BCE as function of prediction
ax1 = axes[0, 0]
p_range = torch.linspace(0.01, 0.99, 99)
loss_y1 = -torch.log(p_range)
loss_y0 = -torch.log(1 - p_range)

ax1.plot(p_range.numpy(), loss_y1.numpy(), 'b-', linewidth=2, label='y=1 (true positive)')
ax1.plot(p_range.numpy(), loss_y0.numpy(), 'r-', linewidth=2, label='y=0 (true negative)')
ax1.set_xlabel('Predicted Probability p', fontsize=11)
ax1.set_ylabel('BCE Loss', fontsize=11)
ax1.set_title('BCE Loss vs Predicted Probability', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 5])

# Plot 2: Loss curves comparison
ax2 = axes[0, 1]
ax2.plot(history1, 'b-', linewidth=2, alpha=0.7, label='BCELoss + Sigmoid')
ax2.plot(history2, 'r--', linewidth=2, alpha=0.7, label='BCEWithLogitsLoss')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Sigmoid vs Stable BCE computation
ax3 = axes[1, 0]
z_range = torch.linspace(-10, 10, 200)
sigmoid_z = torch.sigmoid(z_range)

ax3.plot(z_range.numpy(), sigmoid_z.numpy(), 'b-', linewidth=2, label='σ(z)')
ax3.fill_between(z_range.numpy(), 0, sigmoid_z.numpy(), alpha=0.2)
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Mark regions of potential numerical issues
ax3.axvspan(-10, -5, alpha=0.1, color='red', label='Underflow risk')
ax3.axvspan(5, 10, alpha=0.1, color='red')

ax3.set_xlabel('Logit z', fontsize=11)
ax3.set_ylabel('Probability', fontsize=11)
ax3.set_title('Sigmoid Function and Numerical Stability Regions', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Stable vs Unstable BCE for extreme values
ax4 = axes[1, 1]
extreme_z = torch.linspace(-20, 20, 100)
stable_losses = []
unstable_losses = []

for z in extreme_z:
    z_t = z.unsqueeze(0)
    y_t = torch.ones(1)
    
    stable = F.binary_cross_entropy_with_logits(z_t, y_t).item()
    
    p = torch.sigmoid(z_t)
    p_clipped = torch.clamp(p, 1e-7, 1-1e-7)
    unstable = F.binary_cross_entropy(p_clipped, y_t).item()
    
    stable_losses.append(stable)
    unstable_losses.append(unstable)

ax4.plot(extreme_z.numpy(), stable_losses, 'g-', linewidth=2, label='BCEWithLogitsLoss (stable)')
ax4.plot(extreme_z.numpy(), unstable_losses, 'r--', linewidth=2, label='BCE+Sigmoid (clipped)')
ax4.set_xlabel('Logit z', fontsize=11)
ax4.set_ylabel('Loss (y=1)', fontsize=11)
ax4.set_title('Numerical Stability Comparison', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bce_nll_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved!")
```

---

## Key Comparisons

### When to Use Each Loss

| Loss Function | Model Output | When to Use |
|---------------|--------------|-------------|
| `nn.BCELoss()` | Probabilities (after sigmoid) | When you need probabilities elsewhere |
| `nn.BCEWithLogitsLoss()` | Logits (before sigmoid) | **Recommended for training** |

### BCEWithLogitsLoss Advantages

1. **Numerical Stability**: Handles extreme logits without overflow/underflow
2. **Computational Efficiency**: Combines sigmoid + BCE in one fused operation
3. **Gradient Flow**: Better gradients for very confident predictions

---

## Exercises

1. **Implement** BCEWithLogitsLoss with `pos_weight` for handling class imbalance.

2. **Show** mathematically that $\frac{\partial \text{BCE}}{\partial z} = \sigma(z) - y$ (the elegant gradient form).

3. **Compare** training speed and final accuracy using both loss functions on a real dataset.

---

## Summary

| Concept | Formula | Key Point |
|---------|---------|-----------|
| BCE | $-[y\log p + (1-y)\log(1-p)]$ | Negative log-likelihood |
| Stable BCE | $\max(z,0) - yz + \log(1+e^{-|z|})$ | Numerically stable |
| Gradient | $\sigma(z) - y$ | Same for both formulations |
| PyTorch | `BCEWithLogitsLoss` | **Always prefer this** |

The equivalence between BCE and negative log-likelihood shows that minimizing BCE is equivalent to maximum likelihood estimation — the principled statistical approach to fitting probabilistic models.
