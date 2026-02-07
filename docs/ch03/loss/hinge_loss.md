# Hinge Loss

Hinge loss is the loss function underlying Support Vector Machines (SVMs) and maximum-margin classifiers. Unlike cross-entropy, which operates on probabilistic outputs, hinge loss directly encourages a geometric margin between classes in the feature space. This section derives hinge loss from the maximum-margin principle, establishes its mathematical properties, and demonstrates the PyTorch implementation.

## Geometric Motivation: Maximum Margin

Consider a binary classification problem with labels $y \in \{-1, +1\}$ and a linear decision boundary $f(x) = w^T x + b$. The **functional margin** of sample $(x^{(i)}, y^{(i)})$ is:

$$m^{(i)} = y^{(i)} \cdot f(x^{(i)}) = y^{(i)}(w^T x^{(i)} + b)$$

The margin is positive when the prediction is correct ($y$ and $f(x)$ have the same sign) and negative when incorrect. The larger the margin, the more confident the classification.

The maximum-margin principle seeks parameters $(w, b)$ that maximize the minimum margin across all training samples, subject to correct classification. This leads naturally to a loss function that is zero when the margin exceeds a threshold and grows linearly when it falls below.

## Mathematical Definition

### Binary Hinge Loss

For a single sample with true label $y \in \{-1, +1\}$ and model output (score) $f(x)$:

$$\mathcal{L}_{\text{hinge}}(f(x), y) = \max(0,\; 1 - y \cdot f(x))$$

This can also be written as $[1 - y \cdot f(x)]_+$ where $[\cdot]_+$ denotes the positive part.

The loss is:

- **Zero** when $y \cdot f(x) \geq 1$ — the sample is correctly classified with sufficient margin
- **Linear** when $y \cdot f(x) < 1$ — the sample is either misclassified or within the margin band

The threshold of 1 defines the **margin boundary**: samples must not only be on the correct side of the decision boundary ($y \cdot f(x) > 0$) but must exceed a margin of 1 to incur zero loss.

### Multi-Class Hinge Loss (Weston-Watkins)

For $K$ classes with scores $s_k = f_k(x)$ and true class $c$:

$$\mathcal{L}_{\text{multi-hinge}} = \sum_{k \neq c} \max(0,\; s_k - s_c + \Delta)$$

where $\Delta$ is the desired margin (typically $\Delta = 1$). Each incorrect class contributes a loss if its score comes within $\Delta$ of the true class score.

## Properties

### Gradient Analysis

$$\frac{\partial \mathcal{L}_{\text{hinge}}}{\partial f(x)} = \begin{cases} 0 & \text{if } y \cdot f(x) \geq 1 \\ -y & \text{if } y \cdot f(x) < 1 \end{cases}$$

Two critical properties emerge:

**Sparse gradients.** Once a sample achieves sufficient margin ($y \cdot f(x) \geq 1$), it contributes zero gradient. Only margin violators and support vectors (samples on the margin boundary) influence parameter updates. This sparsity is both a strength (efficient training, robust to well-separated points) and a limitation (no refinement signal for correctly classified points).

**Non-differentiability at the hinge.** The loss has a kink at $y \cdot f(x) = 1$. In practice, subgradient methods handle this without difficulty.

### Comparison with Cross-Entropy

| Property | Hinge Loss | Cross-Entropy |
|----------|------------|---------------|
| **Output type** | Raw scores (any $\mathbb{R}$) | Probabilities via softmax/sigmoid |
| **Label convention** | $y \in \{-1, +1\}$ | $y \in \{0, 1\}$ or class indices |
| **Zero loss** | When margin $\geq 1$ | Never (always positive) |
| **Gradient for correct** | Zero (margin achieved) | Non-zero (pushes toward $p \to 1$) |
| **Probabilistic** | No (scores, not probabilities) | Yes (calibrated probabilities) |
| **Robustness** | Robust (bounded gradient for outliers) | Unbounded loss for confident wrong predictions |

A crucial difference: cross-entropy never stops penalizing, even when the prediction is correct. It always pushes $p$ closer to 1 (or 0). Hinge loss stops once the margin is achieved, which can lead to better-separated decision boundaries but lacks probabilistic calibration.

### Squared Hinge Loss

A smooth variant replaces the linear penalty with a quadratic one:

$$\mathcal{L}_{\text{sq-hinge}}(f(x), y) = \max(0,\; 1 - y \cdot f(x))^2$$

This is differentiable everywhere (including at the hinge point), produces gradients that scale with the margin violation, and is more sensitive to large violations. However, the quadratic penalty makes it more sensitive to outliers than standard hinge loss.

## PyTorch Implementation

### Binary Hinge Loss: `nn.HingeEmbeddingLoss`

PyTorch's `nn.HingeEmbeddingLoss` computes:

$$\mathcal{L} = \begin{cases} x & \text{if } y = 1 \\ \max(0, \Delta - x) & \text{if } y = -1 \end{cases}$$

!!! warning "Interface Difference"
    `nn.HingeEmbeddingLoss` is designed for contrastive/embedding tasks, not standard classification. For standard hinge loss, use the manual implementation or `nn.MultiMarginLoss` below.

### Multi-Class Hinge Loss: `nn.MultiMarginLoss`

`nn.MultiMarginLoss` implements the multi-class hinge loss:

```python
import torch
import torch.nn as nn

# 4 samples, 3 classes
scores = torch.tensor([[2.0, 0.5, 0.1],
                        [0.1, 2.5, 0.3],
                        [1.0, 2.0, 0.5],
                        [2.5, 1.5, 1.0]])
targets = torch.tensor([0, 1, 1, 0])

criterion = nn.MultiMarginLoss(margin=1.0)
loss = criterion(scores, targets)
print(f"Multi-class Hinge Loss: {loss.item():.4f}")
```

### Manual Binary Hinge Loss

For standard binary hinge loss with $y \in \{-1, +1\}$:

```python
def hinge_loss(scores, targets, reduction='mean'):
    """Binary hinge loss.

    Args:
        scores: Model outputs (raw scores), shape (N,).
        targets: Labels in {-1, +1}, shape (N,).
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Hinge loss.
    """
    losses = torch.clamp(1 - targets * scores, min=0)

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    return losses

# Example
scores = torch.tensor([1.5, -0.5, 0.8, -1.2, 0.3])
targets = torch.tensor([1, -1, 1, -1, 1])  # Note: {-1, +1} labels

loss = hinge_loss(scores, targets)
print(f"Hinge Loss: {loss.item():.4f}")

# Per-sample analysis
per_sample = hinge_loss(scores, targets, reduction='none')
for i in range(len(targets)):
    margin = (targets[i] * scores[i]).item()
    print(f"  Sample {i+1}: score={scores[i]:.1f}, y={targets[i]:+d}, "
          f"margin={margin:.2f}, loss={per_sample[i]:.4f}")
```

### Module-Based Implementation

```python
class HingeLoss(nn.Module):
    """Binary hinge loss as an nn.Module.

    Args:
        margin: Minimum desired margin. Default: 1.0.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, scores, targets):
        """
        Args:
            scores: Raw model outputs, shape (N,).
            targets: Labels in {-1, +1}, shape (N,).
        """
        losses = torch.clamp(self.margin - targets * scores, min=0)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        return losses
```

### SVM-Style Training Example

```python
import torch.optim as optim

# Linearly separable 2D data
torch.manual_seed(42)
n = 100

# Generate two clusters
x_pos = torch.randn(n // 2, 2) + torch.tensor([2.0, 2.0])
x_neg = torch.randn(n // 2, 2) + torch.tensor([-2.0, -2.0])
X = torch.cat([x_pos, x_neg], dim=0)
y = torch.cat([torch.ones(n // 2), -torch.ones(n // 2)])

# Linear model
model = nn.Linear(2, 1)
criterion = HingeLoss(margin=1.0)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)  # L2 reg

for epoch in range(200):
    scores = model(X).squeeze()
    loss = criterion(scores, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        accuracy = ((scores * y) > 0).float().mean()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}, acc={accuracy.item():.4f}")
```

Note the `weight_decay` parameter, which adds L2 regularization $\frac{\lambda}{2}\|w\|^2$ — the standard regularizer in SVM formulations that controls the trade-off between margin maximization and training error minimization.

## When to Use Hinge Loss

**Use hinge loss when:**

- You want maximum-margin classification (SVM-like behavior)
- Probabilistic outputs are not needed—only the decision boundary matters
- The task benefits from sparse gradient updates (only margin violators contribute)
- Robustness to confident misclassifications is desired (linear, not exponential, penalty)

**Prefer cross-entropy when:**

- Calibrated probability outputs are needed (e.g., for ranking, thresholding, or downstream tasks)
- The model uses softmax or sigmoid outputs
- You want gradients for all samples, not just margin violators
- The problem requires multi-label classification

## Key Takeaways

Hinge loss embodies the maximum-margin principle: it penalizes samples that fail to achieve a specified margin while ignoring well-classified samples entirely. This produces sparse gradients where only support vectors (margin violators) drive parameter updates. Unlike cross-entropy, hinge loss does not produce probabilistic outputs and stops optimizing once sufficient margin is achieved. In PyTorch, `nn.MultiMarginLoss` provides the multi-class version, while binary hinge loss is straightforward to implement manually. The connection to SVMs motivates the common pairing of hinge loss with L2 regularization (`weight_decay`) to control the margin–error trade-off.
