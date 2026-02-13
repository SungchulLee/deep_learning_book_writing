# Focal Loss

Focal Loss addresses a fundamental failure mode of standard cross-entropy: in datasets with severe class imbalance, easy-to-classify examples from the majority class dominate the gradient signal, preventing the model from learning to identify the rare minority class. Introduced by Lin et al. (2017) for dense object detection, Focal Loss down-weights well-classified examples and focuses training on hard, misclassified samples.

## The Class Imbalance Problem

In heavily imbalanced datasets (e.g., 99% background, 1% objects in detection; 99.9% legitimate, 0.1% fraudulent in fraud detection), standard cross-entropy allows the model to achieve high accuracy by predicting the majority class. The core issue is not accuracy but **gradient dominance**: the vast number of easy negatives collectively produces a large gradient signal that overwhelms the learning signal from the few hard positives.

Consider a detector evaluating $10^5$ candidate regions per image, where only a handful contain objects. Even if each easy negative contributes a small loss, the cumulative gradient from $10^5$ easy negatives far exceeds the signal from the few positives. The model learns to output "background" everywhere—a degenerate but low-loss solution.

## Mathematical Formulation

### Standard Cross-Entropy Baseline

For binary classification, the cross-entropy loss for a single sample is:

$$\text{CE}(p, y) = -y\log(p) - (1 - y)\log(1 - p)$$

Define $p_t$ as the model's estimated probability for the **true class**:

$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases}$$

Then cross-entropy simplifies to $\text{CE}(p_t) = -\log(p_t)$.

### Balanced Cross-Entropy

A first attempt at handling imbalance is to introduce a class-balancing weight $\alpha_t$:

$$\text{CE}_{\text{balanced}} = -\alpha_t \log(p_t)$$

where $\alpha_t = \alpha$ for the positive class and $\alpha_t = 1 - \alpha$ for the negative class. This scales the loss contribution of each class but does not distinguish between easy and hard examples within a class.

### Focal Loss Definition

Focal Loss adds a **modulating factor** $(1 - p_t)^\gamma$ to the cross-entropy:

$$\mathcal{L}_{\text{FL}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where:

- $p_t$ is the model's estimated probability for the true class
- $\gamma \geq 0$ is the **focusing parameter** that controls down-weighting strength
- $\alpha_t$ is the optional class-balancing weight

### How the Modulating Factor Works

The factor $(1 - p_t)^\gamma$ smoothly scales the loss based on prediction confidence:

- **Well-classified examples** ($p_t \to 1$): The factor $(1 - p_t)^\gamma \to 0$, so the loss contribution approaches zero. Easy examples are effectively ignored.
- **Misclassified examples** ($p_t \to 0$): The factor $(1 - p_t)^\gamma \to 1$, so the loss is unaffected. Hard examples retain their full gradient signal.
- **Moderately confident examples** ($p_t \approx 0.5$): The factor provides intermediate down-weighting.

For a well-classified example with $p_t = 0.9$ and $\gamma = 2$:

$$(1 - 0.9)^2 = 0.01$$

The loss is reduced by a factor of 100 compared to standard cross-entropy. For a misclassified example with $p_t = 0.1$:

$$(1 - 0.1)^2 = 0.81$$

The loss is only reduced by $\sim 19\%$. This differential down-weighting redirects the optimization effort toward the examples that matter most.

### Effect of $\gamma$

| $\gamma$ | Behavior |
|----------|----------|
| $\gamma = 0$ | Standard cross-entropy (no modulation) |
| $\gamma = 1$ | Mild down-weighting of easy examples |
| $\gamma = 2$ | Standard choice; strong down-weighting (recommended default) |
| $\gamma = 5$ | Aggressive down-weighting; almost ignores easy examples |

## Gradient Analysis

The gradient of Focal Loss with respect to the logit $z$ (where $p = \sigma(z)$) is:

$$\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z} = \alpha_t(1 - p_t)^{\gamma-1}\left[\gamma p_t \log(p_t) + p_t - 1\right] \cdot \frac{\partial p_t}{\partial z}$$

Compared to cross-entropy's gradient $p - y$, the focal loss gradient is scaled by a factor that depends on the prediction confidence. This means the effective learning rate for easy examples is dramatically reduced, while hard examples receive a gradient magnitude comparable to standard cross-entropy.

## PyTorch Implementation

### Binary Focal Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    https://arxiv.org/abs/1708.02002

    Args:
        alpha: Class-balancing weight for the positive class. Default: 0.25.
        gamma: Focusing parameter. Higher values increase down-weighting
               of easy examples. Default: 2.0.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (before sigmoid), shape (N,).
            targets: Binary labels (0 or 1), shape (N,).
        """
        probs = torch.sigmoid(logits)

        # p_t: probability assigned to the true class
        p_t = torch.where(targets == 1, probs, 1 - probs)

        # alpha_t: class-balancing weight
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma

        # Standard BCE (per sample, numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Focal loss
        loss = alpha_t * focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

### Multi-Class Focal Loss

```python
class MultiClassFocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Args:
        alpha: Per-class weights, shape (K,). If None, no class weighting.
        gamma: Focusing parameter. Default: 2.0.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs, shape (N, K).
            targets: Class indices, shape (N,).
        """
        # Compute softmax probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Gather p_t: probability of the true class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma

        # NLL loss per sample
        nll = F.nll_loss(log_probs, targets, reduction='none')

        # Apply focal weight
        loss = focal_weight * nll

        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

## Demonstration: Effect of $\gamma$

```python
focal_criterion = FocalLoss(alpha=0.25, gamma=2.0)
bce_criterion = nn.BCEWithLogitsLoss(reduction='none')

# Predictions with varying difficulty
logits = torch.tensor([2.0, -1.5, -2.0, 3.0, -1.0])
targets = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
probs = torch.sigmoid(logits)

print("Per-sample comparison:")
bce_per_sample = bce_criterion(logits, targets)
focal_per_sample = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')(logits, targets)

for i in range(len(targets)):
    p = probs[i].item()
    p_t = p if targets[i] == 1 else 1 - p
    print(f"  Sample {i+1}: p_t={p_t:.3f}, "
          f"BCE={bce_per_sample[i]:.4f}, "
          f"Focal={focal_per_sample[i]:.4f}, "
          f"Ratio={focal_per_sample[i]/bce_per_sample[i]:.4f}")
```

```python
# Effect of gamma on loss curves
import matplotlib.pyplot as plt

p_t = torch.linspace(0.01, 0.99, 200)
gammas = [0, 0.5, 1, 2, 5]

plt.figure(figsize=(8, 5))
for gamma in gammas:
    focal = -((1 - p_t) ** gamma) * torch.log(p_t)
    plt.plot(p_t.numpy(), focal.numpy(), label=f'γ={gamma}')

plt.xlabel('p_t (probability of true class)')
plt.ylabel('Focal Loss')
plt.title('Focal Loss for Different γ Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Choosing Hyperparameters

**$\gamma$ (focusing parameter).** Start with $\gamma = 2$ as the default. Increase $\gamma$ for more severe imbalance or when easy examples dominate training. Decrease $\gamma$ toward 0 if the model struggles to learn anything (the curriculum may be too aggressive).

**$\alpha$ (class-balancing weight).** Set $\alpha$ to the inverse class frequency: $\alpha = n_{\text{negative}} / (n_{\text{positive}} + n_{\text{negative}})$ for the positive class. The original paper uses $\alpha = 0.25$ for the positive class in object detection. Tune $\alpha$ and $\gamma$ jointly, as they interact: increasing $\gamma$ already reduces the influence of easy examples, so aggressive $\alpha$ may not be needed.

## Comparison with Alternative Strategies

| Strategy | Mechanism | Limitation |
|----------|-----------|------------|
| **Oversampling** (SMOTE) | Duplicate minority samples | Overfitting to minority class |
| **Undersampling** | Remove majority samples | Discards potentially useful data |
| **Class weights** ($\alpha_t$ alone) | Scale loss per class | Does not distinguish easy/hard |
| **Focal Loss** | Down-weight easy, any class | Requires tuning $\gamma$ |

Focal Loss is complementary to sampling strategies and can be combined with them.

## Key Takeaways

Focal Loss addresses class imbalance by modulating the cross-entropy loss with a factor $(1 - p_t)^\gamma$ that down-weights well-classified examples. The focusing parameter $\gamma$ controls the strength of down-weighting, with $\gamma = 2$ as the standard default. Unlike simple class-weighting, Focal Loss distinguishes between easy and hard examples within each class, directing the optimization effort where it matters most. The implementation builds on `BCEWithLogitsLoss` for numerical stability, requiring only the additional computation of the modulating factor.
