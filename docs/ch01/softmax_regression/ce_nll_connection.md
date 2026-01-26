# Cross-Entropy as Negative Log-Likelihood

## Learning Objectives

By the end of this section, you will be able to:

- Derive cross-entropy loss from maximum likelihood estimation principles
- Understand the information-theoretic interpretation of cross-entropy
- Connect KL divergence to cross-entropy in classification
- Implement and analyze cross-entropy loss in PyTorch
- Recognize the equivalence between NLL and CE in multi-class classification

---

## The Maximum Likelihood Framework

### Setting Up the Problem

In multi-class classification with $K$ classes, we have:

- **Data:** $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{N}$ where $y^{(i)} \in \{1, \ldots, K\}$
- **Model:** Predicts probabilities $\hat{\pi}_k^{(i)} = P(Y = k | \mathbf{x}^{(i)}; \boldsymbol{\theta})$
- **Goal:** Find parameters $\boldsymbol{\theta}$ that maximize the likelihood of observed data

### The Likelihood Function

Assuming independent samples, the likelihood is:

$$\mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^{N} P(Y = y^{(i)} | \mathbf{x}^{(i)}; \boldsymbol{\theta}) = \prod_{i=1}^{N} \hat{\pi}_{y^{(i)}}^{(i)}$$

Using one-hot encoding $\mathbf{y}^{(i)}$ where $y_k^{(i)} = \mathbb{1}[y^{(i)} = k]$:

$$\mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^{N} \prod_{k=1}^{K} \left(\hat{\pi}_k^{(i)}\right)^{y_k^{(i)}}$$

### The Log-Likelihood

Taking the logarithm (which is monotonic, so maximizing log-likelihood = maximizing likelihood):

$$\ell(\boldsymbol{\theta}) = \log \mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log \hat{\pi}_k^{(i)}$$

### Negative Log-Likelihood (NLL)

Since we typically minimize loss functions, we define the **negative log-likelihood**:

$$\text{NLL}(\boldsymbol{\theta}) = -\ell(\boldsymbol{\theta}) = -\sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log \hat{\pi}_k^{(i)}$$

---

## Cross-Entropy: Information-Theoretic View

### Entropy and Information

**Entropy** measures the average uncertainty (or information content) of a distribution:

$$H(\mathbf{p}) = -\sum_{k=1}^{K} p_k \log p_k = \mathbb{E}_{X \sim \mathbf{p}}[-\log p_X]$$

**Key insight:** Entropy is minimized (= 0) when the distribution is deterministic, and maximized ($= \log K$) when uniform.

### Cross-Entropy Definition

The **cross-entropy** between a true distribution $\mathbf{p}$ and a predicted distribution $\mathbf{q}$ is:

$$H(\mathbf{p}, \mathbf{q}) = -\sum_{k=1}^{K} p_k \log q_k = \mathbb{E}_{X \sim \mathbf{p}}[-\log q_X]$$

This measures the average number of bits needed to encode samples from $\mathbf{p}$ using a code optimized for $\mathbf{q}$.

### Cross-Entropy in Classification

For a single sample with one-hot true label $\mathbf{y}$ (true class $c$) and predicted probabilities $\hat{\boldsymbol{\pi}}$:

$$H(\mathbf{y}, \hat{\boldsymbol{\pi}}) = -\sum_{k=1}^{K} y_k \log \hat{\pi}_k = -\log \hat{\pi}_c$$

Since $y_c = 1$ and $y_k = 0$ for $k \neq c$, only the true class term survives.

---

## The Equivalence: NLL = Cross-Entropy

### Mathematical Proof

For the full dataset:

$$\text{Cross-Entropy Loss} = \frac{1}{N} \sum_{i=1}^{N} H(\mathbf{y}^{(i)}, \hat{\boldsymbol{\pi}}^{(i)}) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log \hat{\pi}_k^{(i)}$$

This is exactly $\frac{1}{N} \text{NLL}(\boldsymbol{\theta})$!

$$\boxed{\text{Cross-Entropy Loss} = \frac{1}{N} \text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{\pi}_{y^{(i)}}^{(i)}}$$

### Why This Matters

| Perspective | Interpretation |
|-------------|----------------|
| Statistical | Maximum likelihood estimation |
| Information-theoretic | Minimize coding inefficiency |
| Optimization | Convex loss with nice gradients |
| Practical | Works well empirically |

---

## Connection to KL Divergence

### KL Divergence Definition

The **Kullback-Leibler divergence** (relative entropy) from $\mathbf{q}$ to $\mathbf{p}$ is:

$$D_{KL}(\mathbf{p} \| \mathbf{q}) = \sum_{k=1}^{K} p_k \log \frac{p_k}{q_k} = H(\mathbf{p}, \mathbf{q}) - H(\mathbf{p})$$

### Decomposition

$$\boxed{H(\mathbf{p}, \mathbf{q}) = H(\mathbf{p}) + D_{KL}(\mathbf{p} \| \mathbf{q})}$$

This shows that cross-entropy equals:
- The inherent uncertainty in $\mathbf{p}$ (entropy)
- Plus the "extra cost" of using $\mathbf{q}$ instead of $\mathbf{p}$ (KL divergence)

### In Classification (One-Hot Labels)

When $\mathbf{y}$ is one-hot, $H(\mathbf{y}) = 0$ (no uncertainty in the label). Therefore:

$$H(\mathbf{y}, \hat{\boldsymbol{\pi}}) = D_{KL}(\mathbf{y} \| \hat{\boldsymbol{\pi}})$$

**Minimizing cross-entropy = minimizing KL divergence from true labels!**

---

## Geometric Interpretation

### The Loss Surface

For a single sample with true class $c$, the cross-entropy loss is:

$$\mathcal{L} = -\log \hat{\pi}_c$$

where $\hat{\pi}_c = \sigma(\mathbf{z})_c = \frac{e^{z_c}}{\sum_j e^{z_j}}$ is the softmax probability.

**Properties:**
- $\mathcal{L} = 0$ when $\hat{\pi}_c = 1$ (perfect prediction)
- $\mathcal{L} \to \infty$ when $\hat{\pi}_c \to 0$ (completely wrong)
- $\mathcal{L} = \log K$ when $\hat{\pi}_c = 1/K$ (uniform, random guessing)

### Visualization

```
Loss = -log(p_true)

Loss ↑
  ∞  │╲
     │ ╲
  4  │  ╲
     │   ╲
  2  │    ╲___
     │        ╲____
  0  │             ╲___
     └──────────────────────→ p_true
     0    0.25   0.5    1.0

Key points:
  p_true = 0.01: Loss ≈ 4.6
  p_true = 0.5:  Loss ≈ 0.69
  p_true = 0.9:  Loss ≈ 0.1
  p_true = 1.0:  Loss = 0
```

---

## PyTorch Implementation

### Understanding nn.CrossEntropyLoss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch's CrossEntropyLoss combines:
# 1. Softmax: logits → probabilities
# 2. Log: probabilities → log-probabilities
# 3. NLL: select true class log-prob, negate, average

criterion = nn.CrossEntropyLoss()

# Input: logits (raw scores), NOT probabilities!
logits = torch.tensor([[2.0, 1.0, 0.5],   # Sample 1
                       [0.5, 2.5, 1.0]])  # Sample 2

# Target: class indices, NOT one-hot!
targets = torch.tensor([0, 1])  # Sample 1: class 0, Sample 2: class 1

loss = criterion(logits, targets)
print(f"CrossEntropyLoss: {loss.item():.4f}")
```

### Manual Computation

```python
def cross_entropy_manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Manual implementation of cross-entropy loss.
    
    CE = -mean(log(softmax(logits))[true_class])
       = mean(-log_softmax(logits)[true_class])
       = mean(NLL)
    """
    # Method 1: Step by step
    # probs = F.softmax(logits, dim=1)
    # log_probs = torch.log(probs)
    # nll = -log_probs[range(len(targets)), targets]
    # return nll.mean()
    
    # Method 2: More numerically stable
    log_probs = F.log_softmax(logits, dim=1)
    nll = -log_probs[range(len(targets)), targets]
    return nll.mean()

# Verify equivalence
loss_manual = cross_entropy_manual(logits, targets)
loss_pytorch = F.cross_entropy(logits, targets)
print(f"Manual:  {loss_manual.item():.6f}")
print(f"PyTorch: {loss_pytorch.item():.6f}")
print(f"Match:   {torch.allclose(loss_manual, loss_pytorch)}")
```

### Decomposing CrossEntropyLoss

```python
# CrossEntropyLoss = LogSoftmax + NLLLoss

log_softmax = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()

# Equivalent computation
log_probs = log_softmax(logits)
loss_decomposed = nll_loss(log_probs, targets)

print(f"Decomposed: {loss_decomposed.item():.6f}")
print(f"Direct CE:  {criterion(logits, targets).item():.6f}")
```

---

## Common Variants and Extensions

### Binary Cross-Entropy

For binary classification ($K = 2$), cross-entropy simplifies to **binary cross-entropy (BCE)**:

$$\text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y^{(i)} \log \hat{p}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$

```python
# For binary classification
bce_loss = nn.BCELoss()
bce_with_logits = nn.BCEWithLogitsLoss()

# BCE expects probabilities in [0, 1]
probs = torch.sigmoid(torch.tensor([0.5, -1.0, 2.0]))
binary_targets = torch.tensor([1.0, 0.0, 1.0])
loss_bce = bce_loss(probs, binary_targets)

# BCEWithLogitsLoss is more stable (takes logits)
logits_binary = torch.tensor([0.5, -1.0, 2.0])
loss_bce_logits = bce_with_logits(logits_binary, binary_targets)
```

### Weighted Cross-Entropy

For imbalanced classes, weight the loss by class frequency:

$$\text{Weighted CE} = -\frac{1}{N} \sum_{i=1}^{N} w_{y^{(i)}} \log \hat{\pi}_{y^{(i)}}^{(i)}$$

```python
# Class weights (higher weight = more importance)
class_weights = torch.tensor([1.0, 2.0, 3.0])  # Weight rare classes higher

weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
loss_weighted = weighted_criterion(logits, targets)
```

### Label Smoothing Cross-Entropy

Instead of one-hot targets, use smoothed targets:

$$\tilde{y}_k = \begin{cases}
1 - \epsilon & \text{if } k = c \text{ (true class)} \\
\frac{\epsilon}{K-1} & \text{otherwise}
\end{cases}$$

```python
# PyTorch 1.10+ supports label smoothing directly
criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
loss_smooth = criterion_smooth(logits, targets)

# Manual implementation
def label_smoothing_ce(logits, targets, smoothing=0.1):
    num_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    
    # Create smoothed targets
    with torch.no_grad():
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    
    # Cross-entropy with soft targets
    loss = -(smooth_targets * log_probs).sum(dim=1).mean()
    return loss
```

### Focal Loss

Addresses class imbalance by down-weighting easy examples:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

```python
def focal_loss(logits, targets, alpha=1.0, gamma=2.0):
    """
    Focal Loss for dense object detection.
    
    Args:
        logits: Raw model outputs
        targets: Ground truth class indices
        alpha: Weighting factor
        gamma: Focusing parameter (γ > 0 reduces loss for well-classified examples)
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # pt = probability of true class
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * ce_loss).mean()
```

---

## Numerical Stability Analysis

### The Problem with Naive Implementation

```python
def ce_naive(logits, targets):
    """UNSTABLE: Computing softmax then log loses precision."""
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log(probs)  # log(small number) → large negative
    return F.nll_loss(log_probs, targets)

def ce_stable(logits, targets):
    """STABLE: Using log_softmax directly."""
    log_probs = F.log_softmax(logits, dim=1)
    return F.nll_loss(log_probs, targets)

# Test with extreme logits
extreme_logits = torch.tensor([[100.0, 0.0, 0.0]])  # Very confident
targets = torch.tensor([0])

# Naive may produce nan or inf for extreme values
print(f"Naive:  {ce_naive(extreme_logits, targets).item()}")   # May be unstable
print(f"Stable: {ce_stable(extreme_logits, targets).item()}")  # Always works
print(f"PyTorch: {F.cross_entropy(extreme_logits, targets).item()}")  # Built-in stability
```

### Log-Sum-Exp Trick in Cross-Entropy

The stable computation uses:

$$\log\left(\sum_j e^{z_j}\right) = z_{\max} + \log\left(\sum_j e^{z_j - z_{\max}}\right)$$

So cross-entropy becomes:

$$\text{CE} = -z_c + \log\sum_j e^{z_j} = -z_c + z_{\max} + \log\sum_j e^{z_j - z_{\max}}$$

This avoids both overflow (large exponentials) and underflow (log of small numbers).

---

## Complete Example: Cross-Entropy Analysis

```python
import torch
import torch.nn.functional as F
import numpy as np

class CrossEntropyAnalyzer:
    """Comprehensive analysis of cross-entropy loss."""
    
    @staticmethod
    def loss_vs_probability():
        """Show how loss varies with predicted probability."""
        print("=" * 60)
        print("CROSS-ENTROPY LOSS vs PREDICTED PROBABILITY")
        print("=" * 60)
        
        probs = torch.linspace(0.01, 0.99, 20)
        losses = -torch.log(probs)
        
        print(f"\n{'P(true)':<12} {'Loss':<12} {'Interpretation':<30}")
        print("-" * 54)
        
        for p, l in zip(probs.tolist(), losses.tolist()):
            if p < 0.1:
                interp = "Very wrong prediction"
            elif p < 0.5:
                interp = "Uncertain/wrong"
            elif p < 0.9:
                interp = "Reasonably correct"
            else:
                interp = "Confident and correct"
            print(f"{p:<12.3f} {l:<12.4f} {interp}")
    
    @staticmethod
    def multi_class_example():
        """Detailed multi-class example."""
        print("\n" + "=" * 60)
        print("MULTI-CLASS CROSS-ENTROPY EXAMPLE")
        print("=" * 60)
        
        # 3 samples, 4 classes
        logits = torch.tensor([
            [2.0, 0.5, 0.3, 0.1],   # Should predict class 0
            [0.1, 3.0, 0.2, 0.5],   # Should predict class 1
            [0.5, 0.5, 0.5, 0.5],   # Uniform (uncertain)
        ])
        
        targets = torch.tensor([0, 1, 2])  # True classes
        
        # Compute probabilities
        probs = F.softmax(logits, dim=1)
        
        print("\nLogits:")
        print(logits)
        print("\nProbabilities (softmax):")
        print(probs)
        print(f"\nTrue classes: {targets.tolist()}")
        
        # Per-sample loss
        log_probs = F.log_softmax(logits, dim=1)
        per_sample_loss = -log_probs[range(3), targets]
        
        print(f"\nPer-sample cross-entropy loss:")
        for i in range(3):
            true_prob = probs[i, targets[i]].item()
            loss = per_sample_loss[i].item()
            print(f"  Sample {i}: P(true) = {true_prob:.4f}, Loss = {loss:.4f}")
        
        # Total loss
        total_loss = F.cross_entropy(logits, targets)
        print(f"\nMean loss: {total_loss.item():.4f}")
    
    @staticmethod
    def information_theory_view():
        """Information-theoretic interpretation."""
        print("\n" + "=" * 60)
        print("INFORMATION-THEORETIC INTERPRETATION")
        print("=" * 60)
        
        # True distribution (one-hot for class 0)
        p_true = torch.tensor([1.0, 0.0, 0.0])
        
        # Various predicted distributions
        predictions = {
            "Perfect": torch.tensor([1.0, 0.0, 0.0]),
            "Confident correct": torch.tensor([0.9, 0.05, 0.05]),
            "Uncertain": torch.tensor([0.5, 0.25, 0.25]),
            "Random": torch.tensor([1/3, 1/3, 1/3]),
            "Wrong": torch.tensor([0.1, 0.8, 0.1]),
        }
        
        print(f"\nTrue distribution: {p_true.tolist()}")
        print(f"\n{'Prediction':<20} {'CE(p,q)':<10} {'H(p)':<10} {'KL(p||q)':<10}")
        print("-" * 50)
        
        for name, q in predictions.items():
            # Cross-entropy
            ce = -(p_true * torch.log(q + 1e-10)).sum().item()
            
            # Entropy of p (0 for one-hot)
            h_p = -(p_true * torch.log(p_true + 1e-10)).sum().item()
            
            # KL divergence = CE - H(p)
            kl = ce - h_p
            
            print(f"{name:<20} {ce:<10.4f} {h_p:<10.4f} {kl:<10.4f}")
        
        print("\nNote: For one-hot true distribution, H(p)=0, so CE = KL")

# Run analysis
if __name__ == "__main__":
    CrossEntropyAnalyzer.loss_vs_probability()
    CrossEntropyAnalyzer.multi_class_example()
    CrossEntropyAnalyzer.information_theory_view()
```

---

## Summary

### The Fundamental Equations

**Cross-Entropy Loss:**
$$\boxed{\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{\pi}_{y^{(i)}}^{(i)} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log \hat{\pi}_k^{(i)}}$$

**Equivalences:**
$$\text{Cross-Entropy} = \frac{1}{N} \text{NLL} = H(\mathbf{p}, \mathbf{q}) = H(\mathbf{p}) + D_{KL}(\mathbf{p} \| \mathbf{q})$$

### PyTorch Quick Reference

| Function | Input | Notes |
|----------|-------|-------|
| `nn.CrossEntropyLoss()` | Logits | Most common, stable |
| `F.cross_entropy()` | Logits | Functional version |
| `nn.NLLLoss()` | Log-probabilities | Use with `log_softmax` |
| `F.log_softmax()` | Logits | Stable log-probabilities |

### Key Insights

!!! info "Three Perspectives on Cross-Entropy"
    1. **Statistical:** Maximizing likelihood of observed labels
    2. **Information-theoretic:** Minimizing coding inefficiency
    3. **Geometric:** Measuring "distance" between distributions (via KL divergence)

---

## Next Steps

Understanding cross-entropy loss prepares you for:

1. **Jacobian of Softmax** — Partial derivatives of softmax outputs
2. **Gradient Derivation** — Complete backpropagation through softmax + CE
3. **PyTorch Implementation** — Building end-to-end classifiers

---

## References

1. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*, Chapter 2
2. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Chapter 8.6
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 6.2.2
4. PyTorch Documentation: [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
