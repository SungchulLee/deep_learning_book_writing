# Loss Function Selection

Choosing the right loss function is a modeling decision that encodes assumptions about the data, the task, and the desired optimization behavior. This section provides a systematic framework for loss function selection, connecting theoretical foundations to practical decision criteria.

## Decision Framework

### Step 1: Task Type

The first branch separates regression from classification:

```
Task Type?
├── Regression (continuous output)
│   → MSE, MAE, or Huber
├── Binary Classification (2 classes)
│   → BCE, Focal Loss, or Hinge
├── Multi-Class Classification (K > 2, mutually exclusive)
│   → Cross-Entropy or Hinge
├── Multi-Label Classification (multiple labels per sample)
│   → BCE (per label)
└── Distributional Matching
    → KL Divergence
```

### Step 2: Data Characteristics

Within each task type, data properties determine the optimal choice:

**For Regression:**

| Characteristic | Recommended Loss | Reason |
|---------------|-----------------|--------|
| Clean data, Gaussian noise | MSE | Statistically optimal (Cramér-Rao) |
| Outliers present | MAE or Huber | Bounded influence of extreme values |
| Occasional outliers, mostly clean | Huber | MSE precision near optimum, MAE robustness far away |
| Heavy-tailed noise | MAE | Laplace MLE; median-based estimation |
| Unknown noise distribution | Huber (start) | Safest default; tune $\delta$ |

**For Classification:**

| Characteristic | Recommended Loss | Reason |
|---------------|-----------------|--------|
| Balanced classes | Cross-Entropy / BCE | Standard MLE |
| Moderate imbalance | Weighted Cross-Entropy | Class rebalancing |
| Severe imbalance ($>$10:1) | Focal Loss | Down-weights easy majority examples |
| Need calibrated probabilities | Cross-Entropy / BCE | Probabilistic output |
| Need maximum margin | Hinge Loss | SVM-like decision boundary |

### Step 3: Application-Specific Considerations

| Domain | Common Loss | Rationale |
|--------|------------|-----------|
| Object detection (boxes) | Smooth L1 / GIoU | Bounding box regression with outlier robustness |
| Semantic segmentation | Dice + BCE | Overlap optimization for imbalanced regions |
| Knowledge distillation | KL Divergence | Match teacher distribution |
| VAE training | Reconstruction + KL | ELBO maximization |
| GAN training | BCE / Wasserstein | Discriminator/generator objectives |
| Ranking tasks | Margin-based losses | Pairwise ordering |
| Language modeling | Cross-Entropy | Next-token prediction |

## PyTorch Quick Reference

### Regression Losses

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

predictions = torch.randn(32, 1)
targets = torch.randn(32, 1)

# MSE: default for clean regression
mse = nn.MSELoss()(predictions, targets)

# MAE: robust to outliers
mae = nn.L1Loss()(predictions, targets)

# Huber: hybrid MSE/MAE
huber = nn.HuberLoss(delta=1.0)(predictions, targets)

# Smooth L1: object detection convention
smooth_l1 = nn.SmoothL1Loss(beta=1.0)(predictions, targets)
```

### Classification Losses

```python
logits_binary = torch.randn(32)           # Binary: single logit
labels_binary = torch.randint(0, 2, (32,)).float()

logits_multi = torch.randn(32, 10)        # Multi-class: K logits
labels_multi = torch.randint(0, 10, (32,))

# Binary classification
bce = nn.BCEWithLogitsLoss()(logits_binary, labels_binary)

# Multi-class classification
ce = nn.CrossEntropyLoss()(logits_multi, labels_multi)

# With class weights (for imbalance)
weights = torch.ones(10)
weights[0] = 5.0  # upweight rare class 0
ce_weighted = nn.CrossEntropyLoss(weight=weights)(logits_multi, labels_multi)
```

### Distributional Losses

```python
# KL divergence for discrete distributions
log_probs = F.log_softmax(logits_multi, dim=1)
target_probs = F.softmax(torch.randn(32, 10), dim=1)
kl = nn.KLDivLoss(reduction='batchmean')(log_probs, target_probs)

# KL divergence for VAE (Gaussian)
mu = torch.randn(32, 20)
logvar = torch.randn(32, 20)
kl_vae = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
```

## Probabilistic Interpretation Guide

Every loss function implies a probabilistic model. Choosing a loss is choosing a noise distribution:

| Loss | Implied Model | $p(y \mid x, \theta)$ |
|------|--------------|----------------------|
| MSE | Gaussian noise | $\mathcal{N}(f_\theta(x), \sigma^2)$ |
| MAE | Laplace noise | $\text{Laplace}(f_\theta(x), b)$ |
| Huber | Gaussian core + Laplace tails | Huber distribution |
| BCE | Bernoulli | $\text{Bern}(\sigma(f_\theta(x)))$ |
| Cross-Entropy | Categorical | $\text{Cat}(\text{softmax}(f_\theta(x)))$ |
| Hinge | — | No probabilistic interpretation (geometric) |

If you can specify your beliefs about the data-generating process, the loss function follows from MLE:

$$\mathcal{L}(\theta) = -\frac{1}{m}\sum_{i=1}^m \log p(y^{(i)} \mid x^{(i)}, \theta)$$

## Common Mistakes

### Mistake 1: Applying Activation Before Loss

```python
# WRONG: double sigmoid
probs = torch.sigmoid(logits)
loss = nn.BCEWithLogitsLoss()(probs, targets)  # sigmoid applied twice!

# CORRECT: raw logits
loss = nn.BCEWithLogitsLoss()(logits, targets)
```

```python
# WRONG: softmax before CrossEntropyLoss
probs = F.softmax(logits, dim=1)
loss = nn.CrossEntropyLoss()(probs, targets)  # internal log_softmax + NLL on probs!

# CORRECT: raw logits
loss = nn.CrossEntropyLoss()(logits, targets)
```

### Mistake 2: Wrong Label Format

```python
# WRONG: one-hot labels with CrossEntropyLoss
one_hot = F.one_hot(targets, num_classes=10).float()
loss = nn.CrossEntropyLoss()(logits, one_hot)  # expects integer indices!

# CORRECT: integer class indices
loss = nn.CrossEntropyLoss()(logits, targets)
```

### Mistake 3: Wrong Argument Order

```python
# PyTorch convention: (prediction, target)
loss = nn.MSELoss()(predictions, targets)  # ✓

# Some frameworks use (target, prediction) — be careful
```

### Mistake 4: Using MSE for Classification

```python
# WRONG: MSE for classification
probs = F.softmax(logits, dim=1)
loss = nn.MSELoss()(probs, one_hot_targets)  # poor gradients, slow convergence

# CORRECT: Cross-entropy for classification
loss = nn.CrossEntropyLoss()(logits, targets)
```

MSE gradients for classification vanish when $p \approx 0$ or $p \approx 1$ (the sigmoid/softmax saturation regions), making learning extremely slow. Cross-entropy's gradient $p - y$ does not have this problem.

## Diagnostic Criteria

If training is not progressing as expected, the loss function may be the cause. Consider these diagnostic checks:

**Loss not decreasing:**

- For classification: ensure you are not applying softmax/sigmoid before the loss
- For regression with outliers: switch from MSE to Huber
- Check that labels and predictions have matching shapes and types

**Training unstable (loss oscillating or NaN):**

- Switch from MAE to Huber (smooth gradients near optimum)
- Add gradient clipping
- For custom losses: add epsilon to log arguments

**Model converges but performs poorly:**

- Classification with imbalance: use Focal Loss or class weights
- Segmentation: add Dice Loss to BCE
- Regression with heteroscedastic noise: consider learned variance (negative log-likelihood with $\sigma$ output)

**Model overfits to outliers:**

- Switch from MSE to MAE or Huber
- Add regularization (weight decay)
- Consider robust losses (trimmed mean, Winsorized loss)

## Summary Table

| Loss Function | PyTorch Class | Task | Key Property |
|--------------|---------------|------|-------------|
| MSE | `nn.MSELoss` | Regression | Smooth gradients, Gaussian MLE |
| MAE | `nn.L1Loss` | Regression | Outlier robust, Laplace MLE |
| Huber | `nn.HuberLoss` | Regression | Hybrid MSE/MAE |
| Smooth L1 | `nn.SmoothL1Loss` | Regression | Object detection standard |
| BCE | `nn.BCEWithLogitsLoss` | Binary classification | Bernoulli MLE |
| Cross-Entropy | `nn.CrossEntropyLoss` | Multi-class classification | Categorical MLE |
| NLL | `nn.NLLLoss` | Multi-class (with log-probs) | For beam search/custom softmax |
| Focal | Custom | Imbalanced classification | Down-weights easy examples |
| Hinge | `nn.MultiMarginLoss` | Maximum-margin classification | Sparse gradients |
| KL Div | `nn.KLDivLoss` | Distribution matching | Knowledge distillation |
| Dice | Custom | Segmentation | Overlap optimization |

## Key Takeaways

Loss function selection is a modeling decision, not merely a technical one. The probabilistic interpretation provides the clearest guidance: choose the loss whose implied noise distribution matches your beliefs about the data. For regression, start with Huber as a safe default, then specialize to MSE (clean data) or MAE (heavy outliers). For classification, use `CrossEntropyLoss` for multi-class and `BCEWithLogitsLoss` for binary, adding Focal Loss if class imbalance is severe. Always feed raw logits (not probabilities) to PyTorch's classification losses, and always verify that labels have the expected format. When standard losses are insufficient, custom losses built as `nn.Module` subclasses provide unlimited flexibility while maintaining compatibility with PyTorch's training ecosystem.
