# Mixup

## Overview

Mixup is a data augmentation and regularization technique that trains neural networks on *convex combinations* of pairs of training examples and their labels. By creating virtual training samples that lie between existing data points, Mixup encourages the model to behave linearly between training examples, leading to smoother decision boundaries, improved generalization, and better calibrated predictions.

## Mathematical Formulation

### Core Operation

Given two training examples $(x_i, y_i)$ and $(x_j, y_j)$, Mixup creates a virtual example:

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j
$$

$$
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

where the mixing coefficient $\lambda$ is sampled from a Beta distribution:

$$
\lambda \sim \text{Beta}(\alpha, \alpha), \quad \alpha > 0
$$

The hyperparameter $\alpha$ controls the strength of interpolation. When $\alpha \to 0$, $\lambda$ concentrates at 0 and 1 (no mixing); when $\alpha \to \infty$, $\lambda$ concentrates at 0.5 (maximum mixing).

### Vicinal Risk Minimization

Standard Empirical Risk Minimization (ERM) assumes the training distribution is a set of delta functions centered at the training points:

$$
\mathcal{L}_{\text{ERM}} = \frac{1}{n} \sum_{i=1}^{n} \ell\left(f(x_i), y_i\right)
$$

Mixup implements **Vicinal Risk Minimization (VRM)**, which defines a vicinity distribution around each training point. The Mixup vicinity is the set of all convex combinations:

$$
\mathcal{L}_{\text{Mixup}} = \mathbb{E}_{\lambda \sim \text{Beta}(\alpha, \alpha)} \left[ \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} \ell\left(f(\lambda x_i + (1-\lambda) x_j), \; \lambda y_i + (1-\lambda) y_j\right) \right]
$$

In practice, pairs are sampled by random permutation within each mini-batch rather than computing all $n^2$ pairs.

### Properties of the Beta Distribution

The Beta$(\alpha, \alpha)$ distribution is symmetric around 0.5:

| $\alpha$ | Distribution Shape | Mixing Behavior |
|----------|-------------------|-----------------|
| $\alpha \to 0$ | Concentrated at 0 and 1 | Almost no mixing (recovers ERM) |
| $\alpha = 0.2$ | U-shaped | Mostly one sample, occasionally mixed |
| $\alpha = 1.0$ | Uniform on [0, 1] | All mixing ratios equally likely |
| $\alpha = 2.0$ | Bell-shaped around 0.5 | Mostly equal-weight blends |

For classification, $\alpha \in [0.1, 0.4]$ typically works best, providing enough regularization without overly blurring class boundaries.

### Label Representation

Mixup requires labels to be representable as continuous vectors. For classification with $K$ classes, labels are converted to one-hot vectors before mixing:

$$
y_i = e_{c_i} \in \mathbb{R}^K, \quad \tilde{y} = \lambda \, e_{c_i} + (1-\lambda) \, e_{c_j}
$$

The loss is then computed against these soft targets using cross-entropy:

$$
\ell(\tilde{y}, p) = -\sum_{k=1}^{K} \tilde{y}_k \log p_k = -\lambda \log p_{c_i} - (1-\lambda) \log p_{c_j}
$$

## Why Mixup Works

### Linear Interpolation Prior

Mixup enforces the prior that the model should behave approximately linearly between training examples:

$$
f(\lambda x_i + (1-\lambda) x_j) \approx \lambda f(x_i) + (1-\lambda) f(x_j)
$$

This is a strong but beneficial inductive bias that leads to smoother decision boundaries and reduces oscillations between training points.

### Regularization Effect

Mixup acts as a regularizer by reducing the model's capacity to memorize individual training examples. The network must learn representations that support meaningful interpolation, which favors simpler, more generalizable features.

### Calibration Improvement

By training on a continuous distribution of soft targets rather than hard 0/1 labels, Mixup produces models whose output probabilities are better calibrated — the predicted confidence more closely matches the true accuracy.

### Gradient Analysis

Thulasidasan et al. (2019) showed that Mixup reduces the norm of the Jacobian $\frac{\partial f(x)}{\partial x}$, acting as a form of Jacobian regularization that limits the model's sensitivity to input perturbations.

## PyTorch Implementation

### Basic Mixup

```python
import torch
import torch.nn as nn
import numpy as np


def mixup_data(x: torch.Tensor, y: torch.Tensor, 
               alpha: float = 0.2) -> tuple:
    """
    Apply Mixup to a batch of data.
    
    Args:
        x: Input batch, shape (batch_size, ...)
        y: Labels (class indices), shape (batch_size,)
        alpha: Beta distribution parameter. Higher = more mixing.
        
    Returns:
        mixed_x: Blended inputs
        y_a: Original labels
        y_b: Permuted labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor,
                    lam: float) -> torch.Tensor:
    """
    Compute Mixup loss as weighted combination of two standard losses.
    
    Args:
        criterion: Base loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

### Complete Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader


def train_with_mixup(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    alpha: float = 0.2,
    epochs: int = 100,
    lr: float = 0.001
) -> dict:
    """
    Train model with Mixup augmentation.
    
    Args:
        model: Neural network
        train_loader: Training data
        val_loader: Validation data
        alpha: Mixup interpolation strength
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training with Mixup
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Apply Mixup
            mixed_x, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha)
            
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation (no Mixup)
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Acc={val_correct/val_total:.4f}")
    
    return history
```

## Mixup Variants

### Manifold Mixup

Instead of mixing in input space, Manifold Mixup applies the interpolation to hidden representations at a randomly selected layer:

```python
class ManifoldMixupModel(nn.Module):
    """
    Model supporting Manifold Mixup at random hidden layers.
    
    Reference: Verma et al., "Manifold Mixup: Better Representations by
               Interpolating Hidden States" (ICML 2019)
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        # Build layers as a list for indexing
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ))
            prev_dim = hidden_dim
        self.output = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x, mixup_layer=None, lam=None, index=None):
        """
        Forward pass with optional Manifold Mixup.
        
        Args:
            x: Input tensor
            mixup_layer: Layer index at which to apply Mixup (None = no Mixup)
            lam: Mixing coefficient
            index: Permutation indices for the batch
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply Mixup at the selected layer
            if mixup_layer is not None and i == mixup_layer:
                x = lam * x + (1 - lam) * x[index]
        
        return self.output(x)


def train_step_manifold_mixup(model, X_batch, y_batch, criterion, 
                               optimizer, alpha=0.2):
    """Single training step with Manifold Mixup."""
    optimizer.zero_grad()
    
    # Randomly select a layer for mixing
    n_layers = len(model.layers)
    mixup_layer = np.random.randint(0, n_layers + 1)  # +1 includes input space
    
    # Sample mixing coefficient
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    
    # Create permutation
    batch_size = X_batch.size(0)
    index = torch.randperm(batch_size, device=X_batch.device)
    
    if mixup_layer == 0:
        # Input-space Mixup
        mixed_x = lam * X_batch + (1 - lam) * X_batch[index]
        outputs = model(mixed_x)
    else:
        # Hidden-layer Mixup
        outputs = model(X_batch, mixup_layer=mixup_layer - 1, 
                       lam=lam, index=index)
    
    # Mixed labels
    loss = lam * criterion(outputs, y_batch) + (1 - lam) * criterion(outputs, y_batch[index])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### Batch-Level Mixup Strategies

```python
class BatchMixup:
    """
    Flexible Mixup with different pairing strategies.
    """
    
    def __init__(self, alpha=0.2, strategy='random'):
        """
        Args:
            alpha: Beta distribution parameter
            strategy: Pairing strategy — 'random', 'cross_class', 'same_class'
        """
        self.alpha = alpha
        self.strategy = strategy
    
    def __call__(self, x, y):
        if self.alpha <= 0:
            return x, y, y, 1.0
        
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        
        if self.strategy == 'random':
            index = torch.randperm(batch_size, device=x.device)
        
        elif self.strategy == 'cross_class':
            # Pair each sample with a sample from a different class
            index = self._cross_class_permutation(y)
        
        elif self.strategy == 'same_class':
            # Pair each sample with a sample from the same class
            index = self._same_class_permutation(y)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam
    
    def _cross_class_permutation(self, y):
        """Create permutation pairing different classes."""
        batch_size = y.size(0)
        index = torch.randperm(batch_size, device=y.device)
        
        # Try to ensure cross-class pairing (best effort)
        for i in range(batch_size):
            if y[i] == y[index[i]]:
                # Find a swap partner with different class
                for j in range(i + 1, batch_size):
                    if y[i] != y[index[j]] and y[j] != y[index[i]]:
                        index[i], index[j] = index[j].clone(), index[i].clone()
                        break
        return index
    
    def _same_class_permutation(self, y):
        """Create permutation pairing same classes."""
        batch_size = y.size(0)
        index = torch.arange(batch_size, device=y.device)
        
        # Shuffle within each class
        for c in y.unique():
            class_mask = (y == c).nonzero(as_tuple=True)[0]
            if len(class_mask) > 1:
                perm = class_mask[torch.randperm(len(class_mask))]
                index[class_mask] = perm
        
        return index
```

### Mixup for Regression

Mixup applies directly to regression tasks without any modification to the label mixing:

```python
def mixup_regression(x, y, alpha=0.2):
    """
    Mixup for regression tasks.
    
    Since regression targets are already continuous, the label mixing
    is straightforward interpolation.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y
```

## Combining Mixup with Other Techniques

### Mixup + Label Smoothing

Both produce soft targets. When combined, reduce the label smoothing parameter:

```python
def mixup_with_label_smoothing(model, x, y, alpha=0.2, epsilon=0.05):
    """
    Combine Mixup with mild label smoothing.
    
    Use reduced epsilon since Mixup already softens labels.
    """
    num_classes = 10  # adjust as needed
    
    # Mixup
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    
    # Soft targets from Mixup
    y_onehot = torch.zeros(x.size(0), num_classes, device=x.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    y_onehot_perm = torch.zeros(x.size(0), num_classes, device=x.device)
    y_onehot_perm.scatter_(1, y[index].unsqueeze(1), 1.0)
    
    soft_targets = lam * y_onehot + (1 - lam) * y_onehot_perm
    
    # Apply additional label smoothing
    soft_targets = (1 - epsilon) * soft_targets + epsilon / num_classes
    
    # Compute loss
    logits = model(mixed_x)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1).mean()
    
    return loss
```

### Mixup + CutMix

Randomly choose between Mixup and CutMix per batch:

```python
def mixup_or_cutmix(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, 
                     cutmix_prob=0.5):
    """Randomly apply either Mixup or CutMix per batch."""
    if np.random.random() < cutmix_prob:
        # Apply CutMix (see cutmix.md for implementation)
        return cutmix_data(x, y, alpha=cutmix_alpha)
    else:
        return mixup_data(x, y, alpha=mixup_alpha)
```

## Practical Guidelines

### Hyperparameter Selection

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| $\alpha$ (CIFAR-10) | 0.2 – 1.0 | Higher for smaller datasets |
| $\alpha$ (ImageNet) | 0.1 – 0.4 | 0.2 is standard |
| $\alpha$ (text) | 0.1 – 0.2 | Smaller for discrete data |
| $\alpha$ (regression) | 0.1 – 0.4 | Similar to classification |

### When to Use Mixup

1. **Limited training data**: Mixup is most beneficial when data is scarce
2. **Overconfident models**: When models produce poorly calibrated predictions
3. **Classification with many classes**: Soft targets help with fine-grained distinctions
4. **Adversarial robustness desired**: Mixup improves robustness to small perturbations

### When to Avoid Mixup

1. **Object detection/segmentation**: Spatial label mixing is non-trivial (consider CutMix instead)
2. **Very large $\alpha$**: Heavily mixed examples can confuse the optimizer
3. **Highly structured data**: Where interpolation may not preserve semantic meaning

### Evaluation Note

At evaluation time, Mixup is **never applied**. The model is evaluated on clean, unmodified data.

## References

1. Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR*.
2. Verma, V., et al. (2019). Manifold Mixup: Better Representations by Interpolating Hidden States. *ICML*.
3. Thulasidasan, S., et al. (2019). On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks. *NeurIPS*.
4. Chapelle, O., Weston, J., Bottou, L., & Vapnik, V. (2001). Vicinal Risk Minimization. *NeurIPS*.
