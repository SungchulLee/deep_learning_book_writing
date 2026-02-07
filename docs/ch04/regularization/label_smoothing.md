# Label Smoothing

## Overview

Label smoothing is a regularization technique for classification models that replaces hard one-hot target vectors with soft targets that distribute a small probability mass across all classes. By preventing the model from assigning full probability to the ground-truth class, label smoothing discourages overconfident predictions, improves model calibration, and acts as a form of output regularization that complements weight-based and data-based regularizers.

## Mathematical Formulation

### Hard vs. Soft Targets

For a classification problem with $K$ classes, the standard one-hot target for a sample with true class $k$ is:

$$
y_i^{\text{hard}} = \begin{cases} 1 & \text{if } i = k \\ 0 & \text{if } i \neq k \end{cases}
$$

Label smoothing replaces this with a soft target:

$$
y_i^{\text{smooth}} = \begin{cases} 1 - \varepsilon & \text{if } i = k \\ \frac{\varepsilon}{K - 1} & \text{if } i \neq k \end{cases}
$$

where $\varepsilon \in [0, 1)$ is the smoothing parameter. Equivalently:

$$
y^{\text{smooth}} = (1 - \varepsilon) \, y^{\text{hard}} + \frac{\varepsilon}{K} \, \mathbf{1}
$$

Note the two equivalent formulations: distributing $\varepsilon$ uniformly over the $K-1$ non-target classes (first form), or mixing the one-hot vector with the uniform distribution over all $K$ classes (second form). The first form ensures $\sum_i y_i^{\text{smooth}} = 1$, and so does the second.

### Cross-Entropy with Label Smoothing

The standard cross-entropy loss with hard targets is:

$$
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{K} y_i^{\text{hard}} \log p_i = -\log p_k
$$

With label smoothing:

$$
\mathcal{L}_{\text{LS}} = -\sum_{i=1}^{K} y_i^{\text{smooth}} \log p_i = -(1 - \varepsilon) \log p_k - \frac{\varepsilon}{K-1} \sum_{i \neq k} \log p_i
$$

This can be decomposed as:

$$
\mathcal{L}_{\text{LS}} = (1 - \varepsilon) \, \mathcal{L}_{\text{CE}} + \varepsilon \, \mathcal{L}_{\text{uniform}}
$$

where $\mathcal{L}_{\text{uniform}} = -\frac{1}{K} \sum_{i=1}^{K} \log p_i$ is the cross-entropy with a uniform target distribution.

### KL Divergence Interpretation

The label smoothing loss is equivalent to:

$$
\mathcal{L}_{\text{LS}} = (1 - \varepsilon) \, H(y^{\text{hard}}, p) + \varepsilon \, H(u, p)
$$

where $H(\cdot, \cdot)$ denotes cross-entropy and $u = \frac{1}{K} \mathbf{1}$ is the uniform distribution. Since $H(u, p) = \log K + D_{\text{KL}}(u \| p)$, minimizing label smoothing loss penalizes the KL divergence between the model output and the uniform distribution, which prevents the logits from growing unboundedly.

### Effect on Logits

Without label smoothing, the cross-entropy loss drives the logit of the correct class $z_k \to \infty$ relative to other logits. With label smoothing, the optimal logit configuration satisfies:

$$
z_k - z_j = \log\frac{(1 - \varepsilon)(K - 1)}{\varepsilon} \quad \text{for all } j \neq k
$$

This finite gap prevents the model from becoming infinitely confident and keeps the logit magnitudes bounded.

## Why Label Smoothing Works

### Preventing Overconfidence

Hard targets encourage the model to output $p_k = 1$ for the correct class, which requires logits to grow without bound. This leads to overconfident predictions that generalize poorly, especially when the training data contains label noise or ambiguous examples.

### Improved Calibration

A well-calibrated model's predicted probabilities match empirical frequencies: when the model predicts 80% confidence, it should be correct 80% of the time. Label smoothing improves calibration by preventing the extreme probability values that hard targets encourage.

### Implicit Regularization of Logit Magnitudes

Label smoothing penalizes large logit values, acting similarly to weight decay applied specifically to the output layer. The penalty encourages the model to produce predictions that are confident but not excessively so.

### Clustering Effect on Representations

Müller et al. (2019) showed that label smoothing encourages penultimate-layer representations of the same class to cluster more tightly and equidistantly from representations of other classes, creating more structured and transferable feature spaces.

## PyTorch Implementation

### Using Built-in Cross-Entropy

PyTorch's `CrossEntropyLoss` supports label smoothing directly:

```python
import torch
import torch.nn as nn

# Built-in label smoothing (PyTorch >= 1.10)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Usage
logits = model(inputs)       # (batch_size, num_classes)
loss = criterion(logits, targets)  # targets are class indices (LongTensor)
```

### Manual Implementation

```python
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Args:
        epsilon: Smoothing parameter in [0, 1). Default: 0.1.
        reduction: 'mean', 'sum', or 'none'.
    """
    
    def __init__(self, epsilon: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs, shape (batch_size, num_classes)
            targets: Class indices, shape (batch_size,)
            
        Returns:
            Label-smoothed cross-entropy loss
        """
        num_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Hard target component: -log p_k
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Uniform component: -(1/K) sum_i log p_i
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Combined loss
        loss = (1 - self.epsilon) * nll_loss + self.epsilon * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

### Label Smoothing with Soft Targets

When targets are already probabilities (e.g., from knowledge distillation or Mixup):

```python
class SoftTargetCrossEntropy(nn.Module):
    """
    Cross-entropy loss that accepts soft target distributions.
    
    Supports both one-hot and soft targets. Can optionally apply
    additional label smoothing on top of soft targets.
    """
    
    def __init__(self, epsilon: float = 0.0):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Shape (batch_size, num_classes)
            targets: Shape (batch_size, num_classes) — soft probability targets
        """
        num_classes = logits.size(-1)
        
        # Apply additional smoothing if requested
        if self.epsilon > 0:
            targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1)
        
        return loss.mean()
```

### Integration with Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader


def train_with_label_smoothing(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epsilon: float = 0.1,
    epochs: int = 100,
    lr: float = 0.001
) -> dict:
    """
    Train model with label smoothing.
    
    Args:
        model: Classification network
        train_loader: Training data
        val_loader: Validation data
        epsilon: Label smoothing parameter
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Training history
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=epsilon)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_confidence': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_confidences = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = nn.CrossEntropyLoss()(outputs, y_batch)  # Hard CE for fair comparison
                
                probs = torch.softmax(outputs, dim=-1)
                max_probs, predicted = probs.max(1)
                
                val_loss += loss.item() * X_batch.size(0)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
                all_confidences.append(max_probs)
        
        avg_confidence = torch.cat(all_confidences).mean().item()
        
        history['train_loss'].append(train_loss / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_correct / val_total)
        history['val_confidence'].append(avg_confidence)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Val Acc={val_correct/val_total:.4f}, "
                  f"Avg Confidence={avg_confidence:.4f}")
    
    return history
```

## Measuring Calibration

### Expected Calibration Error (ECE)

```python
import numpy as np


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error.
    
    Args:
        probs: Predicted probabilities for the positive/chosen class, shape (n,)
        labels: Binary correctness indicators, shape (n,)
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value (lower is better calibrated)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        n_in_bin = in_bin.sum()
        
        if n_in_bin > 0:
            avg_confidence = probs[in_bin].mean()
            avg_accuracy = labels[in_bin].mean()
            ece += (n_in_bin / len(probs)) * abs(avg_accuracy - avg_confidence)
    
    return ece


def evaluate_calibration(model, data_loader, device='cpu'):
    """Evaluate model calibration on a dataset."""
    model.eval()
    all_probs = []
    all_correct = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=-1)
            max_probs, predicted = probs.max(1)
            
            all_probs.append(max_probs.cpu().numpy())
            all_correct.append(predicted.eq(y_batch).cpu().numpy())
    
    probs = np.concatenate(all_probs)
    correct = np.concatenate(all_correct)
    
    ece = expected_calibration_error(probs, correct)
    print(f"ECE: {ece:.4f}")
    print(f"Accuracy: {correct.mean():.4f}")
    print(f"Mean confidence: {probs.mean():.4f}")
    
    return ece
```

## Label Smoothing for Specific Architectures

### Transformers

Label smoothing is a standard component of transformer training. The original "Attention Is All You Need" paper uses $\varepsilon = 0.1$:

```python
class TransformerClassifier(nn.Module):
    """Transformer-based classifier with label smoothing."""
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, 
                 num_classes, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, 
            dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)


# Standard transformer training setup
model = TransformerClassifier(
    vocab_size=30000, d_model=512, n_heads=8,
    n_layers=6, num_classes=1000
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

### Knowledge Distillation Connection

Label smoothing is related to self-distillation. The soft targets from label smoothing resemble soft targets produced by a teacher model in knowledge distillation, though with uniform distribution over non-target classes rather than the teacher's learned distribution:

$$
y^{\text{distill}}_i = \begin{cases}
(1 - \varepsilon) + \varepsilon \, p_i^{\text{teacher}} & \text{if } i = k \\
\varepsilon \, p_i^{\text{teacher}} & \text{if } i \neq k
\end{cases}
$$

Label smoothing is the special case where $p^{\text{teacher}}$ is uniform.

## Combining with Other Regularization

Label smoothing interacts with other techniques that modify the training targets:

```python
def combined_augmentation_training_step(
    model, images, labels, criterion,
    use_mixup=True, mixup_alpha=0.2,
    label_smoothing=0.1
):
    """
    Training step combining label smoothing with Mixup.
    
    When using Mixup, reduce label smoothing since Mixup already
    produces soft targets.
    """
    num_classes = 10  # Example
    
    if use_mixup:
        # Mixup produces soft targets — use reduced smoothing
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        index = torch.randperm(images.size(0), device=images.device)
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Create soft targets from Mixup
        targets_a = torch.zeros(images.size(0), num_classes, device=images.device)
        targets_a.scatter_(1, labels.unsqueeze(1), 1.0)
        targets_b = torch.zeros(images.size(0), num_classes, device=images.device)
        targets_b.scatter_(1, labels[index].unsqueeze(1), 1.0)
        
        soft_targets = lam * targets_a + (1 - lam) * targets_b
        
        # Apply mild additional smoothing
        reduced_epsilon = label_smoothing * 0.5
        soft_targets = (1 - reduced_epsilon) * soft_targets + reduced_epsilon / num_classes
        
        logits = model(mixed_images)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
    else:
        logits = model(images)
        loss = criterion(logits, labels)  # criterion has label_smoothing built in
    
    return loss
```

## Practical Guidelines

### Choosing $\varepsilon$

| Scenario | Recommended $\varepsilon$ |
|----------|--------------------------|
| Standard classification | 0.1 |
| Noisy labels | 0.1 – 0.3 |
| Few classes ($K < 10$) | 0.05 – 0.1 |
| Many classes ($K > 100$) | 0.1 – 0.2 |
| With Mixup/CutMix | 0.05 (reduced) |
| Knowledge distillation | 0.0 – 0.05 (teacher provides soft targets) |

### When to Use Label Smoothing

1. **Classification tasks**: Almost always beneficial as a default
2. **Overconfident models**: When predicted probabilities are poorly calibrated
3. **Noisy labels**: Softens the impact of mislabeled examples
4. **Large-scale training**: Standard practice in ImageNet and language model training

### When NOT to Use Label Smoothing

1. **Regression tasks**: Not applicable (targets are continuous)
2. **Binary tasks with extreme class imbalance**: May interfere with learning the rare class
3. **When exact confidence is needed**: Label smoothing systematically reduces confidence
4. **Downstream distillation**: May harm dark knowledge transfer (Müller et al., 2019)

## References

1. Szegedy, C., et al. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR*. (Introduced label smoothing.)
2. Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing Help? *NeurIPS*.
3. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*. (Uses $\varepsilon = 0.1$ for transformer training.)
4. Pereyra, G., et al. (2017). Regularizing Neural Networks by Penalizing Confident Output Distributions. *ICLR Workshop*.
