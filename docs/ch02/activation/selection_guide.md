# Activation Function Selection Guide

## Overview

Choosing the right activation function is crucial for neural network performance. This guide provides a systematic framework for selecting activations based on your architecture, task, and constraints.

## Learning Objectives

By the end of this section, you will understand:

1. How to select activations for different network architectures
2. Task-specific activation choices (classification, regression, etc.)
3. Common pitfalls and anti-patterns
4. Best practices for training stability
5. Implementation guidelines in PyTorch

---

## Quick Reference Decision Tree

```
What are you building?
│
├─ Transformer (NLP/Vision)?
│   ├─ Modern LLM (LLaMA, etc.) ───────> SwiGLU in FFN
│   └─ Standard Transformer ────────────> GELU
│
├─ CNN (Image Classification)?
│   ├─ EfficientNet-style ──────────────> Swish/SiLU
│   ├─ MobileNet/Edge deployment ───────> Hardswish
│   └─ General/ResNet-style ────────────> ReLU + BatchNorm
│
├─ RNN/LSTM/GRU?
│   └─ Gates: Sigmoid, States: Tanh (built-in, don't change)
│
├─ GAN (Generator/Discriminator)?
│   └─ Leaky ReLU (0.2 slope) in both
│
└─ MLP/Fully-Connected?
    ├─ Deep (>5 layers) ────────────────> SELU or ReLU + BatchNorm
    └─ Shallow ─────────────────────────> ReLU
```

---

## Architecture-Specific Guidelines

### Transformers (NLP)

| Component | Recommended | Rationale |
|-----------|-------------|-----------|
| **FFN hidden layer** | GELU | Smooth, works well with attention |
| **Modern LLM FFN** | SwiGLU | Best perplexity, used in LLaMA/PaLM |
| **Output layer** | None (logits) | Use CrossEntropyLoss |

```python
import torch.nn as nn
import torch.nn.functional as F

class TransformerFFN(nn.Module):
    """Standard transformer FFN with GELU."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))  # GELU for transformers
```

### Convolutional Neural Networks

| Component | Recommended | Rationale |
|-----------|-------------|-----------|
| **Conv layers** | ReLU | Simple, fast, well-tested |
| **With BatchNorm** | ReLU | BN handles normalization |
| **Without BatchNorm** | Leaky ReLU or ELU | Prevents dead neurons |
| **Mobile/Edge** | Hardswish | Efficient approximation |
| **EfficientNet-style** | Swish/SiLU | Better accuracy |
| **Output layer** | None (logits) | Use CrossEntropyLoss |

```python
class ConvBlock(nn.Module):
    """Standard conv block with ReLU + BatchNorm."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

### Recurrent Networks (LSTM, GRU)

**Do not modify the internal activations** of LSTM/GRU cells. They are designed with:
- **Gates**: Sigmoid (to produce values in [0, 1])
- **Cell/hidden state**: Tanh (to produce values in [-1, 1])

For layers between RNN and output:
```python
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        # No activation on output - use CrossEntropyLoss
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))  # Logits
```

### Generative Adversarial Networks

| Component | Recommended | Rationale |
|-----------|-------------|-----------|
| **Generator hidden** | ReLU or Leaky ReLU | Stable training |
| **Generator output** | Tanh (images in [-1,1]) | Bounded output |
| **Discriminator hidden** | Leaky ReLU (0.2) | Prevents mode collapse |
| **Discriminator output** | None (use BCEWithLogitsLoss) | Numerical stability |

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, img_channels * 28 * 28),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_channels * 28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # No activation - use BCEWithLogitsLoss
        )
    
    def forward(self, x):
        return self.model(x)
```

---

## Task-Specific Guidelines

### Binary Classification

| Layer | Activation | Loss Function |
|-------|------------|---------------|
| Hidden layers | ReLU, Leaky ReLU, GELU | - |
| Output layer | **None** | `BCEWithLogitsLoss` |

!!! warning "Common Mistake"
    **Never use Sigmoid + BCELoss**. Always use logits + BCEWithLogitsLoss for numerical stability.

```python
# ✅ Correct
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 1)  # No activation!
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Logits

model = BinaryClassifier()
criterion = nn.BCEWithLogitsLoss()  # Applies sigmoid internally

# ❌ Wrong - Numerical issues
class BadBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))  # Bad!

criterion = nn.BCELoss()  # Will have numerical issues
```

### Multiclass Classification

| Layer | Activation | Loss Function |
|-------|------------|---------------|
| Hidden layers | ReLU, Leaky ReLU, GELU | - |
| Output layer | **None** | `CrossEntropyLoss` |

!!! warning "Common Mistake"
    **Never use Softmax + CrossEntropyLoss**. CrossEntropyLoss applies LogSoftmax internally.

```python
# ✅ Correct
class MulticlassClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, num_classes)  # No activation!
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Logits

criterion = nn.CrossEntropyLoss()  # Applies LogSoftmax internally

# ❌ Wrong - Double softmax
def bad_forward(self, x):
    x = F.relu(self.fc1(x))
    return F.softmax(self.fc2(x), dim=-1)  # Bad!
```

### Regression

| Layer | Activation | Loss Function |
|-------|------------|---------------|
| Hidden layers | ReLU, Leaky ReLU | - |
| Output layer | **None** (unbounded) | `MSELoss` |
| Output layer | **Sigmoid** (bounded [0,1]) | `MSELoss` |
| Output layer | **Tanh** (bounded [-1,1]) | `MSELoss` |
| Output layer | **Softplus** (positive only) | `MSELoss` |

```python
class Regressor(nn.Module):
    def __init__(self, output_bounds='none'):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 1)
        self.output_bounds = output_bounds
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        if self.output_bounds == 'none':
            return x  # Unbounded
        elif self.output_bounds == 'positive':
            return F.softplus(x)  # [0, ∞)
        elif self.output_bounds == 'unit':
            return torch.sigmoid(x)  # [0, 1]
        elif self.output_bounds == 'symmetric':
            return torch.tanh(x)  # [-1, 1]
```

---

## Common Anti-Patterns

### 1. Double Activation at Output

```python
# ❌ Bad: Sigmoid applied twice
logits = model(x)
probs = torch.sigmoid(logits)  # Already applied!
loss = F.binary_cross_entropy_with_logits(probs, targets)  # Applies again!

# ✅ Good
logits = model(x)
loss = F.binary_cross_entropy_with_logits(logits, targets)
```

### 2. Sigmoid/Tanh in Deep Hidden Layers

```python
# ❌ Bad: Vanishing gradients
class DeepSigmoidNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 64), nn.Sigmoid(),  # Bad!
            nn.Linear(64, 64), nn.Sigmoid(),  # Bad!
            nn.Linear(64, 64), nn.Sigmoid(),  # Bad!
            nn.Linear(64, 10)
        )

# ✅ Good: Use ReLU family
class DeepReLUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
```

### 3. Wrong Softmax Dimension

```python
x = torch.randn(32, 10)  # [batch, classes]

# ❌ Wrong dimension
probs_wrong = F.softmax(x, dim=0)  # Softmax over batch!

# ✅ Correct dimension
probs_correct = F.softmax(x, dim=1)  # Softmax over classes
```

### 4. Activation Before BatchNorm

The order matters for training dynamics:

```python
# Conventional order (most common)
x = F.relu(self.bn(self.conv(x)))  # Conv → BN → ReLU

# Pre-activation (ResNet v2)
x = self.conv(F.relu(self.bn(x)))  # BN → ReLU → Conv
```

---

## Training Stability Guidelines

### Weight Initialization

| Activation | Initialization | PyTorch |
|------------|----------------|---------|
| ReLU, Leaky ReLU | He (Kaiming) | `kaiming_normal_` |
| Tanh | Xavier (Glorot) | `xavier_normal_` |
| SELU | LeCun Normal | `lecun_normal` (custom) |
| Linear output | Xavier or Default | `xavier_normal_` |

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

model.apply(init_weights)
```

### Gradient Clipping

For activations that can produce unbounded outputs:

```python
optimizer.zero_grad()
loss.backward()

# Clip gradients to prevent explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

### Batch Normalization Placement

```python
class StableBlock(nn.Module):
    """Block with proper normalization for training stability."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # BN after linear, before activation
        return self.relu(self.bn(self.fc(x)))
```

---

## Performance Comparison

### Relative Computational Cost

| Activation | Relative Cost | Notes |
|------------|---------------|-------|
| ReLU | 1.0× | Baseline |
| Leaky ReLU | 1.0× | Same as ReLU |
| Sigmoid | 2-3× | Requires exp |
| Tanh | 2-3× | Requires exp |
| GELU (approx) | 1.5× | With tanh approximation |
| GELU (exact) | 2× | With error function |
| Swish/SiLU | 2× | Requires sigmoid |
| Hardswish | 1.2× | Piecewise linear |

### Accuracy vs Speed Trade-off

For CNNs on ImageNet (rough comparison):

| Activation | Accuracy | Speed |
|------------|----------|-------|
| ReLU | Baseline | Fastest |
| Swish | +0.3-0.5% | Slower |
| GELU | +0.2-0.4% | Slower |
| Hardswish | +0.2-0.3% | Fast |

---

## Summary Tables

### Hidden Layers

| Architecture | First Choice | Alternative |
|--------------|--------------|-------------|
| CNN | ReLU | Leaky ReLU, Swish |
| Transformer | GELU | SwiGLU |
| MLP | ReLU | Leaky ReLU, SELU |
| GAN | Leaky ReLU | ReLU |

### Output Layers

| Task | Activation | Loss |
|------|------------|------|
| Binary classification | None | BCEWithLogitsLoss |
| Multiclass classification | None | CrossEntropyLoss |
| Regression (unbounded) | None | MSELoss |
| Regression ([0,1]) | Sigmoid | MSELoss |
| Regression ([-1,1]) | Tanh | MSELoss |
| Generation (images) | Tanh | Various |

---

## Checklist for New Projects

- [ ] **Hidden layers**: Start with ReLU (CNNs) or GELU (Transformers)
- [ ] **Output layer**: No activation for classification with appropriate loss
- [ ] **Initialization**: He for ReLU, Xavier for Tanh
- [ ] **Loss function**: BCEWithLogitsLoss or CrossEntropyLoss
- [ ] **BatchNorm**: Add for training stability in CNNs
- [ ] **Test baseline**: Verify with standard activations before experimenting
- [ ] **Monitor**: Watch for dead neurons, vanishing/exploding gradients

!!! tip "Final Advice"
    When in doubt, start with **ReLU for CNNs** and **GELU for Transformers**. These are battle-tested defaults. Only experiment with alternatives when you have a specific reason (dead neurons, instability, or seeking marginal improvements).
