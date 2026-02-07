# Softmax

## Overview

**Softmax** converts a vector of raw scores (logits) into a probability distribution where all elements are positive and sum to 1. Unlike element-wise activation functions, softmax operates **across a dimension**, introducing dependencies between elements. It is the standard output activation for multiclass classification and a core component of attention mechanisms in transformers.

## Learning Objectives

By the end of this section, you will understand:

1. The mathematical definition and properties of softmax
2. How softmax converts logits to probabilities
3. The log-sum-exp trick for numerical stability
4. Temperature scaling and its effect on the distribution
5. The critical distinction between softmax as an output layer vs in attention
6. PyTorch implementation and integration with loss functions

---

## Mathematical Definition

Given an input vector $\mathbf{z} = [z_1, z_2, \ldots, z_K]^T \in \mathbb{R}^K$, the softmax function produces:

$$\operatorname{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad i = 1, 2, \ldots, K$$

In vector form:

$$\operatorname{softmax}(\mathbf{z}) = \frac{\exp(\mathbf{z})}{\mathbf{1}^T \exp(\mathbf{z})}$$

---

## Properties

| Property | Value |
|----------|-------|
| **Output range** | $(0, 1)$ for each element |
| **Summation** | $\sum_i \operatorname{softmax}(z_i) = 1$ |
| **Element-wise** | ❌ No — outputs depend on all inputs |
| **Monotonic** | Yes (preserves relative ordering) |
| **Translation invariant** | $\operatorname{softmax}(\mathbf{z} + c) = \operatorname{softmax}(\mathbf{z})$ for any scalar $c$ |

### Key Properties in Detail

**Positivity and normalization**: Every output is strictly positive and the outputs sum to 1, so softmax produces a valid probability distribution.

**Order preservation**: If $z_i > z_j$, then $\operatorname{softmax}(z_i) > \operatorname{softmax}(z_j)$. The ranking of logits is preserved.

**Translation invariance**: Adding a constant to all logits does not change the output. This is exploited for numerical stability.

**Proof of translation invariance:**

$$\frac{e^{z_i + c}}{\sum_j e^{z_j + c}} = \frac{e^c \cdot e^{z_i}}{e^c \cdot \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

---

## Jacobian

Unlike element-wise activations where the derivative is a simple scalar, the softmax Jacobian is a full matrix because each output depends on all inputs. Let $p_i = \operatorname{softmax}(z_i)$:

$$\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i(1 - p_i) & \text{if } i = j \\ -p_i \, p_j & \text{if } i \neq j \end{cases}$$

In matrix form:

$$\frac{\partial \mathbf{p}}{\partial \mathbf{z}} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

!!! info "Practical Note"
    You rarely need to compute the Jacobian explicitly. PyTorch's autograd handles this automatically, and `CrossEntropyLoss` combines log-softmax with the loss computation for efficiency.

---

## Numerical Stability: The Log-Sum-Exp Trick

### The Problem

For large logit values, $e^{z_i}$ overflows to infinity. For large negative values, $e^{z_i}$ underflows to zero:

```python
import torch

z = torch.tensor([1000.0, 1001.0, 1002.0])
# Naive computation would give: exp(1000) = inf!
```

### The Solution

Subtract the maximum logit before exponentiating. By translation invariance, this does not change the result:

$$\operatorname{softmax}(z_i) = \frac{e^{z_i - z_{\max}}}{\sum_j e^{z_j - z_{\max}}}$$

where $z_{\max} = \max_j z_j$.

Now the largest exponent is $e^0 = 1$, and all others are $\leq 1$, preventing overflow.

### Stable Implementation

```python
import torch

def softmax_stable(z, dim=-1):
    """Numerically stable softmax."""
    z_max = z.max(dim=dim, keepdim=True).values
    exp_z = torch.exp(z - z_max)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

# PyTorch's built-in softmax already does this
z = torch.tensor([1000.0, 1001.0, 1002.0])
print(torch.softmax(z, dim=0))  # Works correctly!
# Output: tensor([0.0900, 0.2447, 0.6652])
```

### Log-Softmax

For numerical stability in loss computation, log-softmax is preferred over taking $\log(\operatorname{softmax}(\cdot))$ separately:

$$\log \operatorname{softmax}(z_i) = z_i - \log\sum_j e^{z_j} = z_i - z_{\max} - \log\sum_j e^{z_j - z_{\max}}$$

```python
import torch.nn.functional as F

z = torch.randn(4, 10)  # [batch, classes]

# Numerically stable log-softmax
log_probs = F.log_softmax(z, dim=-1)

# Equivalent but less stable:
# log_probs = torch.log(F.softmax(z, dim=-1))  # Avoid this
```

---

## PyTorch Implementation

### Functional API

```python
import torch
import torch.nn.functional as F

z = torch.tensor([[2.0, 1.0, 0.1]])  # [batch=1, classes=3]

# Softmax over the class dimension
probs = F.softmax(z, dim=-1)
print(f"Logits: {z.tolist()}")
print(f"Probs:  {[f'{v:.4f}' for v in probs[0].tolist()]}")
# Probs: ['0.6590', '0.2424', '0.0986']
print(f"Sum:    {probs.sum().item():.4f}")  # 1.0000
```

### Module API

```python
import torch.nn as nn

softmax = nn.Softmax(dim=-1)
probs = softmax(z)
```

!!! warning "Always Specify `dim`"
    `F.softmax(z)` without `dim` will raise a deprecation warning and may default to an unexpected dimension. Always specify `dim` explicitly.

---

## Softmax and CrossEntropyLoss

### Critical Best Practice

In PyTorch, **do not apply softmax before `CrossEntropyLoss`**. The loss function combines log-softmax and negative log-likelihood internally for numerical stability:

```python
import torch
import torch.nn as nn

# ✅ Correct: Return raw logits, let loss handle softmax
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)  # Raw logits — no softmax!

model = Classifier(64, 10)
criterion = nn.CrossEntropyLoss()  # Applies LogSoftmax internally

logits = model(torch.randn(8, 64))   # [batch, classes]
targets = torch.randint(0, 10, (8,))  # [batch]
loss = criterion(logits, targets)
```

```python
# ❌ Wrong: Double softmax
class BadClassifier(nn.Module):
    def forward(self, x):
        logits = self.fc(x)
        return F.softmax(logits, dim=-1)  # Bad! CrossEntropyLoss applies again

# This produces incorrect gradients and poor training
```

### When You Do Need Softmax Explicitly

- **At inference time** to convert logits to probabilities for interpretation
- **In custom loss functions** that expect probabilities
- **In attention mechanisms** (see below)

```python
# Inference-time conversion
model.eval()
with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=-1)
    predicted_class = probs.argmax(dim=-1)
    confidence = probs.max(dim=-1).values
```

---

## Softmax in Attention Mechanisms

Softmax plays a different role in attention: it converts raw attention scores into attention weights that sum to 1 across the key dimension:

$$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Softmax attention."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax converts scores to attention weights
    attn_weights = F.softmax(scores, dim=-1)  # Sum to 1 across keys
    
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

Note: In attention, the $-\infty$ masking before softmax ensures that masked positions receive zero attention weight (since $e^{-\infty} = 0$).

---

## Temperature Scaling

Temperature $\tau > 0$ controls the "sharpness" of the softmax distribution:

$$\operatorname{softmax}(z_i / \tau) = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$$

| Temperature | Effect | Distribution |
|-------------|--------|-------------|
| $\tau \to 0^+$ | Hard (argmax) | Concentrates on the largest logit |
| $\tau = 1$ | Standard | Normal softmax |
| $\tau \to \infty$ | Uniform | All outputs approach $1/K$ |

### Applications

- **Knowledge distillation**: High temperature ($\tau = 2$–$20$) to extract soft targets from a teacher model
- **Sampling in language models**: Temperature controls randomness of text generation
- **Calibration**: Adjusting temperature post-training to calibrate prediction confidence

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]])

for temp in [0.1, 0.5, 1.0, 2.0, 10.0]:
    probs = F.softmax(logits / temp, dim=-1)
    print(f"τ={temp:4.1f}: {[f'{v:.4f}' for v in probs[0].tolist()]}")

# τ= 0.1: ['0.9999', '0.0001', '0.0000']  — nearly one-hot
# τ= 0.5: ['0.9309', '0.0625', '0.0066']  — sharp
# τ= 1.0: ['0.6590', '0.2424', '0.0986']  — standard
# τ= 2.0: ['0.4684', '0.3181', '0.2135']  — softer
# τ=10.0: ['0.3580', '0.3340', '0.3080']  — near uniform
```

---

## Common Dimension Mistakes

```python
x = torch.randn(32, 10)  # [batch, classes]

# ❌ Wrong: softmax over batch dimension
probs_wrong = F.softmax(x, dim=0)  # Probabilities across the BATCH

# ✅ Correct: softmax over class dimension
probs_correct = F.softmax(x, dim=1)  # Probabilities across CLASSES

# For sequences: [batch, seq_len, vocab_size]
x_seq = torch.randn(4, 128, 50000)
probs_seq = F.softmax(x_seq, dim=-1)  # Softmax over vocabulary
```

---

## Softmax vs Sigmoid for Multi-label

| Scenario | Function | Constraint |
|----------|----------|-----------|
| **Multiclass** (exactly one label) | Softmax | $\sum p_i = 1$ |
| **Multi-label** (any subset of labels) | Sigmoid (per element) | Each $p_i$ independent |

```python
# Multiclass: exactly one class (softmax)
logits_multiclass = model_multiclass(x)  # [batch, num_classes]
# Use CrossEntropyLoss (applies softmax internally)

# Multi-label: multiple classes possible (sigmoid)
logits_multilabel = model_multilabel(x)  # [batch, num_labels]
# Use BCEWithLogitsLoss (applies sigmoid independently per label)
```

---

## Summary

| Aspect | Softmax |
|--------|---------|
| **Formula** | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ |
| **Output range** | $(0, 1)$, summing to 1 |
| **Element-wise** | ❌ No (depends on all inputs) |
| **Use in output** | Multiclass classification (via `CrossEntropyLoss`) |
| **Use in attention** | Converts scores to weights |
| **Numerical stability** | Log-sum-exp trick (built into PyTorch) |
| **Temperature** | $\tau$ controls distribution sharpness |

!!! tip "Key Rules"
    1. **Never apply softmax before `CrossEntropyLoss`** — the loss handles it internally
    2. **Always specify `dim`** when calling `F.softmax`
    3. **Use `F.log_softmax`** when you need log-probabilities (more stable than `log(softmax(...))`)
    4. **Use sigmoid, not softmax**, for multi-label classification
