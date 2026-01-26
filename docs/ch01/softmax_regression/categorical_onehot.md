# Categorical Distribution and One-Hot Encoding

## Learning Objectives

By the end of this section, you will be able to:

- Understand the categorical distribution as a generalization of the Bernoulli distribution
- Derive the probability mass function for multi-class classification
- Implement one-hot encoding and understand its mathematical properties
- Connect the categorical distribution to the softmax regression framework
- Apply these concepts in PyTorch for multi-class classification tasks

---

## From Binary to Multi-Class Classification

### The Bernoulli Distribution Revisited

In binary classification, we model outcomes using the Bernoulli distribution. For a random variable $Y \in \{0, 1\}$:

$$P(Y = y) = p^y (1-p)^{1-y}$$

where $p = P(Y=1)$ is the probability of the positive class. This compact notation elegantly captures both cases:

- When $y = 1$: $P(Y=1) = p^1 (1-p)^0 = p$
- When $y = 0$: $P(Y=0) = p^0 (1-p)^1 = 1-p$

### Generalizing to Multiple Classes

For multi-class classification with $K$ classes, we need the **categorical distribution** (also called the **multinoulli** or **generalized Bernoulli** distribution). Let $Y \in \{1, 2, \ldots, K\}$ denote the class label, and let $\boldsymbol{\pi} = (\pi_1, \pi_2, \ldots, \pi_K)$ be the probability vector where:

$$\pi_k = P(Y = k), \quad \sum_{k=1}^{K} \pi_k = 1, \quad \pi_k \geq 0$$

The probability mass function (PMF) is:

$$P(Y = k) = \pi_k$$

---

## One-Hot Encoding: Mathematical Foundation

### Definition and Motivation

**One-hot encoding** transforms a categorical label $y \in \{1, 2, \ldots, K\}$ into a binary vector $\mathbf{y} \in \{0, 1\}^K$ where exactly one element is 1 and all others are 0:

$$\mathbf{y} = \mathbf{e}_k = (0, \ldots, 0, \underbrace{1}_{k\text{-th position}}, 0, \ldots, 0)^T$$

**Example:** For $K = 3$ classes (e.g., cat, dog, bird):
- Class 1 (cat): $\mathbf{y} = (1, 0, 0)^T$
- Class 2 (dog): $\mathbf{y} = (0, 1, 0)^T$
- Class 3 (bird): $\mathbf{y} = (0, 0, 1)^T$

### Mathematical Properties

One-hot vectors have several important properties:

**Property 1: Unit Sum**
$$\sum_{k=1}^{K} y_k = 1$$

**Property 2: Mutual Exclusivity**
$$y_i \cdot y_j = 0 \quad \forall i \neq j$$

**Property 3: Indicator Function**
$$y_k = \mathbb{1}[Y = k] = \begin{cases} 1 & \text{if } Y = k \\ 0 & \text{otherwise} \end{cases}$$

**Property 4: Inner Product with Probability Vector**
$$\mathbf{y}^T \boldsymbol{\pi} = \pi_k \quad \text{when } Y = k$$

This last property is crucial—it allows us to extract the probability of the true class via a simple dot product.

---

## Categorical Distribution with One-Hot Encoding

### Compact PMF Representation

Using one-hot encoding, we can write the categorical PMF in a elegant product form:

$$P(\mathbf{y} | \boldsymbol{\pi}) = \prod_{k=1}^{K} \pi_k^{y_k}$$

This mirrors the Bernoulli formulation but extends naturally to $K$ classes. Let's verify this works:

**Verification:** If the true class is $c$ (so $y_c = 1$ and $y_k = 0$ for $k \neq c$):

$$P(\mathbf{y} | \boldsymbol{\pi}) = \pi_1^0 \cdot \pi_2^0 \cdots \pi_c^1 \cdots \pi_K^0 = \pi_c$$

### The Log-Likelihood

Taking the logarithm of the PMF:

$$\log P(\mathbf{y} | \boldsymbol{\pi}) = \sum_{k=1}^{K} y_k \log \pi_k$$

For a dataset of $N$ independent samples $\{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^{N}$, the log-likelihood becomes:

$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log \pi_k^{(i)}$$

where $\pi_k^{(i)} = P(Y = k | \mathbf{x}^{(i)}; \boldsymbol{\theta})$ depends on the input through model parameters $\boldsymbol{\theta}$.

---

## Connection to Softmax Regression

### The Model Structure

In softmax regression, we model the class probabilities as:

$$\pi_k = P(Y = k | \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$

This is the **softmax function** applied to the **logits** $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$.

### Why This Parameterization?

The softmax function ensures:

1. **Non-negativity:** $\pi_k > 0$ for all $k$ (exponentials are always positive)
2. **Normalization:** $\sum_{k=1}^{K} \pi_k = 1$ (by construction)
3. **Monotonicity:** Higher logits → higher probabilities
4. **Differentiability:** Smooth gradients for optimization

---

## PyTorch Implementation

### Creating One-Hot Encodings

```python
import torch
import torch.nn.functional as F

# Method 1: Using torch.nn.functional.one_hot
def create_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert class indices to one-hot encoded vectors.
    
    Args:
        labels: Tensor of shape (batch_size,) with class indices [0, num_classes-1]
        num_classes: Total number of classes K
    
    Returns:
        One-hot tensor of shape (batch_size, num_classes)
    """
    return F.one_hot(labels, num_classes=num_classes).float()

# Example usage
labels = torch.tensor([0, 2, 1, 0])  # 4 samples with classes 0, 2, 1, 0
one_hot = create_one_hot(labels, num_classes=3)
print("Labels:", labels)
print("One-hot encoding:")
print(one_hot)
```

Output:
```
Labels: tensor([0, 2, 1, 0])
One-hot encoding:
tensor([[1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.]])
```

### Manual One-Hot Implementation

```python
def one_hot_manual(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Manual implementation of one-hot encoding for understanding.
    
    This creates a zero tensor and uses scatter_ to place 1s at the
    appropriate positions.
    """
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    
    # scatter_(dim, index, src) places values from src into positions specified by index
    # Here we scatter 1s along dimension 1 at positions given by labels
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    
    return one_hot

# Verify equivalence
labels = torch.tensor([0, 2, 1])
assert torch.allclose(one_hot_manual(labels, 3), F.one_hot(labels, 3).float())
print("Manual implementation matches F.one_hot!")
```

### Extracting True Class Probabilities

```python
def get_true_class_probs(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Extract predicted probabilities for the true classes.
    
    This is equivalent to: (one_hot * probs).sum(dim=1)
    But more efficient using gather.
    
    Args:
        probs: Predicted probabilities of shape (batch_size, num_classes)
        labels: True class indices of shape (batch_size,)
    
    Returns:
        Probabilities for true classes of shape (batch_size,)
    """
    # gather(dim, index) selects elements along dim using indices
    return probs.gather(1, labels.unsqueeze(1)).squeeze(1)

# Example
probs = torch.tensor([
    [0.7, 0.2, 0.1],  # Sample 0: class 0 has prob 0.7
    [0.1, 0.3, 0.6],  # Sample 1: class 2 has prob 0.6
    [0.2, 0.5, 0.3],  # Sample 2: class 1 has prob 0.5
])
labels = torch.tensor([0, 2, 1])  # True classes

true_probs = get_true_class_probs(probs, labels)
print(f"True class probabilities: {true_probs}")  # tensor([0.7, 0.6, 0.5])
```

---

## Categorical vs One-Hot in PyTorch's CrossEntropyLoss

### A Critical Implementation Detail

**PyTorch's `nn.CrossEntropyLoss` expects class indices, NOT one-hot vectors!**

```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# Logits from model (NOT probabilities!)
logits = torch.tensor([
    [2.0, 1.0, 0.5],
    [0.5, 2.5, 1.0],
])

# CORRECT: Use class indices
labels_indices = torch.tensor([0, 1])
loss_correct = criterion(logits, labels_indices)
print(f"Loss with indices: {loss_correct.item():.4f}")

# WRONG: Using one-hot encoding with standard CrossEntropyLoss
labels_onehot = F.one_hot(labels_indices, num_classes=3).float()
# loss_wrong = criterion(logits, labels_onehot)  # This would raise an error!
```

### When You Need One-Hot Encoding

Use one-hot encoding explicitly when:

1. **Label smoothing:** Soft targets instead of hard targets
2. **Knowledge distillation:** Matching teacher probability distributions
3. **Custom loss functions:** That require explicit probability targets
4. **Visualization:** Understanding model predictions

```python
class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    Requires one-hot encoded targets internally.
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Create smoothed one-hot targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, labels.unsqueeze(1), self.confidence)
        
        # Compute cross-entropy with soft targets
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        
        return loss
```

---

## Numerical Considerations

### The Log-Sum-Exp Trick

When computing with one-hot vectors and log probabilities, numerical stability is crucial:

```python
def stable_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log-softmax computation.
    
    log(softmax(z)) = z - log(sum(exp(z)))
                    = z - max(z) - log(sum(exp(z - max(z))))
    """
    max_logits = logits.max(dim=1, keepdim=True).values
    shifted = logits - max_logits
    log_sum_exp = shifted.exp().sum(dim=1, keepdim=True).log()
    return shifted - log_sum_exp

# PyTorch's built-in is equivalent and optimized
log_probs_builtin = F.log_softmax(logits, dim=1)
log_probs_manual = stable_log_softmax(logits)
print(f"Max difference: {(log_probs_builtin - log_probs_manual).abs().max():.2e}")
```

---

## Complete Example: Categorical Distribution Sampling

```python
import torch
import matplotlib.pyplot as plt

def sample_categorical(probs: torch.Tensor, num_samples: int = 1000) -> torch.Tensor:
    """
    Sample from a categorical distribution.
    
    Args:
        probs: Probability vector of shape (num_classes,)
        num_samples: Number of samples to draw
    
    Returns:
        Tensor of sampled class indices
    """
    # torch.multinomial samples from the categorical distribution
    return torch.multinomial(probs, num_samples, replacement=True)

# Define a categorical distribution
class_probs = torch.tensor([0.5, 0.3, 0.2])  # Cat, Dog, Bird
class_names = ['Cat', 'Dog', 'Bird']

# Sample from the distribution
samples = sample_categorical(class_probs, num_samples=10000)

# Count occurrences
counts = torch.bincount(samples, minlength=3).float()
empirical_probs = counts / counts.sum()

print("True probabilities:", class_probs.numpy())
print("Empirical probabilities:", empirical_probs.numpy())
print("Class counts:", counts.numpy().astype(int))
```

---

## Summary

### Key Concepts

| Concept | Formula/Description |
|---------|---------------------|
| Categorical PMF | $P(Y=k) = \pi_k$ |
| One-hot encoding | $y_k = \mathbb{1}[Y=k]$ |
| Product form PMF | $P(\mathbf{y}|\boldsymbol{\pi}) = \prod_k \pi_k^{y_k}$ |
| Log-likelihood | $\log P(\mathbf{y}|\boldsymbol{\pi}) = \sum_k y_k \log \pi_k$ |
| Probability extraction | $\mathbf{y}^T \boldsymbol{\pi} = \pi_{\text{true class}}$ |

### PyTorch Functions

| Task | PyTorch Function |
|------|------------------|
| Create one-hot | `F.one_hot(labels, num_classes)` |
| Extract probabilities | `probs.gather(1, labels.unsqueeze(1))` |
| Sample categorical | `torch.multinomial(probs, n)` |
| Cross-entropy loss | `nn.CrossEntropyLoss()` (takes indices!) |

### Common Pitfalls

!!! warning "Critical Mistakes to Avoid"
    1. **Don't pass one-hot vectors to `nn.CrossEntropyLoss`** — it expects class indices
    2. **Don't apply softmax before `nn.CrossEntropyLoss`** — it's applied internally
    3. **Be aware of dimension ordering** — PyTorch uses (batch, classes), not (classes, batch)

---

## Next Steps

Now that you understand the categorical distribution and one-hot encoding, you're ready to explore:

1. **Softmax Function Derivation** — How the softmax function converts logits to probabilities
2. **Cross-Entropy Loss** — The connection between negative log-likelihood and cross-entropy
3. **Gradient Computation** — How to derive gradients for softmax regression

---

## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 4
2. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Chapter 3
3. PyTorch Documentation: [torch.nn.functional.one_hot](https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html)
