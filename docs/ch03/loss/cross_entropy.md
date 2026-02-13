# Cross-Entropy Loss

## Learning Objectives

By the end of this section, you will be able to:

- Derive cross-entropy loss from maximum likelihood estimation principles
- Understand the information-theoretic interpretation of cross-entropy
- Prove the equivalence between NLL and cross-entropy in multiclass classification
- Connect KL divergence to cross-entropy optimization
- Derive the complete gradient of cross-entropy loss with respect to model parameters
- Verify the elegant gradient formula $\nabla_\mathbf{z}\mathcal{L} = \hat{\boldsymbol{\pi}} - \mathbf{y}$ in PyTorch

!!! note "See Also"
    Cross-entropy also appears in **Section 3.4 Softmax Regression**, where it is derived as the natural loss for the softmax classifier and demonstrated through an N-gram language model with PyTorch's three interfaces (`nn.CrossEntropyLoss`, `F.cross_entropy`, `nn.NLLLoss`). This section provides the broader treatment: information-theoretic foundations, full batch gradient derivation with matrix forms, NumPy implementation from scratch, and connections to focal loss.

---

## The Maximum Likelihood Framework

### Setting Up the Problem

In multiclass classification with $K$ classes, we have:

- **Data:** $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{N}$ where $y^{(i)} \in \{1, \ldots, K\}$
- **Model:** Predicts probabilities $\hat{\pi}_k^{(i)} = P(Y = k \mid \mathbf{x}^{(i)};\, \boldsymbol{\theta})$
- **Goal:** Find parameters $\boldsymbol{\theta}$ that maximize the likelihood of observed data

### The Likelihood Function

Assuming independent samples, the likelihood is:

$$\mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^{N} P\bigl(Y = y^{(i)} \mid \mathbf{x}^{(i)};\, \boldsymbol{\theta}\bigr) = \prod_{i=1}^{N} \hat{\pi}_{y^{(i)}}^{(i)}$$

Using one-hot encoding $\mathbf{y}^{(i)}$ where $y_k^{(i)} = \mathbb{1}[y^{(i)} = k]$:

$$\mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^{N} \prod_{k=1}^{K} \bigl(\hat{\pi}_k^{(i)}\bigr)^{y_k^{(i)}}$$

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

Entropy is minimized ($= 0$) when the distribution is deterministic and maximized ($= \log K$) when uniform.

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

This is exactly $\frac{1}{N} \text{NLL}(\boldsymbol{\theta})$:

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

Cross-entropy equals the inherent uncertainty in $\mathbf{p}$ (entropy) plus the "extra cost" of using $\mathbf{q}$ instead of $\mathbf{p}$ (KL divergence).

### In Classification (One-Hot Labels)

When $\mathbf{y}$ is one-hot, $H(\mathbf{y}) = 0$ (no uncertainty in the label). Therefore:

$$H(\mathbf{y}, \hat{\boldsymbol{\pi}}) = D_{KL}(\mathbf{y} \| \hat{\boldsymbol{\pi}})$$

**Minimizing cross-entropy = minimizing KL divergence from true labels.**

---

## Geometric Interpretation

For a single sample with true class $c$, the cross-entropy loss is:

$$\mathcal{L} = -\log \hat{\pi}_c$$

where $\hat{\pi}_c = \sigma(\mathbf{z})_c$ is the softmax probability.

**Properties:**

- $\mathcal{L} = 0$ when $\hat{\pi}_c = 1$ (perfect prediction)
- $\mathcal{L} \to \infty$ when $\hat{\pi}_c \to 0$ (completely wrong)
- $\mathcal{L} = \log K$ when $\hat{\pi}_c = 1/K$ (uniform, random guessing)

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

## Gradient Derivation: Step by Step

### The Softmax Regression Model

For multiclass classification with $K$ classes and input features $\mathbf{x} \in \mathbb{R}^D$:

**Logits (linear scores):**
$$z_k = \mathbf{w}_k^T \mathbf{x} + b_k = \sum_{d=1}^{D} w_{kd} x_d + b_k$$

**Predicted probabilities (softmax):**
$$\hat{\pi}_k = \sigma(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**Parameters:** $\boldsymbol{\theta} = \{\mathbf{W}, \mathbf{b}\}$ where $\mathbf{W} \in \mathbb{R}^{K \times D}$ and $\mathbf{b} \in \mathbb{R}^K$.

**Loss function** for a single sample with true class $c$ (one-hot encoded as $\mathbf{y}$):

$$\mathcal{L} = -\log \hat{\pi}_c = -\sum_{k=1}^{K} y_k \log \hat{\pi}_k$$

### Step 1: Gradient w.r.t. Logits

We compute $\frac{\partial \mathcal{L}}{\partial z_j}$ using the chain rule:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{k=1}^{K} \frac{\partial \mathcal{L}}{\partial \hat{\pi}_k} \cdot \frac{\partial \hat{\pi}_k}{\partial z_j}$$

**Computing $\frac{\partial \mathcal{L}}{\partial \hat{\pi}_k}$:**

$$\frac{\partial \mathcal{L}}{\partial \hat{\pi}_k} = -\frac{y_k}{\hat{\pi}_k}$$

**Using the softmax Jacobian:**

$$\frac{\partial \hat{\pi}_k}{\partial z_j} = \hat{\pi}_k(\delta_{kj} - \hat{\pi}_j)$$

**Combining:**

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{k=1}^{K} \left(-\frac{y_k}{\hat{\pi}_k}\right) \cdot \hat{\pi}_k(\delta_{kj} - \hat{\pi}_j) = -\sum_{k=1}^{K} y_k(\delta_{kj} - \hat{\pi}_j)$$

$$= -\sum_{k=1}^{K} y_k \delta_{kj} + \hat{\pi}_j \sum_{k=1}^{K} y_k$$

Since $\sum_k y_k = 1$ (one-hot) and $\sum_k y_k \delta_{kj} = y_j$:

$$\frac{\partial \mathcal{L}}{\partial z_j} = -y_j + \hat{\pi}_j = \hat{\pi}_j - y_j$$

### The Beautiful Result

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\boldsymbol{\pi}} - \mathbf{y}}$$

The gradient is simply the **difference between predicted probabilities and true labels**.

**Interpretation:**

- If $\hat{\pi}_c \approx 1$ (correct prediction): gradient $\approx 0$ (small update)
- If $\hat{\pi}_c \approx 0$ (wrong prediction): gradient is large (big update)
- The gradient "pushes" predictions toward the true label

### Step 2: Gradient w.r.t. Weights

Using the chain rule with $z_k = \sum_d w_{kd} x_d + b_k$:

$$\frac{\partial \mathcal{L}}{\partial w_{kd}} = \frac{\partial \mathcal{L}}{\partial z_k} \cdot \frac{\partial z_k}{\partial w_{kd}} = (\hat{\pi}_k - y_k) \cdot x_d$$

**In matrix form:**

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = (\hat{\boldsymbol{\pi}} - \mathbf{y}) \mathbf{x}^T}$$

where $(\hat{\boldsymbol{\pi}} - \mathbf{y}) \in \mathbb{R}^K$ and $\mathbf{x} \in \mathbb{R}^D$, giving $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} \in \mathbb{R}^{K \times D}$.

### Step 3: Gradient w.r.t. Biases

$$\frac{\partial \mathcal{L}}{\partial b_k} = \frac{\partial \mathcal{L}}{\partial z_k} \cdot \frac{\partial z_k}{\partial b_k} = (\hat{\pi}_k - y_k) \cdot 1$$

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \hat{\boldsymbol{\pi}} - \mathbf{y}}$$

---

## Batch Gradient Computation

Given a batch of $N$ samples $\{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^{N}$, the total loss is:

$$\mathcal{L}_{\text{total}} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}^{(i)}$$

Let $\mathbf{X} \in \mathbb{R}^{N \times D}$ (samples as rows), $\hat{\mathbf{P}} \in \mathbb{R}^{N \times K}$ (predicted probs), $\mathbf{Y} \in \mathbb{R}^{N \times K}$ (one-hot labels):

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{1}{N} (\hat{\mathbf{P}} - \mathbf{Y})^T \mathbf{X}}$$

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \frac{1}{N} (\hat{\mathbf{P}} - \mathbf{Y})^T \mathbf{1}}$$

### Gradient Properties and Intuition

**Gradient magnitude** depends on prediction confidence:

| True Class Prob $\hat{\pi}_c$ | Gradient Magnitude | Interpretation |
|-------------------------------|-------------------|----------------|
| 0.99 | Small ($\approx 0.01$) | Confident correct → small update |
| 0.50 | Medium ($\approx 0.50$) | Uncertain → moderate update |
| 0.01 | Large ($\approx 0.99$) | Confident wrong → large update |

**Gradient direction:** The true class gradient is negative (pushes logit up), while other class gradients are positive (pushes logits down). The net effect increases separation between the true class and others.

**Gradient boundedness:** $\|\nabla_\mathbf{z} \mathcal{L}\|_2 = \|\hat{\boldsymbol{\pi}} - \mathbf{y}\|_2 \leq \sqrt{2}$, which helps training stability.

---

## Common Variants and Extensions

### Binary Cross-Entropy

For binary classification ($K = 2$), cross-entropy simplifies to:

$$\text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y^{(i)} \log \hat{p}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$

### Weighted Cross-Entropy

For imbalanced classes, weight the loss by class frequency:

$$\text{Weighted CE} = -\frac{1}{N} \sum_{i=1}^{N} w_{y^{(i)}} \log \hat{\pi}_{y^{(i)}}^{(i)}$$

### Label Smoothing Cross-Entropy

Instead of one-hot targets, use smoothed targets:

$$\tilde{y}_k = \begin{cases}
1 - \epsilon & \text{if } k = c \text{ (true class)} \\
\frac{\epsilon}{K-1} & \text{otherwise}
\end{cases}$$

### Focal Loss

Addresses class imbalance by down-weighting easy examples:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $\gamma > 0$ is the focusing parameter and $\alpha_t$ is the weighting factor.

---

## With L2 Regularization

### Regularized Loss

$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{CE} + \frac{\lambda}{2} \|\mathbf{W}\|_F^2$$

### Regularized Gradient

$$\frac{\partial \mathcal{L}_{\text{reg}}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}_{CE}}{\partial \mathbf{W}} + \lambda \mathbf{W}$$

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

### Loss Variants in PyTorch

```python
# With class weights (for imbalanced data)
class_weights = torch.tensor([1.0, 2.0, 3.0])
weighted_ce = nn.CrossEntropyLoss(weight=class_weights)

# With label smoothing
smooth_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

# Ignoring certain labels (e.g., padding)
ignore_ce = nn.CrossEntropyLoss(ignore_index=-100)

# Focal loss (manual implementation)
def focal_loss(logits, targets, alpha=1.0, gamma=2.0):
    """
    Focal Loss for dense object detection.

    Args:
        logits: Raw model outputs
        targets: Ground truth class indices
        alpha: Weighting factor
        gamma: Focusing parameter
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # pt = probability of true class
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * ce_loss).mean()
```

### Verifying Gradient Derivation

```python
def verify_gradient_derivation():
    """
    Verify our analytical gradients match PyTorch autograd.
    """
    torch.manual_seed(42)

    # Setup
    N, D, K = 4, 5, 3  # 4 samples, 5 features, 3 classes

    X = torch.randn(N, D)
    y = torch.randint(0, K, (N,))

    W = torch.randn(K, D, requires_grad=True)
    b = torch.randn(K, requires_grad=True)

    # Forward pass
    logits = X @ W.T + b  # (N, K)
    probs = F.softmax(logits, dim=1)

    # Loss (cross-entropy)
    loss = F.cross_entropy(logits, y)

    # Autograd backward
    loss.backward()

    # Our analytical gradients
    y_onehot = F.one_hot(y, K).float()
    dz = probs.detach() - y_onehot  # (N, K)

    dW_analytical = (1/N) * dz.T @ X  # (K, D)
    db_analytical = (1/N) * dz.sum(dim=0)  # (K,)

    # Compare
    print("Gradient Verification")
    print("=" * 50)
    print(f"dW max error: {(W.grad - dW_analytical).abs().max().item():.2e}")
    print(f"db max error: {(b.grad - db_analytical).abs().max().item():.2e}")
    print(f"Gradients match: {torch.allclose(W.grad, dW_analytical, atol=1e-5)}")

verify_gradient_derivation()
```

### Visualizing the Gradient Flow

```python
def visualize_gradient_flow():
    """
    Show how gradients flow through softmax + cross-entropy.
    """
    torch.manual_seed(42)

    # Single sample for clarity
    logits = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
    true_class = 0

    # Forward
    probs = F.softmax(logits, dim=0)
    loss = -torch.log(probs[true_class])

    # Backward
    loss.backward()

    print("Gradient Flow Visualization")
    print("=" * 50)
    print(f"Logits z:        {logits.detach().numpy().round(4)}")
    print(f"Probabilities π: {probs.detach().numpy().round(4)}")
    print(f"True class:      {true_class}")
    print(f"Loss:            {loss.item():.4f}")
    print()
    print(f"∂L/∂z (autograd):   {logits.grad.numpy().round(4)}")

    # Analytical: π - y
    y_onehot = torch.zeros(3)
    y_onehot[true_class] = 1
    grad_analytical = probs.detach() - y_onehot
    print(f"∂L/∂z (analytical): {grad_analytical.numpy().round(4)}")
    print()
    print("Note: ∂L/∂z = π - y (predicted minus true)")

visualize_gradient_flow()
```

### NumPy Implementation from Scratch

```python
import numpy as np

class SoftmaxRegressionNumPy:
    """
    Softmax regression implemented from scratch.
    Demonstrates the gradient derivations in code.
    """

    def __init__(self, input_dim: int, num_classes: int):
        self.W = np.random.randn(num_classes, input_dim) * 0.01
        self.b = np.zeros(num_classes)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: X → logits → probabilities."""
        self.z = X @ self.W.T + self.b  # (N, K)
        self.probs = self.softmax(self.z)  # (N, K)
        return self.probs

    def compute_loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        N = len(y)
        correct_log_probs = -np.log(probs[np.arange(N), y] + 1e-10)
        return np.mean(correct_log_probs)

    def backward(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Backward pass: compute gradients.

        The key insight: ∂L/∂z = π - y (one-hot)
        """
        N = len(y)

        # Convert y to one-hot encoding
        y_onehot = np.zeros_like(self.probs)
        y_onehot[np.arange(N), y] = 1

        # Gradient w.r.t. logits: dL/dz = π - y
        dz = self.probs - y_onehot  # (N, K)

        # Gradient w.r.t. weights: dL/dW = (1/N) * dz^T @ X
        dW = (1/N) * dz.T @ X  # (K, D)

        # Gradient w.r.t. biases: dL/db = (1/N) * sum(dz)
        db = (1/N) * np.sum(dz, axis=0)  # (K,)

        return dW, db

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        """Single training step: forward, backward, update."""
        probs = self.forward(X)
        loss = self.compute_loss(probs, y)
        dW, db = self.backward(X, y)
        self.W -= lr * dW
        self.b -= lr * db
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

---

## PyTorch Quick Reference

| Function | Input | Notes |
|----------|-------|-------|
| `nn.CrossEntropyLoss()` | Logits | Most common, numerically stable |
| `F.cross_entropy()` | Logits | Functional version |
| `nn.NLLLoss()` | Log-probabilities | Use with `log_softmax` |
| `F.log_softmax()` | Logits | Stable log-probabilities |
| `nn.BCEWithLogitsLoss()` | Logits | Binary classification |

---

## Summary

### The Fundamental Equations

**Cross-Entropy Loss:**

$$\boxed{\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{\pi}_{y^{(i)}}^{(i)} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log \hat{\pi}_k^{(i)}}$$

**Equivalences:**

$$\text{Cross-Entropy} = \frac{1}{N} \text{NLL} = H(\mathbf{p}, \mathbf{q}) = H(\mathbf{p}) + D_{KL}(\mathbf{p} \| \mathbf{q})$$

### Gradient Summary

| Quantity | Single Sample | Batch ($N$ samples) |
|----------|--------------|---------------------|
| w.r.t. logits | $\hat{\boldsymbol{\pi}} - \mathbf{y}$ | — |
| w.r.t. weights | $(\hat{\boldsymbol{\pi}} - \mathbf{y})\mathbf{x}^T$ | $\frac{1}{N}(\hat{\mathbf{P}} - \mathbf{Y})^T \mathbf{X}$ |
| w.r.t. biases | $\hat{\boldsymbol{\pi}} - \mathbf{y}$ | $\frac{1}{N}\mathbf{1}^T(\hat{\mathbf{P}} - \mathbf{Y})$ |

### The Key Insight

$$\boxed{\text{Gradient} = \text{Predicted} - \text{True}}$$

This simple formula is why softmax + cross-entropy is so widely used.

### Three Perspectives on Cross-Entropy

!!! info "Three Perspectives"
    1. **Statistical:** Maximizing likelihood of observed labels
    2. **Information-theoretic:** Minimizing coding inefficiency
    3. **Geometric:** Measuring "distance" between distributions (via KL divergence)

---

## References

1. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*, Chapter 2.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 6.2.2.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 4.3.
4. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Chapter 8.6.
5. Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
6. PyTorch Documentation: [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
