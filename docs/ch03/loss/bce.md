# Binary Cross-Entropy

Binary Cross-Entropy (BCE) is the specialization of cross-entropy loss to two-class problems. It arises from maximum likelihood estimation under a Bernoulli model and is the standard loss for binary classification tasks such as spam detection, medical diagnosis, and sentiment analysis. This section derives BCE from the Bernoulli likelihood, covers the critical distinction between `nn.BCELoss` and `nn.BCEWithLogitsLoss`, and demonstrates practical usage patterns.

## Mathematical Foundation

### Bernoulli Model

In binary classification, the model predicts a probability $p \in [0, 1]$ for the positive class ($y = 1$). The label $y \in \{0, 1\}$ is modeled as a Bernoulli random variable:

$$P(y \mid p) = p^y (1 - p)^{1-y}$$

When $y = 1$, this equals $p$; when $y = 0$, it equals $1 - p$.

### Likelihood and Log-Likelihood

For $m$ independent observations $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$ with predicted probabilities $p^{(i)} = \sigma(f_\theta(x^{(i)}))$:

$$L(\theta) = \prod_{i=1}^m \left(p^{(i)}\right)^{y^{(i)}} \left(1 - p^{(i)}\right)^{1 - y^{(i)}}$$

Taking the logarithm:

$$\ell(\theta) = \sum_{i=1}^m \left[y^{(i)} \log p^{(i)} + (1 - y^{(i)}) \log(1 - p^{(i)})\right]$$

### Cost Function

Negating and averaging gives the Binary Cross-Entropy:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)} \log p^{(i)} + (1 - y^{(i)}) \log(1 - p^{(i)})\right]$$

This is the negative log-likelihood of the Bernoulli model, scaled by $\frac{1}{m}$.

### Per-Sample Behavior

The loss for a single sample depends on the true label:

$$\ell(p, y) = \begin{cases} -\log(p) & \text{if } y = 1 \\ -\log(1 - p) & \text{if } y = 0 \end{cases}$$

Both terms have the same structure: they penalize the model when the predicted probability diverges from the true label. As $p \to y$, the loss approaches 0. As $p$ moves away from $y$, the loss increases without bound (approaching $+\infty$ when $p \to 1 - y$).

## Logits and the Sigmoid Function

Neural networks typically output raw, unbounded values called **logits** before any activation function. The sigmoid function converts logits to probabilities:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

The logit is the inverse of the sigmoid, mapping probabilities to the real line:

$$z = \sigma^{-1}(p) = \log\frac{p}{1 - p}$$

This is the **log-odds** of the positive class.

```python
import torch

# Raw model outputs (logits)
logits = torch.tensor([-2.0, 3.0, -1.5, 2.5, 0.5])

# Convert to probabilities
probabilities = torch.sigmoid(logits)
print(f"Logits:        {logits}")
print(f"Probabilities: {probabilities}")
# Interpretation:
# logit = 0  → probability = 0.5 (maximum uncertainty)
# logit > 0  → probability > 0.5 (predicts positive class)
# logit < 0  → probability < 0.5 (predicts negative class)
```

## Gradient of BCE with Sigmoid

When the sigmoid is composed with BCE, the gradient with respect to the logit $z$ simplifies elegantly:

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = \sigma(z) - y = p - y$$

This is identical in form to the softmax–cross-entropy gradient and enjoys the same computational simplicity. The gradient is bounded in $(-1, 1)$ and vanishes only when the prediction is perfect.

### Derivation

Starting from $\mathcal{L} = -[y\log\sigma(z) + (1-y)\log(1-\sigma(z))]$ and using $\sigma'(z) = \sigma(z)(1-\sigma(z))$:

$$\frac{\partial \mathcal{L}}{\partial z} = -\left[\frac{y \cdot \sigma(z)(1-\sigma(z))}{\sigma(z)} - \frac{(1-y) \cdot \sigma(z)(1-\sigma(z))}{1-\sigma(z)}\right] = -[y(1-\sigma(z)) - (1-y)\sigma(z)]$$

$$= -y + y\sigma(z) + \sigma(z) - y\sigma(z) = \sigma(z) - y$$

## PyTorch Interfaces

### `nn.BCELoss`: Expects Probabilities

`nn.BCELoss` takes **probabilities** (post-sigmoid values in $[0, 1]$) as input:

```python
import torch
import torch.nn as nn

true_labels = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float32)

# Model predictions (probabilities after sigmoid)
predicted_probs = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.6])

bce_criterion = nn.BCELoss()
bce_loss = bce_criterion(predicted_probs, true_labels)
print(f"BCE Loss: {bce_loss.item():.4f}")  # ~0.23

# Per-sample analysis
for i in range(len(true_labels)):
    y = true_labels[i].item()
    p = predicted_probs[i].item()
    sample_loss = -(y * torch.log(torch.tensor(p)) +
                    (1 - y) * torch.log(torch.tensor(1 - p)))
    print(f"  Sample {i+1}: y={int(y)}, p={p:.2f} → loss={sample_loss.item():.4f}")
```

### `nn.BCEWithLogitsLoss` (Recommended)

`nn.BCEWithLogitsLoss` combines sigmoid activation and BCE loss in a single numerically stable operation:

```python
logits = torch.tensor([-2.0, 3.0, -1.5, 2.5, 0.5])

bce_with_logits = nn.BCEWithLogitsLoss()
loss_from_logits = bce_with_logits(logits, true_labels)

# Equivalent to (but more stable than):
manual_loss = nn.BCELoss()(torch.sigmoid(logits), true_labels)

print(f"BCEWithLogitsLoss: {loss_from_logits.item():.4f}")
print(f"Manual approach:   {manual_loss.item():.4f}")
# Both produce the same result
```

!!! warning "Common Mistake"
    Never apply sigmoid before `BCEWithLogitsLoss`—it already includes sigmoid internally. Applying sigmoid twice produces incorrect results.

### Why `BCEWithLogitsLoss` Is More Stable

The numerical instability in the manual approach arises when $\sigma(z)$ is very close to 0 or 1. Computing $\log(\sigma(z))$ when $\sigma(z) \approx 0$ yields $-\infty$, and computing $\log(1 - \sigma(z))$ when $\sigma(z) \approx 1$ also yields $-\infty$.

`BCEWithLogitsLoss` avoids this by using the identity:

$$-[y\log\sigma(z) + (1-y)\log(1-\sigma(z))] = \max(z, 0) - yz + \log(1 + e^{-|z|})$$

This formulation never takes the logarithm of a number close to zero.

## Class Weights and Pos Weight

### Handling Class Imbalance

When positive examples are rare (e.g., 1% positive in fraud detection), the `pos_weight` parameter scales the loss for positive samples:

```python
# If positive class is 10x rarer
pos_weight = torch.tensor([10.0])  # weight positive samples 10x more
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

logits = torch.tensor([0.5, -0.5, 0.1])
targets = torch.tensor([1.0, 0.0, 1.0])
loss = criterion(logits, targets)
```

Mathematically, `pos_weight` $w$ modifies the loss to:

$$\mathcal{L} = -\frac{1}{m}\sum_{i=1}^m \left[w \cdot y^{(i)} \log p^{(i)} + (1 - y^{(i)}) \log(1 - p^{(i)})\right]$$

A good heuristic is $w = n_{\text{negative}} / n_{\text{positive}}$.

### Per-Sample Weights

The `weight` parameter applies per-sample (or per-class) scaling:

```python
# Different importance per sample
sample_weights = torch.tensor([1.0, 2.0, 0.5, 1.5, 1.0])
criterion = nn.BCELoss(weight=sample_weights)
loss = criterion(predicted_probs, true_labels)
```

## Multi-Label Classification

BCE naturally extends to multi-label problems where each sample can belong to multiple classes simultaneously. Each output neuron is treated as an independent binary classifier:

```python
# Multi-label: each image can have multiple tags
# Tags: [cat, outdoor, sunny]
labels = torch.tensor([[1, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1]], dtype=torch.float32)

logits = torch.tensor([[2.0, 1.5, -0.5],
                        [-1.0, -2.0, 3.0],
                        [1.5, -0.3, 2.0]])

criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, labels)
print(f"Multi-label BCE: {loss.item():.4f}")
```

Unlike multi-class classification with `CrossEntropyLoss` (where classes are mutually exclusive), multi-label BCE treats each class independently and does not require probabilities to sum to 1.

## Model Architecture Pattern

```python
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output (logit)
        )

    def forward(self, x):
        return self.layers(x)  # Returns raw logit — no sigmoid!

# Training
model = BinaryClassifier(input_dim=10)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# In the training loop:
logits = model(x_batch)               # shape: (batch, 1)
loss = criterion(logits.squeeze(), y_batch)  # y_batch: (batch,)
loss.backward()

# Inference
with torch.no_grad():
    logits = model(x_test)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).long()
```

## Relationship to Multi-Class Cross-Entropy

Binary cross-entropy is a special case of multi-class cross-entropy with $K = 2$. The two are related by:

| Aspect | `nn.BCEWithLogitsLoss` | `nn.CrossEntropyLoss` |
|--------|------------------------|-----------------------|
| **Classes** | 2 (binary) | $K \geq 2$ |
| **Model output** | 1 logit | $K$ logits |
| **Internal activation** | Sigmoid | Softmax |
| **Label format** | Float (0.0 or 1.0) | Integer index (0 to $K-1$) |
| **Multi-label** | Yes (independent) | No (mutually exclusive) |

For binary problems, both approaches are valid: `BCEWithLogitsLoss` with 1 output neuron, or `CrossEntropyLoss` with 2 output neurons. The single-output approach is slightly more efficient and is the standard convention.

## Key Takeaways

Binary Cross-Entropy is the negative log-likelihood of a Bernoulli model, making it the principled loss for binary classification. The sigmoid activation maps logits to probabilities, and the composed sigmoid–BCE gradient takes the simple form $p - y$. Always use `nn.BCEWithLogitsLoss` over `nn.BCELoss` for numerical stability—it fuses the sigmoid and log operations to avoid overflow and underflow. For class imbalance, use `pos_weight` to upweight the minority class. BCE extends naturally to multi-label classification, where each output neuron is an independent binary classifier.
