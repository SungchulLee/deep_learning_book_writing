# Classification Losses

Classification tasks predict discrete categories rather than continuous values. This fundamental difference requires loss functions that operate on probability distributions over classes. This section covers the primary classification losses and their proper usage in PyTorch.

## From Regression to Classification

While regression predicts a continuous value, classification outputs a probability distribution over $K$ classes:

| Aspect | Regression | Classification |
|--------|------------|----------------|
| Output | Single continuous value | Probability per class |
| Target | Continuous value | Class index or one-hot |
| Activation | None (or bounded) | Sigmoid/Softmax |
| Loss | MSE, MAE, Huber | BCE, Cross-Entropy |

## Binary Classification: Cross-Entropy

Binary classification distinguishes between two classes (e.g., spam/not-spam, positive/negative). The model outputs a single probability $p \in [0, 1]$ representing the likelihood of the positive class.

### Mathematical Foundation

Binary Cross-Entropy (BCE) measures the divergence between predicted probability $p$ and true label $y \in \{0, 1\}$:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

This formula derives from the negative log-likelihood of the Bernoulli distribution. When $y=1$, only the first term contributes ($-\log(p)$), penalizing low predicted probabilities. When $y=0$, only the second term contributes ($-\log(1-p)$), penalizing high predicted probabilities.

### Basic BCE Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Binary classification: Spam detection
# Labels: 0 = Not Spam, 1 = Spam
true_labels = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float32)

# Model predictions (probabilities after sigmoid)
predicted_probs = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.6])

# BCE Loss
bce_criterion = nn.BCELoss()
bce_loss = bce_criterion(predicted_probs, true_labels)
print(f"BCE Loss: {bce_loss.item():.4f}")  # ~0.23

# Per-sample analysis
for i in range(len(true_labels)):
    y = true_labels[i].item()
    p = predicted_probs[i].item()
    if y == 1:
        sample_loss = -torch.log(torch.tensor(p))
    else:
        sample_loss = -torch.log(torch.tensor(1 - p))
    print(f"  Sample {i+1}: y={int(y)}, p={p:.2f} → loss={sample_loss.item():.4f}")
```

### Understanding Logits

Neural networks typically output raw, unbounded values called **logits** before any activation function. The sigmoid function converts logits to probabilities:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

```python
# Raw model outputs (logits)
logits = torch.tensor([-2.0, 3.0, -1.5, 2.5, 0.5])

# Convert to probabilities
probabilities = torch.sigmoid(logits)
print(f"Logits: {logits}")
print(f"Probabilities: {probabilities}")

# Interpretation:
# logit = 0 → probability = 0.5 (maximum uncertainty)
# logit > 0 → probability > 0.5 (predicts positive class)
# logit < 0 → probability < 0.5 (predicts negative class)
```

### BCEWithLogitsLoss (Recommended)

`BCEWithLogitsLoss` combines sigmoid activation and BCE loss in a numerically stable manner:

```python
# Recommended approach: BCEWithLogitsLoss
bce_with_logits = nn.BCEWithLogitsLoss()
loss_from_logits = bce_with_logits(logits, true_labels)

# Equivalent to (but more stable than):
manual_loss = nn.BCELoss()(torch.sigmoid(logits), true_labels)

print(f"BCEWithLogitsLoss: {loss_from_logits.item():.4f}")
print(f"Manual approach: {manual_loss.item():.4f}")
# Both produce the same result, but BCEWithLogitsLoss is more stable
```

!!! warning "Common Mistake"
    Never apply sigmoid before `BCEWithLogitsLoss`—it already includes sigmoid internally. Applying sigmoid twice produces incorrect results.

## Multi-Class Classification: Cross-Entropy

Multi-class classification assigns inputs to one of $K > 2$ mutually exclusive classes. The model outputs a probability distribution over all classes.

### Mathematical Foundation

Cross-Entropy for $K$ classes:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik} \log(p_{ik})$$

where $y_{ik}$ is 1 if sample $i$ belongs to class $k$ and 0 otherwise. Since exactly one class is correct, this simplifies to:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{n}\sum_{i=1}^{n} \log(p_{i,c_i})$$

where $c_i$ is the true class index for sample $i$.

### Softmax Activation

Multi-class models output $K$ logits, converted to probabilities via softmax:

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

This ensures all probabilities sum to 1: $\sum_k p_k = 1$.

```python
# 4 images, 3 classes: Cat (0), Dog (1), Bird (2)
true_classes = torch.tensor([0, 2, 1, 0])  # Class indices

# Model outputs: logits for each class
logits_multi = torch.tensor([
    [3.0, 1.0, 0.5],   # Image 1: High confidence for Cat
    [0.5, 0.8, 2.5],   # Image 2: High confidence for Bird
    [1.0, 2.0, 0.5],   # Image 3: High confidence for Dog
    [2.5, 1.5, 1.0],   # Image 4: High confidence for Cat
])

# Convert to probabilities
probs = F.softmax(logits_multi, dim=1)
print(f"Probabilities:\n{probs}")
print(f"Sum per sample: {probs.sum(dim=1)}")  # Each row sums to 1.0
```

### CrossEntropyLoss (Recommended)

PyTorch's `CrossEntropyLoss` combines log-softmax and negative log-likelihood:

```python
# CrossEntropyLoss expects raw logits, not probabilities!
ce_criterion = nn.CrossEntropyLoss()
ce_loss = ce_criterion(logits_multi, true_classes)
print(f"Cross-Entropy Loss: {ce_loss.item():.4f}")  # ~0.37
```

!!! warning "Critical Points"
    1. `CrossEntropyLoss` expects **raw logits**, not softmax probabilities
    2. Labels should be **class indices** (integers), not one-hot vectors
    3. Never apply softmax before `CrossEntropyLoss`

### Per-Sample Loss Analysis

```python
class_names = ['Cat', 'Dog', 'Bird']

for i in range(len(true_classes)):
    true_class = true_classes[i].item()
    true_prob = probs[i, true_class].item()
    sample_loss = -torch.log(torch.tensor(true_prob))
    
    print(f"Image {i+1}: True={class_names[true_class]}, "
          f"P(true)={true_prob:.4f}, Loss={sample_loss:.4f}")
```

The loss for each sample depends only on the predicted probability of the true class. Lower probability for the true class means higher loss.

## One-Hot Encoding vs Class Indices

PyTorch's `CrossEntropyLoss` uses class indices for efficiency:

```python
# Class indices (what PyTorch expects)
true_classes = torch.tensor([0, 2, 1, 0])

# One-hot encoding (used in some other frameworks)
one_hot = F.one_hot(true_classes, num_classes=3)
print(f"One-hot encoded:\n{one_hot}")

# Convert one-hot back to indices
classes_from_one_hot = torch.argmax(one_hot, dim=1)
assert torch.equal(classes_from_one_hot, true_classes)
```

## Summary: Binary vs Multi-Class

```
╔═══════════════════╦════════════════════════╦═════════════════════════════╗
║                   ║ BINARY                 ║ MULTI-CLASS                 ║
╠═══════════════════╬════════════════════════╬═════════════════════════════╣
║ Number of Classes ║ 2                      ║ K ≥ 2                       ║
║ Model Output      ║ 1 logit                ║ K logits                    ║
║ Activation        ║ Sigmoid                ║ Softmax                     ║
║ Output Range      ║ [0, 1]                 ║ [0, 1] per class, sum to 1  ║
║ Loss Function     ║ BCEWithLogitsLoss      ║ CrossEntropyLoss            ║
║ Label Format      ║ 0 or 1                 ║ Class index (0 to K-1)      ║
╚═══════════════════╩════════════════════════╩═════════════════════════════╝
```

## Common Model Architectures

```python
# Binary Classification Model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output
        )
    
    def forward(self, x):
        return self.layers(x)  # Returns logit

# Usage
model = BinaryClassifier(10)
criterion = nn.BCEWithLogitsLoss()
```

```python
# Multi-Class Classification Model
class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # One output per class
        )
    
    def forward(self, x):
        return self.layers(x)  # Returns K logits

# Usage
model = MultiClassClassifier(10, num_classes=5)
criterion = nn.CrossEntropyLoss()
```

## Making Predictions (Inference)

During inference, convert model outputs to predictions:

```python
# Binary classification
with torch.no_grad():
    logits = binary_model(inputs)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).long()

# Multi-class classification  
with torch.no_grad():
    logits = multiclass_model(inputs)
    probabilities = F.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
```

## Best Practices

**DO:**

- Use `BCEWithLogitsLoss` for binary classification
- Use `CrossEntropyLoss` for multi-class classification
- Let loss functions handle activations internally
- Use class indices for labels with `CrossEntropyLoss`
- Monitor both loss and accuracy during training

**DON'T:**

- Apply sigmoid before `BCEWithLogitsLoss`
- Apply softmax before `CrossEntropyLoss`
- Use `BCELoss` with raw logits
- One-hot encode labels for `CrossEntropyLoss`

## Key Takeaways

Classification losses operate on probability distributions, with the loss function encoding the divergence between predicted and true distributions. Binary classification uses `BCEWithLogitsLoss` with a single output neuron and sigmoid activation. Multi-class classification uses `CrossEntropyLoss` with $K$ output neurons and softmax activation. Both PyTorch loss functions handle the activation internally for numerical stability—never apply sigmoid or softmax before these losses. During inference, apply the appropriate activation and use thresholding (binary) or argmax (multi-class) to obtain discrete predictions.
