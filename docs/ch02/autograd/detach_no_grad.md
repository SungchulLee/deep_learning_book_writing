# Detach and No Grad

## Overview

Controlling when and how PyTorch tracks gradients is essential for memory-efficient inference, correct training loop implementation, and advanced techniques like transfer learning and reinforcement learning. This section covers the mechanisms for enabling, disabling, and selectively controlling gradient tracking: `torch.no_grad()`, `detach()`, `requires_grad`, and parameter freezing.

## Learning Objectives

By the end of this section, you will be able to:

1. Use `torch.no_grad()` for inference and parameter updates
2. Understand the difference between `detach()` and `detach_()`
3. Control gradient tracking with `requires_grad` and `requires_grad_()`
4. Freeze and unfreeze model parameters for transfer learning
5. Configure optimizers to exclude frozen parameters

## `torch.no_grad()` Context Manager

### Purpose

`torch.no_grad()` temporarily disables gradient tracking for **all** operations within its scope. No computational graph is built, saving both memory and compute.

```python
import torch

x = torch.randn(3, requires_grad=True)

# Outside: tracking enabled
y = x ** 2
print(f"Outside no_grad: y.requires_grad = {y.requires_grad}")  # True

# Inside: tracking disabled
with torch.no_grad():
    z = x ** 2
    print(f"Inside no_grad: z.requires_grad = {z.requires_grad}")   # False

# After: tracking restored
w = x ** 2
print(f"After no_grad: w.requires_grad = {w.requires_grad}")   # True
```

### Use Case 1: Inference

During inference, no gradients are needed. Wrapping the forward pass in `torch.no_grad()` avoids building the computational graph entirely:

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)
x = torch.randn(32, 10)

with torch.no_grad():
    predictions = model(x)
    # No computation graph stored — significant memory savings

print(f"predictions.requires_grad: {predictions.requires_grad}")  # False
```

### Use Case 2: Manual Parameter Updates

When updating parameters outside of an optimizer, `torch.no_grad()` prevents the update itself from being tracked:

```python
import torch

w = torch.randn(3, requires_grad=True)
lr = 0.1

loss = (w ** 2).sum()
loss.backward()

# ❌ WRONG: w = w - lr * w.grad creates a new tensor with grad tracking
# ❌ WRONG: w -= lr * w.grad raises error (in-place on leaf)

# ✅ CORRECT: in-place update without tracking
with torch.no_grad():
    w -= lr * w.grad

print(f"w.requires_grad after update: {w.requires_grad}")  # Still True
```

### Use Case 3: Metric Computation

```python
import torch

def compute_accuracy(model, x, y_true):
    """Compute accuracy without building computation graph."""
    with torch.no_grad():
        logits = model(x)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y_true).float().mean()
    return accuracy.item()
```

## Detaching Tensors

### `detach()` — Returns a New View

`detach()` returns a **new tensor** that shares the same underlying storage but is detached from the computational graph:

```python
import torch

a = torch.tensor([1., 2., 3.], requires_grad=True)
b = a ** 2   # Non-leaf, has grad_fn

b_detached = b.detach()

print(f"b.requires_grad:          {b.requires_grad}")            # True
print(f"b_detached.requires_grad: {b_detached.requires_grad}")   # False
print(f"b_detached.grad_fn:       {b_detached.grad_fn}")         # None
print(f"Same object:    {b_detached is b}")                      # False
print(f"Shares storage: {b_detached.data_ptr() == b.data_ptr()}")  # True
```

**Key properties:**

- Returns a new Python object, but shares the same memory
- The detached tensor has `requires_grad=False` and `grad_fn=None`
- Modifications to the detached tensor affect the original (shared storage)

### `detach_()` — In-Place Detach

`detach_()` modifies the tensor **in-place**, removing it from the graph:

```python
import torch

a = torch.tensor([1., 2., 3.], requires_grad=True)
b = a ** 2

print(f"Before: b.grad_fn = {b.grad_fn}")        # PowBackward0

b.detach_()

print(f"After: b.grad_fn = {b.grad_fn}")          # None
print(f"After: b.requires_grad = {b.requires_grad}")  # False
```

**Warning:** Using `detach_()` on a tensor still referenced by an active computation graph silently breaks gradient flow. Prefer `detach()` in most cases.

### Common Pattern: Converting to NumPy

Tensors with `requires_grad=True` cannot be converted directly to NumPy arrays:

```python
import torch

t = torch.randn(4, requires_grad=True)

try:
    arr = t.numpy()
except RuntimeError as e:
    print(f"Error: {e}")

# Correct: detach → cpu → numpy
arr = t.detach().cpu().numpy()
print(f"NumPy array: {arr}")
```

### Common Pattern: Target Networks in RL

In reinforcement learning, target network outputs should not receive gradients:

```python
import torch
import torch.nn as nn

q_network = nn.Linear(10, 4)
target_network = nn.Linear(10, 4)
target_network.load_state_dict(q_network.state_dict())

state = torch.randn(32, 10)
next_state = torch.randn(32, 10)

# Option A: no_grad context
with torch.no_grad():
    target_q = target_network(next_state).max(dim=1).values

# Option B: detach after forward
target_q = target_network(next_state).max(dim=1).values.detach()

# Gradients flow only through q_network
q_values = q_network(state)
```

### `detach()` vs `torch.no_grad()` — When to Use Which

| Scenario | Preferred Method |
|----------|-----------------|
| Inference / validation | `torch.no_grad()` — block-level, clear intent |
| Stopping gradient at a specific tensor | `detach()` — surgical, per-tensor |
| Parameter updates (manual) | `torch.no_grad()` |
| Creating targets for loss computation | `detach()` or `torch.no_grad()` |
| Converting to NumPy | `detach()` |

## The `requires_grad` Attribute

### Controlling Tracking Per-Tensor

```python
import torch

# Set at creation time
w = torch.randn(3, requires_grad=True)

# Or toggle in-place
w = torch.randn(3)
w.requires_grad_(True)
print(f"After requires_grad_(True): {w.requires_grad}")  # True

w.requires_grad_(False)
print(f"After requires_grad_(False): {w.requires_grad}")  # False
```

`requires_grad_()` modifies only metadata — it does not change tensor values and does not require a `torch.no_grad()` wrapper.

### Propagation Rule

If **any** input to an operation has `requires_grad=True`, the output inherits it:

```python
import torch

a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=False)
c = a + b

print(f"c.requires_grad: {c.requires_grad}")  # True (inherited from a)
```

## Parameter Freezing for Transfer Learning

### Freezing Specific Layers

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),   # Layer 0 — freeze
    nn.ReLU(),
    nn.Linear(256, 10)     # Layer 2 — train
)

# Freeze first linear layer
for name, param in model.named_parameters():
    if name.startswith('0.'):
        param.requires_grad_(False)

# Verify
for name, param in model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")
# 0.weight: requires_grad = False
# 0.bias:   requires_grad = False
# 2.weight: requires_grad = True
# 2.bias:   requires_grad = True
```

### Freezing by Module

```python
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = FeatureExtractor()
        self.classifier = Classifier()
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=[2, 3])   # Global average pooling
        return self.classifier(x)

model = Model()

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad_(False)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable} / {total}")
```

### Unfreezing for Fine-Tuning

```python
def unfreeze_all(model):
    """Make all parameters trainable."""
    for param in model.parameters():
        param.requires_grad_(True)

def freeze_bn(model):
    """Freeze batch normalization layers (weights + running stats)."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            for param in module.parameters():
                param.requires_grad_(False)
            module.eval()   # Use running stats, not batch stats
```

### Optimizer Configuration with Frozen Parameters

When parameters are frozen, exclude them from the optimizer to avoid unnecessary state:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# Freeze first layer
for param in model[0].parameters():
    param.requires_grad_(False)

# Option 1: Filter trainable parameters
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# Option 2: Parameter groups with different learning rates
optimizer = torch.optim.Adam([
    {'params': model[2].parameters(), 'lr': 1e-3},   # Trainable head
    # Frozen layers excluded entirely
])
```

## Comparison Table

| Method | Effect | Creates New Tensor | Modifies In-Place |
|--------|--------|-------------------|-------------------|
| `x.requires_grad_(True/False)` | Toggle tracking | No | Yes (metadata) |
| `x.detach()` | Remove from graph | Yes (shared storage) | No |
| `x.detach_()` | Remove from graph | No | Yes |
| `torch.no_grad()` | Disable tracking in scope | N/A | N/A |

## Common Patterns

### Pattern 1: Validation Loop

```python
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        output = model(x)
        loss = criterion(output, y)
        # Metrics computed without gradient overhead
model.train()
```

### Pattern 2: Gradient-Free Logging

```python
with torch.no_grad():
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    weight_norm = sum(p.norm() for p in model.parameters())
```

### Pattern 3: Stopping Gradient Flow at a Boundary

```python
# Encoder-decoder where encoder is pretrained
encoded = encoder(x).detach()   # Gradient stops here
decoded = decoder(encoded)       # Only decoder gets gradients
loss = criterion(decoded, target)
loss.backward()
```

## Summary

| Concept | When to Use |
|---------|-------------|
| `torch.no_grad()` | Inference, validation, manual parameter updates |
| `detach()` | Stopping gradient at a specific tensor boundary |
| `requires_grad_(False)` | Freezing parameters for transfer learning |
| `detach_()` | Rare — in-place detach when you're sure it's safe |

## Common Pitfalls

1. **Using `model.eval()` alone for inference** — this only affects dropout / batch norm, not gradient tracking. Always pair with `torch.no_grad()`.
2. **Detaching when you mean `no_grad`** — `detach()` is per-tensor; for a block of code, `torch.no_grad()` is cleaner.
3. **Forgetting to re-enable training mode** — call `model.train()` after validation.
4. **Modifying detached tensors** that share storage with graph tensors — can corrupt the computational graph.

## References

- PyTorch Autograd Mechanics: [https://pytorch.org/docs/stable/notes/autograd.html](https://pytorch.org/docs/stable/notes/autograd.html)
- Transfer Learning Tutorial: [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
