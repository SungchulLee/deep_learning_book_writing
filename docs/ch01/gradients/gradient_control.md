# Gradient Tracking Control

## Overview

Controlling when and how PyTorch tracks gradients is essential for memory-efficient inference, proper training loop implementation, and advanced techniques like transfer learning. This section covers the mechanisms for enabling, disabling, and manipulating gradient tracking: `requires_grad`, `torch.no_grad()`, `detach()`, and related methods.

## Learning Objectives

By the end of this section, you will be able to:

1. Control gradient tracking with `requires_grad` and `requires_grad_()`
2. Use `torch.no_grad()` for inference and parameter updates
3. Understand the difference between `detach()` and `detach_()`
4. Freeze and unfreeze model parameters for transfer learning
5. Avoid common pitfalls in gradient control

## The `requires_grad` Attribute

### Default Behavior

By default, tensors do **not** track gradients:

```python
import torch

# Data tensors - no gradient tracking by default
x = torch.randn(3, 4)
print(f"x.requires_grad: {x.requires_grad}")  # False

# Parameters - must explicitly enable
w = torch.randn(4, 2, requires_grad=True)
print(f"w.requires_grad: {w.requires_grad}")  # True
```

### Propagation Rule

If **any** input to an operation has `requires_grad=True`, the output will also have `requires_grad=True`:

```python
import torch

a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=False)
c = a + b

print(f"a.requires_grad: {a.requires_grad}")  # True
print(f"b.requires_grad: {b.requires_grad}")  # False
print(f"c.requires_grad: {c.requires_grad}")  # True (inherits from a)
```

### In-Place Modification with `requires_grad_()`

```python
import torch

# Method 1: Set at creation
w = torch.randn(3, requires_grad=True)

# Method 2: In-place modification
w = torch.randn(3)
w.requires_grad_(True)  # Note: trailing underscore = in-place

print(f"After requires_grad_(True): {w.requires_grad}")  # True

# Disable tracking
w.requires_grad_(False)
print(f"After requires_grad_(False): {w.requires_grad}")  # False
```

**Important:** `requires_grad_()` only modifies metadata, not tensor values. It does **not** need `torch.no_grad()` wrapping.

## The `torch.no_grad()` Context Manager

### Purpose

`torch.no_grad()` temporarily disables gradient tracking for all operations within its scope:

```python
import torch

x = torch.randn(3, requires_grad=True)

# Normal operation - tracking enabled
y = x ** 2
print(f"Outside no_grad: y.requires_grad = {y.requires_grad}")  # True

# Inside no_grad - tracking disabled
with torch.no_grad():
    z = x ** 2
    print(f"Inside no_grad: z.requires_grad = {z.requires_grad}")  # False

# After exiting - tracking restored
w = x ** 2
print(f"After no_grad: w.requires_grad = {w.requires_grad}")  # True
```

### Use Case 1: Inference

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 5)
x = torch.randn(32, 10)

# During inference - no gradients needed
with torch.no_grad():
    predictions = model(x)
    # predictions.requires_grad = False
    # No computation graph stored - saves memory!

print(f"predictions.requires_grad: {predictions.requires_grad}")  # False
```

### Use Case 2: Parameter Updates

```python
import torch

w = torch.randn(3, requires_grad=True)
lr = 0.1

# Compute loss and gradients
loss = (w ** 2).sum()
loss.backward()

# ❌ WRONG: Update without no_grad
# w = w - lr * w.grad  # Creates new tensor with grad tracking
# w -= lr * w.grad     # In-place on leaf - error!

# ✅ CORRECT: Update with no_grad
with torch.no_grad():
    w -= lr * w.grad  # In-place update, not tracked
    
print(f"w.requires_grad after update: {w.requires_grad}")  # Still True!
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

### `detach()` - Returns New Tensor

`detach()` returns a **new tensor** that shares storage but has no gradient tracking:

```python
import torch

a = torch.tensor([1., 2., 3.], requires_grad=True)
b = a ** 2  # Non-leaf, has grad_fn

# Detach creates a new tensor
b_detached = b.detach()

print(f"b.requires_grad: {b.requires_grad}")           # True
print(f"b_detached.requires_grad: {b_detached.requires_grad}")  # False
print(f"b_detached.grad_fn: {b_detached.grad_fn}")     # None
print(f"b_detached is b: {b_detached is b}")           # False (new object)
print(f"Shares storage: {b_detached.data_ptr() == b.data_ptr()}")  # True
```

### `detach_()` - In-Place Modification

`detach_()` modifies the tensor **in-place**:

```python
import torch

a = torch.tensor([1., 2., 3.], requires_grad=True)
b = a ** 2

print(f"Before detach_(): b.grad_fn = {b.grad_fn}")  # PowBackward0

b.detach_()  # In-place

print(f"After detach_(): b.grad_fn = {b.grad_fn}")    # None
print(f"After detach_(): b.requires_grad = {b.requires_grad}")  # False
```

**Warning:** Using `detach_()` on a tensor that's still part of an active computation graph can break gradient flow silently.

### Use Case: Converting to NumPy

```python
import torch
import numpy as np

# Tensors with requires_grad=True cannot directly convert to NumPy
t = torch.randn(4, requires_grad=True)

try:
    arr = t.numpy()  # Error!
except RuntimeError as e:
    print(f"Error: {e}")

# Correct approach
arr = t.detach().cpu().numpy()  # detach → cpu → numpy
print(f"NumPy array: {arr}")
```

### Use Case: Target Networks in RL

```python
import torch
import torch.nn as nn

# Q-network and target network
q_network = nn.Linear(10, 4)
target_network = nn.Linear(10, 4)
target_network.load_state_dict(q_network.state_dict())

state = torch.randn(32, 10)
next_state = torch.randn(32, 10)

# Target values should not have gradients
with torch.no_grad():
    target_q = target_network(next_state).max(dim=1).values

# Or equivalently:
target_q = target_network(next_state).max(dim=1).values.detach()

# Now compute loss - gradients only flow through q_network
q_values = q_network(state)
# loss = ...
```

## Parameter Freezing for Transfer Learning

### Freezing Specific Layers

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),   # Layer 0 - freeze this
    nn.ReLU(),             # Layer 1
    nn.Linear(256, 10)     # Layer 2 - train this
)

# Freeze first linear layer
for name, param in model.named_parameters():
    if name.startswith('0.'):  # Layer 0
        param.requires_grad_(False)
        
# Verify
for name, param in model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")
# Output:
# 0.weight: requires_grad = False
# 0.bias: requires_grad = False
# 2.weight: requires_grad = True
# 2.bias: requires_grad = True
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
        x = x.mean(dim=[2, 3])  # Global average pooling
        return self.classifier(x)

model = Model()

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad_(False)

# Only classifier parameters will be updated
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params} / {total_params}")
```

### Unfreezing for Fine-tuning

```python
def unfreeze_all(model):
    """Make all parameters trainable."""
    for param in model.parameters():
        param.requires_grad_(True)

def freeze_bn(model):
    """Freeze batch normalization layers."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            for param in module.parameters():
                param.requires_grad_(False)
            module.eval()  # Also set to eval mode
```

## Optimizer and Frozen Parameters

When parameters are frozen, they should typically be excluded from the optimizer:

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

# Option 1: Filter when creating optimizer
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# Option 2: Use parameter groups
optimizer = torch.optim.Adam([
    {'params': model[0].parameters(), 'lr': 0.0},  # Frozen (or exclude)
    {'params': model[2].parameters(), 'lr': 0.001}  # Trainable
])
```

## Summary Table

| Method | Effect | Creates New Tensor | Modifies In-Place |
|--------|--------|-------------------|-------------------|
| `x.requires_grad_(True/False)` | Toggle tracking | No | Yes |
| `x.detach()` | Remove from graph | Yes | No |
| `x.detach_()` | Remove from graph | No | Yes |
| `torch.no_grad()` | Disable tracking in scope | N/A | N/A |

## Common Patterns

### Pattern 1: Training Loop

```python
for epoch in range(num_epochs):
    for x, y in dataloader:
        # Forward (tracking enabled)
        output = model(x)
        loss = criterion(output, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Update (tracking disabled)
        # optimizer.step() uses no_grad internally
        optimizer.step()
```

### Pattern 2: Validation Loop

```python
model.eval()
with torch.no_grad():  # Disable tracking
    for x, y in val_loader:
        output = model(x)
        loss = criterion(output, y)
        # Metrics computed without gradients
model.train()
```

### Pattern 3: Mixed Precision Logging

```python
# Log values without affecting gradients
with torch.no_grad():
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    weight_norm = sum(p.norm() for p in model.parameters())
```

## References

- PyTorch Autograd Mechanics: https://pytorch.org/docs/stable/notes/autograd.html
- Transfer Learning Tutorial: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
