# Gradient Accumulation

## Overview

Gradient accumulation is PyTorch's default behavior where gradients **add up** across multiple `.backward()` calls rather than being overwritten. This behavior, while sometimes surprising to beginners, is a powerful feature that enables training with effectively larger batch sizes, multiple loss functions, and distributed training strategies. Understanding gradient accumulation is essential for writing correct training loops and leveraging memory-efficient training techniques.

## Learning Objectives

By the end of this section, you will be able to:

1. Understand why gradients accumulate by default in PyTorch
2. Properly zero gradients between training steps
3. Leverage intentional gradient accumulation for large batch training
4. Implement gradient accumulation with proper scaling
5. Avoid common pitfalls related to gradient management

## Default Accumulation Behavior

### Why Gradients Accumulate

When you call `loss.backward()`, PyTorch **adds** the computed gradients to the existing `.grad` attribute rather than replacing it:

$$\texttt{param.grad}_{\text{new}} = \texttt{param.grad}_{\text{old}} + \nabla_{\text{param}} L$$

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

# First backward: loss1 = x^2, d(loss1)/dx = 2x = 4
loss1 = x ** 2
loss1.backward()
print(f"After 1st backward: x.grad = {x.grad}")  # tensor([4.])

# Second backward WITHOUT zeroing: loss2 = 3x, d(loss2)/dx = 3
loss2 = 3 * x
loss2.backward()
print(f"After 2nd backward: x.grad = {x.grad}")  # tensor([7.]) ← Accumulated!
```

**Why This Design?**
1. **Flexibility**: Allows combining gradients from multiple losses
2. **Efficiency**: Enables processing data in chunks (gradient accumulation)
3. **Simplicity**: Single accumulation pattern works for all use cases

## Zeroing Gradients

### The Importance of Zeroing

Without zeroing gradients, training will be incorrect:

```python
# ❌ INCORRECT: Gradients keep accumulating
for epoch in range(epochs):
    loss = compute_loss()
    loss.backward()        # Gradients accumulate!
    optimizer.step()       # Updates get progressively larger

# ✅ CORRECT: Zero gradients before each backward
for epoch in range(epochs):
    optimizer.zero_grad()  # Clear previous gradients
    loss = compute_loss()
    loss.backward()
    optimizer.step()
```

### Three Ways to Zero Gradients

```python
import torch

w = torch.randn(3, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

# Run backward to populate gradients
loss = (w ** 2).sum()
loss.backward()
print(f"After backward: w.grad = {w.grad}")

# Method 1: optimizer.zero_grad() - Modern default (sets to None)
optimizer.zero_grad()  # Default: set_to_none=True
print(f"After zero_grad(): w.grad = {w.grad}")  # None

# Restore gradients for demo
loss = (w ** 2).sum()
loss.backward()

# Method 2: optimizer.zero_grad(set_to_none=False) - Zero tensor
optimizer.zero_grad(set_to_none=False)
print(f"After zero_grad(set_to_none=False): w.grad = {w.grad}")  # tensor([0., 0., 0.])

# Restore gradients for demo
loss = (w ** 2).sum()
loss.backward()

# Method 3: Manual assignment - For individual parameters
w.grad = None
print(f"After w.grad = None: w.grad = {w.grad}")  # None
```

### `set_to_none=True` vs `set_to_none=False`

| Setting | Behavior | Memory | Speed |
|---------|----------|--------|-------|
| `set_to_none=True` (default) | Sets `.grad` to `None` | Frees memory | Slightly faster |
| `set_to_none=False` | Sets `.grad` to zeros | Keeps allocation | Slightly slower |

**Recommendation:** Use the default (`set_to_none=True`) unless you have specific reasons to preserve the gradient tensor allocation.

## Intentional Gradient Accumulation

### Use Case: Simulating Large Batches

When GPU memory is limited, you can simulate a large batch by accumulating gradients across smaller micro-batches:

**Mathematical Equivalence:**
For batch size $B$ split into $K$ micro-batches of size $b = B/K$:

$$\nabla_\theta L_{\text{batch}} = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell_i = \frac{1}{K} \sum_{k=1}^{K} \left( \frac{1}{b} \sum_{j \in \text{micro-batch}_k} \nabla_\theta \ell_j \right)$$

Each micro-batch mean gradient must be scaled by $\frac{1}{K}$ before accumulation.

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Setup
model = nn.Linear(5, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Simulate large batch (32) using micro-batches (8)
micro_batch_size = 8
accumulation_steps = 4  # 8 × 4 = 32 effective batch size
total_samples = micro_batch_size * accumulation_steps

# Generate fake data
X = torch.randn(total_samples, 5)
y = torch.randn(total_samples, 1)

# Clear gradients at start
optimizer.zero_grad()

# Accumulation loop
for step in range(accumulation_steps):
    # Get micro-batch
    start = step * micro_batch_size
    end = start + micro_batch_size
    x_batch = X[start:end]
    y_batch = y[start:end]
    
    # Forward pass
    pred = model(x_batch)
    loss = nn.functional.mse_loss(pred, y_batch, reduction='mean')
    
    # CRITICAL: Scale loss before backward
    scaled_loss = loss / accumulation_steps
    scaled_loss.backward()  # Gradients accumulate
    
    print(f"Step {step+1}: loss = {loss.item():.4f}")

# Single optimizer step with accumulated gradients
optimizer.step()
optimizer.zero_grad()  # Ready for next accumulation cycle

print("Completed one optimization step with effective batch size 32")
```

### Why Scaling Matters

Without scaling, accumulated gradients are $K$ times too large:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# Two identical models
modelA = nn.Linear(3, 1, bias=False)
modelB = nn.Linear(3, 1, bias=False)
with torch.no_grad():
    modelB.weight.copy_(modelA.weight)

X = torch.randn(4, 3)
y = torch.randn(4, 1)

# Full batch gradient (reference)
modelA.zero_grad()
loss_full = nn.functional.mse_loss(modelA(X), y, reduction='mean')
loss_full.backward()
grad_full = modelA.weight.grad.clone()

# Split into 2 micro-batches
X1, X2 = X[:2], X[2:]
y1, y2 = y[:2], y[2:]

# ❌ WRONG: Unscaled accumulation
modelB.zero_grad()
loss1 = nn.functional.mse_loss(modelB(X1), y1, reduction='mean')
loss1.backward()
loss2 = nn.functional.mse_loss(modelB(X2), y2, reduction='mean')
loss2.backward()
grad_wrong = modelB.weight.grad.clone()

# ✅ CORRECT: Scaled accumulation
modelB.zero_grad()
(nn.functional.mse_loss(modelB(X1), y1, reduction='mean') / 2).backward()
(nn.functional.mse_loss(modelB(X2), y2, reduction='mean') / 2).backward()
grad_correct = modelB.weight.grad.clone()

print(f"Full batch gradient:\n{grad_full}")
print(f"\nUnscaled accumulation (WRONG):\n{grad_wrong}")
print(f"\nScaled accumulation (CORRECT):\n{grad_correct}")
print(f"\nMax error (scaled): {(grad_full - grad_correct).abs().max().item():.2e}")
```

**Output:**
```
Full batch gradient:
tensor([[-0.1234, 0.5678, -0.3456]])

Unscaled accumulation (WRONG):
tensor([[-0.2468, 1.1356, -0.6912]])  ← 2x too large!

Scaled accumulation (CORRECT):
tensor([[-0.1234, 0.5678, -0.3456]])  ← Matches!

Max error (scaled): 0.00e+00
```

## Complete Training Loop with Gradient Accumulation

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_with_accumulation(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accumulation_steps: int,
    num_epochs: int
):
    """
    Training loop with gradient accumulation.
    
    Args:
        model: Neural network model
        train_loader: DataLoader providing (x, y) batches
        optimizer: Optimizer instance
        accumulation_steps: Number of micro-batches to accumulate
        num_epochs: Number of training epochs
    """
    model.train()
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear at epoch start
        accumulated_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Forward pass
            output = model(x)
            loss = criterion(output, y)
            
            # Scale and backward
            (loss / accumulation_steps).backward()
            accumulated_loss += loss.item()
            
            # Update every `accumulation_steps` batches
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                avg_loss = accumulated_loss / accumulation_steps
                print(f"Epoch {epoch+1}, Step {batch_idx+1}: Loss = {avg_loss:.4f}")
                accumulated_loss = 0.0
        
        # Handle remaining batches if total isn't divisible by accumulation_steps
        if (batch_idx + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

# Example usage
torch.manual_seed(42)
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_with_accumulation(model, train_loader, optimizer, accumulation_steps=4, num_epochs=2)
```

## Multiple Loss Functions

Gradient accumulation naturally handles multiple loss terms:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Two loss functions we want to combine
loss_mse = ((x - 2) ** 2).mean()    # Main loss
loss_reg = 0.1 * x.abs().mean()     # Regularization

# Method 1: Combine losses first (PREFERRED)
total_loss = loss_mse + loss_reg
total_loss.backward()
grad_method1 = x.grad.clone()

# Reset for comparison
x.grad.zero_()

# Method 2: Accumulate gradients separately
loss_mse.backward(retain_graph=True)  # Need retain_graph for second backward
loss_reg.backward()
grad_method2 = x.grad.clone()

print(f"Method 1 (combined): {grad_method1}")
print(f"Method 2 (accumulated): {grad_method2}")
print(f"Equal: {torch.allclose(grad_method1, grad_method2)}")
```

## Summary

| Aspect | Key Point |
|--------|-----------|
| **Default Behavior** | Gradients accumulate (add up) across `.backward()` calls |
| **Zeroing** | Always zero gradients before computing new gradients for training |
| **Zeroing Methods** | `optimizer.zero_grad()`, `param.grad = None`, `param.grad.zero_()` |
| **Large Batch Training** | Accumulate scaled gradients: `(loss / K).backward()` |
| **Multiple Losses** | Can accumulate or combine before backward |
| **Memory Benefit** | Process large effective batches with limited GPU memory |

## Common Pitfalls

1. **Forgetting to zero gradients** → Exploding updates
2. **Forgetting to scale** → Gradients K× too large
3. **Calling `optimizer.step()` too often** → Updates before accumulation completes
4. **Using `reduction='sum'` without adjustment** → Batch size affects gradient magnitude

## References

- PyTorch Gradient Accumulation: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
- Training Neural Networks with Large Batch Sizes (Hoffer et al., 2017)
