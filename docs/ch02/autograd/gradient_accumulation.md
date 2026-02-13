# Gradient Accumulation

## Overview

Gradient accumulation is PyTorch's default behavior where gradients **add up** across multiple `.backward()` calls rather than being overwritten. While sometimes surprising to beginners, this design is a powerful feature that enables training with effectively larger batch sizes, combining multiple loss functions, and distributed training strategies. Understanding gradient accumulation is essential for writing correct training loops and for memory-efficient training.

## Learning Objectives

By the end of this section, you will be able to:

1. Understand why gradients accumulate by default in PyTorch
2. Properly zero gradients between training steps
3. Leverage intentional gradient accumulation for large-batch training
4. Implement gradient accumulation with correct loss scaling
5. Handle multiple loss functions using accumulation

## Default Accumulation Behavior

### Why Gradients Accumulate

When `loss.backward()` is called, PyTorch **adds** the newly computed gradients to the existing `.grad` attribute:

$$\texttt{param.grad}_{\text{new}} = \texttt{param.grad}_{\text{old}} + \nabla_{\text{param}} L$$

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

# First backward: loss₁ = x², d(loss₁)/dx = 2x = 4
loss1 = x ** 2
loss1.backward()
print(f"After 1st backward: x.grad = {x.grad}")  # tensor([4.])

# Second backward WITHOUT zeroing: loss₂ = 3x, d(loss₂)/dx = 3
loss2 = 3 * x
loss2.backward()
print(f"After 2nd backward: x.grad = {x.grad}")  # tensor([7.]) ← accumulated!
```

**Design rationale:**

1. **Flexibility** — allows combining gradients from multiple losses naturally
2. **Efficiency** — enables processing data in memory-limited chunks
3. **Simplicity** — a single additive pattern works for all use cases

## Zeroing Gradients

### The Importance of Zeroing

Without zeroing, gradients from previous iterations contaminate the current update:

```python
# ❌ INCORRECT: gradients accumulate across iterations
for epoch in range(num_epochs):
    loss = compute_loss()
    loss.backward()         # Gradients keep growing!
    optimizer.step()        # Updates get progressively larger

# ✅ CORRECT: zero gradients before each backward pass
for epoch in range(num_epochs):
    optimizer.zero_grad()   # Clear previous gradients
    loss = compute_loss()
    loss.backward()
    optimizer.step()
```

### Three Ways to Zero Gradients

```python
import torch

w = torch.randn(3, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

# Populate gradients
loss = (w ** 2).sum()
loss.backward()

# Method 1: optimizer.zero_grad() — default sets grad to None
optimizer.zero_grad()                          # set_to_none=True by default
print(f"After zero_grad(): w.grad = {w.grad}")  # None

# Restore gradients
loss = (w ** 2).sum()
loss.backward()

# Method 2: optimizer.zero_grad(set_to_none=False) — sets to zero tensor
optimizer.zero_grad(set_to_none=False)
print(f"After zero_grad(False): w.grad = {w.grad}")  # tensor([0., 0., 0.])

# Restore gradients
loss = (w ** 2).sum()
loss.backward()

# Method 3: Direct assignment
w.grad = None
print(f"After w.grad = None: w.grad = {w.grad}")  # None
```

### `set_to_none=True` vs `set_to_none=False`

| Setting | Behavior | Memory | Speed |
|---------|----------|--------|-------|
| `set_to_none=True` (default) | Sets `.grad` to `None` | Frees memory | Slightly faster |
| `set_to_none=False` | Fills `.grad` with zeros | Keeps allocation | Slightly slower |

**Recommendation:** Use the default (`set_to_none=True`) unless you need the zero tensor to remain allocated (e.g., for in-place operations on `.grad`).

## Intentional Gradient Accumulation

### Use Case: Simulating Large Batches

When GPU memory is insufficient for a desired batch size $B$, you can split it into $K$ micro-batches of size $b = B / K$ and accumulate gradients:

**Mathematical equivalence:**

$$\nabla_\theta L_{\text{batch}} = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell_i = \frac{1}{K} \sum_{k=1}^{K} \underbrace{\left( \frac{1}{b} \sum_{j \in \text{micro-batch}_k} \nabla_\theta \ell_j \right)}_{\text{micro-batch mean gradient}}$$

Each micro-batch's mean gradient must be **scaled by $1/K$** before accumulation.

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

model = nn.Linear(5, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Simulate batch size 32 using 4 micro-batches of size 8
micro_batch_size = 8
accumulation_steps = 4   # 8 × 4 = 32 effective batch size

X = torch.randn(32, 5)
y = torch.randn(32, 1)

optimizer.zero_grad()

for step in range(accumulation_steps):
    start = step * micro_batch_size
    end = start + micro_batch_size
    x_batch = X[start:end]
    y_batch = y[start:end]
    
    pred = model(x_batch)
    loss = nn.functional.mse_loss(pred, y_batch, reduction='mean')
    
    # Scale loss before backward
    scaled_loss = loss / accumulation_steps
    scaled_loss.backward()   # Gradients accumulate
    
    print(f"Step {step+1}: loss = {loss.item():.4f}")

# Single optimizer step with accumulated gradients
optimizer.step()
optimizer.zero_grad()

print("Completed one optimization step with effective batch size 32")
```

### Why Scaling Matters

Without scaling by $1/K$, accumulated gradients are $K$ times too large:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# Two identical models for comparison
modelA = nn.Linear(3, 1, bias=False)
modelB = nn.Linear(3, 1, bias=False)
with torch.no_grad():
    modelB.weight.copy_(modelA.weight)

X = torch.randn(4, 3)
y = torch.randn(4, 1)

# Reference: full-batch gradient
modelA.zero_grad()
loss_full = nn.functional.mse_loss(modelA(X), y, reduction='mean')
loss_full.backward()
grad_full = modelA.weight.grad.clone()

# Split into 2 micro-batches
X1, X2 = X[:2], X[2:]
y1, y2 = y[:2], y[2:]

# ❌ WRONG: unscaled accumulation
modelB.zero_grad()
nn.functional.mse_loss(modelB(X1), y1, reduction='mean').backward()
nn.functional.mse_loss(modelB(X2), y2, reduction='mean').backward()
grad_wrong = modelB.weight.grad.clone()

# ✅ CORRECT: scaled accumulation
modelB.zero_grad()
(nn.functional.mse_loss(modelB(X1), y1, reduction='mean') / 2).backward()
(nn.functional.mse_loss(modelB(X2), y2, reduction='mean') / 2).backward()
grad_correct = modelB.weight.grad.clone()

print(f"Full batch gradient:\n{grad_full}")
print(f"\nUnscaled (WRONG):\n{grad_wrong}")
print(f"\nScaled (CORRECT):\n{grad_correct}")
print(f"\nMax error (scaled): {(grad_full - grad_correct).abs().max().item():.2e}")
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
        accumulation_steps: Number of micro-batches per optimizer step
        num_epochs: Number of training epochs
    """
    model.train()
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Forward pass
            output = model(x)
            loss = criterion(output, y)
            
            # Scale and backward
            (loss / accumulation_steps).backward()
            accumulated_loss += loss.item()
            
            # Update every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                avg_loss = accumulated_loss / accumulation_steps
                print(f"Epoch {epoch+1}, Step {batch_idx+1}: Loss = {avg_loss:.4f}")
                accumulated_loss = 0.0
        
        # Handle remaining batches
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

Gradient accumulation naturally handles multiple loss terms. Two equivalent approaches:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

loss_mse = ((x - 2) ** 2).mean()     # Main loss
loss_reg = 0.1 * x.abs().mean()      # Regularization

# Method 1: Combine losses first (PREFERRED — cleaner and avoids retain_graph)
total_loss = loss_mse + loss_reg
total_loss.backward()
grad_method1 = x.grad.clone()

x.grad.zero_()

# Method 2: Accumulate gradients from separate backward calls
loss_mse.backward(retain_graph=True)   # Need retain_graph for shared subgraph
loss_reg.backward()
grad_method2 = x.grad.clone()

print(f"Method 1 (combined):     {grad_method1}")
print(f"Method 2 (accumulated):  {grad_method2}")
print(f"Equal: {torch.allclose(grad_method1, grad_method2)}")
```

Method 1 is preferred because it avoids the overhead of `retain_graph`.

## Summary

| Aspect | Key Point |
|--------|-----------|
| **Default behavior** | Gradients accumulate (add up) across `.backward()` calls |
| **Zeroing** | Always `optimizer.zero_grad()` before computing training gradients |
| **Zeroing methods** | `optimizer.zero_grad()`, `param.grad = None`, `param.grad.zero_()` |
| **Large-batch simulation** | Accumulate scaled gradients: `(loss / K).backward()` |
| **Multiple losses** | Combine before backward, or accumulate with `retain_graph` |
| **Memory benefit** | Process large effective batches with limited GPU memory |

## Common Pitfalls

1. **Forgetting to zero gradients** → stale gradients cause incorrect, growing updates
2. **Forgetting to scale by $1/K$** → accumulated gradients are $K$ times too large
3. **Calling `optimizer.step()` on every micro-batch** → updates before accumulation completes
4. **Using `reduction='sum'`** without adjusting for batch size → gradient magnitude depends on batch size

## References

- PyTorch Gradient Accumulation: [https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)
- Hoffer, E., et al. (2017). Train Longer, Generalize Better: Closing the Generalization Gap in Large Batch Training of Neural Networks.
