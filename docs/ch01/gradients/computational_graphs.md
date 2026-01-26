# Computational Graphs

## Overview

A computational graph is a directed acyclic graph (DAG) that represents the sequence of operations performed on tensors during the forward pass. PyTorch builds this graph dynamically as operations are executed, then traverses it in reverse during backpropagation to compute gradients. Understanding computational graphs is essential for debugging training issues and optimizing memory usage.

## Learning Objectives

By the end of this section, you will be able to:

1. Understand how PyTorch constructs computational graphs dynamically
2. Trace the relationship between tensor operations and `grad_fn` attributes
3. Use `retain_graph` for multiple backward passes
4. Understand the lifecycle of computational graphs
5. Visualize and debug gradient flow through networks

## Computational Graph Structure

### Nodes and Edges

A computational graph consists of:

**Nodes:**
- **Leaf nodes**: Input tensors with `requires_grad=True`
- **Operation nodes**: Functions that transform tensors (stored as `grad_fn`)
- **Output nodes**: Results of operations

**Edges:**
- Directed edges from inputs to outputs through operations
- Encode dependencies for gradient computation

```
x ─────────→ [**2] ────→ y ────→ [sum] ────→ loss
(leaf)      (PowBackward)      (SumBackward)   (scalar)
```

### Dynamic Graph Construction

PyTorch uses **define-by-run** semantics: the graph is built as operations execute.

```python
import torch

# Step 1: Create leaf tensor
x = torch.tensor([1., 2., 3.], requires_grad=True)
print(f"After creation: x.grad_fn = {x.grad_fn}")  # None (leaf)

# Step 2: First operation - graph expands
y = x ** 2
print(f"After y = x**2: y.grad_fn = {y.grad_fn}")  # PowBackward0

# Step 3: Second operation - graph expands further
z = y.sum()
print(f"After z = y.sum(): z.grad_fn = {z.grad_fn}")  # SumBackward0
```

### Tracing the Graph

Each non-leaf tensor's `grad_fn` stores references to its inputs:

```python
import torch

x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.sum()

# Trace backward through the graph
print(f"z.grad_fn: {z.grad_fn}")
print(f"z.grad_fn.next_functions: {z.grad_fn.next_functions}")
# Shows: ((MulBackward0, 0),)

print(f"y.grad_fn: {y.grad_fn}")
print(f"y.grad_fn.next_functions: {y.grad_fn.next_functions}")
# Shows: ((AccumulateGrad, 0),) - AccumulateGrad is for leaf tensors
```

## Graph Lifecycle and Memory

### Graph Consumption During Backward

By default, `backward()` **frees** the computational graph after execution to save memory:

```python
x = torch.randn(3, requires_grad=True)
loss = (x ** 2).sum()

# First backward - graph is consumed
loss.backward()
print(f"x.grad after 1st backward: {x.grad}")

# Second backward fails - graph is gone!
try:
    loss.backward()
except RuntimeError as e:
    print(f"Error: {e}")
```

**Output:**
```
x.grad after 1st backward: tensor([...])
Error: Trying to backward through the graph a second time...
```

### Using `retain_graph=True`

To perform multiple backward passes on the same graph:

```python
x = torch.randn(3, requires_grad=True)
loss = (x ** 2).sum()

# First backward - retain the graph
loss.backward(retain_graph=True)
grad_after_first = x.grad.clone()

# Second backward - graph still exists
# Gradients ACCUMULATE
loss.backward()
grad_after_second = x.grad.clone()

print(f"After 1st backward: {grad_after_first}")
print(f"After 2nd backward (accumulated): {grad_after_second}")
```

**Common Use Cases for `retain_graph`:**
1. Computing gradients with respect to different upstream gradients
2. Higher-order derivatives (gradients of gradients)
3. Multiple loss functions sharing intermediate computations

### Memory Management Best Practices

```python
# Zero gradients before new computation
x.grad.zero_()

# Build fresh graph
loss = (x ** 2).sum()
loss.backward()

# Gradients now reflect only this forward pass
print(f"Fresh gradients: {x.grad}")
```

## The Backward Function Chain

### Understanding `grad_fn`

Each `grad_fn` is a Python object representing a backward function:

```python
import torch

x = torch.randn(2, 3, requires_grad=True)
y = x @ x.T  # Matrix multiplication
z = y.trace()  # Trace (sum of diagonal)

print(f"Operation: z = trace(x @ x.T)")
print(f"z.grad_fn: {z.grad_fn}")
print(f"Type: {type(z.grad_fn)}")
```

### Chain Rule in Action

The backward pass applies the chain rule through the graph:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}$$

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

# Forward: y = sin(x), z = y^2
y = torch.sin(x)
z = y ** 2

# Manual chain rule:
# dz/dx = dz/dy * dy/dx = 2y * cos(x) = 2*sin(x)*cos(x) = sin(2x)

z.backward()
print(f"x.grad (autograd): {x.grad.item():.6f}")
print(f"sin(2x) (manual): {torch.sin(2*x.detach()).item():.6f}")
```

**Output:**
```
x.grad (autograd): 0.756802
sin(2x) (manual): 0.756802
```

## In-Place Operations and Graphs

### The Version Counter

PyTorch tracks tensor modifications using a version counter:

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2
print(f"y depends on x version: {x._version}")

# In-place modification
x.data += 1  # Modifies x without going through autograd
print(f"After in-place: x._version changed")

# This can cause issues!
try:
    y.sum().backward()
except RuntimeError as e:
    print(f"Error: one of the variables needed for gradient computation has been modified")
```

### Safe vs. Unsafe In-Place Operations

```python
# UNSAFE: Modifying tensor used in computation
x = torch.tensor([1.], requires_grad=True)
y = x * 2
x += 1  # Modifies x - breaks gradient for y!

# SAFE: Modify after backward or detached copy
x = torch.tensor([1.], requires_grad=True)
y = x * 2
y.sum().backward()
with torch.no_grad():
    x += 1  # Safe - backward already completed
```

## Visualizing Computational Graphs

### Manual Tracing

```python
def print_graph(tensor, indent=0):
    """Recursively print computational graph structure."""
    prefix = "  " * indent
    print(f"{prefix}Tensor shape: {tensor.shape}")
    
    if tensor.grad_fn is not None:
        print(f"{prefix}grad_fn: {tensor.grad_fn}")
        for func, idx in tensor.grad_fn.next_functions:
            if func is not None:
                print(f"{prefix}  ↑ {func}")

# Example
x = torch.randn(2, 3, requires_grad=True)
y = x.sum(dim=1)
z = y.mean()

print_graph(z)
```

### Using torchviz (Optional)

For complex networks, the `torchviz` library creates visual diagrams:

```python
# pip install torchviz
from torchviz import make_dot

x = torch.randn(2, 3, requires_grad=True)
y = x ** 2
z = y.sum()

# Creates a visual graph (saves to file)
dot = make_dot(z, params={'x': x})
dot.render('computational_graph', format='png')
```

## Multiple Paths in the Graph

### When Variables Appear Multiple Times

If a tensor is used multiple times, gradients are **summed**:

```python
x = torch.tensor([3.0], requires_grad=True)

# x appears twice in the computation
y = x ** 2 + x ** 3  # y = x^2 + x^3

y.backward()

# dy/dx = 2x + 3x^2 = 2(3) + 3(9) = 6 + 27 = 33
print(f"x.grad: {x.grad}")  # tensor([33.])
```

### Branching and Merging

```python
x = torch.tensor([2.0], requires_grad=True)

# Branch
y1 = x * 2      # y1 = 2x
y2 = x * 3      # y2 = 3x

# Merge
z = y1 + y2     # z = 2x + 3x = 5x

z.backward()
print(f"x.grad: {x.grad}")  # tensor([5.])
```

The gradient correctly sums contributions from both branches.

## Training Loop Pattern

Understanding graphs is crucial for training:

```python
import torch
import torch.nn as nn

# Setup
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(32, 10)
y = torch.randn(32, 1)

for epoch in range(100):
    # ═══════════════════════════════════════
    # FORWARD PASS: Build computational graph
    # ═══════════════════════════════════════
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    
    # ═══════════════════════════════════════
    # BACKWARD PASS: Traverse graph, compute gradients
    # ═══════════════════════════════════════
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Graph is consumed here
    
    # ═══════════════════════════════════════
    # UPDATE: Modify parameters (outside graph)
    # ═══════════════════════════════════════
    optimizer.step()       # Uses no_grad internally
    
    # Graph is now freed; next iteration builds a new one
```

## Summary

| Concept | Description |
|---------|-------------|
| **Dynamic Graphs** | Built during forward pass, consumed during backward |
| **grad_fn** | Stores backward function and references to inputs |
| **retain_graph** | Prevents graph deletion for multiple backward passes |
| **Version Counter** | Tracks tensor modifications; detects invalid gradients |
| **Gradient Summing** | Multiple uses of a tensor sum their gradient contributions |
| **Graph Lifecycle** | Created (forward) → Consumed (backward) → Freed (memory) |

## Common Pitfalls

1. **Calling backward twice without `retain_graph=True`**
2. **In-place operations on tensors needed for gradients**
3. **Forgetting to zero gradients between iterations**
4. **Memory leaks from indefinitely retained graphs**

## References

- PyTorch Autograd Mechanics: https://pytorch.org/docs/stable/notes/autograd.html
- Dynamic vs Static Graphs: https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/
