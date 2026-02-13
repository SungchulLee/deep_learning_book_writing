# Computational Graphs

## Overview

A computational graph is a directed acyclic graph (DAG) that represents the sequence of operations performed on tensors during the forward pass. PyTorch builds this graph **dynamically** as operations execute, then traverses it in reverse during backpropagation to compute gradients. Understanding computational graphs is essential for debugging training issues, optimizing memory usage, and reasoning about gradient flow.

## Learning Objectives

By the end of this section, you will be able to:

1. Understand how PyTorch constructs computational graphs dynamically (define-by-run)
2. Trace the relationship between tensor operations and `grad_fn` attributes
3. Distinguish between leaf and non-leaf tensors
4. Manage graph lifecycle with `retain_graph`
5. Reason about in-place operations and the version counter
6. Visualize and debug gradient flow through networks

## Graph Structure

### Nodes and Edges

A computational graph consists of two types of nodes connected by directed edges:

**Nodes:**

- **Leaf nodes** — Input tensors created directly by the user, with `requires_grad=True` and `grad_fn=None`. Only leaf tensors accumulate gradients by default.
- **Intermediate (operation) nodes** — Tensors produced by operations on other tensors. Each carries a `grad_fn` attribute that records which backward function created it.

**Edges:**

Directed edges encode data dependencies: each intermediate node points back to the nodes that produced its inputs. During backpropagation, these edges are traversed in reverse to propagate gradients.

```
x ─────────→ [**2] ────→ y ────→ [sum] ────→ loss
(leaf)      (PowBackward)      (SumBackward)   (scalar)
```

### Leaf vs Non-Leaf Tensors

A tensor $x$ is a **leaf tensor** if and only if it was created directly by the user (not as the result of an operation). Formally:

- `x.is_leaf == True`
- `x.grad_fn is None`
- Only leaf tensors with `requires_grad=True` store gradients after `.backward()`

```python
import torch

# Leaf tensor — created by user
x = torch.randn(3, requires_grad=True)
print(f"x.is_leaf: {x.is_leaf}")          # True
print(f"x.grad_fn: {x.grad_fn}")          # None

# Non-leaf tensors — results of operations
y = 2 * x
z = y ** 2

print(f"y.is_leaf: {y.is_leaf}")          # False
print(f"y.grad_fn: {y.grad_fn}")          # <MulBackward0>
print(f"z.grad_fn: {z.grad_fn}")          # <PowBackward0>
```

### The `requires_grad` Propagation Rule

If **any** input to an operation has `requires_grad=True`, the output inherits `requires_grad=True`:

```python
import torch

a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=False)
c = a + b

print(f"a.requires_grad: {a.requires_grad}")  # True
print(f"b.requires_grad: {b.requires_grad}")  # False
print(f"c.requires_grad: {c.requires_grad}")  # True (inherits from a)
```

## Dynamic Graph Construction

### Define-by-Run Semantics

Unlike static-graph frameworks, PyTorch builds the computational graph **as operations execute**. Each operation appends a new node to the graph:

```python
import torch

# Step 1: Create leaf tensor — no graph yet
x = torch.tensor([1., 2., 3.], requires_grad=True)
print(f"After creation: x.grad_fn = {x.grad_fn}")    # None (leaf)

# Step 2: First operation — graph grows
y = x ** 2
print(f"After y = x**2: y.grad_fn = {y.grad_fn}")    # PowBackward0

# Step 3: Second operation — graph grows further
z = y.sum()
print(f"After z = y.sum(): z.grad_fn = {z.grad_fn}") # SumBackward0
```

This means the graph can differ on every forward pass — enabling dynamic control flow (loops, conditionals) inside models.

### Tracing the Graph via `next_functions`

Each `grad_fn` stores references to the backward functions of its inputs through `next_functions`:

```python
import torch

x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.sum()

# Trace backward through the graph
print(f"z.grad_fn: {z.grad_fn}")
print(f"z.grad_fn.next_functions: {z.grad_fn.next_functions}")
# → ((MulBackward0, 0),)

print(f"y.grad_fn: {y.grad_fn}")
print(f"y.grad_fn.next_functions: {y.grad_fn.next_functions}")
# → ((AccumulateGrad, 0),)  — AccumulateGrad is the terminal node for leaves
```

A recursive traversal of `next_functions` reconstructs the full graph:

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

x = torch.randn(2, 3, requires_grad=True)
y = x.sum(dim=1)
z = y.mean()

print_graph(z)
```

## Graph Lifecycle and Memory

### Graph Consumption During Backward

By default, `backward()` **frees** the computational graph after execution to reclaim memory:

```python
import torch

x = torch.randn(3, requires_grad=True)
loss = (x ** 2).sum()

# First backward — graph is consumed and freed
loss.backward()
print(f"x.grad after 1st backward: {x.grad}")

# Second backward fails — graph no longer exists
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

To perform multiple backward passes on the same graph, pass `retain_graph=True`:

```python
import torch

x = torch.randn(3, requires_grad=True)
loss = (x ** 2).sum()

# First backward — retain the graph
loss.backward(retain_graph=True)
grad_first = x.grad.clone()

# Second backward succeeds, but gradients ACCUMULATE
loss.backward()
grad_second = x.grad.clone()

print(f"After 1st backward: {grad_first}")
print(f"After 2nd backward (accumulated): {grad_second}")
```

**Common use cases for `retain_graph`:**

1. Computing gradients with respect to different upstream gradient vectors
2. Higher-order derivatives (gradients of gradients; see [Higher-Order Gradients](higher_order_gradients.md))
3. Multiple loss functions sharing intermediate computations

**Warning:** Retaining graphs indefinitely causes memory leaks. Always let the graph be freed after the final backward call.

## The Chain Rule in Action

### Backward Function Chain

The backward pass applies the chain rule through the stored `grad_fn` chain. For a composition $L = h(g(f(x)))$:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial g} \cdot \frac{\partial g}{\partial f} \cdot \frac{\partial f}{\partial x}$$

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

# Forward: y = sin(x), z = y²
y = torch.sin(x)
z = y ** 2

# Chain rule: dz/dx = dz/dy · dy/dx = 2y · cos(x) = 2sin(x)cos(x) = sin(2x)
z.backward()

print(f"x.grad (autograd): {x.grad.item():.6f}")
print(f"sin(2x) (manual):  {torch.sin(2 * x.detach()).item():.6f}")
```

**Output:**
```
x.grad (autograd): 0.756802
sin(2x) (manual):  0.756802
```

### Multiple Paths: Gradient Summing

When a tensor is used multiple times in a computation, gradients from all paths are **summed** (the multivariate chain rule):

```python
import torch

x = torch.tensor([3.0], requires_grad=True)

# x appears in two terms
y = x ** 2 + x ** 3    # y = x² + x³

y.backward()

# dy/dx = 2x + 3x² = 2(3) + 3(9) = 33
print(f"x.grad: {x.grad}")  # tensor([33.])
```

### Branching and Merging

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

# Branch into two paths
y1 = x * 2    # y1 = 2x
y2 = x * 3    # y2 = 3x

# Merge
z = y1 + y2   # z = 5x

z.backward()
print(f"x.grad: {x.grad}")  # tensor([5.])
```

## In-Place Operations and the Version Counter

### The Version Counter

PyTorch tracks modifications to tensor data using an internal version counter. The backward pass checks that tensors have not been modified since the graph was built:

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2
print(f"x._version before modification: {x._version}")

# In-place modification via .data bypasses autograd
x.data += 1
print(f"x._version after modification: {x._version}")

# Backward may fail or give incorrect results
try:
    y.sum().backward()
except RuntimeError as e:
    print(f"Error: {e}")
```

### Safe vs Unsafe In-Place Operations

```python
import torch

# UNSAFE: Modifying a tensor still needed by the graph
x = torch.tensor([1.], requires_grad=True)
y = x * 2
# x += 1  ← would break gradient computation for y

# SAFE: Modify after backward has completed
x = torch.tensor([1.], requires_grad=True)
y = x * 2
y.sum().backward()
with torch.no_grad():
    x += 1  # Safe — backward already consumed the graph
```

**Rule of thumb:** Avoid in-place operations on any tensor that participates in an active computational graph.

## Retaining Gradients for Non-Leaf Tensors

By default, only leaf tensors store gradients. Use `.retain_grad()` before the backward pass to inspect gradients at intermediate nodes:

```python
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = 2 * x
y.retain_grad()       # Request gradient storage for y
z = (y ** 2).sum()

z.backward()

print(f"x.grad: {x.grad}")  # tensor([8., 16., 24.])
print(f"y.grad: {y.grad}")  # tensor([4., 8., 12.])
```

**Verification:** $y = 2x = [2, 4, 6]$, $z = \sum y_i^2 = 56$, $\partial z / \partial y_i = 2y_i$, $\partial z / \partial x_i = 2y_i \cdot 2 = 4y_i$.

## Visualizing Graphs with torchviz

For complex networks, the `torchviz` library generates visual diagrams of computational graphs:

```python
# pip install torchviz
from torchviz import make_dot

x = torch.randn(2, 3, requires_grad=True)
y = x ** 2
z = y.sum()

dot = make_dot(z, params={'x': x})
dot.render('computational_graph', format='png')
```

## Training Loop and Graph Lifecycle

Understanding the graph lifecycle is critical for writing correct training loops. Each iteration builds a **fresh** graph:

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(32, 10)
y = torch.randn(32, 1)

for epoch in range(100):
    # ── FORWARD PASS: build computational graph ──
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    
    # ── BACKWARD PASS: traverse graph, compute gradients ──
    optimizer.zero_grad()   # Clear previous gradients
    loss.backward()         # Graph is consumed here
    
    # ── UPDATE: modify parameters (outside graph) ──
    optimizer.step()        # Uses torch.no_grad() internally
    
    # Graph is now freed; next iteration builds a new one
```

## Summary

| Concept | Description |
|---------|-------------|
| **Dynamic Graphs** | Built during forward pass, consumed during backward |
| **Leaf Tensors** | User-created, `grad_fn=None`, store gradients by default |
| **grad_fn** | Backward function stored on each intermediate tensor |
| **next_functions** | Links from each `grad_fn` to its input backward functions |
| **retain_graph** | Prevents graph deletion for multiple backward passes |
| **Version Counter** | Detects invalid in-place modifications |
| **Gradient Summing** | Multiple uses of a tensor sum gradient contributions |
| **Graph Lifecycle** | Created (forward) → Consumed (backward) → Freed (memory) |

## Common Pitfalls

1. **Calling `backward()` twice** without `retain_graph=True`
2. **In-place operations** on tensors needed for gradient computation
3. **Forgetting to zero gradients** between training iterations (see [Gradient Accumulation](gradient_accumulation.md))
4. **Memory leaks** from indefinitely retained graphs

## References

- PyTorch Autograd Mechanics: [https://pytorch.org/docs/stable/notes/autograd.html](https://pytorch.org/docs/stable/notes/autograd.html)
- Dynamic vs Static Graphs: [https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/](https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/)
