# Memory Management

## Learning Objectives

By the end of this section, you will be able to:

- Understand how PyTorch manages tensor memory through the storage-stride model
- Distinguish between views and copies, and predict which operations produce which
- Use `clone()`, `detach()`, `contiguous()`, and `copy_()` correctly
- Apply in-place operations safely with awareness of autograd constraints
- Manage GPU memory effectively and debug common memory issues
- Write memory-efficient training loops and data pipelines

---

## Overview

Memory management in PyTorch involves understanding how tensors share (or don't share) underlying data, when operations allocate new memory versus reusing existing memory, and how to avoid the subtle bugs that arise from unintended memory sharing. This section unifies three closely related concerns: the view-vs-copy distinction, in-place operations, and GPU memory management.

---

## Views vs Copies

### What is a View?

A **view** is a tensor that shares the same underlying storage as another tensor but interprets it with a different shape, stride, or offset. No data is copied — both tensors point to the same memory:

```python
import torch

original = torch.arange(6)
view = original.view(2, 3)

# Same underlying storage
print(original.storage().data_ptr() == view.storage().data_ptr())  # True

# Modifying view affects original
view[0, 0] = 99
print(original[0])  # 99
```

### What is a Copy?

A **copy** has its own storage. Modifications are completely independent:

```python
original = torch.arange(6)
copy = original.clone()

print(original.storage().data_ptr() == copy.storage().data_ptr())  # False

copy[0] = 99
print(original[0])  # 0 — unchanged
```

### Quick Test for Shared Storage

```python
def shares_storage(a, b):
    """Check if two tensors share underlying storage."""
    return a.storage().data_ptr() == b.storage().data_ptr()

x = torch.randn(3, 4)
print(shares_storage(x, x.view(-1)))     # True — view
print(shares_storage(x, x.clone()))       # False — copy
print(shares_storage(x, x.T))             # True — view
print(shares_storage(x, x.contiguous()))  # True (already contiguous)
```

### Which Operations Create Views vs Copies?

| Operation | Creates | Notes |
|-----------|---------|-------|
| `view()`, `reshape()`* | View | *`reshape` copies if non-contiguous |
| `transpose()`, `permute()`, `.T` | View | Changes strides, not data |
| Basic slicing `t[a:b]` | View | Adjusts offset and strides |
| `squeeze()`, `unsqueeze()` | View | Adjusts strides |
| `expand()` | View | Uses stride 0 |
| `clone()` | Copy | Always copies data |
| `contiguous()` | Copy if needed | Returns self if already contiguous |
| `repeat()` | Copy | Always copies data |
| Boolean/fancy indexing | Copy | Always copies data |
| `to(device)` / `to(dtype)` | Copy | New storage (unless same device/dtype) |

---

## Clone, Detach, and Copy Operations

### `clone()` — Independent Copy with Gradient Tracking

`clone()` creates a new tensor with its own storage, preserving `requires_grad` and creating a node in the computation graph:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.clone()

print(y.requires_grad)  # True
print(shares_storage(x, y))  # False — independent memory

# Gradient flows through clone
z = y.sum()
z.backward()
print(x.grad)  # tensor([1., 1., 1.])
```

Cloning a non-contiguous tensor produces a contiguous copy:

```python
t = torch.randn(3, 4).T  # Non-contiguous
clone = t.clone()
print(clone.is_contiguous())  # True
```

### `detach()` — Remove from Computation Graph

`detach()` creates a tensor that shares storage but doesn't track gradients:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.detach()

print(y.requires_grad)  # False
print(shares_storage(x, y))  # True — same memory!

# Modifying y affects x!
y[0] = 99
print(x[0])  # tensor(99.)
```

### `detach().clone()` — The Standard Snapshot Pattern

For a completely independent copy without gradient tracking:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
snapshot = x.detach().clone()

print(snapshot.requires_grad)  # False
print(shares_storage(x, snapshot))  # False — independent

# Common use cases:
# 1. Logging during training
loss_value = loss.detach().clone()

# 2. Converting to NumPy
arr = tensor.detach().cpu().numpy()

# 3. Saving checkpoints
best_weights = model.state_dict()
best_weights = {k: v.detach().clone() for k, v in best_weights.items()}
```

### `copy_()` — In-Place Copy

Copies data from source into target without allocating new storage:

```python
target = torch.zeros(3)
source = torch.tensor([1.0, 2.0, 3.0])

target.copy_(source)
print(target)  # tensor([1., 2., 3.])

# Supports broadcasting
target = torch.zeros(3, 3)
target.copy_(torch.tensor([1.0, 2.0, 3.0]))  # Broadcasts to each row
```

### `contiguous()` — Ensure Contiguous Memory Layout

Returns the tensor itself if already contiguous, otherwise creates a contiguous copy:

```python
t = torch.randn(3, 4)
t_cont = t.contiguous()
print(t is t_cont)  # True — already contiguous, returns self

t_T = t.T
t_T_cont = t_T.contiguous()
print(shares_storage(t_T, t_T_cont))  # False — copy was made
print(t_T_cont.is_contiguous())  # True
```

### Creating Tensors with Matching Properties

```python
x = torch.randn(3, 4, dtype=torch.float64, device='cpu')

# *_like functions: match shape, dtype, and device
zeros = torch.zeros_like(x)
ones = torch.ones_like(x)
rand = torch.randn_like(x)

# new_* methods: match dtype and device, custom shape
y = x.new_zeros(2, 3)
z = x.new_full((2, 2), 3.14)
```

### The `.data` Attribute (Avoid)

`.data` provides a view without gradient tracking but bypasses autograd safety checks:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
x_data = x.data  # No gradient tracking, shares storage

# Dangerous: can silently corrupt gradients
# Prefer detach() instead
```

### Decision Guide

**Need an independent copy?**

- With gradients: `tensor.clone()`
- Without gradients: `tensor.detach().clone()`

**Need to stop gradient flow?**

- Keep storage: `tensor.detach()`
- New storage: `tensor.detach().clone()`

**Need contiguous memory?**

- `tensor.contiguous()` — copies only if needed

**Need same properties, different values?**

- `torch.zeros_like(tensor)`, `torch.randn_like(tensor)`, etc.

**Need to copy into existing tensor?**

- `target.copy_(source)`

---

## In-Place Operations

In-place operations modify tensors directly without allocating new storage. They are identified by an underscore suffix.

### Common In-Place Operations

```python
x = torch.tensor([1., 2., 3., 4., 5.])

# Arithmetic
x.add_(10)       # x += 10
x.sub_(5)        # x -= 5
x.mul_(2)        # x *= 2
x.div_(4)        # x /= 4
x.pow_(2)        # x **= 2

# Clamping and rounding
x.clamp_(min=0)              # ReLU-like
x.clamp_(min=-1, max=1)      # Clamp to range
x.floor_()                   # Round down
x.round_()                   # Round to nearest

# Initialization
x.fill_(7.0)                 # All elements = 7
x.zero_()                    # All zeros
x.uniform_(-1, 1)            # Uniform random
x.normal_(mean=0, std=1)     # Normal random

# Activation functions
x.sigmoid_()
x.tanh_()
x.relu_()
```

### Aliasing: Views Share Storage

In-place operations modify the underlying storage, affecting **all** tensors that share it:

```python
x = torch.arange(12)
view = x.view(3, 4)

view.fill_(0)
print(x)  # All zeros — x was modified through the view!

# Slices are also views
y = x[2:5]
y.mul_(10)  # Modifies elements 2, 3, 4 of x
```

### Autograd Restrictions

In-place operations on tensors involved in gradient computation are **prohibited**:

```python
# On leaf tensors with requires_grad
leaf = torch.tensor([1., 2., 3.], requires_grad=True)
try:
    leaf.add_(1)  # RuntimeError!
except RuntimeError:
    print("Cannot modify leaf tensor in-place")

# On intermediate computation results
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
try:
    y.add_(1)  # RuntimeError!
except RuntimeError:
    print("Cannot modify intermediate tensor in-place")
```

### Safe In-Place Patterns

**1. Inside `torch.no_grad()` (parameter updates):**

```python
with torch.no_grad():
    for param in model.parameters():
        param.add_(-learning_rate * param.grad)
```

**2. On tensors without gradient tracking (preprocessing):**

```python
data = torch.randn(100, 50)  # requires_grad=False by default
data.clamp_(0, 1)
data.div_(data.max())
```

**3. On gradients (gradient clipping):**

```python
loss.backward()
for param in model.parameters():
    if param.grad is not None:
        param.grad.clamp_(-max_norm, max_norm)
```

**4. Weight initialization:**

```python
def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(0, 0.02)
        if module.bias is not None:
            module.bias.data.zero_()
model.apply(init_weights)
```

**5. Attention masking:**

```python
scores = torch.randn(4, 4)
mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))
```

### Indexed Assignment is Implicitly In-Place

```python
x = torch.zeros(5)
x[0] = 1.0
x[1:4] = torch.tensor([10., 20., 30.])
x[x > 15] = -1

# scatter operations
x.scatter_(0, torch.tensor([0, 2, 4]), torch.tensor([1., 2., 3.]))
x.index_fill_(0, torch.tensor([1, 3]), -1)
```

### When to Use vs Avoid

**Use in-place for**: initialization, preprocessing without gradients, parameter updates inside `no_grad()`, memory-critical situations, gradient modification.

**Avoid in-place for**: leaf tensors with `requires_grad=True`, intermediate computation results, forward pass operations in `nn.Module`, when code clarity matters more than memory savings.

### Common Mistake: Unintended Aliasing

```python
# Bug: modifies the original tensor
def process(tensor):
    tensor.mul_(2)  # Modifies the original!
    return tensor

# Fix: clone first
def process(tensor):
    result = tensor.clone()
    result.mul_(2)
    return result
```

---

## GPU Memory Management

### Memory Allocation on GPU

```python
if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device='cuda')

    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
```

### Releasing GPU Memory

```python
# Delete tensor reference
del x

# Release cached memory back to GPU
torch.cuda.empty_cache()

# Synchronize before measuring
torch.cuda.synchronize()
```

### Memory-Efficient Training Pattern

```python
def train_step(model, data, target, optimizer, criterion):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()  # .item() converts to Python scalar — frees graph
```

### Pre-Allocated Buffers

For repetitive operations, pre-allocate output tensors:

```python
# Without pre-allocation: creates new tensor each iteration
for i in range(1000):
    result = torch.matmul(A, B)

# With pre-allocation: reuses memory
result = torch.empty(A.shape[0], B.shape[1], device=A.device)
for i in range(1000):
    torch.matmul(A, B, out=result)
```

---

## Common Memory Issues and Solutions

### Issue 1: Accumulating Computation Graphs

```python
# WRONG: keeps entire graph in memory
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss)  # Graph accumulates!

# CORRECT: detach or convert to scalar
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss.item())  # Frees graph
```

### Issue 2: Hidden References Through Views

```python
# WRONG: slice keeps entire storage alive
x = torch.randn(10000, 10000)
y = x[0, :]  # y shares storage with x
del x  # Storage NOT freed — y still references it

# CORRECT: clone if you need only the slice
y = x[0, :].clone()
del x  # Storage now freed
```

### Issue 3: Forgetting to Detach for Logging

```python
# WRONG: accumulates graphs across epochs
all_outputs = []
for data in dataloader:
    output = model(data)
    all_outputs.append(output)  # Keeps graph!

# CORRECT
all_outputs = []
for data in dataloader:
    output = model(data)
    all_outputs.append(output.detach())  # Breaks graph
```

---

## Memory Profiling

### Using PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True
) as prof:
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_memory_usage"))
```

### Memory Summary

```python
if torch.cuda.is_available():
    print(torch.cuda.memory_summary())
```

### Memory Snapshots

```python
torch.cuda.memory._record_memory_history()
# ... your code ...
snapshot = torch.cuda.memory._snapshot()
```

---

## Best Practices Summary

| Practice | Benefit |
|----------|---------|
| Use `.item()` for scalar loss values | Prevents graph accumulation |
| Use `detach().clone()` for snapshots | Independent copy, no gradient |
| Clone slices if original not needed | Allows garbage collection |
| Use `torch.no_grad()` for inference | Disables gradient tracking |
| Pre-allocate buffers for loops | Avoids repeated allocations |
| Use in-place ops only when safe | Reduces memory allocation |
| Call `empty_cache()` judiciously | Releases unused cached memory |
| Use `*_like` functions | Matches dtype/device automatically |
| Avoid `.data` — use `detach()` | Respects autograd safety checks |

---

## Quick Reference

| Operation | New Storage? | Gradient | Memory Sharing |
|-----------|-------------|----------|----------------|
| `clone()` | Yes | Preserved | Independent |
| `detach()` | No | Removed | Shared |
| `detach().clone()` | Yes | Removed | Independent |
| `contiguous()` | If needed | Preserved | Depends |
| `to(device/dtype)` | Yes | Preserved | Independent |
| `copy_(src)` | No (in-place) | N/A | Target modified |
| `view()`, `reshape()`* | No | Preserved | Shared |
| In-place `*_()` ops | No | N/A | Same tensor |

---

## See Also

- [Memory Layout and Strides](memory_layout_strides.md) — Stride-based memory access patterns
- [Reshaping and Views](reshaping_view.md) — View operations in detail
- [Dtype and Device](dtype_device.md) — Type casting and device transfer
- [Shape Manipulation](shape_manipulation.md) — Indexing view/copy behavior
