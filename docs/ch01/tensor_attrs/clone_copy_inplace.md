# Clone, Copy, and In-Place Operations

## Overview

Understanding when PyTorch creates copies versus views is crucial for memory management and avoiding subtle bugs. This section covers the various methods for duplicating tensors and the implications of in-place operations.

## Views vs Copies

### What is a View?

A **view** is a tensor that shares storage with another tensor. Changes to one affect the other.

```python
import torch

original = torch.arange(6)
view = original.view(2, 3)

print(f"Same storage: {original.storage().data_ptr() == view.storage().data_ptr()}")
# True

# Modifying view affects original
view[0, 0] = 99
print(f"Original after view modification: {original}")
# tensor([99, 1, 2, 3, 4, 5])
```

### What is a Copy?

A **copy** has its own storage. Changes are independent.

```python
original = torch.arange(6)
copy = original.clone()

print(f"Same storage: {original.storage().data_ptr() == copy.storage().data_ptr()}")
# False

# Modifying copy doesn't affect original
copy[0] = 99
print(f"Original after copy modification: {original}")
# tensor([0, 1, 2, 3, 4, 5])
```

## The `clone()` Method

### Basic Cloning

```python
t = torch.tensor([1.0, 2.0, 3.0])

# Create independent copy
t_clone = t.clone()

# Verify independence
t_clone[0] = 99.0
print(f"Original: {t}")       # tensor([1., 2., 3.])
print(f"Clone: {t_clone}")    # tensor([99., 2., 3.])
```

### Clone with Gradients

`clone()` preserves gradient tracking and creates a node in the computation graph:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.clone()

print(f"x.requires_grad: {x.requires_grad}")  # True
print(f"y.requires_grad: {y.requires_grad}")  # True

# y is in the computation graph
z = y.sum()
z.backward()
print(f"x.grad: {x.grad}")  # tensor([1., 1., 1.])
```

### Memory Layout Preservation

```python
# Clone preserves memory layout
t = torch.randn(3, 4)
t_T = t.T  # Non-contiguous

print(f"Transposed contiguous: {t_T.is_contiguous()}")  # False

clone_T = t_T.clone()
print(f"Clone contiguous: {clone_T.is_contiguous()}")  # True (made contiguous!)
```

## The `detach()` Method

### Removing from Computation Graph

`detach()` creates a tensor that shares storage but doesn't track gradients:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# Detach from graph
y_detached = y.detach()

print(f"y.requires_grad: {y.requires_grad}")           # True
print(f"y_detached.requires_grad: {y_detached.requires_grad}")  # False

# Still shares storage!
print(f"Same storage: {y.storage().data_ptr() == y_detached.storage().data_ptr()}")
# True
```

### Common Pattern: `detach().clone()`

For a completely independent copy without gradient tracking:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# Independent copy, no gradients
independent = y.detach().clone()

print(f"requires_grad: {independent.requires_grad}")  # False
print(f"Same storage: {y.storage().data_ptr() == independent.storage().data_ptr()}")
# False
```

## In-Place Operations

### Identifying In-Place Operations

In-place operations end with an underscore (`_`):

```python
t = torch.tensor([1.0, 2.0, 3.0])

# Out-of-place (creates new tensor)
t_new = t.add(10)
print(f"Original: {t}")      # tensor([1., 2., 3.])
print(f"New: {t_new}")       # tensor([11., 12., 13.])

# In-place (modifies existing tensor)
t.add_(10)
print(f"After add_: {t}")    # tensor([11., 12., 13.])
```

### Common In-Place Operations

```python
t = torch.tensor([1.0, 2.0, 3.0])

# Arithmetic
t.add_(10)       # t = t + 10
t.sub_(5)        # t = t - 5
t.mul_(2)        # t = t * 2
t.div_(2)        # t = t / 2

# Assignment
t.zero_()        # Fill with zeros
t.fill_(5)       # Fill with value
t.uniform_()     # Fill with uniform random
t.normal_()      # Fill with normal random

# Clipping
t.clamp_(0, 10)  # Clamp to range

# Copying
t.copy_(other)   # Copy from another tensor
```

### In-Place and Autograd

In-place operations can cause issues with autograd:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# This will cause an error during backward!
# y.add_(1)  # RuntimeError: modified by an in-place operation

# Safe: use out-of-place
y_safe = y + 1
```

### Version Counter

PyTorch tracks tensor modifications:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

print(f"y version before: {y._version}")  # 0

# In-place modification
y.data.add_(1)  # Bypasses autograd (dangerous!)

print(f"y version after: {y._version}")   # 1

# backward() may now fail or give wrong gradients
```

## The `data` Attribute

### Accessing Underlying Data

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# .data gives view without gradient tracking
x_data = x.data

print(f"x.requires_grad: {x.requires_grad}")          # True
print(f"x_data.requires_grad: {x_data.requires_grad}") # False

# Same storage
print(f"Same storage: {x.storage().data_ptr() == x_data.storage().data_ptr()}")
# True
```

!!! warning "Avoid Using `.data`"
    Using `.data` can lead to incorrect gradients. Prefer `detach()` for
    creating tensors without gradient tracking.

## Copy Operations

### `copy_()` - In-Place Copy

```python
target = torch.zeros(3)
source = torch.tensor([1.0, 2.0, 3.0])

# Copy source into target
target.copy_(source)
print(f"Target after copy_: {target}")  # tensor([1., 2., 3.])

# Supports broadcasting
target = torch.zeros(3, 3)
source = torch.tensor([1.0, 2.0, 3.0])
target.copy_(source)  # Broadcasts source to each row
print(f"Broadcast copy:\n{target}")
```

### Creating Copies with Same Properties

```python
x = torch.randn(3, 4, dtype=torch.float32, device='cpu')

# zeros_like, ones_like, etc. create new tensors with same properties
zeros = torch.zeros_like(x)
ones = torch.ones_like(x)
empty = torch.empty_like(x)
rand = torch.rand_like(x)

print(f"All have dtype={zeros.dtype}, device={zeros.device}, shape={zeros.shape}")
```

## Contiguous Copy

### Making Contiguous

```python
t = torch.randn(3, 4)
t_T = t.T

print(f"Transposed contiguous: {t_T.is_contiguous()}")  # False

# contiguous() creates a contiguous copy
t_T_cont = t_T.contiguous()

print(f"After contiguous: {t_T_cont.is_contiguous()}")  # True
print(f"Same storage: {t_T.storage().data_ptr() == t_T_cont.storage().data_ptr()}")
# False - new memory
```

### When Contiguous is Needed

```python
t = torch.randn(3, 4).T  # Non-contiguous

# view() requires contiguous
# t.view(-1)  # Error!

# Options:
# 1. Make contiguous first
flat1 = t.contiguous().view(-1)

# 2. Use reshape (handles automatically)
flat2 = t.reshape(-1)

# 3. Use flatten
flat3 = t.flatten()
```

## Practical Patterns

### Safe Tensor Modification in Training

```python
# BAD: In-place modification of leaf tensor
x = torch.tensor([1.0], requires_grad=True)
# x.add_(1)  # Error!

# GOOD: Out-of-place
x = x + 1  # Creates new tensor, reassigns

# Or use clone
x_modified = x.clone()
x_modified.add_(1)  # OK on clone
```

### Efficient Batch Assembly

```python
# Pre-allocate and copy
batch_size = 32
feature_dim = 256

batch = torch.empty(batch_size, feature_dim)
for i, sample in enumerate(samples[:batch_size]):
    batch[i].copy_(sample)  # In-place copy into pre-allocated
```

### Parameter Updates

```python
# Common pattern in optimizers
param = torch.randn(10, requires_grad=True)
grad = param.grad

# Update in-place using .data
with torch.no_grad():
    param.data.add_(-0.01 * grad)

# Or better, use optimizer
optimizer = torch.optim.SGD([param], lr=0.01)
optimizer.step()
```

### Moving Between Devices

```python
# to() creates a copy on the new device
cpu_tensor = torch.randn(3, 4)
gpu_tensor = cpu_tensor.to('cuda')  # Copy to GPU

print(f"Same storage: {cpu_tensor.storage().data_ptr() == gpu_tensor.storage().data_ptr()}")
# False - different devices, different storage

# In-place device move (if supported)
# tensor.cuda_()  # Some versions support this
```

## Summary

| Operation | Creates Copy? | Shares Storage? | Tracks Gradients? |
|-----------|---------------|-----------------|-------------------|
| `view()` | No | Yes | Yes |
| `reshape()` | Maybe | Maybe | Yes |
| `clone()` | Yes | No | Yes |
| `detach()` | No | Yes | No |
| `detach().clone()` | Yes | No | No |
| `contiguous()` | If needed | No (if copied) | Yes |
| `to(device)` | Yes | No | Yes |
| `copy_()` | No (in-place) | N/A | N/A |

## See Also

- [Memory Layout and Strides](../tensors/memory_layout_strides.md) - Storage details
- [Reshaping and View Operations](reshaping_view.md) - View operations
- [Autograd Fundamentals](../gradients/autograd_fundamentals.md) - Gradient tracking
