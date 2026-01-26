# Memory Layout and Strides

## Overview

Understanding how PyTorch stores tensors in memory is crucial for writing efficient code and avoiding subtle bugs. This section covers contiguity, strides, and the difference between views and copies.

## Memory Layout Fundamentals

### Contiguous Memory

A tensor is **contiguous** when its elements are stored in memory in the same order as they would be accessed when iterating through dimensions from last to first (row-major order, also known as C-order).

```python
import torch

# Create a contiguous tensor
t = torch.arange(12).reshape(3, 4)
print(f"Values:\n{t}")
print(f"Is contiguous: {t.is_contiguous()}")  # True

# Memory layout (row-major):
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#  ↑        ↑        ↑
#  row 0    row 1    row 2
```

### Strides Explained

**Strides** indicate how many elements to skip in memory to move one position along each dimension:

$$\text{memory\_offset} = \sum_{i=0}^{n-1} \text{index}_i \times \text{stride}_i$$

```python
t = torch.arange(12).reshape(3, 4)
print(f"Shape: {t.shape}")    # torch.Size([3, 4])
print(f"Stride: {t.stride()}") # (4, 1)

# Interpretation:
# - Moving 1 step in dim 0 (down a row): skip 4 elements
# - Moving 1 step in dim 1 (across a column): skip 1 element

# Verify: t[1, 2] is at position 1*4 + 2*1 = 6
print(f"t[1, 2] = {t[1, 2]}")  # tensor(6)
```

### Visualizing Strides

```
Memory:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
          ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓   ↓   ↓
Logical: [[0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 10, 11]]

Shape (3, 4):
- stride[0] = 4 (jump 4 to next row)
- stride[1] = 1 (jump 1 to next column)
```

## Views vs Copies

### Views Share Memory

Many operations return **views** that share the underlying data:

```python
original = torch.arange(12).reshape(3, 4)

# reshape() returns a view if possible
reshaped = original.reshape(4, 3)

# Verify they share storage
print(f"Same storage: {original.storage().data_ptr() == reshaped.storage().data_ptr()}")
# True

# Modifying one affects the other!
original[0, 0] = 99
print(f"reshaped[0, 0] after change: {reshaped[0, 0]}")  # tensor(99)
```

### Transposition Creates Non-Contiguous Views

```python
mat = torch.arange(6).reshape(2, 3)
print(f"Original:\n{mat}")
print(f"Original stride: {mat.stride()}")  # (3, 1)
print(f"Original contiguous: {mat.is_contiguous()}")  # True

# Transpose creates a view with different strides
mat_T = mat.T
print(f"\nTransposed:\n{mat_T}")
print(f"Transposed stride: {mat_T.stride()}")  # (1, 3)
print(f"Transposed contiguous: {mat_T.is_contiguous()}")  # False
```

The transposed tensor is **not contiguous** because the strides don't follow the expected pattern for its shape. The elements are still accessed from the same memory, just in a different order.

```
Original memory: [0, 1, 2, 3, 4, 5]
                  ↓  ↓  ↓  ↓  ↓  ↓
Original view:   [[0, 1, 2],
                  [3, 4, 5]]

Transposed view: [[0, 3],    (same memory, different access pattern)
                  [1, 4],
                  [2, 5]]
```

## The `view()` vs `reshape()` Distinction

### `view()` - Requires Contiguity

```python
t = torch.arange(6).reshape(2, 3)
t_T = t.T  # Non-contiguous

# view() fails on non-contiguous tensors
try:
    flat = t_T.view(-1)
except RuntimeError as e:
    print(f"Error: view() requires contiguous tensor")
```

### `reshape()` - Works Always

```python
# reshape() handles non-contiguous tensors by copying if needed
flat = t_T.reshape(-1)  # Works!
print(f"Reshaped: {flat}")

# But it may create a copy
print(f"Same storage: {t.storage().data_ptr() == flat.storage().data_ptr()}")
# False - a copy was made
```

### Making Tensors Contiguous

```python
t_T_contiguous = t_T.contiguous()
print(f"Now contiguous: {t_T_contiguous.is_contiguous()}")  # True

# Now view() works
flat_view = t_T_contiguous.view(-1)
```

!!! tip "When to Use Which"
    - Use `view()` when you know the tensor is contiguous (faster, no copy)
    - Use `reshape()` when you're unsure (safer, may copy)
    - Call `contiguous()` explicitly when you need guaranteed contiguous memory

## Clone vs Detach

### `clone()` - Copy Data

```python
original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# clone() creates a copy with same gradient tracking
cloned = original.clone()

print(f"Original: {original}")
print(f"Cloned: {cloned}")
print(f"Same storage: {original.storage().data_ptr() == cloned.storage().data_ptr()}")
# False - different memory

print(f"Clone requires_grad: {cloned.requires_grad}")  # True
```

### `detach()` - Remove from Graph

```python
original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# detach() creates a view without gradient tracking
detached = original.detach()

print(f"Detached requires_grad: {detached.requires_grad}")  # False
print(f"Same storage: {original.storage().data_ptr() == detached.storage().data_ptr()}")
# True - same memory!
```

### Common Pattern: `detach().clone()`

When you need an independent copy without gradient tracking:

```python
# Creates independent copy without gradient graph
independent = original.detach().clone()
```

## In-Place Operations

In-place operations modify tensors directly and end with an underscore:

```python
t = torch.tensor([1.0, 2.0, 3.0])
print(f"Original id: {id(t)}")

# In-place addition
t.add_(10)
print(f"After add_: {t}")
print(f"Same id: {id(t)}")  # Same object

# Out-of-place addition
t2 = t.add(10)
print(f"After add: original {t}, new {t2}")  # Original unchanged
```

!!! warning "In-Place Operations and Autograd"
    In-place operations can break gradient computation if they modify
    tensors that are needed for backward pass. Use with caution during training.

## Storage Object

All tensors use a **Storage** object that holds the actual data:

```python
t = torch.arange(6).reshape(2, 3)
storage = t.storage()

print(f"Storage size: {storage.size()}")  # 6
print(f"Storage type: {type(storage)}")   # torch.storage.TypedStorage
print(f"Data pointer: {storage.data_ptr()}")

# Multiple tensors can share storage
t_view = t[0]  # First row
print(f"Same storage: {t.storage().data_ptr() == t_view.storage().data_ptr()}")
# True
```

## Memory Inspection Utilities

```python
def tensor_info(t, name="Tensor"):
    """Display comprehensive tensor memory information."""
    print(f"=== {name} ===")
    print(f"  Shape: {t.shape}")
    print(f"  Stride: {t.stride()}")
    print(f"  Contiguous: {t.is_contiguous()}")
    print(f"  Storage offset: {t.storage_offset()}")
    print(f"  Storage size: {t.storage().size()}")
    print(f"  Data pointer: {t.storage().data_ptr()}")
    print()

# Example usage
t = torch.arange(12).reshape(3, 4)
tensor_info(t, "Original")
tensor_info(t.T, "Transposed")
tensor_info(t[1:], "Sliced")
```

## Practical Examples

### Efficient Batch Processing

```python
# Bad: Creates many small tensors
batch_bad = [torch.randn(224, 224) for _ in range(32)]

# Good: One contiguous tensor
batch_good = torch.randn(32, 224, 224)

# Access individual images via views (no copy)
first_image = batch_good[0]  # View into batch
```

### Avoiding Unnecessary Copies

```python
# Scenario: Normalize data
data = torch.randn(1000, 100)

# Method 1: Creates intermediate copies
mean = data.mean(dim=0)
std = data.std(dim=0)
normalized_v1 = (data - mean) / std  # Creates copies

# Method 2: In-place (if data can be modified)
data.sub_(data.mean(dim=0))
data.div_(data.std(dim=0))  # Modifies data in-place
```

## Summary

| Concept | Description |
|---------|-------------|
| Contiguous | Elements in memory match iteration order |
| Stride | Steps to skip for each dimension |
| View | Shares memory with original |
| Copy | Independent memory allocation |
| `view()` | Reshape contiguous tensors (fast) |
| `reshape()` | Reshape any tensor (may copy) |
| `clone()` | Create independent copy |
| `detach()` | Remove from computation graph |
| `contiguous()` | Make tensor contiguous (may copy) |

## See Also

- [Tensor Basics](tensor_basics.md) - Fundamental concepts
- [Reshaping and View Operations](../tensor_attrs/reshaping_view.md) - Detailed reshaping
- [Broadcasting Rules](../tensor_attrs/broadcasting_rules.md) - Implicit expansion
