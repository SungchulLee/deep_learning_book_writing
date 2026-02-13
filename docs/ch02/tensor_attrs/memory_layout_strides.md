# Memory Layout and Strides

## Learning Objectives

By the end of this section, you will be able to:

- Understand how tensors are stored in memory using the storage-stride model
- Explain the difference between row-major and column-major ordering
- Calculate memory offsets from multi-dimensional indices using strides
- Predict whether operations create views or copies
- Optimize tensor operations by leveraging memory layout knowledge

---

## Overview

Understanding how PyTorch stores tensors in memory is crucial for writing efficient code and avoiding subtle bugs. PyTorch tensors are multi-dimensional arrays stored in contiguous blocks of memory. The **stride** mechanism determines how multi-dimensional indices map to positions in this linear memory block. Understanding strides is fundamental to efficient tensor manipulation and explains many behaviors of reshaping and view operations.

---

## The Storage-Stride Model

### Storage: The Raw Data

Every PyTorch tensor is backed by a **Storage** object—a flat, one-dimensional array of typed elements:

```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# View the underlying storage
print(x.storage())        # [1, 2, 3, 4, 5, 6]
print(len(x.storage()))   # 6 elements total
print(type(x.storage()))  # torch.storage.TypedStorage
print(x.storage().data_ptr())  # Memory address
```

Multiple tensors can share the same storage:

```python
t = torch.arange(6).reshape(2, 3)
t_view = t[0]  # First row

# Both point to the same underlying data
print(t.storage().data_ptr() == t_view.storage().data_ptr())  # True
```

### Strides: The Access Pattern

**Strides** specify how many elements to skip in storage to move one position along each dimension:

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x.stride())  # (3, 1)
# stride[0] = 3: skip 3 elements to move to next row
# stride[1] = 1: skip 1 element to move to next column
```

The element at position $(i, j)$ is located at:

$$
\text{storage\_index} = \text{offset} + i \times \text{stride}[0] + j \times \text{stride}[1]
$$

More generally, for an $n$-dimensional tensor:

$$
\text{storage\_index} = \text{offset} + \sum_{k=0}^{n-1} \text{index}_k \times \text{stride}_k
$$

**Example verification:**

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# Access x[1, 2] manually: offset(0) + 1*3 + 2*1 = 5
print(x.storage()[5])  # 6
print(x[1, 2])         # 6

# Another example
t = torch.arange(12).reshape(3, 4)
print(f"Shape: {t.shape}")     # torch.Size([3, 4])
print(f"Stride: {t.stride()}")  # (4, 1)

# t[1, 2] is at position 1*4 + 2*1 = 6
print(f"t[1, 2] = {t[1, 2]}")  # tensor(6)
```

---

## Row-Major vs Column-Major Order

### Row-Major (C-style) Order

PyTorch uses **row-major** ordering by default, where elements of the same row are stored contiguously:

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
# Storage: [1, 2, 3, 4, 5, 6]
#          |row 0 | row 1 |

print(x.stride())  # (3, 1) - row stride > column stride
```

Visual representation:

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

### Column-Major (Fortran-style) Order

Some libraries (NumPy with `order='F'`, MATLAB) use column-major ordering:

```python
import numpy as np

# NumPy column-major array
arr_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')
# Memory: [1, 4, 2, 5, 3, 6]
#         |c0| |c1| |c2|

# Convert to PyTorch (maintains memory layout)
x_f = torch.from_numpy(arr_f)
print(x_f.stride())  # (1, 2) - column stride > row stride
```

### Contiguity

A tensor is **contiguous** if its memory layout matches the expected row-major pattern:

```python
x = torch.randn(3, 4)
print(x.is_contiguous())  # True

# Transpose changes strides but not storage
x_t = x.t()
print(x_t.stride())         # (1, 4) instead of (4, 1)
print(x_t.is_contiguous())  # False
```

---

## Views vs Copies

### Views Share Memory

Many operations return **views** that share the underlying data:

```python
original = torch.arange(12).reshape(3, 4)

# reshape() returns a view if possible
reshaped = original.reshape(4, 3)

# Verify they share storage
print(original.storage().data_ptr() == reshaped.storage().data_ptr())  # True

# Modifying one affects the other!
original[0, 0] = 99
print(reshaped[0, 0])  # tensor(99)
```

### Transposition Creates Non-Contiguous Views

```python
mat = torch.arange(6).reshape(2, 3)
print(f"Original:\n{mat}")
print(f"Original stride: {mat.stride()}")       # (3, 1)
print(f"Original contiguous: {mat.is_contiguous()}")  # True

# Transpose creates a view with different strides
mat_T = mat.T
print(f"\nTransposed:\n{mat_T}")
print(f"Transposed stride: {mat_T.stride()}")       # (1, 3)
print(f"Transposed contiguous: {mat_T.is_contiguous()}")  # False
```

The transposed tensor is **not contiguous** because the strides don't follow the expected pattern for its shape. The elements are still accessed from the same memory, just in a different order:

```
Original memory: [0, 1, 2, 3, 4, 5]
                  ↓  ↓  ↓  ↓  ↓  ↓
Original view:   [[0, 1, 2],
                  [3, 4, 5]]

Transposed view: [[0, 3],    (same memory, different access pattern)
                  [1, 4],
                  [2, 5]]
```

Detailed memory layout visualization:

```
Tensor x (2x3):
[[a, b, c],
 [d, e, f]]

Row-major storage (stride=(3,1)):
┌───┬───┬───┬───┬───┬───┐
│ a │ b │ c │ d │ e │ f │
└───┴───┴───┴───┴───┴───┘
  0   1   2   3   4   5

x[0,0]=storage[0]   x[0,1]=storage[1]   x[0,2]=storage[2]
x[1,0]=storage[3]   x[1,1]=storage[4]   x[1,2]=storage[5]

After transpose x.t() (stride=(1,3)):
Shape is (3x2), but storage unchanged!
x_t[0,0]=storage[0]   x_t[0,1]=storage[3]
x_t[1,0]=storage[1]   x_t[1,1]=storage[4]
x_t[2,0]=storage[2]   x_t[2,1]=storage[5]
```

---

## How Operations Affect Strides

### Operations That Preserve Strides (Views)

These operations create new tensors sharing the same storage:

```python
x = torch.arange(24).reshape(4, 6)

# Slicing
y = x[1:3, 2:5]
print(y.stride())  # Same as x: (6, 1)
print(y.storage().data_ptr() == x.storage().data_ptr())  # True

# Reshape (when contiguous)
z = x.reshape(2, 12)
print(z.storage().data_ptr() == x.storage().data_ptr())  # True
```

### Operations That Change Strides

```python
x = torch.arange(12).reshape(3, 4)
print(f"Original: shape={x.shape}, stride={x.stride()}")
# Original: shape=torch.Size([3, 4]), stride=(4, 1)

# Transpose reverses strides
x_t = x.t()
print(f"Transposed: shape={x_t.shape}, stride={x_t.stride()}")
# Transposed: shape=torch.Size([4, 3]), stride=(1, 4)

# Permute reorders strides
y = torch.randn(2, 3, 4)
y_p = y.permute(2, 0, 1)
print(f"Permuted: stride {y.stride()} -> {y_p.stride()}")
# Permuted: stride (12, 4, 1) -> (1, 12, 4)
```

### Slicing and Storage Offset

Slicing preserves strides and adjusts the storage offset:

```python
x = torch.arange(20).reshape(4, 5)
# Storage: [0, 1, 2, ..., 19]

y = x[1:3, 2:4]  # 2x2 slice
print(f"Offset: {y.storage_offset()}")  # 7 (position of x[1,2])
print(f"Stride: {y.stride()}")          # (5, 1) - unchanged
```

### Adding Dimensions

`unsqueeze` inserts dimensions with adjusted strides:

```python
x = torch.randn(3, 4)
print(x.stride())  # (4, 1)

y = x.unsqueeze(0)  # Add batch dimension
print(y.shape)     # (1, 3, 4)
print(y.stride())  # (12, 4, 1)

y = x.unsqueeze(1)  # Add channel dimension
print(y.shape)     # (3, 1, 4)
print(y.stride())  # (4, 4, 1) - note repeated stride
```

### Broadcasting and Expand

`expand` uses stride 0 to repeat data without copying:

```python
x = torch.tensor([1, 2, 3])
print(x.stride())  # (1,)

y = x.expand(4, 3)  # Repeat to 4x3
print(y.shape)     # (4, 3)
print(y.stride())  # (0, 1) - stride 0 means "don't move in storage"

# All rows point to same data
print(y[0].data_ptr() == y[1].data_ptr())  # True
```

---

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
print(t.storage().data_ptr() == flat.storage().data_ptr())  # False - a copy was made
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

---

## Clone vs Detach

### `clone()` - Copy Data

```python
original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# clone() creates a copy with same gradient tracking
cloned = original.clone()

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

---

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

---

## Performance Implications

### Contiguous Access is Faster

Modern CPUs optimize for sequential memory access:

```python
import time

def benchmark(x, iterations=1000):
    start = time.time()
    for _ in range(iterations):
        _ = x.sum()
    return time.time() - start

x = torch.randn(1000, 1000)
x_t = x.t()

t_contig = benchmark(x)
t_noncontig = benchmark(x_t)

print(f"Contiguous: {t_contig:.4f}s")
print(f"Non-contiguous: {t_noncontig:.4f}s")
```

### When to Call `.contiguous()`

```python
# Required: operations that need contiguous memory
x_t = x.t()
x_view = x_t.contiguous().view(-1)

# Optional: performance optimization
# Only if you'll perform many operations on the tensor
x_fast = x_t.contiguous()

# Unnecessary: reshape handles non-contiguous
x_flat = x_t.reshape(-1)  # Works without .contiguous()
```

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

---

## Memory Inspection Utilities

```python
def inspect_tensor(t, name="tensor"):
    """Display comprehensive tensor memory information."""
    print(f"=== {name} ===")
    print(f"  Shape: {t.shape}")
    print(f"  Stride: {t.stride()}")
    print(f"  Contiguous: {t.is_contiguous()}")
    print(f"  Storage offset: {t.storage_offset()}")
    print(f"  Storage size: {len(t.storage())}")
    print(f"  Data pointer: {t.storage().data_ptr()}")
    print()

# Example usage
x = torch.arange(12).reshape(3, 4)
inspect_tensor(x, "Original")
inspect_tensor(x.T, "Transposed")
inspect_tensor(x[1:], "Sliced")
inspect_tensor(x[:, ::2], "Strided slice")
```

---

## Common Issues and Solutions

| Symptom | Cause | Solution |
|---------|-------|----------|
| `view()` fails | Non-contiguous tensor | Use `.contiguous().view()` or `.reshape()` |
| Unexpected data modification | Tensors share storage | Use `.clone()` for independent copy |
| Slow operations | Non-contiguous access | Call `.contiguous()` if many operations follow |

---

## Exercises

1. **Stride Prediction**: Given a 3D tensor of shape `(2, 3, 4)`, predict the strides. Then verify using PyTorch.

2. **View vs Reshape**: Create scenarios where `view()` fails but `reshape()` succeeds. Explain why.

3. **Memory Detective**: Create two tensors that appear different but share the same storage. Modify one and observe the effect on the other.

4. **Performance Comparison**: Benchmark the same operation on contiguous vs non-contiguous tensors of various sizes.

---

## Summary

| Concept | Description |
|---------|-------------|
| Storage | Flat 1D array holding actual tensor data |
| Stride | Steps to skip in storage for each dimension |
| Contiguous | Elements in memory match row-major iteration order |
| View | Shares memory with original tensor |
| Copy | Independent memory allocation |
| `view()` | Reshape contiguous tensors only (fast, no copy) |
| `reshape()` | Reshape any tensor (may copy if needed) |
| `clone()` | Create independent copy with gradient tracking |
| `detach()` | Remove from computation graph (shares memory) |
| `contiguous()` | Make tensor contiguous (copies if needed) |

The stride mechanism is PyTorch's way of mapping multi-dimensional tensor indices to linear memory positions. Understanding strides explains why some operations create views (sharing memory) while others require copies. Key takeaways:

- Strides determine how many storage elements to skip per dimension
- Row-major order means row stride > column stride
- Transpose and permute change strides without copying data
- Contiguous tensors have strides matching row-major expectations
- Non-contiguous tensors may need `.contiguous()` for certain operations

---

## See Also

- [Dtype and Device](dtype_device.md) — Data type and device attributes
- [Reshaping and Views](reshaping_view.md) — Detailed reshaping operations
- [Broadcasting Rules](broadcasting_rules.md) — Implicit tensor expansion
- [Shape Manipulation](shape_manipulation.md) — Indexing, concatenation, and splitting
- [Memory Management](memory_management.md) — Views, copies, and GPU memory
