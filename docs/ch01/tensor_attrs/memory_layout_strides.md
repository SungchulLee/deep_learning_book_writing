# Memory Layout and Strides

## Learning Objectives

By the end of this section, you will be able to:

- Understand how tensors are stored in memory using strides
- Explain the difference between row-major and column-major ordering
- Predict whether operations create views or copies
- Optimize tensor operations by leveraging memory layout knowledge

---

## Overview

PyTorch tensors are multi-dimensional arrays stored in contiguous blocks of memory. The **stride** mechanism determines how multi-dimensional indices map to positions in this linear memory block. Understanding strides is fundamental to efficient tensor manipulation and explains many behaviors of reshaping and view operations.

---

## The Storage-Stride Model

### Storage: The Raw Data

Every PyTorch tensor is backed by a **Storage** object—a flat, one-dimensional array of typed elements:

```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# View the underlying storage
print(x.storage())  # [1, 2, 3, 4, 5, 6]
print(len(x.storage()))  # 6 elements total
```

### Strides: The Access Pattern

**Strides** specify how many elements to skip in storage to move one position along each dimension:

```python
print(x.stride())  # (3, 1)
# stride[0] = 3: skip 3 elements to move to next row
# stride[1] = 1: skip 1 element to move to next column
```

The element at position $(i, j)$ is located at:

$$
\text{storage\_index} = \text{offset} + i \times \text{stride}[0] + j \times \text{stride}[1]
$$

For our example tensor `x[1, 2]`:
```python
# Manual calculation: offset(0) + 1*3 + 2*1 = 5
print(x.storage()[5])  # 6
print(x[1, 2])         # 6
```

---

## Row-Major vs Column-Major Order

### Row-Major (C-style) Order

PyTorch uses **row-major** ordering by default, where elements of the same row are stored contiguously:

```python
# Row-major: rows are contiguous in memory
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
# Storage: [1, 2, 3, 4, 5, 6]
#          |row 0 | row 1 |

print(x.stride())  # (3, 1) - row stride > column stride
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

### Contiguity Check

A tensor is **contiguous** if its memory layout matches the expected row-major pattern:

```python
x = torch.randn(3, 4)
print(x.is_contiguous())  # True

# Transpose changes strides but not storage
x_t = x.t()
print(x_t.stride())        # (1, 4) instead of (4, 1)
print(x_t.is_contiguous()) # False
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

### Operations That Require Copying

When strides cannot represent the new view, a copy is made:

```python
x = torch.arange(12).reshape(3, 4)
x_t = x.t()  # Non-contiguous

# view() requires contiguous tensor
try:
    z = x_t.view(12)
except RuntimeError as e:
    print(f"Error: {e}")

# reshape() handles non-contiguous by copying
z = x_t.reshape(12)  # Creates new storage
print(z.storage().data_ptr() == x.storage().data_ptr())  # False
```

---

## Stride Patterns for Common Operations

### Slicing

Slicing preserves strides and adjusts the storage offset:

```python
x = torch.arange(20).reshape(4, 5)
# Storage: [0, 1, 2, ..., 19]

y = x[1:3, 2:4]  # 2x2 slice
print(f"Offset: {y.storage_offset()}")  # 7 (position of x[1,2])
print(f"Stride: {y.stride()}")          # (5, 1) - unchanged
```

### Adding Dimensions

`unsqueeze` inserts dimensions with stride 0 or adjusted strides:

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

## Memory Layout Visualization

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

---

## Debugging Stride Issues

### Inspection Functions

```python
def inspect_tensor(t, name="tensor"):
    print(f"{name}:")
    print(f"  shape: {t.shape}")
    print(f"  stride: {t.stride()}")
    print(f"  contiguous: {t.is_contiguous()}")
    print(f"  storage offset: {t.storage_offset()}")
    print(f"  storage size: {len(t.storage())}")
    print()

x = torch.arange(12).reshape(3, 4)
inspect_tensor(x, "original")
inspect_tensor(x.t(), "transposed")
inspect_tensor(x[:2, 1:3], "sliced")
```

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| `view()` fails | Non-contiguous tensor | Use `.contiguous().view()` or `.reshape()` |
| Unexpected data modification | Tensors share storage | Use `.clone()` for independent copy |
| Slow operations | Non-contiguous access | Call `.contiguous()` if many operations follow |

---

## Connection to Other Topics

- **[Reshaping and View Operations](reshaping_view.md)**: How view and reshape use strides
- **[Memory Management](memory_management.md)**: Efficient memory usage patterns
- **[NumPy Interoperability](../tensors/numpy_interop.md)**: Stride compatibility with NumPy

---

## Exercises

1. **Stride Prediction**: Given a 3D tensor of shape `(2, 3, 4)`, predict the strides. Then verify using PyTorch.

2. **View vs Reshape**: Create scenarios where `view()` fails but `reshape()` succeeds. Explain why.

3. **Memory Detective**: Create two tensors that appear different but share the same storage. Modify one and observe the effect on the other.

4. **Performance Comparison**: Benchmark the same operation on contiguous vs non-contiguous tensors of various sizes.

---

## Summary

The stride mechanism is PyTorch's way of mapping multi-dimensional tensor indices to linear memory positions. Understanding strides explains why some operations create views (sharing memory) while others require copies. Key takeaways:

- Strides determine how many storage elements to skip per dimension
- Row-major order means row stride > column stride
- Transpose and permute change strides without copying data
- Contiguous tensors have strides matching row-major expectations
- Non-contiguous tensors may need `.contiguous()` for certain operations
