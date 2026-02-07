# Indexing and Slicing

## Overview

PyTorch tensor indexing follows NumPy conventions with extensions for GPU tensors and autograd. Understanding indexing is critical for efficient data manipulation—proper indexing avoids unnecessary copies and enables vectorized operations on subsets of data.

---

## Basic Indexing

Basic indexing uses integers and slices and always returns a **view** (shared memory, no copy).

### Integer Indexing

```python
import torch

t = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Single element
print(t[0, 1])       # tensor(2)
print(t[2, 2])       # tensor(9)

# Entire row
print(t[0])           # tensor([1, 2, 3])

# Negative indexing (from the end)
print(t[-1])          # tensor([7, 8, 9])
print(t[-1, -1])      # tensor(9)
```

### Slice Indexing

Slices use `start:stop:step` syntax, where `stop` is exclusive:

```python
t = torch.arange(10)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print(t[2:5])         # tensor([2, 3, 4])
print(t[:3])          # tensor([0, 1, 2])       — first 3
print(t[7:])          # tensor([7, 8, 9])       — from index 7
print(t[::2])         # tensor([0, 2, 4, 6, 8]) — every other element
print(t[::-1])        # tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) — reversed
```

### Multi-dimensional Slicing

```python
A = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]])

# Submatrix
print(A[0:2, 1:3])
# tensor([[1, 2],
#          [5, 6]])

# Entire column
print(A[:, 2])         # tensor([ 2,  6, 10])

# Entire row
print(A[1, :])         # tensor([4, 5, 6, 7])

# Every other row, every other column
print(A[::2, ::2])
# tensor([[ 0,  2],
#          [ 8, 10]])
```

### Views vs Copies

Basic indexing returns a **view** — modifying the result modifies the original:

```python
A = torch.arange(6).reshape(2, 3)
B = A[0]       # B is a view of the first row of A

B[0] = 99
print(A)
# tensor([[99,  1,  2],
#          [ 3,  4,  5]])
```

To get an independent copy, use `.clone()`:

```python
B = A[0].clone()  # Independent copy
B[0] = -1
print(A[0, 0])    # Still 99 — A is unaffected
```

---

## Advanced Indexing

Advanced indexing uses tensors or lists as indices and always returns a **copy** (new memory).

### Integer Tensor Indexing

```python
t = torch.tensor([10, 20, 30, 40, 50])

# Index with a tensor of indices
idx = torch.tensor([0, 2, 4])
print(t[idx])         # tensor([10, 30, 50])

# Index with a Python list
print(t[[0, 2, 4]])   # tensor([10, 30, 50])

# Duplicate indices are allowed
print(t[[0, 0, 1, 1]])  # tensor([10, 10, 20, 20])
```

### Multi-dimensional Advanced Indexing

```python
A = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]])

# Select specific elements: A[0,1], A[1,2], A[2,3]
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([1, 2, 3])
print(A[rows, cols])   # tensor([ 1,  6, 11])

# Select specific rows
row_idx = torch.tensor([0, 2])
print(A[row_idx])
# tensor([[ 0,  1,  2,  3],
#          [ 8,  9, 10, 11]])

# Select specific columns
col_idx = torch.tensor([1, 3])
print(A[:, col_idx])
# tensor([[ 1,  3],
#          [ 5,  7],
#          [ 9, 11]])
```

---

## Boolean Masking

Boolean tensors can be used as indices to select elements where the condition is `True`:

```python
t = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])

# Mask for positive values
mask = t > 0
print(mask)          # tensor([ True, False,  True, False,  True])

# Select elements (returns 1D tensor — always a copy)
print(t[mask])       # tensor([1., 3., 5.])

# Assign to masked elements
t[mask] = 0
print(t)             # tensor([ 0., -2.,  0., -4.,  0.])
```

### 2D Masking

```python
A = torch.randn(3, 3)

# Mask based on condition
mask = A > 0
print(A[mask])           # 1D tensor of all positive elements

# Set negative values to zero (ReLU-like)
A[A < 0] = 0
print(A)
```

### Combining Masks

```python
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Logical AND
mask = (x > 3) & (x < 8)
print(x[mask])           # tensor([4, 5, 6, 7])

# Logical OR
mask = (x < 3) | (x > 8)
print(x[mask])           # tensor([ 1,  2,  9, 10])

# Logical NOT
mask = ~(x == 5)
print(x[mask])           # tensor([ 1,  2,  3,  4,  6,  7,  8,  9, 10])
```

!!! warning "Use `&`, `|`, `~` — Not `and`, `or`, `not`"
    Python's `and`, `or`, `not` operators do not work element-wise on tensors.
    Always use the bitwise operators `&`, `|`, `~` with parentheses around each condition.

---

## Ellipsis (`...`)

The ellipsis `...` represents "all remaining dimensions":

```python
t = torch.randn(2, 3, 4, 5)

# These are equivalent
print(t[0, ...].shape)          # torch.Size([3, 4, 5])
print(t[0, :, :, :].shape)     # torch.Size([3, 4, 5])

# Select last dimension
print(t[..., 0].shape)         # torch.Size([2, 3, 4])
print(t[:, :, :, 0].shape)     # torch.Size([2, 3, 4])

# Select first and last
print(t[0, ..., -1].shape)     # torch.Size([3, 4])
```

---

## `None` for Dimension Insertion

Using `None` in an index is equivalent to `unsqueeze`:

```python
x = torch.tensor([1, 2, 3])    # Shape: (3,)

print(x[None, :].shape)        # torch.Size([1, 3])  — row vector
print(x[:, None].shape)        # torch.Size([3, 1])  — column vector
print(x[None, :, None].shape)  # torch.Size([1, 3, 1])
```

This is commonly used for broadcasting:

```python
# Pairwise distance matrix using None indexing
a = torch.tensor([1.0, 3.0, 5.0])
b = torch.tensor([2.0, 4.0])

# a[:, None] has shape (3, 1), b[None, :] has shape (1, 2)
diff = a[:, None] - b[None, :]  # Shape: (3, 2)
print(diff)
# tensor([[-1., -3.],
#          [ 1., -1.],
#          [ 3.,  1.]])
```

---

## `torch.gather` and `torch.scatter_`

### `gather` — Indexing Along a Dimension

`gather(dim, index)` collects elements from a specific dimension according to `index`:

```python
# Scores for 3 samples across 4 classes
scores = torch.tensor([[0.1, 0.9, 0.3, 0.5],
                        [0.8, 0.2, 0.7, 0.1],
                        [0.3, 0.4, 0.6, 0.9]])

# Predicted class indices
labels = torch.tensor([[1], [0], [3]])  # Shape: (3, 1)

# Gather scores of predicted classes
selected = scores.gather(dim=1, index=labels)
print(selected)
# tensor([[0.9000],
#          [0.8000],
#          [0.9000]])
```

### `scatter_` — Writing to Specific Positions

`scatter_(dim, index, src)` is the inverse of `gather`:

```python
# Create one-hot encoding
labels = torch.tensor([2, 0, 1])
one_hot = torch.zeros(3, 3)
one_hot.scatter_(dim=1, index=labels.unsqueeze(1), value=1.0)
print(one_hot)
# tensor([[0., 0., 1.],
#          [1., 0., 0.],
#          [0., 1., 0.]])
```

---

## `torch.index_select` and `torch.take`

```python
A = torch.arange(12).reshape(3, 4)

# Select specific indices along a dimension
idx = torch.tensor([0, 2])
print(torch.index_select(A, dim=0, index=idx))  # Rows 0 and 2
print(torch.index_select(A, dim=1, index=idx))  # Columns 0 and 2

# take: flat index into any-shaped tensor
print(torch.take(A, torch.tensor([0, 5, 11])))  # tensor([ 0,  5, 11])
```

---

## `torch.nonzero`

Returns indices of non-zero elements:

```python
t = torch.tensor([0, 1, 0, 3, 0, 5])

# Indices of non-zero elements
print(torch.nonzero(t))
# tensor([[1],
#          [3],
#          [5]])

# As a tuple (similar to NumPy's behavior)
print(torch.nonzero(t, as_tuple=True))
# (tensor([1, 3, 5]),)

# Useful for finding positions matching a condition
A = torch.tensor([[1, 0], [0, 2]])
rows, cols = torch.nonzero(A, as_tuple=True)
print(f"Non-zero at rows={rows}, cols={cols}")
# Non-zero at rows=tensor([0, 1]), cols=tensor([0, 1])
```

---

## Indexing Assignment

Values can be assigned using any indexing method:

```python
t = torch.zeros(5)

# Integer index
t[0] = 1.0

# Slice assignment
t[1:3] = torch.tensor([2.0, 3.0])

# Boolean mask assignment
mask = t == 0
t[mask] = -1.0

print(t)  # tensor([ 1.,  2.,  3., -1., -1.])

# Advanced index assignment
A = torch.zeros(3, 3)
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([2, 1, 0])
A[rows, cols] = torch.tensor([1.0, 2.0, 3.0])
print(A)
# tensor([[0., 0., 1.],
#          [0., 2., 0.],
#          [3., 0., 0.]])
```

---

## Common Patterns

### Selecting Diagonal Elements

```python
A = torch.arange(9).reshape(3, 3)
diag = A[torch.arange(3), torch.arange(3)]
print(diag)  # tensor([0, 4, 8])
```

### Top-k Selection

```python
scores = torch.tensor([0.3, 0.9, 0.1, 0.7, 0.5])

values, indices = torch.topk(scores, k=3)
print(f"Top-3 values: {values}")     # tensor([0.9000, 0.7000, 0.5000])
print(f"Top-3 indices: {indices}")   # tensor([1, 3, 4])
```

### Batch-wise Indexing

```python
# Select one element per sample in a batch
batch_scores = torch.randn(4, 10)    # 4 samples, 10 classes
labels = torch.tensor([3, 7, 1, 5])  # Target class per sample

# Gather the score of the correct class
selected = batch_scores[torch.arange(4), labels]
print(selected.shape)                 # torch.Size([4])
```

---

## Summary

| Method | Returns | Memory |
|--------|---------|--------|
| Integer / slice indexing | View | Shared |
| Advanced (tensor) indexing | Copy | New |
| Boolean masking | Copy (1D) | New |
| `gather` / `scatter_` | Copy / In-place | New / In-place |
| `index_select` | Copy | New |
| `.clone()` | Copy | New |

## See Also

- [Tensor Basics](tensor_basics.md) — Core tensor concepts
- [Tensor Operations](tensor_operations.md) — Reshaping, broadcasting, conditional operations
- [NumPy Interoperability](numpy_interop.md) — Indexing consistency with NumPy
