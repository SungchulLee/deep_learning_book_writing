# Indexing and Slicing

PyTorch provides powerful indexing and slicing capabilities that closely mirror NumPy but with additional features optimized for deep learning workflows. Mastering these operations is essential for efficient data manipulation, memory management, and building performant neural networks.

## Memory Model Foundation

Before diving into indexing operations, understanding PyTorch's memory model is crucial for writing efficient code.

### Tensors, Storage, and Strides

Every PyTorch tensor consists of three components:

```python
import torch

t = torch.arange(12).reshape(3, 4)

# 1. Storage: contiguous 1D array holding actual data
print(t.storage())  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 2. Shape: logical dimensions
print(t.shape)  # torch.Size([3, 4])

# 3. Strides: steps to move one element in each dimension
print(t.stride())  # (4, 1) - move 4 elements for next row, 1 for next column
```

**Why strides matter**: Strides enable views without copying data. A transposed tensor shares the same storage but has reversed strides:

```python
t_transposed = t.T
print(t_transposed.stride())  # (1, 4) - reversed!
print(t.storage().data_ptr() == t_transposed.storage().data_ptr())  # True
```

### Storage Offset

The storage offset indicates where the tensor's data begins in storage:

```python
original = torch.arange(10)
sliced = original[3:7]

print(sliced.storage_offset())  # 3 - data starts at position 3 in storage
print(sliced.storage())  # Still the full [0, 1, ..., 9]
```

This explains why modifying a slice affects the original—they share the same underlying storage.

## Basic Indexing

### Single Element Access

```python
# 1D tensor
vec = torch.tensor([10, 20, 30, 40, 50])

print(vec[0])    # tensor(10) - first element
print(vec[2])    # tensor(30) - third element
print(vec[-1])   # tensor(50) - last element
print(vec[-2])   # tensor(40) - second to last

# Returns a 0-dimensional tensor, not a Python scalar
print(type(vec[0]))      # <class 'torch.Tensor'>
print(vec[0].dim())      # 0
print(vec[0].item())     # 10 - extract Python scalar
```

### Multi-Dimensional Indexing

```python
# 2D tensor (matrix)
mat = torch.arange(12).reshape(3, 4)
print(f"Matrix:\n{mat}")
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# Access single element - returns 0-dim tensor
print(mat[0, 0])     # tensor(0) - top-left
print(mat[1, 2])     # tensor(6) - row 1, col 2
print(mat[-1, -1])   # tensor(11) - bottom-right

# Access entire row - returns 1D tensor (view)
print(mat[0])        # tensor([0, 1, 2, 3])

# Access entire column - requires slicing
print(mat[:, 0])     # tensor([0, 4, 8])
```

### 3D and Higher Dimensions

```python
# 3D tensor: interpret as (batch, height, width) or (channels, rows, cols)
cube = torch.arange(24).reshape(2, 3, 4)
print(f"Shape: {cube.shape}")  # torch.Size([2, 3, 4])

# Single element
print(cube[0, 1, 2])  # Element at position (0, 1, 2)

# Entire 2D slice (batch/channel selection)
print(cube[0])        # First 2D slice, shape [3, 4]
print(cube[0, 1])     # First slice, second row, shape [4]

# Cross-sectional slices
print(cube[:, 0, :])  # All first rows from each slice, shape [2, 4]
print(cube[:, :, 0])  # All first columns from each slice, shape [2, 3]
```

## Slicing Operations

### Basic Slicing Syntax

The slice notation `start:stop:step` follows Python conventions. **Basic slicing always creates views**.

```python
vec = torch.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Basic slices
print(vec[2:5])      # tensor([2, 3, 4]) - indices 2, 3, 4
print(vec[:3])       # tensor([0, 1, 2]) - first 3 elements
print(vec[7:])       # tensor([7, 8, 9]) - from index 7 to end
print(vec[::2])      # tensor([0, 2, 4, 6, 8]) - every other element
print(vec[1::2])     # tensor([1, 3, 5, 7, 9]) - odd indices

# Negative indices
print(vec[-3:])      # tensor([7, 8, 9]) - last 3 elements
print(vec[:-2])      # tensor([0, 1, 2, 3, 4, 5, 6, 7]) - all but last 2
print(vec[::-1])     # tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) - reversed
```

### 2D Slicing

```python
mat = torch.arange(20).reshape(4, 5)
print(f"Matrix:\n{mat}")
# tensor([[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 19]])

# Row slices
print(mat[1:3])      # Rows 1 and 2 (shape [2, 5])

# Column slices
print(mat[:, 1:4])   # Columns 1-3 (shape [4, 3])

# Combined slicing - extract submatrix
print(mat[1:3, 2:4]) # Rows 1-2, columns 2-3 (shape [2, 2])

# Strided access
print(mat[::2])      # Rows 0 and 2 (shape [2, 5])
print(mat[:, ::2])   # Columns 0, 2, 4 (shape [4, 3])
```

### View Semantics: The Critical Distinction

Basic slicing creates **views** that share memory with the original tensor:

```python
original = torch.arange(10)
sliced = original[2:5]

# Verify shared storage
print(original.storage().data_ptr() == sliced.storage().data_ptr())  # True

# Modifying slice affects original!
sliced[0] = 99
print(original)  # tensor([ 0,  1, 99,  3,  4,  5,  6,  7,  8,  9])

# 2D example
a = torch.arange(12).reshape(3, 4)
view = a[1:3, 2:4]
view[0, 0] = 999
print(a[1, 2])  # 999
```

**When you need a copy**, use `.clone()`:

```python
original = torch.arange(10)
copied = original[2:5].clone()  # Independent copy
copied[0] = 999
print(original[2])  # Still 2
```

## Advanced Indexing

Advanced indexing (boolean masks and integer arrays) creates **copies**, not views.

### Boolean Indexing (Masking)

```python
t = torch.tensor([1, -2, 3, -4, 5, -6])

# Create boolean mask
positive_mask = t > 0
print(f"Mask: {positive_mask}")  # tensor([True, False, True, False, True, False])

# Select elements where mask is True (returns COPY, always 1D)
positives = t[positive_mask]
print(f"Positive values: {positives}")  # tensor([1, 3, 5])

# Verify it's a copy
positives[0] = 100
print(t[0])  # Still 1
```

### Boolean Assignment (In-Place)

```python
t = torch.tensor([1, -2, 3, -4, 5, -6])

# Modify elements based on condition
t[t < 0] = 0
print(f"After zeroing negatives: {t}")  # tensor([1, 0, 3, 0, 5, 0])
```

### Compound Boolean Conditions

Use `&` (and), `|` (or), `~` (not). **Parentheses are required** due to operator precedence:

```python
x = torch.randn(10)

# WRONG - will raise error
# mask = x > -0.5 & x < 0.5

# CORRECT - parentheses required!
in_range = (x > -0.5) & (x < 0.5)
out_of_range = (x <= -0.5) | (x >= 0.5)
negative = ~(x > 0)

# Apply combined mask
x[in_range] = 0  # Zero out values in range
```

### Integer Array Indexing (Fancy Indexing)

```python
t = torch.tensor([10, 20, 30, 40, 50])
indices = torch.tensor([0, 2, 4])

# Select specific indices (returns COPY)
selected = t[indices]
print(f"Selected: {selected}")  # tensor([10, 30, 50])

# Can have repeated indices
repeated_indices = torch.tensor([0, 0, 1, 1, 2])
print(t[repeated_indices])  # tensor([10, 10, 20, 20, 30])
```

### Multi-Dimensional Fancy Indexing

```python
mat = torch.arange(12).reshape(3, 4)

# Coordinate-based selection: selects (0,1), (2,3), (1,2)
row_idx = torch.tensor([0, 2, 1])
col_idx = torch.tensor([1, 3, 2])
print(mat[row_idx, col_idx])  # tensor([1, 11, 6])

# Select specific rows
row_indices = torch.tensor([0, 2])
selected_rows = mat[row_indices]  # Shape: (2, 4)

# Select specific columns
col_indices = torch.tensor([1, 3])
selected_cols = mat[:, col_indices]  # Shape: (3, 2)
```

### Broadcasting in Fancy Indexing

```python
x = torch.arange(12).reshape(3, 4)

# Create a subgrid via broadcasting
row_idx = torch.tensor([[0], [1], [2]])  # Shape: (3, 1)
col_idx = torch.tensor([[0, 2]])          # Shape: (1, 2)

# Broadcasts to select 3×2 subgrid
subgrid = x[row_idx, col_idx]
print(subgrid.shape)  # torch.Size([3, 2])
print(subgrid)
# tensor([[ 0,  2],
#         [ 4,  6],
#         [ 8, 10]])
```

### Combining Index Types

```python
mat = torch.arange(20).reshape(4, 5)

# Integer index + slice
print(mat[1, :3])        # Row 1, first 3 columns

# Boolean + integer
mask = torch.tensor([True, False, True, False])
print(mat[mask, 2])      # Rows 0 and 2, column 2

# Multiple integer arrays
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([1, 2, 3])
print(mat[rows, cols])   # Elements (0,1), (1,2), (2,3)
```

## Special Indexing Operations

### Ellipsis (`...`)

The ellipsis represents "all remaining dimensions"—essential for dimension-agnostic code:

```python
# 4D tensor (e.g., batch × channels × height × width)
t = torch.randn(2, 3, 4, 5)

# These are equivalent
print(t[0, :, :, 0].shape)   # torch.Size([3, 4])
print(t[0, ..., 0].shape)    # torch.Size([3, 4])

# All but first and last dimensions
print(t[0, ..., -1].shape)   # torch.Size([3, 4])

# Dimension-agnostic function
def get_last_channel(x):
    """Works for any number of leading dimensions."""
    return x[..., -1]

# Works on 2D, 3D, 4D tensors
print(get_last_channel(torch.randn(3, 4)).shape)      # (3,)
print(get_last_channel(torch.randn(2, 3, 4)).shape)   # (2, 3)
print(get_last_channel(torch.randn(2, 3, 4, 5)).shape)  # (2, 3, 4)
```

### None (newaxis)

Use `None` to insert dimensions of size 1:

```python
vec = torch.tensor([1, 2, 3, 4, 5])
print(f"Original shape: {vec.shape}")  # torch.Size([5])

# Add dimension at front (row vector)
row = vec[None, :]
print(f"Row vector: {row.shape}")  # torch.Size([1, 5])

# Add dimension at end (column vector)
col = vec[:, None]
print(f"Column vector: {col.shape}")  # torch.Size([5, 1])

# Add multiple dimensions
expanded = vec[None, :, None]
print(f"Expanded: {expanded.shape}")  # torch.Size([1, 5, 1])

# Equivalent to unsqueeze
print(vec.unsqueeze(0).shape)  # torch.Size([1, 5])
print(vec.unsqueeze(1).shape)  # torch.Size([5, 1])
```

### `torch.where()` for Conditional Selection

Select elements from two tensors based on a condition:

```python
condition = torch.tensor([True, False, True, False])
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([10, 20, 30, 40])

# Select from a where True, from b where False
result = torch.where(condition, a, b)
print(result)  # tensor([1, 20, 3, 40])

# Common patterns
x = torch.randn(5)

# ReLU-like operation
relu = torch.where(x > 0, x, torch.zeros_like(x))

# Replace NaNs
data = torch.tensor([1., float('nan'), 3.])
cleaned = torch.where(torch.isnan(data), torch.tensor(0.0), data)

# Clamp to range [-1, 1]
clamped = torch.where(x < -1, torch.tensor(-1.0), 
                      torch.where(x > 1, torch.tensor(1.0), x))
```

### `torch.where()` Index Form

Returns tuple of indices where condition is True:

```python
x = torch.tensor([[0, 1, 0], [2, 0, 3], [0, 4, 0]])

row_idx, col_idx = torch.where(x > 0)
print(row_idx)  # tensor([0, 1, 1, 2])
print(col_idx)  # tensor([1, 0, 2, 1])

# Use indices to get values
values = x[row_idx, col_idx]  # tensor([1, 2, 3, 4])
```

### `torch.nonzero()`

Returns indices of non-zero elements:

```python
x = torch.tensor([[0, 1, 0], [2, 0, 3], [0, 4, 0]])

indices = x.nonzero()
# tensor([[0, 1],
#         [1, 0],
#         [1, 2],
#         [2, 1]])

# Each row is the coordinate of a non-zero element
```

## Masked Operations

### `masked_fill`: Fill Values Based on Mask

```python
x = torch.randn(3, 4)
mask = x > 0

# Fill positive values with -999 (returns new tensor)
filled = x.masked_fill(mask, value=-999)

# In-place version
x.masked_fill_(x.abs() < 0.5, value=0)
```

**Critical use case**: Attention masking

```python
# Create causal mask for transformer attention
scores = torch.randn(4, 4)
mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))
# Upper triangle becomes -inf, softmax will zero these
```

### `masked_select`: Extract Matching Values

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask = x > 4

selected = x.masked_select(mask)  # tensor([5, 6, 7, 8, 9])
# Returns 1D tensor with all matching elements
```

### `masked_scatter`: Scatter Values Based on Mask

```python
x = torch.zeros(3, 4)
mask = torch.tensor([[True, False, True, False],
                     [False, True, False, True],
                     [True, True, False, False]])
source = torch.arange(1, 7).float()

result = x.masked_scatter(mask, source)
# tensor([[1, 0, 2, 0],
#         [0, 3, 0, 4],
#         [5, 6, 0, 0]])
```

## Gather and Scatter Operations

These operations are fundamental for advanced indexing patterns in deep learning.

### `gather`: Indexed Selection Along a Dimension

```python
# Select elements along a dimension using indices
x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

# Pick specific column indices per row (dim=1)
idx = torch.tensor([[0, 2], [1, 3], [0, 0]])

gathered = x.gather(dim=1, index=idx)
# tensor([[1, 3],
#         [6, 8],
#         [9, 9]])
# Row 0: columns 0, 2 → [1, 3]
# Row 1: columns 1, 3 → [6, 8]
# Row 2: columns 0, 0 → [9, 9]
```

**Understanding gather**: For `dim=1`, each index selects a column within its row. The output shape matches the index shape.

### `scatter_`: Indexed Assignment

```python
# Scatter values to positions specified by indices
target = torch.zeros(3, 4)
idx = torch.tensor([[0, 2], [1, 3], [0, 0]])
source = torch.tensor([[9., 9.], [8., 8.], [7., 7.]])

target.scatter_(dim=1, index=idx, src=source)
# tensor([[9, 0, 9, 0],
#         [0, 8, 0, 8],
#         [7, 0, 0, 0]])
# Note: position (2,0) gets 7 (last write wins for duplicates)
```

### `scatter_add_`: Accumulate at Indices

```python
# Sum values at the same index instead of overwriting
target = torch.zeros(3, 4)
idx = torch.tensor([[0, 0], [1, 1], [2, 2]])
source = torch.ones(3, 2)

target.scatter_add_(dim=1, index=idx, src=source)
# tensor([[2, 0, 0, 0],   # position (0,0) got 1+1=2
#         [0, 2, 0, 0],   # position (1,1) got 1+1=2
#         [0, 0, 2, 0]])  # position (2,2) got 1+1=2
```

## In-Place Indexing Assignment

### Basic Assignment

```python
t = torch.zeros(5)

# Single element
t[2] = 7
print(t)  # tensor([0., 0., 7., 0., 0.])

# Slice assignment
t[1:4] = 1
print(t)  # tensor([0., 1., 1., 1., 0.])

# Broadcast assignment
mat = torch.zeros(3, 4)
mat[0] = torch.tensor([1, 2, 3, 4])  # Entire row
mat[:, -1] = 9  # Entire column
print(mat)
```

## View vs Copy: Complete Reference

| Operation | Creates | Memory | Notes |
|-----------|---------|--------|-------|
| Basic slice `t[a:b]` | View | Shared | Fast, no allocation |
| Step slice `t[::2]` | View | Shared | Non-contiguous |
| Single index `t[i]` | View | Shared | 0-dim tensor |
| Boolean mask `t[mask]` | Copy | New | Always 1D output |
| Integer array `t[indices]` | Copy | New | Allows repeats |
| `None` indexing `t[None]` | View | Shared | Adds dimension |
| `.contiguous()` | Maybe | Maybe | Copy only if needed |
| `.clone()` | Copy | New | Explicit copy |

```python
t = torch.arange(10)

# View operations
slice_view = t[2:5]
print(slice_view.storage().data_ptr() == t.storage().data_ptr())  # True

# Copy operations
idx = torch.tensor([2, 3, 4])
idx_copy = t[idx]
print(idx_copy.storage().data_ptr() == t.storage().data_ptr())  # False

mask_copy = t[t > 5]
print(mask_copy.storage().data_ptr() == t.storage().data_ptr())  # False
```

## Performance Considerations

### Contiguity and Memory Layout

```python
mat = torch.randn(1000, 1000)

# Row access (contiguous in C-order)
row = mat[500]  # Fast: sequential memory access

# Column access (strided)
col = mat[:, 500]  # Slower: skips through memory
print(col.is_contiguous())  # False

# Make contiguous for repeated operations
col_contiguous = col.contiguous()
print(col_contiguous.is_contiguous())  # True
```

### Avoiding Unnecessary Copies

```python
# BAD: Creates multiple intermediate copies
result = data[mask][indices]

# BETTER: Combine operations when possible
combined_mask = mask.clone()
combined_mask[~mask] = False  # Modify in-place
result = data[combined_mask]

# BEST: Use boolean operations
final_mask = mask & (indices_condition)
result = data[final_mask]
```

### GPU Considerations

```python
# Boolean indexing on GPU can be slow due to irregular access
# Consider using torch.where for better GPU utilization

# Slow on GPU
gpu_tensor = torch.randn(10000, device='cuda')
mask = gpu_tensor > 0
selected = gpu_tensor[mask]  # Irregular memory access

# Often faster: use where to create regular tensor
selected_padded = torch.where(mask, gpu_tensor, torch.tensor(0., device='cuda'))
```

## Practical Patterns

### Batch Element Selection

```python
# Select different indices per batch item
batch = torch.randn(4, 10)  # 4 samples, 10 features
indices = torch.tensor([2, 5, 7, 1])  # One index per sample

# Advanced indexing with arange
batch_idx = torch.arange(4)
selected = batch[batch_idx, indices]
print(f"Selected shape: {selected.shape}")  # torch.Size([4])
```

### Selecting Class Predictions

```python
def select_predictions(logits, targets):
    """Select specific logits using per-sample indices."""
    batch_size = logits.size(0)
    batch_indices = torch.arange(batch_size, device=logits.device)
    return logits[batch_indices, targets]

# Usage: cross-entropy loss component
logits = torch.randn(32, 10)  # Batch of 32, 10 classes
targets = torch.randint(0, 10, (32,))
selected = select_predictions(logits, targets)  # Shape: (32,)
```

### Image Patch Extraction

```python
# Extract patches from images
images = torch.randn(32, 3, 224, 224)  # Batch of images (B, C, H, W)

# Extract top-left 64×64 patch from all images
patches = images[:, :, :64, :64]
print(f"Patches shape: {patches.shape}")  # torch.Size([32, 3, 64, 64])

# Extract center crop
h, w = 224, 224
crop_h, crop_w = 128, 128
h_start = (h - crop_h) // 2
w_start = (w - crop_w) // 2
center_crop = images[:, :, h_start:h_start+crop_h, w_start:w_start+crop_w]
print(f"Center crop shape: {center_crop.shape}")  # torch.Size([32, 3, 128, 128])
```

### Sequence Masking

```python
# Mask padded positions in sequences
sequences = torch.randn(16, 50, 256)  # batch, seq_len, features
lengths = torch.randint(20, 51, (16,))  # Actual lengths per sequence

# Create mask: True for valid positions
mask = torch.arange(50).expand(16, 50) < lengths.unsqueeze(1)
print(f"Mask shape: {mask.shape}")  # torch.Size([16, 50])

# Zero out padded positions
sequences[~mask] = 0

# Or use masked_fill for attention scores
attention_scores = torch.randn(16, 50, 50)
attention_scores.masked_fill_(~mask.unsqueeze(1), float('-inf'))
```

### Attention Masking

```python
def apply_causal_mask(scores):
    """Apply causal (look-ahead) mask to attention scores."""
    seq_len = scores.size(-1)
    # Upper triangular mask (positions that should be masked)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
    return scores.masked_fill(mask, float('-inf'))

# Usage
attention_scores = torch.randn(8, 12, 64, 64)  # (batch, heads, seq, seq)
masked_scores = apply_causal_mask(attention_scores)
```

### Data Filtering by Class

```python
def filter_by_label(data, labels, target_class):
    """Select samples belonging to a specific class."""
    mask = labels == target_class
    return data[mask], labels[mask]

# Usage
data = torch.randn(100, 5)  # 100 samples, 5 features
labels = torch.randint(0, 3, (100,))  # 3 classes

class_1_data, class_1_labels = filter_by_label(data, labels, 1)
```

### Top-k Selection

```python
def select_topk_per_row(scores, k):
    """Select top-k indices per row."""
    values, indices = scores.topk(k, dim=-1)
    return values, indices

# Usage
scores = torch.randn(10, 100)  # 10 samples, 100 classes
top5_values, top5_indices = select_topk_per_row(scores, 5)
print(top5_indices.shape)  # (10, 5)
```

## Common Pitfalls and Solutions

### Pitfall 1: Unintended View Modification

```python
# Problem
original = torch.arange(10)
subset = original[2:5]
subset *= 2  # Modifies original!

# Solution: Clone when you need independence
subset = original[2:5].clone()
subset *= 2  # original unchanged
```

### Pitfall 2: Boolean Operator Precedence

```python
x = torch.randn(10)

# WRONG - bitwise & has higher precedence than comparison
# mask = x > 0 & x < 1  # Error!

# CORRECT - use parentheses
mask = (x > 0) & (x < 1)
```

### Pitfall 3: Shape Mismatch in Assignment

```python
mat = torch.zeros(3, 4)

# WRONG - shape mismatch
# mat[0] = torch.tensor([1, 2, 3])  # Error: expected 4 elements

# CORRECT
mat[0] = torch.tensor([1, 2, 3, 4])
```

### Pitfall 4: Expecting Views from Advanced Indexing

```python
t = torch.arange(10)
indices = torch.tensor([1, 3, 5])

# This is a COPY, not a view
selected = t[indices]
selected[0] = 999
print(t[1])  # Still 1, not 999

# Use scatter_ for indexed in-place updates
t.scatter_(0, indices, torch.tensor([100, 300, 500]))
```

## Summary

| Operation | Syntax | Returns | Memory |
|-----------|--------|---------|--------|
| Single element | `t[i]` or `t[i, j]` | 0-dim tensor | View |
| Row | `t[i]` | Tensor | View |
| Column | `t[:, j]` | Tensor | View |
| Slice | `t[a:b]` | Tensor | View |
| Boolean mask | `t[mask]` | 1D tensor | Copy |
| Integer indices | `t[idx]` | Tensor | Copy |
| Ellipsis | `t[..., i]` | Tensor | View |
| New axis | `t[None]` | Tensor | View |
| Where (3-arg) | `torch.where(c, a, b)` | Tensor | Copy |
| Where (1-arg) | `torch.where(c)` | Tuple of indices | Copy |
| Gather | `t.gather(dim, idx)` | Tensor | Copy |
| Scatter | `t.scatter_(dim, idx, src)` | In-place | — |

## Key Takeaways

1. **Basic slicing creates views**—modifications affect the original tensor
2. **Boolean and fancy indexing create copies**—safe to modify independently
3. **Always use parentheses** with compound boolean conditions (`&`, `|`, `~`)
4. **`torch.where`** is powerful for conditional operations and often GPU-friendly
5. **Ellipsis (`...`)** enables dimension-agnostic code
6. **`None` indexing** adds dimensions (equivalent to `unsqueeze`)
7. **`gather`/`scatter_`** are essential for advanced index-based operations
8. **Always verify** whether you have a view or copy when debugging memory issues
9. **Contiguity matters** for performance, especially with column access
10. **Use `.clone()`** explicitly when you need an independent copy

## See Also

- [Shape and Dimensions](shape_dimensions.md) - Understanding tensor structure
- [Reshaping and View Operations](reshaping_view.md) - Changing tensor layout
- [Broadcasting Rules](broadcasting_rules.md) - Implicit expansion
- [Memory Management](memory_management.md) - Storage and optimization
