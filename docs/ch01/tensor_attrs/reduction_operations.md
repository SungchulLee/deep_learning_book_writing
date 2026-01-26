# Reduction Operations

Reduction operations aggregate tensor values along specified dimensions, producing outputs with fewer dimensions than the input. These operations are fundamental for computing losses, normalizing data, and implementing attention mechanisms.

## Understanding Reductions

### The `dim` Parameter

The `dim` parameter specifies which dimension to collapse:

```python
import torch

x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])

# Sum along dim 0 (collapse rows, keep columns)
print(x.sum(dim=0))  # tensor([12., 15., 18.])

# Sum along dim 1 (collapse columns, keep rows)
print(x.sum(dim=1))  # tensor([6., 15., 24.])
```

**Mental model**: The dimension you specify is the one that disappears.

### The `keepdim` Parameter

`keepdim=True` preserves reduced dimensions as size 1—essential for broadcasting:

```python
x = torch.randn(3, 4, 5)

# Without keepdim (default): dimension removed
mean = x.mean(dim=1)
print(mean.shape)  # torch.Size([3, 5])

# With keepdim: dimension retained as size 1
mean_keep = x.mean(dim=1, keepdim=True)
print(mean_keep.shape)  # torch.Size([3, 1, 5])

# Essential for broadcasting operations
normalized = x - mean_keep  # Broadcasts correctly!
```

**Rule of thumb**: Always use `keepdim=True` when the result will be used in subsequent broadcasting operations with the original tensor.

### Multiple Dimensions

Reduce over multiple dimensions simultaneously:

```python
x = torch.randn(2, 3, 4, 5)

# Reduce over spatial dimensions
spatial_mean = x.mean(dim=(2, 3))
print(spatial_mean.shape)  # torch.Size([2, 3])

# Reduce over batch and spatial (common in batch normalization)
channel_stats = x.mean(dim=(0, 2, 3), keepdim=True)
print(channel_stats.shape)  # torch.Size([1, 3, 1, 1])
```

## Basic Reductions

### Sum

```python
t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# Global sum
print(torch.sum(t))              # tensor(21.)

# Sum along dimension 0 (columns)
print(torch.sum(t, dim=0))       # tensor([5., 7., 9.])

# Sum along dimension 1 (rows)
print(torch.sum(t, dim=1))       # tensor([6., 15.])

# Method syntax (equivalent)
print(t.sum(dim=1))              # tensor([6., 15.])
```

### Mean

```python
t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# Global mean
print(torch.mean(t))             # tensor(3.5)

# Mean along dimensions
print(torch.mean(t, dim=0))      # tensor([2.5, 3.5, 4.5])
print(torch.mean(t, dim=1))      # tensor([2., 5.])
```

### Product

```python
t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# Global product
print(torch.prod(t))             # tensor(720.)

# Product along dimensions
print(torch.prod(t, dim=0))      # tensor([4., 10., 18.])
print(torch.prod(t, dim=1))      # tensor([6., 120.])
```

## Min and Max Operations

### Global Min/Max

```python
x = torch.randn(3, 4)

print(x.min())  # Minimum value (scalar)
print(x.max())  # Maximum value (scalar)
```

### Dimension-wise Min/Max

Returns both values AND indices when `dim` is specified:

```python
t = torch.tensor([[3.0, 1.0, 4.0],
                  [1.0, 5.0, 9.0]])

# Min along dim 1 (per row)
min_vals, min_idx = torch.min(t, dim=1)
print(f"Min values: {min_vals}")   # tensor([1., 1.])
print(f"Min indices: {min_idx}")   # tensor([1, 0])

# Max along dim 0 (per column)
max_vals, max_idx = torch.max(t, dim=0)
print(f"Max values: {max_vals}")   # tensor([3., 5., 9.])
print(f"Max indices: {max_idx}")   # tensor([0, 1, 1])
```

### argmin and argmax

Get only the indices:

```python
t = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])

# Global indices (flattened position)
print(torch.argmin(t))           # tensor(1)
print(torch.argmax(t))           # tensor(5)

# Dimension-wise
t2d = torch.tensor([[3.0, 1.0, 4.0],
                    [1.0, 5.0, 9.0]])

print(torch.argmax(t2d, dim=0))  # tensor([0, 1, 1]) - per column
print(torch.argmax(t2d, dim=1))  # tensor([2, 2])    - per row
```

### aminmax

Get both min and max efficiently in one call:

```python
x = torch.randn(3, 4)

min_val, max_val = x.aminmax()

# Dimension-wise
min_vals, max_vals = x.aminmax(dim=1)
```

### Element-wise Min/Max Between Tensors

```python
a = torch.tensor([1.0, 5.0, 3.0])
b = torch.tensor([2.0, 3.0, 4.0])

# Element-wise minimum
print(torch.minimum(a, b))       # tensor([1., 3., 3.])

# Element-wise maximum
print(torch.maximum(a, b))       # tensor([2., 5., 4.])
```

## Top-K and K-th Value

### Top-K

```python
t = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])

# Top 3 largest
values, indices = torch.topk(t, k=3)
print(f"Top 3 values: {values}")    # tensor([9., 6., 5.])
print(f"Top 3 indices: {indices}")  # tensor([5, 7, 4])

# Top 3 smallest
values, indices = torch.topk(t, k=3, largest=False)
print(f"Bottom 3 values: {values}") # tensor([1., 1., 2.])

# Along a dimension
t2d = torch.randn(4, 10)
top_vals, top_idx = torch.topk(t2d, k=3, dim=1)
print(top_vals.shape)  # torch.Size([4, 3])
```

### K-th Value

```python
t = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])

# 3rd smallest value (k is 1-indexed)
value, index = torch.kthvalue(t, k=3)
print(f"3rd smallest: {value} at index {index}")  # 3.0 at index 0
```

## Cumulative Operations

Cumulative operations preserve shape—they don't reduce dimensions.

### Cumulative Sum

```python
t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

cumsum = torch.cumsum(t, dim=0)
print(cumsum)  # tensor([1., 3., 6., 10., 15.])

# 2D cumulative sum
t2d = torch.tensor([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]])

# Along rows (dim=0)
print(torch.cumsum(t2d, dim=0))
# tensor([[1., 2., 3.],
#         [5., 7., 9.]])

# Along columns (dim=1)
print(torch.cumsum(t2d, dim=1))
# tensor([[ 1.,  3.,  6.],
#         [ 4.,  9., 15.]])
```

### Cumulative Product

```python
t = torch.tensor([1.0, 2.0, 3.0, 4.0])

cumprod = torch.cumprod(t, dim=0)
print(cumprod)  # tensor([1., 2., 6., 24.])
```

### Cumulative Max/Min

```python
t = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])

# Cumulative maximum (running max)
cummax_vals, cummax_idx = torch.cummax(t, dim=0)
print(f"Cummax values: {cummax_vals}")   # tensor([3., 3., 4., 4., 5., 9.])
print(f"Cummax indices: {cummax_idx}")   # tensor([0, 0, 2, 2, 4, 5])

# Cumulative minimum (running min)
cummin_vals, cummin_idx = torch.cummin(t, dim=0)
print(f"Cummin values: {cummin_vals}")   # tensor([3., 1., 1., 1., 1., 1.])
```

## Logical Reductions

### all and any

```python
x = torch.tensor([True, True, False, True])

print(x.all())  # tensor(False) - not all True
print(x.any())  # tensor(True)  - at least one True

# Practical usage with comparisons
values = torch.randn(5)
all_positive = (values > 0).all()
any_negative = (values < 0).any()
```

### Dimension-wise Logical Reductions

```python
x = torch.tensor([[True, True, True],
                  [True, False, True],
                  [False, False, False]])

print(x.all(dim=1))  # tensor([True, False, False]) - all True per row
print(x.any(dim=1))  # tensor([True, True, False])  - any True per row
```

## Counting Operations

### numel

Total number of elements:

```python
x = torch.randn(2, 3, 4)
print(x.numel())  # 24
```

### count_nonzero

```python
x = torch.tensor([[1, 0, 3], [0, 0, 6], [7, 8, 0]])

print(torch.count_nonzero(x))          # tensor(5)

# Per dimension
print(torch.count_nonzero(x, dim=0))   # tensor([2, 1, 2])
print(torch.count_nonzero(x, dim=1))   # tensor([2, 1, 2])
```

### Unique Values and Counts

```python
x = torch.tensor([1, 2, 2, 3, 3, 3, 4])

# Unique values
unique = torch.unique(x)  # tensor([1, 2, 3, 4])

# With counts
unique_vals, counts = torch.unique(x, return_counts=True)
# unique_vals: tensor([1, 2, 3, 4])
# counts: tensor([1, 2, 3, 1])

# With inverse indices (for reconstruction)
unique_vals, inverse = torch.unique(x, return_inverse=True)
# x can be reconstructed as unique_vals[inverse]
```

## Norm Reductions

```python
t = torch.tensor([3.0, 4.0])

# L2 norm (Euclidean)
l2 = torch.norm(t, p=2)
print(f"L2 norm: {l2}")          # tensor(5.) = sqrt(3² + 4²)

# L1 norm (Manhattan)
l1 = torch.norm(t, p=1)
print(f"L1 norm: {l1}")          # tensor(7.) = |3| + |4|

# L-infinity norm
linf = torch.norm(t, p=float('inf'))
print(f"L∞ norm: {linf}")        # tensor(4.)

# Frobenius norm for matrices
mat = torch.randn(3, 4)
fro = torch.norm(mat, p='fro')

# Along specific dimensions
x = torch.randn(32, 128)
row_norms = torch.norm(x, p=2, dim=1)  # L2 norm per row
print(row_norms.shape)  # torch.Size([32])
```

## Practical Examples

### Softmax Normalization

```python
logits = torch.randn(32, 10)  # Batch of 32, 10 classes

# Manual softmax: exp(x) / sum(exp(x))
exp_logits = torch.exp(logits)
softmax_manual = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

# Built-in (numerically stable)
softmax_builtin = torch.softmax(logits, dim=1)

print(torch.allclose(softmax_manual, softmax_builtin))  # True
```

### Log-Sum-Exp (Numerically Stable)

```python
logits = torch.randn(32, 10)

# Numerically stable log-sum-exp
lse = torch.logsumexp(logits, dim=1, keepdim=True)

# Log-softmax
log_probs = logits - lse
```

### Attention Weights Normalization

```python
# Attention scores before softmax
scores = torch.randn(8, 4, 10, 20)  # (batch, heads, queries, keys)

# Normalize over keys dimension
attn_weights = torch.softmax(scores, dim=-1)

# Verify each query sums to 1
print(attn_weights.sum(dim=-1)[0, 0, :5])  # All ~1.0
```

### Finding Best Predictions

```python
logits = torch.randn(32, 10)  # 32 samples, 10 classes

# Get predicted classes
predictions = torch.argmax(logits, dim=1)
print(predictions.shape)  # torch.Size([32])

# Get top-3 predictions with confidence
top_values, top_indices = torch.topk(torch.softmax(logits, dim=1), k=3, dim=1)
print(top_indices.shape)  # torch.Size([32, 3])
```

### Masking with Reductions

```python
# Check if any position in sequence is padding
sequences = torch.randint(0, 100, (32, 50))  # (batch, seq_len)
padding_token = 0

has_padding = (sequences == padding_token).any(dim=1)
print(has_padding.shape)  # torch.Size([32])

# Count non-padding tokens per sequence
non_padding_counts = (sequences != padding_token).sum(dim=1)
```

## Quick Reference

| Operation | Function | Returns |
|-----------|----------|---------|
| `sum(dim)` | `torch.sum` | Sum of elements |
| `mean(dim)` | `torch.mean` | Arithmetic mean |
| `prod(dim)` | `torch.prod` | Product of elements |
| `min(dim)` | `torch.min` | Values and indices |
| `max(dim)` | `torch.max` | Values and indices |
| `argmin(dim)` | `torch.argmin` | Indices only |
| `argmax(dim)` | `torch.argmax` | Indices only |
| `aminmax(dim)` | `torch.aminmax` | Min and max together |
| `topk(k, dim)` | `torch.topk` | Top k values and indices |
| `kthvalue(k, dim)` | `torch.kthvalue` | K-th smallest value |
| `cumsum(dim)` | `torch.cumsum` | Running sum (no reduction) |
| `cumprod(dim)` | `torch.cumprod` | Running product (no reduction) |
| `cummax(dim)` | `torch.cummax` | Running max with indices |
| `cummin(dim)` | `torch.cummin` | Running min with indices |
| `all(dim)` | `torch.all` | All True? |
| `any(dim)` | `torch.any` | Any True? |
| `count_nonzero(dim)` | `torch.count_nonzero` | Non-zero count |
| `norm(p, dim)` | `torch.norm` | p-norm |

## Key Takeaways

1. **`dim` specifies which dimension disappears** in the output
2. **Use `keepdim=True`** when broadcasting with the original tensor
3. **`min`/`max` return values AND indices** when dim is specified; use `argmin`/`argmax` for indices only
4. **Multiple dimensions** can be reduced simultaneously with a tuple
5. **Cumulative operations preserve shape**—they're not true reductions
6. **`logsumexp`** is the numerically stable way to compute log of sum of exponentials

## See Also

- [Statistics Operations](statistics_operations.md) - Variance, std, quantiles
- [Broadcasting Rules](broadcasting_rules.md) - Shape compatibility
- [Shape and Dimensions](shape_dimensions.md) - Dimension concepts
