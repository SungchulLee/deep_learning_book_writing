# Logical and Comparison Operations

## Overview

Logical and comparison operations create boolean tensors used for masking, conditional selection, and flow control. These operations are essential for data filtering, attention mechanisms, and loss masking.

## Element-wise Comparisons

### Basic Comparisons

```python
import torch

a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([3, 2, 1, 4, 6])

# Greater than
print(a > b)         # tensor([False, False, True, False, False])
print(torch.gt(a, b)) # Equivalent

# Greater than or equal
print(a >= b)        # tensor([False, True, True, True, False])
print(torch.ge(a, b))

# Less than
print(a < b)         # tensor([True, False, False, False, True])
print(torch.lt(a, b))

# Less than or equal
print(a <= b)        # tensor([True, True, False, True, True])
print(torch.le(a, b))

# Equal
print(a == b)        # tensor([False, True, False, True, False])
print(torch.eq(a, b))

# Not equal
print(a != b)        # tensor([True, False, True, False, True])
print(torch.ne(a, b))
```

### Scalar Comparisons

```python
t = torch.tensor([1, 2, 3, 4, 5])

# Compare with scalar
print(t > 3)         # tensor([False, False, False, True, True])
print(t == 3)        # tensor([False, False, True, False, False])
print(t <= 2)        # tensor([True, True, False, False, False])
```

## Tensor Equality

### `equal()` - Full Tensor Comparison

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 2, 3])
c = torch.tensor([1, 2, 4])

# Check if tensors are identical
print(torch.equal(a, b))  # True
print(torch.equal(a, c))  # False
```

### `allclose()` - Approximate Equality

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([1.0001, 2.0001, 3.0001])

# Exact equality fails
print(torch.equal(a, b))  # False

# Approximate equality (within tolerance)
print(torch.allclose(a, b, rtol=1e-3, atol=1e-3))  # True

# Default tolerances: rtol=1e-5, atol=1e-8
print(torch.allclose(a, b))  # False (default is stricter)
```

## Logical Operations

### AND, OR, NOT, XOR

```python
a = torch.tensor([True, True, False, False])
b = torch.tensor([True, False, True, False])

# Logical AND
print(torch.logical_and(a, b))  # tensor([True, False, False, False])
print(a & b)                     # Bitwise equivalent for bool

# Logical OR
print(torch.logical_or(a, b))   # tensor([True, True, True, False])
print(a | b)

# Logical NOT
print(torch.logical_not(a))     # tensor([False, False, True, True])
print(~a)

# Logical XOR
print(torch.logical_xor(a, b))  # tensor([False, True, True, False])
print(a ^ b)
```

### Combining Conditions

```python
t = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Multiple conditions
# Values between 3 and 7 (inclusive)
mask = (t >= 3) & (t <= 7)
print(f"3 <= t <= 7: {mask}")
print(f"Values: {t[mask]}")  # tensor([3, 4, 5, 6, 7])

# Values less than 3 OR greater than 7
mask = (t < 3) | (t > 7)
print(f"t < 3 or t > 7: {mask}")
print(f"Values: {t[mask]}")  # tensor([1, 2, 8, 9, 10])
```

## Boolean Reductions

### all() and any()

```python
t = torch.tensor([True, True, False, True])

# Are all elements True?
print(torch.all(t))   # tensor(False)

# Is any element True?
print(torch.any(t))   # tensor(True)

# Along dimensions
t2d = torch.tensor([[True, True],
                    [True, False],
                    [False, False]])

print(torch.all(t2d, dim=1))  # tensor([True, False, False])
print(torch.any(t2d, dim=1))  # tensor([True, True, False])
```

### With Comparisons

```python
scores = torch.tensor([85, 90, 72, 88, 65])

# All passing (>= 70)?
print(torch.all(scores >= 70))  # tensor(True)

# Any perfect (== 100)?
print(torch.any(scores == 100)) # tensor(False)

# Any failing (< 70)?
print(torch.any(scores < 70))   # tensor(True)
```

## Conditional Selection

### `where()` - Conditional Element Selection

```python
condition = torch.tensor([True, False, True, False])
x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([10, 20, 30, 40])

# Select from x where True, y where False
result = torch.where(condition, x, y)
print(result)  # tensor([1, 20, 3, 40])
```

### With Comparison Conditions

```python
t = torch.tensor([-2, -1, 0, 1, 2])

# Replace negatives with 0 (ReLU-like)
result = torch.where(t > 0, t, torch.zeros_like(t))
print(result)  # tensor([0, 0, 0, 1, 2])

# Clamp to range [-1, 1]
clamped = torch.where(t > 1, torch.ones_like(t), 
          torch.where(t < -1, -torch.ones_like(t), t))
print(clamped)  # tensor([-1, -1, 0, 1, 1])
```

### Broadcasting with `where()`

```python
# Broadcast scalar
t = torch.tensor([1, 2, 3, 4, 5])
result = torch.where(t > 3, t, 0)  # Scalar 0 broadcasts
print(result)  # tensor([0, 0, 0, 4, 5])
```

## Finding Elements

### `nonzero()` - Find Non-Zero Indices

```python
t = torch.tensor([0, 1, 0, 2, 0, 3])

# Indices of non-zero elements
indices = torch.nonzero(t)
print(indices)  # tensor([[1], [3], [5]])

# As tuple (useful for indexing)
indices_tuple = torch.nonzero(t, as_tuple=True)
print(indices_tuple)  # (tensor([1, 3, 5]),)

# Find where condition is True
mask = t > 1
where_gt_1 = torch.nonzero(mask)
print(where_gt_1)  # tensor([[3], [5]])
```

### 2D Nonzero

```python
t2d = torch.tensor([[0, 1, 0],
                    [2, 0, 3],
                    [0, 0, 0]])

indices = torch.nonzero(t2d)
print(indices)
# tensor([[0, 1],
#         [1, 0],
#         [1, 2]])

# As tuple for direct indexing
rows, cols = torch.nonzero(t2d, as_tuple=True)
print(f"Rows: {rows}, Cols: {cols}")
# Rows: tensor([0, 1, 1]), Cols: tensor([1, 0, 2])
```

## Special Value Detection

### NaN and Infinity

```python
t = torch.tensor([1.0, float('nan'), float('inf'), -float('inf'), 2.0])

# Detect NaN
print(torch.isnan(t))    # tensor([False, True, False, False, False])

# Detect infinity
print(torch.isinf(t))    # tensor([False, False, True, True, False])

# Detect positive/negative infinity
print(torch.isposinf(t)) # tensor([False, False, True, False, False])
print(torch.isneginf(t)) # tensor([False, False, False, True, False])

# Detect finite (not nan or inf)
print(torch.isfinite(t)) # tensor([True, False, False, False, True])
```

### Handling NaN

```python
t = torch.tensor([1.0, float('nan'), 3.0, float('nan'), 5.0])

# Replace NaN with value
t_clean = torch.where(torch.isnan(t), torch.zeros_like(t), t)
print(t_clean)  # tensor([1., 0., 3., 0., 5.])

# Using nan_to_num
t_replaced = torch.nan_to_num(t, nan=0.0, posinf=1e10, neginf=-1e10)
print(t_replaced)
```

## Masking Operations

### Boolean Masking

```python
features = torch.randn(5, 3)
mask = torch.tensor([True, False, True, False, True])

# Select rows where mask is True
selected = features[mask]
print(f"Selected shape: {selected.shape}")  # torch.Size([3, 3])
```

### Masked Fill

```python
t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

mask = torch.tensor([[False, True, False],
                     [True, False, True]])

# Fill masked positions with value
filled = t.masked_fill(mask, -float('inf'))
print(filled)
# tensor([[1., -inf, 3.],
#         [-inf, 5., -inf]])
```

### Attention Masking

```python
# Causal attention mask
seq_len = 4
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print(f"Causal mask:\n{causal_mask}")
# tensor([[False, True, True, True],
#         [False, False, True, True],
#         [False, False, False, True],
#         [False, False, False, False]])

# Apply to attention scores
scores = torch.randn(seq_len, seq_len)
masked_scores = scores.masked_fill(causal_mask, -float('inf'))

# Softmax will zero out masked positions
attention = torch.softmax(masked_scores, dim=-1)
```

### Padding Mask

```python
# Sequence padding mask
batch_size, max_len = 3, 5
lengths = torch.tensor([3, 5, 2])

# Create mask: True where padding
positions = torch.arange(max_len).expand(batch_size, max_len)
padding_mask = positions >= lengths.unsqueeze(1)
print(f"Padding mask:\n{padding_mask}")
# tensor([[False, False, False, True, True],
#         [False, False, False, False, False],
#         [False, False, True, True, True]])
```

## Practical Examples

### Data Filtering

```python
# Filter financial data
prices = torch.tensor([100.0, 150.0, 50.0, 200.0, 75.0])
volumes = torch.tensor([1000, 500, 2000, 800, 1500])

# Find high-volume, high-price
mask = (prices > 100) & (volumes > 700)
indices = torch.nonzero(mask, as_tuple=True)[0]
print(f"Qualifying indices: {indices}")
print(f"Prices: {prices[indices]}")
print(f"Volumes: {volumes[indices]}")
```

### Outlier Detection

```python
data = torch.randn(1000)

# Z-score based outlier detection
mean, std = data.mean(), data.std()
z_scores = torch.abs((data - mean) / std)

# Flag outliers (|z| > 3)
outliers = z_scores > 3
print(f"Number of outliers: {outliers.sum()}")

# Remove outliers
clean_data = data[~outliers]
```

### Loss Masking

```python
# Ignore padding in loss calculation
predictions = torch.randn(32, 50, 100)  # batch, seq, vocab
targets = torch.randint(0, 100, (32, 50))  # batch, seq
padding_token = 0

# Create mask for non-padding positions
mask = targets != padding_token

# Compute loss only on non-padding
loss_unreduced = torch.nn.functional.cross_entropy(
    predictions.view(-1, 100), 
    targets.view(-1), 
    reduction='none'
).view(32, 50)

# Apply mask and average
masked_loss = loss_unreduced * mask
loss = masked_loss.sum() / mask.sum()
print(f"Masked loss: {loss}")
```

## Summary

| Operation | Function | Description |
|-----------|----------|-------------|
| `>`, `<`, `>=`, `<=` | `gt`, `lt`, `ge`, `le` | Comparisons |
| `==`, `!=` | `eq`, `ne` | Equality |
| `&`, `\|`, `~`, `^` | `logical_and/or/not/xor` | Logical ops |
| `all`, `any` | `torch.all`, `torch.any` | Boolean reductions |
| `where` | `torch.where` | Conditional selection |
| `nonzero` | `torch.nonzero` | Find indices |
| `isnan`, `isinf` | `torch.isnan`, `torch.isinf` | Special values |
| `masked_fill` | `t.masked_fill` | Fill at mask positions |

## See Also

- [Indexing and Slicing](indexing_slicing.md) - Boolean indexing
- [Reduction Operations](reduction_operations.md) - Boolean reductions
- [Broadcasting Rules](broadcasting_rules.md) - Comparison broadcasting
