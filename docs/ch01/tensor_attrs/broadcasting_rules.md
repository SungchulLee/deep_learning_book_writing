# Broadcasting Rules

Broadcasting is a powerful mechanism that allows PyTorch to perform element-wise operations on tensors of different shapes by automatically expanding them to compatible shapes without copying data. Understanding broadcasting is essential for writing efficient, correct, and memory-efficient deep learning code.

## The Broadcasting Concept

When performing element-wise operations between two tensors, PyTorch automatically expands the smaller tensor to match the shape of the larger one using stride tricks—no data is physically copied.

```python
import torch

# Without broadcasting: manual expansion needed
a = torch.tensor([1, 2, 3])
b = torch.tensor([[1], [2], [3]])
# You'd need to manually tile/repeat

# With broadcasting: automatic and memory-efficient
a = torch.tensor([1, 2, 3])       # Shape: (3,)
b = torch.tensor([[1], [2], [3]]) # Shape: (3, 1)
c = a + b  # Shape: (3, 3) - automatic!
```

## The Three Broadcasting Rules

Two tensors are **broadcastable** if, iterating from the last (rightmost) dimension:

1. **Dimensions are equal**, OR
2. **One dimension is 1**, OR
3. **One dimension doesn't exist** (tensor has fewer dimensions)

### Rule 1: Dimension Alignment (Right-to-Left)

**Dimensions are aligned from the right (trailing dimensions), not the left.**

```python
a = torch.randn(3, 4, 5)  # Shape: (3, 4, 5)
b = torch.randn(5)        # Shape: (5,)

# Alignment (right-justified):
#   a: (3, 4, 5)
#   b:       (5)  ← implicitly (1, 1, 5)

result = a + b  # Shape: (3, 4, 5)
```

### Rule 2: Prepending Ones

**The shorter tensor is prepended with 1s to match the number of dimensions.**

```python
a = torch.randn(3, 4, 5)  # Shape: (3, 4, 5)
b = torch.randn(4, 5)     # Shape: (4, 5)

# After prepending 1:
#   a: (3, 4, 5)
#   b: (1, 4, 5)  ← 1 prepended

result = a + b  # Shape: (3, 4, 5)
```

### Rule 3: Size-1 Expansion

**Dimensions with size 1 are virtually expanded to match the corresponding dimension.**

```python
a = torch.randn(3, 1, 5)  # Shape: (3, 1, 5)
b = torch.randn(1, 4, 5)  # Shape: (1, 4, 5)

# Broadcasting:
#   a: (3, 1, 5) → (3, 4, 5)  (middle dim expands)
#   b: (1, 4, 5) → (3, 4, 5)  (first dim expands)

result = a + b  # Shape: (3, 4, 5)
```

### Rule Visualization

```
Shapes are compared right-to-left:

Shape A:     (3, 1)
Shape B:        (4)     ← B gets prepended with 1s → (1, 4)
              -----
Compare:     (3, 1)
             (1, 4)
              ↓  ↓
Result:      (3, 4)     ← max of each dimension
```

## Compatibility Requirement

For each dimension, the sizes must either:

1. **Be equal**, or
2. **One of them must be 1**

```python
# Compatible
(5, 1, 7) and (   3, 7) → (5, 3, 7)  ✓
(3, 1)    and (1, 4)    → (3, 4)     ✓
(8, 1, 6, 1) and (7, 1, 5) → (8, 7, 6, 5)  ✓

# Incompatible
(3, 4) and (5,) → ERROR  ✗  (4 ≠ 5)
(2, 3) and (3, 2) → ERROR  ✗  (2≠3 and 3≠2)
```

## Basic Examples

### Scalar Broadcasting

Scalars (0-D tensors) broadcast to any shape:

```python
# Scalar broadcasts to any shape
vec = torch.tensor([1, 2, 3, 4])
result = vec + 10
print(result)  # tensor([11, 12, 13, 14])

mat = torch.ones(3, 4)
result = mat * 5
print(result)  # 3x4 tensor of 5s

# Using 0-D tensor
x = torch.randn(3, 4)
scalar = torch.tensor(5.0)  # Shape: ()
result = x * scalar
print(result.shape)  # torch.Size([3, 4])
```

### Vector with Matrix

```python
mat = torch.arange(12).reshape(3, 4)
print(f"Matrix (3x4):\n{mat}")

# Row vector (shape 4) broadcasts with matrix (shape 3x4)
row_vec = torch.tensor([10, 20, 30, 40])
result = mat + row_vec
print(f"\nMatrix + row vector:\n{result}")

# Column vector (shape 3x1) broadcasts with matrix (shape 3x4)
col_vec = torch.tensor([[100], [200], [300]])
result = mat + col_vec
print(f"\nMatrix + column vector:\n{result}")
```

### Outer Product via Broadcasting

```python
# Column vector (3x1) with row vector (1x4) → (3x4)
col = torch.arange(3).reshape(3, 1)
row = torch.arange(4).reshape(1, 4)

print(f"Column (3x1):\n{col}")
print(f"Row (1x4):\n{row}")

outer = col + row
print(f"\nOuter sum (3x4):\n{outer}")

# Multiplication gives outer product
outer_prod = col * row
print(f"\nOuter product (3x4):\n{outer_prod}")
```

## Broadcasting Mechanics: Step-by-Step Process

```python
# Example: (2, 3, 1) + (3, 5)

# Step 1: Align shapes from the right, prepend 1s
# A: (2, 3, 1)
# B:    (3, 5) → (1, 3, 5)

# Step 2: Compare each dimension (right to left)
# Dim 2: 1 vs 5 → 5 (1 broadcasts)
# Dim 1: 3 vs 3 → 3 (equal)
# Dim 0: 2 vs 1 → 2 (1 broadcasts)

# Result shape: (2, 3, 5)

A = torch.randn(2, 3, 1)
B = torch.randn(3, 5)
C = A + B
print(f"(2,3,1) + (3,5) = {C.shape}")  # torch.Size([2, 3, 5])
```

### Compatibility Check Function

```python
def check_broadcast(shape1, shape2):
    """Check if two shapes can broadcast and compute result shape."""
    # Pad shorter shape with 1s on the left
    len_diff = len(shape1) - len(shape2)
    if len_diff > 0:
        shape2 = (1,) * len_diff + shape2
    else:
        shape1 = (1,) * (-len_diff) + shape1
    
    result = []
    for d1, d2 in zip(shape1, shape2):
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise ValueError(f"Cannot broadcast {shape1} with {shape2}")
    
    return tuple(result)

# Examples
print(check_broadcast((3, 1), (1, 4)))      # (3, 4)
print(check_broadcast((2, 3, 1), (3, 5)))   # (2, 3, 5)
print(check_broadcast((5,), (4, 5)))        # (4, 5)
```

## When Broadcasting Fails

```python
# Incompatible shapes
A = torch.randn(3, 4)
B = torch.randn(2, 3)

try:
    C = A + B
except RuntimeError as e:
    print(f"Error: Cannot broadcast (3,4) with (2,3)")
    print(f"Reason: Dim 1: 4 ≠ 3 (neither is 1)")
    print(f"        Dim 0: 3 ≠ 2 (neither is 1)")
```

### Common Incompatible Patterns

| Shape A | Shape B | Why It Fails |
|---------|---------|--------------|
| (3, 4) | (2, 4) | First dims differ (3 ≠ 2) |
| (5, 3) | (5, 2) | Last dims differ (3 ≠ 2) |
| (2, 3, 4) | (2, 5, 4) | Middle dims differ (3 ≠ 5) |
| (3, 4) | (5,) | 4 ≠ 5 |
| (2, 3) | (3, 2) | 2≠3 and 3≠2 |
| (3,) | (4,) | 3 ≠ 4 |

## Machine Learning Applications

### Adding Bias to Layer Output

```python
# Common pattern: add bias to batch of features
batch_size, features = 32, 256
output = torch.randn(batch_size, features)  # (32, 256)
bias = torch.randn(features)                 # (256,)

# Bias broadcasts: (256,) → (1, 256) → (32, 256)
output_biased = output + bias
print(f"Output with bias: {output_biased.shape}")  # torch.Size([32, 256])
```

### Data Normalization

```python
# Normalize features (subtract mean, divide by std)
data = torch.randn(100, 5)  # 100 samples, 5 features

# Compute statistics along batch dimension
mean = data.mean(dim=0, keepdim=True)  # (1, 5)
std = data.std(dim=0, keepdim=True)    # (1, 5)

# Broadcasting handles the normalization
normalized = (data - mean) / std
print(f"Normalized shape: {normalized.shape}")  # torch.Size([100, 5])
```

### Channel-wise Operations (Images)

Per-channel scaling in images:

```python
# (batch, channels, H, W) * (channels, 1, 1) → (batch, channels, H, W)
images = torch.randn(8, 3, 64, 64)  # Batch of RGB images
scale = torch.randn(3, 1, 1)        # Per-channel scale

scaled = images * scale  # Shape: (8, 3, 64, 64)
```

### Pairwise Distances

```python
# Compute all pairwise distances between two sets of points
points_a = torch.randn(5, 2)  # 5 points in 2D
points_b = torch.randn(3, 2)  # 3 points in 2D

# Reshape for broadcasting: (5, 1, 2) - (1, 3, 2) = (5, 3, 2)
diff = points_a.unsqueeze(1) - points_b.unsqueeze(0)
distances = torch.sqrt((diff ** 2).sum(dim=2))

print(f"Points A: {points_a.shape}")      # torch.Size([5, 2])
print(f"Points B: {points_b.shape}")      # torch.Size([3, 2])
print(f"Distances: {distances.shape}")    # torch.Size([5, 3])
```

### Attention Scores

```python
# Scaled dot-product attention uses broadcasting
batch, heads, seq_len, d_k = 4, 8, 100, 64

Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)

# Attention scores: (B, H, L, d) @ (B, H, d, L) → (B, H, L, L)
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
print(f"Attention scores: {scores.shape}")  # torch.Size([4, 8, 100, 100])
```

### Comparing All Pairs of Vectors (Attention Pattern)

```python
# Compute pairwise distances for attention
queries = torch.randn(8, 10, 64)   # (batch, q_len, dim)
keys = torch.randn(8, 20, 64)      # (batch, k_len, dim)

# Unsqueeze for broadcasting
q = queries.unsqueeze(2)  # (8, 10, 1, 64)
k = keys.unsqueeze(1)     # (8, 1, 20, 64)

# Broadcast and compute difference
diff = q - k  # (8, 10, 20, 64)
distances = diff.pow(2).sum(-1)  # (8, 10, 20)
```

## The `keepdim` Parameter

The `keepdim=True` parameter preserves dimensions for easy broadcasting:

```python
data = torch.randn(4, 5)

# Without keepdim
row_sum = data.sum(dim=1)
print(f"sum(dim=1): {row_sum.shape}")  # torch.Size([4]) - lost dimension!

# With keepdim
row_sum_keep = data.sum(dim=1, keepdim=True)
print(f"sum(dim=1, keepdim=True): {row_sum_keep.shape}")  # torch.Size([4, 1])

# Now broadcasting works naturally
normalized = data / row_sum_keep  # (4, 5) / (4, 1) → (4, 5)
print(f"Row-normalized: {normalized.shape}")
```

## Explicit Broadcasting Methods

### `.expand()` - Create Broadcast View

The `expand()` method explicitly broadcasts without copying data:

```python
vec = torch.tensor([1, 2, 3])
print(f"Original: {vec.shape}")  # torch.Size([3])

# Expand to (4, 3)
expanded = vec.expand(4, 3)
print(f"Expanded:\n{expanded}")
print(f"Shape: {expanded.shape}")  # torch.Size([4, 3])

# Verify no copy (same storage)
print(f"Same storage: {vec.storage().data_ptr() == expanded.storage().data_ptr()}")
# True - no memory copied!

# Stride of 0 means "repeat"
print(expanded.stride())  # (0, 1)
```

### `.broadcast_to()` - Explicit Target Shape

```python
x = torch.tensor([1., 2., 3.])
target_shape = (4, 3)

broadcasted = x.broadcast_to(target_shape)
print(broadcasted.shape)  # torch.Size([4, 3])
```

### `torch.broadcast_shapes()` - Compute Result Shape

```python
shape1 = (3, 1, 5)
shape2 = (1, 4, 5)

result_shape = torch.broadcast_shapes(shape1, shape2)
print(result_shape)  # torch.Size([3, 4, 5])
```

### `torch.broadcast_tensors()` - Broadcast Multiple Tensors

```python
a = torch.randn(3, 1)
b = torch.randn(1, 4)
c = torch.randn(3, 4)

# All tensors broadcast to common shape
a_bc, b_bc, c_bc = torch.broadcast_tensors(a, b, c)
print(a_bc.shape, b_bc.shape, c_bc.shape)
# All: torch.Size([3, 4])
```

### `expand()` vs `repeat()`

```python
vec = torch.tensor([1, 2, 3])

# expand: creates a view (no copy, memory efficient)
expanded = vec.expand(4, 3)

# repeat: creates a copy (uses more memory)
repeated = vec.repeat(4, 1)

print(f"expand same storage: {vec.storage().data_ptr() == expanded.storage().data_ptr()}")
# True

print(f"repeat same storage: {vec.storage().data_ptr() == repeated.storage().data_ptr()}")
# False - actually copied
```

!!! warning "Don't Modify Expanded Tensors In-Place"
    Since `expand()` creates views, in-place modifications can have
    unexpected effects. Use `expand().clone()` if you need to modify.

## Memory Efficiency

Broadcasting is memory-efficient because no data is actually copied:

```python
# Small tensor
small = torch.tensor([[1, 2, 3]])  # (1, 3)

# Large tensor
large = torch.randn(1000, 3)

# Broadcasting happens virtually
result = small + large  # (1000, 3)

print(f"Small tensor size: {small.numel()} elements")   # 3
print(f"Large tensor size: {large.numel()} elements")   # 3000
print(f"Result size: {result.numel()} elements")        # 3000

# But small wasn't actually expanded to 1000x3 in memory!

# Demonstrating stride tricks
expanded = small.expand(1000, 3)
print(f"Only {small.storage().size()} elements stored, not {1000*3}")
print(f"Stride: {expanded.stride()}")  # (0, 1) - stride 0 means "repeat"
```

## Einstein Summation (einsum)

`torch.einsum` provides explicit control over broadcasting and contraction:

```python
# Matrix multiplication with einsum
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# 'ij,jk->ik' means: A[i,j] * B[j,k] → C[i,k]
C = torch.einsum('ij,jk->ik', A, B)
print(f"Matrix multiply: {C.shape}")  # torch.Size([3, 5])

# Batch matrix multiply
batch_A = torch.randn(32, 3, 4)
batch_B = torch.randn(32, 4, 5)
batch_C = torch.einsum('bij,bjk->bik', batch_A, batch_B)
print(f"Batch matmul: {batch_C.shape}")  # torch.Size([32, 3, 5])

# Outer product
a = torch.randn(3)
b = torch.randn(4)
outer = torch.einsum('i,j->ij', a, b)
print(f"Outer product: {outer.shape}")  # torch.Size([3, 4])
```

## Broadcasting with Operations

### Element-wise Operations

```python
a = torch.randn(3, 4)
b = torch.randn(4)

a + b       # Addition
a - b       # Subtraction
a * b       # Multiplication
a / b       # Division
a ** b      # Power
torch.max(a, b)  # Element-wise max
```

### Comparison Operations

```python
a = torch.randn(3, 4)
threshold = torch.tensor(0.5)

a > threshold   # Broadcasts scalar
a < torch.randn(4)  # Broadcasts vector
```

### torch.where with Broadcasting

```python
condition = torch.randn(3, 4) > 0
a = torch.randn(3, 4)
b = torch.randn(4)  # Will broadcast

result = torch.where(condition, a, b)  # Shape: (3, 4)
```

## Common Pitfalls and Solutions

### Pitfall 1: Missing `keepdim`

```python
data = torch.randn(10, 5)

# Wrong: loses dimension
mean = data.mean(dim=1)  # shape (10,)
# centered = data - mean  # Error: shapes (10,5) and (10,) incompatible

# Correct: keep dimension
mean = data.mean(dim=1, keepdim=True)  # shape (10, 1)
centered = data - mean  # Works: (10, 5) - (10, 1) → (10, 5)
```

### Pitfall 2: Unintended Broadcasting

```python
# Bug: forgot keepdim, unexpected broadcast
images = torch.randn(10, 3, 32, 32)

# Computing per-channel mean
wrong_mean = images.mean(dim=(2, 3))  # Shape: (10, 3)
# This will broadcast incorrectly!

# Fix: use keepdim=True
correct_mean = images.mean(dim=(2, 3), keepdim=True)  # Shape: (10, 3, 1, 1)
normalized = images - correct_mean  # Correct broadcasting
```

### Pitfall 3: Unexpected Broadcasting to Higher Dimensions

```python
# Intended: element-wise multiply two vectors
a = torch.tensor([1, 2, 3])      # (3,)
b = torch.tensor([[4], [5]])     # (2, 1)

# Actually broadcasts to (2, 3)!
result = a * b
print(f"Unexpected shape: {result.shape}")  # torch.Size([2, 3])
```

### Pitfall 4: Broadcast vs Batch Dimension Confusion

```python
# Be careful with batch dimensions
weights = torch.randn(10)        # (10,) - feature weights
features = torch.randn(32, 10)   # (32, 10) - batch of features

# This works due to broadcasting
weighted = features * weights    # (32, 10) * (10,) → (32, 10)

# But make sure you intended this, not element-wise across batch!
```

### Pitfall 5: Wrong Dimension Order

```python
# (batch, features) + (batch,) - wrong!
data = torch.randn(32, 128)
scale = torch.randn(32)  # Per-sample scale

# This broadcasts scale across features, not samples!
# Wrong: result = data * scale

# Correct: unsqueeze to (32, 1)
correct = data * scale.unsqueeze(1)
```

### Pitfall 6: Shape Mismatch Silent Error Prevention

```python
# Subtle bug: wrong dimension sizing
batch = torch.randn(32, 128)
weights = torch.randn(256)  # Oops! Should be 128

try:
    result = batch * weights  # Error: 128 ≠ 256
except RuntimeError as e:
    print("Caught:", e)
```

### Pitfall 7: Accidental Outer Product Attempt

```python
a = torch.randn(5)
b = torch.randn(3)

# Intended: element-wise (need same length)
# Actual: attempt to broadcast (will fail)
try:
    c = a + b  # Error: shapes don't broadcast
except RuntimeError:
    print("5 and 3 don't broadcast for element-wise ops")
```

## Debugging Broadcasting

### Print Shapes Explicitly

```python
def broadcast_debug(a, b, op_name="+"):
    try:
        result_shape = torch.broadcast_shapes(a.shape, b.shape)
        print(f"{a.shape} {op_name} {b.shape} → {result_shape}")
        return True
    except RuntimeError as e:
        print(f"{a.shape} {op_name} {b.shape} → INCOMPATIBLE")
        return False
```

### Visualize Broadcasting

```python
def show_broadcast(shape_a, shape_b):
    """Show how shapes align for broadcasting."""
    # Right-align shapes
    max_dim = max(len(shape_a), len(shape_b))
    a_padded = (1,) * (max_dim - len(shape_a)) + tuple(shape_a)
    b_padded = (1,) * (max_dim - len(shape_b)) + tuple(shape_b)
    
    print(f"a: {a_padded}")
    print(f"b: {b_padded}")
    
    result = []
    for da, db in zip(a_padded, b_padded):
        if da == db:
            result.append(da)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        else:
            return "INCOMPATIBLE"
    print(f"→  {tuple(result)}")

show_broadcast((3, 1, 5), (4, 5))
# a: (3, 1, 5)
# b: (1, 4, 5)
# →  (3, 4, 5)
```

## Quick Reference Tables

### Common Broadcasting Patterns

| Pattern | Shape A | Shape B | Result |
|---------|---------|---------|--------|
| Scalar + tensor | () | (3, 4) | (3, 4) |
| Row broadcast | (1, 4) | (3, 4) | (3, 4) |
| Column broadcast | (3, 1) | (3, 4) | (3, 4) |
| Outer operation | (3, 1) | (1, 4) | (3, 4) |
| Dimension prepend | (4,) | (3, 4) | (3, 4) |
| Bias addition | (features,) | (batch, features) | (batch, features) |
| Channel-wise | (C, 1, 1) | (B, C, H, W) | (B, C, H, W) |

### Compatible Shape Examples

| Shape A | Shape B | Result |
|---------|---------|--------|
| `(5, 1, 7)` | `(3, 7)` | `(5, 3, 7)` |
| `(3, 1)` | `(1, 4)` | `(3, 4)` |
| `(8, 1, 6, 1)` | `(7, 1, 5)` | `(8, 7, 6, 5)` |
| `(5,)` | `(3, 5)` | `(3, 5)` |
| `()` | `(3, 4)` | `(3, 4)` |

### Incompatible Shape Examples

| Shape A | Shape B | Why It Fails |
|---------|---------|--------------|
| `(3, 4)` | `(5,)` | 4 ≠ 5 |
| `(2, 3)` | `(3, 2)` | 2≠3 and 3≠2 |
| `(3,)` | `(4,)` | 3 ≠ 4 |

## Best Practices

1. **Always use `keepdim=True`** when reducing dimensions for later broadcasting:
   ```python
   mean = x.mean(dim=1, keepdim=True)
   normalized = x - mean
   ```

2. **Explicit unsqueeze** for clarity:
   ```python
   # Clear: adding column vector
   col = weights.unsqueeze(1)
   result = matrix * col
   ```

3. **Verify shapes** before operations:
   ```python
   assert a.shape[-1] == b.shape[-1], "Last dimension must match"
   ```

4. **Use `broadcast_shapes`** to check compatibility:
   ```python
   result_shape = torch.broadcast_shapes(a.shape, b.shape)
   ```

5. **Document expected shapes** in function docstrings:
   ```python
   def normalize(x, mean, std):
       """
       Args:
           x: (batch, features)
           mean: (features,) - broadcasts to x
           std: (features,) - broadcasts to x
       """
       return (x - mean) / std
   ```

6. **Use `expand()` for memory efficiency** when you need explicit broadcasting without data copy.

7. **Prefer `expand()` over `repeat()`** unless you actually need a separate copy of the data.

## Key Takeaways

1. **Broadcasting aligns from the right** (trailing dimensions)
2. **Size-1 dimensions expand** to match
3. **No data is copied** — broadcasting uses stride tricks for memory efficiency
4. **Use `keepdim=True`** to preserve dimensions for subsequent broadcasting
5. **Explicit `unsqueeze`** makes code clearer and prevents errors
6. **Check shapes early** to catch broadcasting errors
7. **Scalars broadcast to anything**
8. **Incompatible shapes raise RuntimeError** — no silent failures
9. **`expand()` creates views**, `repeat()` creates copies
10. **Document expected shapes** in function signatures for maintainability

## See Also

- [Shape and Dimensions](shape_dimensions.md) — Understanding shapes
- [Reshaping and View Operations](reshaping_view.md) — Preparing for broadcast
- [Linear Algebra Operations](linalg_operations.md) — Matrix operations
