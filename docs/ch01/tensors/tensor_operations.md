# Tensor Operations

PyTorch supports comprehensive operations on tensors spanning arithmetic, mathematical functions, reductions, statistics, and linear algebra. This section covers the full operational toolkit.

## Basic Arithmetic

### Element-wise Operations

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([5.0, 6.0, 7.0, 8.0])

# Addition
add_result = a + b              # tensor([6., 8., 10., 12.])

# Subtraction
sub_result = a - b              # tensor([-4., -4., -4., -4.])

# Multiplication (element-wise, NOT matrix)
mul_result = a * b              # tensor([5., 12., 21., 32.])

# Division
div_result = a / b              # tensor([0.2000, 0.3333, 0.4286, 0.5000])
```

### Scalar Operations

```python
t = torch.tensor([1.0, 2.0, 3.0])

print(t + 10)   # tensor([11., 12., 13.])
print(t * 3)    # tensor([3., 6., 9.])
print(t / 2)    # tensor([0.5, 1.0, 1.5])
print(t ** 2)   # tensor([1., 4., 9.])
```

### Functional Interface

Every operator has a corresponding function and method form:

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Equivalent forms
print(a + b)                    # Operator
print(torch.add(a, b))          # Function
print(a.add(b))                 # Method

# Functions allow output tensor specification
out = torch.empty(3)
torch.add(a, b, out=out)        # Store result in pre-allocated tensor
```

## Mathematical Functions

### Power and Roots

```python
t = torch.tensor([1.0, 4.0, 9.0, 16.0])

print(torch.pow(t, 2))          # tensor([1., 16., 81., 256.])
print(torch.sqrt(t))            # tensor([1., 2., 3., 4.])
print(torch.pow(t, 1/3))        # Cube root
```

### Exponential and Logarithm

```python
t = torch.tensor([1.0, 2.0, 3.0])

print(torch.exp(t))             # e^t: tensor([2.7183, 7.3891, 20.0855])
print(torch.log(t))             # ln(t): tensor([0., 0.6931, 1.0986])
print(torch.log10(t))           # Log base 10
print(torch.log2(t))            # Log base 2
print(torch.log1p(t))           # ln(1 + t) — numerically stable near 0
```

### Trigonometric Functions

```python
t = torch.tensor([0.0, torch.pi/6, torch.pi/4, torch.pi/3, torch.pi/2])

print(torch.sin(t))             # tensor([0., 0.5, 0.7071, 0.8660, 1.])
print(torch.cos(t))             # tensor([1., 0.8660, 0.7071, 0.5, 0.])
print(torch.tan(t))             # tensor([0., 0.5774, 1., 1.7321, inf])

# Inverse trig
x = torch.tensor([0.0, 0.5, 1.0])
print(torch.asin(x))            # arcsin
print(torch.acos(x))            # arccos
print(torch.atan(x))            # arctan

# Two-argument arctan (preserves quadrant)
y = torch.tensor([1.0, 1.0, -1.0, -1.0])
x = torch.tensor([1.0, -1.0, -1.0, 1.0])
print(torch.atan2(y, x))
```

### Hyperbolic Functions

```python
t = torch.tensor([0.0, 1.0, 2.0])

print(torch.sinh(t))            # Hyperbolic sine
print(torch.cosh(t))            # Hyperbolic cosine
print(torch.tanh(t))            # Hyperbolic tangent (common activation)
```

## Rounding, Sign, and Clipping

### Rounding Operations

```python
t = torch.tensor([-1.7, -0.5, 0.3, 1.5, 2.8])

print(torch.round(t))           # tensor([-2., -0., 0., 2., 3.])
print(torch.floor(t))           # tensor([-2., -1., 0., 1., 2.])
print(torch.ceil(t))            # tensor([-1., -0., 1., 2., 3.])
print(torch.trunc(t))           # tensor([-1., -0., 0., 1., 2.])
print(torch.frac(t))            # Fractional part
```

### Sign and Absolute Value

```python
t = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

print(torch.abs(t))             # tensor([3., 1., 0., 1., 3.])
print(torch.sign(t))            # tensor([-1., -1., 0., 1., 1.])
print(torch.neg(t))             # tensor([3., 1., -0., -1., -3.])
```

### Clipping and Clamping

```python
t = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])

# Clamp to range
clamped = torch.clamp(t, min=-2.0, max=2.0)   # tensor([-2., -1., 0., 1., 2.])

# Clamp minimum only (ReLU-like)
print(torch.clamp(t, min=0))                   # tensor([0., 0., 0., 1., 5.])
```

## Division Variants

```python
a = torch.tensor([7.0, 8.0, 9.0])
b = torch.tensor([2.0, 3.0, 4.0])

print(a / b)                    # True division: tensor([3.5, 2.6667, 2.25])
print(a // b)                   # Floor division: tensor([3., 2., 2.])
print(a % b)                    # Modulo: tensor([1., 2., 1.])
print(torch.fmod(a, b))         # C-style remainder
print(torch.remainder(a, b))    # Python-style remainder
```

## Special Functions

```python
# Error function (Gaussian CDF related)
t = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(torch.erf(t))

# Reciprocal and reciprocal square root
t = torch.tensor([1.0, 2.0, 4.0, 8.0])
print(torch.reciprocal(t))      # 1/x
print(torch.rsqrt(t))           # 1/√x

# Log-gamma (for generalized factorial)
t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(torch.lgamma(t))
```

## Reduction Operations

Reductions aggregate tensor values along specified dimensions, producing outputs with fewer dimensions.

### The `dim` Parameter

The `dim` parameter specifies which dimension to collapse:

```python
x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])

# Sum along dim 0 (collapse rows, keep columns)
print(x.sum(dim=0))  # tensor([12., 15., 18.])

# Sum along dim 1 (collapse columns, keep rows)
print(x.sum(dim=1))  # tensor([6., 15., 24.])
```

**Mental model**: the dimension you specify is the one that disappears.

### The `keepdim` Parameter

`keepdim=True` preserves reduced dimensions as size 1—essential for broadcasting:

```python
x = torch.randn(3, 4, 5)

# Without keepdim: dimension removed
mean = x.mean(dim=1)
print(mean.shape)  # torch.Size([3, 5])

# With keepdim: dimension retained as size 1
mean_keep = x.mean(dim=1, keepdim=True)
print(mean_keep.shape)  # torch.Size([3, 1, 5])

# Essential for broadcasting operations
normalized = x - mean_keep  # Broadcasts correctly
```

**Rule of thumb**: always use `keepdim=True` when the result will participate in subsequent broadcasting with the original tensor.

### Sum, Mean, Product

```python
t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(torch.sum(t))              # tensor(21.)
print(torch.mean(t))             # tensor(3.5)
print(torch.prod(t))             # tensor(720.)

# Along dimensions
print(torch.sum(t, dim=0))       # tensor([5., 7., 9.])
print(torch.mean(t, dim=1))      # tensor([2., 5.])
```

### Multiple Dimensions

```python
x = torch.randn(2, 3, 4, 5)

# Reduce over spatial dimensions
spatial_mean = x.mean(dim=(2, 3))
print(spatial_mean.shape)  # torch.Size([2, 3])

# Common in batch normalization: reduce over batch and spatial
channel_stats = x.mean(dim=(0, 2, 3), keepdim=True)
print(channel_stats.shape)  # torch.Size([1, 3, 1, 1])
```

### Min and Max

When `dim` is specified, `min`/`max` return both values AND indices:

```python
t = torch.tensor([[3.0, 1.0, 4.0],
                  [1.0, 5.0, 9.0]])

# Min along dim 1 (per row)
min_vals, min_idx = torch.min(t, dim=1)
print(f"Min values: {min_vals}")   # tensor([1., 1.])
print(f"Min indices: {min_idx}")   # tensor([1, 0])

# argmin/argmax return indices only
print(torch.argmax(t, dim=1))     # tensor([2, 2])
```

### Element-wise Min/Max Between Tensors

```python
a = torch.tensor([1.0, 5.0, 3.0])
b = torch.tensor([2.0, 3.0, 4.0])

print(torch.minimum(a, b))       # tensor([1., 3., 3.])
print(torch.maximum(a, b))       # tensor([2., 5., 4.])
```

### Top-K

```python
t = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])

values, indices = torch.topk(t, k=3)
print(f"Top 3 values: {values}")    # tensor([9., 6., 5.])
print(f"Top 3 indices: {indices}")  # tensor([5, 7, 4])

# Bottom K
values, indices = torch.topk(t, k=3, largest=False)
```

### Cumulative Operations

Cumulative operations preserve shape—they do not reduce dimensions:

```python
t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

print(torch.cumsum(t, dim=0))   # tensor([1., 3., 6., 10., 15.])
print(torch.cumprod(t, dim=0))  # tensor([1., 2., 6., 24., 120.])

# Cumulative max returns values and indices
cummax_vals, cummax_idx = torch.cummax(t, dim=0)
```

### Logical Reductions

```python
x = torch.tensor([True, True, False, True])

print(x.all())  # tensor(False) — not all True
print(x.any())  # tensor(True)  — at least one True

# Practical: check if all values meet a condition
values = torch.randn(5)
print((values > 0).all())
print((values < 0).any())
```

### Counting and Unique Values

```python
x = torch.tensor([[1, 0, 3], [0, 0, 6], [7, 8, 0]])
print(torch.count_nonzero(x))          # tensor(5)
print(torch.count_nonzero(x, dim=0))   # tensor([2, 1, 2])

# Unique values with counts
x = torch.tensor([1, 2, 2, 3, 3, 3, 4])
unique_vals, counts = torch.unique(x, return_counts=True)
```

### Norms

```python
t = torch.tensor([3.0, 4.0])

print(torch.norm(t, p=2))              # L2: tensor(5.) = √(9+16)
print(torch.norm(t, p=1))              # L1: tensor(7.) = 3+4
print(torch.norm(t, p=float('inf')))   # L∞: tensor(4.)

# Per-row norms
x = torch.randn(32, 128)
row_norms = torch.norm(x, p=2, dim=1)  # Shape: (32,)
```

## Statistics Operations

### Variance and Standard Deviation

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

std = x.std()    # tensor(1.5811) — Bessel-corrected (N-1)
var = x.var()    # tensor(2.5)

# Biased (population): divides by N
std_biased = x.std(correction=0)     # tensor(1.4142)
var_biased = x.var(correction=0)     # tensor(2.0)
```

When to use which: `correction=1` (default) when estimating population parameters from a sample. `correction=0` when you have the entire population or for neural network normalization layers (BatchNorm, LayerNorm).

### Dimension-wise Statistics

```python
data = torch.randn(100, 10)  # 100 samples, 10 features

# Per-feature statistics (reduce batch dimension)
mean_per_feature = data.mean(dim=0)    # shape: [10]
std_per_feature = data.std(dim=0)      # shape: [10]

# Per-sample statistics (reduce feature dimension)
mean_per_sample = data.mean(dim=1)     # shape: [100]
```

### Weighted Mean

```python
values = torch.tensor([1.0, 2.0, 3.0, 4.0])
weights = torch.tensor([0.1, 0.2, 0.3, 0.4])

weighted_mean = (values * weights).sum() / weights.sum()  # tensor(3.0)
```

### Median and Mode

```python
x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
print(x.median())  # tensor(3.)

# Along dimension (returns value and index)
x2d = torch.tensor([[1., 5., 3.],
                    [2., 4., 6.]])
med_vals, med_idx = x2d.median(dim=1)

# Mode (most frequent, for discrete data)
x = torch.tensor([1, 2, 2, 3, 3, 3, 4])
mode_val, mode_idx = x.mode()  # tensor(3), tensor(3)
```

### Quantiles

```python
x = torch.randn(1000)

q25 = x.quantile(0.25)   # 25th percentile (Q1)
q50 = x.quantile(0.50)   # Median
q75 = x.quantile(0.75)   # 75th percentile (Q3)

# Multiple quantiles at once
quantiles = x.quantile(torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
```

### Covariance and Correlation

```python
def covariance(X, correction=1):
    """Covariance matrix. X: (n_samples, n_features)."""
    n = X.size(0)
    X_centered = X - X.mean(dim=0, keepdim=True)
    return (X_centered.T @ X_centered) / (n - correction)

def pearson_correlation(x, y):
    """Pearson correlation between two vectors."""
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    return numerator / denominator

X = torch.randn(100, 5)
cov = covariance(X)  # Shape: (5, 5)
```

### Histograms

```python
x = torch.randn(10000)

# Histogram with bin edges
hist, bin_edges = torch.histogram(x, bins=50)

# Integer counting
x_int = torch.tensor([0, 1, 1, 2, 2, 2, 3])
counts = torch.bincount(x_int)  # tensor([1, 2, 3, 1])
```

### Normalization Functions

```python
# Z-score normalization
def z_score_normalize(x, dim=None):
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + 1e-8)

# Min-max normalization
def min_max_normalize(x, dim=None):
    if dim is None:
        x_min, x_max = x.min(), x.max()
    else:
        x_min = x.min(dim=dim, keepdim=True).values
        x_max = x.max(dim=dim, keepdim=True).values
    return (x - x_min) / (x_max - x_min + 1e-8)
```

### Batch and Layer Normalization Statistics

```python
# Batch normalization: per-channel stats across batch and spatial dims
images = torch.randn(32, 3, 224, 224)
channel_mean = images.mean(dim=(0, 2, 3), keepdim=True)  # [1, 3, 1, 1]
channel_var = images.var(dim=(0, 2, 3), keepdim=True, correction=0)
normalized = (images - channel_mean) / torch.sqrt(channel_var + 1e-5)

# Layer normalization: per-sample stats across feature dims
features = torch.randn(32, 10, 64)
mean = features.mean(dim=-1, keepdim=True)    # [32, 10, 1]
var = features.var(dim=-1, keepdim=True, correction=0)
```

## Numerical Stability

### Log-Sum-Exp

```python
t = torch.tensor([1000.0, 1001.0, 1002.0])

# Naive: overflow risk
# naive = torch.log(torch.sum(torch.exp(t)))  # inf!

# Stable version
stable = torch.logsumexp(t, dim=0)
```

### Softmax and Log-Softmax

```python
logits = torch.tensor([1.0, 2.0, 3.0, 4.0])

softmax = torch.softmax(logits, dim=0)           # Numerically stable
log_softmax = torch.log_softmax(logits, dim=0)   # More stable for loss
```

## Quick Reference

| Category | Operations |
|----------|------------|
| Basic arithmetic | `+`, `-`, `*`, `/`, `**`, `//`, `%` |
| Math functions | `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh` |
| Rounding | `round`, `floor`, `ceil`, `trunc` |
| Sign/Abs | `abs`, `sign`, `neg` |
| Clipping | `clamp`, `clip` |
| Reductions | `sum`, `mean`, `prod`, `min`, `max` |
| Statistics | `std`, `var`, `median`, `mode`, `quantile` |
| Counting | `argmin`, `argmax`, `topk`, `unique`, `bincount` |
| Cumulative | `cumsum`, `cumprod`, `cummax`, `cummin` |
| Logical | `all`, `any`, `count_nonzero` |
| Norms | `norm` (L1, L2, Frobenius) |
