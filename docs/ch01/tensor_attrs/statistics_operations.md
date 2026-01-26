# Statistics Operations

Statistical operations compute descriptive measures of tensor data. These operations are essential for normalization, data analysis, and understanding model behavior.

## Variance and Standard Deviation

### Basic Computation

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Standard deviation
std = x.std()
print(f"Std: {std}")  # tensor(1.5811)

# Variance
var = x.var()
print(f"Var: {var}")  # tensor(2.5)

# Relationship: var = std²
print(torch.allclose(var, std ** 2))  # True
```

### Bessel's Correction

By default, PyTorch uses Bessel's correction (unbiased estimator, dividing by N-1):

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Unbiased (default): divides by N-1 (sample variance)
std_unbiased = x.std(correction=1)  # or just x.std()
var_unbiased = x.var(correction=1)
print(f"Sample std: {std_unbiased}")   # tensor(1.5811)
print(f"Sample var: {var_unbiased}")   # tensor(2.5)

# Biased: divides by N (population variance)
std_biased = x.std(correction=0)
var_biased = x.var(correction=0)
print(f"Population std: {std_biased}") # tensor(1.4142)
print(f"Population var: {var_biased}") # tensor(2.0)
```

**When to use which:**
- `correction=1` (default): When computing statistics from a sample to estimate population parameters
- `correction=0`: When you have the entire population, or for neural network normalization layers

### Dimension-wise Statistics

```python
data = torch.randn(100, 10)  # 100 samples, 10 features

# Per-feature statistics (reduce batch dimension)
mean_per_feature = data.mean(dim=0)    # shape: [10]
std_per_feature = data.std(dim=0)      # shape: [10]
var_per_feature = data.var(dim=0)      # shape: [10]

# Per-sample statistics (reduce feature dimension)
mean_per_sample = data.mean(dim=1)     # shape: [100]
std_per_sample = data.std(dim=1)       # shape: [100]
```

### Multiple Dimensions

```python
# Image batch: (batch, channels, height, width)
images = torch.randn(32, 3, 224, 224)

# Per-channel statistics across batch and spatial dims
# Common in batch normalization
channel_mean = images.mean(dim=(0, 2, 3))           # shape: [3]
channel_std = images.std(dim=(0, 2, 3))             # shape: [3]

# With keepdim for broadcasting
channel_mean_k = images.mean(dim=(0, 2, 3), keepdim=True)  # shape: [1, 3, 1, 1]
channel_std_k = images.std(dim=(0, 2, 3), keepdim=True)    # shape: [1, 3, 1, 1]
```

## Mean

### Basic Mean

```python
x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

# Global mean
print(x.mean())        # tensor(3.5)

# Along dimensions
print(x.mean(dim=0))   # tensor([2.5, 3.5, 4.5]) - column means
print(x.mean(dim=1))   # tensor([2., 5.])        - row means
```

### Weighted Mean

```python
values = torch.tensor([1.0, 2.0, 3.0, 4.0])
weights = torch.tensor([0.1, 0.2, 0.3, 0.4])

weighted_mean = (values * weights).sum() / weights.sum()
print(f"Weighted mean: {weighted_mean}")  # tensor(3.0)
```

### Running/Exponential Moving Average

```python
def exponential_moving_average(values, alpha=0.1):
    """
    Compute EMA: ema[t] = alpha * value[t] + (1-alpha) * ema[t-1]
    """
    ema = torch.zeros_like(values)
    ema[0] = values[0]
    for t in range(1, len(values)):
        ema[t] = alpha * values[t] + (1 - alpha) * ema[t-1]
    return ema

values = torch.randn(100)
ema = exponential_moving_average(values, alpha=0.1)
```

## Median and Mode

### Median

```python
x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])

median = x.median()
print(f"Median: {median}")  # tensor(3.)

# Along dimension (returns value and index)
x2d = torch.tensor([[1., 5., 3.],
                    [2., 4., 6.]])

med_vals, med_idx = x2d.median(dim=1)
print(f"Row medians: {med_vals}")    # tensor([3., 4.])
print(f"Median indices: {med_idx}")  # tensor([2, 1])
```

### Mode

Most frequent value (for discrete data):

```python
x = torch.tensor([1, 2, 2, 3, 3, 3, 4])

mode_val, mode_idx = x.mode()
print(f"Mode: {mode_val}")          # tensor(3)
print(f"First occurrence: {mode_idx}")  # tensor(3)

# Along dimension
x2d = torch.tensor([[1, 2, 2],
                    [3, 3, 4]])

mode_vals, mode_idx = x2d.mode(dim=1)
print(f"Row modes: {mode_vals}")    # tensor([2, 3])
```

## Quantiles and Percentiles

### Single Quantile

```python
x = torch.randn(1000)

# Specific quantiles
q25 = x.quantile(0.25)   # 25th percentile (Q1)
q50 = x.quantile(0.50)   # 50th percentile (median)
q75 = x.quantile(0.75)   # 75th percentile (Q3)

print(f"Q1: {q25:.3f}, Median: {q50:.3f}, Q3: {q75:.3f}")
```

### Multiple Quantiles

```python
x = torch.randn(1000)

# Multiple quantiles at once
quantile_points = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
quantiles = x.quantile(quantile_points)

print(f"Deciles and quartiles: {quantiles}")
```

### Along Dimensions

```python
data = torch.randn(100, 10)  # 100 samples, 10 features

# Per-feature quartiles
q_points = torch.tensor([0.25, 0.5, 0.75])
feature_quartiles = torch.quantile(data, q_points, dim=0)
print(feature_quartiles.shape)  # torch.Size([3, 10])
```

### Interquartile Range (IQR)

```python
def iqr(x, dim=None):
    """Compute interquartile range."""
    q25 = x.quantile(0.25, dim=dim)
    q75 = x.quantile(0.75, dim=dim)
    return q75 - q25

x = torch.randn(1000)
print(f"IQR: {iqr(x):.3f}")
```

## Covariance and Correlation

### Covariance Matrix

```python
def covariance(X, correction=1):
    """
    Compute covariance matrix.
    Args:
        X: (n_samples, n_features)
        correction: 0 for population, 1 for sample (default)
    Returns:
        (n_features, n_features) covariance matrix
    """
    n = X.size(0)
    X_centered = X - X.mean(dim=0, keepdim=True)
    return (X_centered.T @ X_centered) / (n - correction)

X = torch.randn(100, 5)
cov = covariance(X)
print(cov.shape)  # torch.Size([5, 5])
```

### Correlation Matrix

```python
def correlation(X):
    """
    Compute Pearson correlation matrix.
    Args:
        X: (n_samples, n_features)
    Returns:
        (n_features, n_features) correlation matrix
    """
    # Center and normalize
    X_centered = X - X.mean(dim=0, keepdim=True)
    X_normalized = X_centered / X_centered.std(dim=0, keepdim=True)
    
    n = X.size(0)
    return (X_normalized.T @ X_normalized) / (n - 1)

X = torch.randn(100, 5)
corr = correlation(X)
print(corr.shape)  # torch.Size([5, 5])

# Diagonal should be all 1s
print(torch.diag(corr))  # tensor([1., 1., 1., 1., 1.])
```

### Pairwise Correlation

```python
def pearson_correlation(x, y):
    """Compute Pearson correlation between two vectors."""
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    
    return numerator / denominator

x = torch.randn(100)
y = 0.8 * x + 0.2 * torch.randn(100)  # Correlated
print(f"Correlation: {pearson_correlation(x, y):.3f}")  # ~0.97
```

## Histograms and Binning

### Histogram

```python
x = torch.randn(10000)

# Simple histogram
hist = torch.histc(x, bins=50, min=-3, max=3)
print(hist.shape)  # torch.Size([50])

# Histogram with bin edges
hist, bin_edges = torch.histogram(x, bins=50)
print(f"Counts shape: {hist.shape}")        # torch.Size([50])
print(f"Bin edges shape: {bin_edges.shape}")  # torch.Size([51])
```

### Bincount

For counting integer occurrences:

```python
x = torch.tensor([0, 1, 1, 2, 2, 2, 3])

counts = torch.bincount(x)
print(counts)  # tensor([1, 2, 3, 1])

# With weights
weights = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
weighted_counts = torch.bincount(x, weights=weights)
print(weighted_counts)  # tensor([0.5, 2.5, 7.5, 3.5])
```

## Normalization Techniques

### Z-Score Normalization (Standardization)

```python
def z_score_normalize(x, dim=None, keepdim=False):
    """Normalize to zero mean and unit variance."""
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    normalized = (x - mean) / (std + 1e-8)
    
    if not keepdim and dim is not None:
        return normalized.squeeze(dim)
    return normalized

x = torch.randn(32, 10)
x_normalized = z_score_normalize(x, dim=1)
print(f"Mean: {x_normalized.mean(dim=1)[:5]}")  # ~0
print(f"Std: {x_normalized.std(dim=1)[:5]}")    # ~1
```

### Min-Max Normalization

```python
def min_max_normalize(x, dim=None, new_min=0, new_max=1):
    """Scale to [new_min, new_max] range."""
    if dim is None:
        x_min, x_max = x.min(), x.max()
    else:
        x_min = x.min(dim=dim, keepdim=True).values
        x_max = x.max(dim=dim, keepdim=True).values
    
    normalized = (x - x_min) / (x_max - x_min + 1e-8)
    return normalized * (new_max - new_min) + new_min

x = torch.randn(32, 10)
x_scaled = min_max_normalize(x, dim=1)
print(f"Min: {x_scaled.min(dim=1).values[:5]}")  # All 0
print(f"Max: {x_scaled.max(dim=1).values[:5]}")  # All 1
```

### Layer Normalization Statistics

```python
def layer_norm_stats(x, normalized_shape):
    """
    Compute layer normalization statistics.
    Args:
        x: Input tensor
        normalized_shape: Dimensions to normalize over (from the end)
    """
    # Determine which dims to normalize
    dims = tuple(range(-len(normalized_shape), 0))
    
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, keepdim=True, correction=0)
    
    return mean, var

x = torch.randn(32, 10, 64)  # (batch, seq, features)
mean, var = layer_norm_stats(x, normalized_shape=[64])
print(f"Mean shape: {mean.shape}")  # torch.Size([32, 10, 1])
```

### Batch Normalization Statistics

```python
def batch_norm_stats(x):
    """
    Compute batch normalization statistics for 4D input.
    Args:
        x: (batch, channels, height, width)
    Returns:
        Per-channel mean and variance
    """
    # Reduce over batch and spatial dimensions
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = x.var(dim=(0, 2, 3), keepdim=True, correction=0)
    
    return mean, var

features = torch.randn(32, 64, 28, 28)
mean, var = batch_norm_stats(features)
print(f"Mean shape: {mean.shape}")  # torch.Size([1, 64, 1, 1])
print(f"Var shape: {var.shape}")    # torch.Size([1, 64, 1, 1])

# Normalize
normalized = (features - mean) / torch.sqrt(var + 1e-5)
```

## Statistical Summaries

### Five-Number Summary

```python
def five_number_summary(x):
    """Return min, Q1, median, Q3, max."""
    return {
        'min': x.min().item(),
        'Q1': x.quantile(0.25).item(),
        'median': x.quantile(0.5).item(),
        'Q3': x.quantile(0.75).item(),
        'max': x.max().item()
    }

x = torch.randn(1000)
summary = five_number_summary(x)
print(summary)
```

### Comprehensive Summary

```python
def describe(tensor, name="Tensor"):
    """Print comprehensive statistics for a tensor."""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min: {tensor.min().item():.4f}")
    print(f"  Max: {tensor.max().item():.4f}")
    print(f"  Mean: {tensor.mean().item():.4f}")
    print(f"  Std: {tensor.std().item():.4f}")
    print(f"  Median: {tensor.median().item():.4f}")
    
    # Check for issues
    print(f"  Has NaN: {torch.isnan(tensor).any().item()}")
    print(f"  Has Inf: {torch.isinf(tensor).any().item()}")

x = torch.randn(100, 50)
describe(x, "Random Matrix")
```

### Detecting Outliers

```python
def detect_outliers_iqr(x, factor=1.5):
    """Detect outliers using IQR method."""
    q25 = x.quantile(0.25)
    q75 = x.quantile(0.75)
    iqr = q75 - q25
    
    lower_bound = q25 - factor * iqr
    upper_bound = q75 + factor * iqr
    
    outliers = (x < lower_bound) | (x > upper_bound)
    return outliers, lower_bound, upper_bound

x = torch.randn(1000)
outliers, low, high = detect_outliers_iqr(x)
print(f"Outliers: {outliers.sum().item()} ({outliers.float().mean()*100:.1f}%)")
```

### Distribution Checks

```python
def check_normality(x):
    """Basic normality checks (skewness and kurtosis)."""
    mean = x.mean()
    std = x.std()
    x_standardized = (x - mean) / std
    
    # Skewness (should be ~0 for normal)
    skewness = (x_standardized ** 3).mean()
    
    # Excess kurtosis (should be ~0 for normal, normal has kurtosis=3)
    kurtosis = (x_standardized ** 4).mean() - 3
    
    return {
        'skewness': skewness.item(),
        'excess_kurtosis': kurtosis.item(),
        'approximately_normal': abs(skewness) < 0.5 and abs(kurtosis) < 1
    }

x = torch.randn(10000)
result = check_normality(x)
print(result)
```

## Quick Reference

| Operation | Function | Notes |
|-----------|----------|-------|
| Mean | `torch.mean(x, dim)` | Arithmetic average |
| Variance | `torch.var(x, dim)` | Use `correction` param |
| Std | `torch.std(x, dim)` | Use `correction` param |
| Median | `torch.median(x, dim)` | Middle value |
| Mode | `torch.mode(x, dim)` | Most frequent |
| Quantile | `torch.quantile(x, q, dim)` | q in [0, 1] |
| Histogram | `torch.histogram(x, bins)` | Counts and edges |
| Histc | `torch.histc(x, bins)` | Counts only |
| Bincount | `torch.bincount(x)` | Integer counts |

## Key Takeaways

1. **Bessel's correction**: Use `correction=1` (default) for sample statistics, `correction=0` for population or neural network normalization
2. **Dimension matters**: Understand which dimension to reduce for per-feature vs per-sample statistics
3. **Use `keepdim=True`** when normalizing to enable broadcasting
4. **Covariance vs Correlation**: Correlation is scale-invariant, covariance is not
5. **Quantiles are robust**: Less sensitive to outliers than mean/std
6. **Check for NaN/Inf** in statistics—they propagate and can indicate numerical issues

## See Also

- [Reduction Operations](reduction_operations.md) - Sum, min, max, cumulative ops
- [Broadcasting Rules](broadcasting_rules.md) - Shape compatibility
- [Linear Algebra Operations](linalg_operations.md) - Matrix operations for covariance
