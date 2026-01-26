# Pandas Interoperability

## Overview

Pandas DataFrames are ubiquitous in data science workflows. Converting between Pandas and PyTorch is essential for data preprocessing pipelines, especially when working with tabular financial data.

## Converting DataFrame to Tensor

### Basic Conversion

```python
import pandas as pd
import torch
import numpy as np

# Create a sample DataFrame
df = pd.DataFrame({
    'feature_1': [1.0, 2.0, 3.0, 4.0],
    'feature_2': [5.0, 6.0, 7.0, 8.0],
    'feature_3': [9.0, 10.0, 11.0, 12.0]
})

print("DataFrame:")
print(df)

# Convert to tensor via NumPy
tensor = torch.tensor(df.values)
print(f"\nTensor shape: {tensor.shape}")
print(f"Tensor dtype: {tensor.dtype}")
print(tensor)
```

### Handling Data Types

```python
# Mixed dtypes require attention
df_mixed = pd.DataFrame({
    'int_col': [1, 2, 3],
    'float_col': [1.5, 2.5, 3.5],
    'str_col': ['a', 'b', 'c']  # Can't convert to tensor!
})

# Select only numeric columns
df_numeric = df_mixed.select_dtypes(include=[np.number])
tensor_numeric = torch.tensor(df_numeric.values)
print(f"Numeric tensor: {tensor_numeric}")

# Or specify columns explicitly
tensor_specific = torch.tensor(df_mixed[['int_col', 'float_col']].values)
```

### Explicit dtype Control

```python
df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})

# Default: inherits dtype from DataFrame
tensor_default = torch.tensor(df.values)
print(f"Default dtype: {tensor_default.dtype}")  # torch.int64

# Force float32 (common for deep learning)
tensor_float32 = torch.tensor(df.values, dtype=torch.float32)
print(f"Float32 dtype: {tensor_float32.dtype}")

# Or convert DataFrame first
tensor_via_astype = torch.tensor(df.astype(np.float32).values)
```

## Converting Series to Tensor

```python
# Pandas Series
series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name='values')

# Convert to 1D tensor
tensor_1d = torch.tensor(series.values)
print(f"1D tensor: {tensor_1d.shape}")  # torch.Size([5])

# Convert to column vector (2D)
tensor_col = torch.tensor(series.values).unsqueeze(1)
print(f"Column tensor: {tensor_col.shape}")  # torch.Size([5, 1])
```

## Handling Missing Values

### Detecting NaN

```python
df_with_nan = pd.DataFrame({
    'a': [1.0, 2.0, np.nan, 4.0],
    'b': [5.0, np.nan, 7.0, 8.0]
})

tensor = torch.tensor(df_with_nan.values)
print(f"Contains NaN: {torch.isnan(tensor).any()}")

# Count NaN per column
nan_counts = torch.isnan(tensor).sum(dim=0)
print(f"NaN counts per column: {nan_counts}")
```

### Strategies for Missing Values

```python
# Strategy 1: Drop rows with NaN
df_clean = df_with_nan.dropna()
tensor_clean = torch.tensor(df_clean.values)
print(f"After dropna: {tensor_clean.shape}")

# Strategy 2: Fill with value
df_filled = df_with_nan.fillna(0)
tensor_filled = torch.tensor(df_filled.values)
print(f"After fillna(0):\n{tensor_filled}")

# Strategy 3: Fill with column mean
df_mean_filled = df_with_nan.fillna(df_with_nan.mean())
tensor_mean_filled = torch.tensor(df_mean_filled.values)
print(f"After mean fill:\n{tensor_mean_filled}")

# Strategy 4: Forward fill (for time series)
df_ffill = df_with_nan.ffill()
tensor_ffill = torch.tensor(df_ffill.values)
```

## Time Series Data

### DateTime Index

```python
# Time series DataFrame
dates = pd.date_range('2023-01-01', periods=5, freq='D')
df_ts = pd.DataFrame({
    'close': [100.0, 101.5, 99.8, 102.3, 103.1],
    'volume': [1000, 1200, 800, 1500, 1100]
}, index=dates)

# Convert values (lose datetime index)
tensor_values = torch.tensor(df_ts.values, dtype=torch.float32)

# Preserve datetime as ordinal or timestamp
timestamps = torch.tensor(df_ts.index.astype(np.int64).values)
print(f"Timestamps (nanoseconds): {timestamps}")

# Convert to days since epoch
days = (df_ts.index - pd.Timestamp('1970-01-01')).days
tensor_days = torch.tensor(days.values)
```

### Sequence Formatting for RNNs

```python
def create_sequences(df, seq_length):
    """Create sequences for time series modeling."""
    values = torch.tensor(df.values, dtype=torch.float32)
    sequences = []
    
    for i in range(len(values) - seq_length):
        seq = values[i:i + seq_length]
        sequences.append(seq)
    
    return torch.stack(sequences)

# Example
df_long = pd.DataFrame({
    'feature': np.random.randn(100)
})

sequences = create_sequences(df_long, seq_length=10)
print(f"Sequences shape: {sequences.shape}")  # torch.Size([90, 10, 1])
```

## Categorical Data

### Label Encoding

```python
df_cat = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B'],
    'value': [1.0, 2.0, 3.0, 4.0, 5.0]
})

# Convert category to numeric codes
df_cat['category_code'] = df_cat['category'].astype('category').cat.codes
tensor = torch.tensor(df_cat[['category_code', 'value']].values)
print(f"With encoded category:\n{tensor}")

# Store mapping for later use
category_mapping = dict(enumerate(df_cat['category'].astype('category').cat.categories))
print(f"Mapping: {category_mapping}")
```

### One-Hot Encoding

```python
# One-hot encode categorical column
one_hot = pd.get_dummies(df_cat['category'], prefix='cat')
df_encoded = pd.concat([one_hot, df_cat[['value']]], axis=1)

tensor_onehot = torch.tensor(df_encoded.values, dtype=torch.float32)
print(f"One-hot encoded shape: {tensor_onehot.shape}")
print(tensor_onehot)
```

## Converting Tensor Back to DataFrame

```python
# Tensor to DataFrame
tensor = torch.randn(5, 3)
column_names = ['feature_a', 'feature_b', 'feature_c']

# Must convert to NumPy first
df_from_tensor = pd.DataFrame(
    tensor.numpy(),  # or tensor.detach().cpu().numpy() for GPU tensors
    columns=column_names
)
print(df_from_tensor)
```

### Handling GPU Tensors

```python
if torch.cuda.is_available():
    tensor_gpu = torch.randn(5, 3, device='cuda')
    
    # Must move to CPU before converting
    df = pd.DataFrame(tensor_gpu.cpu().numpy())
    print(df)
```

### Handling Gradient-Tracking Tensors

```python
tensor_grad = torch.randn(5, 3, requires_grad=True)

# Must detach from computation graph
df = pd.DataFrame(tensor_grad.detach().numpy())
print(df)
```

## Practical Financial Data Example

```python
# Simulated OHLCV data
def prepare_ohlcv_data(df):
    """Prepare OHLCV DataFrame for deep learning."""
    
    # Ensure numeric types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[numeric_cols].astype(np.float32)
    
    # Handle missing values
    df = df.ffill().bfill()
    
    # Normalize (example: min-max scaling)
    df_normalized = (df - df.min()) / (df.max() - df.min())
    
    # Convert to tensor
    tensor = torch.tensor(df_normalized.values)
    
    return tensor

# Example usage
df_ohlcv = pd.DataFrame({
    'open': [100.0, 101.0, 102.0, 101.5, 103.0],
    'high': [101.5, 102.5, 103.0, 102.0, 104.0],
    'low': [99.5, 100.5, 101.5, 100.5, 102.5],
    'close': [101.0, 102.0, 101.5, 103.0, 103.5],
    'volume': [1000, 1200, 800, 1500, 1100]
})

tensor_ohlcv = prepare_ohlcv_data(df_ohlcv)
print(f"OHLCV tensor shape: {tensor_ohlcv.shape}")
print(f"OHLCV tensor dtype: {tensor_ohlcv.dtype}")
```

## Best Practices

### Memory Efficiency

```python
# For large DataFrames, be mindful of memory

# Check memory usage
df_large = pd.DataFrame(np.random.randn(100000, 100))
print(f"DataFrame memory: {df_large.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Convert efficiently using float32
tensor = torch.tensor(df_large.values.astype(np.float32))
print(f"Tensor memory: {tensor.numel() * 4 / 1e6:.1f} MB")  # 4 bytes per float32
```

### Type Consistency

```python
def df_to_tensor(df, dtype=torch.float32):
    """Safely convert DataFrame to tensor."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns found")
    
    # Handle NaN
    if numeric_df.isna().any().any():
        print("Warning: DataFrame contains NaN values")
        numeric_df = numeric_df.fillna(0)
    
    return torch.tensor(numeric_df.values, dtype=dtype)
```

## Summary

| Operation | Method |
|-----------|--------|
| DataFrame → Tensor | `torch.tensor(df.values)` |
| Series → Tensor | `torch.tensor(series.values)` |
| Tensor → DataFrame | `pd.DataFrame(tensor.numpy())` |
| GPU Tensor → DataFrame | `pd.DataFrame(tensor.cpu().numpy())` |
| Grad Tensor → DataFrame | `pd.DataFrame(tensor.detach().numpy())` |
| Handle NaN | `df.fillna()` or `df.dropna()` before conversion |
| Force dtype | `torch.tensor(df.values, dtype=torch.float32)` |

## See Also

- [NumPy Interoperability](numpy_interop.md) - NumPy conversion details
- [Tensor Creation and dtypes](tensor_creation_dtypes.md) - Data type options
- [Memory Layout and Strides](memory_layout_strides.md) - Memory efficiency
