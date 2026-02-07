# NumPy Interoperability

## Overview

PyTorch and NumPy are designed to work seamlessly together. Understanding the different conversion methods—and critically, which ones share memory versus copy data—is essential for efficient workflows. This page also covers Pandas interoperability, since Pandas-to-PyTorch conversion goes through NumPy.

---

## NumPy → PyTorch: The Three Methods

| Method | Memory Behavior | When to Use |
|--------|-----------------|-------------|
| `torch.from_numpy()` | **SHARE** (no copy) | Zero-copy when possible |
| `torch.as_tensor()` | **TRY TO SHARE** | Best-effort sharing |
| `torch.tensor()` | **COPY** (always) | Safe, independent tensor |

### Decision Flowchart

```
NumPy → PyTorch:
    ├─ Want safety, no surprises?       → torch.tensor()      [COPY]
    ├─ Want zero-copy, understand risks? → torch.from_numpy()  [SHARE]
    └─ Want auto-decide?                → torch.as_tensor()   [TRY SHARE]

PyTorch → NumPy:
    ├─ On CPU, no gradients? → tensor.numpy()               [SHARE]
    └─ GPU or has gradients? → tensor.detach().cpu().numpy() [COPY]
```

---

## `torch.from_numpy()` — Zero-Copy Sharing

Creates a PyTorch tensor that **shares memory** with the NumPy array. Mutations propagate both ways:

```python
import numpy as np
import torch

arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
t = torch.from_numpy(arr)

# Mutations propagate both ways
arr[0] = 99.0
print(t)  # tensor([99., 2., 3.])

t[1] = -7.0
print(arr)  # [99. -7.  3.]
```

### Verifying Shared Memory

```python
def ptr_numpy(a):
    """Return the base data pointer of a NumPy array (as a Python int)."""
    return a.__array_interface__['data'][0]

def ptr_torch(t):
    """Return the base data pointer of a PyTorch tensor's storage."""
    return t.untyped_storage().data_ptr()

arr = np.array([1, 2, 3], dtype=np.float32)
t = torch.from_numpy(arr)

print(f"NumPy pointer:  {ptr_numpy(arr)}")
print(f"PyTorch pointer: {ptr_torch(t)}")
print(f"Same memory: {ptr_numpy(arr) == ptr_torch(t)}")  # True
```

### Requirements for `from_numpy()`

`torch.from_numpy()` requires:

1. **Numeric dtype** (not object arrays)
2. **Writable array** (not read-only)
3. **Compatible strides** (positive strides only; negative strides like `arr[::-1]` are not supported)

You cannot pass `dtype` or `device` arguments to `from_numpy()`. The tensor's dtype is derived from the ndarray, and the device is always CPU:

```python
arr64 = np.array([1.0, 2.0], dtype=np.float64)

# Cannot do: torch.from_numpy(arr64, dtype=torch.float32)  # TypeError!

# Convert the NumPy array first
arr32 = arr64.astype(np.float32, copy=True)  # New array
t = torch.from_numpy(arr32)                   # SHARE with arr32 (not arr64)

# Need GPU? Move after creating
t_cpu = torch.from_numpy(arr32)   # SHARE on CPU
t_gpu = t_cpu.to('cuda')          # COPY to GPU (sharing broken)
```

### Fortran-Ordered Arrays

If the ndarray is Fortran-ordered (column-major), `from_numpy()` **still shares** memory. The resulting tensor will have column-major-style strides and is typically non-contiguous in PyTorch's row-major sense:

```python
arr_2d = np.arange(6, dtype=np.float32).reshape(2, 3)
arr_f = np.asfortranarray(arr_2d)

t_f = torch.from_numpy(arr_f)   # SHARE — but non-contiguous

print(f"t_f.is_contiguous(): {t_f.is_contiguous()}")  # likely False
print(f"t_f.stride():        {t_f.stride()}")          # column-major strides
```

Many operations work fine on non-contiguous tensors, but operations that require contiguous memory will either internally make a copy or require `.contiguous()` (**COPY**, breaks sharing):

```python
# If you want row-major sharing from the start
arr_c = np.ascontiguousarray(arr_f)   # COPY if needed
t_shared = torch.from_numpy(arr_c)    # SHARE with row-major layout
```

### Read-Only Arrays

```python
ro = np.array([1.0, 2.0, 3.0], dtype=np.float32)
ro.setflags(write=False)

# from_numpy will error on read-only arrays
# Safe fallback: make a writable copy
t = torch.from_numpy(np.array(ro, copy=True))  # COPY
```

!!! warning "Negative Strides"
    Reversed arrays (`arr[::-1]`) and other negative-stride views are not supported
    by `from_numpy()`. Use `torch.as_tensor()` (may **COPY**) or make a contiguous
    copy first.

---

## `torch.as_tensor()` — Best-Effort Sharing

Shares when possible, copies when a dtype or device conversion is needed:

```python
arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)

# Shares (same dtype, same device)
t_shared = torch.as_tensor(arr)

arr[1] = 222.0
print(t_shared)  # Shows 222.0 — shared
```

### Sharing Rules

`as_tensor()` **shares** (no copy) when the ndarray is numeric, writable, and has compatible strides (C-order or Fortran-order).

`as_tensor()` **copies** when:

- The ndarray is read-only
- Strides/layout are unsupported (e.g., negative strides like `arr[::-1]`)
- A dtype change is requested: `as_tensor(ndarray, dtype=...)` may copy
- A device change is requested: `as_tensor(..., device=...)` → copy to that device

### Comparison with `from_numpy()`

The key difference: `from_numpy()` **always shares** (and raises an error if it cannot), while `as_tensor()` silently falls back to copying when sharing is impossible. Additionally, `as_tensor()` accepts `dtype` and `device` arguments:

```python
arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)

# as_tensor can request a different dtype (may trigger COPY)
t_converted = torch.as_tensor(arr, dtype=torch.float32)

# from_numpy cannot — you must convert the array first
arr32 = arr.astype(np.float32)
t_from = torch.from_numpy(arr32)
```

---

## `torch.tensor()` — Safe Copying

Always creates an independent copy:

```python
arr = np.array([10, 20, 30], dtype=np.int64)
t = torch.tensor(arr)

arr[0] = 123
print(t)  # tensor([10, 20, 30])  — independent
```

**Advantages**: safest—no surprises from later NumPy mutations. Can set `dtype` and `device` directly. Works with any input (lists, tuples, scalars, arrays).

---

## PyTorch → NumPy

### `tensor.numpy()` — Shares if Possible

```python
t = torch.tensor([1.0, 2.0, 3.0])
arr = t.numpy()  # SHARES memory

t[0] = 99.0
print(arr)  # [99. 2. 3.]
```

### GPU or Gradient Tensors

Tensors on GPU or with gradient tracking require extra steps:

```python
# GPU tensor
if torch.cuda.is_available():
    t_gpu = torch.tensor([1.0, 2.0], device='cuda')
    arr = t_gpu.cpu().numpy()

# Tensor with gradients
t_grad = torch.tensor([1.0, 2.0], requires_grad=True)
arr = t_grad.detach().numpy()

# Universal pattern (works everywhere)
arr = t.detach().cpu().numpy()
```

---

## Non-Contiguous Arrays and Strided Views

Strided views with positive step still share memory:

```python
base = np.arange(10, dtype=np.float32)  # [0,1,2,...,9]
view = base[::2]                        # [0,2,4,6,8] — non-contiguous

t_view = torch.from_numpy(view)         # SHARE with view (and base)

view[0] = 999.0
print(base)    # Shows 999 at index 0
print(t_view)  # Also shows 999 — all three aliases share memory
```

Negative strides require special handling:

```python
arr = np.arange(5, dtype=np.float32)
reversed_view = arr[::-1]  # negative stride

# from_numpy may fail
# torch.from_numpy(reversed_view)  # RuntimeError

# as_tensor will silently COPY
t = torch.as_tensor(reversed_view)  # COPY (no sharing)
```

---

## Dtype Mappings

| NumPy dtype | PyTorch dtype |
|-------------|---------------|
| `np.float32` | `torch.float32` |
| `np.float64` | `torch.float64` |
| `np.int32` | `torch.int32` |
| `np.int64` | `torch.int64` |
| `np.uint8` | `torch.uint8` |
| `np.bool_` | `torch.bool` |
| `np.complex64` | `torch.complex64` |
| `np.complex128` | `torch.complex128` |

```python
for np_dtype in [np.float32, np.float64, np.int64, np.int32, np.uint8, np.bool_]:
    arr = np.array([0, 1, 2], dtype=np_dtype)
    t = torch.from_numpy(arr)
    print(f"NumPy dtype {arr.dtype.name:>8} → Torch dtype {t.dtype}")
```

!!! warning "Default dtype mismatch"
    NumPy defaults to `float64`, PyTorch defaults to `float32`. Always be explicit
    when converting: use `dtype=np.float32` in NumPy or `dtype=torch.float32` in PyTorch.

### Complex Dtypes

```python
cplx = np.array([1+2j, 3+4j], dtype=np.complex128)

try:
    t_cplx = torch.from_numpy(cplx)
except Exception as e:
    # Workaround: split into real and imaginary parts
    t_real = torch.from_numpy(np.real(cplx).astype(np.float64))
    t_imag = torch.from_numpy(np.imag(cplx).astype(np.float64))
```

---

## Pandas Interoperability

Pandas DataFrames convert to PyTorch tensors via NumPy. This is essential for tabular data preprocessing pipelines, especially with financial data.

### DataFrame → Tensor

```python
import pandas as pd

df = pd.DataFrame({
    'feature_1': [1.0, 2.0, 3.0, 4.0],
    'feature_2': [5.0, 6.0, 7.0, 8.0],
    'feature_3': [9.0, 10.0, 11.0, 12.0]
})

# Convert to tensor via NumPy
tensor = torch.tensor(df.values)
print(f"Shape: {tensor.shape}, Dtype: {tensor.dtype}")
```

### Handling Mixed Types

```python
df_mixed = pd.DataFrame({
    'int_col': [1, 2, 3],
    'float_col': [1.5, 2.5, 3.5],
    'str_col': ['a', 'b', 'c']  # Can't convert to tensor
})

# Select only numeric columns
df_numeric = df_mixed.select_dtypes(include=[np.number])
tensor_numeric = torch.tensor(df_numeric.values)

# Or specify columns explicitly
tensor_specific = torch.tensor(df_mixed[['int_col', 'float_col']].values)
```

### Explicit dtype Control

```python
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# Default: inherits dtype from DataFrame
tensor_default = torch.tensor(df.values)
print(f"Default dtype: {tensor_default.dtype}")  # torch.int64

# Force float32 (common for deep learning)
tensor_float32 = torch.tensor(df.values, dtype=torch.float32)
```

### Series → Tensor

```python
series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

# 1D tensor
tensor_1d = torch.tensor(series.values)  # Shape: (5,)

# Column vector
tensor_col = tensor_1d.unsqueeze(1)      # Shape: (5, 1)
```

### Handling Missing Values

```python
df_nan = pd.DataFrame({
    'a': [1.0, 2.0, np.nan, 4.0],
    'b': [5.0, np.nan, 7.0, 8.0]
})

# Strategy 1: Drop rows
tensor_clean = torch.tensor(df_nan.dropna().values)

# Strategy 2: Fill with value
tensor_filled = torch.tensor(df_nan.fillna(0).values)

# Strategy 3: Fill with column mean
tensor_mean = torch.tensor(df_nan.fillna(df_nan.mean()).values)

# Strategy 4: Forward fill (for time series)
tensor_ffill = torch.tensor(df_nan.ffill().values)
```

### Categorical Data

```python
df_cat = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B'],
    'value': [1.0, 2.0, 3.0, 4.0, 5.0]
})

# Label encoding
df_cat['code'] = df_cat['category'].astype('category').cat.codes
tensor = torch.tensor(df_cat[['code', 'value']].values)

# One-hot encoding
one_hot = pd.get_dummies(df_cat['category'], prefix='cat')
df_encoded = pd.concat([one_hot, df_cat[['value']]], axis=1)
tensor_onehot = torch.tensor(df_encoded.values, dtype=torch.float32)
```

### Tensor → DataFrame

```python
tensor = torch.randn(5, 3)
columns = ['feature_a', 'feature_b', 'feature_c']

# Must convert to NumPy first
df = pd.DataFrame(tensor.numpy(), columns=columns)

# For GPU tensors or tensors with gradients
# df = pd.DataFrame(tensor.detach().cpu().numpy(), columns=columns)
```

### Time Series Data

```python
dates = pd.date_range('2023-01-01', periods=5, freq='D')
df_ts = pd.DataFrame({
    'close': [100.0, 101.5, 99.8, 102.3, 103.1],
    'volume': [1000, 1200, 800, 1500, 1100]
}, index=dates)

# Convert values (datetime index lost)
tensor_values = torch.tensor(df_ts.values, dtype=torch.float32)

# Preserve datetime as days since epoch
days = (df_ts.index - pd.Timestamp('1970-01-01')).days
tensor_days = torch.tensor(days.values)
```

### Sequence Formatting for RNNs

```python
def create_sequences(df, seq_length):
    """Create sliding-window sequences for time series modeling."""
    values = torch.tensor(df.values, dtype=torch.float32)
    sequences = []

    for i in range(len(values) - seq_length):
        seq = values[i:i + seq_length]
        sequences.append(seq)

    return torch.stack(sequences)

# Example
df_long = pd.DataFrame({'feature': np.random.randn(100)})
sequences = create_sequences(df_long, seq_length=10)
print(f"Sequences shape: {sequences.shape}")  # torch.Size([90, 10, 1])
```

---

## Practical Financial Data Example

```python
def prepare_ohlcv_data(df):
    """Prepare OHLCV DataFrame for deep learning."""
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[numeric_cols].astype(np.float32)

    # Handle missing values
    df = df.ffill().bfill()

    # Min-max normalization
    df_normalized = (df - df.min()) / (df.max() - df.min())

    return torch.tensor(df_normalized.values)

df_ohlcv = pd.DataFrame({
    'open': [100.0, 101.0, 102.0, 101.5, 103.0],
    'high': [101.5, 102.5, 103.0, 102.0, 104.0],
    'low': [99.5, 100.5, 101.5, 100.5, 102.5],
    'close': [101.0, 102.0, 101.5, 103.0, 103.5],
    'volume': [1000, 1200, 800, 1500, 1100]
})

tensor_ohlcv = prepare_ohlcv_data(df_ohlcv)
print(f"Shape: {tensor_ohlcv.shape}, Dtype: {tensor_ohlcv.dtype}")
```

---

## Data Loading Pipeline

```python
def load_data():
    """Common pattern: load with NumPy, train with PyTorch."""
    data = np.load('data.npy')
    labels = np.load('labels.npy')

    # Use tensor() for safety in training loops
    X = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    return X, y
```

### Safe Conversion Utility

```python
def df_to_tensor(df, dtype=torch.float32):
    """Safely convert DataFrame to tensor."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("No numeric columns found")

    if numeric_df.isna().any().any():
        print("Warning: DataFrame contains NaN values")
        numeric_df = numeric_df.fillna(0)

    return torch.tensor(numeric_df.values, dtype=dtype)
```

---

## Common Pitfalls

### Pitfall 1: Unexpected Mutations from Shared Memory

```python
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)

arr[0] = 0  # Oops — tensor also changed
```

**Solution**: Use `torch.tensor()` when independence is needed, or `.clone()` on the tensor.

### Pitfall 2: dtype Mismatch

```python
arr = np.array([1.0, 2.0])          # float64
t = torch.from_numpy(arr)
print(t.dtype)                       # torch.float64 (not float32!)

# Fix: explicit dtype
arr32 = np.array([1.0, 2.0], dtype=np.float32)
```

### Pitfall 3: GPU Tensors Cannot Convert Directly

```python
if torch.cuda.is_available():
    t_gpu = torch.randn(3, device='cuda')
    # t_gpu.numpy()  # RuntimeError!
    arr = t_gpu.cpu().numpy()  # Move to CPU first
```

---

## Tips

- **Autograd**: Tensors created from NumPy have `requires_grad=False` by default. Set it on float/complex dtypes if you want gradients.
- **Device**: `from_numpy` / `as_tensor` always create CPU tensors. Moving to CUDA/MPS always causes a **COPY**, breaking sharing.
- **Independence after sharing**: Use `.clone()` on the tensor to get an independent copy.
- **Negative/odd strides**: Some NumPy views (e.g., `arr[::-1]`) are incompatible with `from_numpy`; `as_tensor` will **COPY** instead.
- **Pandas**: Always go through `.values` (or `.to_numpy()`) for DataFrame → Tensor conversion.

---

## Summary

| Operation | Memory | Use When |
|-----------|--------|----------|
| `torch.from_numpy(arr)` | Share | Need speed, array won't be mutated |
| `torch.as_tensor(arr)` | Try share | Best-effort sharing |
| `torch.tensor(arr)` | Copy | Need safety, independence |
| `tensor.numpy()` | Share | CPU tensor, no grad |
| `tensor.detach().cpu().numpy()` | Copy | GPU tensor or has grad |
| `torch.tensor(df.values)` | Copy | DataFrame → Tensor |
| `pd.DataFrame(tensor.numpy())` | Share/Copy | Tensor → DataFrame |

## See Also

- [Tensor Basics](tensor_basics.md) — Fundamental tensor concepts
- [Tensor Creation](tensor_creation.md) — Creation methods and dtype system
- [Indexing and Slicing](indexing_slicing.md) — Accessing tensor elements
