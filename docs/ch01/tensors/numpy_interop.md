# NumPy Interoperability

## Overview

PyTorch and NumPy are designed to work seamlessly together. Understanding the different conversion methods—and critically, which ones share memory versus copy data—is essential for efficient workflows.

## The Three Conversion Methods

| Method | Memory Behavior | When to Use |
|--------|-----------------|-------------|
| `torch.from_numpy()` | **SHARE** (no copy) | Zero-copy when possible |
| `torch.as_tensor()` | **TRY TO SHARE** | Best-effort sharing |
| `torch.tensor()` | **COPY** (always) | Safe, independent tensor |

## `torch.from_numpy()` - Zero-Copy Sharing

Creates a PyTorch tensor that **shares memory** with the NumPy array:

```python
import numpy as np
import torch

# Create NumPy array
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# Create tensor that SHARES memory
t_shared = torch.from_numpy(arr)

print(f"NumPy array: {arr}")
print(f"PyTorch tensor: {t_shared}")

# Mutations propagate both ways!
arr[0] = 99.0
print(f"After arr[0]=99: tensor is {t_shared}")  # tensor([99., 2., 3.])

t_shared[1] = -7.0
print(f"After t_shared[1]=-7: array is {arr}")  # [99. -7.  3.]
```

### Verifying Shared Memory

```python
def ptr_numpy(a):
    """Get data pointer of NumPy array."""
    return a.__array_interface__['data'][0]

def ptr_torch(t):
    """Get data pointer of PyTorch tensor storage."""
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
3. **Compatible strides** (positive strides work; negative may fail)

```python
# Read-only array fails
ro_arr = np.array([1.0, 2.0, 3.0])
ro_arr.setflags(write=False)

try:
    torch.from_numpy(ro_arr)
except Exception as e:
    print(f"Read-only error: {type(e).__name__}")

# Solution: make a copy first
t_copy = torch.from_numpy(np.array(ro_arr, copy=True))
```

## `torch.as_tensor()` - Best-Effort Sharing

Attempts to share memory when possible, otherwise copies:

```python
# Shares when possible
arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)
t_as = torch.as_tensor(arr)

arr[1] = 222.0
print(f"After arr change: {t_as}")  # Shared, so change is visible

# Copies when necessary (dtype conversion requested)
t_converted = torch.as_tensor(arr, dtype=torch.float32)
arr[2] = 333.0
print(f"After dtype conversion: {t_converted}")  # No change - copied
```

### When `as_tensor()` Copies

- Read-only arrays
- Negative strides (e.g., `arr[::-1]`)
- Requested dtype differs from array dtype
- Requested device differs (e.g., `device='cuda'`)

## `torch.tensor()` - Safe Copying

Always creates an independent copy:

```python
arr = np.array([10, 20, 30], dtype=np.int64)
t_copy = torch.tensor(arr)

# Independent - no sharing
arr[0] = 123
print(f"After arr change: {t_copy}")  # Still [10, 20, 30]
```

### Advantages of `torch.tensor()`

- Safest option - no surprises from mutations
- Can specify dtype and device directly
- Works with any input (lists, tuples, scalars, arrays)

```python
# Direct dtype and device specification
t = torch.tensor(arr, dtype=torch.float32, device='cpu')
```

## Dtype Mappings

Common NumPy to PyTorch dtype correspondences:

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
for np_dtype in [np.float32, np.float64, np.int32, np.int64, np.uint8, np.bool_]:
    arr = np.array([0, 1, 2], dtype=np_dtype)
    t = torch.from_numpy(arr)
    print(f"NumPy {arr.dtype:>8} → PyTorch {t.dtype}")
```

## Non-Contiguous Arrays

### Strided Views Still Share

```python
base = np.arange(10, dtype=np.float32)  # [0, 1, 2, ..., 9]
view = base[::2]  # [0, 2, 4, 6, 8] - non-contiguous view

t_view = torch.from_numpy(view)  # Still SHARES memory

view[0] = 999.0
print(f"After view[0]=999: base is {base}")  # Shows 999 at index 0
print(f"PyTorch tensor: {t_view}")  # Also shows 999
```

### Fortran-Order Arrays

Fortran-order (column-major) arrays can be shared but result in non-contiguous tensors:

```python
arr_f = np.asfortranarray(np.arange(6).reshape(2, 3))
t_f = torch.from_numpy(arr_f)

print(f"Fortran array order: {arr_f.flags['F_CONTIGUOUS']}")  # True
print(f"Tensor contiguous: {t_f.is_contiguous()}")  # False

# Many operations work; some may require .contiguous()
t_f_cont = t_f.contiguous()  # Creates a COPY
```

## Converting PyTorch to NumPy

### `tensor.numpy()` - Shares if Possible

```python
t = torch.tensor([1.0, 2.0, 3.0])
arr = t.numpy()  # SHARES memory (if tensor is on CPU, no grad)

t[0] = 99.0
print(f"After tensor change: {arr}")  # Shows 99.0
```

### Requirements for `numpy()`

1. Tensor must be on CPU
2. Tensor must not require gradients (or use `.detach()`)

```python
# GPU tensor
if torch.cuda.is_available():
    t_gpu = torch.tensor([1.0, 2.0], device='cuda')
    # t_gpu.numpy()  # Error!
    arr = t_gpu.cpu().numpy()  # Move to CPU first

# Tensor with gradients
t_grad = torch.tensor([1.0, 2.0], requires_grad=True)
# t_grad.numpy()  # Error!
arr = t_grad.detach().numpy()  # Detach first
```

## Practical Patterns

### Data Loading Pipeline

```python
# Common pattern: Load with NumPy, train with PyTorch
def load_data():
    # NumPy for loading (rich ecosystem)
    data = np.load('data.npy')
    labels = np.load('labels.npy')
    
    # Convert to PyTorch for training
    # Use tensor() for safety in training loops
    X = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    
    return X, y
```

### Visualization with Matplotlib

```python
import matplotlib.pyplot as plt

# PyTorch tensor
t = torch.randn(100, 100)

# Convert to NumPy for plotting
# Use .detach().cpu().numpy() for maximum compatibility
arr = t.detach().cpu().numpy()

plt.imshow(arr)
plt.colorbar()
plt.show()
```

### Mixed Operations

```python
# NumPy array
arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)

# PyTorch operations with zero-copy
t = torch.from_numpy(arr)
t_squared = t ** 2

# Result back to NumPy
result = t_squared.numpy()
print(f"Squared values: {result}")
```

## Decision Flowchart

```
Need NumPy array → PyTorch tensor?
    │
    ├─ Want safety, no surprises? → torch.tensor() [COPY]
    │
    ├─ Want zero-copy, understand risks? → torch.from_numpy() [SHARE]
    │
    └─ Want auto-decide? → torch.as_tensor() [TRY SHARE]

Need PyTorch tensor → NumPy array?
    │
    ├─ On CPU, no gradients? → tensor.numpy() [SHARE]
    │
    └─ GPU or has gradients? → tensor.detach().cpu().numpy()
```

## Common Pitfalls

### Pitfall 1: Unexpected Mutations

```python
# Danger: modifying shared array affects tensor
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)

# Later in code, arr gets modified...
arr[0] = 0  # Oops! Tensor also changed
```

**Solution**: Use `torch.tensor()` for safety, or document shared-memory assumptions.

### Pitfall 2: Device Mismatch

```python
arr = np.array([1.0, 2.0])

# from_numpy always creates CPU tensor
t_cpu = torch.from_numpy(arr)

# Moving to GPU breaks sharing
t_gpu = t_cpu.to('cuda')  # COPY - no longer shares with arr
```

### Pitfall 3: dtype Surprises

```python
# Python floats default to float64 in NumPy
arr = np.array([1.0, 2.0])  # float64!
t = torch.from_numpy(arr)
print(t.dtype)  # torch.float64

# PyTorch defaults to float32
t2 = torch.tensor([1.0, 2.0])
print(t2.dtype)  # torch.float32

# Solution: explicit dtype
arr32 = np.array([1.0, 2.0], dtype=np.float32)
```

## Summary

| Operation | Memory | Use When |
|-----------|--------|----------|
| `torch.from_numpy(arr)` | Share | Need speed, array won't be mutated |
| `torch.as_tensor(arr)` | Try share | Best-effort sharing |
| `torch.tensor(arr)` | Copy | Need safety, independence |
| `tensor.numpy()` | Share | CPU tensor, no grad |
| `tensor.detach().cpu().numpy()` | Copy | GPU tensor or has grad |

## See Also

- [Tensor Basics](tensor_basics.md) - Fundamental concepts
- [Memory Layout and Strides](memory_layout_strides.md) - Memory details
- [Tensor Creation and dtypes](tensor_creation_dtypes.md) - Creation methods
