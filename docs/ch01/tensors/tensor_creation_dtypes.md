# Tensor Creation and Data Types

## Overview

PyTorch provides multiple methods for creating tensors, each optimized for different use cases. Understanding these methods and their associated data types is crucial for efficient deep learning workflows.

## Data Types (dtypes)

PyTorch supports a comprehensive set of data types:

### Floating-Point Types

| dtype | Bits | Range | Use Case |
|-------|------|-------|----------|
| `torch.float16` / `torch.half` | 16 | ±6.5×10⁴ | Mixed precision training |
| `torch.bfloat16` | 16 | ±3.4×10³⁸ | TPU/modern GPU training |
| `torch.float32` / `torch.float` | 32 | ±3.4×10³⁸ | Default for most operations |
| `torch.float64` / `torch.double` | 64 | ±1.8×10³⁰⁸ | High precision computation |

### Integer Types

| dtype | Bits | Range | Use Case |
|-------|------|-------|----------|
| `torch.int8` | 8 | -128 to 127 | Quantized models |
| `torch.uint8` | 8 | 0 to 255 | Image data |
| `torch.int16` / `torch.short` | 16 | -32,768 to 32,767 | Memory-efficient integers |
| `torch.int32` / `torch.int` | 32 | -2³¹ to 2³¹-1 | General integer operations |
| `torch.int64` / `torch.long` | 64 | -2⁶³ to 2⁶³-1 | Indices, labels |

### Other Types

| dtype | Description | Use Case |
|-------|-------------|----------|
| `torch.bool` | Boolean | Masks, conditions |
| `torch.complex64` | Complex (float32 parts) | Signal processing |
| `torch.complex128` | Complex (float64 parts) | High precision complex |

## Creating Tensors from Python Data

### From Lists

```python
import torch

# 1D tensor from list
vec = torch.tensor([1, 2, 3, 4, 5])
print(f"Shape: {vec.shape}, Dtype: {vec.dtype}")
# Shape: torch.Size([5]), Dtype: torch.int64

# 2D tensor from nested lists
mat = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Shape: {mat.shape}, Dtype: {mat.dtype}")
# Shape: torch.Size([2, 3]), Dtype: torch.int64

# Float tensor
float_vec = torch.tensor([1.0, 2.0, 3.0])
print(f"Dtype: {float_vec.dtype}")  # torch.float32

# Explicit dtype
int8_tensor = torch.tensor([1, 2, 3], dtype=torch.int8)
print(f"Dtype: {int8_tensor.dtype}")  # torch.int8
```

### From Ranges

```python
# arange: like Python range
t1 = torch.arange(10)           # 0 to 9
t2 = torch.arange(2, 10)        # 2 to 9
t3 = torch.arange(0, 10, 2)     # 0, 2, 4, 6, 8

print(f"arange(10): {t1}")
print(f"arange(2, 10): {t2}")
print(f"arange(0, 10, 2): {t3}")

# linspace: evenly spaced values
lin = torch.linspace(0, 1, steps=5)  # [0.00, 0.25, 0.50, 0.75, 1.00]
print(f"linspace(0, 1, 5): {lin}")

# logspace: logarithmically spaced values
log = torch.logspace(0, 2, steps=3)  # [10^0, 10^1, 10^2] = [1, 10, 100]
print(f"logspace(0, 2, 3): {log}")
```

## Factory Functions

Factory functions create tensors with specific patterns or properties.

### Zeros and Ones

```python
# All zeros
zeros_1d = torch.zeros(5)
zeros_2d = torch.zeros(3, 4)
zeros_3d = torch.zeros(2, 3, 4)

print(f"zeros(5): shape {zeros_1d.shape}")
print(f"zeros(3, 4):\n{zeros_2d}")

# All ones
ones_1d = torch.ones(5)
ones_2d = torch.ones(3, 4)

print(f"ones(5): {ones_1d}")

# Match shape of existing tensor
template = torch.randn(2, 3)
zeros_like = torch.zeros_like(template)
ones_like = torch.ones_like(template)

print(f"zeros_like shape: {zeros_like.shape}")  # torch.Size([2, 3])
```

### Filled Tensors

```python
# Fill with specific value
filled = torch.full((3, 4), fill_value=3.14)
print(f"full((3, 4), 3.14):\n{filled}")

# Fill like existing tensor
filled_like = torch.full_like(template, fill_value=7)
print(f"full_like shape: {filled_like.shape}")
```

### Identity and Diagonal

```python
# Identity matrix
eye = torch.eye(4)
print(f"eye(4):\n{eye}")

# Diagonal matrix from vector
diag = torch.diag(torch.tensor([1, 2, 3]))
print(f"diag([1,2,3]):\n{diag}")

# Extract diagonal from matrix
mat = torch.arange(9).reshape(3, 3)
diagonal = torch.diag(mat)
print(f"Diagonal of mat: {diagonal}")  # tensor([0, 4, 8])
```

### Empty Tensors

```python
# Uninitialized tensor (fast, but contains garbage values!)
empty = torch.empty(3, 4)
print(f"empty(3, 4): may contain any values")

# Use empty for pre-allocated buffers when you'll overwrite all values
```

!!! warning "Empty Tensors Contain Garbage"
    `torch.empty()` does not initialize memory. Always overwrite all values
    before using the tensor, or use `torch.zeros()` instead.

## Random Tensors

### Uniform Distribution

```python
# Uniform [0, 1)
rand = torch.rand(3, 4)
print(f"rand(3, 4):\n{rand}")

# Uniform integers in range [low, high)
randint = torch.randint(low=0, high=10, size=(3, 4))
print(f"randint(0, 10, (3,4)):\n{randint}")
```

### Normal Distribution

```python
# Standard normal (mean=0, std=1)
randn = torch.randn(3, 4)
print(f"randn(3, 4):\n{randn}")

# Custom normal distribution
normal = torch.normal(mean=5.0, std=2.0, size=(3, 4))
print(f"normal(5, 2, (3,4)):\n{normal}")
```

### Reproducibility with Seeds

```python
# Set seed for reproducibility
torch.manual_seed(42)
t1 = torch.randn(3)

torch.manual_seed(42)  # Reset seed
t2 = torch.randn(3)

print(f"Same seed produces same values: {torch.equal(t1, t2)}")  # True

# Generator for local reproducibility
gen = torch.Generator().manual_seed(123)
t3 = torch.randn(3, generator=gen)
```

## Type Conversion

### Using `to()`

```python
float_tensor = torch.tensor([1.5, 2.5, 3.5])

# Convert to integer
int_tensor = float_tensor.to(torch.int32)
print(f"to(int32): {int_tensor}")  # tensor([1, 2, 3])

# Convert to different float precision
double_tensor = float_tensor.to(torch.float64)
print(f"to(float64) dtype: {double_tensor.dtype}")
```

### Convenience Methods

```python
t = torch.tensor([1, 2, 3])

# Common conversions
t_float = t.float()    # to float32
t_double = t.double()  # to float64
t_half = t.half()      # to float16
t_int = t.int()        # to int32
t_long = t.long()      # to int64
t_bool = t.bool()      # to bool

print(f"int tensor: {t.dtype}")
print(f"after .float(): {t_float.dtype}")
print(f"after .bool(): {t_bool}")  # tensor([True, True, True])
```

### Type Casting Warnings

```python
# Narrowing conversion (potential data loss)
large_float = torch.tensor([1000.7])
as_int8 = large_float.to(torch.int8)
print(f"1000.7 as int8: {as_int8}")  # tensor([-24]) - overflow!

# Always verify range when using narrow types
```

## Default Data Type

```python
# Check default dtype
print(f"Default float: {torch.get_default_dtype()}")  # torch.float32

# Change default (affects new tensor creation)
torch.set_default_dtype(torch.float64)
t = torch.tensor([1.0, 2.0])
print(f"After set_default_dtype(float64): {t.dtype}")  # torch.float64

# Reset to float32
torch.set_default_dtype(torch.float32)
```

## Device Specification

```python
# CPU tensor (default)
cpu_tensor = torch.tensor([1, 2, 3])
print(f"Device: {cpu_tensor.device}")  # cpu

# GPU tensor (if available)
if torch.cuda.is_available():
    gpu_tensor = torch.tensor([1, 2, 3], device='cuda')
    print(f"Device: {gpu_tensor.device}")  # cuda:0
    
    # Move tensor to GPU
    moved = cpu_tensor.to('cuda')
    print(f"Moved to: {moved.device}")

# Create directly on device with dtype
t = torch.zeros(3, 4, dtype=torch.float32, device='cpu')
```

## Best Practices

### Choosing the Right dtype

| Scenario | Recommended dtype |
|----------|-------------------|
| General training | `float32` |
| Mixed precision training | `float16` or `bfloat16` |
| Class labels, indices | `int64` |
| Image pixel values | `uint8` or `float32` |
| Binary masks | `bool` |
| Memory-constrained | `float16` |
| High precision needed | `float64` |

### Memory Estimation

```python
def estimate_memory(shape, dtype=torch.float32):
    """Estimate tensor memory in bytes."""
    numel = 1
    for dim in shape:
        numel *= dim
    
    bits_per_element = {
        torch.float16: 16, torch.float32: 32, torch.float64: 64,
        torch.int8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64,
        torch.bool: 8
    }
    
    bytes_needed = numel * bits_per_element[dtype] // 8
    return bytes_needed

# Example: ImageNet batch
shape = (32, 3, 224, 224)  # batch, channels, height, width
mem_fp32 = estimate_memory(shape, torch.float32)
mem_fp16 = estimate_memory(shape, torch.float16)

print(f"Batch memory (fp32): {mem_fp32 / 1e6:.1f} MB")
print(f"Batch memory (fp16): {mem_fp16 / 1e6:.1f} MB")
```

## Summary

| Function | Description | Example |
|----------|-------------|---------|
| `torch.tensor()` | From Python data | `torch.tensor([1,2,3])` |
| `torch.arange()` | Range of values | `torch.arange(0, 10, 2)` |
| `torch.zeros()` | All zeros | `torch.zeros(3, 4)` |
| `torch.ones()` | All ones | `torch.ones(3, 4)` |
| `torch.rand()` | Uniform [0, 1) | `torch.rand(3, 4)` |
| `torch.randn()` | Standard normal | `torch.randn(3, 4)` |
| `torch.eye()` | Identity matrix | `torch.eye(4)` |
| `torch.full()` | Filled with value | `torch.full((3,4), 7)` |

## See Also

- [Tensor Basics](tensor_basics.md) - Fundamental tensor concepts
- [Memory Layout and Strides](memory_layout_strides.md) - Storage details
- [NumPy Interoperability](numpy_interop.md) - NumPy conversion
