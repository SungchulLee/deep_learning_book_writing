# Dtype and Device

## Learning Objectives

By the end of this section, you will be able to:

- Identify and select appropriate data types for different deep learning tasks
- Understand the precision–memory–speed tradeoffs across dtypes
- Move tensors between CPU and GPU devices
- Perform safe type casting and device transfers
- Apply mixed-precision patterns for efficient training

---

## Overview

Every PyTorch tensor carries two fundamental attributes beyond its shape: the **data type** (`dtype`) that determines how each element is stored and interpreted, and the **device** that specifies where the tensor physically resides in memory. Choosing the right dtype–device combination is essential for numerical correctness, memory efficiency, and computational performance.

---

## Data Types (dtype)

### Inspecting dtype

```python
import torch

# Default types depend on input
x_int = torch.tensor([1, 2, 3])
print(x_int.dtype)  # torch.int64

x_float = torch.tensor([1.0, 2.0, 3.0])
print(x_float.dtype)  # torch.float32

x_bool = torch.tensor([True, False, True])
print(x_bool.dtype)  # torch.bool
```

PyTorch infers dtype from the input data: Python integers become `int64`, Python floats become `float32`, and Python booleans become `bool`.

### Complete dtype Reference

#### Floating-Point Types

| dtype | Alias | Bits | Range (approx.) | Precision | Typical Use |
|-------|-------|------|------------------|-----------|-------------|
| `torch.float16` | `torch.half` | 16 | ±65504 | ~3.3 decimal digits | Inference, mixed precision |
| `torch.bfloat16` | — | 16 | ±3.4×10³⁸ | ~2.4 decimal digits | Training, mixed precision |
| `torch.float32` | `torch.float` | 32 | ±3.4×10³⁸ | ~7.2 decimal digits | Default for most training |
| `torch.float64` | `torch.double` | 64 | ±1.8×10³⁰⁸ | ~15.9 decimal digits | Scientific computing |

```python
# Creating tensors with specific float dtypes
f16 = torch.tensor([1.0, 2.0], dtype=torch.float16)
bf16 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
f32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
f64 = torch.tensor([1.0, 2.0], dtype=torch.float64)

# Memory comparison
for t, name in [(f16, "float16"), (bf16, "bfloat16"),
                (f32, "float32"), (f64, "float64")]:
    print(f"{name}: {t.element_size()} bytes/element, "
          f"{t.numel() * t.element_size()} bytes total")
# float16:  2 bytes/element, 4 bytes total
# bfloat16: 2 bytes/element, 4 bytes total
# float32:  4 bytes/element, 8 bytes total
# float64:  8 bytes/element, 16 bytes total
```

#### float16 vs bfloat16

Both use 16 bits but allocate them differently:

```python
# float16: 1 sign + 5 exponent + 10 mantissa
# - Higher precision, smaller range
# - Can overflow/underflow more easily during training

# bfloat16: 1 sign + 8 exponent + 7 mantissa
# - Lower precision, same range as float32
# - More numerically stable for training gradients

# Demonstrating range difference
large_val = torch.tensor(100000.0)
print(large_val.to(torch.float16))   # tensor(inf) — overflow!
print(large_val.to(torch.bfloat16))  # tensor(98304.) — representable (with rounding)
```

#### Integer Types

| dtype | Bits | Range | Signed | Typical Use |
|-------|------|-------|--------|-------------|
| `torch.int8` | 8 | −128 to 127 | Yes | Quantized models |
| `torch.uint8` | 8 | 0 to 255 | No | Image pixel values |
| `torch.int16` | 16 | −32768 to 32767 | Yes | Rare |
| `torch.int32` | 32 | −2³¹ to 2³¹−1 | Yes | Indices, counts |
| `torch.int64` | 64 | −2⁶³ to 2⁶³−1 | Yes | Default integer, indices |

```python
# Integer dtypes
labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)  # Class labels
pixels = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)  # Image
quant = torch.tensor([1, -2, 3], dtype=torch.int8)  # Quantized weights
```

#### Other Types

| dtype | Bits | Description |
|-------|------|-------------|
| `torch.bool` | 8 | Boolean (True/False) |
| `torch.complex64` | 64 | Complex with float32 real/imag |
| `torch.complex128` | 128 | Complex with float64 real/imag |

```python
# Boolean tensors
mask = torch.tensor([True, False, True])
print(mask.dtype)  # torch.bool

# Complex tensors
z = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
print(z.real)  # tensor([1., 3.])
print(z.imag)  # tensor([2., 4.])
```

### Setting the Default dtype

```python
# Check current default
print(torch.get_default_dtype())  # torch.float32

# Change default (affects tensor creation functions)
torch.set_default_dtype(torch.float64)
x = torch.randn(3)
print(x.dtype)  # torch.float64

# Reset to standard
torch.set_default_dtype(torch.float32)
```

!!! warning "Changing Default dtype"
    Changing the default dtype affects all subsequent tensor creation functions
    (`torch.randn`, `torch.zeros`, etc.) but not `torch.tensor()` which infers
    dtype from input data. Change with caution—it can introduce subtle bugs in
    codebases that assume `float32`.

---

## Type Casting

### The `.to()` Method

The primary and most flexible method for type conversion:

```python
x = torch.tensor([1.5, 2.7, 3.9])

# Cast to integer (truncates toward zero)
x_int = x.to(torch.int32)
print(x_int)  # tensor([1, 2, 3])

# Cast to double precision
x_double = x.to(torch.float64)
print(x_double.dtype)  # torch.float64

# Cast to half precision
x_half = x.to(torch.float16)
print(x_half)  # tensor([1.5000, 2.7002, 3.8984]) — reduced precision
```

### Convenience Methods

```python
x = torch.tensor([1, 2, 3])

x.float()    # → torch.float32
x.double()   # → torch.float64
x.half()     # → torch.float16
x.bfloat16() # → torch.bfloat16
x.int()      # → torch.int32
x.long()     # → torch.int64
x.short()    # → torch.int16
x.byte()     # → torch.uint8
x.bool()     # → torch.bool
```

### Type Promotion Rules

When operating on tensors with different dtypes, PyTorch promotes to the more general type:

```python
# Integer + float → float
a = torch.tensor([1, 2, 3])        # int64
b = torch.tensor([0.5, 1.5, 2.5])  # float32
c = a + b
print(c.dtype)  # torch.float32

# float32 + float64 → float64
d = torch.tensor([1.0])             # float32
e = torch.tensor([2.0], dtype=torch.float64)
f = d + e
print(f.dtype)  # torch.float64

# Scalar promotion
g = torch.tensor([1.0, 2.0])  # float32
h = g + 1                     # Python int → preserves tensor dtype
print(h.dtype)  # torch.float32
```

### Casting Preserves vs Creates Storage

Type casting **always** creates a new tensor with new storage (even if the dtype is the same, `.to()` may return the original tensor as an optimization):

```python
x = torch.randn(3, dtype=torch.float32)

# Different dtype → new storage
y = x.to(torch.float64)
print(x.storage().data_ptr() == y.storage().data_ptr())  # False

# Same dtype → may return self
z = x.to(torch.float32)
print(x.storage().data_ptr() == z.storage().data_ptr())  # True (optimization)
```

### Precision Loss Awareness

```python
# float → int truncates (does not round)
x = torch.tensor([1.9, -1.9, 2.5])
print(x.to(torch.int64))  # tensor([1, -1, 2])

# float32 → float16 can overflow
large = torch.tensor([100000.0])
print(large.to(torch.float16))  # tensor([inf])

# float32 → float16 loses precision
precise = torch.tensor([1.00001])
print(precise.to(torch.float16))  # tensor([1.]) — precision lost
```

---

## Device Management

### Inspecting Device

```python
cpu_tensor = torch.randn(3)
print(cpu_tensor.device)  # device(type='cpu')

if torch.cuda.is_available():
    gpu_tensor = torch.randn(3, device='cuda')
    print(gpu_tensor.device)  # device(type='cuda', index=0)
```

### Creating Tensors on Specific Devices

```python
# Directly on GPU
if torch.cuda.is_available():
    x = torch.randn(3, 4, device='cuda')
    y = torch.zeros(3, 4, device='cuda:0')  # Explicit GPU index

# On CPU (default)
z = torch.randn(3, 4, device='cpu')
```

### Moving Between Devices

```python
x = torch.randn(3, 4)

if torch.cuda.is_available():
    # CPU → GPU (all equivalent)
    x_gpu = x.to('cuda')
    x_gpu = x.to(torch.device('cuda', 0))
    x_gpu = x.cuda()
    x_gpu = x.cuda(0)  # Specific GPU

    # GPU → CPU (all equivalent)
    x_cpu = x_gpu.to('cpu')
    x_cpu = x_gpu.cpu()
```

!!! warning "Cross-Device Operations"
    Tensors on different devices cannot participate in the same operation:

    ```python
    cpu_t = torch.randn(3)
    gpu_t = torch.randn(3, device='cuda')
    # cpu_t + gpu_t  # RuntimeError: Expected all tensors on same device
    ```

### Device-Agnostic Code

Write code that works on both CPU and GPU:

```python
# Pattern 1: Derive device from existing tensor
def add_bias(x, bias_value):
    bias = torch.full_like(x, bias_value)  # Same device and dtype as x
    return x + bias

# Pattern 2: Use a device variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyModel().to(device)
data = torch.randn(32, 784).to(device)
output = model(data)

# Pattern 3: Match device of another tensor
def create_mask(reference_tensor, threshold):
    return (reference_tensor > threshold).to(reference_tensor.device)
```

### The `.is_cuda` Property

```python
x = torch.randn(3)
print(x.is_cuda)  # False

if torch.cuda.is_available():
    y = x.cuda()
    print(y.is_cuda)  # True
```

### Multi-GPU Device Management

```python
if torch.cuda.is_available():
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Place tensors on specific GPUs
    x0 = torch.randn(3, device='cuda:0')
    if num_gpus > 1:
        x1 = torch.randn(3, device='cuda:1')
```

---

## Combined dtype and Device Operations

### `.to()` for Simultaneous Conversion

The `.to()` method can change both dtype and device in a single call:

```python
x = torch.tensor([1, 2, 3])  # int64, cpu

if torch.cuda.is_available():
    # Change both dtype and device at once
    x_gpu_float = x.to(device='cuda', dtype=torch.float32)
    print(x_gpu_float.dtype)   # torch.float32
    print(x_gpu_float.device)  # cuda:0

# Match another tensor's dtype and device
reference = torch.randn(3, dtype=torch.float16, device='cuda')
matched = x.to(reference)  # Same dtype and device as reference
```

### Creating Tensors with Matching Properties

```python
x = torch.randn(3, 4, dtype=torch.float64, device='cpu')

# *_like functions: match shape, dtype, and device
zeros = torch.zeros_like(x)
ones = torch.ones_like(x)
rand = torch.randn_like(x)
full = torch.full_like(x, 3.14)

print(f"All have dtype={zeros.dtype}, device={zeros.device}")

# new_* methods: match dtype and device, custom shape
y = x.new_zeros(2, 3)
z = x.new_ones(5)
w = x.new_empty(4, 4)
v = x.new_full((2, 2), 7.0)
```

---

## Mixed Precision Training

### Automatic Mixed Precision (AMP)

Mixed precision uses `float16` or `bfloat16` for speed-critical operations while keeping `float32` for numerically sensitive ones:

```python
if torch.cuda.is_available():
    model = MyModel().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler()

    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()

        # Forward pass in mixed precision
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target)

        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### Why Gradient Scaling?

In `float16`, small gradient values can underflow to zero. The `GradScaler` multiplies the loss by a scale factor before `backward()`, then unscales gradients before the optimizer step, preserving small gradient magnitudes.

### bfloat16 Advantage

With `bfloat16`, gradient scaling is typically unnecessary because the exponent range matches `float32`:

```python
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)

# No scaler needed with bfloat16
loss.backward()
optimizer.step()
```

---

## Memory and Performance Implications

### Memory Estimation

```python
def estimate_tensor_memory(shape, dtype=torch.float32):
    """Estimate memory usage in bytes."""
    numel = 1
    for s in shape:
        numel *= s
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    total_bytes = numel * bytes_per_element
    return total_bytes

# Example: ImageNet batch
shape = (32, 3, 224, 224)
for dtype in [torch.float32, torch.float16, torch.float64]:
    mem = estimate_tensor_memory(shape, dtype)
    print(f"{dtype}: {mem / 1e6:.1f} MB")
# torch.float32: 19.3 MB
# torch.float16: 9.6 MB
# torch.float64: 38.5 MB
```

### Performance: dtype Selection Impact

```python
import time

def benchmark_matmul(n, dtype, device='cpu', iterations=100):
    a = torch.randn(n, n, dtype=dtype, device=device)
    b = torch.randn(n, n, dtype=dtype, device=device)
    if device != 'cpu':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = a @ b
    if device != 'cpu':
        torch.cuda.synchronize()
    return (time.time() - start) / iterations

# Compare dtypes (CPU)
for dtype in [torch.float32, torch.float64]:
    t = benchmark_matmul(512, dtype)
    print(f"{dtype}: {t*1000:.2f} ms")
```

---

## NumPy Interoperability

### dtype Mapping

```python
import numpy as np

# NumPy → PyTorch
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
t = torch.from_numpy(arr)
print(t.dtype)  # torch.float32

# PyTorch → NumPy (CPU tensors only, no gradient)
t = torch.randn(3, dtype=torch.float32)
arr = t.numpy()
print(arr.dtype)  # float32

# GPU tensor → NumPy requires CPU transfer
if torch.cuda.is_available():
    t_gpu = torch.randn(3, device='cuda')
    arr = t_gpu.cpu().numpy()
```

### Shared Memory with NumPy

```python
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
t = torch.from_numpy(arr)  # Shares memory!

arr[0] = 99.0
print(t[0])  # tensor(99.) — shared memory

# To avoid sharing: clone
t_independent = torch.from_numpy(arr).clone()
```

---

## Tensor Inspection Utilities

```python
def tensor_info(t, name="Tensor"):
    """Display comprehensive dtype and device information."""
    print(f"=== {name} ===")
    print(f"  dtype: {t.dtype}")
    print(f"  device: {t.device}")
    print(f"  shape: {t.shape}")
    print(f"  element_size: {t.element_size()} bytes")
    print(f"  total memory: {t.numel() * t.element_size()} bytes")
    print(f"  requires_grad: {t.requires_grad}")
    if t.is_cuda:
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print()

# Usage
x = torch.randn(32, 3, 224, 224)
tensor_info(x, "ImageNet Batch")
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Unintended Integer Arithmetic

```python
# Python integers create int64 tensors
x = torch.tensor([1, 2, 3])
print(x.dtype)  # torch.int64

# Division produces integer result for int tensors in some contexts
print(x / 2)  # tensor([0.5000, 1.0000, 1.5000]) — PyTorch promotes to float

# But floor division stays integer
print(x // 2)  # tensor([0, 1, 1])

# Fix: explicitly use float
x = torch.tensor([1.0, 2.0, 3.0])  # or x.float()
```

### Pitfall 2: Cross-Device Operations

```python
if torch.cuda.is_available():
    a = torch.randn(3, device='cpu')
    b = torch.randn(3, device='cuda')

    try:
        c = a + b  # RuntimeError!
    except RuntimeError:
        c = a.cuda() + b  # Move a to GPU first
```

### Pitfall 3: dtype Mismatch in Operations

```python
a = torch.tensor([1.0], dtype=torch.float32)
b = torch.tensor([2.0], dtype=torch.float64)

# This works (auto-promotes to float64), but may be unexpected
c = a + b
print(c.dtype)  # torch.float64

# Loss functions often require matching dtypes
# Ensure model output and target have compatible types
```

### Pitfall 4: Half-Precision Overflow

```python
x = torch.tensor([60000.0], dtype=torch.float16)
y = x * 2
print(y)  # tensor([inf]) — overflow!

# Use bfloat16 for large-range values
x_bf = torch.tensor([60000.0], dtype=torch.bfloat16)
y_bf = x_bf * 2
print(y_bf)  # tensor([122880.]) — no overflow
```

---

## Best Practices

1. **Use `float32` as the default** for training unless you have a specific reason to use other dtypes.

2. **Use `float16` or `bfloat16`** for inference and mixed-precision training to reduce memory and improve throughput.

3. **Prefer `bfloat16` over `float16`** for training — it has the same dynamic range as `float32` and rarely needs gradient scaling.

4. **Use `int64` for indices and labels** — this is PyTorch's default for integer tensors and is expected by most loss functions.

5. **Write device-agnostic code** using `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` or by deriving device from existing tensors.

6. **Use `*_like` functions** (`torch.zeros_like`, `torch.randn_like`) to automatically match dtype and device of existing tensors.

7. **Check `.element_size()`** to understand memory usage per element.

8. **Use `.to()` for simultaneous** dtype and device conversion.

---

## Summary

| Attribute/Method | Description | Returns |
|-----------------|-------------|---------|
| `.dtype` | Data type of tensor elements | `torch.dtype` |
| `.device` | Device where tensor resides | `torch.device` |
| `.is_cuda` | Whether tensor is on GPU | `bool` |
| `.element_size()` | Bytes per element | `int` |
| `.to(dtype)` | Cast to new dtype | `Tensor` |
| `.to(device)` | Move to device | `Tensor` |
| `.to(dtype, device)` | Cast and move simultaneously | `Tensor` |
| `.float()` / `.double()` / `.half()` | Convenience casting | `Tensor` |
| `.cuda()` / `.cpu()` | Device transfer shortcuts | `Tensor` |
| `torch.zeros_like(t)` | Match dtype/device/shape | `Tensor` |
| `t.new_zeros(shape)` | Match dtype/device, custom shape | `Tensor` |

---

## See Also

- [Memory Layout and Strides](memory_layout_strides.md) — How tensor data is arranged in memory
- [Memory Management](memory_management.md) — GPU memory allocation and optimization
- [Broadcasting Rules](broadcasting_rules.md) — Type promotion during broadcasting
