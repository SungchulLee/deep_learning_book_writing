# Shape and Dimensions

Understanding tensor shapes and dimensions is fundamental to working effectively with PyTorch. Every tensor operation requires careful attention to shape compatibility, and debugging shape mismatches is one of the most common tasks in deep learning development. This section covers essential attributes and methods for inspecting and manipulating tensor dimensionality.

## Core Shape Attributes

Every PyTorch tensor has a shape that describes its dimensions. The shape tells you how many elements exist along each axis.

### The `.shape` Attribute

The most common way to inspect a tensor's dimensions:

```python
import torch

# 1D tensor
vec = torch.tensor([1, 2, 3, 4, 5])
print(f"Shape: {vec.shape}")  # torch.Size([5])

# 2D tensor
mat = torch.randn(3, 4)
print(f"Shape: {mat.shape}")  # torch.Size([3, 4])

# 3D tensor
cube = torch.randn(2, 3, 4)
print(f"Shape: {cube.shape}")  # torch.Size([2, 3, 4])
```

`torch.Size` is a tuple subclass, so you can use standard indexing:

```python
x = torch.arange(12).reshape(3, 4)
batch_size = x.shape[0]      # 3
num_features = x.shape[-1]   # 4 (negative indexing works)
```

### The `.size()` Method

Functionally equivalent to `.shape`, but allows querying a specific dimension directly:

```python
t = torch.randn(2, 3, 4)

# These are equivalent
print(t.shape)    # torch.Size([2, 3, 4])
print(t.size())   # torch.Size([2, 3, 4])

# Get specific dimension
print(t.size(0))  # 2
print(t.size(1))  # 3
print(t.size(-1)) # 4 (last dimension)
```

### Number of Dimensions: `.ndim`

The **rank** or **order** of a tensor is its number of dimensions:

```python
scalar = torch.tensor(42)
vec = torch.tensor([1, 2, 3])
mat = torch.randn(3, 4)
cube = torch.randn(2, 3, 4)

print(f"Scalar ndim: {scalar.ndim}")  # 0
print(f"Vector ndim: {vec.ndim}")     # 1
print(f"Matrix ndim: {mat.ndim}")     # 2
print(f"3D tensor ndim: {cube.ndim}") # 3

# Alternative: len(shape)
print(f"Dimensions: {len(cube.shape)}")  # 3
```

### Total Number of Elements: `.numel()`

Returns the total number of elements in the tensor:

```python
t = torch.randn(2, 3, 4)

# numel() returns total element count
print(f"Total elements: {t.numel()}")  # 24 (= 2 × 3 × 4)

# Useful for memory estimation
bytes_per_element = 4  # float32
memory_bytes = t.numel() * bytes_per_element
print(f"Memory: {memory_bytes} bytes")
```

## Data Type and Device Attributes

### Data Type: `.dtype`

Every tensor has a data type that determines precision and memory usage:

```python
x = torch.tensor([1, 2, 3])
print(x.dtype)  # torch.int64

y = torch.tensor([1.0, 2.0, 3.0])
print(y.dtype)  # torch.float32

z = torch.randn(3, dtype=torch.float64)
print(z.dtype)  # torch.float64
```

Common data types:

| dtype | Description | Size |
|-------|-------------|------|
| `torch.float32` | Single precision float | 4 bytes |
| `torch.float64` | Double precision float | 8 bytes |
| `torch.float16` | Half precision float | 2 bytes |
| `torch.bfloat16` | Brain floating point | 2 bytes |
| `torch.int64` | 64-bit integer | 8 bytes |
| `torch.int32` | 32-bit integer | 4 bytes |
| `torch.bool` | Boolean | 1 byte |

### Device: `.device`

Indicates where the tensor's data resides:

```python
cpu_tensor = torch.randn(3)
print(cpu_tensor.device)  # cpu

if torch.cuda.is_available():
    gpu_tensor = torch.randn(3, device='cuda')
    print(gpu_tensor.device)  # cuda:0
```

### Device Check: `.is_cuda`

Boolean check for GPU residence:

```python
x = torch.randn(3)
print(x.is_cuda)  # False

if torch.cuda.is_available():
    y = x.cuda()
    print(y.is_cuda)  # True
```

## Memory Layout Attributes

### Contiguity: `.is_contiguous()`

Determines if tensor elements are stored contiguously in memory:

```python
x = torch.arange(12).reshape(3, 4)
print(x.is_contiguous())  # True

# Transpose changes memory layout
xt = x.t()
print(xt.is_contiguous())  # False
```

Non-contiguous tensors may require `.contiguous()` before certain operations like `.view()`.

### Strides: `.stride()`

Returns the number of elements to skip in memory to advance one position along each dimension:

```python
x = torch.arange(12).reshape(3, 4)
print(x.stride())  # (4, 1)
# Moving along dim 0 skips 4 elements
# Moving along dim 1 skips 1 element

xt = x.t()
print(xt.stride())  # (1, 4)
# After transpose, strides swap
```

Understanding strides is crucial for:

- Debugging memory layout issues
- Understanding when operations create views vs copies
- Optimizing memory access patterns

### Storage Offset: `.storage_offset()`

Returns the offset into the underlying storage:

```python
x = torch.arange(12)
y = x[3:8]  # View of x
print(y.storage_offset())  # 3
```

### Layout: `.layout`

Describes the tensor's memory layout:

```python
dense = torch.randn(3, 3)
print(dense.layout)  # torch.strided

sparse = torch.sparse_coo_tensor([[0, 1], [1, 0]], [3., 4.], (3, 3))
print(sparse.layout)  # torch.sparse_coo
```

## Dimension Naming Conventions

In deep learning, specific dimension positions have conventional meanings:

### Image Data (NCHW vs NHWC)

```python
# PyTorch convention: NCHW (batch, channels, height, width)
batch_nchw = torch.randn(32, 3, 224, 224)
print(f"Batch size: {batch_nchw.shape[0]}")   # 32
print(f"Channels: {batch_nchw.shape[1]}")     # 3
print(f"Height: {batch_nchw.shape[2]}")       # 224
print(f"Width: {batch_nchw.shape[3]}")        # 224

# TensorFlow/NumPy convention: NHWC
batch_nhwc = torch.randn(32, 224, 224, 3)
```

### Sequence Data

```python
# Common convention: (batch, sequence_length, features)
sequences = torch.randn(16, 50, 256)
print(f"Batch size: {sequences.shape[0]}")       # 16
print(f"Sequence length: {sequences.shape[1]}")  # 50
print(f"Feature dim: {sequences.shape[2]}")      # 256

# RNN packed convention: (sequence_length, batch, features)
packed = torch.randn(50, 16, 256)
```

## Transpose Helpers

### `.T` - Simple Transpose

For 2D matrices, transposes rows and columns:

```python
x = torch.arange(6).reshape(2, 3)
print(x.T.shape)  # torch.Size([3, 2])
```

!!! warning "Multi-dimensional Tensors"
    For tensors with more than 2 dimensions, `.T` reverses **all** dimensions,
    which may not be what you want.

### `.mT` - Matrix Transpose

Transposes the last two dimensions only, useful for batched matrices:

```python
batch = torch.randn(5, 3, 4)  # 5 matrices of shape 3×4
print(batch.mT.shape)  # torch.Size([5, 4, 3])
```

### `.H` - Hermitian Transpose

Conjugate transpose for complex tensors (same as `.mT` for real tensors):

```python
z = torch.randn(2, 3, dtype=torch.complex64)
print(z.H.shape)  # torch.Size([3, 2])
```

## Autograd-Related Attributes

### `.requires_grad`

Indicates whether gradients should be computed for this tensor:

```python
x = torch.randn(3, requires_grad=True)
print(x.requires_grad)  # True

y = torch.randn(3)
print(y.requires_grad)  # False
```

### `.is_leaf`

Leaf tensors are created by the user (not by operations):

```python
w = torch.randn(3, requires_grad=True)
print(w.is_leaf)  # True

y = w * 2
print(y.is_leaf)  # False (result of operation)
```

### `.grad_fn`

Points to the function that created this tensor (for non-leaf tensors):

```python
x = torch.randn(3, requires_grad=True)
print(x.grad_fn)  # None (leaf tensor)

y = x ** 2
print(y.grad_fn)  # <PowBackward0 object>
```

### `.grad`

Stores accumulated gradients after `.backward()`:

```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # Gradient of y with respect to x
```

## Dynamic Shape Handling

### Getting Shape at Runtime

```python
def process_batch(images):
    """Process any batch of images."""
    batch_size = images.shape[0]
    channels = images.shape[1]
    height, width = images.shape[2], images.shape[3]
    
    print(f"Processing {batch_size} images of size {height}x{width}")
    
    # Use shapes for dynamic computation
    flat_size = channels * height * width
    flat_images = images.reshape(batch_size, flat_size)
    
    return flat_images

# Works with any batch size
small_batch = torch.randn(8, 3, 64, 64)
process_batch(small_batch)

large_batch = torch.randn(64, 3, 128, 128)
process_batch(large_batch)
```

### Inferred Dimensions with -1

```python
t = torch.arange(24)

# Let PyTorch compute one dimension
mat = t.reshape(4, -1)  # -1 → 6
print(f"reshape(4, -1): {mat.shape}")  # torch.Size([4, 6])

mat2 = t.reshape(-1, 3)  # -1 → 8  
print(f"reshape(-1, 3): {mat2.shape}")  # torch.Size([8, 3])

mat3 = t.reshape(2, -1, 4)  # -1 → 3
print(f"reshape(2, -1, 4): {mat3.shape}")  # torch.Size([2, 3, 4])
```

## Shape Inspection Utilities

### Comprehensive Info Function

```python
def tensor_info(t, name="Tensor"):
    """Comprehensive tensor information."""
    print(f"\n=== {name} ===")
    print(f"  Shape: {t.shape}")
    print(f"  ndim: {t.ndim}")
    print(f"  numel: {t.numel()}")
    print(f"  dtype: {t.dtype}")
    print(f"  device: {t.device}")
    print(f"  requires_grad: {t.requires_grad}")
    print(f"  is_contiguous: {t.is_contiguous()}")
    print(f"  stride: {t.stride()}")
    
    for i, dim_size in enumerate(t.shape):
        print(f"  Dim {i}: {dim_size}")
    print()

# Usage
batch = torch.randn(32, 3, 224, 224)
tensor_info(batch, "ImageNet Batch")
```

### Shape Assertions

```python
def verify_shape(t, expected_shape, name="tensor"):
    """Assert tensor has expected shape."""
    if t.shape != torch.Size(expected_shape):
        raise ValueError(
            f"{name} has shape {t.shape}, expected {expected_shape}"
        )

# Usage in code
x = torch.randn(32, 784)
verify_shape(x, (32, 784), "input")
```

### Debug Shapes for Multiple Tensors

```python
def debug_shapes(tensors_dict):
    """Print shapes for debugging."""
    for name, tensor in tensors_dict.items():
        print(f"{name}: {tensor.shape}, dtype={tensor.dtype}")

# Usage
x = torch.randn(32, 128)
W = torch.randn(128, 64)
debug_shapes({'input': x, 'weight': W})
```

## Common Shape Operations

### Checking Shape Compatibility

```python
def shapes_broadcastable(shape1, shape2):
    """Check if two shapes can be broadcast together."""
    for d1, d2 in zip(reversed(shape1), reversed(shape2)):
        if d1 != d2 and d1 != 1 and d2 != 1:
            return False
    return True

# Examples
print(shapes_broadcastable((3, 4), (4,)))      # True
print(shapes_broadcastable((3, 4), (3, 1)))    # True
print(shapes_broadcastable((3, 4), (2, 4)))    # False
```

### Computing Output Shapes

```python
def broadcast_shapes(shape1, shape2):
    """Compute the broadcast result shape."""
    result = []
    for d1, d2 in zip(
        reversed(shape1 + (1,) * (len(shape2) - len(shape1))),
        reversed(shape2 + (1,) * (len(shape1) - len(shape2)))
    ):
        result.append(max(d1, d2))
    return tuple(reversed(result))

# Examples
print(broadcast_shapes((3, 1), (1, 4)))    # (3, 4)
print(broadcast_shapes((5,), (4, 5)))      # (4, 5)
print(broadcast_shapes((2, 3, 1), (3, 5))) # (2, 3, 5)
```

## Shape in Neural Network Layers

### Linear Layer

```python
import torch.nn as nn

# Linear layer: input_features → output_features
linear = nn.Linear(784, 256)

# Input: (batch_size, 784)
x = torch.randn(32, 784)
y = linear(x)
print(f"Linear: {x.shape} → {y.shape}")  # [32, 784] → [32, 256]
```

### Convolutional Layer

```python
# Conv2d: (N, C_in, H, W) → (N, C_out, H', W')
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

x = torch.randn(32, 3, 224, 224)
y = conv(x)
print(f"Conv2d: {x.shape} → {y.shape}")
# [32, 3, 224, 224] → [32, 64, 224, 224]
```

### Computing Convolutional Output Size

$$H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding} - \text{dilation} \times (\text{kernel\_size} - 1) - 1}{\text{stride}}\right\rfloor + 1$$

```python
def conv_output_size(in_size, kernel_size, padding=0, stride=1, dilation=1):
    """Compute output size of convolution."""
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

# Example: 224x224 with 7x7 kernel, stride 2, padding 3
print(conv_output_size(224, 7, padding=3, stride=2))  # 112
```

## Debugging Shape Issues

### Common Error Messages

```python
# Shape mismatch in matrix multiply
A = torch.randn(3, 4)
B = torch.randn(5, 6)

try:
    C = A @ B
except RuntimeError as e:
    print(f"MatMul error: {e}")
    print(f"A.shape={A.shape}, B.shape={B.shape}")
    print("Requirement: A.shape[-1] == B.shape[-2]")
```

### Shape Debugging Pattern

```python
def debug_forward(model, x):
    """Print shapes through model forward pass."""
    print(f"Input: {x.shape}")
    
    for name, layer in model.named_children():
        x = layer(x)
        print(f"After {name}: {x.shape}")
    
    return x

# Usage
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

x = torch.randn(32, 784)
debug_forward(model, x)
```

## Practical Patterns

### Batch Dimension Handling

```python
def ensure_batch_dim(x, expected_ndim=4):
    """Ensure tensor has batch dimension."""
    if x.ndim == expected_ndim:
        return x
    elif x.ndim == expected_ndim - 1:
        return x.unsqueeze(0)
    else:
        raise ValueError(f"Expected {expected_ndim-1}D or {expected_ndim}D tensor, got {x.ndim}D")

# Image processing example
def predict(model, x):
    """Predict with automatic batch handling."""
    # Add batch dimension if missing
    if x.ndim == 3:  # Single image (C, H, W)
        x = x.unsqueeze(0)  # → (1, C, H, W)
    
    output = model(x)
    return output

# Works with both single images and batches
single_image = torch.randn(3, 224, 224)
batch = torch.randn(32, 3, 224, 224)
```

### Channel-Last to Channel-First Conversion

```python
# Convert NHWC (TensorFlow style) to NCHW (PyTorch style)
nhwc = torch.randn(32, 224, 224, 3)
nchw = nhwc.permute(0, 3, 1, 2)
print(f"NHWC {nhwc.shape} → NCHW {nchw.shape}")
# [32, 224, 224, 3] → [32, 3, 224, 224]

# Reverse conversion
nchw_back = nchw.permute(0, 2, 3, 1)
print(f"NCHW {nchw.shape} → NHWC {nchw_back.shape}")
```

## Summary Tables

### Shape and Dimension Attributes

| Attribute/Method | Description | Returns |
|-----------------|-------------|---------|
| `.shape` | Tensor dimensions | `torch.Size` |
| `.size()` | Same as shape | `torch.Size` |
| `.size(dim)` | Size of specific dimension | `int` |
| `.ndim` | Number of dimensions | `int` |
| `.numel()` | Total number of elements | `int` |
| `len(t)` | Size of dim 0 | `int` |

### Data Type and Device Attributes

| Attribute/Method | Description | Returns |
|-----------------|-------------|---------|
| `.dtype` | Data type | `torch.dtype` |
| `.device` | Device (cpu/cuda) | `torch.device` |
| `.is_cuda` | Check if on GPU | `bool` |

### Memory Layout Attributes

| Attribute/Method | Description | Returns |
|-----------------|-------------|---------|
| `.is_contiguous()` | Check memory layout | `bool` |
| `.stride()` | Memory stride per dimension | `tuple` |
| `.storage_offset()` | Offset into storage | `int` |
| `.layout` | Memory layout type | `torch.layout` |

### Autograd Attributes

| Attribute/Method | Description | Returns |
|-----------------|-------------|---------|
| `.requires_grad` | Gradient tracking enabled | `bool` |
| `.is_leaf` | Leaf tensor check | `bool` |
| `.grad_fn` | Gradient function | `Function` or `None` |
| `.grad` | Accumulated gradients | `Tensor` or `None` |

### Transpose Operations

| Attribute | Description | Behavior |
|-----------|-------------|----------|
| `.T` | Simple transpose | Reverses all dims (2D: swap rows/cols) |
| `.mT` | Matrix transpose | Swaps last two dims only |
| `.H` | Hermitian transpose | Conjugate transpose (complex) |

## Key Takeaways

1. **Always verify shapes** when debugging neural networks
2. **Use `.ndim`** to check tensor dimensionality before operations
3. **Monitor `.dtype`** to prevent precision issues
4. **Check `.is_contiguous()`** before using `.view()`
5. **Understand strides** for memory-efficient operations
6. **Track `.requires_grad`** for proper gradient computation
7. **Use `-1` in reshape** to let PyTorch infer one dimension
8. **Know the conventions** — PyTorch uses NCHW for images, (batch, seq, features) for sequences
9. **Use `.mT`** instead of `.T` for batched matrix operations
10. **Add assertions** for expected shapes to catch errors early

## See Also

- [Indexing and Slicing](indexing_slicing.md) — Accessing elements
- [Reshaping and View Operations](reshaping_view.md) — Changing shapes
- [Broadcasting Rules](broadcasting_rules.md) — Implicit shape matching
