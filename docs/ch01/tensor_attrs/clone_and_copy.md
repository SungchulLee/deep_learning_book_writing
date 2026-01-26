# Clone and Copy Operations

Understanding when PyTorch creates copies versus views is crucial for memory management and avoiding subtle bugs. This section covers the various methods for duplicating tensors and their implications.

## Views vs Copies

### What is a View?

A **view** is a tensor that shares storage with another tensor. Changes to one affect the other:

```python
import torch

original = torch.arange(6)
view = original.view(2, 3)

print(f"Same storage: {original.storage().data_ptr() == view.storage().data_ptr()}")
# True

# Modifying view affects original
view[0, 0] = 99
print(f"Original after view modification: {original}")
# tensor([99, 1, 2, 3, 4, 5])
```

### What is a Copy?

A **copy** has its own storage. Changes are independent:

```python
original = torch.arange(6)
copy = original.clone()

print(f"Same storage: {original.storage().data_ptr() == copy.storage().data_ptr()}")
# False

# Modifying copy doesn't affect original
copy[0] = 99
print(f"Original after copy modification: {original}")
# tensor([0, 1, 2, 3, 4, 5])
```

### Quick Test for Shared Storage

```python
def shares_storage(a, b):
    """Check if two tensors share underlying storage."""
    return a.storage().data_ptr() == b.storage().data_ptr()

x = torch.randn(3, 4)
print(shares_storage(x, x.view(-1)))     # True - view
print(shares_storage(x, x.clone()))       # False - copy
print(shares_storage(x, x.T))             # True - view
print(shares_storage(x, x.contiguous()))  # True (already contiguous)
```

## The `clone()` Method

### Basic Cloning

`clone()` creates an independent copy with its own storage:

```python
t = torch.tensor([1.0, 2.0, 3.0])

t_clone = t.clone()

# Verify independence
t_clone[0] = 99.0
print(f"Original: {t}")       # tensor([1., 2., 3.])
print(f"Clone: {t_clone}")    # tensor([99., 2., 3.])
```

### Clone Preserves Gradient Tracking

`clone()` preserves `requires_grad` and creates a node in the computation graph:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.clone()

print(f"x.requires_grad: {x.requires_grad}")  # True
print(f"y.requires_grad: {y.requires_grad}")  # True

# y is connected to x in the computation graph
z = y.sum()
z.backward()
print(f"x.grad: {x.grad}")  # tensor([1., 1., 1.])
```

### Clone Makes Contiguous

Cloning a non-contiguous tensor produces a contiguous copy:

```python
t = torch.randn(3, 4)
t_T = t.T  # Non-contiguous transpose

print(f"Transposed contiguous: {t_T.is_contiguous()}")  # False

clone_T = t_T.clone()
print(f"Clone contiguous: {clone_T.is_contiguous()}")   # True
```

## The `detach()` Method

### Removing from Computation Graph

`detach()` creates a tensor that shares storage but doesn't track gradients:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# Detach from graph
y_detached = y.detach()

print(f"y.requires_grad: {y.requires_grad}")              # True
print(f"y_detached.requires_grad: {y_detached.requires_grad}")  # False

# Still shares storage!
print(f"Same storage: {y.storage().data_ptr() == y_detached.storage().data_ptr()}")
# True
```

### The `detach().clone()` Pattern

For a completely independent copy without gradient tracking:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# Independent copy, no gradients, no shared storage
independent = y.detach().clone()

print(f"requires_grad: {independent.requires_grad}")  # False
print(f"Same storage: {y.storage().data_ptr() == independent.storage().data_ptr()}")
# False
```

**Common use cases:**
- Saving intermediate values for logging without affecting gradients
- Creating numpy arrays from tensors: `tensor.detach().cpu().numpy()`
- Freezing parts of a network during training

## The `copy_()` Method

### In-Place Copy

`copy_()` copies data from source into target in-place:

```python
target = torch.zeros(3)
source = torch.tensor([1.0, 2.0, 3.0])

target.copy_(source)
print(f"Target after copy_: {target}")  # tensor([1., 2., 3.])
```

### Copy with Broadcasting

`copy_()` supports broadcasting:

```python
target = torch.zeros(3, 3)
source = torch.tensor([1.0, 2.0, 3.0])

target.copy_(source)  # Broadcasts source to each row
print(f"Broadcast copy:\n{target}")
# tensor([[1., 2., 3.],
#         [1., 2., 3.],
#         [1., 2., 3.]])
```

### Copy Across Devices

```python
cpu_tensor = torch.randn(3, 4)
gpu_tensor = torch.empty(3, 4, device='cuda')

gpu_tensor.copy_(cpu_tensor)  # Copies CPU → GPU
```

## The `contiguous()` Method

### Making Contiguous

Non-contiguous tensors (from transpose, slicing, etc.) sometimes need to be made contiguous:

```python
t = torch.randn(3, 4)
t_T = t.T

print(f"Transposed contiguous: {t_T.is_contiguous()}")  # False

t_T_cont = t_T.contiguous()
print(f"After contiguous: {t_T_cont.is_contiguous()}")  # True

# Creates new storage only if needed
print(f"Same storage: {t_T.storage().data_ptr() == t_T_cont.storage().data_ptr()}")
# False - new memory allocated
```

### When Contiguous is Required

```python
t = torch.randn(3, 4).T  # Non-contiguous

# view() requires contiguous memory
# t.view(-1)  # RuntimeError!

# Options:
# 1. Make contiguous first
flat1 = t.contiguous().view(-1)

# 2. Use reshape (handles automatically)
flat2 = t.reshape(-1)

# 3. Use flatten
flat3 = t.flatten()
```

### Contiguous on Already-Contiguous Tensor

If the tensor is already contiguous, `contiguous()` returns itself (no copy):

```python
t = torch.randn(3, 4)  # Already contiguous
t_cont = t.contiguous()

print(f"Same object: {t is t_cont}")  # True
print(f"Same storage: {t.storage().data_ptr() == t_cont.storage().data_ptr()}")  # True
```

## Creating Tensors with Same Properties

### `*_like` Functions

Create new tensors with the same dtype, device, and shape:

```python
x = torch.randn(3, 4, dtype=torch.float32, device='cpu')

zeros = torch.zeros_like(x)     # All zeros
ones = torch.ones_like(x)       # All ones
empty = torch.empty_like(x)     # Uninitialized
rand = torch.rand_like(x)       # Uniform [0, 1)
randn = torch.randn_like(x)     # Normal(0, 1)
full = torch.full_like(x, 7.0)  # All 7.0

print(f"All have dtype={zeros.dtype}, device={zeros.device}, shape={zeros.shape}")
```

### `new_*` Methods

Create tensors on the same device with same dtype:

```python
x = torch.randn(3, 4, dtype=torch.float64, device='cuda')

zeros = x.new_zeros(2, 3)       # Shape (2, 3), same dtype/device
ones = x.new_ones(5)            # Shape (5,)
empty = x.new_empty(4, 4)       # Uninitialized
full = x.new_full((2, 2), 3.14) # All 3.14
tensor = x.new_tensor([1, 2, 3]) # From data
```

## Moving Between Devices

### `to()` Creates a Copy

```python
cpu_tensor = torch.randn(3, 4)

# Moving to different device creates a copy
gpu_tensor = cpu_tensor.to('cuda')

print(f"Same storage: {cpu_tensor.storage().data_ptr() == gpu_tensor.storage().data_ptr()}")
# False - different devices, different storage

# Moving to same device/dtype may return self
same_tensor = cpu_tensor.to('cpu')  # May be same object
```

### `to()` with dtype

```python
float_tensor = torch.randn(3, 4)

# Changing dtype creates a copy
double_tensor = float_tensor.to(torch.float64)
int_tensor = float_tensor.to(torch.int32)

print(f"Same storage: {float_tensor.storage().data_ptr() == double_tensor.storage().data_ptr()}")
# False
```

## The `.data` Attribute (Deprecated Pattern)

### What `.data` Does

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# .data gives view without gradient tracking
x_data = x.data

print(f"x.requires_grad: {x.requires_grad}")          # True
print(f"x_data.requires_grad: {x_data.requires_grad}") # False
print(f"Same storage: {x.storage().data_ptr() == x_data.storage().data_ptr()}")
# True
```

### Why to Avoid `.data`

Using `.data` bypasses autograd safety checks and can lead to incorrect gradients:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# Dangerous: modifying through .data
y.data.add_(1)  # Bypasses version counter!

# backward() may now give wrong gradients
```

**Prefer `detach()`** for creating tensors without gradient tracking.

## Practical Patterns

### Safe Logging During Training

```python
def log_activations(tensor, name):
    """Log tensor statistics without affecting gradients."""
    with torch.no_grad():
        t = tensor.detach()
        print(f"{name}: mean={t.mean():.4f}, std={t.std():.4f}")
```

### Converting to NumPy

```python
# For tensor without gradients
t = torch.randn(3, 4)
arr = t.numpy()  # Shares memory with tensor

# For tensor with gradients
t_grad = torch.randn(3, 4, requires_grad=True)
arr = t_grad.detach().cpu().numpy()  # Safe conversion

# For GPU tensor
t_gpu = torch.randn(3, 4, device='cuda')
arr = t_gpu.cpu().numpy()
```

### Efficient Batch Assembly

```python
# Pre-allocate and copy for efficiency
batch_size = 32
feature_dim = 256

batch = torch.empty(batch_size, feature_dim)
for i, sample in enumerate(samples[:batch_size]):
    batch[i].copy_(sample)  # In-place copy into pre-allocated
```

### Freezing Part of a Model

```python
def forward_with_frozen_encoder(self, x):
    # Encoder output doesn't contribute to gradients
    with torch.no_grad():
        encoded = self.encoder(x)
    
    # Decoder still trains
    return self.decoder(encoded)

# Or using detach
def forward_with_frozen_encoder(self, x):
    encoded = self.encoder(x).detach()  # Breaks gradient flow
    return self.decoder(encoded)
```

### Saving Checkpoints

```python
def save_for_inference(model, path):
    """Save model state without gradient info."""
    state_dict = {
        k: v.detach().clone() 
        for k, v in model.state_dict().items()
    }
    torch.save(state_dict, path)
```

## Quick Reference

| Operation | Creates Copy? | Shares Storage? | Gradient Tracking |
|-----------|---------------|-----------------|-------------------|
| `view()`, `reshape()`* | No | Yes | Preserved |
| `clone()` | Yes | No | Preserved |
| `detach()` | No | Yes | Removed |
| `detach().clone()` | Yes | No | Removed |
| `contiguous()` | If needed | No (if copied) | Preserved |
| `to(device)` | Yes | No | Preserved |
| `to(dtype)` | Yes | No | Preserved |
| `copy_(src)` | No (in-place) | N/A | N/A |
| `.data` | No | Yes | Removed (unsafe) |

*`reshape()` may or may not create a copy depending on memory layout.

## Decision Guide

**Need an independent copy?**
- With gradients: `tensor.clone()`
- Without gradients: `tensor.detach().clone()`

**Need to stop gradient flow?**
- Keep storage: `tensor.detach()`
- New storage: `tensor.detach().clone()`

**Need contiguous memory?**
- `tensor.contiguous()` (copies only if needed)

**Need same properties, different values?**
- `torch.zeros_like(tensor)`, `torch.randn_like(tensor)`, etc.

**Need to copy into existing tensor?**
- `target.copy_(source)`

## Key Takeaways

1. **Views share storage** — modifying one affects the other
2. **`clone()` creates independent copy** with gradient tracking
3. **`detach()` removes from graph** but shares storage
4. **`detach().clone()` for fully independent** non-gradient copy
5. **`contiguous()` copies only when needed** — returns self if already contiguous
6. **Avoid `.data`** — use `detach()` instead
7. **`to(device/dtype)` creates copies** — different memory locations
8. **Use `*_like` functions** to create tensors with matching properties

## See Also

- [In-place Operations](inplace_operations.md) - Modifying tensors in-place
- [Memory Layout and Strides](../tensors/memory_layout_strides.md) - Storage details
- [Reshaping and View Operations](reshaping_view.md) - View operations
