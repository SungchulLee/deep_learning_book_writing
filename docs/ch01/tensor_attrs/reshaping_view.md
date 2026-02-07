# Reshaping and View Operations

Reshaping operations are fundamental to deep learning, enabling data transformations between layers and processing stages. Understanding the distinction between views and copies is crucial for both correctness and performance—a misunderstanding here can lead to subtle bugs or significant memory overhead.

## Memory Model: Views vs Copies

Before diving into operations, understanding PyTorch's memory model is essential.

### What is a View?

A **view** is a tensor that shares the same underlying storage as another tensor but interprets it with a different shape or stride. No data is copied—both tensors point to the same memory.

```python
import torch

x = torch.arange(12)
y = x.view(3, 4)  # View of x

# Same underlying storage
print(x.storage().data_ptr() == y.storage().data_ptr())  # True

# Modifying y affects x
y[0, 0] = 999
print(x[0])  # 999

# They share the exact same storage object
print(x.storage())  # [999, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

### What is a Copy?

A **copy** creates new storage with duplicated data. Modifications are completely independent.

```python
x = torch.arange(12).reshape(3, 4)
y = x.clone()  # Explicit copy

# Different storage
print(x.storage().data_ptr() == y.storage().data_ptr())  # False

y[0, 0] = 999
print(x[0, 0])  # 0 (unchanged)
```

### The Storage-Stride-Offset Model

Every tensor is defined by three components:

```python
t = torch.arange(12).reshape(3, 4)

# 1. Storage: The actual data buffer
print(t.storage())  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 2. Shape: Logical dimensions
print(t.shape)  # torch.Size([3, 4])

# 3. Strides: Steps in storage to move one element in each dimension
print(t.stride())  # (4, 1)

# 4. Storage offset: Where this tensor's data starts
print(t.storage_offset())  # 0
```

**Stride interpretation**: For a tensor with stride `(4, 1)`:
- Moving one step in dimension 0 (next row) skips 4 elements in storage
- Moving one step in dimension 1 (next column) skips 1 element in storage

This model explains why certain operations create views and others require copies.

## Core Reshaping Operations

### `reshape()` — Universal Reshaping

The safest and most flexible reshaping method. Returns a view when possible, creates a copy when necessary.

```python
vec = torch.arange(12)
print(f"Original: {vec.shape}")  # torch.Size([12])

# Reshape to various dimensions
mat = vec.reshape(3, 4)
print(f"3×4 matrix:\n{mat}")

cube = vec.reshape(2, 2, 3)
print(f"2×2×3 cube shape: {cube.shape}")

# Total elements must match
# vec.reshape(3, 5)  # RuntimeError: shape '[3, 5]' is invalid for input of size 12
```

**When does `reshape()` copy?**

```python
# View case: contiguous tensor
x = torch.arange(12)
y = x.reshape(3, 4)
print(y.storage().data_ptr() == x.storage().data_ptr())  # True (view)

# Copy case: non-contiguous tensor
x = torch.arange(12).reshape(3, 4).T  # Transpose makes it non-contiguous
y = x.reshape(6, 2)
print(y.storage().data_ptr() == x.storage().data_ptr())  # False (copy)
```

### `view()` — Fast Reshaping for Contiguous Tensors

`view()` is stricter than `reshape()`: it **always** returns a view and fails if this is impossible.

```python
vec = torch.arange(12)

# Works on contiguous tensors
mat = vec.view(3, 4)
print(f"View shape: {mat.shape}")

# Transpose creates non-contiguous tensor
mat_T = mat.T
print(f"Is contiguous: {mat_T.is_contiguous()}")  # False

# view() fails on non-contiguous tensors
try:
    mat_T.view(-1)
except RuntimeError as e:
    print(f"Error: view size is not compatible with input tensor's size and stride")
```

**Solution for non-contiguous tensors:**

```python
# Option 1: Make contiguous first (creates copy)
flat = mat_T.contiguous().view(-1)

# Option 2: Use reshape (handles it automatically)
flat = mat_T.reshape(-1)
```

### `reshape()` vs `view()` Decision Guide

| Scenario | Recommendation | Reason |
|----------|----------------|--------|
| Performance-critical, known contiguous | `view()` | Guaranteed no copy, fails fast if assumption wrong |
| Unsure about contiguity | `reshape()` | Handles both cases automatically |
| Need guaranteed no-copy | `view()` | Will fail rather than silently copy |
| General-purpose code | `reshape()` | More robust |
| Debugging memory issues | `view()` | Makes contiguity problems explicit |

```python
# Best practice: use reshape() unless you need guaranteed view behavior
mat = torch.randn(3, 4)
flat = mat.reshape(-1)  # Safe for any tensor

# Use view() when you want to catch non-contiguous tensors
def process_contiguous_only(x):
    """Process tensor that must be contiguous."""
    return x.view(-1)  # Will fail if x is non-contiguous
```

## Automatic Size Inference with -1

Use `-1` to let PyTorch compute one dimension automatically:

```python
vec = torch.arange(24)

# Infer number of rows: 24 / 4 = 6
auto_rows = vec.reshape(-1, 4)
print(f"reshape(-1, 4): {auto_rows.shape}")  # torch.Size([6, 4])

# Infer number of columns: 24 / 3 = 8
auto_cols = vec.reshape(3, -1)
print(f"reshape(3, -1): {auto_cols.shape}")  # torch.Size([3, 8])

# Infer middle dimension: 24 / (2 * 4) = 3
auto_mid = vec.reshape(2, -1, 4)
print(f"reshape(2, -1, 4): {auto_mid.shape}")  # torch.Size([2, 3, 4])

# Flatten to 1D
flat = vec.reshape(-1)
print(f"reshape(-1): {flat.shape}")  # torch.Size([24])

# Only one -1 allowed per reshape
# vec.reshape(-1, -1)  # RuntimeError: only one dimension can be inferred
```

**Why only one `-1`?** With two unknowns, the equation `a × b = total` has infinite solutions.

## Adding and Removing Dimensions

### `unsqueeze()` — Add Dimension of Size 1

```python
vec = torch.tensor([1, 2, 3, 4, 5])
print(f"Original: {vec.shape}")  # torch.Size([5])

# Add dimension at position 0 (row vector)
row = vec.unsqueeze(0)
print(f"unsqueeze(0): {row.shape}")  # torch.Size([1, 5])

# Add dimension at position 1 (column vector)
col = vec.unsqueeze(1)
print(f"unsqueeze(1): {col.shape}")  # torch.Size([5, 1])

# Add at end with -1
end = vec.unsqueeze(-1)
print(f"unsqueeze(-1): {end.shape}")  # torch.Size([5, 1])

# Multiple unsqueezes
expanded = vec.unsqueeze(0).unsqueeze(2)
print(f"Double unsqueeze: {expanded.shape}")  # torch.Size([1, 5, 1])
```

### Alternative: Indexing with `None`

More concise for complex patterns:

```python
vec = torch.tensor([1, 2, 3, 4, 5])

# These are equivalent
row1 = vec.unsqueeze(0)
row2 = vec[None, :]
print(torch.equal(row1, row2))  # True

col1 = vec.unsqueeze(1)
col2 = vec[:, None]
print(torch.equal(col1, col2))  # True

# Complex patterns are clearer with None
expanded = vec[None, :, None, None]
print(f"Multi-None: {expanded.shape}")  # torch.Size([1, 5, 1, 1])
```

### `squeeze()` — Remove Dimensions of Size 1

```python
t = torch.randn(1, 5, 1, 3, 1)
print(f"Original: {t.shape}")  # torch.Size([1, 5, 1, 3, 1])

# Remove ALL size-1 dimensions
squeezed_all = t.squeeze()
print(f"squeeze(): {squeezed_all.shape}")  # torch.Size([5, 3])

# Remove specific dimension (only if size 1)
squeezed_0 = t.squeeze(0)
print(f"squeeze(0): {squeezed_0.shape}")  # torch.Size([5, 1, 3, 1])

squeezed_2 = t.squeeze(2)
print(f"squeeze(2): {squeezed_2.shape}")  # torch.Size([1, 5, 3, 1])

# No effect if dimension size > 1
t2 = torch.randn(2, 3)
print(f"squeeze() on (2,3): {t2.squeeze().shape}")  # torch.Size([2, 3])
```

**Caution with `squeeze()`:** Removing all size-1 dimensions can cause bugs when batch size is 1:

```python
def forward(self, x):
    # x shape: (batch, features) or (1, features) for single sample
    output = self.model(x)
    return output.squeeze()  # Dangerous! Shape varies with batch size

# Better: be explicit
def forward(self, x):
    output = self.model(x)
    return output.squeeze(1)  # Only squeeze dimension 1
```

## Flattening Operations

### `flatten()` — Collapse Dimensions

```python
t = torch.arange(24).reshape(2, 3, 4)
print(f"Original: {t.shape}")  # torch.Size([2, 3, 4])

# Flatten all dimensions
flat_all = t.flatten()
print(f"flatten(): {flat_all.shape}")  # torch.Size([24])

# Flatten from start_dim (keep batch dimension)
flat_1 = t.flatten(start_dim=1)
print(f"flatten(start_dim=1): {flat_1.shape}")  # torch.Size([2, 12])

# Flatten range of dimensions
flat_range = t.flatten(start_dim=0, end_dim=1)
print(f"flatten(0, 1): {flat_range.shape}")  # torch.Size([6, 4])
```

**Critical use case**: CNN to fully-connected transition:

```python
# CNN features: (batch, channels, height, width)
features = torch.randn(32, 128, 7, 7)

# Flatten spatial dimensions for FC layer (keep batch)
flat = features.flatten(start_dim=1)
print(flat.shape)  # torch.Size([32, 6272])
```

### `ravel()` — Flatten to Contiguous 1D

```python
x = torch.arange(12).reshape(3, 4)
flat = x.ravel()  # Similar to flatten()
print(flat.shape)  # torch.Size([12])

# Difference: ravel() on non-contiguous always returns contiguous
x_T = x.T
print(x_T.ravel().is_contiguous())  # True
```

### Flattening Methods Compared

```python
t = torch.randn(2, 3, 4)

# All produce shape (24,) for full flatten
flat1 = t.reshape(-1)      # View if possible, copy if needed
flat2 = t.view(-1)         # View only (fails if non-contiguous)
flat3 = t.flatten()        # View if contiguous
flat4 = t.ravel()          # View if contiguous, else contiguous copy
```

## Transposition and Permutation

### `transpose()` — Swap Two Dimensions

```python
mat = torch.arange(12).reshape(3, 4)
print(f"Original:\n{mat}")
print(f"Stride: {mat.stride()}")  # (4, 1)

# Transpose (swap dims 0 and 1)
mat_T = mat.transpose(0, 1)
print(f"Transposed:\n{mat_T}")
print(f"Stride: {mat_T.stride()}")  # (1, 4) — strides swapped!
print(f"Contiguous: {mat_T.is_contiguous()}")  # False

# Shorthand for 2D matrices
mat_T2 = mat.T
print(torch.equal(mat_T, mat_T2))  # True
```

**Why does transpose make tensors non-contiguous?**

The underlying storage is unchanged—only strides are swapped. Data is no longer in row-major order:

```python
mat = torch.arange(6).reshape(2, 3)
# Storage: [0, 1, 2, 3, 4, 5]
# mat[0] = [0, 1, 2], mat[1] = [3, 4, 5]

mat_T = mat.T
# Same storage: [0, 1, 2, 3, 4, 5]
# mat_T[0] = [0, 3], mat_T[1] = [1, 4], mat_T[2] = [2, 5]
# Elements of each row are not adjacent in storage → non-contiguous
```

### `permute()` — Rearrange All Dimensions

```python
# 4D tensor (batch, channels, height, width)
t = torch.randn(2, 3, 4, 5)
print(f"Original: {t.shape}")  # torch.Size([2, 3, 4, 5])

# Reorder dimensions
permuted = t.permute(3, 1, 0, 2)
print(f"permute(3,1,0,2): {permuted.shape}")  # torch.Size([5, 3, 2, 4])
```

**Common use case**: Image format conversion:

```python
# HWC (Height, Width, Channels) → CHW (Channels, Height, Width)
img_hwc = torch.randn(224, 224, 3)
img_chw = img_hwc.permute(2, 0, 1)
print(img_chw.shape)  # torch.Size([3, 224, 224])

# Batch version: NHWC → NCHW (TensorFlow → PyTorch format)
batch_hwc = torch.randn(32, 224, 224, 3)
batch_chw = batch_hwc.permute(0, 3, 1, 2)
print(batch_chw.shape)  # torch.Size([32, 3, 224, 224])

# Reverse: NCHW → NHWC (PyTorch → TensorFlow format)
batch_hwc_back = batch_chw.permute(0, 2, 3, 1)
print(batch_hwc_back.shape)  # torch.Size([32, 224, 224, 3])
```

### `movedim()` — Move Dimensions to New Positions

More intuitive than `permute()` for simple moves:

```python
x = torch.randn(2, 3, 4, 5)

# Move dim 1 to position 3
moved = torch.movedim(x, 1, 3)
print(moved.shape)  # torch.Size([2, 4, 5, 3])

# Move multiple dimensions
moved2 = torch.movedim(x, [0, 1], [2, 3])
print(moved2.shape)  # torch.Size([4, 5, 2, 3])
```

### `swapdims()` — Alias for `transpose()`

```python
x = torch.randn(2, 3, 4)
swapped = torch.swapdims(x, 0, 2)
print(swapped.shape)  # torch.Size([4, 3, 2])
```

## Contiguity Deep Dive

### What Makes a Tensor Contiguous?

A tensor is contiguous when its elements are stored in memory in the same order as they would be accessed by iterating through the tensor in row-major order.

```python
# Contiguous: elements are sequential in storage
t = torch.arange(12).reshape(3, 4)
print(t.is_contiguous())  # True
print(t.stride())  # (4, 1)

# After transpose: same storage, different access pattern
t_T = t.T
print(t_T.is_contiguous())  # False
print(t_T.stride())  # (1, 4)

# The mathematical condition:
# stride[i] == stride[i+1] * size[i+1] for all i
# For t: stride[0]=4 == stride[1]*size[1] = 1*4 ✓
# For t_T: stride[0]=1 != stride[1]*size[1] = 4*3 ✗
```

### Making Tensors Contiguous

```python
t = torch.arange(12).reshape(3, 4).T

print(t.is_contiguous())  # False

# contiguous() creates a copy with proper layout
t_cont = t.contiguous()
print(t_cont.is_contiguous())  # True

# New storage (copy was made)
print(t.storage().data_ptr() == t_cont.storage().data_ptr())  # False
```

### Operations That Require Contiguity

```python
t = torch.arange(12).reshape(3, 4).T

# view() requires contiguous
# t.view(-1)  # RuntimeError

# Some NumPy conversions
# t.numpy()  # May fail or create copy

# Certain CUDA operations may be slower on non-contiguous tensors
```

## Expand and Repeat

### `expand()` — Broadcast Without Copying

Creates a view with size-1 dimensions expanded using stride tricks:

```python
x = torch.tensor([1., 2., 3.])  # Shape: (3,)

# Expand to (4, 3) — no memory allocation!
expanded = x.expand(4, 3)
print(expanded.shape)  # torch.Size([4, 3])
print(expanded.stride())  # (0, 1) — stride 0 means same data repeated

# Verify: same storage
print(x.storage().data_ptr() == expanded.storage().data_ptr())  # True

# All rows are the same data
print(expanded)
# tensor([[1., 2., 3.],
#         [1., 2., 3.],
#         [1., 2., 3.],
#         [1., 2., 3.]])
```

**Warning**: Never modify expanded tensors in-place:

```python
x = torch.tensor([1., 2., 3.])
expanded = x.expand(4, 3)

# This modifies ALL rows because they share data!
# expanded[0, 0] = 999  # Undefined behavior!

# If you need to modify, clone first
expanded_copy = expanded.clone()
expanded_copy[0, 0] = 999  # Safe
```

### `repeat()` — Actually Copy Data

```python
x = torch.tensor([1., 2., 3.])  # Shape: (3,)

# Repeat 2 times along new dim 0, 3 times along dim 1
repeated = x.repeat(2, 3)
print(repeated.shape)  # torch.Size([2, 9])
print(repeated)
# tensor([[1., 2., 3., 1., 2., 3., 1., 2., 3.],
#         [1., 2., 3., 1., 2., 3., 1., 2., 3.]])

# Data is actually duplicated
print(repeated.storage().size())  # 18 (2 * 9 elements)
```

### `expand()` vs `repeat()` Decision Guide

| Operation | Memory | Modifiable | Use Case |
|-----------|--------|------------|----------|
| `expand()` | No additional | No (dangerous) | Broadcasting for computation |
| `repeat()` | Copies data | Yes | When actual copies needed |

```python
# Use expand() for efficient broadcasting
bias = torch.randn(1, 128)
batch_bias = bias.expand(32, 128)  # Free, no memory allocated

# Use repeat() when you need independent copies
templates = torch.randn(1, 100)
copies = templates.repeat(10, 1)  # 10 independent copies
```

### `expand_as()` and `repeat_interleave()`

```python
# expand_as: match another tensor's shape
x = torch.randn(1, 3)
y = torch.randn(4, 3)
x_expanded = x.expand_as(y)
print(x_expanded.shape)  # torch.Size([4, 3])

# repeat_interleave: repeat elements along a dimension
x = torch.tensor([1, 2, 3])
repeated = x.repeat_interleave(2)
print(repeated)  # tensor([1, 1, 2, 2, 3, 3])

# With different repeat counts per element
x = torch.tensor([1, 2, 3])
repeated = x.repeat_interleave(torch.tensor([1, 2, 3]))
print(repeated)  # tensor([1, 2, 2, 3, 3, 3])
```

## Clone and Detach

### `clone()` — Create Independent Copy

```python
x = torch.randn(3, requires_grad=True)
y = x.clone()  # Independent copy

# Clone preserves requires_grad and gradient flows through
print(y.requires_grad)  # True

z = y.sum()
z.backward()
print(x.grad is not None)  # True — gradient flows through clone
```

### `detach()` — Remove from Computation Graph

```python
x = torch.randn(3, requires_grad=True)
y = x.detach()  # Shares storage, no gradient

print(y.requires_grad)  # False

# y shares data with x (view!)
y[0] = 999
print(x[0])  # tensor(999.) — both changed!
```

### `detach().clone()` — Snapshot Without Gradient

The standard pattern for creating a gradient-free, independent copy:

```python
x = torch.randn(3, requires_grad=True)
snapshot = x.detach().clone()  # Independent copy, no gradient

print(snapshot.requires_grad)  # False

# Modifications don't affect original
snapshot[0] = 999
print(x[0] != 999)  # True — independent

# Common in training loops
with torch.no_grad():
    best_weights = model.weights.detach().clone()
```

## Understanding Strides

Strides are the key to understanding view operations:

```python
# Basic stride interpretation
x = torch.arange(12).reshape(3, 4)
print(f"Shape: {x.shape}, Stride: {x.stride()}")
# Shape: torch.Size([3, 4]), Stride: (4, 1)

# Access pattern: x[i, j] = storage[i*4 + j*1]
print(x[1, 2])  # storage[1*4 + 2*1] = storage[6] = 6

# After transpose: strides swap
xt = x.t()
print(f"Shape: {xt.shape}, Stride: {xt.stride()}")
# Shape: torch.Size([4, 3]), Stride: (1, 4)

# Access pattern: xt[i, j] = storage[i*1 + j*4]
print(xt[2, 1])  # storage[2*1 + 1*4] = storage[6] = 6 (same element!)

# After expand: stride 0 for broadcasted dimensions
y = torch.tensor([1, 2, 3])
ye = y.expand(4, 3)
print(f"Shape: {ye.shape}, Stride: {ye.stride()}")
# Shape: torch.Size([4, 3]), Stride: (0, 1)

# Access pattern: ye[i, j] = storage[i*0 + j*1] = storage[j]
# Row index doesn't matter — all rows are the same!
```

### Stride-Based View Detection

```python
def is_view_of(a, b):
    """Check if a is a view of b (shares storage)."""
    return a.storage().data_ptr() == b.storage().data_ptr()

x = torch.arange(12)
y = x.view(3, 4)
z = x.reshape(3, 4)
w = x.clone().view(3, 4)

print(is_view_of(y, x))  # True
print(is_view_of(z, x))  # True (reshape returned view)
print(is_view_of(w, x))  # False (clone created new storage)
```

## Common Patterns

### Adding/Removing Batch Dimension

```python
# Single image → batch of 1
image = torch.randn(3, 224, 224)
batched = image.unsqueeze(0)
print(batched.shape)  # torch.Size([1, 3, 224, 224])

# After inference, remove batch dimension
output = torch.randn(1, 1000)  # Single prediction
prediction = output.squeeze(0)
print(prediction.shape)  # torch.Size([1000])
```

### CNN to FC Layer Transition

```python
# After convolutional layers, before fully connected
conv_output = torch.randn(32, 64, 7, 7)  # batch, channels, H, W

# Method 1: reshape
flat = conv_output.reshape(32, -1)
print(flat.shape)  # torch.Size([32, 3136])

# Method 2: flatten (preferred — more explicit)
flat = conv_output.flatten(start_dim=1)
print(flat.shape)  # torch.Size([32, 3136])
```

### Broadcasting Preparation

```python
# Prepare tensors for element-wise operations
features = torch.randn(100)  # 1D feature vector

# Need to broadcast with (batch, features) shape
row = features.unsqueeze(0)  # (1, 100)
# or
row = features[None, :]  # (1, 100)

# Both broadcast correctly with (32, 100)
batch = torch.randn(32, 100)
result = batch + row  # Broadcasting adds row to each batch element
print(result.shape)  # torch.Size([32, 100])
```

### Multi-Head Attention Reshape

```python
# Standard transformer reshape pattern
batch, seq_len, d_model = 32, 50, 512
num_heads, d_k = 8, 64

x = torch.randn(batch, seq_len, d_model)

# Step 1: Reshape to separate heads
# (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
x_heads = x.reshape(batch, seq_len, num_heads, d_k)

# Step 2: Transpose for parallel attention computation
# (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
x_heads = x_heads.permute(0, 2, 1, 3)
print(x_heads.shape)  # torch.Size([32, 8, 50, 64])

# After attention, reverse the process
# (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
x_combined = x_heads.permute(0, 2, 1, 3)
# (batch, seq_len, num_heads, d_k) → (batch, seq_len, d_model)
x_out = x_combined.reshape(batch, seq_len, d_model)
```

### Vision Transformer Patch Embedding

```python
# Reshape image into patches for ViT
batch, channels, height, width = 8, 3, 224, 224
patch_size = 16
num_patches = (height // patch_size) * (width // patch_size)  # 196

images = torch.randn(batch, channels, height, width)

# Reshape to patches: (B, C, H, W) → (B, num_patches, patch_dim)
# Step 1: Reshape to grid of patches
patches = images.reshape(batch, channels, 
                         height // patch_size, patch_size,
                         width // patch_size, patch_size)
# Shape: (8, 3, 14, 16, 14, 16)

# Step 2: Rearrange dimensions
patches = patches.permute(0, 2, 4, 1, 3, 5)
# Shape: (8, 14, 14, 3, 16, 16)

# Step 3: Flatten to sequence of patch embeddings
patches = patches.reshape(batch, num_patches, -1)
# Shape: (8, 196, 768)  where 768 = 3 * 16 * 16
```

### Combining Spatial Dimensions for Attention

```python
# Flatten spatial dimensions for self-attention
features = torch.randn(8, 256, 14, 14)  # (batch, channels, H, W)

# Method 1: flatten + permute
flat = features.flatten(2)  # (8, 256, 196)
seq = flat.permute(0, 2, 1)  # (8, 196, 256) — (batch, seq_len, channels)

# Method 2: reshape + permute
flat = features.reshape(8, 256, -1)  # (8, 256, 196)
seq = flat.permute(0, 2, 1)  # (8, 196, 256)
```

## Performance Tips

### 1. Prefer Views Over Copies

```python
# Good: view (no allocation)
x = torch.randn(1000, 1000)
flat = x.view(-1)

# Slower: unnecessary copy
flat = x.clone().view(-1)
```

### 2. Use `contiguous()` Sparingly

```python
# Only call contiguous() when actually needed
t = some_transpose_operation()

# Bad: always calling contiguous
t_cont = t.contiguous()  # May be unnecessary copy

# Good: only when required
if not t.is_contiguous():
    t = t.contiguous()

# Better: use reshape which handles it
t = t.reshape(desired_shape)
```

### 3. Chain Operations to Avoid Intermediates

```python
# Good: single chain
result = x.flatten(1).unsqueeze(0)

# Less efficient: named intermediates
y = x.flatten(1)
z = y.unsqueeze(0)
```

### 4. Be Aware of Contiguity After Permute/Transpose

```python
x = torch.randn(32, 3, 224, 224)
x = x.permute(0, 2, 3, 1)  # Now non-contiguous

# If you need to view after this, make contiguous
x = x.contiguous()
# or use reshape which handles it
```

### 5. Memory Layout for GPU Performance

```python
# Contiguous memory access is faster on GPU
# After permute, consider making contiguous for subsequent operations
x = x.permute(0, 2, 3, 1).contiguous()

# Use channels_last memory format for CNNs (PyTorch 1.5+)
x = x.to(memory_format=torch.channels_last)
```

## Quick Reference

| Operation | Creates View? | Notes |
|-----------|--------------|-------|
| `view()` | Always | Requires contiguous |
| `reshape()` | Usually | Copies if non-contiguous |
| `flatten()` | Usually | Copies if non-contiguous |
| `ravel()` | Usually | Returns contiguous result |
| `squeeze()` | Always | |
| `unsqueeze()` | Always | |
| `transpose()` | Always | Makes non-contiguous |
| `permute()` | Always | Makes non-contiguous |
| `expand()` | Always | Uses stride tricks |
| `repeat()` | Never | Always copies |
| `clone()` | Never | Explicit copy |
| `contiguous()` | Maybe | Copies if non-contiguous |

## Key Takeaways

1. **Views share memory** — modifications affect all tensors sharing storage
2. **`view()` requires contiguity**, `reshape()` handles both cases
3. **`transpose()`/`permute()` create views** with different strides, making tensors non-contiguous
4. **Use `-1`** for automatic dimension inference (only one per operation)
5. **`expand()` is memory-free** (uses stride tricks), `repeat()` copies data
6. **Never modify `expand()`ed tensors in-place** — undefined behavior
7. **Check `is_contiguous()`** when debugging memory/performance issues
8. **`squeeze()`/`unsqueeze()`** for adding/removing singleton dimensions
9. **`detach().clone()`** is the standard pattern for gradient-free snapshots
10. **Document shape transformations** in comments for maintainability

## See Also

- [Dtype and Device](dtype_device.md) — Data type and device attributes
- [Shape Manipulation](shape_manipulation.md) — Indexing, concatenation, and splitting
- [Broadcasting Rules](broadcasting_rules.md) — Implicit expansion
- [Memory Layout and Strides](memory_layout_strides.md) — Deep dive into memory
- [Memory Management](memory_management.md) — Views, copies, and GPU memory
