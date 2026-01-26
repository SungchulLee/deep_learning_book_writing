# Concatenation and Stacking

Combining tensors is fundamental to deep learning workflows—batching samples, fusing features from multiple sources, and assembling outputs. Understanding the distinction between `cat` and `stack` is essential: they serve different purposes and have different requirements.

## Core Concept: cat vs stack

| Operation | Behavior | Dimension Change | Shape Requirement |
|-----------|----------|------------------|-------------------|
| `cat` | Joins along **existing** dimension | Same ndim | Match all dims except concat dim |
| `stack` | Joins along **new** dimension | Adds one dimension | ALL dimensions must match exactly |

```python
import torch

# Visual comparison with 1D tensors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(f"Original shapes: {a.shape}, {b.shape}")  # [3], [3]

# cat: joins along existing dimension (extends it)
catted = torch.cat([a, b], dim=0)
print(f"cat result: {catted}")        # tensor([1, 2, 3, 4, 5, 6])
print(f"cat shape: {catted.shape}")   # torch.Size([6])

# stack: creates new dimension (groups tensors)
stacked = torch.stack([a, b], dim=0)
print(f"stack result:\n{stacked}")
# tensor([[1, 2, 3],
#         [4, 5, 6]])
print(f"stack shape: {stacked.shape}")  # torch.Size([2, 3])
```

**Mental model**:
- `cat` = "glue tensors end-to-end along a dimension"
- `stack` = "arrange tensors as layers along a new dimension"

## Concatenation with `torch.cat()`

### Basic Concatenation

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Concatenate along dimension 0 (vertically - more rows)
cat_dim0 = torch.cat([a, b], dim=0)
print(f"cat dim=0:\n{cat_dim0}")
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])
print(f"Shape: {cat_dim0.shape}")  # torch.Size([4, 2])

# Concatenate along dimension 1 (horizontally - more columns)
cat_dim1 = torch.cat([a, b], dim=1)
print(f"cat dim=1:\n{cat_dim1}")
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])
print(f"Shape: {cat_dim1.shape}")  # torch.Size([2, 4])
```

### Multiple Tensors

```python
t1 = torch.randn(2, 3)
t2 = torch.randn(2, 3)
t3 = torch.randn(2, 3)

# Concatenate any number of tensors
combined = torch.cat([t1, t2, t3], dim=0)
print(f"Combined shape: {combined.shape}")  # torch.Size([6, 3])
```

### Shape Requirements for cat

All tensors must have the same shape **except** in the concatenation dimension:

```python
a = torch.randn(2, 3, 4)
b = torch.randn(5, 3, 4)  # Different size in dim 0

# Works: concatenating along dim 0 (where sizes differ)
result = torch.cat([a, b], dim=0)
print(f"Shape: {result.shape}")  # torch.Size([7, 3, 4])

# Fails: other dimensions must match
c = torch.randn(2, 5, 4)  # Different size in dim 1
try:
    torch.cat([a, c], dim=0)  # Error!
except RuntimeError as e:
    print(f"Error: Sizes of tensors must match except in dimension 0")
```

### Negative Dimension Indexing

```python
t = torch.randn(2, 3, 4)
u = torch.randn(2, 3, 4)

# dim=-1 means last dimension
combined = torch.cat([t, u], dim=-1)
print(combined.shape)  # torch.Size([2, 3, 8])

# dim=-2 means second-to-last
combined = torch.cat([t, u], dim=-2)
print(combined.shape)  # torch.Size([2, 6, 4])
```

## Stacking with `torch.stack()`

### Basic Stacking

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.tensor([7, 8, 9])

# Stack along new dimension 0 (creates "batch" dimension)
stacked_0 = torch.stack([a, b, c], dim=0)
print(f"stack dim=0:\n{stacked_0}")
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])
print(f"Shape: {stacked_0.shape}")  # torch.Size([3, 3])

# Stack along new dimension 1 (interleaves elements)
stacked_1 = torch.stack([a, b, c], dim=1)
print(f"stack dim=1:\n{stacked_1}")
# tensor([[1, 4, 7],
#         [2, 5, 8],
#         [3, 6, 9]])
print(f"Shape: {stacked_1.shape}")  # torch.Size([3, 3])
```

### Understanding stack Dimension Placement

```python
a = torch.randn(3, 4)  # Shape: (3, 4)
b = torch.randn(3, 4)

# dim=0: new dimension at position 0
s0 = torch.stack([a, b], dim=0)
print(f"dim=0: {s0.shape}")  # torch.Size([2, 3, 4])

# dim=1: new dimension at position 1
s1 = torch.stack([a, b], dim=1)
print(f"dim=1: {s1.shape}")  # torch.Size([3, 2, 4])

# dim=2 (or -1): new dimension at position 2
s2 = torch.stack([a, b], dim=2)
print(f"dim=2: {s2.shape}")  # torch.Size([3, 4, 2])
```

### Shape Requirements for stack

**All tensors must have exactly the same shape**:

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = torch.randn(3, 4)

# All same shape — works
stacked = torch.stack([a, b, c], dim=0)
print(f"Shape: {stacked.shape}")  # torch.Size([3, 3, 4])

# Different shapes — fails
d = torch.randn(3, 5)  # Different!
try:
    torch.stack([a, d], dim=0)
except RuntimeError:
    print("Error: stack expects each tensor to be equal size")
```

### stack as unsqueeze + cat

Under the hood, `stack` is equivalent to unsqueezing each tensor then concatenating:

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)

# stack dim=0 is equivalent to:
stacked = torch.stack([a, b], dim=0)
manual = torch.cat([a.unsqueeze(0), b.unsqueeze(0)], dim=0)

print(torch.equal(stacked, manual))  # True

# stack dim=1 is equivalent to:
stacked = torch.stack([a, b], dim=1)
manual = torch.cat([a.unsqueeze(1), b.unsqueeze(1)], dim=1)

print(torch.equal(stacked, manual))  # True
```

## Specialized Stacking Functions

### `vstack` — Vertical Stack

Stacks along dimension 0 (adds rows):

```python
# 1D inputs: treats them as row vectors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

vstacked = torch.vstack([a, b])
print(f"vstack 1D:\n{vstacked}")
# tensor([[1, 2, 3],
#         [4, 5, 6]])
print(f"Shape: {vstacked.shape}")  # torch.Size([2, 3])

# 2D inputs: same as cat(dim=0)
m1 = torch.randn(2, 3)
m2 = torch.randn(4, 3)  # Different number of rows OK
vstacked_2d = torch.vstack([m1, m2])
print(f"vstack 2D shape: {vstacked_2d.shape}")  # torch.Size([6, 3])
```

### `hstack` — Horizontal Stack

Stacks along dimension 1 (adds columns):

```python
# 1D inputs: concatenates (no new dimension)
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

hstacked = torch.hstack([a, b])
print(f"hstack 1D: {hstacked}")  # tensor([1, 2, 3, 4, 5, 6])
print(f"Shape: {hstacked.shape}")  # torch.Size([6])

# 2D inputs: same as cat(dim=1)
m1 = torch.randn(3, 2)
m2 = torch.randn(3, 4)  # Different number of columns OK
hstacked_2d = torch.hstack([m1, m2])
print(f"hstack 2D shape: {hstacked_2d.shape}")  # torch.Size([3, 6])
```

### `dstack` — Depth Stack

Stacks along dimension 2 (depth):

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

dstacked = torch.dstack([a, b])
print(f"dstack shape: {dstacked.shape}")  # torch.Size([2, 2, 2])
print(f"dstack result:\n{dstacked}")
# tensor([[[1, 5],
#          [2, 6]],
#         [[3, 7],
#          [4, 8]]])
```

### Comparison of vstack, hstack, dstack

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

print(f"vstack: {torch.vstack([a, b]).shape}")  # (4, 3) — like cat(dim=0)
print(f"hstack: {torch.hstack([a, b]).shape}")  # (2, 6) — like cat(dim=1)
print(f"dstack: {torch.dstack([a, b]).shape}")  # (2, 3, 2) — adds dim 2
```

## Practical Applications

### Building Batches from Samples

The most common use of `stack`:

```python
# Individual samples (e.g., images)
samples = [torch.randn(3, 224, 224) for _ in range(32)]

# Stack into batch: (batch, channels, height, width)
batch = torch.stack(samples, dim=0)
print(f"Batch shape: {batch.shape}")  # torch.Size([32, 3, 224, 224])
```

### Feature Concatenation (Multi-Modal Fusion)

```python
# Features from different modalities
text_features = torch.randn(32, 256)    # From text encoder
image_features = torch.randn(32, 512)   # From image encoder
audio_features = torch.randn(32, 128)   # From audio encoder

# Early fusion: concatenate features
combined = torch.cat([text_features, image_features, audio_features], dim=1)
print(f"Combined features: {combined.shape}")  # torch.Size([32, 896])
```

### Multi-Head Attention Assembly

```python
# Simulate multi-head attention outputs
num_heads = 8
batch_size = 32
seq_len = 50
head_dim = 64

# Each head produces independent output
head_outputs = [torch.randn(batch_size, seq_len, head_dim) for _ in range(num_heads)]

# Concatenate heads along feature dimension
concatenated = torch.cat(head_outputs, dim=-1)
print(f"Concatenated shape: {concatenated.shape}")  # torch.Size([32, 50, 512])
```

### Sequence Concatenation

```python
# Concatenate sequences of different lengths
prefix = torch.randn(5, 256)    # 5 tokens
content = torch.randn(20, 256)  # 20 tokens
suffix = torch.randn(3, 256)    # 3 tokens

full_sequence = torch.cat([prefix, content, suffix], dim=0)
print(f"Full sequence: {full_sequence.shape}")  # torch.Size([28, 256])
```

### Time Series Stacking

```python
# Daily measurements: (hours, features)
monday = torch.randn(24, 5)
tuesday = torch.randn(24, 5)
wednesday = torch.randn(24, 5)
# ... etc

daily_data = [monday, tuesday, wednesday]

# Stack into (days, hours, features)
week_data = torch.stack(daily_data, dim=0)
print(f"Week data: {week_data.shape}")  # torch.Size([3, 24, 5])
```

### Image Tiling and Reconstruction

```python
def tile_image(image, tile_h, tile_w):
    """Split image into non-overlapping tiles."""
    c, h, w = image.shape
    tiles = []
    for i in range(0, h, tile_h):
        for j in range(0, w, tile_w):
            tile = image[:, i:i+tile_h, j:j+tile_w]
            tiles.append(tile)
    return torch.stack(tiles, dim=0)

def untile_image(tiles, grid_h, grid_w):
    """Reconstruct image from tiles."""
    rows = []
    for i in range(grid_h):
        row_tiles = tiles[i*grid_w:(i+1)*grid_w]
        row = torch.cat(list(row_tiles), dim=2)  # Horizontal concat
        rows.append(row)
    return torch.cat(rows, dim=1)  # Vertical concat

# Example
image = torch.randn(3, 256, 256)
tiles = tile_image(image, 64, 64)
print(f"Tiles: {tiles.shape}")  # torch.Size([16, 3, 64, 64])

reconstructed = untile_image(tiles, 4, 4)
print(f"Reconstructed: {reconstructed.shape}")  # torch.Size([3, 256, 256])
```

### Coordinate Grid Creation

```python
x = torch.linspace(-1, 1, 5)
y = torch.linspace(-1, 1, 5)

# Create meshgrid
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

# Stack into coordinate pairs (H, W, 2)
coords = torch.stack([grid_x, grid_y], dim=-1)
print(f"Coords shape: {coords.shape}")  # torch.Size([5, 5, 2])

# Each position contains [x, y] coordinates
print(f"Center coord: {coords[2, 2]}")  # tensor([0., 0.])
```

### Multi-GPU Output Gathering

```python
# Simulated outputs from multiple GPUs
gpu_outputs = [torch.randn(8, 10) for _ in range(4)]  # 4 GPUs, batch 8 each

# Gather all outputs
all_outputs = torch.cat(gpu_outputs, dim=0)
print(f"Gathered: {all_outputs.shape}")  # torch.Size([32, 10])
```

### Ensemble Predictions

```python
# Predictions from multiple models
model_predictions = [torch.randn(32, 10) for _ in range(5)]  # 5 models

# Stack for ensemble: (num_models, batch, classes)
ensemble = torch.stack(model_predictions, dim=0)
print(f"Ensemble shape: {ensemble.shape}")  # torch.Size([5, 32, 10])

# Average predictions
avg_prediction = ensemble.mean(dim=0)
print(f"Average prediction: {avg_prediction.shape}")  # torch.Size([32, 10])
```

## Performance Considerations

### Avoid Iterative Concatenation

```python
# BAD: O(n²) memory allocations
def slow_batch(tensors):
    result = tensors[0].unsqueeze(0)
    for t in tensors[1:]:
        result = torch.cat([result, t.unsqueeze(0)], dim=0)
    return result

# GOOD: O(n) single allocation
def fast_batch(tensors):
    return torch.stack(tensors, dim=0)

# Benchmark
import time
tensors = [torch.randn(100, 100) for _ in range(100)]

start = time.time()
slow_batch(tensors)
print(f"Slow: {time.time() - start:.4f}s")

start = time.time()
fast_batch(tensors)
print(f"Fast: {time.time() - start:.4f}s")
```

### Pre-allocate When Output Size is Known

```python
# When processing sequentially with known output size
num_items = 100
feature_dim = 256

# Pre-allocate output tensor
output = torch.empty(num_items, feature_dim)

for i, data in enumerate(data_loader):
    output[i] = process(data)  # In-place assignment (no allocation)
```

### Memory: cat/stack Always Copy

```python
# cat and stack create new tensors (copy data)
a = torch.randn(1000, 1000)
b = torch.randn(1000, 1000)

combined = torch.cat([a, b], dim=0)

# Verify: different storage
print(combined.storage().data_ptr() != a.storage().data_ptr())  # True
```

## Common Pitfalls

### Pitfall 1: Confusing cat and stack

```python
samples = [torch.randn(3, 32, 32) for _ in range(16)]

# WRONG: cat doesn't create batch dimension
# wrong = torch.cat(samples, dim=0)  # Shape: (48, 32, 32) — channels merged!

# CORRECT: stack creates batch dimension
correct = torch.stack(samples, dim=0)  # Shape: (16, 3, 32, 32)
```

### Pitfall 2: Dimension Mismatch in cat

```python
a = torch.randn(2, 3)
b = torch.randn(2, 4)  # Different size in dim 1

# Can only cat along the different dimension
# torch.cat([a, b], dim=0)  # Error: dim 1 doesn't match

torch.cat([a, b], dim=1)  # Works: (2, 7)
```

### Pitfall 3: Forgetting stack Requires Identical Shapes

```python
sequences = [
    torch.randn(10, 64),  # Length 10
    torch.randn(15, 64),  # Length 15 — different!
]

# torch.stack(sequences, dim=0)  # Error!

# Solution 1: Pad to same length first
max_len = max(s.size(0) for s in sequences)
padded = [torch.nn.functional.pad(s, (0, 0, 0, max_len - s.size(0))) 
          for s in sequences]
stacked = torch.stack(padded, dim=0)

# Solution 2: Use cat if batch dimension isn't needed
concatenated = torch.cat(sequences, dim=0)  # Shape: (25, 64)
```

## Quick Reference

| Function | Description | Creates New Dim? | Shape Requirement |
|----------|-------------|------------------|-------------------|
| `cat(tensors, dim)` | Join along existing dim | No | Match all except dim |
| `stack(tensors, dim)` | Join along new dim | Yes | ALL must match |
| `vstack(tensors)` | Vertical (dim 0) | Sometimes | Columns must match |
| `hstack(tensors)` | Horizontal (dim 1) | No | Rows must match |
| `dstack(tensors)` | Depth (dim 2) | Yes | Must match |

## Key Takeaways

1. **`cat` concatenates along existing dimension** — extends that dimension
2. **`stack` creates a new dimension** — groups tensors as "layers"
3. **`cat` allows different sizes** in the concatenation dimension only
4. **`stack` requires identical shapes** for all input tensors
5. **Use `stack` for batching** individual samples
6. **Use `cat` for combining** sequences or features
7. **Avoid iterative `cat`** — collect tensors and call once
8. **`stack` = unsqueeze + cat** under the hood

## See Also

- [Splitting Operations](splitting.md) - Inverse operations
- [Reshaping and View Operations](reshaping_view.md) - Shape manipulation
- [Indexing and Slicing](indexing_slicing.md) - Extracting parts
