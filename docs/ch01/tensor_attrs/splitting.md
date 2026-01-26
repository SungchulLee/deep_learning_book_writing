# Splitting Operations

Splitting operations divide tensors into smaller pieces—the inverse of concatenation and stacking. These operations are essential for data partitioning, unpacking batches, implementing attention mechanisms, and creating train/validation/test splits.

## Overview of Splitting Operations

| Operation | Description | Output Type | Inverse Of |
|-----------|-------------|-------------|------------|
| `split` | Split into specified sizes | Tuple of tensors | `cat` |
| `chunk` | Split into N equal parts | Tuple of tensors | `cat` |
| `unbind` | Remove dimension, return slices | Tuple of tensors | `stack` |
| `tensor_split` | Flexible splitting | Tuple of tensors | `cat` |

## `torch.split()` — Split by Size

Split a tensor into chunks of specified sizes along a dimension.

### Split into Equal-Sized Chunks

```python
import torch

t = torch.arange(10)
print(f"Original: {t}")  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Split into chunks of size 2
chunks = torch.split(t, split_size_or_sections=2)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")
# Chunk 0: tensor([0, 1])
# Chunk 1: tensor([2, 3])
# Chunk 2: tensor([4, 5])
# Chunk 3: tensor([6, 7])
# Chunk 4: tensor([8, 9])
```

### Split into Specified Sizes

```python
t = torch.arange(10)

# Split into specific sizes: [3, 3, 4]
chunks = torch.split(t, split_size_or_sections=[3, 3, 4])
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")
# Chunk 0: tensor([0, 1, 2])
# Chunk 1: tensor([3, 4, 5])
# Chunk 2: tensor([6, 7, 8, 9])

# Sizes must sum to tensor size
# torch.split(t, [3, 3, 3])  # Error: sizes sum to 9, tensor has 10
```

### Split Along Different Dimensions

```python
t = torch.arange(24).reshape(4, 6)
print(f"Original shape: {t.shape}")  # (4, 6)

# Split along dim 0 (rows)
row_chunks = torch.split(t, 2, dim=0)
print(f"Row chunks: {len(row_chunks)} chunks")
for chunk in row_chunks:
    print(f"  Shape: {chunk.shape}")  # (2, 6) each

# Split along dim 1 (columns)
col_chunks = torch.split(t, [2, 2, 2], dim=1)
print(f"Column chunks: {len(col_chunks)} chunks")
for chunk in col_chunks:
    print(f"  Shape: {chunk.shape}")  # (4, 2) each
```

### Handling Uneven Splits

When the tensor size isn't evenly divisible, the last chunk is smaller:

```python
t = torch.arange(10)

# Split into chunks of size 3 (10 not divisible by 3)
chunks = torch.split(t, 3)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}, size: {len(chunk)}")
# Chunk 0: tensor([0, 1, 2]), size: 3
# Chunk 1: tensor([3, 4, 5]), size: 3
# Chunk 2: tensor([6, 7, 8]), size: 3
# Chunk 3: tensor([9]), size: 1  ← smaller last chunk
```

## `torch.chunk()` — Split into N Equal Parts

Split a tensor into a specified **number** of chunks (vs specified sizes).

### Basic Chunking

```python
t = torch.arange(12)

# Split into 3 equal chunks
chunks = torch.chunk(t, chunks=3)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")
# Chunk 0: tensor([0, 1, 2, 3])
# Chunk 1: tensor([4, 5, 6, 7])
# Chunk 2: tensor([8, 9, 10, 11])
```

### Chunking with Uneven Division

```python
t = torch.arange(10)

# Split into 3 chunks (10 / 3 = 3.33...)
chunks = torch.chunk(t, chunks=3)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}, size: {len(chunk)}")
# Chunk 0: tensor([0, 1, 2, 3]), size: 4
# Chunk 1: tensor([4, 5, 6, 7]), size: 4
# Chunk 2: tensor([8, 9]), size: 2  ← smaller
```

**Note**: `chunk` tries to make chunks as equal as possible. The last chunk may be smaller if not evenly divisible.

### Multi-Dimensional Chunking

```python
t = torch.arange(24).reshape(4, 6)

# Chunk along dim 0
row_chunks = torch.chunk(t, 2, dim=0)
print(f"Row chunks: {[c.shape for c in row_chunks]}")
# [(2, 6), (2, 6)]

# Chunk along dim 1
col_chunks = torch.chunk(t, 3, dim=1)
print(f"Col chunks: {[c.shape for c in col_chunks]}")
# [(4, 2), (4, 2), (4, 2)]
```

## `torch.unbind()` — Remove Dimension and Unpack

`unbind` removes a dimension and returns a tuple of tensors—the inverse of `stack`.

### Basic Unbinding

```python
t = torch.arange(12).reshape(3, 4)
print(f"Original:\n{t}")
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# Unbind along dim 0 (returns rows)
rows = torch.unbind(t, dim=0)
print(f"Number of rows: {len(rows)}")  # 3
for i, row in enumerate(rows):
    print(f"Row {i}: {row}")
# Row 0: tensor([0, 1, 2, 3])
# Row 1: tensor([4, 5, 6, 7])
# Row 2: tensor([8, 9, 10, 11])
```

### Unbind Along Different Dimensions

```python
t = torch.arange(12).reshape(3, 4)

# Unbind along dim 1 (returns columns)
cols = torch.unbind(t, dim=1)
print(f"Number of columns: {len(cols)}")  # 4
for i, col in enumerate(cols):
    print(f"Column {i}: {col}")
# Column 0: tensor([0, 4, 8])
# Column 1: tensor([1, 5, 9])
# Column 2: tensor([2, 6, 10])
# Column 3: tensor([3, 7, 11])
```

### unbind vs split/chunk

```python
t = torch.arange(12).reshape(3, 4)

# unbind: removes the dimension (3D → 2D would become 2D tensors)
unbound = torch.unbind(t, dim=0)
print(f"unbind shapes: {[u.shape for u in unbound]}")
# [torch.Size([4]), torch.Size([4]), torch.Size([4])]

# split with size 1: keeps the dimension
split_1 = torch.split(t, 1, dim=0)
print(f"split shapes: {[s.shape for s in split_1]}")
# [torch.Size([1, 4]), torch.Size([1, 4]), torch.Size([1, 4])]
```

**Key difference**: `unbind` reduces dimensionality (like indexing), while `split`/`chunk` preserve it (like slicing).

## `torch.tensor_split()` — Flexible Splitting

More flexible version of `split` that can split at specific indices.

### Split at Indices

```python
t = torch.arange(10)

# Split at indices 3 and 7
chunks = torch.tensor_split(t, [3, 7])
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")
# Chunk 0: tensor([0, 1, 2])      — indices 0:3
# Chunk 1: tensor([3, 4, 5, 6])   — indices 3:7
# Chunk 2: tensor([7, 8, 9])      — indices 7:end
```

### Split into N Parts

```python
t = torch.arange(10)

# Split into 3 parts (similar to chunk but more flexible)
chunks = torch.tensor_split(t, 3)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")
# Chunk 0: tensor([0, 1, 2, 3])
# Chunk 1: tensor([4, 5, 6])
# Chunk 2: tensor([7, 8, 9])
```

## Practical Applications

### Train/Validation/Test Split

```python
# Dataset tensor
data = torch.randn(1000, 10)
labels = torch.randint(0, 5, (1000,))

# Split: 70% train, 15% val, 15% test
train_data, val_data, test_data = torch.split(data, [700, 150, 150], dim=0)
train_labels, val_labels, test_labels = torch.split(labels, [700, 150, 150], dim=0)

print(f"Train: {train_data.shape}")  # (700, 10)
print(f"Val: {val_data.shape}")      # (150, 10)
print(f"Test: {test_data.shape}")    # (150, 10)
```

### Unpacking Batch Dimension

```python
# Batch of predictions
batch_predictions = torch.randn(4, 10)  # 4 samples, 10 classes

# Unpack into individual predictions
individual = torch.unbind(batch_predictions, dim=0)
for i, pred in enumerate(individual):
    print(f"Sample {i} prediction shape: {pred.shape}")  # (10,)
```

### Multi-Head Attention Split

```python
# Combined QKV projection
batch, seq_len, d_model = 32, 50, 512
num_heads = 8
head_dim = d_model // num_heads

# Projected QKV: (batch, seq_len, 3 * d_model)
qkv = torch.randn(batch, seq_len, 3 * d_model)

# Split into Q, K, V
q, k, v = torch.split(qkv, d_model, dim=-1)
print(f"Q shape: {q.shape}")  # (32, 50, 512)
print(f"K shape: {k.shape}")  # (32, 50, 512)
print(f"V shape: {v.shape}")  # (32, 50, 512)

# Alternative: chunk into 3 equal parts
q, k, v = torch.chunk(qkv, 3, dim=-1)
```

### Splitting Heads in Multi-Head Attention

```python
batch, seq_len, d_model = 32, 50, 512
num_heads = 8
head_dim = d_model // num_heads

# Query tensor
q = torch.randn(batch, seq_len, d_model)

# Reshape to separate heads: (batch, seq_len, num_heads, head_dim)
q = q.reshape(batch, seq_len, num_heads, head_dim)

# Transpose for attention: (batch, num_heads, seq_len, head_dim)
q = q.permute(0, 2, 1, 3)

# Split into individual heads (for debugging/analysis)
heads = torch.unbind(q, dim=1)
print(f"Number of heads: {len(heads)}")  # 8
print(f"Each head shape: {heads[0].shape}")  # (32, 50, 64)
```

### Gradient Checkpointing Segments

```python
# Split model layers for gradient checkpointing
class LargeModel(torch.nn.Module):
    def __init__(self, num_layers=24):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(512, 512) for _ in range(num_layers)
        ])
    
    def forward_segment(self, x, segment_layers):
        for layer in segment_layers:
            x = torch.relu(layer(x))
        return x
    
    def forward(self, x):
        # Split layers into segments for checkpointing
        num_segments = 4
        layers_per_segment = len(self.layers) // num_segments
        
        for i in range(num_segments):
            start = i * layers_per_segment
            end = start + layers_per_segment
            segment = self.layers[start:end]
            x = torch.utils.checkpoint.checkpoint(
                self.forward_segment, x, segment
            )
        return x
```

### Variable-Length Sequence Processing

```python
# Packed sequences with recorded lengths
combined_sequences = torch.randn(45, 64)  # Total of 45 timesteps
lengths = [10, 15, 8, 12]  # Individual sequence lengths

# Split back into individual sequences
sequences = torch.split(combined_sequences, lengths, dim=0)
for i, seq in enumerate(sequences):
    print(f"Sequence {i}: shape {seq.shape}")
# Sequence 0: shape torch.Size([10, 64])
# Sequence 1: shape torch.Size([15, 64])
# Sequence 2: shape torch.Size([8, 64])
# Sequence 3: shape torch.Size([12, 64])
```

### Image Channel Separation

```python
# RGB image batch
images = torch.randn(32, 3, 224, 224)  # (batch, channels, H, W)

# Split into individual channels
r, g, b = torch.unbind(images, dim=1)
print(f"Red channel: {r.shape}")    # (32, 224, 224)
print(f"Green channel: {g.shape}")  # (32, 224, 224)
print(f"Blue channel: {b.shape}")   # (32, 224, 224)

# Alternative using chunk
r, g, b = torch.chunk(images, 3, dim=1)
print(f"Red channel: {r.shape}")    # (32, 1, 224, 224) — keeps dim!
```

### Sliding Window Creation

```python
def create_windows_with_split(sequence, window_size, stride=1):
    """Create overlapping windows using unfold (more efficient than split)."""
    # unfold(dim, size, step) creates sliding windows
    windows = sequence.unfold(0, window_size, stride)
    return windows

# Example
time_series = torch.arange(20).float()
windows = create_windows_with_split(time_series, window_size=5, stride=2)
print(f"Windows shape: {windows.shape}")  # (8, 5)
print(f"First window: {windows[0]}")  # tensor([0., 1., 2., 3., 4.])
print(f"Second window: {windows[1]}")  # tensor([2., 3., 4., 5., 6.])
```

### K-Fold Cross Validation

```python
def k_fold_split(data, k=5):
    """Split data into k folds for cross-validation."""
    fold_size = len(data) // k
    folds = torch.chunk(data, k, dim=0)
    return folds

def get_fold(folds, fold_idx):
    """Get train and validation sets for a specific fold."""
    val_fold = folds[fold_idx]
    train_folds = [f for i, f in enumerate(folds) if i != fold_idx]
    train_data = torch.cat(train_folds, dim=0)
    return train_data, val_fold

# Example
data = torch.randn(100, 10)
folds = k_fold_split(data, k=5)

for i in range(5):
    train, val = get_fold(folds, i)
    print(f"Fold {i}: train {train.shape}, val {val.shape}")
```

## Views vs Copies

**Important**: `split`, `chunk`, and `unbind` return **views** when possible:

```python
t = torch.arange(12).reshape(3, 4)

# split returns views
chunks = torch.split(t, 1, dim=0)
print(chunks[0].storage().data_ptr() == t.storage().data_ptr())  # True

# Modifying a chunk affects the original!
chunks[0][0, 0] = 999
print(t[0, 0])  # tensor(999)
```

To get independent copies:

```python
# Clone each chunk
chunks = [chunk.clone() for chunk in torch.split(t, 1, dim=0)]
```

## split vs chunk Comparison

| Aspect | `split` | `chunk` |
|--------|---------|---------|
| Argument | Size(s) of chunks | Number of chunks |
| Usage | `split(t, 3)` or `split(t, [2,3,5])` | `chunk(t, 3)` |
| Uneven handling | Last chunk smaller | Tries to balance |
| Flexibility | Can specify exact sizes | Equal parts only |

```python
t = torch.arange(10)

# split with size: chunks of size 3
split_result = torch.split(t, 3)
print([len(c) for c in split_result])  # [3, 3, 3, 1]

# chunk with count: 3 chunks
chunk_result = torch.chunk(t, 3)
print([len(c) for c in chunk_result])  # [4, 4, 2]
```

## Performance Tips

### Avoid Repeated Splitting

```python
# BAD: Splitting inside a loop
for epoch in range(100):
    batches = torch.split(data, batch_size)  # Repeated work
    for batch in batches:
        process(batch)

# GOOD: Split once, reuse
batches = torch.split(data, batch_size)
for epoch in range(100):
    for batch in batches:
        process(batch)
```

### Use Views When Possible

```python
# Views are free (no memory allocation)
chunks = torch.split(t, chunk_size)  # Returns views

# Only clone if you need independent copies
independent_chunks = [c.clone() for c in chunks]  # Allocates memory
```

## Quick Reference

| Function | Description | Arguments |
|----------|-------------|-----------|
| `split(t, size, dim)` | Split into chunks of `size` | Size or list of sizes |
| `chunk(t, n, dim)` | Split into `n` chunks | Number of chunks |
| `unbind(t, dim)` | Remove dim, return slices | Dimension to remove |
| `tensor_split(t, indices, dim)` | Split at indices | Indices or count |

## Key Takeaways

1. **`split` specifies chunk sizes**, `chunk` specifies number of chunks
2. **`unbind` removes the dimension** (inverse of `stack`)
3. **`split`/`chunk` preserve the dimension** (inverse of `cat`)
4. **All return views** when possible—modifications affect original
5. **Last chunk may be smaller** when not evenly divisible
6. **Use `split` for exact control**, `chunk` for equal partitioning
7. **Clone chunks** if you need independent copies
8. **`tensor_split`** offers index-based splitting flexibility

## See Also

- [Concatenation and Stacking](concat_stack.md) - Inverse operations
- [Reshaping and View Operations](reshaping_view.md) - Shape manipulation
- [Indexing and Slicing](indexing_slicing.md) - Extracting parts
