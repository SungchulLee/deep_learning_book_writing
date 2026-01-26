# Batching and Shuffling

## Learning Objectives

By the end of this section, you will be able to:

- Master the `batch_size` parameter and its implications
- Understand when and why to shuffle data
- Implement reproducible shuffling with generators
- Use `drop_last` appropriately for training vs validation

## The Role of Batching

Batching is fundamental to efficient neural network training. Rather than processing samples one at a time or computing gradients over the entire dataset, we process small groups of samples together.

### Why Batch Processing?

The benefits of batching are threefold:

**1. Computational Efficiency**

Modern GPUs are optimized for parallel matrix operations. Processing a batch of 32 samples is nearly as fast as processing 1 sample:

```python
import torch
import time

x_single = torch.randn(1, 1000)
x_batch = torch.randn(32, 1000)
weights = torch.randn(1000, 100)

# Single sample: minimal GPU utilization
start = time.time()
for _ in range(1000):
    _ = x_single @ weights
single_time = time.time() - start

# Batch: excellent GPU utilization
start = time.time()
for _ in range(1000):
    _ = x_batch @ weights
batch_time = time.time() - start

# Batch processing is ~30x faster per sample
print(f"Single: {single_time:.3f}s, Batch: {batch_time:.3f}s")
print(f"Speedup: {single_time / batch_time:.1f}x")
```

**2. Gradient Stability**

Averaging gradients over multiple samples reduces variance:

$$
\text{Var}\left[\frac{1}{B}\sum_{i=1}^{B} g_i\right] = \frac{\text{Var}[g_i]}{B}
$$

**3. Memory Predictability**

Fixed batch sizes enable deterministic memory allocation, crucial for GPU training.

## Batch Size Configuration

### Basic Usage

```python
from torch.utils.data import DataLoader

# Create DataLoader with specified batch size
dataloader = DataLoader(
    dataset,
    batch_size=32  # Process 32 samples at a time
)
```

### Batch Size Trade-offs

| Batch Size | Pros | Cons |
|------------|------|------|
| **Small (8-16)** | Lower memory, more updates per epoch, gradient noise may help generalization | Slower training, less stable gradients |
| **Medium (32-64)** | Good balance of speed and stability | — |
| **Large (128-512)** | Faster training, more stable gradients | High memory usage, may need LR scaling |
| **Very Large (1024+)** | Maximum throughput | Requires LR warmup, may generalize worse |

### Dynamic Batch Size Calculation

When memory is constrained, you can calculate the maximum batch size:

```python
def find_max_batch_size(model, input_shape, device='cuda', start=2, max_attempts=10):
    """
    Binary search for maximum batch size that fits in GPU memory.
    """
    model = model.to(device)
    batch_size = start
    
    for _ in range(max_attempts):
        try:
            x = torch.randn(batch_size, *input_shape, device=device)
            _ = model(x)
            torch.cuda.empty_cache()
            batch_size *= 2
        except RuntimeError:  # Out of memory
            torch.cuda.empty_cache()
            return batch_size // 2
    
    return batch_size
```

## The Shuffle Parameter

Shuffling randomizes the order in which samples are presented to the model each epoch.

### Why Shuffle?

**Without Shuffling** (shuffle=False):

- Samples always appear in the same order
- Model may learn spurious patterns from data ordering
- Batches always contain the same samples

**With Shuffling** (shuffle=True):

- Order changes each epoch
- Different samples grouped into batches
- Prevents overfitting to data order

### Shuffling Demonstration

```python
from torch.utils.data import DataLoader, TensorDataset
import torch

# Create simple dataset with identifiable samples
data = torch.arange(10)
dataset = TensorDataset(data)

# Without shuffle - same order every epoch
loader_no_shuffle = DataLoader(dataset, batch_size=3, shuffle=False)

print("Without shuffle:")
for epoch in range(2):
    batches = [batch[0].tolist() for batch in loader_no_shuffle]
    print(f"  Epoch {epoch + 1}: {batches}")
```

**Output:**
```
Without shuffle:
  Epoch 1: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
  Epoch 2: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
```

```python
# With shuffle - different order each epoch
loader_shuffle = DataLoader(dataset, batch_size=3, shuffle=True)

print("With shuffle:")
for epoch in range(2):
    batches = [batch[0].tolist() for batch in loader_shuffle]
    print(f"  Epoch {epoch + 1}: {batches}")
```

**Output:**
```
With shuffle:
  Epoch 1: [[7, 2, 5], [0, 9, 1], [4, 8, 3], [6]]
  Epoch 2: [[3, 1, 8], [6, 0, 4], [9, 2, 7], [5]]
```

### When to Shuffle

| Scenario | Shuffle? | Reason |
|----------|----------|--------|
| **Training** | ✅ Yes | Prevent learning from data order |
| **Validation** | ❌ No | Reproducible evaluation metrics |
| **Test** | ❌ No | Reproducible final results |
| **Time Series** | ❌ No | Temporal order matters |
| **Debugging** | ❌ No | Reproducible behavior |

## Reproducible Shuffling with Generators

For reproducible experiments, control the shuffling randomness with a Generator:

### Basic Generator Usage

```python
# Create a generator with fixed seed
gen = torch.Generator().manual_seed(42)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    generator=gen  # Controls shuffle randomness
)
```

### Full Reproducibility Pattern

```python
def create_reproducible_dataloader(dataset, batch_size, seed):
    """
    Create a DataLoader with fully reproducible shuffling.
    """
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=gen
    )

# Same seed = identical shuffle order
loader1 = create_reproducible_dataloader(dataset, batch_size=4, seed=42)
loader2 = create_reproducible_dataloader(dataset, batch_size=4, seed=42)

batches1 = [batch[0].tolist() for batch in loader1]
batches2 = [batch[0].tolist() for batch in loader2]

print(f"Identical? {batches1 == batches2}")  # True
```

!!! warning "Generator Persistence"
    The generator maintains state across epochs. If you need the same shuffle order for multiple runs, create a new generator for each run or re-seed before each epoch.

### Per-Epoch Seeding

For debugging specific epochs:

```python
def train_with_epoch_seeds(dataset, num_epochs, base_seed):
    for epoch in range(num_epochs):
        # New generator per epoch with deterministic seed
        epoch_seed = base_seed + epoch
        gen = torch.Generator().manual_seed(epoch_seed)
        
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            generator=gen
        )
        
        # Training loop...
        for batch in loader:
            pass
```

## The drop_last Parameter

The `drop_last` parameter controls how to handle the final batch when the dataset size doesn't divide evenly by batch_size.

### Behavior Comparison

```python
dataset = TensorDataset(torch.arange(10))  # 10 samples

# drop_last=False (default): keep all samples
loader_keep = DataLoader(dataset, batch_size=3, drop_last=False)
batches_keep = [batch[0].tolist() for batch in loader_keep]
print(f"drop_last=False: {batches_keep}")
# Output: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

# drop_last=True: discard incomplete final batch
loader_drop = DataLoader(dataset, batch_size=3, drop_last=True)
batches_drop = [batch[0].tolist() for batch in loader_drop]
print(f"drop_last=True:  {batches_drop}")
# Output: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
```

### When to Use drop_last

| Scenario | drop_last | Reason |
|----------|-----------|--------|
| **Training** | True | Consistent batch sizes, required for some normalization |
| **Validation/Test** | False | Use all data for accurate metrics |
| **BatchNorm layers** | True | BatchNorm requires batch_size > 1 |
| **Small datasets** | False | Don't waste precious samples |

### Mathematical Impact

With dataset size $N$ and batch size $B$:

**drop_last=False:**
$$
\text{Samples per epoch} = N
$$
$$
\text{Batches per epoch} = \lceil N/B \rceil
$$

**drop_last=True:**
$$
\text{Samples per epoch} = B \cdot \lfloor N/B \rfloor
$$
$$
\text{Batches per epoch} = \lfloor N/B \rfloor
$$

```python
N, B = 100, 32

samples_keep = N                    # 100
samples_drop = B * (N // B)         # 96 (4 dropped)

batches_keep = -(-N // B)           # 4 (ceiling division)
batches_drop = N // B               # 3
```

## Recommended Configurations

### Training Configuration

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,              # Randomize each epoch
    drop_last=True,            # Consistent batch sizes
    generator=torch.Generator().manual_seed(42),
    num_workers=4,
    pin_memory=True
)
```

### Validation Configuration

```python
val_loader = DataLoader(
    val_dataset,
    batch_size=64,             # Can be larger (no backprop)
    shuffle=False,             # Reproducible evaluation
    drop_last=False,           # Use all validation data
    num_workers=4,
    pin_memory=True
)
```

### Test Configuration

```python
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,             # Reproducible final results
    drop_last=False
)
```

## Advanced: Dynamic Batching

For variable-length sequences or memory-constrained scenarios, you can implement dynamic batching:

```python
class DynamicBatchSampler:
    """
    Creates batches based on total tokens rather than sample count.
    Useful for NLP tasks with variable-length sequences.
    """
    
    def __init__(self, lengths, max_tokens, shuffle=True):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
    
    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)
        
        batch = []
        batch_length = 0
        
        for idx in indices:
            if batch_length + self.lengths[idx] > self.max_tokens and batch:
                yield batch
                batch = []
                batch_length = 0
            
            batch.append(idx)
            batch_length += self.lengths[idx]
        
        if batch:
            yield batch
```

## Summary

| Parameter | Training | Validation | Test |
|-----------|----------|------------|------|
| `batch_size` | 32-128 typical | Larger OK (no gradients) | Match validation |
| `shuffle` | True | False | False |
| `drop_last` | True (usually) | False | False |
| `generator` | Use for reproducibility | Not needed | Not needed |

## Practice Exercises

1. **Memory Analysis**: Calculate the GPU memory required for batch sizes of 16, 32, 64, and 128 for a model with 10M parameters processing 224×224 RGB images.

2. **Reproducibility Test**: Create two training runs with different seeds and verify that shuffling produces different results. Then use the same seed and verify identical shuffling.

3. **drop_last Impact**: For a dataset of 1000 samples with batch_size=64, calculate:
   - How many samples are dropped per epoch with drop_last=True?
   - What percentage of data is unused?
   - Over 100 epochs, how many gradient updates are lost?

## What's Next

The next section covers **DataLoader Parameters**, providing a complete guide to all configuration options including `num_workers`, `pin_memory`, `timeout`, and more.
