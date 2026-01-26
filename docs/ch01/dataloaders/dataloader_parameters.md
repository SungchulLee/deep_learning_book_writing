# DataLoader Parameters

## Learning Objectives

By the end of this section, you will be able to:

- Understand every DataLoader parameter and its purpose
- Configure optimal settings for training, validation, and debugging
- Balance performance trade-offs in different scenarios
- Implement production-ready DataLoader configurations

## Complete Parameter Reference

The DataLoader constructor accepts numerous parameters for fine-grained control:

```python
DataLoader(
    dataset,                    # Required: Dataset to load from
    batch_size=1,               # Samples per batch
    shuffle=False,              # Randomize order
    sampler=None,               # Custom sampling strategy
    batch_sampler=None,         # Custom batch sampling
    num_workers=0,              # Parallel loading processes
    collate_fn=None,            # Custom batch construction
    pin_memory=False,           # Pin memory for GPU transfer
    drop_last=False,            # Drop incomplete final batch
    timeout=0,                  # Worker timeout
    worker_init_fn=None,        # Worker initialization
    multiprocessing_context=None,
    generator=None,             # Random number generator
    prefetch_factor=None,       # Batches to prefetch per worker
    persistent_workers=False,   # Keep workers alive between epochs
    pin_memory_device=""        # Device for pinned memory
)
```

## Core Parameters Deep Dive

### batch_size

Controls how many samples are grouped into each batch.

```python
# Standard configuration
loader = DataLoader(dataset, batch_size=32)

# Memory-constrained configuration
loader = DataLoader(dataset, batch_size=8)

# Full-batch gradient descent (rarely used)
loader = DataLoader(dataset, batch_size=len(dataset))
```

**Guidelines:**

| Model Type | Recommended batch_size |
|------------|----------------------|
| Small MLPs | 64-256 |
| Standard CNNs (ResNet) | 32-128 |
| Large CNNs (EfficientNet) | 16-64 |
| Transformers (BERT-base) | 16-32 |
| Large Transformers (GPT) | 8-16 |
| Vision Transformers | 32-128 |

### shuffle

Randomizes sample order each epoch.

```python
# Training: always shuffle
train_loader = DataLoader(train_dataset, shuffle=True)

# Validation/Test: never shuffle
val_loader = DataLoader(val_dataset, shuffle=False)
```

!!! warning "Mutual Exclusivity"
    `shuffle=True` and `sampler` are mutually exclusive. If you provide a sampler, do not set shuffle=True—the sampler controls ordering.

### generator

Controls randomness for reproducible shuffling.

```python
# Reproducible shuffling
gen = torch.Generator().manual_seed(42)
loader = DataLoader(
    dataset,
    shuffle=True,
    generator=gen
)
```

**Best Practice:** Always use a generator for reproducible experiments:

```python
def get_reproducible_loaders(train_ds, val_ds, seed=42):
    """Create reproducible train and validation loaders."""
    train_gen = torch.Generator().manual_seed(seed)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        generator=train_gen
    )
    
    # Validation doesn't need generator (no shuffling)
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False
    )
    
    return train_loader, val_loader
```

### drop_last

Determines handling of the final incomplete batch.

```python
# Dataset with 100 samples, batch_size=32
# Batches: [32, 32, 32, 4]

# Keep all samples (default)
loader = DataLoader(dataset, batch_size=32, drop_last=False)
# 4 batches, last has 4 samples

# Drop incomplete batch
loader = DataLoader(dataset, batch_size=32, drop_last=True)
# 3 batches, each has 32 samples (4 samples dropped)
```

**When to use drop_last=True:**

- BatchNorm layers (require batch_size > 1)
- Synchronized distributed training
- When you need consistent tensor shapes

## Performance Parameters

### num_workers

Number of subprocesses for parallel data loading.

```python
# Single-threaded (simple, good for debugging)
loader = DataLoader(dataset, num_workers=0)

# Multi-process (faster, more complex)
loader = DataLoader(dataset, num_workers=4)
```

**Optimal num_workers selection:**

```python
import os

def get_optimal_workers():
    """
    Heuristic for optimal num_workers.
    """
    cpu_count = os.cpu_count()
    
    # Rule of thumb: 4 workers per GPU, up to CPU count
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    suggested = min(4 * gpu_count, cpu_count)
    
    return suggested
```

| Scenario | Recommended num_workers |
|----------|------------------------|
| Debugging | 0 |
| Single GPU | 4-8 |
| Multi-GPU | 4 per GPU |
| CPU-only | 2-4 |
| I/O-bound data | Increase until no benefit |
| CPU-bound data | Don't exceed CPU cores |

!!! note "Worker Overhead"
    Each worker spawns a separate process with its own memory copy of the dataset. Too many workers can exhaust system memory.

### pin_memory

Allocates data in pinned (page-locked) memory for faster GPU transfers.

```python
# Without pin_memory: Standard allocation
loader = DataLoader(dataset, pin_memory=False)

# With pin_memory: Faster GPU transfer
loader = DataLoader(dataset, pin_memory=True)
```

**How it works:**

```
Standard Memory:
CPU Memory → Copy → Pinned Memory → GPU Memory
                (extra step)

Pinned Memory:
CPU Pinned Memory → GPU Memory (direct DMA transfer)
```

**Guidelines:**

- ✅ Always use `pin_memory=True` when training on GPU
- ✅ Combine with `non_blocking=True` in `.to(device)` calls
- ❌ Don't use on CPU-only training
- ⚠️ Increases CPU memory usage

```python
# Complete GPU transfer pattern
loader = DataLoader(dataset, pin_memory=True, num_workers=4)

for batch in loader:
    # Non-blocking transfer overlaps with next batch loading
    inputs = batch[0].to(device, non_blocking=True)
    targets = batch[1].to(device, non_blocking=True)
```

### persistent_workers

Keeps worker processes alive between epochs.

```python
# Workers terminate after each epoch (default)
loader = DataLoader(dataset, num_workers=4, persistent_workers=False)

# Workers persist across epochs (faster)
loader = DataLoader(dataset, num_workers=4, persistent_workers=True)
```

**Benefits:**

- No worker spawn overhead per epoch
- Workers maintain state (open file handles, caches)
- Significantly faster for multi-epoch training

**Requirements:**

- Only works with `num_workers > 0`
- Uses more memory (workers stay alive)

### prefetch_factor

Number of batches each worker pre-loads ahead of time.

```python
# Default: 2 batches per worker
loader = DataLoader(
    dataset,
    num_workers=4,
    prefetch_factor=2  # 8 total batches prefetched (4 workers × 2)
)

# Higher prefetch for slow data loading
loader = DataLoader(
    dataset,
    num_workers=4,
    prefetch_factor=4  # 16 total batches prefetched
)
```

| prefetch_factor | Memory Usage | GPU Idle Time |
|-----------------|--------------|---------------|
| 1 | Low | May have gaps |
| 2 (default) | Moderate | Usually smooth |
| 4+ | High | Minimal gaps |

### timeout

Maximum time (seconds) to wait for a batch from workers.

```python
# No timeout (wait indefinitely)
loader = DataLoader(dataset, num_workers=4, timeout=0)

# 60-second timeout
loader = DataLoader(dataset, num_workers=4, timeout=60)
```

**Use cases:**

- Set non-zero timeout to detect hanging workers
- Debugging deadlocks in data loading
- Production systems requiring guaranteed response times

```python
# Debugging configuration with timeout
debug_loader = DataLoader(
    dataset,
    num_workers=4,
    timeout=30  # Fail fast if worker hangs
)
```

### worker_init_fn

Function called when each worker process starts.

```python
import random
import numpy as np

def worker_init_fn(worker_id):
    """
    Initialize each worker with unique random seed.
    
    Without this, all workers may generate identical random numbers
    in data augmentation.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)

loader = DataLoader(
    dataset,
    num_workers=4,
    worker_init_fn=worker_init_fn
)
```

**Common use cases:**

- Setting different random seeds per worker
- Opening database connections
- Loading shared resources
- Configuring logging

## Sampler Parameters

### sampler

Custom sampling strategy replacing shuffle.

```python
from torch.utils.data import SequentialSampler, RandomSampler

# Explicit sequential access
loader = DataLoader(dataset, sampler=SequentialSampler(dataset))

# Equivalent to shuffle=True
loader = DataLoader(dataset, sampler=RandomSampler(dataset))
```

### batch_sampler

Custom batch-level sampling (advanced).

```python
from torch.utils.data import BatchSampler, SequentialSampler

# Create custom batch sampler
batch_sampler = BatchSampler(
    SequentialSampler(dataset),
    batch_size=32,
    drop_last=False
)

loader = DataLoader(dataset, batch_sampler=batch_sampler)
```

!!! note "Mutual Exclusivity"
    When using `batch_sampler`, you cannot specify `batch_size`, `shuffle`, `sampler`, or `drop_last`.

### collate_fn

Custom function to combine samples into batches.

```python
def custom_collate(batch):
    """
    Custom collation for variable-length sequences.
    """
    sequences, labels = zip(*batch)
    
    # Pad sequences to max length
    padded = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True
    )
    
    labels = torch.stack(labels)
    
    return padded, labels

loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=custom_collate
)
```

## Configuration Recipes

### Training Configuration (GPU)

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
    generator=torch.Generator().manual_seed(42),
    worker_init_fn=worker_init_fn
)
```

### Validation Configuration (GPU)

```python
val_loader = DataLoader(
    val_dataset,
    batch_size=64,            # Larger OK (no gradients)
    shuffle=False,            # Reproducible
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False           # Use all data
)
```

### Debug Configuration

```python
debug_loader = DataLoader(
    train_dataset,
    batch_size=4,             # Small for quick iteration
    shuffle=False,            # Reproducible
    num_workers=0,            # Single process (easier debugging)
    pin_memory=False,
    drop_last=False
)
```

### High-Performance Configuration

```python
fast_loader = DataLoader(
    train_dataset,
    batch_size=128,           # Maximize GPU utilization
    shuffle=True,
    num_workers=8,            # Many workers
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,        # Aggressive prefetching
    drop_last=True,
    generator=torch.Generator().manual_seed(42)
)
```

### Memory-Constrained Configuration

```python
memory_efficient_loader = DataLoader(
    train_dataset,
    batch_size=8,             # Small batches
    shuffle=True,
    num_workers=2,            # Few workers
    pin_memory=True,
    persistent_workers=False, # Free memory between epochs
    prefetch_factor=1,        # Minimal prefetching
    drop_last=False           # Don't waste samples
)
```

## Parameter Interaction Matrix

| Parameter | Works With | Conflicts With |
|-----------|------------|----------------|
| `shuffle` | `generator`, `drop_last` | `sampler`, `batch_sampler` |
| `sampler` | `drop_last` | `shuffle`, `batch_sampler` |
| `batch_sampler` | — | `batch_size`, `shuffle`, `sampler`, `drop_last` |
| `num_workers` | All | — |
| `pin_memory` | All | — |
| `persistent_workers` | `num_workers > 0` | `num_workers = 0` |
| `prefetch_factor` | `num_workers > 0` | `num_workers = 0` |

## Summary

| Parameter | Default | Training | Validation | Debug |
|-----------|---------|----------|------------|-------|
| batch_size | 1 | 32-128 | 64-256 | 4-8 |
| shuffle | False | True | False | False |
| num_workers | 0 | 4+ | 4+ | 0 |
| pin_memory | False | True | True | False |
| drop_last | False | True | False | False |
| persistent_workers | False | True | True | False |
| prefetch_factor | 2 | 2-4 | 2 | — |

## Practice Exercises

1. **Configuration Comparison**: Create three DataLoaders with different configurations (debug, standard, high-performance) and measure their iteration speed.

2. **Worker Analysis**: Profile data loading time with num_workers = 0, 2, 4, 8. Plot the relationship between workers and throughput.

3. **Memory Investigation**: Use `torch.cuda.memory_allocated()` to measure GPU memory with different pin_memory settings.

## What's Next

The next section covers **Samplers**, showing how to implement custom sampling strategies for imbalanced datasets, stratified sampling, and curriculum learning.
