# DataLoader Basics

## Overview

The `DataLoader` wraps a `Dataset` and provides an iterable over batches, handling batching, shuffling, parallel loading, and memory management. It is the bridge between data storage and the training loop.

## Core Interface

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,          # Samples per batch
    shuffle=True,           # Randomize order each epoch
    num_workers=4,          # Parallel data loading processes
    pin_memory=True,        # Pin memory for faster GPU transfer
    drop_last=False         # Drop incomplete final batch
)

for batch_x, batch_y in loader:
    # batch_x: (32, ...), batch_y: (32, ...)
    pass
```

## Iteration Mechanics

Each iteration over a `DataLoader` constitutes one epoch. The loader internally:

1. Creates an index sequence (shuffled if `shuffle=True`) via a `Sampler`.
2. Groups indices into batches of size `batch_size` via a `BatchSampler`.
3. Calls `dataset[idx]` for each index in a batch (possibly in parallel workers).
4. Collates individual samples into batched tensors via a `collate_fn`.

```python
# Manual equivalent of what DataLoader does:
indices = torch.randperm(len(dataset)) if shuffle else torch.arange(len(dataset))
for i in range(0, len(indices), batch_size):
    batch_indices = indices[i : i + batch_size]
    samples = [dataset[idx] for idx in batch_indices]
    batch = default_collate(samples)
    yield batch
```

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `batch_size` | 1 | Number of samples per batch |
| `shuffle` | False | Randomize sample order each epoch |
| `num_workers` | 0 | Number of subprocess workers for loading |
| `pin_memory` | False | Copy tensors to CUDA pinned memory |
| `drop_last` | False | Drop last batch if smaller than `batch_size` |
| `collate_fn` | `default_collate` | Function to merge samples into batch |
| `sampler` | None | Strategy for drawing samples |
| `prefetch_factor` | None | Batches prefetched per worker |
| `persistent_workers` | False | Keep workers alive between epochs |

## Basic Usage Patterns

```python
# Training: shuffle, drop incomplete batch, pin memory
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)

# Validation: no shuffle, keep all samples, larger batch for speed
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                        num_workers=4, pin_memory=True)

# Test: same as validation
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                         num_workers=4, pin_memory=True)
```

## Accessing Batch Contents

```python
# Single batch inspection
batch = next(iter(train_loader))
images, labels = batch
print(f"Batch shape: {images.shape}")  # e.g., torch.Size([64, 3, 224, 224])
print(f"Labels shape: {labels.shape}") # e.g., torch.Size([64])

# Full epoch
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # ... training step
```

## Key Takeaways

- `DataLoader` automates batching, shuffling, parallel loading, and memory pinning.
- Use `shuffle=True` and `drop_last=True` for training; `shuffle=False` for evaluation.
- `num_workers > 0` enables parallel data loading, which is essential for I/O-bound datasets.
- The DataLoader is iterable—each full iteration is one epoch over the dataset.
