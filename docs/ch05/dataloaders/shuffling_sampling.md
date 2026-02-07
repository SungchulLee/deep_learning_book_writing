# Shuffling and Sampling

## Overview

Shuffling randomizes the order in which samples are presented to the model each epoch, breaking correlations between consecutive batches. Samplers provide fine-grained control over this process.

## Why Shuffle?

Without shuffling, the model sees samples in the same order every epoch. If the dataset is sorted (e.g., by class), the model receives long runs of identical labels, causing biased gradient updates and poor convergence. Shuffling ensures each batch is a representative sample of the data distribution.

```python
# Shuffled training
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Deterministic evaluation (never shuffle)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
```

## Built-in Samplers

PyTorch provides several sampler classes in `torch.utils.data`:

**`SequentialSampler`**: Yields indices 0, 1, 2, ..., N-1. Used by default when `shuffle=False`.

**`RandomSampler`**: Yields a random permutation of indices. Used by default when `shuffle=True`.

```python
from torch.utils.data import RandomSampler, SequentialSampler

# Equivalent to shuffle=True
loader = DataLoader(dataset, batch_size=64,
                    sampler=RandomSampler(dataset))

# With replacement (bootstrap sampling)
loader = DataLoader(dataset, batch_size=64,
                    sampler=RandomSampler(dataset, replacement=True,
                                         num_samples=10000))
```

**`SubsetRandomSampler`**: Samples randomly from a specified subset of indices—useful for creating train/validation splits without copying data:

```python
from torch.utils.data import SubsetRandomSampler

indices = list(range(len(dataset)))
split = int(0.8 * len(dataset))
train_indices, val_indices = indices[:split], indices[split:]

train_loader = DataLoader(dataset, batch_size=64,
                          sampler=SubsetRandomSampler(train_indices))
val_loader = DataLoader(dataset, batch_size=256,
                        sampler=SubsetRandomSampler(val_indices))
```

Note: When using a custom `sampler`, set `shuffle=False` (or omit it) to avoid a conflict.

## BatchSampler

`BatchSampler` wraps any sampler and yields batches of indices:

```python
from torch.utils.data import BatchSampler, SequentialSampler

batch_sampler = BatchSampler(
    SequentialSampler(dataset),
    batch_size=32,
    drop_last=False
)

loader = DataLoader(dataset, batch_sampler=batch_sampler)
```

This is useful when you need custom batch construction logic (e.g., grouping by length or ensuring each batch contains samples from multiple classes).

## Temporal Sampling for Financial Data

Standard shuffling violates the temporal ordering required for financial time series. Instead, use sequential sampling for evaluation and block-aware sampling for training:

```python
class BlockShuffleSampler(Sampler):
    """Shuffle blocks of contiguous indices to preserve local temporal order."""
    def __init__(self, data_source, block_size=20):
        self.n = len(data_source)
        self.block_size = block_size

    def __iter__(self):
        blocks = [list(range(i, min(i + self.block_size, self.n)))
                  for i in range(0, self.n, self.block_size)]
        random.shuffle(blocks)
        return iter([idx for block in blocks for idx in block])

    def __len__(self):
        return self.n
```

## Key Takeaways

- Always shuffle training data; never shuffle evaluation data.
- Samplers provide fine-grained control over the sample ordering strategy.
- `SubsetRandomSampler` enables train/val splits without data duplication.
- Financial time series require temporal-aware sampling that preserves local ordering.
