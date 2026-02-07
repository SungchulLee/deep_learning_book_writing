# Custom Samplers

## Overview

Custom samplers implement the `Sampler` interface to define application-specific sampling strategies. By overriding `__iter__` and `__len__`, you control exactly which samples appear in each epoch and in what order.

## The Sampler Interface

```python
from torch.utils.data import Sampler

class MySampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # Yield indices in desired order
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
```

## Stratified Sampler

Ensures each batch contains a proportional representation of each class:

```python
class StratifiedSampler(Sampler):
    """Yield indices such that each batch is approximately class-balanced."""
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.classes = np.unique(self.labels)
        self.class_indices = {
            c: np.where(self.labels == c)[0] for c in self.classes
        }

    def __iter__(self):
        # Shuffle within each class
        for c in self.classes:
            np.random.shuffle(self.class_indices[c])

        indices = []
        per_class = self.batch_size // len(self.classes)
        max_batches = min(len(v) // per_class for v in self.class_indices.values())

        for i in range(max_batches):
            for c in self.classes:
                start = i * per_class
                indices.extend(self.class_indices[c][start:start + per_class])
        return iter(indices)

    def __len__(self):
        per_class = self.batch_size // len(self.classes)
        max_batches = min(len(v) // per_class for v in self.class_indices.values())
        return max_batches * self.batch_size
```

## Curriculum Sampler

Implements curriculum learning by progressively introducing harder samples:

```python
class CurriculumSampler(Sampler):
    """Start with easy samples and gradually introduce harder ones."""
    def __init__(self, difficulties, epoch, total_epochs):
        self.difficulties = np.array(difficulties)
        self.sorted_indices = np.argsort(self.difficulties)
        # Fraction of data available increases linearly with epoch
        fraction = min(1.0, 0.3 + 0.7 * epoch / total_epochs)
        self.n_available = int(fraction * len(self.sorted_indices))

    def __iter__(self):
        available = self.sorted_indices[:self.n_available]
        np.random.shuffle(available)
        return iter(available.tolist())

    def __len__(self):
        return self.n_available
```

## Walk-Forward Sampler for Finance

For financial backtesting, a walk-forward sampler implements expanding or rolling windows:

```python
class WalkForwardSampler(Sampler):
    """Expanding-window sampler for time series cross-validation."""
    def __init__(self, n_samples, train_start, train_end):
        self.indices = list(range(train_start, train_end))

    def __iter__(self):
        # No shuffling — preserve temporal order
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
```

## Using Custom Samplers

```python
sampler = StratifiedSampler(labels=train_labels, batch_size=64)
loader = DataLoader(dataset, batch_size=64, sampler=sampler)
# Note: shuffle must be False (default) when using a custom sampler
```

## Key Takeaways

- Custom samplers implement `__iter__` (yielding indices) and `__len__`.
- Stratified sampling ensures class balance within batches.
- Curriculum learning progressively introduces harder samples.
- Walk-forward samplers enforce temporal ordering for financial backtesting.
