# Dataset Class

## Overview

PyTorch's `torch.utils.data.Dataset` is an abstract class that represents a dataset. Every custom dataset must inherit from it and implement two core methods, creating a uniform interface that the rest of the training pipeline depends on.

## The Dataset Interface

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, target
```

The two required methods are:

- **`__len__`**: Returns the total number of samples. Called by `DataLoader` to determine epoch length and by samplers to define index ranges.
- **`__getitem__`**: Returns a single sample given an index. This is where data loading, preprocessing, and augmentation occur.

## Map-Style vs. Iterable-Style Datasets

PyTorch supports two dataset paradigms:

**Map-style datasets** (`Dataset`) support random access via integer indexing. They are the standard choice when the entire dataset fits in memory or when individual samples can be loaded efficiently from disk.

**Iterable-style datasets** (`IterableDataset`) yield samples sequentially and are designed for streaming data, very large datasets, or data generated on-the-fly:

```python
from torch.utils.data import IterableDataset

class StreamDataset(IterableDataset):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        for record in self.data_source:
            yield self.process(record)
```

Iterable-style datasets do not support `__len__` or `__getitem__`, which means certain `DataLoader` features (e.g., custom samplers, shuffling) require additional care.

## Lazy vs. Eager Loading

A critical design decision is **when** data is loaded into memory:

```python
# Eager loading: entire dataset in memory at __init__
class EagerDataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.load(file_path)  # All data loaded upfront

    def __getitem__(self, idx):
        return self.data[idx]

# Lazy loading: samples loaded on demand
class LazyDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths  # Store paths only

    def __getitem__(self, idx):
        return torch.load(self.file_paths[idx])  # Load per access
```

Eager loading provides fast `__getitem__` at the cost of memory. Lazy loading conserves memory but incurs I/O overhead on each access—overhead that can be mitigated by multi-process data loading (Section 5.2).

## The Transform Pattern

The `Dataset` class conventionally accepts a `transform` argument, establishing a clean separation between raw data storage and preprocessing:

```python
dataset = MyDataset(data, targets, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
]))
```

This design allows the same underlying data to be used with different preprocessing pipelines (e.g., different augmentation for training vs. validation) without data duplication.

## Quantitative Finance Application

Financial datasets require careful temporal handling:

```python
class TimeSeriesDataset(Dataset):
    """Rolling-window dataset for financial time series."""
    def __init__(self, prices, window_size, horizon=1):
        self.prices = prices
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.prices) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        x = self.prices[idx : idx + self.window_size]
        y = self.prices[idx + self.window_size : idx + self.window_size + self.horizon]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
```

Key considerations: temporal ordering must be preserved (no shuffling across time for train/test splits), and look-ahead bias must be carefully avoided in the `__getitem__` implementation.

## Key Takeaways

- `Dataset` provides a uniform interface (`__len__`, `__getitem__`) that decouples data representation from the training loop.
- Map-style datasets support random access; iterable-style datasets support sequential streaming.
- The transform pattern cleanly separates raw data from preprocessing, enabling different pipelines for training and evaluation.
- Financial datasets must respect temporal ordering to avoid look-ahead bias.
