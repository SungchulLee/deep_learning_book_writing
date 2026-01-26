# Built-in Datasets

## Learning Objectives

By the end of this section, you will be able to:

- Access and use PyTorch's built-in dataset utilities
- Work with `TensorDataset` for simple tensor-based data
- Apply `random_split` and `Subset` for data splitting
- Use `ConcatDataset` to combine multiple datasets
- Understand the `ChainDataset` utility for iterable datasets

---

## Overview

PyTorch provides several built-in dataset classes and utilities in `torch.utils.data` that cover common use cases without requiring custom implementations. These utilities follow the same `Dataset` interface and integrate seamlessly with `DataLoader`.

---

## TensorDataset

`TensorDataset` wraps tensors into a map-style dataset where each sample is a tuple of corresponding elements from each tensor.

### Basic Usage

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Create feature and label tensors
X = torch.randn(100, 10)  # 100 samples, 10 features
y = torch.randint(0, 2, (100,))  # 100 binary labels

# Wrap in TensorDataset
dataset = TensorDataset(X, y)

# Access like any map-style dataset
print(f"Dataset length: {len(dataset)}")  # 100

sample = dataset[0]
print(f"Sample type: {type(sample)}")  # tuple
print(f"Features shape: {sample[0].shape}")  # torch.Size([10])
print(f"Label: {sample[1]}")  # tensor(0) or tensor(1)
```

### How It Works

```python
# TensorDataset is essentially:
class TensorDataset(Dataset):
    def __init__(self, *tensors):
        # All tensors must have same first dimension
        assert all(tensors[0].size(0) == t.size(0) for t in tensors)
        self.tensors = tensors
    
    def __len__(self):
        return self.tensors[0].size(0)
    
    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.tensors)
```

### Multiple Tensors

```python
# Works with any number of tensors
X = torch.randn(100, 10)  # Features
y = torch.randint(0, 5, (100,))  # Labels
weights = torch.rand(100)  # Sample weights
ids = torch.arange(100)  # Sample IDs

dataset = TensorDataset(X, y, weights, ids)

features, label, weight, sample_id = dataset[0]
print(f"Feature shape: {features.shape}")
print(f"Label: {label}, Weight: {weight:.3f}, ID: {sample_id}")
```

### With DataLoader

```python
loader = DataLoader(
    TensorDataset(X, y),
    batch_size=32,
    shuffle=True
)

for batch_X, batch_y in loader:
    print(f"Batch X: {batch_X.shape}, Batch y: {batch_y.shape}")
    break
```

---

## random_split

`random_split` divides a dataset into non-overlapping `Subset` objects.

### Basic Splitting

```python
from torch.utils.data import random_split

# Create a dataset
dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))

# Split into train/val/test (70/15/15)
train_size = 70
val_size = 15
test_size = 15

train_ds, val_ds, test_ds = random_split(
    dataset, 
    [train_size, val_size, test_size]
)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
```

### Reproducible Splits

```python
# Use a Generator for reproducibility
generator = torch.Generator().manual_seed(42)

train_ds, val_ds = random_split(
    dataset,
    [80, 20],
    generator=generator
)

print(f"First train index: {train_ds.indices[0]}")  # Same every run with seed 42
```

### Percentage-Based Splitting

```python
def split_by_percentage(dataset, train_pct=0.7, val_pct=0.15, seed=42):
    """Split dataset by percentages."""
    n = len(dataset)
    train_n = int(n * train_pct)
    val_n = int(n * val_pct)
    test_n = n - train_n - val_n  # Remainder goes to test
    
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_n, val_n, test_n], generator=generator)

train_ds, val_ds, test_ds = split_by_percentage(dataset, 0.7, 0.15)
```

---

## Subset

`Subset` provides a view into a dataset at specific indices without copying data.

### Creating Subsets

```python
from torch.utils.data import Subset

dataset = TensorDataset(torch.arange(100).float(), torch.arange(100))

# Create subset with specific indices
indices = [0, 10, 20, 30, 40]
subset = Subset(dataset, indices)

print(f"Subset length: {len(subset)}")  # 5
print(f"Subset[0]: {subset[0]}")  # (tensor(0.), tensor(0))
print(f"Subset[2]: {subset[2]}")  # (tensor(20.), tensor(20))
```

### Understanding the Reference

```python
# Subset references the original dataset (no copy)
print(f"Subset.dataset is original: {subset.dataset is dataset}")  # True

# Modifying original affects subset (if tensors are modified)
# But typically you create new tensors, so this isn't an issue
```

### Nested Subsets

```python
# You can create subsets of subsets
train_ds, val_ds = random_split(dataset, [80, 20])

# Take first 10 samples from training set
mini_train = Subset(train_ds, list(range(10)))
print(f"Mini train length: {len(mini_train)}")  # 10
```

### Practical Use Case: Stratified Sampling

```python
def get_class_indices(labels, class_value):
    """Get indices where label equals class_value."""
    return (labels == class_value).nonzero(as_tuple=True)[0].tolist()

# Create balanced subset
X = torch.randn(100, 10)
y = torch.cat([torch.zeros(70), torch.ones(30)]).long()  # Imbalanced

dataset = TensorDataset(X, y)

# Get 20 samples from each class
class_0_indices = get_class_indices(y, 0)[:20]
class_1_indices = get_class_indices(y, 1)[:20]
balanced_indices = class_0_indices + class_1_indices

balanced_subset = Subset(dataset, balanced_indices)
print(f"Balanced subset size: {len(balanced_subset)}")  # 40
```

---

## ConcatDataset

`ConcatDataset` combines multiple map-style datasets into a single dataset.

### Basic Concatenation

```python
from torch.utils.data import ConcatDataset

# Create separate datasets
ds1 = TensorDataset(torch.randn(50, 10), torch.zeros(50).long())
ds2 = TensorDataset(torch.randn(30, 10), torch.ones(30).long())
ds3 = TensorDataset(torch.randn(20, 10), torch.full((20,), 2).long())

# Concatenate
combined = ConcatDataset([ds1, ds2, ds3])

print(f"Combined length: {len(combined)}")  # 100
print(f"Sample 0 (from ds1): label = {combined[0][1]}")  # 0
print(f"Sample 60 (from ds2): label = {combined[60][1]}")  # 1
print(f"Sample 90 (from ds3): label = {combined[90][1]}")  # 2
```

### How Indexing Works

```python
# ConcatDataset maintains cumulative sizes
# Index 0-49 → ds1[0-49]
# Index 50-79 → ds2[0-29]
# Index 80-99 → ds3[0-19]

# Access the cumulative sizes
print(f"Cumulative sizes: {combined.cumulative_sizes}")  # [50, 80, 100]
```

### Practical Use Case: Multi-Source Data

```python
# Combine data from multiple files/sources
def load_dataset_from_file(filepath):
    """Load dataset from a file (mock implementation)."""
    n = torch.randint(50, 100, (1,)).item()
    return TensorDataset(torch.randn(n, 10), torch.randint(0, 5, (n,)))

# Load and combine
datasets = [load_dataset_from_file(f"data_{i}.pt") for i in range(5)]
combined = ConcatDataset(datasets)

print(f"Total samples: {len(combined)}")
```

---

## ChainDataset

`ChainDataset` is the iterable-style equivalent of `ConcatDataset`.

### Basic Usage

```python
from torch.utils.data import ChainDataset, IterableDataset

class SimpleStream(IterableDataset):
    def __init__(self, n, value):
        self.n = n
        self.value = value
    
    def __iter__(self):
        for _ in range(self.n):
            yield torch.tensor([self.value])

# Create iterable datasets
stream1 = SimpleStream(5, 1.0)
stream2 = SimpleStream(3, 2.0)
stream3 = SimpleStream(4, 3.0)

# Chain them
chained = ChainDataset([stream1, stream2, stream3])

# Iterate through
for sample in chained:
    print(sample.item(), end=" ")  # 1.0 1.0 1.0 1.0 1.0 2.0 2.0 2.0 3.0 3.0 3.0 3.0
```

### Practical Use Case: Multiple Data Files

```python
class FileStreamDataset(IterableDataset):
    """Stream data from a file."""
    
    def __init__(self, filepath):
        self.filepath = filepath
    
    def __iter__(self):
        with open(self.filepath, 'r') as f:
            for line in f:
                yield process_line(line)

# Chain multiple file streams
file_datasets = [FileStreamDataset(f"data_{i}.txt") for i in range(10)]
combined_stream = ChainDataset(file_datasets)
```

---

## Comparison Table

| Utility | Type | Purpose | Use Case |
|---------|------|---------|----------|
| `TensorDataset` | Map-style | Wrap tensors | Simple tensor data |
| `Subset` | Map-style | View at indices | Splitting, filtering |
| `random_split` | Utility | Random partition | Train/val/test splits |
| `ConcatDataset` | Map-style | Combine datasets | Multi-source data |
| `ChainDataset` | Iterable | Chain iterables | Sequential streams |

---

## Practical Examples

### Example 1: Complete Training Pipeline

```python
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

# Generate data
torch.manual_seed(42)
X = torch.randn(1000, 20)
y = (X.sum(dim=1) > 0).long()  # Binary classification

# Create dataset
dataset = TensorDataset(X, y)

# Split into train/val/test
generator = torch.Generator().manual_seed(42)
train_ds, val_ds, test_ds = random_split(
    dataset, 
    [700, 150, 150], 
    generator=generator
)

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Training loop
for epoch in range(5):
    for batch_X, batch_y in train_loader:
        # Forward pass, loss, backward pass...
        pass
    
    # Validation
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            # Evaluate...
            pass
```

### Example 2: Combining Multiple Data Sources

```python
from torch.utils.data import ConcatDataset, TensorDataset, Subset

# Simulate loading from multiple files
datasets = []
for i in range(5):
    n = torch.randint(100, 200, (1,)).item()
    X = torch.randn(n, 10) + i  # Different distributions
    y = torch.full((n,), i)  # Different labels
    datasets.append(TensorDataset(X, y))

# Combine all
full_dataset = ConcatDataset(datasets)
print(f"Total samples: {len(full_dataset)}")

# Create balanced subset (equal samples per source)
samples_per_source = 50
balanced_indices = []
offset = 0
for ds in datasets:
    indices = list(range(offset, offset + samples_per_source))
    balanced_indices.extend(indices)
    offset += len(ds)

balanced_dataset = Subset(full_dataset, balanced_indices)
print(f"Balanced subset size: {len(balanced_dataset)}")  # 250
```

### Example 3: K-Fold Cross Validation

```python
def k_fold_split(dataset, k=5, fold=0, seed=42):
    """
    Create train/val split for k-fold cross validation.
    
    Args:
        dataset: Full dataset
        k: Number of folds
        fold: Current fold index (0 to k-1)
        seed: Random seed
    
    Returns:
        train_ds, val_ds: Subset objects
    """
    n = len(dataset)
    
    # Create shuffled indices
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    
    # Calculate fold boundaries
    fold_size = n // k
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold < k - 1 else n
    
    # Split indices
    val_indices = indices[val_start:val_end]
    train_indices = indices[:val_start] + indices[val_end:]
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

# Usage
dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))

for fold in range(5):
    train_ds, val_ds = k_fold_split(dataset, k=5, fold=fold)
    print(f"Fold {fold}: train={len(train_ds)}, val={len(val_ds)}")
```

---

## Summary

| Utility | Key Point |
|---------|-----------|
| **TensorDataset** | Quick wrapper for tensor data |
| **random_split** | Creates non-overlapping Subsets |
| **Subset** | View into dataset at specific indices |
| **ConcatDataset** | Combine map-style datasets |
| **ChainDataset** | Chain iterable datasets |
| **Generator** | Use for reproducible splits |

---

## Further Reading

- Section 1.12.1: Map-Style Datasets (custom implementations)
- Section 1.13: DataLoaders (batching, workers)
- [torch.utils.data Documentation](https://pytorch.org/docs/stable/data.html)
