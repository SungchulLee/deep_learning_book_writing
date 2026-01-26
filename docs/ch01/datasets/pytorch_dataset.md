# PyTorch Dataset Class

## Learning Objectives

By the end of this section, you will be able to:

- Understand the fundamental role of the `Dataset` class in PyTorch's data loading pipeline
- Distinguish between map-style and iterable-style datasets
- Implement the required methods (`__len__`, `__getitem__`) for map-style datasets
- Understand Python's sequence protocol and how it enables iteration over datasets
- Make informed decisions about when to use each dataset style

---

## Overview

The `torch.utils.data.Dataset` class is the foundational abstraction in PyTorch for representing datasets. It serves as the interface between your raw data (files, arrays, databases) and PyTorch's training machinery. Understanding this abstraction is essential for building efficient, maintainable deep learning pipelines.

### The Core Abstraction

A dataset in PyTorch is fundamentally a **mapping** from indices to samples:

$$
\text{Dataset}: \{0, 1, 2, \ldots, N-1\} \rightarrow \text{Samples}
$$

This simple mathematical formulation captures the essence of what a dataset does: given an integer index $i$, it returns the $i$-th sample in your collection.

---

## Map-Style vs Iterable-Style Datasets

PyTorch provides two dataset paradigms, each optimized for different use cases:

| Aspect | Map-Style | Iterable-Style |
|--------|-----------|----------------|
| **Base Class** | `torch.utils.data.Dataset` | `torch.utils.data.IterableDataset` |
| **Required Methods** | `__len__()`, `__getitem__(idx)` | `__iter__()` |
| **Access Pattern** | Random access by index | Sequential iteration only |
| **Size Knowledge** | Known upfront | May be unknown |
| **Shuffling** | Native support via `DataLoader` | Manual implementation required |
| **Best For** | Finite, indexable data | Streams, very large datasets |

### When to Use Map-Style

Map-style datasets are the default choice for most machine learning tasks:

- **Image classification**: Load images from disk by index
- **Tabular data**: Access rows from in-memory arrays
- **Fixed-size text corpora**: Index into preprocessed documents
- **Any scenario requiring shuffling**: Random sampling is trivial

### When to Use Iterable-Style

Iterable-style datasets excel in streaming scenarios:

- **Real-time data feeds**: Stock prices, sensor readings
- **Database queries**: Streaming results from SQL
- **Very large datasets**: When random access is impractical
- **Distributed data sources**: Sharded across multiple files

---

## Implementing a Map-Style Dataset

### The Minimal Implementation

A map-style dataset requires exactly two methods:

```python
import torch
from torch.utils.data import Dataset

class MinimalDataset(Dataset):
    """The simplest possible map-style dataset."""
    
    # Class attribute (shared across instances)
    # In practice, use instance attributes via __init__
    data = torch.tensor([10, 20, 30, 40])
    
    def __len__(self):
        """Return the total number of samples."""
        return self.data.numel()
    
    def __getitem__(self, idx):
        """Map index to sample (random-access)."""
        return self.data[idx]
```

**Key Implementation Rules:**

1. `__len__()` must return an integer representing the dataset size $N$
2. `__getitem__(idx)` must accept indices in $\{0, 1, \ldots, N-1\}$
3. `__getitem__` should be **pure** (no side effects) and **fast**
4. Return tensors (or tuples of tensors) so `DataLoader` can batch them

### The Standard Pattern with `__init__`

In practice, datasets receive configuration through their constructor:

```python
class StandardDataset(Dataset):
    """Map-style dataset with proper initialization."""
    
    def __init__(self, n: int = 8):
        """
        Initialize the dataset.
        
        Args:
            n: Number of samples to generate
        """
        # Store tensor data: [0, 1, 2, ..., n-1]
        # In real applications, you'd load/prepare data here
        self.data = torch.arange(n)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.data.numel()
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Map index to sample.
        
        Args:
            idx: Sample index in [0, len-1]
            
        Returns:
            The idx-th sample as a tensor
        """
        return self.data[idx]
```

### Returning Multiple Values

Most supervised learning tasks require input-target pairs:

```python
from typing import Tuple

class SupervisedDataset(Dataset):
    """Dataset returning (input, target) pairs."""
    
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert len(X) == len(y), "Features and targets must have same length"
        self.X = X  # Features
        self.y = y  # Targets
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (feature_i, target_i) tuple."""
        return self.X[idx], self.y[idx]
```

---

## The Sequence Protocol

A subtle but important point: `Dataset` does **not** define `__iter__`. Yet you can still iterate over a dataset:

```python
ds = StandardDataset(n=5)

# This works despite no __iter__ method!
for sample in ds:
    print(sample)
```

This works because Python's **sequence protocol** allows iteration over any object with `__len__` and `__getitem__`. Python automatically generates indices 0, 1, 2, ... until `__getitem__` raises `IndexError`.

### Practical Implications

```python
from collections.abc import Iterable, Iterator

ds = StandardDataset(n=5)

# Is it iterable? Python checks for __iter__ first
print(isinstance(ds, Iterable))  # May be False

# Is it an iterator? Definitely not (no __next__)
print(isinstance(ds, Iterator))  # False

# Yet iteration works via sequence protocol
it = iter(ds)  # Creates a sequence iterator
print(next(it))  # Works!

# But the dataset itself is not an iterator
try:
    next(ds)  # Raises TypeError
except TypeError as e:
    print(f"next(ds) fails: {e}")
```

**Key Insight**: A `Dataset` is an **indexed collection**, not an iterator. You access it by index (`ds[i]`), and Python's sequence protocol enables `for` loops and `iter()`.

---

## Mathematical Perspective

### Dataset as a Function

Formally, a dataset $\mathcal{D}$ with $N$ samples defines a function:

$$
\mathcal{D}: \mathbb{Z}_N \rightarrow \mathcal{X} \times \mathcal{Y}
$$

where $\mathbb{Z}_N = \{0, 1, \ldots, N-1\}$ is the index space, $\mathcal{X}$ is the feature space, and $\mathcal{Y}$ is the target space.

For a sample at index $i$:

$$
\mathcal{D}(i) = (x_i, y_i)
$$

### Random Access Enables Shuffling

The map-style design enables the **uniform sampling** required for stochastic gradient descent:

$$
i \sim \text{Uniform}(\{0, 1, \ldots, N-1\})
$$

$$
(x_i, y_i) = \mathcal{D}(i)
$$

This random access is precisely what `DataLoader` exploits when `shuffle=True`.

---

## Best Practices

### 1. Keep `__getitem__` Pure and Fast

```python
# ❌ BAD: Side effects in __getitem__
def __getitem__(self, idx):
    self.access_count += 1  # Side effect!
    return self.data[idx]

# ✅ GOOD: Pure function, no side effects
def __getitem__(self, idx):
    return self.data[idx]
```

### 2. Return Tensors for DataLoader Compatibility

```python
# ❌ BAD: Returns Python list
def __getitem__(self, idx):
    return [self.data[idx], self.labels[idx]]

# ✅ GOOD: Returns tensor tuple
def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]
```

### 3. Type Hints Improve Clarity

```python
from typing import Tuple

def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit types aid understanding and IDE support."""
    return self.X[idx], self.y[idx]
```

### 4. Document Index Bounds

```python
def __getitem__(self, idx: int) -> torch.Tensor:
    """
    Retrieve sample by index.
    
    Args:
        idx: Index in range [0, len(self)-1]
        
    Returns:
        Sample tensor at the given index
        
    Raises:
        IndexError: If idx is out of bounds
    """
    if idx < 0 or idx >= len(self):
        raise IndexError(f"Index {idx} out of range [0, {len(self)-1}]")
    return self.data[idx]
```

---

## Complete Example

```python
#!/usr/bin/env python3
"""Complete map-style dataset example."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class ToyRegressionDataset(Dataset):
    """
    Synthetic regression dataset: y = 3x + 1 + noise.
    
    Demonstrates all best practices for map-style datasets.
    """
    
    def __init__(self, n: int = 64, noise_std: float = 0.3, seed: int = 0):
        """
        Generate synthetic regression data.
        
        Args:
            n: Number of samples
            noise_std: Standard deviation of Gaussian noise
            seed: Random seed for reproducibility
        """
        g = torch.Generator().manual_seed(seed)
        
        # Features: x ~ Uniform[-2, 2]
        self.x = torch.empty(n, 1).uniform_(-2, 2, generator=g)
        
        # Targets: y = 3x + 1 + epsilon
        self.y = 3 * self.x + 1 + noise_std * torch.randn(self.x.shape, generator=g)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.x.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (feature, target) pair at index idx."""
        return self.x[idx], self.y[idx]

def main():
    # Create dataset
    ds = ToyRegressionDataset(n=100, seed=42)
    print(f"Dataset size: {len(ds)}")
    
    # Access individual samples
    x0, y0 = ds[0]
    print(f"First sample: x={x0.item():.3f}, y={y0.item():.3f}")
    
    # Iterate (via sequence protocol)
    print("\nFirst 3 samples via iteration:")
    for i, (x, y) in enumerate(ds):
        if i >= 3:
            break
        print(f"  [{i}] x={x.item():.3f}, y={y.item():.3f}")
    
    # Use with DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    print(f"\nDataLoader batches: {len(loader)}")
    
    for batch_x, batch_y in loader:
        print(f"Batch shape: x={batch_x.shape}, y={batch_y.shape}")
        break  # Just show first batch

if __name__ == "__main__":
    main()
```

**Output:**
```
Dataset size: 100
First sample: x=0.683, y=3.049

First 3 samples via iteration:
  [0] x=0.683, y=3.049
  [1] x=-1.127, y=-2.380
  [2] x=1.891, y=6.673

DataLoader batches: 4
Batch shape: x=torch.Size([32, 1]), y=torch.Size([32, 1])
```

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Dataset Abstraction** | Maps indices {0, ..., N-1} to samples |
| **Required Methods** | `__len__()` returns size, `__getitem__(idx)` returns sample |
| **Sequence Protocol** | Enables iteration without explicit `__iter__` |
| **Purity** | `__getitem__` should have no side effects |
| **Return Type** | Tensors or tensor tuples for DataLoader compatibility |
| **Map vs Iterable** | Use map-style for random access, iterable for streams |

---

## Further Reading

- [PyTorch Data Loading Documentation](https://pytorch.org/docs/stable/data.html)
- [Writing Custom Datasets Tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- Section 1.13: DataLoaders (batching, shuffling, multiprocessing)
