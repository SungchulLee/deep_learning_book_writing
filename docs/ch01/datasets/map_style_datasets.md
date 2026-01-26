# Map-Style Datasets

## Learning Objectives

By the end of this section, you will be able to:

- Implement map-style datasets for various data storage strategies
- Choose between in-memory, lazy-loading, and memory-mapped approaches
- Understand the trade-offs between memory usage and access speed
- Implement efficient random splitting and subsetting operations
- Apply transforms within the `__getitem__` method

---

## Overview

Map-style datasets support **random access**: you can retrieve any sample by its index in $O(1)$ time (for in-memory data) or $O(1)$ seeks (for disk-based data). This is the fundamental dataset type in PyTorch, enabling shuffled batching essential for stochastic gradient descent.

### The Map-Style Contract

A map-style dataset must implement:

```python
class MapStyleDataset(Dataset):
    def __len__(self) -> int:
        """Return N, the total number of samples."""
        pass
    
    def __getitem__(self, idx: int) -> Any:
        """Return the sample at index idx ∈ {0, 1, ..., N-1}."""
        pass
```

---

## Storage Strategies

### Strategy 1: In-Memory (RAM)

Store all data as tensors in memory for fastest access.

```python
from typing import Tuple
import torch
from torch.utils.data import Dataset

class InMemoryDataset(Dataset):
    """
    Keep features and labels in RAM as tensors.
    
    Pros:
        - Fastest access (no I/O in __getitem__)
        - Works well with DataLoader multiprocessing
    
    Cons:
        - Limited by available RAM
        - Initial loading time if data is large
    
    Best for:
        - Small to medium datasets (<10GB)
        - Data that fits comfortably in memory
        - When training speed is critical
    """
    
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert len(X) == len(y), "Feature and target count must match"
        self.X = X  # Features tensor
        self.y = y  # Labels tensor
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Direct tensor indexing - very fast
        return self.X[idx], self.y[idx]
```

**Usage Example:**

```python
# Create synthetic data in memory
X = torch.linspace(-1, 1, 1000).unsqueeze(1)  # [1000, 1]
y = (3 * X + 1).squeeze(1)  # [1000]

ds = InMemoryDataset(X, y)
print(f"Dataset size: {len(ds)}")
print(f"Sample 0: x={ds[0][0].item():.3f}, y={ds[0][1].item():.3f}")
```

---

### Strategy 2: Lazy Loading (Disk)

Store file paths; load data on demand in `__getitem__`.

```python
import os
from typing import List
import torch
from torch.utils.data import Dataset

class LazyTextDataset(Dataset):
    """
    Hold file paths in memory; read file contents on-demand.
    
    Pros:
        - Minimal RAM footprint (stores only paths)
        - Scales to datasets larger than RAM
        - Parallelizable via DataLoader workers
    
    Cons:
        - Each __getitem__ incurs I/O (slower per sample)
        - File system becomes bottleneck under heavy load
    
    Best for:
        - Large datasets that don't fit in memory
        - Image datasets with many files
        - When you need to process samples differently each epoch
    """
    
    def __init__(self, paths: List[str]):
        """
        Args:
            paths: List of file paths to load lazily
        """
        self.paths = list(paths)  # Just the paths; no file data loaded
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and parse file content when sample is requested."""
        path = self.paths[idx]
        
        with open(path, 'r') as f:
            content = f.read().strip()
        
        # Parse content (example: "value:42" -> tensor(42))
        value = int(content.split(':')[1])
        return torch.tensor(value, dtype=torch.long)
```

**Creating and Using Lazy Datasets:**

```python
import tempfile
import os

def create_sample_files(n: int = 10) -> List[str]:
    """Create temporary text files for demonstration."""
    tmpdir = tempfile.mkdtemp(prefix="lazy_demo_")
    paths = []
    
    for i in range(n):
        path = os.path.join(tmpdir, f"sample_{i}.txt")
        with open(path, 'w') as f:
            f.write(f"value:{i * 10}\n")
        paths.append(path)
    
    return paths

# Usage
paths = create_sample_files(10)
ds = LazyTextDataset(paths)

# Each access triggers file I/O
print(ds[0])   # Reads sample_0.txt
print(ds[5])   # Reads sample_5.txt
print(ds[0])   # Reads sample_0.txt again (no caching)
```

---

### Strategy 3: Memory-Mapped Files

Use NumPy `memmap` for OS-backed, on-demand paging.

```python
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

class MemmapDataset(Dataset):
    """
    Wrap a NumPy memmap file (on-demand paging by OS).
    
    Pros:
        - Works for arrays larger than RAM
        - OS handles paging efficiently
        - Random access with minimal memory footprint
        - Zero-copy tensor creation with torch.from_numpy
    
    Cons:
        - Requires contiguous array format on disk
        - May be slower than RAM for random access patterns
        - Platform-dependent behavior
    
    Best for:
        - Very large pre-processed datasets
        - When data naturally fits array format
        - Training on machines with limited RAM
    """
    
    def __init__(self, path: str, shape: Tuple[int, ...], dtype: str = 'float32'):
        """
        Args:
            path: Path to the .dat memmap file
            shape: Array shape (N, features_dim) or (N,)
            dtype: NumPy dtype string
        """
        # Read-only mapping; change to 'r+' for writable views
        self.mm = np.memmap(path, mode='r', dtype=dtype, shape=shape)
    
    def __len__(self) -> int:
        return self.mm.shape[0]
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Return sample as tensor.
        
        Note: torch.from_numpy shares memory with the ndarray view
        (zero-copy), but the mmap region is read-only.
        """
        arr = np.asarray(self.mm[idx])  # View into mmap'd region
        return torch.from_numpy(arr).float()
```

**Creating and Using Memory-Mapped Datasets:**

```python
import tempfile
import os

def create_memmap_file(shape: Tuple[int, int] = (10000, 128)) -> str:
    """Create a memmap file with sample data."""
    tmpdir = tempfile.mkdtemp(prefix="memmap_demo_")
    path = os.path.join(tmpdir, "data.dat")
    
    # Create and fill memmap
    mm = np.memmap(path, mode='w+', dtype='float32', shape=shape)
    for i in range(shape[0]):
        mm[i] = np.random.randn(shape[1]).astype('float32')
    mm.flush()  # Ensure data is written to disk
    
    return path, shape

# Usage
path, shape = create_memmap_file((10000, 128))
ds = MemmapDataset(path, shape)

print(f"Dataset size: {len(ds)}")
print(f"Sample shape: {ds[0].shape}")
print(f"Memory footprint: minimal (OS pages on demand)")
```

---

## Storage Strategy Comparison

| Strategy | RAM Usage | Access Speed | Scalability | Best Use Case |
|----------|-----------|--------------|-------------|---------------|
| **In-Memory** | High (all data) | Fastest | Limited by RAM | Small-medium datasets |
| **Lazy Loading** | Low (paths only) | Slower (I/O) | Excellent | Large file collections |
| **Memory-Mapped** | Low (paged) | Medium | Excellent | Large arrays |

### Decision Flowchart

```
Does data fit in RAM?
├── Yes → Use In-Memory
└── No → Is data in array format?
    ├── Yes → Use Memory-Mapped
    └── No → Use Lazy Loading
```

---

## Dataset Splitting

PyTorch provides utilities for splitting map-style datasets without copying data.

### Using `random_split`

```python
import torch
from torch.utils.data import Dataset, random_split

class SimpleDataset(Dataset):
    def __init__(self, n: int = 100):
        self.data = torch.arange(n)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

# Create dataset
ds = SimpleDataset(n=100)

# Split into train/val/test (70/20/10)
generator = torch.Generator().manual_seed(42)  # For reproducibility
train_ds, val_ds, test_ds = random_split(
    ds, 
    [70, 20, 10], 
    generator=generator
)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
print(f"Train indices (first 5): {train_ds.indices[:5]}")
```

### Understanding `Subset`

`random_split` returns `Subset` objects, which are lightweight wrappers:

```python
from torch.utils.data import Subset

# Subset references original dataset by indices (no data copy)
print(f"Original dataset: {id(ds)}")
print(f"Train subset references: {id(train_ds.dataset)}")  # Same id!

# Access through subset uses original indexing
original_idx = train_ds.indices[0]
print(f"train_ds[0] == ds[{original_idx}]")
print(f"  train_ds[0]: {train_ds[0]}")
print(f"  ds[{original_idx}]: {ds[original_idx]}")
```

### Creating Custom Subsets

```python
# Take specific indices from a dataset
specific_subset = Subset(ds, indices=[0, 5, 10, 15, 20])
print(f"Subset length: {len(specific_subset)}")
print(f"Elements: {[specific_subset[i].item() for i in range(len(specific_subset))]}")

# Nested subsetting (no performance penalty)
nested = Subset(train_ds, indices=list(range(10)))
print(f"Nested subset length: {len(nested)}")
```

---

## Manual Batching (Without DataLoader)

Understanding the mechanics helps debug DataLoader issues.

```python
import torch
from torch.utils.data import Dataset
from typing import Iterator, Tuple

class TinyDataset(Dataset):
    def __init__(self, n: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 2, generator=g)  # [N, 2]
        self.y = (self.X.sum(dim=1) > 0).long()  # [N]
    
    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

def manual_batches(
    ds: Dataset, 
    batch_size: int = 4, 
    shuffle: bool = True, 
    seed: int = 123
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate batches manually (for educational purposes).
    
    This mirrors what DataLoader does internally.
    """
    n = len(ds)
    indices = torch.arange(n)
    
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        indices = indices[torch.randperm(n, generator=g)]
    
    for start in range(0, n, batch_size):
        batch_indices = indices[start:start + batch_size].tolist()
        
        # Collect samples
        samples = [ds[i] for i in batch_indices]
        
        # Separate features and targets, then stack
        Xb, yb = zip(*samples)
        Xb = torch.stack(Xb)  # [B, 2]
        yb = torch.stack(yb)  # [B]
        
        yield Xb, yb

# Usage
ds = TinyDataset(n=9)

print("Batches with batch_size=4:")
for Xb, yb in manual_batches(ds, batch_size=4, shuffle=False):
    print(f"  X: {Xb.shape}, y: {yb.shape}")
```

**Output:**
```
Batches with batch_size=4:
  X: torch.Size([4, 2]), y: torch.Size([4])
  X: torch.Size([4, 2]), y: torch.Size([4])
  X: torch.Size([1, 2]), y: torch.Size([1])  # Last batch smaller
```

---

## Working with Transforms

Transforms modify data on-the-fly during `__getitem__`.

### Transform Classes

```python
import torch
from typing import Callable, Optional

class Compose:
    """Chain multiple transforms: x → f_n(...f_2(f_1(x)))."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class Standardize:
    """Standardize: x → (x - μ) / (σ + ε)."""
    
    def __init__(self, mean: float, std: float, eps: float = 1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

class AddNoise:
    """Add Gaussian noise: x → x + ε, ε ~ N(0, σ²)."""
    
    def __init__(self, std: float = 0.05, seed: int = 0):
        self.std = std
        self.generator = torch.Generator().manual_seed(seed)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.normal(0, self.std, x.shape, generator=self.generator)
        return x + noise
```

### Dataset with Transforms

```python
from typing import Tuple, Optional, Callable

class TransformableDataset(Dataset):
    """
    Dataset with optional input and target transforms.
    
    Transforms are applied on-the-fly in __getitem__, not stored.
    """
    
    def __init__(
        self, 
        n: int = 64, 
        noise_std: float = 0.3, 
        seed: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        g = torch.Generator().manual_seed(seed)
        
        # Generate base data: y = 3x + 1 + noise
        self.x = torch.empty(n, 1).uniform_(-2, 2, generator=g)
        self.y = 3 * self.x + 1 + noise_std * torch.randn(self.x.shape, generator=g)
        
        self.transform = transform
        self.target_transform = target_transform
        
        # Store statistics for potential normalization
        self.x_mean = self.x.mean()
        self.x_std = self.x.std()
        self.y_mean = self.y.mean()
        self.y_std = self.y.std()
    
    def __len__(self) -> int:
        return self.x.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xi = self.x[idx]
        yi = self.y[idx]
        
        if self.transform is not None:
            xi = self.transform(xi)
        
        if self.target_transform is not None:
            yi = self.target_transform(yi)
        
        return xi, yi
```

**Using Transforms:**

```python
# Create base dataset to get statistics
base_ds = TransformableDataset(n=64, seed=42)

# Build transform pipeline
x_transform = Compose([
    Standardize(base_ds.x_mean.item(), base_ds.x_std.item()),
    AddNoise(std=0.1, seed=123)
])

y_transform = Standardize(base_ds.y_mean.item(), base_ds.y_std.item())

# Create transformed dataset
transformed_ds = TransformableDataset(
    n=64, 
    seed=42,
    transform=x_transform,
    target_transform=y_transform
)

# Compare raw vs transformed
print("Raw sample 0:")
print(f"  x={base_ds[0][0].item():.4f}, y={base_ds[0][1].item():.4f}")

print("Transformed sample 0:")
print(f"  x={transformed_ds[0][0].item():.4f}, y={transformed_ds[0][1].item():.4f}")
```

---

## Performance Considerations

### 1. Avoid Repeated Computation in `__getitem__`

```python
# ❌ BAD: Recomputes statistics every call
def __getitem__(self, idx):
    mean = self.data.mean()  # O(N) operation!
    return (self.data[idx] - mean)

# ✅ GOOD: Precompute in __init__
def __init__(self, data):
    self.data = data
    self.mean = data.mean()  # Computed once

def __getitem__(self, idx):
    return self.data[idx] - self.mean  # O(1)
```

### 2. Use Efficient Data Types

```python
# Memory comparison for 1M samples, 128 features
# float64: 1M × 128 × 8 bytes = 1 GB
# float32: 1M × 128 × 4 bytes = 512 MB
# float16: 1M × 128 × 2 bytes = 256 MB

# For most training, float32 is sufficient
data = data.to(torch.float32)
```

### 3. Consider DataLoader Workers

```python
from torch.utils.data import DataLoader

# For lazy-loading datasets, parallel workers help
loader = DataLoader(
    lazy_dataset,
    batch_size=32,
    num_workers=4,  # Parallel I/O
    pin_memory=True  # Faster GPU transfer
)
```

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **In-Memory** | Fast but limited by RAM |
| **Lazy Loading** | Scales infinitely, slower per sample |
| **Memory-Mapped** | OS-paged, great for large arrays |
| **random_split** | Creates non-overlapping Subsets |
| **Subset** | References original data (no copy) |
| **Transforms** | Applied on-the-fly in `__getitem__` |

---

## Further Reading

- Section 1.12.2: Custom Datasets (advanced patterns)
- Section 1.12.4: Data Transforms (torchvision integration)
- Section 1.13: DataLoaders (batching, multiprocessing)
