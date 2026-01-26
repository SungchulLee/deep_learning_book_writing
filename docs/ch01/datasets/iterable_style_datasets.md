# Iterable-Style Datasets

## Learning Objectives

By the end of this section, you will be able to:

- Implement iterable-style datasets using generators and iterator objects
- Understand why `DataLoader` does not sample `IterableDataset` 
- Implement worker-aware sharding for multi-process data loading
- Apply shuffle buffer and block shuffle techniques for streaming data
- Choose between map-style and iterable-style datasets appropriately

---

## Overview

Iterable-style datasets provide **sequential access** to data through Python's iterator protocol. Unlike map-style datasets, they don't support random indexing—you can only consume data in order by iterating.

### The Iterable-Style Contract

```python
from torch.utils.data import IterableDataset

class MyIterableDataset(IterableDataset):
    def __iter__(self):
        """Yield samples one by one."""
        yield sample_0
        yield sample_1
        # ...
```

### Key Characteristics

| Property | Iterable-Style | Map-Style |
|----------|---------------|-----------|
| **Access Pattern** | Sequential only | Random access |
| **Required Method** | `__iter__()` | `__len__()`, `__getitem__()` |
| **Size Knowledge** | Often unknown | Always known |
| **DataLoader Shuffling** | Not supported | Native support |
| **Worker Sharding** | Manual | Automatic |

---

## When to Use Iterable-Style

### Ideal Use Cases

1. **Streaming data**: Real-time feeds, sensor data, API responses
2. **Database queries**: Streaming SQL results without loading all rows
3. **Very large datasets**: When random access is impractical
4. **Infinite data**: Procedurally generated samples
5. **Distributed data sources**: Files spread across storage systems

### When NOT to Use

- When you need shuffling (use map-style instead)
- When dataset size is needed for progress bars
- When random sampling is required for training

---

## Implementation Patterns

### Pattern 1: Generator-Based

The simplest approach uses Python generators in `__iter__`:

```python
import torch
from torch.utils.data import IterableDataset

class RandomStreamGenerator(IterableDataset):
    """
    Iterable dataset using a generator.
    
    Each pass through the dataset yields the SAME sequence
    (useful for reproducibility in demonstrations).
    """
    
    def __init__(self, total: int = 5, seed: int = 0):
        """
        Args:
            total: Number of samples to generate per iteration
            seed: Random seed (re-applied each iteration)
        """
        self.total = total
        self.seed = seed
    
    def __iter__(self):
        """Yield samples via generator."""
        # Re-seed each iteration for reproducibility
        g = torch.Generator().manual_seed(self.seed)
        
        for _ in range(self.total):
            yield torch.randn(3, generator=g)
```

**Usage:**

```python
ds = RandomStreamGenerator(total=4, seed=42)

print("Pass 1:")
for sample in ds:
    print(f"  {sample.tolist()}")

print("\nPass 2 (same sequence due to re-seeding):")
for sample in ds:
    print(f"  {sample.tolist()}")
```

### Pattern 2: Iterator Object

For more complex state management, return a dedicated iterator object:

```python
class _RandomStreamIterator:
    """
    Stateful iterator with its own RNG and position cursor.
    
    This object is created fresh by __iter__, so each iteration
    gets independent state.
    """
    
    def __init__(self, total: int, seed: int):
        self.total = total
        self.position = 0
        self.generator = torch.Generator().manual_seed(seed)
    
    def __iter__(self):
        """Iterator must return itself."""
        return self
    
    def __next__(self):
        """Generate next sample or raise StopIteration."""
        if self.position >= self.total:
            raise StopIteration
        
        self.position += 1
        return torch.randn(3, generator=self.generator)


class RandomStreamIterator(IterableDataset):
    """
    Iterable dataset returning fresh iterator objects.
    
    State lives in _RandomStreamIterator, not in the dataset itself.
    This ensures clean iteration semantics.
    """
    
    def __init__(self, total: int = 5, seed: int = 0):
        self.total = total
        self.seed = seed
    
    def __iter__(self):
        """Return fresh iterator with reset state."""
        return _RandomStreamIterator(self.total, self.seed)
```

**Why Use This Pattern?**

- State (position, RNG) is encapsulated in the iterator
- Multiple concurrent iterations don't interfere
- Clean separation of concerns

---

## DataLoader Behavior

**Critical**: `DataLoader` does NOT use a `Sampler` for `IterableDataset`. It simply consumes whatever `__iter__` yields.

### Implications

```python
from torch.utils.data import DataLoader

ds = RandomStreamGenerator(total=10, seed=0)

# shuffle=True is IGNORED for IterableDataset
loader = DataLoader(ds, batch_size=4, shuffle=True)  # Warning: shuffle ineffective!

for batch in loader:
    print(batch.shape)  # Batches are in sequential order
```

### The Multi-Worker Problem

With `num_workers > 0`, each worker calls `__iter__` independently. Without coordination, **every worker generates the same data**:

```python
# ❌ WRONG: All workers produce identical data
loader = DataLoader(ds, batch_size=4, num_workers=2)

# This would see duplicated samples!
```

**Solution**: Implement worker-aware sharding (see next section).

---

## Worker-Aware Sharding

### Using `get_worker_info()`

PyTorch provides `torch.utils.data.get_worker_info()` to coordinate workers:

```python
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

class WorkerAwareStream(IterableDataset):
    """
    Iterable dataset with proper worker sharding.
    
    Each worker receives a disjoint subset of the data,
    preventing duplicates in multi-worker loading.
    """
    
    def __init__(self, n: int = 20, base_seed: int = 0):
        self.n = n
        self.base_seed = base_seed
    
    def __iter__(self):
        info = get_worker_info()
        
        if info is None:
            # Single-process: yield all items
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process: compute this worker's shard
            worker_id = info.id
            num_workers = info.num_workers
        
        # Per-worker RNG for reproducible but different randomness
        seed = self.base_seed + 1000 * worker_id
        g = torch.Generator().manual_seed(seed)
        
        # Shard by modulo: worker w gets items where i % num_workers == w
        for i in range(self.n):
            if i % num_workers == worker_id:
                # Generate sample (noisy version of index)
                sample = i + torch.randn(1, generator=g).item()
                yield torch.tensor([sample])
```

### Sharding Pattern

Given $N$ items and $W$ workers, worker $w$ processes items where:

$$
\text{item } i \text{ goes to worker } w \iff i \mod W = w
$$

This ensures:
- Each item is processed by exactly one worker
- Workers process disjoint subsets
- Load is balanced (approximately $N/W$ items per worker)

```python
# Example with 10 items and 3 workers:
# Worker 0: items 0, 3, 6, 9
# Worker 1: items 1, 4, 7
# Worker 2: items 2, 5, 8
```

---

## Shuffling Strategies for Streams

Since `DataLoader` can't shuffle `IterableDataset`, we implement shuffling manually.

### Shuffle Buffer

Maintains a fixed-size buffer and randomly samples from it:

```python
class ShuffleBuffer(IterableDataset):
    """
    Streaming shuffle using a fixed-size buffer.
    
    Algorithm:
    1. Fill buffer with first K items
    2. Randomly pick and yield one item
    3. Refill buffer slot with next source item
    4. Repeat until source exhausted, then drain buffer
    
    Trade-offs:
    - Larger K → better mixing, more memory
    - K=1 → no shuffle (identity)
    - K=N → perfect shuffle (but defeats streaming purpose)
    """
    
    def __init__(
        self, 
        source: IterableDataset, 
        buffer_size: int = 1024, 
        seed: int = 0
    ):
        self.source = source
        self.buffer_size = max(1, buffer_size)
        self.seed = seed
    
    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info else 0
        
        # Worker-specific RNG
        rng = torch.Generator().manual_seed(self.seed + 1337 * worker_id)
        
        # Shard source across workers
        source_iter = self._shard(iter(self.source))
        
        # Fill buffer (warm-up)
        buffer = []
        for _ in range(self.buffer_size):
            try:
                buffer.append(next(source_iter))
            except StopIteration:
                break
        
        # Streaming shuffle loop
        while buffer:
            # Random selection
            idx = int(torch.randint(len(buffer), (1,), generator=rng))
            yield buffer.pop(idx)
            
            # Refill
            try:
                buffer.append(next(source_iter))
            except StopIteration:
                pass  # Just keep draining
    
    def _shard(self, iterator):
        """Yield only this worker's items."""
        info = get_worker_info()
        if info is None:
            yield from iterator
        else:
            for i, item in enumerate(iterator):
                if i % info.num_workers == info.id:
                    yield item
```

### Block Shuffle

Read blocks of K items, permute within each block:

```python
class BlockShuffle(IterableDataset):
    """
    Block-based shuffling for streaming data.
    
    Algorithm:
    1. Read K items into a block
    2. Randomly permute the block
    3. Yield all K items
    4. Repeat for next block
    
    Properties:
    - O(K) memory
    - Inter-block order preserved (weaker mixing than shuffle buffer)
    - Good cache locality
    """
    
    def __init__(
        self, 
        source: IterableDataset, 
        block_size: int = 1024, 
        seed: int = 0
    ):
        self.source = source
        self.block_size = max(1, block_size)
        self.seed = seed
    
    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info else 0
        
        rng = torch.Generator().manual_seed(self.seed + 2024 * worker_id)
        source_iter = self._shard(iter(self.source))
        
        while True:
            # Fill block
            block = []
            for _ in range(self.block_size):
                try:
                    block.append(next(source_iter))
                except StopIteration:
                    break
            
            if not block:
                break  # Source exhausted
            
            # Permute and yield
            if len(block) > 1:
                perm = torch.randperm(len(block), generator=rng).tolist()
                for idx in perm:
                    yield block[idx]
            else:
                yield block[0]
    
    def _shard(self, iterator):
        """Yield only this worker's items."""
        info = get_worker_info()
        if info is None:
            yield from iterator
        else:
            for i, item in enumerate(iterator):
                if i % info.num_workers == info.id:
                    yield item
```

### Shuffle Strategy Comparison

| Strategy | Memory | Mixing Quality | Locality | Best For |
|----------|--------|---------------|----------|----------|
| **No Shuffle** | O(1) | None | Perfect | Ordered data |
| **Shuffle Buffer** | O(K) | Good | Poor | General streaming |
| **Block Shuffle** | O(K) | Moderate | Good | Cache-friendly |
| **Full Shuffle** | O(N) | Perfect | N/A | Small datasets |

---

## Complete Example: Worker-Aware Streaming

```python
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

class WorkerAwareShuffledStream(IterableDataset):
    """
    Production-ready iterable dataset with:
    - Worker-aware sharding
    - Local shuffle buffer
    - Reproducible randomness
    """
    
    def __init__(
        self, 
        n: int = 100, 
        buffer_size: int = 16, 
        base_seed: int = 0
    ):
        self.n = n
        self.buffer_size = buffer_size
        self.base_seed = base_seed
    
    def __iter__(self):
        info = get_worker_info()
        
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers
        
        # Per-worker deterministic RNG
        rng = torch.Generator().manual_seed(self.base_seed + 1000 * worker_id)
        
        # Local shuffle buffer
        buffer = []
        
        for i in range(self.n):
            # Sharding: only process items for this worker
            if i % num_workers != worker_id:
                continue
            
            # Generate sample
            sample = torch.randn(3, generator=rng)
            buffer.append(sample)
            
            # Yield when buffer full (local shuffle)
            if len(buffer) >= self.buffer_size:
                perm = torch.randperm(len(buffer), generator=rng).tolist()
                for idx in perm:
                    yield buffer[idx]
                buffer.clear()
        
        # Flush remaining
        if buffer:
            perm = torch.randperm(len(buffer), generator=rng).tolist()
            for idx in perm:
                yield buffer[idx]


def main():
    ds = WorkerAwareShuffledStream(n=20, buffer_size=4, base_seed=42)
    
    print("Single-process iteration:")
    count = 0
    for sample in ds:
        print(f"  {sample[:2].tolist()}...")
        count += 1
    print(f"  Total: {count} samples\n")
    
    print("Multi-worker DataLoader (2 workers):")
    loader = DataLoader(ds, batch_size=4, num_workers=2)
    total_samples = 0
    for batch in loader:
        print(f"  Batch shape: {batch.shape}")
        total_samples += batch.shape[0]
    print(f"  Total: {total_samples} samples")


if __name__ == "__main__":
    main()
```

**Output:**
```
Single-process iteration:
  [0.123, -0.456]...
  [0.789, 0.012]...
  ...
  Total: 20 samples

Multi-worker DataLoader (2 workers):
  Batch shape: torch.Size([4, 3])
  Batch shape: torch.Size([4, 3])
  ...
  Total: 20 samples
```

---

## Epoch Management

For varying data across epochs, incorporate epoch information:

```python
class EpochAwareStream(IterableDataset):
    """Vary data based on epoch number."""
    
    def __init__(self, n: int = 100, base_seed: int = 0):
        self.n = n
        self.base_seed = base_seed
        self.epoch = 0  # Updated externally
    
    def set_epoch(self, epoch: int):
        """Call before each epoch to vary data."""
        self.epoch = epoch
    
    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info else 0
        
        # Epoch + worker seed for per-epoch variation
        seed = self.base_seed + 1000 * self.epoch + worker_id
        rng = torch.Generator().manual_seed(seed)
        
        for i in range(self.n):
            if info is None or i % info.num_workers == worker_id:
                yield torch.randn(3, generator=rng)


# Usage in training loop
ds = EpochAwareStream(n=100)
loader = DataLoader(ds, batch_size=32, num_workers=2)

for epoch in range(10):
    ds.set_epoch(epoch)  # Different data each epoch
    for batch in loader:
        # train...
        pass
```

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **IterableDataset** | Implements `__iter__()` for sequential access |
| **No Shuffling** | DataLoader ignores `shuffle=True` for iterables |
| **Worker Sharding** | Use `get_worker_info()` to partition data |
| **Shuffle Buffer** | Good mixing with O(K) memory |
| **Block Shuffle** | Cache-friendly with moderate mixing |
| **Epoch Variation** | Vary seed per epoch for different data |

---

## Further Reading

- Section 1.12.1: Map-Style Datasets (for comparison)
- Section 1.13: DataLoaders (advanced configurations)
- [PyTorch IterableDataset Docs](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)
