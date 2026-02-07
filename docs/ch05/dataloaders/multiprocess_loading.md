# Multi-Process Loading

## Overview

Data loading—reading from disk, decoding images, applying transforms—can bottleneck training if performed synchronously. Multi-process loading uses `num_workers` subprocesses to prepare batches in parallel, keeping the GPU fed with data.

## Enabling Multi-Process Loading

```python
loader = DataLoader(dataset, batch_size=64, num_workers=4)
```

With `num_workers=0` (default), data loading happens in the main process. With `num_workers > 0`, the `DataLoader` spawns worker processes that prefetch batches in parallel.

## How It Works

Each worker independently loads and preprocesses samples:

1. The main process sends batch indices to workers via a shared queue.
2. Workers call `dataset[idx]` for their assigned indices.
3. Workers send completed batches back to the main process via another queue.
4. The main process consumes batches and feeds them to the GPU.

The `prefetch_factor` parameter (default 2) controls how many batches each worker prepares in advance:

```python
loader = DataLoader(dataset, batch_size=64, num_workers=4,
                    prefetch_factor=2)  # 4 workers × 2 = 8 batches prefetched
```

## Choosing `num_workers`

A common heuristic: set `num_workers` to the number of CPU cores divided by the number of GPUs. Start with 4 and adjust based on profiling:

```python
import os
num_cpus = os.cpu_count()
num_workers = min(num_cpus, 8)  # Cap to avoid excessive process overhead
```

Too few workers → GPU starves for data. Too many workers → excessive memory usage and process management overhead. Profile both data loading time and GPU utilization to find the optimum.

## Persistent Workers

By default, workers are respawned at the start of each epoch. With `persistent_workers=True`, workers stay alive, avoiding the overhead of process creation and dataset re-initialization:

```python
loader = DataLoader(dataset, batch_size=64, num_workers=4,
                    persistent_workers=True)
```

This is particularly beneficial when `__init__` is expensive (e.g., opening database connections or memory-mapping large files).

## Worker Initialization

Each worker process gets a copy of the dataset. For datasets using random number generators, each worker must have a unique seed to avoid producing identical augmentations:

```python
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(worker_id)

loader = DataLoader(dataset, batch_size=64, num_workers=4,
                    worker_init_fn=worker_init_fn)
```

## Common Pitfalls

**Shared state**: Workers are separate processes. In-memory mutations to the dataset in one worker are not visible to others. Use shared memory (`multiprocessing.Array`) or memory-mapped files if workers must share state.

**File descriptor limits**: Each worker may open file handles. With many workers and lazy-loading datasets, the process can hit OS file descriptor limits.

**Debugging**: Errors inside workers can produce cryptic stack traces. Debug with `num_workers=0` first, then enable multi-processing.

## Key Takeaways

- Multi-process loading overlaps data preparation with GPU computation.
- Start with `num_workers=4` and tune based on GPU utilization.
- Use `persistent_workers=True` to amortize worker startup costs across epochs.
- Debug with `num_workers=0`, then enable parallelism.
