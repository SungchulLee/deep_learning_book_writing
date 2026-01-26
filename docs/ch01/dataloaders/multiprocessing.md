# Multiprocessing

## Learning Objectives

By the end of this section, you will be able to:

- Understand the multi-process data loading architecture
- Configure `num_workers` optimally for your hardware
- Use `persistent_workers` and `prefetch_factor` effectively
- Manage randomness correctly across worker processes
- Debug common multiprocessing issues

## Why Multiprocessing?

Data loading often involves expensive operations such as disk I/O, decompression, preprocessing, and feature extraction. Without multiprocessing, these operations block the main process, leaving the GPU idle while waiting for data.

### Single-Process vs Multi-Process

**Without multiprocessing:**
```
Time → [Load batch 1] [Train batch 1] [Load batch 2] [Train batch 2]
GPU  →      IDLE         COMPUTE          IDLE         COMPUTE
```

**With multiprocessing:**
```
Workers →  [Load 1][Load 2][Load 3][Load 4][Load 5]...
Main    →  [Setup] [Train 1][Train 2][Train 3][Train 4]...
GPU     →   IDLE   [COMPUTE][COMPUTE][COMPUTE][COMPUTE]...
```

Data loading happens in parallel with training, maximizing GPU utilization.

## Worker Architecture

When `num_workers > 0`, DataLoader spawns separate worker processes:

```
Main Process
    │
    ├── Worker 0 → Loads batches 0, 4, 8, 12, ...
    ├── Worker 1 → Loads batches 1, 5, 9, 13, ...
    ├── Worker 2 → Loads batches 2, 6, 10, 14, ...
    └── Worker 3 → Loads batches 3, 7, 11, 15, ...
```

Workers load in round-robin fashion; the main process consumes batches in order.

### Accessing Worker Information

Inside `__getitem__`, you can access worker details:

```python
def __getitem__(self, idx):
    worker_info = torch.utils.data.get_worker_info()
    
    if worker_info is None:
        # Single-process loading (num_workers=0)
        worker_id = 0
    else:
        # Multi-process loading
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
    
    return self.data[idx]
```

## Configuring num_workers

### Basic Usage

```python
# Single-process (default)
loader = DataLoader(dataset, num_workers=0)

# Multi-process
loader = DataLoader(dataset, num_workers=4)
```

### Performance Comparison

| num_workers | Relative Speed | Notes |
|-------------|---------------|-------|
| 0 | 1.0x (baseline) | Single process, blocking |
| 2 | 2-3x | Good for most datasets |
| 4 | 3-4x | Sweet spot for many setups |
| 8 | 3-5x | Diminishing returns |
| 16+ | Varies | May decrease due to overhead |

### Selection Guidelines

```python
import os
import torch

def get_recommended_workers():
    cpu_count = os.cpu_count()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    return min(4 * gpu_count, cpu_count, 8)

# Scenario-specific recommendations
CONFIGS = {
    'debugging': 0,
    'light_preprocessing': 4,
    'heavy_preprocessing': 8,
    'memory_limited': 2,
}
```

## persistent_workers

Keeps worker processes alive between epochs instead of respawning.

```python
loader = DataLoader(
    dataset,
    num_workers=4,
    persistent_workers=True  # Requires num_workers > 0
)
```

| Aspect | False (default) | True |
|--------|----------------|------|
| Memory | Lower | Higher |
| Startup | Slow per epoch | Fast (one-time) |
| Use case | Debugging | Production training |

## prefetch_factor

Controls how many batches each worker prefetches:

```python
loader = DataLoader(
    dataset,
    num_workers=4,
    prefetch_factor=2  # 4 workers × 2 = 8 batches ahead
)
```

Increase if GPU shows stalls; decrease if running out of memory.

## Managing Worker Randomness

### The Problem

Without intervention, all workers share the same random seed:

```python
# WRONG - same random values across workers
class AugmentedDataset(Dataset):
    def __getitem__(self, idx):
        noise = torch.rand(self.data[idx].shape)  # Identical across workers!
        return self.data[idx] + noise
```

### The Solution

```python
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed % (2**32 - 1) + worker_id
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

loader = DataLoader(
    dataset,
    num_workers=4,
    worker_init_fn=worker_init_fn
)
```

## Production Configuration

```python
def create_train_loader(dataset, batch_size=32, seed=42):
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % (2**32 - 1) + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed)
    )
```

## Common Issues

### Workers Hang
- Use `timeout=60` to detect
- Reduce `num_workers`
- Debug with `num_workers=0`

### Memory Growing
- Set `persistent_workers=False`
- Reduce `prefetch_factor`
- Check for leaks in `__getitem__`

### Same Augmentation Across Workers
- Always use `worker_init_fn`

### Slow First Batch
- Use `persistent_workers=True`
- This is normal spawn overhead

## Performance Profiling

```python
def compare_configs(dataset, num_batches=100):
    configs = [
        ('baseline', {'num_workers': 0}),
        ('4_workers', {'num_workers': 4}),
        ('4_persistent', {'num_workers': 4, 'persistent_workers': True}),
    ]
    
    for name, config in configs:
        loader = DataLoader(dataset, batch_size=32, **config)
        
        start = time.time()
        for i, _ in enumerate(loader):
            if i >= num_batches:
                break
        elapsed = time.time() - start
        
        print(f"{name}: {num_batches/elapsed:.1f} batches/sec")
```

## Summary

| Parameter | Default | Training | Debugging |
|-----------|---------|----------|-----------|
| num_workers | 0 | 4-8 | 0 |
| persistent_workers | False | True | False |
| prefetch_factor | 2 | 2-4 | N/A |
| worker_init_fn | None | Required | N/A |

## Practice Exercises

1. Measure throughput with different `num_workers` values and find the point of diminishing returns.

2. Verify that `worker_init_fn` creates different random sequences across workers.

3. Monitor memory usage with different `prefetch_factor` values.

## What's Next

The next section covers **Performance Profiling and Debugging**, providing tools for identifying bottlenecks and optimizing your data loading pipeline.
