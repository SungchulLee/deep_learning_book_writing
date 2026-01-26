# Performance Profiling and Debugging

## Learning Objectives

By the end of this section, you will be able to:

- Measure data loading overhead systematically
- Identify common bottlenecks in the data pipeline
- Debug DataLoader issues effectively
- Implement a complete optimization workflow

## Measuring Data Loading Overhead

The fundamental question: **Is data loading slowing down my training?**

### The Loading vs Computation Ratio

For optimal GPU utilization:

$$
\text{Target: } \frac{\text{Data Loading Time}}{\text{Total Time}} < 0.3
$$

If data loading exceeds 30% of total time, you have a bottleneck.

### Measuring Overhead

```python
import time
import torch
from torch.utils.data import DataLoader

def measure_loading_overhead(loader, num_batches=50, compute_time=0.01):
    """
    Measure the ratio of loading time to total time.
    
    Args:
        loader: DataLoader to profile
        num_batches: Number of batches to measure
        compute_time: Simulated GPU computation time per batch
    
    Returns:
        Dict with timing statistics
    """
    load_times = []
    compute_times = []
    
    iterator = iter(loader)
    
    for i in range(num_batches):
        # Measure loading time
        load_start = time.time()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        load_time = time.time() - load_start
        load_times.append(load_time)
        
        # Simulate computation
        compute_start = time.time()
        time.sleep(compute_time)  # Replace with actual forward/backward
        compute_times.append(time.time() - compute_start)
    
    total_load = sum(load_times)
    total_compute = sum(compute_times)
    total_time = total_load + total_compute
    
    return {
        'load_time': total_load,
        'compute_time': total_compute,
        'total_time': total_time,
        'load_ratio': total_load / total_time,
        'avg_load': total_load / len(load_times),
        'avg_compute': total_compute / len(compute_times),
    }


# Usage
results = measure_loading_overhead(train_loader)
print(f"Loading overhead: {results['load_ratio']:.1%}")

if results['load_ratio'] > 0.3:
    print("⚠️  Data loading is a bottleneck!")
else:
    print("✓ Data loading is not a bottleneck")
```

## DataLoader Profiler

A reusable profiler for continuous monitoring:

```python
class DataLoaderProfiler:
    """
    Profile DataLoader performance across epochs.
    """
    
    def __init__(self):
        self.batch_times = []
        self.epoch_stats = []
        self._start_time = None
    
    def start_epoch(self):
        """Call at the start of each epoch."""
        self.batch_times = []
        self._start_time = time.time()
    
    def record_batch(self):
        """Call after each batch is loaded."""
        if self._start_time is not None:
            self.batch_times.append(time.time() - self._start_time)
        self._start_time = time.time()
    
    def end_epoch(self):
        """Call at the end of each epoch. Returns stats."""
        if not self.batch_times:
            return None
        
        times = torch.tensor(self.batch_times)
        stats = {
            'num_batches': len(self.batch_times),
            'total_time': times.sum().item(),
            'mean_time': times.mean().item(),
            'std_time': times.std().item(),
            'min_time': times.min().item(),
            'max_time': times.max().item(),
            'throughput': len(self.batch_times) / times.sum().item(),
        }
        self.epoch_stats.append(stats)
        return stats
    
    def report(self):
        """Print a summary report."""
        if not self.epoch_stats:
            print("No data collected")
            return
        
        print("\n" + "="*60)
        print("DataLoader Performance Report")
        print("="*60)
        
        for i, stats in enumerate(self.epoch_stats):
            print(f"\nEpoch {i+1}:")
            print(f"  Batches:    {stats['num_batches']}")
            print(f"  Total:      {stats['total_time']:.2f}s")
            print(f"  Mean/batch: {stats['mean_time']*1000:.1f}ms")
            print(f"  Std:        {stats['std_time']*1000:.1f}ms")
            print(f"  Throughput: {stats['throughput']:.1f} batches/sec")


# Usage in training loop
profiler = DataLoaderProfiler()

for epoch in range(num_epochs):
    profiler.start_epoch()
    
    for batch in train_loader:
        profiler.record_batch()
        
        # Training step
        outputs = model(batch)
        loss.backward()
        optimizer.step()
    
    stats = profiler.end_epoch()
    print(f"Epoch {epoch}: {stats['throughput']:.1f} batches/sec")

profiler.report()
```

## Identifying Bottlenecks

### Bottleneck Detection Checklist

```python
def diagnose_dataloader(dataset, batch_size=32):
    """
    Systematic bottleneck diagnosis.
    """
    print("="*60)
    print("DataLoader Bottleneck Diagnosis")
    print("="*60)
    
    # Test 1: Single sample loading
    print("\n1. Single Sample Loading Time")
    times = []
    for i in range(10):
        start = time.time()
        _ = dataset[i]
        times.append(time.time() - start)
    avg_sample = sum(times) / len(times)
    print(f"   Average: {avg_sample*1000:.1f}ms per sample")
    
    if avg_sample > 0.01:  # 10ms
        print("   ⚠️  __getitem__ is slow - optimize preprocessing")
    
    # Test 2: num_workers scaling
    print("\n2. Worker Scaling Test")
    worker_counts = [0, 2, 4, 8]
    
    for nw in worker_counts:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=nw
        )
        
        start = time.time()
        for i, _ in enumerate(loader):
            if i >= 20:
                break
        elapsed = time.time() - start
        
        throughput = 20 / elapsed
        print(f"   num_workers={nw}: {throughput:.1f} batches/sec")
    
    # Test 3: Batch size impact
    print("\n3. Batch Size Impact")
    batch_sizes = [8, 16, 32, 64]
    
    for bs in batch_sizes:
        loader = DataLoader(
            dataset, 
            batch_size=bs, 
            num_workers=4
        )
        
        start = time.time()
        for i, _ in enumerate(loader):
            if i >= 20:
                break
        elapsed = time.time() - start
        
        samples_per_sec = (20 * bs) / elapsed
        print(f"   batch_size={bs}: {samples_per_sec:.0f} samples/sec")
    
    print("\n" + "="*60)
```

### Common Bottleneck Patterns

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| GPU utilization < 50% | Data loading too slow | Increase num_workers |
| More workers doesn't help | CPU-bound preprocessing | Optimize __getitem__ |
| Memory keeps growing | Memory leak | Use persistent_workers=False |
| First batch very slow | Worker spawn overhead | Use persistent_workers=True |
| Inconsistent batch times | I/O variability | Use SSD, prefetch more |

## Debugging Common Issues

### Issue 1: Workers Timing Out

```python
# Symptom: RuntimeError: DataLoader timed out

# Cause: Workers are hanging or too slow

# Solution 1: Increase timeout
loader = DataLoader(dataset, num_workers=4, timeout=120)

# Solution 2: Debug with single process
loader = DataLoader(dataset, num_workers=0)

# Solution 3: Add timing to __getitem__
class DebugDataset(Dataset):
    def __getitem__(self, idx):
        start = time.time()
        # ... load data ...
        elapsed = time.time() - start
        if elapsed > 1.0:
            print(f"WARNING: Sample {idx} took {elapsed:.1f}s")
        return data
```

### Issue 2: CUDA Out of Memory

```python
# Symptom: RuntimeError: CUDA out of memory

# Cause 1: Batch size too large
loader = DataLoader(dataset, batch_size=8)  # Reduce from 32

# Cause 2: Data not moved to GPU correctly
for batch in loader:
    x = batch[0].to(device)  # Ensure old tensors are freed
    # Don't keep references to old batches

# Cause 3: Memory accumulation
import gc
gc.collect()
torch.cuda.empty_cache()
```

### Issue 3: Memory Leaks

```python
# Detection
import tracemalloc

tracemalloc.start()

for i, batch in enumerate(loader):
    if i % 100 == 0:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Batch {i}: Current={current/1e6:.1f}MB, Peak={peak/1e6:.1f}MB")
    
    if i >= 500:
        break

tracemalloc.stop()

# Common fixes:
# 1. Don't store batches in lists
# 2. Close file handles in __getitem__
# 3. Use persistent_workers=False to reset worker state
```

### Issue 4: Slow First Epoch

```python
# This is often normal due to:
# - Worker process spawning
# - Disk cache warmup
# - JIT compilation

# Solutions:
# 1. Use persistent_workers=True
loader = DataLoader(dataset, num_workers=4, persistent_workers=True)

# 2. Warmup before timing
iterator = iter(loader)
for _ in range(3):
    _ = next(iterator)

# Now start actual training/timing
```

## Optimization Workflow

A systematic approach to optimizing DataLoader performance:

### Step 1: Baseline Measurement

```python
# Start with simplest configuration
baseline_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=0,
    shuffle=True
)

baseline_throughput = measure_throughput(baseline_loader)
print(f"Baseline: {baseline_throughput:.1f} batches/sec")
```

### Step 2: Add Workers

```python
# Test different worker counts
for nw in [2, 4, 8]:
    loader = DataLoader(dataset, batch_size=32, num_workers=nw)
    throughput = measure_throughput(loader)
    speedup = throughput / baseline_throughput
    print(f"num_workers={nw}: {speedup:.1f}x speedup")
```

### Step 3: Enable Optimizations

```python
# Apply performance optimizations
optimized_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,        # Best from step 2
    pin_memory=True,      # Fast GPU transfer
    persistent_workers=True,  # Avoid spawn overhead
    prefetch_factor=2,    # Keep GPU fed
)

optimized_throughput = measure_throughput(optimized_loader)
total_speedup = optimized_throughput / baseline_throughput
print(f"Total speedup: {total_speedup:.1f}x")
```

### Step 4: Profile __getitem__

If throughput is still low:

```python
import cProfile
import pstats

# Profile dataset access
profiler = cProfile.Profile()
profiler.enable()

for i in range(100):
    _ = dataset[i]

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(10)  # Top 10 time consumers
```

### Step 5: Monitor in Production

```python
class MonitoredDataLoader:
    """Wrapper for production monitoring."""
    
    def __init__(self, loader, log_interval=100):
        self.loader = loader
        self.log_interval = log_interval
        self.batch_count = 0
        self.total_time = 0
    
    def __iter__(self):
        start = time.time()
        
        for batch in self.loader:
            load_time = time.time() - start
            self.total_time += load_time
            self.batch_count += 1
            
            if self.batch_count % self.log_interval == 0:
                avg = self.total_time / self.batch_count
                print(f"[DataLoader] Batch {self.batch_count}: "
                      f"avg load time = {avg*1000:.1f}ms")
            
            yield batch
            start = time.time()
```

## Summary

| Metric | Target | Action if Exceeded |
|--------|--------|-------------------|
| Load ratio | < 30% | Increase workers, optimize __getitem__ |
| GPU utilization | > 80% | Data loading is not bottleneck |
| Batch time variance | Low | Check I/O, use SSD |
| Memory growth | Stable | Fix leaks, reduce prefetch |

## Debugging Decision Tree

```
Is GPU utilization low?
├── Yes → Is data loading time high?
│   ├── Yes → Increase num_workers
│   │   └── Still slow? → Optimize __getitem__
│   └── No → Check model/optimizer
└── No → System is balanced ✓

Are workers timing out?
├── Yes → Reduce num_workers
│   └── Still timing out? → Debug __getitem__ with num_workers=0
└── No → Continue

Is memory growing?
├── Yes → Set persistent_workers=False
│   └── Still growing? → Check __getitem__ for leaks
└── No → Continue
```

## Practice Exercises

1. **Full Diagnosis**: Run the `diagnose_dataloader` function on your dataset and interpret the results.

2. **Optimization Challenge**: Start with `num_workers=0` and systematically apply optimizations to achieve maximum throughput. Document each step's improvement.

3. **Memory Profiling**: Use `tracemalloc` to monitor memory usage over 1000 batches with different configurations.

## What's Next

The next section covers **Distributed Data Loading**, showing how to scale data loading across multiple GPUs using DistributedSampler.
