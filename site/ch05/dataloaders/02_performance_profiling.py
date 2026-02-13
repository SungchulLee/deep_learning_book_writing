#!/usr/bin/env python3
"""
=============================================================================
DataLoader Performance Profiling and Debugging
=============================================================================

OVERVIEW:
---------
This tutorial shows how to:
  • Identify data loading bottlenecks
  • Profile DataLoader performance
  • Debug common issues
  • Optimize for maximum throughput

COMMON ISSUES:
--------------
❌ GPU waiting for data (low GPU utilization)
❌ Worker processes timing out or hanging
❌ Memory leaks in multi-process loading
❌ Slow data loading pipeline
❌ Inconsistent batch timing

LEARNING OBJECTIVES:
-------------------
✓ Measure data loading vs computation time
✓ Identify and fix bottlenecks
✓ Debug worker process issues
✓ Optimize end-to-end training pipeline

DIFFICULTY: ⭐⭐⭐ Advanced
TIME: 20 minutes
"""

import torch
from torch.utils.data import Dataset, DataLoader
import time
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Benchmark Dataset
# =============================================================================
class BenchmarkDataset(Dataset):
    """
    Dataset for benchmarking with configurable loading time.
    """
    
    def __init__(self, num_samples=100, load_time=0.01, data_size=1000):
        """
        Args:
            num_samples: Number of samples
            load_time: Simulated loading time (seconds)
            data_size: Size of data tensor
        """
        self.num_samples = num_samples
        self.load_time = load_time
        self.data_size = data_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Simulate data loading time
        time.sleep(self.load_time)
        
        # Return realistic-sized tensors
        x = torch.randn(self.data_size)
        y = torch.tensor(idx % 10)
        
        return x, y


# =============================================================================
# DEMO 1: Measuring Data Loading Overhead
# =============================================================================
def demo_measure_overhead():
    """
    Measure the ratio of data loading time vs computation time.
    
    Ideal: Data loading << Computation
    Problem: Data loading >= Computation → GPU idle
    """
    print("\n" + "="*60)
    print("DEMO 1: Measuring Data Loading Overhead")
    print("="*60)
    
    dataset = BenchmarkDataset(num_samples=50, load_time=0.02)
    
    # Single-worker loader
    loader_slow = DataLoader(dataset, batch_size=10, num_workers=0)
    
    # Multi-worker loader
    loader_fast = DataLoader(dataset, batch_size=10, num_workers=4)
    
    def simulate_training(loader, desc):
        """Simulate training with timing."""
        load_times = []
        compute_times = []
        
        for batch_idx, (x, y) in enumerate(loader):
            # Measure data loading time (already loaded by this point)
            # So we'll use iterator to measure properly
            pass
        
        # Proper timing with explicit iterator
        iterator = iter(loader)
        
        print(f"\n{desc}:")
        total_start = time.time()
        
        for batch_idx in range(5):  # First 5 batches
            # Measure data loading
            load_start = time.time()
            batch = next(iterator)
            load_time = time.time() - load_start
            load_times.append(load_time)
            
            # Simulate computation
            compute_start = time.time()
            time.sleep(0.01)  # Simulate GPU forward/backward
            compute_time = time.time() - compute_start
            compute_times.append(compute_time)
            
            total_batch = load_time + compute_time
            print(f"  Batch {batch_idx+1}:")
            print(f"    Load: {load_time:.3f}s ({load_time/total_batch*100:.1f}%)")
            print(f"    Compute: {compute_time:.3f}s ({compute_time/total_batch*100:.1f}%)")
        
        total_time = time.time() - total_start
        avg_load = sum(load_times) / len(load_times)
        avg_compute = sum(compute_times) / len(compute_times)
        
        print(f"\n  Summary:")
        print(f"    Total time: {total_time:.3f}s")
        print(f"    Avg load time: {avg_load:.3f}s")
        print(f"    Avg compute time: {avg_compute:.3f}s")
        print(f"    Load overhead: {avg_load/(avg_load+avg_compute)*100:.1f}%")
    
    simulate_training(loader_slow, "🐌 Single Worker (Slow)")
    simulate_training(loader_fast, "⚡ Multi-Worker (Fast)")
    
    print("\n💡 Goal:")
    print("   Keep data loading time << computation time")
    print("   If loading > 30% → Add more workers!")


# =============================================================================
# DEMO 2: Profiling DataLoader Throughput
# =============================================================================
class DataLoaderProfiler:
    """
    Profile DataLoader throughput and timing.
    """
    
    def __init__(self):
        self.batch_times = []
        self.start_time = None
    
    def start(self):
        """Start profiling."""
        self.batch_times = []
        self.start_time = time.time()
    
    def record_batch(self):
        """Record timing for one batch."""
        if self.start_time is not None:
            self.batch_times.append(time.time() - self.start_time)
            self.start_time = time.time()
    
    def report(self, name="DataLoader"):
        """Print profiling report."""
        if not self.batch_times:
            print(f"No data recorded for {name}")
            return
        
        total_time = sum(self.batch_times)
        avg_time = total_time / len(self.batch_times)
        min_time = min(self.batch_times)
        max_time = max(self.batch_times)
        throughput = len(self.batch_times) / total_time
        
        print(f"\n📊 {name} Profile:")
        print(f"  Batches: {len(self.batch_times)}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg time/batch: {avg_time:.3f}s")
        print(f"  Min time: {min_time:.3f}s")
        print(f"  Max time: {max_time:.3f}s")
        print(f"  Throughput: {throughput:.2f} batches/sec")
        print(f"  Std dev: {torch.tensor(self.batch_times).std().item():.3f}s")


def demo_profiler():
    """
    Demonstrate DataLoader profiling.
    """
    print("\n" + "="*60)
    print("DEMO 2: Profiling DataLoader Throughput")
    print("="*60)
    
    dataset = BenchmarkDataset(num_samples=50, load_time=0.01)
    
    configs = [
        {"num_workers": 0, "desc": "Single process"},
        {"num_workers": 2, "desc": "2 workers"},
        {"num_workers": 4, "desc": "4 workers"},
    ]
    
    for cfg in configs:
        loader = DataLoader(
            dataset,
            batch_size=10,
            num_workers=cfg["num_workers"]
        )
        
        profiler = DataLoaderProfiler()
        profiler.start()
        
        for _ in loader:
            profiler.record_batch()
        
        profiler.report(f"{cfg['desc']}")
    
    print("\n💡 Analysis:")
    print("   • Look for consistent batch times (low std dev)")
    print("   • Compare throughput across configurations")
    print("   • Higher throughput = better configuration")


# =============================================================================
# DEMO 3: Identifying Bottlenecks
# =============================================================================
def demo_bottleneck_analysis():
    """
    Systematic approach to finding bottlenecks.
    """
    print("\n" + "="*60)
    print("DEMO 3: Bottleneck Analysis")
    print("="*60)
    
    print("\n🔍 Bottleneck Detection Checklist:\n")
    
    # Test 1: GPU utilization
    print("1. CHECK GPU UTILIZATION")
    print("   Run: nvidia-smi dmon -s mu")
    print("   Low GPU usage? → Data loading is bottleneck\n")
    
    # Test 2: Worker count
    print("2. TEST DIFFERENT num_workers")
    dataset = BenchmarkDataset(num_samples=40, load_time=0.02)
    
    for nw in [0, 2, 4]:
        loader = DataLoader(dataset, batch_size=10, num_workers=nw)
        start = time.time()
        for _ in loader:
            pass
        elapsed = time.time() - start
        print(f"   num_workers={nw}: {elapsed:.3f}s")
    
    print("\n   If time decreases with more workers → Add workers")
    print("   If time same/increases → CPU bottleneck\n")
    
    # Test 3: Batch size
    print("3. TEST DIFFERENT batch_size")
    for bs in [5, 10, 20]:
        loader = DataLoader(dataset, batch_size=bs, num_workers=2)
        start = time.time()
        for _ in loader:
            pass
        elapsed = time.time() - start
        batches = len(dataset) // bs
        time_per_batch = elapsed / batches
        print(f"   batch_size={bs}: {time_per_batch:.3f}s per batch")
    
    print("\n   Larger batches usually faster (amortized overhead)")
    
    # Test 4: Data complexity
    print("\n4. PROFILE __getitem__ COMPLEXITY")
    print("   Add timing inside __getitem__:")
    print("   ```python")
    print("   start = time.time()")
    print("   # ... data loading ...")
    print("   print(f'Load time: {time.time()-start:.3f}s')")
    print("   ```")
    
    print("\n💡 Common Bottlenecks:")
    print("   • Too few workers → Add num_workers")
    print("   • Expensive __getitem__ → Optimize preprocessing")
    print("   • I/O bound → Use SSD, cache data in RAM")
    print("   • Small batch size → Increase batch_size")


# =============================================================================
# DEMO 4: Debugging Common Issues
# =============================================================================
def demo_debug_issues():
    """
    Common DataLoader issues and how to debug them.
    """
    print("\n" + "="*60)
    print("DEMO 4: Debugging Common Issues")
    print("="*60)
    
    print("\n🐛 ISSUE 1: Workers Timing Out")
    print("   Symptom: RuntimeError: DataLoader timed out")
    print("   Cause: Worker processes hang or are too slow")
    print("   Solutions:")
    print("     • Increase timeout: DataLoader(..., timeout=60)")
    print("     • Reduce num_workers")
    print("     • Check for blocking operations in __getitem__")
    print("     • Avoid large global variables (copied to each worker)")
    
    print("\n🐛 ISSUE 2: Memory Leaks")
    print("   Symptom: Memory usage keeps increasing")
    print("   Cause: References not released in workers")
    print("   Solutions:")
    print("     • Use persistent_workers=False during debugging")
    print("     • Check for circular references in Dataset")
    print("     • Don't store large objects in Dataset instance")
    print("     • Close file handles in __getitem__")
    
    print("\n🐛 ISSUE 3: Slow First Batch")
    print("   Symptom: First batch takes much longer")
    print("   Cause: Worker spawning overhead")
    print("   Solutions:")
    print("     • Use persistent_workers=True")
    print("     • Accept first batch warmup (normal)")
    print("     • Profile excluding first batch")
    
    print("\n🐛 ISSUE 4: Inconsistent Randomness")
    print("   Symptom: Same augmentations across workers")
    print("   Cause: Workers share same random seed")
    print("   Solution:")
    print("     def worker_init_fn(worker_id):")
    print("         np.random.seed(np.random.get_state()[1][0] + worker_id)")
    
    print("\n🐛 ISSUE 5: CUDA Out of Memory")
    print("   Symptom: RuntimeError: CUDA out of memory")
    print("   Cause: Batch too large for GPU")
    print("   Solutions:")
    print("     • Reduce batch_size")
    print("     • Use gradient accumulation")
    print("     • Enable mixed precision (fp16)")
    print("     • Check for memory leaks in model")


# =============================================================================
# DEMO 5: Optimization Workflow
# =============================================================================
def demo_optimization_workflow():
    """
    Step-by-step workflow for optimizing DataLoader.
    """
    print("\n" + "="*60)
    print("DEMO 5: Optimization Workflow")
    print("="*60)
    
    print("\n📋 Optimization Workflow:\n")
    
    print("STEP 1: Baseline Measurement")
    print("  • Start with num_workers=0")
    print("  • Measure time per epoch")
    print("  • Record GPU utilization\n")
    
    dataset = BenchmarkDataset(num_samples=40, load_time=0.02)
    loader_baseline = DataLoader(dataset, batch_size=10, num_workers=0)
    
    start = time.time()
    for _ in loader_baseline:
        pass
    baseline_time = time.time() - start
    print(f"  Baseline time: {baseline_time:.3f}s\n")
    
    print("STEP 2: Add Workers")
    print("  • Test num_workers=2,4,8")
    print("  • Find best value\n")
    
    best_workers = 0
    best_time = baseline_time
    
    for nw in [2, 4]:
        loader = DataLoader(dataset, batch_size=10, num_workers=nw)
        start = time.time()
        for _ in loader:
            pass
        elapsed = time.time() - start
        
        speedup = baseline_time / elapsed
        print(f"  num_workers={nw}: {elapsed:.3f}s ({speedup:.2f}x speedup)")
        
        if elapsed < best_time:
            best_time = elapsed
            best_workers = nw
    
    print(f"\n  Best: num_workers={best_workers}\n")
    
    print("STEP 3: Enable Optimizations")
    print("  • pin_memory=True (if using GPU)")
    print("  • persistent_workers=True")
    print("  • Tune prefetch_factor\n")
    
    loader_optimized = DataLoader(
        dataset,
        batch_size=10,
        num_workers=best_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Run a few epochs
    start = time.time()
    for epoch in range(2):
        for _ in loader_optimized:
            pass
    optimized_time = (time.time() - start) / 2  # Average per epoch
    
    total_speedup = baseline_time / optimized_time
    print(f"  Optimized time: {optimized_time:.3f}s")
    print(f"  Total speedup: {total_speedup:.2f}x\n")
    
    print("STEP 4: Profile and Verify")
    print("  • Check GPU utilization increased")
    print("  • Verify no memory leaks")
    print("  • Monitor worker CPU usage\n")
    
    print("STEP 5: Document Configuration")
    print("  • Save optimal parameters")
    print("  • Add comments explaining choices")
    print("  • Include hardware specs (for reproducibility)")
    
    print("\n✅ Final Configuration:")
    print(f"  DataLoader(")
    print(f"    dataset,")
    print(f"    batch_size=10,")
    print(f"    num_workers={best_workers},")
    print(f"    pin_memory=True,")
    print(f"    persistent_workers=True,")
    print(f"    prefetch_factor=2,")
    print(f"  )")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATALOADER PERFORMANCE & DEBUGGING")
    print("="*60)
    
    demo_measure_overhead()
    demo_profiler()
    demo_bottleneck_analysis()
    demo_debug_issues()
    demo_optimization_workflow()
    
    print("\n" + "="*60)
    print("✓ TUTORIAL COMPLETE!")
    print("="*60)
    print("\n📚 Key Takeaways:")
    print("  1. Profile before optimizing - measure everything!")
    print("  2. Data loading should be < 30% of total time")
    print("  3. Increase num_workers until no further speedup")
    print("  4. Use persistent_workers and pin_memory for production")
    print("  5. Debug systematically using the workflow")
    print("\n🎯 Next Steps:")
    print("  → Profile your actual data pipeline")
    print("  → Implement the optimization workflow")
    print("  → Monitor GPU utilization continuously")
    print("  → Document your optimal configuration")
