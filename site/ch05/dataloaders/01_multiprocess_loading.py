#!/usr/bin/env python3
"""
=============================================================================
Multi-Process Data Loading: Workers and Performance
=============================================================================

OVERVIEW:
---------
Multi-process data loading is PyTorch's way to parallelize data preprocessing.
Instead of loading data in the main process (blocking GPU), worker processes
load and preprocess data in parallel.

KEY CONCEPTS:
-------------
• num_workers=0: Single process (simple, good for debugging)
• num_workers>0: Multiple worker processes (fast, complex)
• persistent_workers: Keep workers alive between epochs
• prefetch_factor: How many batches each worker pre-loads
• worker_init_fn: Initialize each worker (set seeds, open files)

LEARNING OBJECTIVES:
-------------------
✓ Understand multi-process data loading architecture
✓ Choose optimal num_workers value
✓ Use persistent_workers for efficiency
✓ Manage randomness across workers
✓ Avoid common pitfalls and deadlocks

DIFFICULTY: ⭐⭐⭐ Advanced
TIME: 25 minutes
"""

import torch
from torch.utils.data import Dataset, DataLoader
import time
import random
import numpy as np


# =============================================================================
# Expensive Dataset (simulates real preprocessing)
# =============================================================================
class ExpensiveDataset(Dataset):
    """
    Dataset with expensive preprocessing to demonstrate worker benefits.
    
    Simulates:
      • Image decoding
      • Data augmentation
      • Feature extraction
      • Any CPU-intensive operation
    """
    
    def __init__(self, num_samples=100, processing_time=0.01, seed=0):
        """
        Args:
            num_samples: Total samples
            processing_time: Simulated preprocessing time (seconds)
            seed: Random seed
        """
        self.num_samples = num_samples
        self.processing_time = processing_time
        torch.manual_seed(seed)
        
        # Pre-generated data (like image paths in real scenarios)
        self.data_ids = list(range(num_samples))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Simulate expensive preprocessing
        time.sleep(self.processing_time)
        
        # Generate data (in reality: load image, decode, augment)
        x = torch.randn(32)  # Features
        y = torch.tensor(self.data_ids[idx] % 2)  # Binary label
        
        return x, y


# =============================================================================
# DEMO 1: num_workers Comparison
# =============================================================================
def demo_num_workers_performance():
    """
    Compare loading speed with different num_workers.
    
    Key Insight:
    -----------
    More workers → Faster data loading → Better GPU utilization
    But: Diminishing returns after 4-8 workers
    """
    print("\n" + "="*60)
    print("DEMO 1: num_workers Performance Impact")
    print("="*60)
    
    dataset = ExpensiveDataset(num_samples=50, processing_time=0.02)
    
    worker_configs = [0, 1, 2, 4]
    
    print("\n⏱️  Loading speed comparison:")
    print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<10} {'Note'}")
    print("-" * 65)
    
    baseline_time = None
    
    for num_workers in worker_configs:
        loader = DataLoader(
            dataset,
            batch_size=10,
            num_workers=num_workers,
            shuffle=False
        )
        
        # Measure time to iterate through all batches
        start_time = time.time()
        for _ in loader:
            pass
        elapsed = time.time() - start_time
        
        if baseline_time is None:
            baseline_time = elapsed
            speedup_str = "Baseline"
        else:
            speedup = baseline_time / elapsed
            speedup_str = f"{speedup:.2f}x faster"
        
        note = ""
        if num_workers == 0:
            note = "Main process only"
        elif num_workers == 1:
            note = "One worker"
        else:
            note = f"{num_workers} parallel workers"
        
        print(f"{num_workers:<10} {elapsed:<12.3f} {speedup_str:<10} {note}")
    
    print("\n💡 Observations:")
    print("   • num_workers=0: Slowest, blocks main process")
    print("   • num_workers=2-4: Sweet spot for most cases")
    print("   • More workers → Faster, but diminishing returns")
    print("   • Too many workers → Memory overhead, slower startup")


# =============================================================================
# DEMO 2: Worker Process Architecture
# =============================================================================
def demo_worker_architecture():
    """
    Understand how worker processes work.
    
    Architecture:
    ------------
    Main Process:
      ├─ Worker 1: Loads batches 0, 4, 8, ...
      ├─ Worker 2: Loads batches 1, 5, 9, ...
      ├─ Worker 3: Loads batches 2, 6, 10, ...
      └─ Worker 4: Loads batches 3, 7, 11, ...
    
    Each worker handles a subset of batches in round-robin fashion.
    """
    print("\n" + "="*60)
    print("DEMO 2: Worker Process Architecture")
    print("="*60)
    
    class WorkerAwareDataset(Dataset):
        """Dataset that reports which worker loads each sample."""
        
        def __init__(self, num_samples=16):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Get worker info (only available in worker processes)
            worker_info = torch.utils.data.get_worker_info()
            
            if worker_info is None:
                worker_id = "Main"
            else:
                worker_id = worker_info.id
            
            return torch.tensor(idx), f"Worker {worker_id}"
    
    dataset = WorkerAwareDataset(num_samples=16)
    
    print("\n🔍 With 4 workers:")
    loader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)
    
    for batch_idx, (indices, worker_ids) in enumerate(loader):
        print(f"  Batch {batch_idx}: indices {indices.tolist()}")
        print(f"            loaded by {worker_ids}")
    
    print("\n💡 Pattern:")
    print("   Each worker handles batches in round-robin order")
    print("   Worker 0 → batches 0, 4, 8, ...")
    print("   Worker 1 → batches 1, 5, 9, ...")
    print("   This ensures load balancing")


# =============================================================================
# DEMO 3: persistent_workers Parameter
# =============================================================================
def demo_persistent_workers():
    """
    persistent_workers keeps worker processes alive between epochs.
    
    Benefits:
    --------
    ✓ No process spawn overhead per epoch
    ✓ Faster epoch start
    ✓ Workers can maintain state (open files, caches)
    
    Cost:
    ----
    ✗ More memory (workers stay alive)
    ✗ Must be careful with stateful operations
    """
    print("\n" + "="*60)
    print("DEMO 3: persistent_workers")
    print("="*60)
    
    dataset = ExpensiveDataset(num_samples=20, processing_time=0.01)
    
    # Test with persistent_workers=False (default)
    print("\n⏱️  persistent_workers=False:")
    print("   Workers spawn → load data → terminate (each epoch)")
    
    loader_normal = DataLoader(
        dataset,
        batch_size=5,
        num_workers=2,
        persistent_workers=False
    )
    
    epoch_times_normal = []
    for epoch in range(3):
        start = time.time()
        for _ in loader_normal:
            pass
        epoch_time = time.time() - start
        epoch_times_normal.append(epoch_time)
        print(f"   Epoch {epoch+1}: {epoch_time:.3f}s (spawn + load)")
    
    # Test with persistent_workers=True
    print("\n⏱️  persistent_workers=True:")
    print("   Workers spawn once → reuse for all epochs")
    
    loader_persistent = DataLoader(
        dataset,
        batch_size=5,
        num_workers=2,
        persistent_workers=True
    )
    
    epoch_times_persistent = []
    for epoch in range(3):
        start = time.time()
        for _ in loader_persistent:
            pass
        epoch_time = time.time() - start
        epoch_times_persistent.append(epoch_time)
        
        if epoch == 0:
            print(f"   Epoch {epoch+1}: {epoch_time:.3f}s (spawn + load)")
        else:
            print(f"   Epoch {epoch+1}: {epoch_time:.3f}s (reuse workers)")
    
    print(f"\n📊 Average time per epoch:")
    print(f"   Without persistent: {np.mean(epoch_times_normal):.3f}s")
    print(f"   With persistent: {np.mean(epoch_times_persistent):.3f}s")
    
    print("\n💡 Recommendation:")
    print("   Use persistent_workers=True for training (multiple epochs)")
    print("   Use persistent_workers=False for one-time loading")


# =============================================================================
# DEMO 4: Managing Randomness Across Workers
# =============================================================================
def worker_init_fn_example(worker_id):
    """
    Initialize each worker with different random seed.
    
    Why needed:
    ----------
    All workers start with SAME random seed by default!
    This causes:
      • Same augmentations across workers
      • Identical shuffling
      • Non-random behavior
    
    Solution:
    --------
    Use worker_init_fn to set different seeds per worker.
    """
    # Get the dataset object
    worker_info = torch.utils.data.get_worker_info()
    
    # Set different seeds for each worker
    seed = worker_info.seed % (2**32 - 1)  # Get base seed
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
    
    print(f"  Worker {worker_id} initialized with seed {seed + worker_id}")


def demo_worker_randomness():
    """
    Show the importance of worker_init_fn for randomness.
    """
    print("\n" + "="*60)
    print("DEMO 4: Managing Randomness Across Workers")
    print("="*60)
    
    class RandomAugmentDataset(Dataset):
        """Dataset that applies random augmentation."""
        
        def __init__(self, num_samples=8):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Simulate random augmentation
            noise = torch.rand(1).item()  # Random value 0-1
            return torch.tensor(idx), torch.tensor(noise)
    
    dataset = RandomAugmentDataset(num_samples=8)
    
    print("\n❌ WITHOUT worker_init_fn:")
    print("   (all workers might use same random seed)")
    loader_bad = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        shuffle=False
    )
    
    for batch_idx, (indices, noise_vals) in enumerate(loader_bad):
        print(f"   Batch {batch_idx}: noise values {noise_vals.tolist()}")
    
    print("\n✅ WITH worker_init_fn:")
    print("   (each worker has different random seed)")
    loader_good = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        shuffle=False,
        worker_init_fn=worker_init_fn_example
    )
    
    for batch_idx, (indices, noise_vals) in enumerate(loader_good):
        print(f"   Batch {batch_idx}: noise values {noise_vals.tolist()}")
    
    print("\n💡 Always use worker_init_fn when:")
    print("   • Applying random data augmentation")
    print("   • Using any randomness in __getitem__")
    print("   • num_workers > 0")


# =============================================================================
# DEMO 5: prefetch_factor Parameter
# =============================================================================
def demo_prefetch_factor():
    """
    prefetch_factor controls how many batches each worker pre-loads.
    
    How it works:
    ------------
    Each worker maintains a queue of pre-loaded batches.
    Higher prefetch_factor → More batches ready → Less waiting
    
    Trade-off:
    ---------
    ✓ Higher: Less GPU idle time, smoother training
    ✗ Higher: More CPU memory usage
    
    Default: 2 (good for most cases)
    Increase to 4-8 if GPU utilization is low
    """
    print("\n" + "="*60)
    print("DEMO 5: prefetch_factor")
    print("="*60)
    
    dataset = ExpensiveDataset(num_samples=40, processing_time=0.02)
    
    configs = [
        {"prefetch_factor": 2, "desc": "Default (2 batches ahead)"},
        {"prefetch_factor": 4, "desc": "Higher (4 batches ahead)"},
    ]
    
    print("\n⏱️  Prefetch factor comparison:")
    print(f"{'Prefetch':<12} {'Time (s)':<12} {'Description'}")
    print("-" * 65)
    
    for cfg in configs:
        loader = DataLoader(
            dataset,
            batch_size=10,
            num_workers=2,
            prefetch_factor=cfg["prefetch_factor"]
        )
        
        start = time.time()
        for _ in loader:
            time.sleep(0.001)  # Simulate GPU computation
        elapsed = time.time() - start
        
        print(f"{cfg['prefetch_factor']:<12} {elapsed:<12.3f} {cfg['desc']}")
    
    print("\n💡 Tuning Guide:")
    print("   • GPU idle? → Increase prefetch_factor")
    print("   • Out of memory? → Decrease prefetch_factor")
    print("   • Default (2) works well for most cases")


# =============================================================================
# DEMO 6: Complete Production Configuration
# =============================================================================
def demo_production_config():
    """
    Recommended DataLoader configuration for production training.
    """
    print("\n" + "="*60)
    print("DEMO 6: Production Configuration")
    print("="*60)
    
    dataset = ExpensiveDataset(num_samples=100, processing_time=0.01)
    
    print("\n🏭 Recommended Training Configuration:\n")
    
    # Worker initialization
    def worker_init(worker_id):
        seed = torch.initial_seed() % (2**32 - 1)
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
    
    # Create loader
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,                # 4 workers (adjust based on CPU)
        pin_memory=True,              # Fast GPU transfer
        persistent_workers=True,      # Keep workers alive
        prefetch_factor=2,            # 2 batches ahead per worker
        worker_init_fn=worker_init,   # Different seeds per worker
        drop_last=True,               # Consistent batch sizes
    )
    
    print("Configuration:")
    print(f"  • batch_size:         32")
    print(f"  • shuffle:            True")
    print(f"  • num_workers:        4")
    print(f"  • pin_memory:         True")
    print(f"  • persistent_workers: True")
    print(f"  • prefetch_factor:    2")
    print(f"  • worker_init_fn:     Custom (seed setting)")
    print(f"  • drop_last:          True")
    
    print("\n📊 Performance characteristics:")
    start = time.time()
    batch_count = 0
    for _ in train_loader:
        batch_count += 1
        if batch_count >= 3:  # Just test first 3 batches
            break
    elapsed = time.time() - start
    
    print(f"  • First 3 batches: {elapsed:.3f}s")
    print(f"  • Time per batch: {elapsed/3:.3f}s")
    
    print("\n💡 This configuration:")
    print("   ✓ Maximizes throughput")
    print("   ✓ Minimizes GPU idle time")
    print("   ✓ Proper randomness handling")
    print("   ✓ Memory efficient with pinning")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MULTI-PROCESS DATA LOADING: COMPLETE GUIDE")
    print("="*60)
    
    demo_num_workers_performance()
    demo_worker_architecture()
    demo_persistent_workers()
    demo_worker_randomness()
    demo_prefetch_factor()
    demo_production_config()
    
    print("\n" + "="*60)
    print("✓ TUTORIAL COMPLETE!")
    print("="*60)
    print("\n📚 Key Takeaways:")
    print("  1. num_workers>0 enables parallel data loading")
    print("  2. persistent_workers=True saves process spawn overhead")
    print("  3. worker_init_fn is essential for proper randomness")
    print("  4. prefetch_factor controls memory vs speed trade-off")
    print("  5. Optimal config depends on your hardware and data")
    print("\n🎯 Next Steps:")
    print("  → Profile your data loading bottlenecks")
    print("  → Tune num_workers for your specific GPU/CPU")
    print("  → Learn about DistributedSampler for multi-GPU")
