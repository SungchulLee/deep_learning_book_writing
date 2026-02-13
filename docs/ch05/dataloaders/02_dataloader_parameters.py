#!/usr/bin/env python3
"""
=============================================================================
DataLoader Parameters: Complete Configuration Guide
=============================================================================

OVERVIEW:
---------
DataLoader has many parameters that control its behavior. This tutorial
covers all the important ones and when to use each.

PARAMETERS COVERED:
------------------
✓ batch_size: Number of samples per batch
✓ shuffle: Whether to randomize sample order
✓ num_workers: Number of parallel data loading processes
✓ pin_memory: Speed up CPU → GPU transfer
✓ drop_last: Handle the final incomplete batch
✓ timeout: Worker timeout for debugging hangs
✓ generator: Control randomness for reproducibility

LEARNING OBJECTIVES:
-------------------
✓ Understand each DataLoader parameter
✓ Learn when to use each configuration
✓ See performance implications
✓ Master common usage patterns

DIFFICULTY: ⭐⭐ Intermediate
TIME: 15 minutes
"""

import torch
import time
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Demo Dataset
# =============================================================================
class SyntheticDataset(Dataset):
    """Simple dataset for parameter demonstrations."""
    
    def __init__(self, num_samples=100, feature_dim=10, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.X = torch.randn(num_samples, feature_dim, generator=gen)
        self.y = torch.randint(0, 2, (num_samples,), generator=gen)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Simulate some processing time
        time.sleep(0.001)  # 1ms per sample
        return self.X[idx], self.y[idx]


# =============================================================================
# DEMO 1: batch_size Parameter
# =============================================================================
def demo_batch_size():
    """
    batch_size controls the number of samples per batch.
    
    Trade-offs:
    -----------
    LARGER batch_size:
      ✓ Faster training (better GPU utilization)
      ✓ More stable gradients
      ✗ More GPU memory required
      ✗ May lead to worse generalization
    
    SMALLER batch_size:
      ✓ Less memory required
      ✓ More noise in gradients (can help escape local minima)
      ✗ Slower training
      ✗ Less stable gradients
    
    Common Values:
    --------------
    - Small models: 32, 64, 128
    - Large models (CNNs): 16, 32, 64
    - Transformers: 8, 16, 32 (due to large memory footprint)
    """
    print("\n" + "="*60)
    print("DEMO 1: batch_size Parameter")
    print("="*60)
    
    dataset = SyntheticDataset(num_samples=100, seed=1)
    
    configs = [
        {"batch_size": 10, "desc": "Small batch (more updates)"},
        {"batch_size": 32, "desc": "Medium batch (balanced)"},
        {"batch_size": 100, "desc": "Full batch (one update)"},
    ]
    
    print(f"\n📊 Batch Size Comparison (100 samples):\n")
    print(f"{'Batch Size':<12} {'Batches':<10} {'Last Batch':<12} {'Description'}")
    print("-" * 70)
    
    for cfg in configs:
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)
        
        batch_sizes = [len(batch[0]) for batch in loader]
        num_batches = len(batch_sizes)
        last_batch_size = batch_sizes[-1]
        
        print(f"{cfg['batch_size']:<12} {num_batches:<10} {last_batch_size:<12} {cfg['desc']}")
    
    print("\n💡 Rule of Thumb:")
    print("   Start with batch_size=32, then adjust based on:")
    print("   - GPU memory (larger if available)")
    print("   - Training stability (larger for stability)")
    print("   - Generalization (smaller sometimes helps)")


# =============================================================================
# DEMO 2: shuffle and generator Parameters
# =============================================================================
def demo_shuffle_and_generator():
    """
    shuffle controls randomization, generator controls reproducibility.
    
    When to Use:
    -----------
    shuffle=True:
      - Training (prevents learning from data order)
      - Each epoch sees different sample order
    
    shuffle=False:
      - Validation/Test (reproducible evaluation)
      - When order matters (time series)
      - Debugging
    
    generator:
      - Ensures reproducible shuffling
      - Important for experiment reproducibility
      - Separate from model randomness
    """
    print("\n" + "="*60)
    print("DEMO 2: shuffle and generator Parameters")
    print("="*60)
    
    dataset = SyntheticDataset(num_samples=10, seed=2)
    
    # Without shuffle
    print("\n🔹 shuffle=False (same order every epoch):")
    loader_no_shuffle = DataLoader(dataset, batch_size=3, shuffle=False)
    
    for epoch in range(2):
        labels = [batch[1].tolist() for batch in loader_no_shuffle]
        print(f"  Epoch {epoch+1}: {labels}")
    
    # With shuffle, no generator (non-reproducible)
    print("\n🔹 shuffle=True, no generator (random every run):")
    print("  Note: Order would differ if you run this script again")
    loader_random = DataLoader(dataset, batch_size=3, shuffle=True)
    
    for epoch in range(2):
        labels = [batch[1].tolist() for batch in loader_random]
        print(f"  Epoch {epoch+1}: {labels}")
    
    # With shuffle and generator (reproducible)
    print("\n🔹 shuffle=True, with generator (reproducible):")
    print("  Note: Same order every time you run this script")
    
    for run in range(2):
        gen = torch.Generator().manual_seed(42)  # Same seed
        loader_repro = DataLoader(dataset, batch_size=3, shuffle=True, generator=gen)
        
        labels = [batch[1].tolist() for batch in loader_repro]
        print(f"  Run {run+1}: {labels}")
    
    print("\n💡 Best Practice:")
    print("   ALWAYS use generator with shuffle=True for reproducible experiments!")


# =============================================================================
# DEMO 3: drop_last Parameter
# =============================================================================
def demo_drop_last():
    """
    drop_last controls what happens to the final incomplete batch.
    
    When to Use:
    -----------
    drop_last=True:
      - When you need consistent batch sizes (some models require this)
      - When last batch is very small (e.g., 1-2 samples)
      - Batch normalization (needs >1 sample per batch)
    
    drop_last=False (default):
      - When you want to use all data
      - Validation/Test sets (don't waste data)
      - When batch size doesn't matter
    """
    print("\n" + "="*60)
    print("DEMO 3: drop_last Parameter")
    print("="*60)
    
    dataset = SyntheticDataset(num_samples=10, seed=3)
    
    # Without drop_last (default)
    print("\n🔹 drop_last=False (keep all data):")
    loader_keep = DataLoader(dataset, batch_size=3, drop_last=False)
    
    for batch_idx, (features, _) in enumerate(loader_keep, 1):
        print(f"  Batch {batch_idx}: {features.shape[0]} samples")
    
    print(f"  Total samples processed: {sum(f.shape[0] for f, _ in loader_keep)}")
    
    # With drop_last
    print("\n🔹 drop_last=True (uniform batch sizes):")
    loader_drop = DataLoader(dataset, batch_size=3, drop_last=True)
    
    for batch_idx, (features, _) in enumerate(loader_drop, 1):
        print(f"  Batch {batch_idx}: {features.shape[0]} samples")
    
    total_processed = sum(f.shape[0] for f, _ in loader_drop)
    total_dropped = len(dataset) - total_processed
    print(f"  Total samples processed: {total_processed}")
    print(f"  Samples dropped: {total_dropped}")
    
    print("\n💡 Recommendation:")
    print("   - Training: drop_last=True (for consistency)")
    print("   - Validation/Test: drop_last=False (use all data)")


# =============================================================================
# DEMO 4: num_workers Parameter
# =============================================================================
def demo_num_workers():
    """
    num_workers enables parallel data loading.
    
    How It Works:
    ------------
    - num_workers=0: Main process loads data (default, simple)
    - num_workers>0: Spawn worker processes to load data in parallel
    
    Benefits:
    ---------
    ✓ Speeds up data loading (GPU doesn't wait for CPU)
    ✓ Crucial when data preprocessing is expensive
    ✓ Can improve GPU utilization significantly
    
    Considerations:
    --------------
    ✗ More memory usage (each worker has copy of dataset)
    ✗ Slower startup (spawning processes)
    ✗ Can cause issues with some libraries (CUDA, matplotlib)
    
    Optimal Value:
    -------------
    - Start with num_workers=4
    - Increase if GPU utilization is low
    - Decrease if running out of memory
    - Never exceed CPU cores
    """
    print("\n" + "="*60)
    print("DEMO 4: num_workers Parameter")
    print("="*60)
    
    dataset = SyntheticDataset(num_samples=50, seed=4)
    
    configs = [
        {"num_workers": 0, "desc": "Single process (simple)"},
        {"num_workers": 2, "desc": "2 workers (faster)"},
        {"num_workers": 4, "desc": "4 workers (even faster)"},
    ]
    
    print("\n⏱️  Loading Speed Comparison:")
    print(f"{'Workers':<10} {'Time (sec)':<12} {'Speedup':<10} {'Description'}")
    print("-" * 65)
    
    baseline_time = None
    
    for cfg in configs:
        loader = DataLoader(
            dataset,
            batch_size=10,
            num_workers=cfg["num_workers"],
            shuffle=False
        )
        
        # Measure loading time
        start = time.time()
        for _ in loader:
            pass  # Just iterate through
        elapsed = time.time() - start
        
        if baseline_time is None:
            baseline_time = elapsed
            speedup = "1.0x (baseline)"
        else:
            speedup = f"{baseline_time/elapsed:.1f}x"
        
        print(f"{cfg['num_workers']:<10} {elapsed:<12.3f} {speedup:<10} {cfg['desc']}")
    
    print("\n💡 Best Practice:")
    print("   - Development: num_workers=0 (easier debugging)")
    print("   - Training: num_workers=4 or more (faster)")
    print("   - CPU-bound task: Set to CPU core count")


# =============================================================================
# DEMO 5: pin_memory Parameter
# =============================================================================
def demo_pin_memory():
    """
    pin_memory speeds up CPU → GPU data transfer.
    
    How It Works:
    ------------
    - Allocates data in pinned (page-locked) memory
    - GPU can directly access this memory
    - Enables asynchronous transfer (non-blocking)
    
    When to Use:
    -----------
    ✓ ALWAYS use when training on GPU
    ✓ Especially with large batches
    ✓ When data loading is bottleneck
    
    Trade-offs:
    ----------
    ✓ Faster GPU transfer (5-10% speedup)
    ✗ More CPU memory used
    ✗ Slightly slower to allocate
    
    Note: Only beneficial when using GPU!
    """
    print("\n" + "="*60)
    print("DEMO 5: pin_memory Parameter")
    print("="*60)
    
    dataset = SyntheticDataset(num_samples=100, seed=5)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if device.type == 'cpu':
        print("  ⚠️  GPU not available - pin_memory has no effect on CPU")
        print("     Running demo anyway for educational purposes...")
    
    configs = [
        {"pin_memory": False, "desc": "Standard allocation"},
        {"pin_memory": True, "desc": "Pinned memory (faster GPU)"},
    ]
    
    print(f"\n{'Pin Memory':<12} {'Status':<15} {'Recommendation'}")
    print("-" * 60)
    
    for cfg in configs:
        loader = DataLoader(
            dataset,
            batch_size=32,
            pin_memory=cfg["pin_memory"],
            num_workers=0
        )
        
        # Get a batch and check if it's pinned
        batch = next(iter(loader))
        is_pinned = batch[0].is_pinned()
        
        status = "Pinned ✓" if is_pinned else "Not pinned"
        
        print(f"{cfg['pin_memory']!s:<12} {status:<15} {cfg['desc']}")
    
    print("\n💡 Always Use:")
    print("   pin_memory=True when training on GPU!")
    print("\n   Full GPU training configuration:")
    print("   DataLoader(..., pin_memory=True, num_workers=4)")


# =============================================================================
# DEMO 6: Complete Configuration Example
# =============================================================================
def demo_complete_configuration():
    """
    Putting it all together: recommended configurations for different scenarios.
    """
    print("\n" + "="*60)
    print("DEMO 6: Recommended Configurations")
    print("="*60)
    
    dataset = SyntheticDataset(num_samples=100, seed=6)
    
    # Training configuration
    print("\n🏋️  Training Configuration:")
    print("   Goal: Fast, randomized, efficient")
    gen_train = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,         # Randomize each epoch
        num_workers=4,        # Parallel loading
        pin_memory=True,      # Fast GPU transfer
        drop_last=True,       # Consistent batch sizes
        generator=gen_train,  # Reproducible shuffling
    )
    
    print(f"   • batch_size={train_loader.batch_size}")
    print(f"   • shuffle={True}")
    print(f"   • num_workers={train_loader.num_workers}")
    print(f"   • pin_memory={train_loader.pin_memory}")
    print(f"   • drop_last={train_loader.drop_last}")
    
    # Validation configuration
    print("\n✅ Validation Configuration:")
    print("   Goal: Reproducible, use all data")
    val_loader = DataLoader(
        dataset,
        batch_size=64,        # Can be larger (no backprop)
        shuffle=False,        # Fixed order for reproducibility
        num_workers=4,        # Still parallelize
        pin_memory=True,      # Still use GPU
        drop_last=False,      # Use all validation data
    )
    
    print(f"   • batch_size={val_loader.batch_size} (larger ok)")
    print(f"   • shuffle={False} (reproducible)")
    print(f"   • num_workers={val_loader.num_workers}")
    print(f"   • pin_memory={val_loader.pin_memory}")
    print(f"   • drop_last={val_loader.drop_last} (use all data)")
    
    # Development/Debugging configuration
    print("\n🐛 Development/Debug Configuration:")
    print("   Goal: Simple, reproducible, easy to debug")
    debug_loader = DataLoader(
        dataset,
        batch_size=8,         # Small for quick iteration
        shuffle=False,        # Consistent order
        num_workers=0,        # Single process (easier debugging)
        pin_memory=False,     # Simpler
        drop_last=False,
    )
    
    print(f"   • batch_size={debug_loader.batch_size} (small)")
    print(f"   • shuffle={False}")
    print(f"   • num_workers={debug_loader.num_workers} (single process)")
    print(f"   • pin_memory={debug_loader.pin_memory}")
    print(f"   • drop_last={debug_loader.drop_last}")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATALOADER PARAMETERS: COMPLETE GUIDE")
    print("="*60)
    
    demo_batch_size()
    demo_shuffle_and_generator()
    demo_drop_last()
    demo_num_workers()
    demo_pin_memory()
    demo_complete_configuration()
    
    print("\n" + "="*60)
    print("✓ TUTORIAL COMPLETE!")
    print("="*60)
    print("\n📚 Key Takeaways:")
    print("  1. batch_size: Trade-off between speed and memory")
    print("  2. shuffle + generator: Randomize while staying reproducible")
    print("  3. drop_last: True for training, False for validation")
    print("  4. num_workers: 4+ for fast loading, 0 for debugging")
    print("  5. pin_memory: Always True when using GPU")
    print("\n🎯 Next Steps:")
    print("  → Learn about custom samplers")
    print("  → Master collate_fn for custom batching")
    print("  → Explore advanced multiprocessing options")
