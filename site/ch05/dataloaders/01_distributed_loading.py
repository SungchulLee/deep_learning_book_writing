#!/usr/bin/env python3
"""
=============================================================================
Distributed Data Loading: Multi-GPU Training
=============================================================================

OVERVIEW:
---------
When training on multiple GPUs, each GPU needs DIFFERENT data to avoid
redundant computation. DistributedSampler ensures each GPU sees unique batches.

KEY CONCEPTS:
-------------
• DistributedSampler: Splits dataset across multiple processes/GPUs
• Rank: ID of current process (0, 1, 2, ... for each GPU)
• World Size: Total number of processes/GPUs
• Shuffle: Must be handled carefully in distributed setting

USE CASES:
----------
✓ Multi-GPU training (single machine)
✓ Multi-node training (multiple machines)
✓ Data parallelism with DDP (DistributedDataParallel)

LEARNING OBJECTIVES:
-------------------
✓ Understand distributed data loading architecture
✓ Use DistributedSampler correctly
✓ Handle shuffling and randomness
✓ Avoid common distributed pitfalls

DIFFICULTY: ⭐⭐⭐ Advanced
TIME: 25 minutes
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import List


# =============================================================================
# Demo Dataset
# =============================================================================
class SimpleDataset(Dataset):
    """Simple dataset for demonstrating distributed loading."""
    
    def __init__(self, num_samples=32):
        self.num_samples = num_samples
        self.data = list(range(num_samples))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])


# =============================================================================
# DEMO 1: Understanding DistributedSampler
# =============================================================================
def demo_distributed_sampler_basics():
    """
    Show how DistributedSampler splits data across processes.
    
    With 4 GPUs and 16 samples:
      GPU 0 gets samples: 0, 4, 8, 12
      GPU 1 gets samples: 1, 5, 9, 13
      GPU 2 gets samples: 2, 6, 10, 14
      GPU 3 gets samples: 3, 7, 11, 15
    
    Each GPU sees different data (no overlap).
    """
    print("\n" + "="*60)
    print("DEMO 1: DistributedSampler Basics")
    print("="*60)
    
    dataset = SimpleDataset(num_samples=16)
    
    print("\n📊 Simulating 4 GPUs (4 processes):\n")
    
    # Simulate 4 processes (GPUs)
    world_size = 4
    
    for rank in range(world_size):
        # Create sampler for this process
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,  # Total number of processes
            rank=rank,                 # ID of this process
            shuffle=False              # Disable for clear demo
        )
        
        # Create DataLoader with this sampler
        loader = DataLoader(
            dataset,
            batch_size=2,
            sampler=sampler
        )
        
        # Collect samples seen by this GPU
        samples_seen = []
        for batch in loader:
            samples_seen.extend(batch.tolist())
        
        print(f"  GPU {rank} sees samples: {samples_seen}")
    
    print("\n💡 Key Points:")
    print("   • Each GPU sees different samples")
    print("   • No overlap between GPUs")
    print("   • Samples distributed in round-robin fashion")
    print("   • MUST use DistributedSampler for proper DDP training")


# =============================================================================
# DEMO 2: Handling Shuffling in Distributed Setting
# =============================================================================
def demo_distributed_shuffling():
    """
    Shuffling in distributed setting requires special handling.
    
    Challenges:
    ----------
    • Each process needs different shuffle order
    • But shuffle should be consistent within epoch
    • Must set epoch number for proper shuffling
    
    Solution:
    --------
    Call sampler.set_epoch(epoch) at start of each epoch
    """
    print("\n" + "="*60)
    print("DEMO 2: Distributed Shuffling")
    print("="*60)
    
    dataset = SimpleDataset(num_samples=12)
    world_size = 3
    
    print("\n🔀 Shuffle WITH set_epoch():\n")
    
    for rank in range(world_size):
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,  # Enable shuffling
            seed=42        # Shared seed for reproducibility
        )
        
        print(f"  GPU {rank}:")
        
        for epoch in range(2):
            # CRITICAL: Set epoch before each epoch
            sampler.set_epoch(epoch)
            
            loader = DataLoader(dataset, batch_size=2, sampler=sampler)
            samples = []
            for batch in loader:
                samples.extend(batch.tolist())
            
            print(f"    Epoch {epoch}: {samples}")
    
    print("\n💡 Critical:")
    print("   ALWAYS call sampler.set_epoch(epoch) before each epoch!")
    print("   Without this, shuffle order will be the same every epoch")


# =============================================================================
# DEMO 3: Uneven Data Distribution
# =============================================================================
def demo_uneven_distribution():
    """
    What happens when dataset size doesn't divide evenly by num_replicas?
    
    Example: 10 samples, 3 GPUs
      GPU 0: 4 samples
      GPU 1: 3 samples
      GPU 2: 3 samples
    
    DistributedSampler handles this by padding or dropping samples.
    """
    print("\n" + "="*60)
    print("DEMO 3: Uneven Data Distribution")
    print("="*60)
    
    # Dataset with 10 samples (not divisible by 3)
    dataset = SimpleDataset(num_samples=10)
    world_size = 3
    
    print("\n📊 10 samples split across 3 GPUs:\n")
    
    print("  drop_last=False (default - pad to equal):")
    for rank in range(world_size):
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False  # Pad to make equal
        )
        
        loader = DataLoader(dataset, batch_size=4, sampler=sampler)
        samples = []
        for batch in loader:
            samples.extend(batch.tolist())
        
        print(f"    GPU {rank}: {samples} ({len(samples)} samples)")
    
    print("\n  drop_last=True (drop to make equal):")
    for rank in range(world_size):
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=True  # Drop to make equal
        )
        
        loader = DataLoader(dataset, batch_size=4, sampler=sampler)
        samples = []
        for batch in loader:
            samples.extend(batch.tolist())
        
        print(f"    GPU {rank}: {samples} ({len(samples)} samples)")
    
    print("\n💡 Recommendation:")
    print("   Training: drop_last=False (use all data)")
    print("   Validation: Either works (small difference)")


# =============================================================================
# DEMO 4: Complete Distributed Training Setup
# =============================================================================
def demo_distributed_training_setup():
    """
    Complete example of DataLoader setup for distributed training.
    
    This shows the typical pattern you'd use with
    torch.nn.parallel.DistributedDataParallel (DDP).
    """
    print("\n" + "="*60)
    print("DEMO 4: Complete Distributed Training Setup")
    print("="*60)
    
    print("\n📝 Typical DDP Training Code Pattern:\n")
    
    code_example = '''
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup_distributed(rank, world_size):
    """Initialize distributed process group."""
    dist.init_process_group(
        backend='nccl',      # Use NCCL for GPU
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

def train(rank, world_size):
    # 1. Setup distributed
    setup_distributed(rank, world_size)
    torch.cuda.set_device(rank)
    
    # 2. Create dataset
    dataset = YourDataset()
    
    # 3. Create DistributedSampler
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    # 4. Create DataLoader with sampler
    train_loader = DataLoader(
        dataset,
        batch_size=32,           # Per-GPU batch size
        sampler=train_sampler,   # Use distributed sampler
        num_workers=4,           # Per-GPU workers
        pin_memory=True,
        persistent_workers=True
    )
    
    # 5. Create model and wrap with DDP
    model = YourModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 6. Training loop
    for epoch in range(num_epochs):
        # CRITICAL: Set epoch for proper shuffling
        train_sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(rank)
            target = target.to(rank)
            
            # Forward, backward, optimize
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # 7. Cleanup
    dist.destroy_process_group()

# Launch with: torchrun --nproc_per_node=4 train.py
'''
    
    print(code_example)
    
    print("\n💡 Key Points:")
    print("   1. Create DistributedSampler with rank and world_size")
    print("   2. Pass sampler to DataLoader (NOT shuffle=True)")
    print("   3. Call sampler.set_epoch() before each epoch")
    print("   4. batch_size is PER GPU (total = batch_size × num_gpus)")
    print("   5. Use torchrun or torch.distributed.launch to run")


# =============================================================================
# DEMO 5: Validation with Distributed Sampler
# =============================================================================
def demo_distributed_validation():
    """
    Validation/testing in distributed setting.
    
    Important differences from training:
    ------------------------------------
    • No shuffling needed
    • Want to evaluate on ALL data exactly once
    • Each GPU evaluates different subset
    • Must aggregate metrics across GPUs
    """
    print("\n" + "="*60)
    print("DEMO 5: Distributed Validation")
    print("="*60)
    
    dataset = SimpleDataset(num_samples=12)
    world_size = 3
    
    print("\n✅ Proper validation setup:\n")
    
    validation_code = '''
def validate(rank, world_size, model, dataset):
    # 1. Create sampler (no shuffle for validation)
    val_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,        # No shuffling for validation
        drop_last=False       # Use all validation data
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=64,        # Can be larger (no backprop)
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            
            # Accumulate metrics
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += data.size(0)
    
    # 2. Aggregate metrics across all GPUs
    import torch.distributed as dist
    
    # Convert to tensors for all_reduce
    total_loss_tensor = torch.tensor(total_loss).to(rank)
    total_correct_tensor = torch.tensor(total_correct).to(rank)
    total_samples_tensor = torch.tensor(total_samples).to(rank)
    
    # Sum across all GPUs
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    
    # Calculate final metrics
    avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
    accuracy = total_correct_tensor.item() / total_samples_tensor.item()
    
    if rank == 0:  # Only print on main process
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy
'''
    
    print(validation_code)
    
    print("\n💡 Critical Steps:")
    print("   1. Use shuffle=False for validation")
    print("   2. Each GPU processes different subset")
    print("   3. Aggregate metrics with dist.all_reduce()")
    print("   4. Only print results on rank 0")


# =============================================================================
# DEMO 6: Common Pitfalls
# =============================================================================
def demo_common_pitfalls():
    """
    Common mistakes when using distributed data loading.
    """
    print("\n" + "="*60)
    print("DEMO 6: Common Pitfalls and Solutions")
    print("="*60)
    
    print("\n❌ PITFALL 1: Using shuffle=True with DistributedSampler")
    print("   Problem:")
    print("     loader = DataLoader(dataset, shuffle=True, sampler=sampler)")
    print("   Error: Cannot use shuffle with sampler")
    print("   Solution:")
    print("     sampler = DistributedSampler(dataset, shuffle=True)")
    print("     loader = DataLoader(dataset, sampler=sampler)  # No shuffle arg")
    
    print("\n❌ PITFALL 2: Forgetting set_epoch()")
    print("   Problem:")
    print("     for epoch in range(10):")
    print("         for batch in loader:")
    print("             # Training...")
    print("   Result: Same shuffle order every epoch")
    print("   Solution:")
    print("     for epoch in range(10):")
    print("         sampler.set_epoch(epoch)  # ← Add this!")
    print("         for batch in loader:")
    print("             # Training...")
    
    print("\n❌ PITFALL 3: Wrong batch_size calculation")
    print("   Problem: Thinking batch_size is total across all GPUs")
    print("   Reality: batch_size is PER GPU")
    print("   Example:")
    print("     4 GPUs, batch_size=32")
    print("     → Effective batch size = 32 × 4 = 128")
    print("   Solution: Adjust batch_size accordingly")
    
    print("\n❌ PITFALL 4: Not aggregating validation metrics")
    print("   Problem: Each GPU computes metrics on its subset only")
    print("   Result: Metrics are wrong (only 1/N of data)")
    print("   Solution: Use dist.all_reduce() to sum across GPUs")
    
    print("\n❌ PITFALL 5: Incorrect world_size or rank")
    print("   Problem: Manually setting wrong values")
    print("   Solution: Get from torch.distributed:")
    print("     rank = dist.get_rank()")
    print("     world_size = dist.get_world_size()")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DISTRIBUTED DATA LOADING: COMPLETE GUIDE")
    print("="*60)
    
    demo_distributed_sampler_basics()
    demo_distributed_shuffling()
    demo_uneven_distribution()
    demo_distributed_training_setup()
    demo_distributed_validation()
    demo_common_pitfalls()
    
    print("\n" + "="*60)
    print("✓ TUTORIAL COMPLETE!")
    print("="*60)
    print("\n📚 Key Takeaways:")
    print("  1. Use DistributedSampler for multi-GPU training")
    print("  2. ALWAYS call sampler.set_epoch(epoch) before each epoch")
    print("  3. batch_size is per-GPU (not total)")
    print("  4. Aggregate validation metrics with all_reduce()")
    print("  5. Never use shuffle=True with DistributedSampler")
    print("\n🎯 Next Steps:")
    print("  → Practice with real DDP training")
    print("  → Learn about gradient accumulation")
    print("  → Explore model parallelism vs data parallelism")
