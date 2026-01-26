# Distributed Data Loading

## Learning Objectives

By the end of this section, you will be able to:

- Understand how DistributedSampler splits data across GPUs
- Configure DataLoaders for multi-GPU training
- Handle shuffling correctly in distributed settings
- Implement proper validation in distributed training
- Avoid common distributed data loading pitfalls

## Why Distributed Data Loading?

When training on multiple GPUs using Data Parallel or Distributed Data Parallel (DDP), each GPU should process **different** data to avoid redundant computation.

### Without DistributedSampler

```
GPU 0: [Batch 0] [Batch 1] [Batch 2] ...
GPU 1: [Batch 0] [Batch 1] [Batch 2] ...  ← Same data!
GPU 2: [Batch 0] [Batch 1] [Batch 2] ...  ← Wasted work!
GPU 3: [Batch 0] [Batch 1] [Batch 2] ...
```

### With DistributedSampler

```
GPU 0: [Batch 0] [Batch 4] [Batch 8]  ...
GPU 1: [Batch 1] [Batch 5] [Batch 9]  ...  ← Different data!
GPU 2: [Batch 2] [Batch 6] [Batch 10] ...  ← Efficient!
GPU 3: [Batch 3] [Batch 7] [Batch 11] ...
```

Each GPU processes unique samples, achieving true parallelism.

## DistributedSampler Basics

### Core Concepts

| Term | Definition |
|------|------------|
| **World Size** | Total number of processes (usually = number of GPUs) |
| **Rank** | Unique ID for each process (0, 1, 2, ..., world_size-1) |
| **Shard** | Portion of dataset assigned to one process |

### How Splitting Works

For a dataset of size $N$ and world size $W$:

$$
\text{Samples per GPU} = \lceil N / W \rceil
$$

Distribution follows round-robin pattern:
- Rank 0: indices 0, W, 2W, 3W, ...
- Rank 1: indices 1, W+1, 2W+1, 3W+1, ...
- Rank r: indices r, W+r, 2W+r, 3W+r, ...

### Basic Usage

```python
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

# In each process, after dist.init_process_group()
rank = dist.get_rank()
world_size = dist.get_world_size()

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,  # Total GPUs
    rank=rank,                 # This GPU's ID
    shuffle=True
)

loader = DataLoader(
    dataset,
    batch_size=32,            # Per-GPU batch size
    sampler=sampler,          # Use distributed sampler
    num_workers=4
)
```

### Demonstration (Simulated)

```python
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch

class SimpleDataset(Dataset):
    def __init__(self, n=16):
        self.data = list(range(n))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

# Simulate 4 GPUs
dataset = SimpleDataset(n=16)
world_size = 4

for rank in range(world_size):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    loader = DataLoader(dataset, batch_size=2, sampler=sampler)
    samples = [batch.tolist() for batch in loader]
    
    print(f"GPU {rank}: {samples}")

# Output:
# GPU 0: [[0, 4], [8, 12]]
# GPU 1: [[1, 5], [9, 13]]
# GPU 2: [[2, 6], [10, 14]]
# GPU 3: [[3, 7], [11, 15]]
```

## Shuffling in Distributed Training

### The Epoch Problem

DistributedSampler uses the epoch number as part of the random seed. Without calling `set_epoch()`, every epoch uses the same shuffle order:

```python
# WRONG - same shuffle every epoch
for epoch in range(num_epochs):
    for batch in loader:
        train(batch)
```

### The Solution: set_epoch()

```python
# CORRECT - different shuffle each epoch
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Update shuffle seed
    
    for batch in loader:
        train(batch)
```

### Why set_epoch Matters

```python
sampler = DistributedSampler(dataset, shuffle=True, seed=42)

# Without set_epoch - identical order
for epoch in range(3):
    samples = list(sampler)[:5]
    print(f"Epoch {epoch}: {samples}")

# Output: Same samples every epoch!
# Epoch 0: [7, 2, 14, 8, 11]
# Epoch 1: [7, 2, 14, 8, 11]
# Epoch 2: [7, 2, 14, 8, 11]
```

```python
# With set_epoch - proper shuffling
for epoch in range(3):
    sampler.set_epoch(epoch)  # ← Critical!
    samples = list(sampler)[:5]
    print(f"Epoch {epoch}: {samples}")

# Output: Different each epoch!
# Epoch 0: [7, 2, 14, 8, 11]
# Epoch 1: [3, 9, 0, 15, 5]
# Epoch 2: [12, 1, 6, 10, 4]
```

## Handling Uneven Splits

When dataset size doesn't divide evenly by world size:

### Default Behavior: Padding

```python
# 10 samples, 3 GPUs → 4 samples each (10 padded to 12)
dataset = SimpleDataset(n=10)

for rank in range(3):
    sampler = DistributedSampler(
        dataset,
        num_replicas=3,
        rank=rank,
        shuffle=False,
        drop_last=False  # Default: pad to equal
    )
    print(f"GPU {rank}: {len(list(sampler))} samples")

# Output:
# GPU 0: 4 samples (one is duplicate)
# GPU 1: 4 samples (one is duplicate)
# GPU 2: 4 samples
```

### Alternative: drop_last

```python
# 10 samples, 3 GPUs → 3 samples each (9 used, 1 dropped)
sampler = DistributedSampler(
    dataset,
    num_replicas=3,
    rank=rank,
    drop_last=True  # Drop to make equal
)
```

| Setting | Behavior | Use Case |
|---------|----------|----------|
| drop_last=False | Pad with duplicates | Training (use all data) |
| drop_last=True | Drop extras | When exact balance matters |

## Complete Training Setup

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size):
    """Initialize distributed process group."""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()

def train(rank, world_size, dataset, num_epochs):
    setup(rank, world_size)
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    # Create DataLoader (no shuffle - sampler handles it)
    train_loader = DataLoader(
        dataset,
        batch_size=32,            # Per-GPU batch size
        sampler=train_sampler,    # Distributed sampler
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model and wrap with DDP
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        # CRITICAL: Set epoch for proper shuffling
        train_sampler.set_epoch(epoch)
        
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(rank, non_blocking=True)
            target = target.to(rank, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Only print on rank 0
        if rank == 0:
            print(f"Epoch {epoch} completed")
    
    cleanup()

# Launch with:
# torchrun --nproc_per_node=4 train.py
```

## Distributed Validation

Validation in distributed training requires aggregating metrics across all GPUs:

```python
def validate(rank, world_size, model, val_dataset):
    # Create validation sampler (no shuffle)
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False  # Use all validation data
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    model.eval()
    
    # Accumulate metrics locally
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(rank, non_blocking=True)
            target = target.to(rank, non_blocking=True)
            
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += data.size(0)
    
    # Aggregate across all GPUs
    metrics = torch.tensor([total_loss, total_correct, total_samples],
                          device=rank)
    
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    # Calculate final metrics
    avg_loss = metrics[0].item() / metrics[2].item()
    accuracy = metrics[1].item() / metrics[2].item()
    
    if rank == 0:
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.2%}")
    
    return avg_loss, accuracy
```

## Common Pitfalls

### Pitfall 1: Using shuffle=True with DistributedSampler

```python
# WRONG - will raise error
loader = DataLoader(dataset, shuffle=True, sampler=sampler)

# CORRECT - sampler handles shuffling
loader = DataLoader(dataset, sampler=sampler)  # No shuffle
```

### Pitfall 2: Forgetting set_epoch()

```python
# WRONG - same order every epoch
for epoch in range(epochs):
    for batch in loader:
        train(batch)

# CORRECT
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Add this!
    for batch in loader:
        train(batch)
```

### Pitfall 3: Wrong Batch Size Understanding

```python
# Per-GPU batch size, not total
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# With 4 GPUs: effective batch size = 32 × 4 = 128
# Learning rate may need adjustment!
```

### Pitfall 4: Not Aggregating Validation Metrics

```python
# WRONG - metrics only from this GPU's shard
accuracy = correct / total  # Wrong!

# CORRECT - aggregate across all GPUs
dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
accuracy = correct_tensor / total_tensor
```

### Pitfall 5: Manual Rank/World Size

```python
# WRONG - error-prone
sampler = DistributedSampler(dataset, num_replicas=4, rank=0)

# CORRECT - get from distributed runtime
sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank()
)
```

## Summary

| Aspect | Single GPU | Multi-GPU |
|--------|-----------|-----------|
| Sampler | RandomSampler | DistributedSampler |
| shuffle | shuffle=True | In sampler |
| Epoch handling | N/A | sampler.set_epoch(epoch) |
| batch_size | Total | Per-GPU |
| Validation | Direct | Aggregate with all_reduce |

## Checklist for Distributed Training

- [ ] Initialize process group with `dist.init_process_group()`
- [ ] Create DistributedSampler with correct rank and world_size
- [ ] Pass sampler to DataLoader (don't set shuffle=True)
- [ ] Call `sampler.set_epoch(epoch)` before each epoch
- [ ] Wrap model with DistributedDataParallel
- [ ] Aggregate validation metrics with `dist.all_reduce()`
- [ ] Only log/save on rank 0

## Practice Exercises

1. **Simulation**: Simulate 4-GPU training locally by creating 4 DataLoaders with different ranks and verify each sees different samples.

2. **Epoch Shuffling**: Verify that `set_epoch()` changes the order by comparing sample indices across epochs.

3. **Metric Aggregation**: Implement a distributed accuracy calculation and verify it matches single-GPU accuracy on the same dataset.

## What's Next

You've now completed the comprehensive DataLoader tutorial series. Key takeaways:

1. **Fundamentals**: Dataset → DataLoader → Batch pipeline
2. **Configuration**: Optimal parameters for training/validation
3. **Samplers**: Handle class imbalance and custom sampling
4. **Collate Functions**: Variable-length sequences and multi-modal data
5. **Multiprocessing**: Parallel data loading with workers
6. **Performance**: Profiling and optimization
7. **Distributed**: Multi-GPU training with DistributedSampler
