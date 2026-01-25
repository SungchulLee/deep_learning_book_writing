# Distributed Training Checkpoints

## Overview

Training neural networks across multiple GPUs or nodes introduces unique challenges for model saving and loading. The **`module.` prefix** issue, device synchronization, and checkpoint consistency across processes require careful handling. This section covers best practices for checkpointing in DataParallel (DP) and DistributedDataParallel (DDP) settings.

## Learning Objectives

By the end of this section, you will be able to:

- Handle the `module.` prefix when saving/loading distributed models
- Implement checkpoint saving that works across GPU configurations
- Create portable checkpoints that load on any number of GPUs
- Synchronize checkpoint operations in multi-process training
- Resume distributed training from checkpoints correctly

## Understanding Distributed Training Wrappers

### DataParallel vs DistributedDataParallel

| Feature | DataParallel (DP) | DistributedDataParallel (DDP) |
|---------|------------------|-------------------------------|
| Process Model | Single process, multiple threads | Multiple processes |
| Communication | Implicit via Python threading | Explicit via NCCL/Gloo |
| State Dict | Has `module.` prefix | Has `module.` prefix |
| Scalability | Limited by GIL | Scales to multiple nodes |
| Checkpoint Saving | Any GPU can save | Typically rank 0 only |

### The `module.` Prefix Problem

When a model is wrapped with `DataParallel` or `DistributedDataParallel`, the state dict keys are prefixed with `module.`:

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

# Original model
model = SimpleModel()
print("Original state dict keys:")
for key in model.state_dict().keys():
    print(f"  {key}")

# Wrapped model
model_dp = nn.DataParallel(model, device_ids=[0])
print("\nDataParallel state dict keys:")
for key in model_dp.state_dict().keys():
    print(f"  {key}")
```

**Output:**
```
Original state dict keys:
  fc.weight
  fc.bias

DataParallel state dict keys:
  module.fc.weight
  module.fc.bias
```

## Prefix Management Utilities

### Remove `module.` Prefix

```python
from collections import OrderedDict
from typing import Dict
import torch


def remove_module_prefix(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    """
    Remove 'module.' prefix from state dict keys.
    
    Use when: Loading a checkpoint saved from DataParallel/DDP 
    into an unwrapped model.
    
    Args:
        state_dict: State dictionary with 'module.' prefix
    
    Returns:
        State dictionary without prefix
    """
    new_state_dict = OrderedDict()
    
    for key, value in state_dict.items():
        # Remove 'module.' prefix (7 characters)
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value
    
    return new_state_dict


def add_module_prefix(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    """
    Add 'module.' prefix to state dict keys.
    
    Use when: Loading a checkpoint saved from unwrapped model
    into a DataParallel/DDP wrapped model.
    
    Args:
        state_dict: State dictionary without 'module.' prefix
    
    Returns:
        State dictionary with prefix
    """
    new_state_dict = OrderedDict()
    
    for key, value in state_dict.items():
        new_key = f'module.{key}' if not key.startswith('module.') else key
        new_state_dict[new_key] = value
    
    return new_state_dict


def normalize_state_dict(
    state_dict: Dict[str, torch.Tensor],
    remove_prefix: bool = True
) -> OrderedDict:
    """
    Normalize state dict by optionally removing 'module.' prefix.
    
    Args:
        state_dict: Input state dictionary
        remove_prefix: If True, remove prefix; if False, keep as-is
    
    Returns:
        Normalized state dictionary
    """
    if remove_prefix:
        return remove_module_prefix(state_dict)
    return OrderedDict(state_dict)
```

## DataParallel Checkpointing

### Saving DataParallel Models

```python
import torch
import torch.nn as nn
from pathlib import Path


def save_dataparallel_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str,
    save_portable: bool = True,
    **extra_state
) -> None:
    """
    Save checkpoint from DataParallel training.
    
    Args:
        model: Model (may or may not be wrapped with DataParallel)
        optimizer: Optimizer instance
        epoch: Current epoch
        filepath: Save path
        save_portable: If True, save without 'module.' prefix for portability
        **extra_state: Additional state to save
    """
    # Get underlying model state dict
    if hasattr(model, 'module'):
        # Model is wrapped - get underlying model
        model_state = model.module.state_dict() if save_portable else model.state_dict()
        is_wrapped = True
    else:
        # Model is not wrapped
        model_state = model.state_dict()
        is_wrapped = False
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'is_wrapped': is_wrapped,
        'save_portable': save_portable,
        **extra_state
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")
    print(f"  Portable format: {save_portable}")


def load_dataparallel_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: torch.device,
    use_dataparallel: bool = False,
    device_ids: list = None
) -> tuple:
    """
    Load checkpoint into model, optionally wrapping with DataParallel.
    
    Args:
        model: Base model (unwrapped)
        optimizer: Optimizer instance
        filepath: Checkpoint path
        device: Target device
        use_dataparallel: Whether to wrap model with DataParallel
        device_ids: GPU device IDs for DataParallel
    
    Returns:
        Tuple of (model, epoch)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Get state dict
    state_dict = checkpoint['model_state_dict']
    
    # Check if we need to adjust for prefix mismatch
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    # Detect prefix situation
    has_module_prefix = any(k.startswith('module.') for k in ckpt_keys)
    
    if has_module_prefix:
        # Checkpoint has prefix, model doesn't - remove prefix
        state_dict = remove_module_prefix(state_dict)
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Optionally wrap with DataParallel
    if use_dataparallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"Model wrapped with DataParallel on devices: {device_ids or 'all'}")
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return model, epoch
```

## DistributedDataParallel Checkpointing

### DDP Checkpoint Manager

```python
import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any


class DDPCheckpointManager:
    """
    Checkpoint manager for DistributedDataParallel training.
    
    Key features:
    - Only rank 0 saves to avoid file conflicts
    - Synchronization barriers for consistency
    - Portable checkpoint format
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        rank: int,
        world_size: int
    ):
        """
        Args:
            checkpoint_dir: Directory for checkpoints
            rank: Process rank
            world_size: Total number of processes
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.rank = rank
        self.world_size = world_size
        
        # Only rank 0 creates directory
        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Synchronize after directory creation
        if dist.is_initialized():
            dist.barrier()
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float] = None,
        scheduler: Any = None,
        is_best: bool = False,
        filename: str = None
    ) -> Optional[str]:
        """
        Save checkpoint (only on rank 0).
        
        Args:
            model: DDP-wrapped model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Validation metrics
            scheduler: Learning rate scheduler
            is_best: Whether this is the best model
            filename: Custom filename (default: checkpoint_epochN.pt)
        
        Returns:
            Path to saved checkpoint (on rank 0) or None (other ranks)
        """
        # Synchronize before saving
        if dist.is_initialized():
            dist.barrier()
        
        # Only rank 0 saves
        if self.rank != 0:
            return None
        
        # Get state dict without 'module.' prefix for portability
        if hasattr(model, 'module'):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {},
            'world_size': self.world_size,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Generate filename
        if filename is None:
            filename = f'checkpoint_epoch{epoch:04d}.pt'
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"[Rank 0] Saved best model")
        
        print(f"[Rank 0] Checkpoint saved: {filepath}")
        
        # Synchronize after saving
        if dist.is_initialized():
            dist.barrier()
        
        return str(filepath)
    
    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        filepath: str = None,
        load_best: bool = False,
        scheduler: Any = None,
        map_location: str = None
    ) -> tuple:
        """
        Load checkpoint on all ranks.
        
        Args:
            model: Model (wrapped or unwrapped)
            optimizer: Optimizer
            filepath: Specific checkpoint path
            load_best: If True, load best_model.pt
            scheduler: Learning rate scheduler
            map_location: Device mapping
        
        Returns:
            Tuple of (epoch, metrics)
        """
        # Determine checkpoint path
        if load_best:
            filepath = self.checkpoint_dir / 'best_model.pt'
        elif filepath is None:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch*.pt'))
            if not checkpoints:
                print(f"[Rank {self.rank}] No checkpoints found")
                return 0, {}
            filepath = checkpoints[-1]
        
        # Determine map location
        if map_location is None:
            if torch.cuda.is_available():
                map_location = f'cuda:{self.rank}'
            else:
                map_location = 'cpu'
        
        # All ranks load checkpoint
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Load model state
        state_dict = checkpoint['model_state_dict']
        
        # Handle DDP wrapper
        if hasattr(model, 'module'):
            # Model is wrapped - load into underlying model
            model.module.load_state_dict(state_dict)
        else:
            # Check for prefix mismatch
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = remove_module_prefix(state_dict)
            model.load_state_dict(state_dict)
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        print(f"[Rank {self.rank}] Loaded checkpoint from epoch {epoch}")
        
        # Synchronize after loading
        if dist.is_initialized():
            dist.barrier()
        
        return epoch, metrics
```

### Complete DDP Training Example

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os


def setup_ddp(rank: int, world_size: int):
    """Initialize distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def train_ddp(
    rank: int,
    world_size: int,
    num_epochs: int,
    resume_from: str = None
):
    """
    DDP training function to run on each process.
    
    Args:
        rank: Process rank
        world_size: Total processes
        num_epochs: Training epochs
        resume_from: Checkpoint path to resume from
    """
    # Setup
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # Create model
    model = SimpleModel().to(device)
    model = DDP(model, device_ids=[rank])
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Checkpoint manager
    ckpt_manager = DDPCheckpointManager(
        checkpoint_dir='ddp_checkpoints',
        rank=rank,
        world_size=world_size
    )
    
    # Resume if specified
    start_epoch = 0
    if resume_from:
        start_epoch, _ = ckpt_manager.load(
            model, optimizer, filepath=resume_from
        )
        start_epoch += 1  # Start from next epoch
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # ... training code ...
        
        # Save checkpoint
        is_best = False  # Determine based on validation
        ckpt_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={'train_loss': 0.1},  # Replace with actual
            is_best=is_best
        )
    
    cleanup_ddp()


# Launch DDP training
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train_ddp,
        args=(world_size, 10, None),
        nprocs=world_size,
        join=True
    )
```

## Cross-Configuration Loading

### Universal Checkpoint Loader

```python
def load_checkpoint_universal(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device = None,
    strict: bool = True
) -> nn.Module:
    """
    Load checkpoint regardless of how it was saved (DP/DDP/single GPU).
    
    Args:
        model: Target model (unwrapped)
        checkpoint_path: Path to checkpoint
        device: Target device
        strict: Strict loading mode
    
    Returns:
        Loaded model
    """
    device = device or torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Check for prefix
    sample_key = next(iter(state_dict.keys()))
    has_prefix = sample_key.startswith('module.')
    
    # Check model expects prefix
    model_sample_key = next(iter(model.state_dict().keys()))
    model_has_prefix = model_sample_key.startswith('module.')
    
    # Adjust if mismatch
    if has_prefix and not model_has_prefix:
        state_dict = remove_module_prefix(state_dict)
        print("Removed 'module.' prefix from checkpoint")
    elif not has_prefix and model_has_prefix:
        state_dict = add_module_prefix(state_dict)
        print("Added 'module.' prefix to checkpoint")
    
    # Load
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    
    return model
```

## Best Practices

### Checkpoint Portability Guidelines

1. **Always save without `module.` prefix** for maximum portability:
   ```python
   state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
   ```

2. **Document the save configuration** in checkpoint metadata:
   ```python
   checkpoint = {
       'model_state_dict': state_dict,
       'saved_with_wrapper': hasattr(model, 'module'),
       'world_size': world_size,
   }
   ```

3. **Use synchronization barriers** in DDP to prevent race conditions

4. **Save only on rank 0** to avoid file conflicts and redundancy

5. **Load on all ranks** when resuming distributed training

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `Missing keys: ['module.fc.weight']` | Loading prefixed checkpoint into unwrapped model | Use `remove_module_prefix()` |
| `Unexpected keys: ['fc.weight']` | Loading unprefixed checkpoint into wrapped model | Use `add_module_prefix()` |
| Corrupt checkpoint | Race condition during save | Use `dist.barrier()` before/after save |
| OOM when loading | Loading to wrong GPU | Use `map_location=f'cuda:{rank}'` |
| Different results after resume | RNG state not restored | Save and load random states |

## Summary

Distributed training checkpointing requires:

1. **Prefix handling**: Manage `module.` prefix for portability
2. **Rank coordination**: Only rank 0 saves, all ranks load
3. **Synchronization**: Use barriers to prevent race conditions
4. **Universal loading**: Handle any checkpoint format gracefully

Following these patterns ensures checkpoints work seamlessly across different GPU configurations and training setups.

## References

- [PyTorch Distributed Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [DistributedDataParallel Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)
