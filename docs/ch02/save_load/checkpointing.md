# Checkpointing

## Overview

**Checkpointing** is the practice of periodically saving the complete training state during model training, enabling recovery from interruptions and systematic tracking of model evolution. Beyond simple model persistence, effective checkpointing encompasses training state management, best model tracking, automatic cleanup, and metadata logging for experiment reproducibility.

## Learning Objectives

By the end of this section, you will be able to:

- Design comprehensive checkpoint schemas that capture full training state
- Implement checkpoint managers with automatic best-model tracking
- Configure retention policies for efficient storage management
- Resume training seamlessly from any checkpoint
- Version and organize checkpoints for experiment tracking

## Mathematical Motivation

### Training State as a Dynamical System

Training can be viewed as a discrete dynamical system with state $\mathbf{S}_t$ at iteration $t$:

$$\mathbf{S}_t = (\theta_t, \mathbf{m}_t, \mathbf{v}_t, t, \text{seed}_t)$$

where:
- $\theta_t$ are model parameters
- $\mathbf{m}_t$ are momentum accumulators (e.g., first moment in Adam)
- $\mathbf{v}_t$ are velocity accumulators (e.g., second moment in Adam)
- $t$ is the iteration count
- $\text{seed}_t$ captures random state for reproducibility

The update rule $\mathbf{S}_{t+1} = \Phi(\mathbf{S}_t, \mathcal{B}_t)$ depends on both current state and mini-batch $\mathcal{B}_t$. Checkpointing captures $\mathbf{S}_t$ at regular intervals, enabling exact trajectory continuation.

### Checkpoint Interval Analysis

The optimal checkpoint frequency balances I/O overhead against recovery cost. If training takes $T$ total time and checkpoint writing takes $c$ time, with failure probability $p$ per interval:

$$\text{Expected Overhead} = \frac{T}{\Delta} \cdot c$$

$$\text{Expected Recovery Cost} = p \cdot T \cdot \frac{\Delta}{2T} = \frac{p \cdot \Delta}{2}$$

Minimizing total expected cost yields optimal interval:

$$\Delta^* = \sqrt{\frac{2cT}{p}}$$

For practical purposes, checkpointing every epoch or every $N$ steps (where $N$ balances these concerns) is standard.

## Checkpoint Schema Design

### Minimal Checkpoint

```python
# Bare minimum for training resumption
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}
```

### Comprehensive Checkpoint

```python
from datetime import datetime
import torch

def create_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    global_step: int,
    train_loss: float,
    val_loss: float,
    val_metrics: dict,
    config: dict,
    rng_states: dict = None
) -> dict:
    """
    Create a comprehensive training checkpoint.
    
    Args:
        model: The neural network model
        optimizer: The optimizer instance
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        global_step: Total training steps completed
        train_loss: Training loss for current epoch
        val_loss: Validation loss for current epoch
        val_metrics: Dictionary of validation metrics
        config: Training configuration/hyperparameters
        rng_states: Optional random number generator states
    
    Returns:
        Complete checkpoint dictionary
    """
    
    checkpoint = {
        # Core training state
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        
        # Training progress
        'epoch': epoch,
        'global_step': global_step,
        
        # Metrics
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_metrics': val_metrics,
        
        # Configuration for reproducibility
        'config': config,
        
        # Metadata
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
    }
    
    # Optional: Random states for exact reproducibility
    if rng_states is None:
        rng_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states['cuda'] = torch.cuda.get_rng_state_all()
    
    checkpoint['rng_states'] = rng_states
    
    return checkpoint
```

## Checkpoint Manager Implementation

### Full-Featured CheckpointManager Class

```python
import os
import glob
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class CheckpointInfo:
    """Metadata container for a checkpoint."""
    filepath: Path
    epoch: int
    global_step: int
    val_loss: float
    val_metrics: Dict[str, float]
    timestamp: str
    is_best: bool = False


class CheckpointManager:
    """
    Production-grade checkpoint management system.
    
    Features:
    - Automatic checkpoint saving with configurable frequency
    - Best model tracking based on monitored metric
    - Automatic cleanup with configurable retention policy
    - Metadata logging and checkpoint inspection
    - Seamless training resumption
    
    Args:
        checkpoint_dir: Directory for storing checkpoints
        max_checkpoints: Maximum regular checkpoints to retain
        monitor_metric: Metric name to track for best model selection
        monitor_mode: 'min' for loss-like metrics, 'max' for accuracy-like
        save_best_only: If True, only save when metric improves
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        max_checkpoints: int = 5,
        monitor_metric: str = 'val_loss',
        monitor_mode: str = 'min',
        save_best_only: bool = False
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.save_best_only = save_best_only
        
        # Track best score
        self.best_score = float('inf') if monitor_mode == 'min' else float('-inf')
        self.best_checkpoint_path: Optional[Path] = None
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint registry
        self._registry: List[CheckpointInfo] = []
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load existing checkpoints into registry."""
        registry_path = self.checkpoint_dir / 'registry.json'
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                data = json.load(f)
                self._registry = [
                    CheckpointInfo(**info) for info in data['checkpoints']
                ]
                self.best_score = data.get('best_score', self.best_score)
                best_path = data.get('best_checkpoint_path')
                self.best_checkpoint_path = Path(best_path) if best_path else None
    
    def _save_registry(self) -> None:
        """Persist registry to disk."""
        registry_path = self.checkpoint_dir / 'registry.json'
        data = {
            'checkpoints': [
                {
                    'filepath': str(info.filepath),
                    'epoch': info.epoch,
                    'global_step': info.global_step,
                    'val_loss': info.val_loss,
                    'val_metrics': info.val_metrics,
                    'timestamp': info.timestamp,
                    'is_best': info.is_best,
                }
                for info in self._registry
            ],
            'best_score': self.best_score,
            'best_checkpoint_path': str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
        }
        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _is_improvement(self, current_score: float) -> bool:
        """Check if current score is better than best score."""
        if self.monitor_mode == 'min':
            return current_score < self.best_score
        return current_score > self.best_score
    
    def _get_checkpoint_filename(self, epoch: int, global_step: int, val_loss: float) -> str:
        """Generate descriptive checkpoint filename."""
        return f"checkpoint_epoch{epoch:04d}_step{global_step:08d}_loss{val_loss:.4f}.pt"
    
    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        global_step: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Optional[Dict[str, float]] = None,
        scheduler: Optional[Any] = None,
        config: Optional[Dict] = None,
        extra_state: Optional[Dict] = None
    ) -> Optional[Path]:
        """
        Save a checkpoint.
        
        Returns:
            Path to saved checkpoint, or None if save_best_only and not improved
        """
        val_metrics = val_metrics or {}
        
        # Get the metric to monitor
        if self.monitor_metric == 'val_loss':
            current_score = val_loss
        else:
            current_score = val_metrics.get(self.monitor_metric, val_loss)
        
        is_improvement = self._is_improvement(current_score)
        
        # Check if we should save
        if self.save_best_only and not is_improvement:
            return None
        
        # Create checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': config or {},
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if extra_state is not None:
            checkpoint['extra_state'] = extra_state
        
        # Generate filename and save
        filename = self._get_checkpoint_filename(epoch, global_step, val_loss)
        filepath = self.checkpoint_dir / filename
        
        torch.save(checkpoint, filepath)
        
        # Create registry entry
        info = CheckpointInfo(
            filepath=filepath,
            epoch=epoch,
            global_step=global_step,
            val_loss=val_loss,
            val_metrics=val_metrics,
            timestamp=checkpoint['timestamp'],
            is_best=is_improvement
        )
        self._registry.append(info)
        
        # Handle best model
        if is_improvement:
            self.best_score = current_score
            best_path = self.checkpoint_dir / 'best_model.pt'
            shutil.copy(filepath, best_path)
            self.best_checkpoint_path = best_path
            print(f"✓ New best model! {self.monitor_metric}: {current_score:.6f}")
        
        # Cleanup old checkpoints
        self._cleanup()
        
        # Save registry
        self._save_registry()
        
        print(f"Checkpoint saved: {filename}")
        return filepath
    
    def _cleanup(self) -> None:
        """Remove old checkpoints exceeding retention limit."""
        # Sort by epoch (oldest first)
        regular_checkpoints = [
            info for info in self._registry 
            if not info.is_best and info.filepath.exists()
        ]
        regular_checkpoints.sort(key=lambda x: x.epoch)
        
        # Remove excess checkpoints
        while len(regular_checkpoints) > self.max_checkpoints:
            oldest = regular_checkpoints.pop(0)
            if oldest.filepath.exists():
                oldest.filepath.unlink()
                print(f"Removed old checkpoint: {oldest.filepath.name}")
            self._registry.remove(oldest)
    
    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[int, Dict]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Tuple of (epoch, checkpoint_data)
        """
        if not self._registry:
            print("No checkpoints found")
            return 0, {}
        
        # Find most recent
        latest = max(self._registry, key=lambda x: x.epoch)
        return self._load_checkpoint(latest.filepath, model, optimizer, scheduler, device)
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[int, Dict]:
        """
        Load the best checkpoint based on monitored metric.
        
        Returns:
            Tuple of (epoch, checkpoint_data)
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        if not best_path.exists():
            print("No best model checkpoint found")
            return 0, {}
        
        return self._load_checkpoint(best_path, model, optimizer, scheduler, device)
    
    def _load_checkpoint(
        self,
        filepath: Path,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
        scheduler: Optional[Any],
        device: Optional[torch.device]
    ) -> Tuple[int, Dict]:
        """Internal method to load a checkpoint."""
        map_location = device if device else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Move model to device
        if device is not None:
            model.to(device)
        
        epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {epoch}")
        print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
        
        return epoch, checkpoint
    
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all available checkpoints with metadata."""
        print(f"\n{'='*70}")
        print(f"Available Checkpoints ({len(self._registry)} total)")
        print(f"{'='*70}")
        
        for info in sorted(self._registry, key=lambda x: x.epoch):
            status = "★ BEST" if info.is_best else ""
            print(f"\nEpoch {info.epoch:4d} | Step {info.global_step:8d} | "
                  f"Val Loss: {info.val_loss:.6f} {status}")
            print(f"  Path: {info.filepath.name}")
            print(f"  Time: {info.timestamp}")
            if info.val_metrics:
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in info.val_metrics.items())
                print(f"  Metrics: {metrics_str}")
        
        return self._registry
```

## Training Loop Integration

### Complete Training Example with Checkpointing

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_with_checkpointing(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    num_epochs: int,
    checkpoint_manager: CheckpointManager,
    device: torch.device,
    config: dict,
    resume_from: str = None
) -> nn.Module:
    """
    Training loop with comprehensive checkpointing.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        criterion: Loss function
        num_epochs: Total epochs to train
        checkpoint_manager: CheckpointManager instance
        device: Training device
        config: Training configuration
        resume_from: Path to checkpoint to resume from
    
    Returns:
        Trained model
    """
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    
    if resume_from:
        if resume_from == 'latest':
            start_epoch, ckpt = checkpoint_manager.load_latest(
                model, optimizer, scheduler, device
            )
        elif resume_from == 'best':
            start_epoch, ckpt = checkpoint_manager.load_best(
                model, optimizer, scheduler, device
            )
        else:
            start_epoch, ckpt = checkpoint_manager._load_checkpoint(
                Path(resume_from), model, optimizer, scheduler, device
            )
        global_step = ckpt.get('global_step', 0)
        print(f"Resuming from epoch {start_epoch}, step {global_step}")
    
    model.to(device)
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            global_step += 1
        
        train_loss /= num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step(val_loss)  # For ReduceLROnPlateau
            # scheduler.step()  # For other schedulers
        
        # Log progress
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val Acc:    {val_accuracy:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics={'accuracy': val_accuracy},
            scheduler=scheduler,
            config=config
        )
    
    return model
```

## Advanced Checkpointing Patterns

### Gradient Checkpointing for Memory Efficiency

For very large models, gradient checkpointing trades compute for memory by not storing intermediate activations:

```python
from torch.utils.checkpoint import checkpoint_sequential

class LargeModel(nn.Module):
    def __init__(self, num_layers: int = 24):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=512, nhead=8)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, use_checkpoint: bool = True) -> torch.Tensor:
        if use_checkpoint and self.training:
            # Checkpoint every 4 layers
            segments = 4
            return checkpoint_sequential(self.layers, segments, x)
        return self.layers(x)
```

!!! note "Gradient Checkpointing vs Training Checkpoints"
    These are different concepts: **Gradient checkpointing** reduces memory during forward/backward by recomputing activations. **Training checkpoints** save model state to disk for recovery. The former is a memory optimization technique; the latter is for fault tolerance.

### Asynchronous Checkpoint Saving

For large models where checkpoint I/O is slow:

```python
import threading
from queue import Queue


class AsyncCheckpointSaver:
    """Asynchronous checkpoint saving to avoid blocking training."""
    
    def __init__(self, num_workers: int = 1):
        self.queue = Queue()
        self.workers = []
        
        for _ in range(num_workers):
            worker = threading.Thread(target=self._save_worker, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _save_worker(self):
        """Worker thread that processes save queue."""
        while True:
            checkpoint, filepath = self.queue.get()
            try:
                torch.save(checkpoint, filepath)
                print(f"[Async] Saved: {filepath}")
            except Exception as e:
                print(f"[Async] Error saving {filepath}: {e}")
            finally:
                self.queue.task_done()
    
    def save(self, checkpoint: dict, filepath: str):
        """Queue a checkpoint for async saving."""
        # Deep copy tensors to CPU to avoid race conditions
        checkpoint_copy = {
            k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
            for k, v in checkpoint.items()
        }
        self.queue.put((checkpoint_copy, filepath))
    
    def wait(self):
        """Wait for all pending saves to complete."""
        self.queue.join()
```

### Distributed Training Checkpoints

When training with multiple GPUs/nodes:

```python
import torch.distributed as dist


def save_distributed_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    filepath: str,
    rank: int = 0
) -> None:
    """
    Save checkpoint in distributed training.
    Only rank 0 saves to avoid file conflicts.
    """
    # Synchronize all processes
    if dist.is_initialized():
        dist.barrier()
    
    # Only rank 0 saves
    if rank == 0:
        # Handle DistributedDataParallel wrapper
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }
        
        torch.save(checkpoint, filepath)
        print(f"[Rank 0] Checkpoint saved: {filepath}")
    
    # Synchronize again after save
    if dist.is_initialized():
        dist.barrier()


def load_distributed_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    filepath: str,
    device: torch.device
) -> int:
    """
    Load checkpoint for distributed training.
    All ranks load from the same file.
    """
    # All ranks load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Handle DistributedDataParallel wrapper
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch']
```

## Checkpoint File Organization

### Recommended Directory Structure

```
experiments/
└── experiment_001/
    ├── checkpoints/
    │   ├── checkpoint_epoch0001_step00001000_loss0.4523.pt
    │   ├── checkpoint_epoch0002_step00002000_loss0.3821.pt
    │   ├── checkpoint_epoch0003_step00003000_loss0.3156.pt
    │   ├── best_model.pt
    │   └── registry.json
    ├── logs/
    │   ├── train.log
    │   └── tensorboard/
    ├── config.yaml
    └── README.md
```

### Naming Conventions

| Component | Format | Example |
|-----------|--------|---------|
| Epoch | `epoch{NNNN}` | `epoch0042` |
| Step | `step{NNNNNNNN}` | `step00012500` |
| Metric | `{metric}{value:.4f}` | `loss0.1234` |
| Timestamp | ISO 8601 | `2024-01-15T10:30:00` |

## Best Practices

### Do's

- ✅ Save complete training state (model, optimizer, scheduler, epoch)
- ✅ Use descriptive filenames with epoch/step/metric information
- ✅ Track best model separately for easy deployment
- ✅ Include configuration and metadata for reproducibility
- ✅ Implement automatic cleanup to manage disk space
- ✅ Validate checkpoints after saving (verify loadability)
- ✅ Use CPU mapping when loading to avoid GPU memory issues

### Don'ts

- ❌ Save only model weights if you need to resume training
- ❌ Overwrite checkpoints without keeping history
- ❌ Store checkpoints on slow networked file systems during training
- ❌ Forget to save random number generator states for exact reproducibility
- ❌ Assume same device configuration when loading

## Summary

Effective checkpointing is essential for:

1. **Fault tolerance**: Recover from crashes without losing training progress
2. **Experiment management**: Track model evolution and compare versions
3. **Model selection**: Automatically identify and preserve best-performing models
4. **Reproducibility**: Enable exact training continuation with complete state capture

The `CheckpointManager` pattern provides a production-ready solution for managing checkpoints throughout the training lifecycle.

## References

- [PyTorch Checkpoint Tutorial](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [Distributed Training Best Practices](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
