# Checkpointing

## Overview

Checkpointing saves the complete training state at regular intervals, enabling recovery from interruptions, training analysis, and model selection. Robust checkpointing is critical for long-running training jobs common in quantitative finance applications where models train on years of market data.

## Basic Checkpointing

```python
import torch
import torch.nn as nn
import os
import json
from datetime import datetime

def save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, path):
    """Save complete training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path, device='cpu'):
    """Load training checkpoint and resume state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint.get('metrics', {})
```

## Checkpoint Manager

```python
class CheckpointManager:
    """Manages checkpoints with rotation and best-model tracking."""
    
    def __init__(self, directory, max_checkpoints=5, metric_name='val_loss',
                 mode='min'):
        self.directory = directory
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints = []
        os.makedirs(directory, exist_ok=True)
    
    def save(self, model, optimizer, scheduler, epoch, metrics):
        """Save checkpoint with automatic rotation."""
        path = os.path.join(self.directory, f'checkpoint_epoch_{epoch:04d}.pt')
        
        save_checkpoint(model, optimizer, scheduler, epoch, 
                       metrics.get('train_loss', 0), metrics, path)
        self.checkpoints.append(path)
        
        # Track best model
        current = metrics.get(self.metric_name, 0)
        is_best = (self.mode == 'min' and current < self.best_metric) or \
                  (self.mode == 'max' and current > self.best_metric)
        
        if is_best:
            self.best_metric = current
            best_path = os.path.join(self.directory, 'best_model.pt')
            torch.save(model.state_dict(), best_path)
        
        # Rotate old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old = self.checkpoints.pop(0)
            if os.path.exists(old):
                os.remove(old)
        
        return is_best
    
    def load_best(self, model):
        """Load the best model."""
        best_path = os.path.join(self.directory, 'best_model.pt')
        model.load_state_dict(
            torch.load(best_path, weights_only=True)
        )
        return model
```

## Gradient Checkpointing (Memory Optimization)

Gradient checkpointing trades compute for memory by not storing all intermediate activations during forward pass, instead recomputing them during backward:

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    """Uses gradient checkpointing for large models."""
    
    def __init__(self, num_layers=12, hidden_size=512):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            # Recompute activations during backward pass
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

**Memory savings**: Reduces activation memory from $O(L)$ to $O(\sqrt{L})$ for $L$ layers, at the cost of ~33% more computation.

## Training Loop with Checkpointing

```python
def training_loop(model, train_loader, val_loader, epochs, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    ckpt_manager = CheckpointManager(
        'checkpoints/', max_checkpoints=5,
        metric_name='val_loss', mode='min'
    )
    
    # Resume from latest checkpoint if available
    start_epoch = 0
    latest = sorted([f for f in os.listdir('checkpoints/') 
                     if f.startswith('checkpoint_')]) if os.path.exists('checkpoints/') else []
    if latest:
        start_epoch, _, _ = load_checkpoint(
            model, optimizer, scheduler,
            os.path.join('checkpoints/', latest[-1]), device
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                val_loss += criterion(model(data), target).item()
        
        scheduler.step()
        
        # Checkpoint
        metrics = {
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
        }
        is_best = ckpt_manager.save(model, optimizer, scheduler, epoch, metrics)
        
        status = " ★ Best" if is_best else ""
        print(f"Epoch {epoch}: train_loss={metrics['train_loss']:.4f}, "
              f"val_loss={metrics['val_loss']:.4f}{status}")
```

## Best Practices

- **Checkpoint frequently** for long-running jobs—at least every epoch, more often for multi-day training
- **Rotate checkpoints** to avoid filling disk; keep last N plus the best
- **Save optimizer and scheduler state** to ensure exact training resumption
- **Use gradient checkpointing** for memory-constrained training of large models
- **Test checkpoint recovery** before starting long training runs
- **Include metadata** (git hash, data version, config) for full reproducibility

## References

1. PyTorch Checkpointing Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
2. Gradient Checkpointing: https://pytorch.org/docs/stable/checkpoint.html
