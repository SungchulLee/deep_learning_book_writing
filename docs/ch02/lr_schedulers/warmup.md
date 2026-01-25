# Warmup Strategies

## Overview

Learning rate warmup is a technique where training begins with a small learning rate that gradually increases to the target value over an initial period. This approach has become essential for training large models, particularly transformers, and is widely used when working with large batch sizes.

## Why Warmup?

### The Problem with Cold Starts

When training begins, the network weights are randomly initialized. Starting with a high learning rate can cause:

1. **Gradient explosion** - Large updates destabilize training
2. **Loss spikes** - Sudden increases in loss that may not recover
3. **Poor convergence** - Model converges to suboptimal solutions
4. **Batch norm instability** - Running statistics are unreliable early

### The Solution: Gradual Warmup

By starting with a small learning rate and gradually increasing it:

1. **Stable early training** - Small updates prevent instability
2. **Running statistics converge** - Batch norm statistics become reliable
3. **Better feature learning** - Model learns good representations
4. **Improved final performance** - Often better than no warmup

## Mathematical Formulations

### Linear Warmup

The most common warmup strategy, linearly increases LR from 0 to target:

$$\eta_t = \eta_{target} \cdot \frac{t + 1}{T_{warmup}}$$

where $t$ is the current step and $T_{warmup}$ is the total warmup steps.

### Exponential Warmup

Increases LR exponentially from a small initial value:

$$\eta_t = \eta_{start} \cdot \left(\frac{\eta_{target}}{\eta_{start}}\right)^{t / T_{warmup}}$$

where $\eta_{start}$ is typically very small (e.g., $10^{-7}$).

### Cosine Warmup

Follows a smooth cosine curve for gradual acceleration:

$$\eta_t = \eta_{target} \cdot \frac{1 - \cos\left(\frac{t \cdot \pi}{T_{warmup}}\right)}{2}$$

### Polynomial Warmup

General form with adjustable power:

$$\eta_t = \eta_{target} \cdot \left(\frac{t + 1}{T_{warmup}}\right)^p$$

where $p$ controls the curve shape (p=1 is linear, p<1 is faster, p>1 is slower).

## Warmup Curves Comparison

```
LR
 │
 │                    ─────── target
 │              __---
 │          __/
 │       _/
 │     _/    ← linear
 │    /
 │  _/
 │_/         ← cosine (slower start)
 └─────────────────────→ step
   0      T_warmup
```

## PyTorch Implementations

### Linear Warmup

```python
class LinearWarmup:
    """
    Linear learning rate warmup.
    
    LR increases linearly from 0 (or start_lr) to base_lr.
    """
    
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        start_lr: float = 0.0
    ):
        """
        Args:
            base_lr: Target learning rate after warmup
            warmup_steps: Number of warmup steps
            start_lr: Initial learning rate (default: 0)
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.current_step = 0
    
    def get_lr(self, step: int = None) -> float:
        """Get learning rate for given step."""
        if step is None:
            step = self.current_step
        
        if step >= self.warmup_steps:
            return self.base_lr
        
        # Linear interpolation
        progress = (step + 1) / self.warmup_steps
        return self.start_lr + progress * (self.base_lr - self.start_lr)
    
    def step(self):
        """Advance one step."""
        self.current_step += 1
        return self.get_lr()
```

### Exponential Warmup

```python
import math

class ExponentialWarmup:
    """
    Exponential learning rate warmup.
    
    LR increases exponentially from start_lr to base_lr.
    Slower initial increase, useful for sensitive models.
    """
    
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        start_lr: float = 1e-7
    ):
        """
        Args:
            base_lr: Target learning rate after warmup
            warmup_steps: Number of warmup steps
            start_lr: Initial learning rate (should be very small)
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.current_step = 0
        
        # Compute per-step multiplier
        self.lr_ratio = base_lr / start_lr
    
    def get_lr(self, step: int = None) -> float:
        """Get learning rate for given step."""
        if step is None:
            step = self.current_step
        
        if step >= self.warmup_steps:
            return self.base_lr
        
        # Exponential interpolation
        progress = (step + 1) / self.warmup_steps
        return self.start_lr * (self.lr_ratio ** progress)
    
    def step(self):
        """Advance one step."""
        self.current_step += 1
        return self.get_lr()
```

### Cosine Warmup

```python
class CosineWarmup:
    """
    Cosine learning rate warmup.
    
    LR follows cosine curve for smooth acceleration.
    Slower at start and end, faster in middle.
    """
    
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int
    ):
        """
        Args:
            base_lr: Target learning rate after warmup
            warmup_steps: Number of warmup steps
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def get_lr(self, step: int = None) -> float:
        """Get learning rate for given step."""
        if step is None:
            step = self.current_step
        
        if step >= self.warmup_steps:
            return self.base_lr
        
        # Cosine interpolation (0 to base_lr)
        progress = (step + 1) / self.warmup_steps
        return self.base_lr * (1 - math.cos(progress * math.pi)) / 2
    
    def step(self):
        """Advance one step."""
        self.current_step += 1
        return self.get_lr()
```

### Warmup + Decay Combined

The most common pattern in transformer training:

```python
class WarmupWithDecay:
    """
    Complete schedule: Linear warmup followed by cosine decay.
    
    This is the standard schedule for transformer training
    (BERT, GPT, etc.).
    """
    
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        """
        Args:
            base_lr: Peak learning rate after warmup
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum LR at end of training
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
    
    def get_lr(self, step: int = None) -> float:
        """Get learning rate for given step."""
        if step is None:
            step = self.current_step
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step + 1) / self.warmup_steps
        
        # Cosine decay after warmup
        decay_steps = self.total_steps - self.warmup_steps
        decay_progress = (step - self.warmup_steps) / decay_steps
        decay_progress = min(decay_progress, 1.0)
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
    
    def step(self):
        """Advance one step."""
        self.current_step += 1
        return self.get_lr()


class WarmupLinearDecay:
    """
    Linear warmup followed by linear decay.
    Alternative to cosine for some applications.
    """
    
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
    
    def get_lr(self, step: int = None) -> float:
        if step is None:
            step = self.current_step
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step + 1) / self.warmup_steps
        
        # Linear decay
        decay_steps = self.total_steps - self.warmup_steps
        decay_progress = (step - self.warmup_steps) / decay_steps
        decay_progress = min(decay_progress, 1.0)
        
        return self.base_lr - decay_progress * (self.base_lr - self.min_lr)
    
    def step(self):
        self.current_step += 1
        return self.get_lr()
```

## Using with PyTorch Optimizers

### Pattern 1: Manual LR Updates

```python
import torch.optim as optim

# Create warmup scheduler
warmup = LinearWarmup(base_lr=1e-3, warmup_steps=1000)

# Create optimizer with initial LR
optimizer = optim.Adam(model.parameters(), lr=0)

# Training loop
for step, batch in enumerate(dataloader):
    # Update learning rate
    lr = warmup.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Training step
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### Pattern 2: Combining with PyTorch Scheduler

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

# Warmup + PyTorch scheduler
warmup_scheduler = LinearWarmup(base_lr=1e-3, warmup_steps=1000)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)

# Training loop
global_step = 0
for epoch in range(total_epochs):
    for batch in dataloader:
        if global_step < warmup_steps:
            # Warmup phase
            lr = warmup_scheduler.get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Train step
        train_step(batch)
        global_step += 1
    
    # After warmup, step cosine scheduler per epoch
    if epoch >= warmup_epochs:
        cosine_scheduler.step()
```

### Pattern 3: Using LambdaLR

```python
from torch.optim.lr_scheduler import LambdaLR

def warmup_lambda(step, warmup_steps=1000, base_lr=1e-3):
    """Lambda function for warmup + constant."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return 1.0

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = LambdaLR(optimizer, lr_lambda=lambda s: warmup_lambda(s))

# Training loop
for step, batch in enumerate(dataloader):
    train_step(batch)
    scheduler.step()  # Step after each batch
```

## Practical Guidelines

### Choosing Warmup Duration

| Model Size | Batch Size | Recommended Warmup |
|------------|------------|-------------------|
| Small (<1M params) | <64 | 100-500 steps |
| Medium (1-10M) | 64-256 | 500-2000 steps |
| Large (10-100M) | 256-1024 | 2000-5000 steps |
| Very Large (>100M) | >1024 | 5000-10000 steps |

**Rule of thumb:** 5-10% of total training steps

### Scaling with Batch Size

Larger batches benefit from longer warmup:

```python
def compute_warmup_steps(batch_size, base_warmup=1000, base_batch=32):
    """Scale warmup with batch size."""
    return int(base_warmup * (batch_size / base_batch))

# Example: batch_size=256, base_warmup=1000
warmup_steps = compute_warmup_steps(256)  # 8000 steps
```

### Transformer Training Recipe

The standard transformer warmup schedule (from "Attention is All You Need"):

```python
class TransformerSchedule:
    """
    Original transformer LR schedule.
    
    LR = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    
    def __init__(
        self,
        d_model: int,
        warmup_steps: int,
        factor: float = 1.0
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.current_step = 0
    
    def get_lr(self, step: int = None) -> float:
        if step is None:
            step = self.current_step
        step = max(step, 1)  # Avoid division by zero
        
        return self.factor * (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * self.warmup_steps ** -1.5
        )
    
    def step(self):
        self.current_step += 1
        return self.get_lr()


# Usage for BERT-base (d_model=768)
scheduler = TransformerSchedule(d_model=768, warmup_steps=10000)
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train_with_warmup(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    base_lr: float = 1e-3,
    warmup_epochs: int = 5,
    device: str = 'cuda'
):
    """
    Train model with warmup + cosine decay schedule.
    """
    model = model.to(device)
    
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    # Create combined warmup + decay scheduler
    scheduler = WarmupWithDecay(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=base_lr / 100
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_acc': [], 'lr': []}
    
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Update learning rate
            lr = scheduler.get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        history['train_loss'].append(epoch_loss / steps_per_epoch)
        history['val_acc'].append(100 * correct / total)
        history['lr'].append(scheduler.get_lr())
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"LR: {history['lr'][-1]:.2e} | "
              f"Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Acc: {history['val_acc'][-1]:.2f}%")
    
    return history
```

## Visualization

```python
import matplotlib.pyplot as plt

def visualize_warmup_strategies(warmup_steps=1000, total_steps=10000):
    """Compare different warmup strategies."""
    
    schedulers = {
        'Linear': LinearWarmup(base_lr=1e-3, warmup_steps=warmup_steps),
        'Exponential': ExponentialWarmup(base_lr=1e-3, warmup_steps=warmup_steps),
        'Cosine': CosineWarmup(base_lr=1e-3, warmup_steps=warmup_steps),
        'Warmup+Decay': WarmupWithDecay(
            base_lr=1e-3, warmup_steps=warmup_steps,
            total_steps=total_steps, min_lr=1e-6
        )
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Warmup phase only
    steps = range(min(warmup_steps * 2, total_steps))
    for name, sched in schedulers.items():
        if name != 'Warmup+Decay':
            lrs = [sched.get_lr(s) for s in steps]
            axes[0].plot(steps, lrs, label=name, linewidth=2)
    
    axes[0].axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.5,
                    label=f'Warmup end ({warmup_steps})')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Learning Rate')
    axes[0].set_title('Warmup Strategies Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Full schedule with decay
    steps = range(total_steps)
    sched = schedulers['Warmup+Decay']
    lrs = [sched.get_lr(s) for s in steps]
    axes[1].plot(steps, lrs, 'b-', linewidth=2)
    axes[1].axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.5,
                    label=f'Warmup end')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Warmup + Cosine Decay')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Advantages and Disadvantages

### Advantages

✅ **Stabilizes early training** - Prevents loss explosions

✅ **Better final performance** - Often improves accuracy

✅ **Enables larger batch sizes** - Critical for distributed training

✅ **Works with any main scheduler** - Combinable with decay strategies

✅ **Essential for transformers** - Required for BERT, GPT training

### Disadvantages

❌ **Adds complexity** - Another hyperparameter to tune

❌ **Slower start** - Delays reaching full learning rate

❌ **May not help small models** - Overhead not always justified

## When to Use Warmup

**Always use when:**

- Training transformers or large language models
- Using large batch sizes (>256)
- Training from scratch on large datasets
- Experiencing early training instability

**May skip when:**

- Fine-tuning pre-trained models (shorter warmup)
- Small models with small batch sizes
- Training is already stable

## Summary

Warmup is a crucial technique for training large models, especially transformers. Linear warmup combined with cosine decay has become the standard recipe for many deep learning applications.

**Key takeaways:**

1. Linear warmup is the most common and often sufficient
2. Use 5-10% of total steps for warmup
3. Scale warmup duration with batch size
4. Combine with cosine decay for complete schedule
5. Essential for transformer training
