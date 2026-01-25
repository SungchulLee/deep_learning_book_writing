# Cosine Annealing

## Overview

Cosine annealing is one of the most popular learning rate scheduling strategies in modern deep learning. It decreases the learning rate following a cosine curve, providing smooth transitions that are fast at the extremes and slow in the middle. This schedule has become the default choice for training vision models and is widely used in image classification benchmarks.

## Mathematical Formulation

### Basic Cosine Annealing

The standard cosine annealing schedule is:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t \cdot \pi}{T_{max}}\right)\right)$$

where:
- $\eta_t$ is the learning rate at epoch $t$
- $\eta_{max}$ is the initial (maximum) learning rate
- $\eta_{min}$ is the minimum learning rate
- $T_{max}$ is the total number of epochs

### Derivation

Starting from the cosine function over $[0, \pi]$:

$$\cos(0) = 1, \quad \cos(\pi) = -1$$

Mapping $t \in [0, T_{max}]$ to $[0, \pi]$:

$$\theta = \frac{t \cdot \pi}{T_{max}}$$

The cosine term ranges from 1 to -1, which we transform to [0, 1]:

$$\frac{1 + \cos(\theta)}{2} \in [1, 0]$$

Scaling to the learning rate range:

$$\eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \cdot \frac{1 + \cos(\theta)}{2}$$

## Learning Rate Curve

```
Learning Rate
    │
η_max├─╮
    │  ╲
    │   ╲
    │    ╲___
    │        ╲____
    │             ╲_______
η_min├───────────────────────╲
    │
    └────────────────────────────→ Epoch
     0                      T_max
     
     Smooth cosine decay curve
```

The cosine curve has the property of being:
- **Fast at the start** - Large initial reduction
- **Slow in the middle** - Gradual refinement
- **Fast at the end** - Quick convergence to minimum

## PyTorch Implementation

### Using CosineAnnealingLR

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Create CosineAnnealingLR scheduler
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,      # Total epochs
    eta_min=0       # Minimum learning rate
)

# Training loop
for epoch in range(100):
    train_one_epoch(model, optimizer, train_loader)
    validate(model, val_loader)
    
    # Step scheduler after each epoch
    scheduler.step()
    
    print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]:.6f}")
```

### Custom Implementation

```python
import math
from typing import Optional

class CosineAnnealingScheduler:
    """
    Cosine annealing learning rate scheduler.
    
    LR follows a cosine curve from max_lr to min_lr over T_max epochs.
    """
    
    def __init__(
        self,
        max_lr: float,
        T_max: int,
        min_lr: float = 0.0
    ):
        """
        Args:
            max_lr: Maximum (initial) learning rate
            T_max: Number of epochs for one cosine cycle
            min_lr: Minimum learning rate
        """
        self.max_lr = max_lr
        self.T_max = T_max
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def get_lr(self, epoch: Optional[int] = None) -> float:
        """
        Calculate learning rate for given epoch.
        
        Args:
            epoch: Epoch number (uses current if None)
        
        Returns:
            Learning rate for the epoch
        """
        if epoch is None:
            epoch = self.current_epoch
        
        # Cosine annealing formula
        cos_value = math.cos(math.pi * epoch / self.T_max)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + cos_value)
        return lr
    
    def step(self):
        """Advance to next epoch."""
        self.current_epoch += 1
    
    def get_schedule(self) -> list:
        """Get complete LR schedule for visualization."""
        return [self.get_lr(e) for e in range(self.T_max + 1)]


class CosineAnnealingWithWarmup:
    """
    Cosine annealing with linear warmup.
    
    Popular schedule for transformer training.
    """
    
    def __init__(
        self,
        max_lr: float,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0.0
    ):
        """
        Args:
            max_lr: Peak learning rate after warmup
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            min_lr: Minimum learning rate
        """
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def get_lr(self, epoch: Optional[int] = None) -> float:
        """Calculate learning rate for given epoch."""
        if epoch is None:
            epoch = self.current_epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cos_value = math.cos(math.pi * progress)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + cos_value)
    
    def step(self):
        """Advance to next epoch."""
        self.current_epoch += 1


# Per-step cosine annealing (for batch-level updates)
class CosineAnnealingPerStep:
    """
    Cosine annealing computed per training step.
    """
    
    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        min_lr: float = 0.0,
        warmup_steps: int = 0
    ):
        """
        Args:
            max_lr: Maximum learning rate
            total_steps: Total training steps
            min_lr: Minimum learning rate
            warmup_steps: Number of warmup steps
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def get_lr(self, step: Optional[int] = None) -> float:
        """Calculate learning rate for given step."""
        if step is None:
            step = self.current_step
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * (step + 1) / self.warmup_steps
        
        # Cosine annealing after warmup
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)  # Clamp to [0, 1]
        
        cos_value = math.cos(math.pi * progress)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + cos_value)
    
    def step(self):
        """Advance one training step."""
        self.current_step += 1
        return self.get_lr()
```

## SGDR: Cosine Annealing with Warm Restarts

SGDR (Stochastic Gradient Descent with Warm Restarts) extends cosine annealing by periodically restarting the schedule, potentially helping escape local minima.

### Mathematical Formulation

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_i} \pi\right)\right)$$

where:
- $T_{cur}$ is the number of epochs since the last restart
- $T_i$ is the number of epochs in the current restart period
- After each restart: $T_{i+1} = T_i \cdot T_{mult}$

### PyTorch Implementation

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Create scheduler with warm restarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,        # Epochs until first restart
    T_mult=2,      # Multiply period after each restart
    eta_min=1e-6   # Minimum learning rate
)

# Restart schedule:
# Restart 1: epochs 0-9 (T=10)
# Restart 2: epochs 10-29 (T=20)
# Restart 3: epochs 30-69 (T=40)
# etc.
```

### Custom SGDR Implementation

```python
class CosineAnnealingWarmRestarts:
    """
    SGDR: Cosine annealing with warm restarts.
    
    Reference: "SGDR: Stochastic Gradient Descent with Warm Restarts"
    https://arxiv.org/abs/1608.03983
    """
    
    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        T_0: int,
        T_mult: int = 1
    ):
        """
        Args:
            max_lr: Maximum learning rate at restart
            min_lr: Minimum learning rate
            T_0: Number of epochs/steps for first restart period
            T_mult: Factor to increase period after each restart
        """
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.current_step = 0
    
    def get_lr(self, step: Optional[int] = None) -> float:
        """Calculate learning rate considering warm restarts."""
        if step is None:
            step = self.current_step
        
        # Find which restart cycle we're in
        T_cur = step
        T_i = self.T_0
        
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
        
        # Cosine annealing within current cycle
        progress = T_cur / T_i
        cos_value = math.cos(math.pi * progress)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + cos_value)
    
    def step(self):
        """Advance one step."""
        self.current_step += 1
    
    def get_restart_epochs(self, max_epochs: int) -> list:
        """Get list of epochs where restarts occur."""
        restarts = []
        current = 0
        T_i = self.T_0
        
        while current < max_epochs:
            current += T_i
            if current < max_epochs:
                restarts.append(current)
            T_i *= self.T_mult
        
        return restarts


# Example: Visualize SGDR schedule
def visualize_sgdr():
    import matplotlib.pyplot as plt
    
    scheduler = CosineAnnealingWarmRestarts(
        max_lr=0.1,
        min_lr=1e-5,
        T_0=10,
        T_mult=2
    )
    
    total_epochs = 100
    lrs = []
    for epoch in range(total_epochs):
        lrs.append(scheduler.get_lr(epoch))
    
    restarts = scheduler.get_restart_epochs(total_epochs)
    
    plt.figure(figsize=(12, 4))
    plt.plot(range(total_epochs), lrs, 'b-', linewidth=2)
    for r in restarts:
        plt.axvline(x=r, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('SGDR: Cosine Annealing with Warm Restarts')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
```

## Practical Guidelines

### Choosing T_max

`T_max` should typically equal the total number of training epochs:

```python
epochs = 100
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
```

**Warning:** If `T_max < epochs`, the schedule will complete before training ends, resulting in constant minimum LR for remaining epochs.

### Choosing eta_min

Common choices for minimum learning rate:

| Setting | eta_min | Use Case |
|---------|---------|----------|
| Zero | 0 | Standard choice |
| Small | 1e-6 | Continued learning |
| Fraction | max_lr / 100 | Proportional minimum |

### SGDR Parameters

| Parameter | Recommendation | Effect |
|-----------|---------------|--------|
| T_0 | 10-50 epochs | First cycle length |
| T_mult | 1 or 2 | Cycle growth rate |
| eta_min | 1e-6 to 1e-4 | Minimum at restart |

**T_mult values:**
- `T_mult = 1`: Fixed cycle length (constant restarts)
- `T_mult = 2`: Doubling cycle length (fewer restarts over time)

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

def train_with_cosine_annealing(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 100,
    max_lr: float = 0.1,
    min_lr: float = 0,
    warmup_epochs: int = 0,
    device: str = 'cuda'
):
    """
    Train model with cosine annealing learning rate schedule.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Total training epochs
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        warmup_epochs: Number of warmup epochs
        device: Training device
    
    Returns:
        Training history dictionary
    """
    model = model.to(device)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=max_lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # Create scheduler
    if warmup_epochs > 0:
        # Use custom warmup + cosine
        scheduler = CosineAnnealingWithWarmup(
            max_lr=max_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            min_lr=min_lr
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_lr
        )
    
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }
    
    for epoch in range(epochs):
        # Get current LR and set it
        if warmup_epochs > 0:
            current_lr = scheduler.get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
        
        # Record metrics
        if warmup_epochs > 0:
            history['lr'].append(current_lr)
        else:
            history['lr'].append(scheduler.get_last_lr()[0])
            scheduler.step()
        
        history['train_loss'].append(train_loss / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['train_acc'].append(100 * train_correct / train_total)
        history['val_acc'].append(100 * val_correct / val_total)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"LR: {history['lr'][-1]:.6f} | "
                  f"Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Val Acc: {history['val_acc'][-1]:.2f}%")
    
    return history


def plot_cosine_comparison():
    """Compare different cosine annealing variants."""
    import matplotlib.pyplot as plt
    
    epochs = 100
    
    # Standard cosine
    standard = CosineAnnealingScheduler(max_lr=0.1, T_max=epochs, min_lr=0)
    
    # With warmup
    warmup = CosineAnnealingWithWarmup(
        max_lr=0.1, warmup_epochs=10, total_epochs=epochs, min_lr=0
    )
    
    # SGDR
    sgdr = CosineAnnealingWarmRestarts(
        max_lr=0.1, min_lr=1e-5, T_0=20, T_mult=2
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (name, sched) in zip(axes, [
        ('Standard Cosine', standard),
        ('Cosine + Warmup', warmup),
        ('SGDR (Warm Restarts)', sgdr)
    ]):
        lrs = [sched.get_lr(e) for e in range(epochs)]
        ax.plot(range(epochs), lrs, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Advantages and Disadvantages

### Advantages

✅ **Smooth decay** - No sudden drops, stable training

✅ **Fast at extremes** - Quick initial progress and final convergence

✅ **Slow in middle** - Careful refinement during transition

✅ **Modern standard** - Widely used and well-understood

✅ **Minimal tuning** - Just T_max and optional eta_min

✅ **Strong empirical performance** - Proven on ImageNet and CIFAR

### Disadvantages

❌ **T_max must match epochs** - Requires knowing training length

❌ **No adaptivity** - Doesn't respond to training dynamics

❌ **Single cycle** - May benefit from restarts (use SGDR)

❌ **May be suboptimal early** - Consider warmup for large models

## When to Use Cosine Annealing

**Best use cases:**

- Image classification (ResNet, EfficientNet, etc.)
- Modern deep learning architectures
- When smooth decay is preferred
- CIFAR-10, ImageNet training
- Competition settings

**Consider alternatives:**

- Very short training → OneCycleLR
- Unknown dataset → ReduceLROnPlateau
- Transformer training → Warmup + Cosine
- Stuck in local minima → SGDR

## Comparison with Other Schedulers

| Aspect | Cosine | Step | OneCycle |
|--------|--------|------|----------|
| Smoothness | Smooth curve | Sharp drops | Smooth cycle |
| Warmup | Optional | No | Built-in |
| Restarts | Via SGDR | No | No |
| Popularity | Very high | High | High |
| Tuning effort | Low | Medium | Low |

## Research Applications

Cosine annealing has been particularly successful in:

1. **Image Classification** - Standard for ResNet, DenseNet training
2. **Object Detection** - Used in YOLO, Faster R-CNN
3. **Semantic Segmentation** - Common in DeepLab, U-Net
4. **Neural Architecture Search** - Default in DARTS, ProxylessNAS
5. **Self-supervised Learning** - Used in SimCLR, MoCo

## Summary

Cosine annealing provides an elegant, effective learning rate schedule that has become a default choice in computer vision. Its smooth decay profile, combined with minimal hyperparameters, makes it both easy to use and highly effective.

**Key takeaways:**

1. Set `T_max` equal to total training epochs
2. Consider warmup for large models or batch sizes
3. Use SGDR (warm restarts) if training seems stuck
4. `eta_min = 0` is a reasonable default
5. Combines well with momentum-based optimizers
