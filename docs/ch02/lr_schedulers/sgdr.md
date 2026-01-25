# SGDR: Warm Restarts

## Overview

SGDR (Stochastic Gradient Descent with Warm Restarts) combines cosine annealing with periodic restarts, where the learning rate is reset to its maximum value after each cycle. This technique can help escape local minima and often leads to better final performance, especially in long training runs.

## The Warm Restart Concept

Standard cosine annealing decreases the learning rate monotonically to a minimum. SGDR extends this by:

1. **Cosine decay** to minimum within each cycle
2. **Sharp restart** to maximum at cycle boundaries
3. **Increasing cycle lengths** (optionally) for gradual convergence

This creates a "snapshot" effect where multiple good solutions are explored.

## Mathematical Formulation

Within each cycle $i$, the learning rate follows:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)$$

where:
- $T_{cur}$ = steps since last restart
- $T_i$ = length of current cycle
- After restart: $T_{i+1} = T_{mult} \times T_i$

### Cycle Length Progression

With $T_0 = 10$ epochs and $T_{mult} = 2$:

| Cycle | Length | Cumulative |
|-------|--------|------------|
| 1 | 10 | 10 |
| 2 | 20 | 30 |
| 3 | 40 | 70 |
| 4 | 80 | 150 |

## Learning Rate Curve

```
LR
 │
max├──╮    ╭──╮        ╭────╮
 │   │    │  │        │    │
 │   │    │  │        │    │
 │   ╰──╮ │  ╰──╮     │    ╰────╮
 │      ╰─╯     ╰─────╯         ╰────
min├──────────────────────────────────
 │
 └──┬────┬────────┬──────────────────→
    T₀  T₀+T₁   T₀+T₁+T₂          steps
    
    ↑    ↑        ↑
  restart restart restart
```

## PyTorch Implementation

### Using Built-in CosineAnnealingWarmRestarts

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Create SGDR scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,        # First cycle length (epochs or steps)
    T_mult=2,      # Multiply cycle length after each restart
    eta_min=1e-6   # Minimum learning rate
)

# Training loop
for epoch in range(epochs):
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
    
    # Step with fractional epoch for smooth updates
    scheduler.step(epoch + batch_idx / len(train_loader))
```

### Custom Implementation

```python
import math
from typing import Optional, List

class CosineAnnealingWarmRestarts:
    """
    SGDR: Stochastic Gradient Descent with Warm Restarts.
    
    Reference: "SGDR: Stochastic Gradient Descent with Warm Restarts"
    https://arxiv.org/abs/1608.03983
    """
    
    def __init__(
        self,
        max_lr: float,
        min_lr: float = 0.0,
        T_0: int = 10,
        T_mult: int = 1,
        decay_rate: float = 1.0
    ):
        """
        Args:
            max_lr: Maximum (restart) learning rate
            min_lr: Minimum learning rate
            T_0: First cycle length in steps/epochs
            T_mult: Factor to increase cycle length (1 = constant)
            decay_rate: Factor to reduce max_lr after each restart
        """
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.decay_rate = decay_rate
        self.current_step = 0
    
    def get_lr(self, step: Optional[float] = None) -> float:
        """Calculate learning rate for given step."""
        if step is None:
            step = self.current_step
        
        # Find which cycle we're in and position within it
        T_cur = step
        T_i = self.T_0
        cycle = 0
        
        while T_cur >= T_i:
            T_cur -= T_i
            T_i = int(T_i * self.T_mult)
            cycle += 1
        
        # Decay max_lr based on cycle
        current_max_lr = self.max_lr * (self.decay_rate ** cycle)
        
        # Cosine annealing within current cycle
        progress = T_cur / T_i
        cos_value = math.cos(math.pi * progress)
        lr = self.min_lr + 0.5 * (current_max_lr - self.min_lr) * (1 + cos_value)
        
        return lr
    
    def step(self, epoch: Optional[float] = None):
        """Advance scheduler."""
        if epoch is not None:
            self.current_step = epoch
        else:
            self.current_step += 1
        return self.get_lr()
    
    def get_restart_steps(self, max_steps: int) -> List[int]:
        """Get list of steps where restarts occur."""
        restarts = []
        current = 0
        T_i = self.T_0
        
        while current < max_steps:
            current += T_i
            if current < max_steps:
                restarts.append(current)
            T_i = int(T_i * self.T_mult)
        
        return restarts
    
    def get_schedule(self, total_steps: int) -> list:
        """Generate complete LR schedule."""
        return [self.get_lr(s) for s in range(total_steps)]
```

## Snapshot Ensembles

A powerful application of SGDR is creating ensembles from snapshots taken at cycle minima:

```python
class SnapshotEnsemble:
    """
    Create ensemble from SGDR snapshots.
    
    Save model at each LR minimum and average predictions.
    """
    
    def __init__(self):
        self.snapshots = []
    
    def save_snapshot(self, model):
        """Save a copy of model state."""
        import copy
        snapshot_state = copy.deepcopy(model.state_dict())
        self.snapshots.append(snapshot_state)
    
    def predict(self, model, x):
        """Average predictions from all snapshots."""
        predictions = []
        original_state = model.state_dict()
        
        with torch.no_grad():
            for state in self.snapshots:
                model.load_state_dict(state)
                model.eval()
                pred = model(x)
                predictions.append(pred)
        
        # Restore original state
        model.load_state_dict(original_state)
        
        return torch.stack(predictions).mean(dim=0)
    
    def predict_with_uncertainty(self, model, x):
        """Predictions with uncertainty estimate."""
        predictions = []
        original_state = model.state_dict()
        
        with torch.no_grad():
            for state in self.snapshots:
                model.load_state_dict(state)
                model.eval()
                pred = torch.softmax(model(x), dim=-1)
                predictions.append(pred)
        
        model.load_state_dict(original_state)
        
        preds = torch.stack(predictions)
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)
        
        return mean_pred, std_pred
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def train_with_sgdr(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 150,
    max_lr: float = 0.1,
    min_lr: float = 1e-6,
    T_0: int = 10,
    T_mult: int = 2,
    device: str = 'cuda'
):
    """
    Train model with SGDR scheduler.
    """
    model = model.to(device)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=max_lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=min_lr
    )
    
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }
    
    # Calculate restart epochs for logging
    restart_epochs = []
    T_i = T_0
    current = 0
    while current < epochs:
        current += T_i
        if current < epochs:
            restart_epochs.append(current)
        T_i *= T_mult
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        
        # Step scheduler with fractional epoch
        scheduler.step(epoch + batch_idx / len(train_loader))
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation
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
        
        # Record history
        history['train_loss'].append(epoch_loss / total)
        history['val_loss'].append(val_loss / val_total)
        history['train_acc'].append(100 * correct / total)
        history['val_acc'].append(100 * val_correct / val_total)
        history['lr'].append(current_lr)
        
        # Mark restarts
        restart_marker = " *** RESTART ***" if epoch in restart_epochs else ""
        print(f"Epoch {epoch+1}/{epochs} | "
              f"LR: {current_lr:.6f} | "
              f"Val Acc: {history['val_acc'][-1]:.2f}%{restart_marker}")
    
    return history
```

## Visualization

```python
import matplotlib.pyplot as plt

def visualize_sgdr(T_0: int = 10, T_mult: int = 2, epochs: int = 150):
    """Visualize SGDR learning rate schedule."""
    
    scheduler = CosineAnnealingWarmRestarts(
        max_lr=0.1,
        min_lr=1e-6,
        T_0=T_0,
        T_mult=T_mult
    )
    
    steps = range(epochs)
    lrs = scheduler.get_schedule(epochs)
    restarts = scheduler.get_restart_steps(epochs)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(steps, lrs, 'b-', linewidth=2)
    
    # Mark restarts
    for r in restarts:
        ax.axvline(x=r, color='r', linestyle='--', alpha=0.5,
                   label='Restart' if r == restarts[0] else '')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'SGDR Schedule (T_0={T_0}, T_mult={T_mult})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig
```

## Practical Guidelines

### Choosing T_0

| T_0 Value | Effect | Use Case |
|-----------|--------|----------|
| 5-10 | Frequent restarts | Quick exploration |
| 10-20 | Balanced | Standard training |
| 20+ | Infrequent restarts | Long training |

### Choosing T_mult

| T_mult | Pattern | Best For |
|--------|---------|----------|
| 1 | Constant cycle | Snapshot ensembles |
| 2 | Doubling (common) | General training |
| 3+ | Rapid increase | Very long training |

## Advantages and Disadvantages

### Advantages

✅ **Escapes local minima** - Restarts provide fresh exploration

✅ **Enables snapshot ensembles** - Multiple good models from one run

✅ **Better generalization** - Often improves final accuracy

✅ **Theoretically motivated** - Explores loss landscape effectively

### Disadvantages

❌ **More hyperparameters** - T_0 and T_mult to tune

❌ **Longer training needed** - Benefits show over multiple cycles

❌ **May waste computation** - Some restarts may not help

## When to Use SGDR

**Good use cases:**

- Long training runs (50+ epochs)
- Want snapshot ensembles
- Training seems stuck
- Research experiments

**Avoid when:**

- Short training budget
- Need fast convergence
- Transfer learning

## Summary

SGDR combines cosine annealing with periodic restarts, enabling exploration of multiple solutions. Key takeaways:

1. T_mult=2 is a good default
2. Use T_mult=1 for snapshot ensembles
3. Restarts help escape local minima
4. Best for long training runs
