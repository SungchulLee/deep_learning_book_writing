# Cyclic Learning Rates

## Overview

Cyclic Learning Rates (CLR) is a technique that oscillates the learning rate between a minimum and maximum value during training. Proposed by Leslie Smith in 2015, this approach can help escape local minima, find better solutions, and reduce the need for learning rate tuning.

## The Intuition Behind Cycling

Traditional scheduling decreases the learning rate monotonically. Cyclic learning rates challenge this by:

1. **Periodically increasing LR** - May escape shallow local minima
2. **Reducing sensitivity to LR choice** - Explores a range of values
3. **Providing implicit regularization** - Large LR periods prevent overfitting
4. **Accelerating training** - Can converge faster than fixed schedules

## Cycle Modes

### Triangular (Basic)

Learning rate oscillates linearly between bounds with constant amplitude:

```
LR
 │     /\    /\    /\
max├───/  \  /  \  /  \
 │   /    \/    \/    \
base├──/                 \
 └─────────────────────────→ step
```

### Triangular2 (Decreasing Amplitude)

Amplitude halves after each cycle:

```
LR
 │     /\
max├───/  \
 │   /    \   /\
 │  /      \ /  \
base├─/        \    /\
 └────────────────────→ step
```

### Exp_Range (Exponential Decay)

Amplitude decays exponentially:

```
LR
 │     /\
max├───/  \
 │   /    \_/\__
 │  /           \_/\___
base├─/                    \_
 └──────────────────────────→ step
```

## Mathematical Formulation

### Triangular Mode

For a cycle with half-period `step_size`:

$$\text{cycle} = \left\lfloor 1 + \frac{t}{2 \cdot \text{step\_size}} \right\rfloor$$

$$x = \left| \frac{t}{\text{step\_size}} - 2 \cdot \text{cycle} + 1 \right|$$

$$\eta_t = \eta_{base} + (\eta_{max} - \eta_{base}) \cdot \max(0, 1 - x)$$

### Triangular2 Mode

Same as triangular with scaling factor:

$$\text{scale} = \frac{1}{2^{\text{cycle} - 1}}$$

$$\eta_t = \eta_{base} + (\eta_{max} - \eta_{base}) \cdot \max(0, 1 - x) \cdot \text{scale}$$

### Exp_Range Mode

With exponential decay factor $\gamma$:

$$\text{scale} = \gamma^t$$

$$\eta_t = \eta_{base} + (\eta_{max} - \eta_{base}) \cdot \max(0, 1 - x) \cdot \text{scale}$$

## PyTorch Implementation

### Using Built-in CyclicLR

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Create CyclicLR scheduler
scheduler = CyclicLR(
    optimizer,
    base_lr=0.001,         # Minimum LR
    max_lr=0.1,            # Maximum LR
    step_size_up=2000,     # Steps to reach max_lr
    step_size_down=2000,   # Steps to reach base_lr (optional)
    mode='triangular',     # 'triangular', 'triangular2', 'exp_range'
    gamma=1.0,             # For exp_range mode
    cycle_momentum=True,   # Cycle momentum inversely
    base_momentum=0.8,     # Min momentum
    max_momentum=0.9       # Max momentum
)

# Training loop - step after each BATCH
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        # Step scheduler after each batch!
        scheduler.step()
```

### Custom Implementation

```python
import math
from typing import Literal, Optional

class CyclicLR:
    """
    Cyclical Learning Rate scheduler.
    
    Reference: "Cyclical Learning Rates for Training Neural Networks"
    https://arxiv.org/abs/1506.01186
    """
    
    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        step_size: int,
        mode: Literal['triangular', 'triangular2', 'exp_range'] = 'triangular',
        gamma: float = 0.99994
    ):
        """
        Args:
            base_lr: Minimum learning rate
            max_lr: Maximum learning rate
            step_size: Half-cycle length (steps from base to max)
            mode: Cycling mode
            gamma: Decay factor for exp_range mode
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.current_step = 0
    
    def get_lr(self, step: Optional[int] = None) -> float:
        """Calculate learning rate for given step."""
        if step is None:
            step = self.current_step
        
        # Determine cycle and position
        cycle = math.floor(1 + step / (2 * self.step_size))
        x = abs(step / self.step_size - 2 * cycle + 1)
        
        # Calculate scale factor based on mode
        if self.mode == 'triangular':
            scale = 1.0
        elif self.mode == 'triangular2':
            scale = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale = self.gamma ** step
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Calculate learning rate
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) * scale
        return lr
    
    def step(self):
        """Advance one step."""
        self.current_step += 1
        return self.get_lr()
    
    def get_cycle_length(self) -> int:
        """Get full cycle length (steps)."""
        return 2 * self.step_size
    
    def get_schedule(self, total_steps: int) -> list:
        """Get complete LR schedule."""
        return [self.get_lr(s) for s in range(total_steps)]


class CyclicLRWithMomentum:
    """
    Cyclic LR with inverse momentum cycling.
    
    When LR is high, momentum is low and vice versa.
    """
    
    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        base_momentum: float,
        max_momentum: float,
        step_size: int,
        mode: str = 'triangular'
    ):
        self.lr_scheduler = CyclicLR(base_lr, max_lr, step_size, mode)
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.step_size = step_size
        self.current_step = 0
    
    def get_lr_momentum(self, step: Optional[int] = None) -> tuple:
        """Get both LR and momentum for given step."""
        if step is None:
            step = self.current_step
        
        lr = self.lr_scheduler.get_lr(step)
        
        # Inverse relationship: high LR → low momentum
        lr_fraction = (lr - self.lr_scheduler.base_lr) / \
                      (self.lr_scheduler.max_lr - self.lr_scheduler.base_lr)
        momentum = self.max_momentum - lr_fraction * (self.max_momentum - self.base_momentum)
        
        return lr, momentum
    
    def step(self):
        """Advance one step."""
        self.current_step += 1
        return self.get_lr_momentum()
```

## Finding the LR Range

A critical step for CLR is finding appropriate `base_lr` and `max_lr`. Leslie Smith's LR Range Test helps:

```python
def lr_range_finder(
    model,
    train_loader,
    criterion,
    optimizer_class=torch.optim.SGD,
    start_lr: float = 1e-7,
    end_lr: float = 10,
    num_iterations: int = 100,
    smooth_factor: float = 0.05
):
    """
    Find optimal LR range for cyclic learning rates.
    
    Returns:
        dict with learning rates and losses
    """
    # Create fresh model copy
    model = model.train()
    device = next(model.parameters()).device
    
    optimizer = optimizer_class(model.parameters(), lr=start_lr)
    
    # Exponential LR increase
    lr_mult = (end_lr / start_lr) ** (1 / num_iterations)
    
    results = {'lr': [], 'loss': [], 'smoothed_loss': []}
    smoothed_loss = 0
    best_loss = float('inf')
    
    data_iter = iter(train_loader)
    
    for i in range(num_iterations):
        # Get batch
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, targets = next(data_iter)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Check divergence
        if loss.item() > 4 * best_loss and i > 10:
            print(f"Stopping early at iteration {i}: loss diverged")
            break
        
        # Record
        current_lr = optimizer.param_groups[0]['lr']
        results['lr'].append(current_lr)
        results['loss'].append(loss.item())
        
        # Smooth loss
        if i == 0:
            smoothed_loss = loss.item()
        else:
            smoothed_loss = smooth_factor * loss.item() + (1 - smooth_factor) * smoothed_loss
        results['smoothed_loss'].append(smoothed_loss)
        best_loss = min(best_loss, smoothed_loss)
        
        # Backward and update
        loss.backward()
        optimizer.step()
        
        # Increase LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
    
    return results


def suggest_lr_bounds(results: dict) -> tuple:
    """
    Suggest base_lr and max_lr from range test results.
    
    Returns:
        (base_lr, max_lr)
    """
    losses = results['smoothed_loss']
    lrs = results['lr']
    
    # Find where loss starts decreasing significantly
    min_grad_idx = 0
    min_grad = float('inf')
    
    for i in range(1, len(losses) - 1):
        # Compute gradient of loss curve
        grad = (losses[i+1] - losses[i-1]) / (lrs[i+1] - lrs[i-1])
        if grad < min_grad:
            min_grad = grad
            min_grad_idx = i
    
    # Find minimum loss point
    min_loss_idx = losses.index(min(losses))
    
    # Suggestions
    base_lr = lrs[min_grad_idx] / 10  # Start of good region / 10
    max_lr = lrs[min_loss_idx] / 3    # Minimum point / 3
    
    return base_lr, max_lr
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

def train_with_cyclic_lr(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 50,
    base_lr: float = 0.001,
    max_lr: float = 0.1,
    step_size_epochs: int = 4,
    mode: str = 'triangular2',
    device: str = 'cuda'
):
    """
    Train model with cyclic learning rate.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Total training epochs
        base_lr: Minimum learning rate
        max_lr: Maximum learning rate
        step_size_epochs: Epochs for half cycle
        mode: 'triangular', 'triangular2', or 'exp_range'
        device: Training device
    
    Returns:
        Training history
    """
    model = model.to(device)
    
    # Calculate step size in iterations
    steps_per_epoch = len(train_loader)
    step_size = step_size_epochs * steps_per_epoch
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    scheduler = CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size,
        mode=mode,
        cycle_momentum=True,
        base_momentum=0.8,
        max_momentum=0.9
    )
    
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }
    
    global_step = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        epoch_lrs = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Step scheduler after each batch
            scheduler.step()
            epoch_lrs.append(scheduler.get_last_lr()[0])
            
            epoch_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            global_step += 1
        
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
        history['lr'].append(sum(epoch_lrs) / len(epoch_lrs))
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | "
              f"LR: {history['lr'][-1]:.6f} | "
              f"Train Acc: {history['train_acc'][-1]:.2f}% | "
              f"Val Acc: {history['val_acc'][-1]:.2f}%")
    
    return history
```

## Visualization

```python
import matplotlib.pyplot as plt

def visualize_cyclic_modes(total_steps: int = 20000, step_size: int = 2000):
    """Compare different cyclic LR modes."""
    
    modes = {
        'triangular': CyclicLR(0.001, 0.1, step_size, 'triangular'),
        'triangular2': CyclicLR(0.001, 0.1, step_size, 'triangular2'),
        'exp_range': CyclicLR(0.001, 0.1, step_size, 'exp_range', gamma=0.99994)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    steps = range(total_steps)
    
    for ax, (name, scheduler) in zip(axes, modes.items()):
        lrs = [scheduler.get_lr(s) for s in steps]
        ax.plot(steps, lrs, 'b-', linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'{name.upper()} Mode')
        ax.grid(True, alpha=0.3)
        
        # Mark cycle boundaries
        cycle_length = 2 * step_size
        for i in range(0, total_steps, cycle_length):
            ax.axvline(x=i, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Advantages and Disadvantages

### Advantages

✅ **Escapes local minima** - Periodic LR increases can jump over barriers

✅ **Reduces tuning burden** - Explores LR range automatically

✅ **Can improve generalization** - High LR periods act as regularization

✅ **Enables LR range discovery** - Range test finds optimal bounds

✅ **Works with momentum cycling** - Additional regularization benefit

### Disadvantages

❌ **Requires finding LR bounds** - Must run range test

❌ **More complex than monotonic schedules** - Multiple hyperparameters

❌ **Updates per batch** - Higher overhead than epoch-level schedulers

❌ **May not converge to best solution** - Final LR might not be optimal

❌ **Amplitude decay complicates analysis** - Behavior changes over training

## Practical Guidelines

### Choosing step_size

| Cycle Length | Use Case |
|--------------|----------|
| 2-4 epochs | Frequent exploration |
| 4-8 epochs | Balanced (recommended) |
| 8+ epochs | Stable with occasional exploration |

### Choosing Mode

| Mode | When to Use |
|------|-------------|
| triangular | Initial experiments, understanding behavior |
| triangular2 | Most common, good balance |
| exp_range | Long training, gradual convergence |

### Finding LR Bounds

1. Run LR range test
2. Find where loss starts decreasing → `base_lr`
3. Find where loss reaches minimum → `max_lr`
4. Use `base_lr / 10` and `max_lr / 3` as conservative estimates

## When to Use Cyclic LR

**Good use cases:**

- Training seems stuck in local minima
- Want to explore optimal LR range
- Moderate training budget (10-50 epochs)
- When step decay isn't working well

**Avoid when:**

- Very short training (use OneCycleLR instead)
- Need fastest convergence
- Training transformers (use warmup + cosine)
- Final fine-tuning stage

## Summary

Cyclic Learning Rates provide a principled approach to LR scheduling that can escape local minima and reduce hyperparameter sensitivity. The key is finding appropriate LR bounds through range testing.

**Key takeaways:**

1. Run LR range test to find base_lr and max_lr
2. triangular2 mode is a good default
3. Step scheduler after each batch, not epoch
4. Consider combining with momentum cycling
5. Best for moderate training lengths
