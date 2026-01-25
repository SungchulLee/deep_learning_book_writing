# Exponential Decay

## Overview

Exponential decay provides a smooth, continuous decrease in learning rate throughout training. Unlike step decay which produces sharp drops, exponential decay reduces the learning rate by a constant factor each epoch, resulting in a gradual, predictable decline.

## Mathematical Formulation

The exponential learning rate schedule is defined as:

$$\eta_t = \eta_0 \cdot \gamma^t$$

where:
- $\eta_t$ is the learning rate at epoch $t$
- $\eta_0$ is the initial learning rate
- $\gamma$ is the decay rate (typically 0.9 to 0.99)
- $t$ is the current epoch

### Derivation from Continuous Decay

The exponential schedule can be derived from continuous exponential decay:

$$\frac{d\eta}{dt} = -\lambda \eta$$

Solving this differential equation:

$$\eta(t) = \eta_0 e^{-\lambda t}$$

In discrete form with $\gamma = e^{-\lambda}$:

$$\eta_t = \eta_0 \cdot \gamma^t$$

### Relationship to Half-Life

The "half-life" $\tau_{1/2}$ (epochs to halve the learning rate) relates to $\gamma$ by:

$$\tau_{1/2} = \frac{\ln(2)}{-\ln(\gamma)} = \frac{\ln(2)}{\ln(1/\gamma)}$$

For common gamma values:

| $\gamma$ | Half-Life (epochs) |
|----------|-------------------|
| 0.99 | 69 |
| 0.95 | 14 |
| 0.90 | 7 |
| 0.80 | 3.1 |

## Learning Rate Progression

**Example with $\eta_0 = 0.1$ and $\gamma = 0.95$:**

| Epoch | Calculation | Learning Rate |
|-------|-------------|---------------|
| 0 | $0.1 \times 0.95^0$ | 0.1000 |
| 10 | $0.1 \times 0.95^{10}$ | 0.0599 |
| 20 | $0.1 \times 0.95^{20}$ | 0.0358 |
| 50 | $0.1 \times 0.95^{50}$ | 0.0077 |
| 100 | $0.1 \times 0.95^{100}$ | 0.00059 |

## Learning Rate Curve

```
LR
 │
 │╲
 │ ╲
 │  ╲_
 │    ╲__
 │       ╲___
 │           ╲_____
 │                  ╲________
 └──────────────────────────────→ Epoch
   Smooth exponential decay
```

## PyTorch Implementation

### Using Built-in ExponentialLR

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Create ExponentialLR scheduler
scheduler = ExponentialLR(
    optimizer,
    gamma=0.95  # Decay factor per epoch
)

# Training loop
for epoch in range(100):
    train_one_epoch(model, optimizer, train_loader)
    
    # Step scheduler after each epoch
    scheduler.step()
    
    # Current learning rate
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}: LR = {current_lr:.6f}")
```

### Custom Implementation

```python
import math

class ExponentialDecayLR:
    """
    Exponential learning rate decay scheduler.
    
    LR = initial_lr * gamma^epoch
    """
    
    def __init__(
        self,
        initial_lr: float,
        gamma: float,
        min_lr: float = 1e-7
    ):
        """
        Args:
            initial_lr: Starting learning rate
            gamma: Decay rate per epoch (0 < gamma < 1)
            min_lr: Minimum learning rate floor
        """
        if not 0 < gamma < 1:
            raise ValueError("gamma must be between 0 and 1")
        
        self.initial_lr = initial_lr
        self.gamma = gamma
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def get_lr(self, epoch: int = None) -> float:
        """Calculate learning rate for given epoch."""
        if epoch is None:
            epoch = self.current_epoch
        
        lr = self.initial_lr * (self.gamma ** epoch)
        return max(lr, self.min_lr)
    
    def step(self):
        """Advance to next epoch."""
        self.current_epoch += 1
        return self.get_lr()
    
    def half_life(self) -> float:
        """Calculate half-life in epochs."""
        return math.log(2) / math.log(1 / self.gamma)
    
    @classmethod
    def from_half_life(cls, initial_lr: float, half_life: float, **kwargs):
        """
        Create scheduler from desired half-life.
        
        Args:
            initial_lr: Starting learning rate
            half_life: Epochs to halve learning rate
        
        Returns:
            ExponentialDecayLR instance
        """
        gamma = 0.5 ** (1 / half_life)
        return cls(initial_lr, gamma, **kwargs)
    
    @classmethod
    def from_final_lr(
        cls,
        initial_lr: float,
        final_lr: float,
        total_epochs: int,
        **kwargs
    ):
        """
        Create scheduler that reaches target LR at end of training.
        
        Args:
            initial_lr: Starting learning rate
            final_lr: Target learning rate at end
            total_epochs: Total training epochs
        
        Returns:
            ExponentialDecayLR instance
        """
        gamma = (final_lr / initial_lr) ** (1 / total_epochs)
        return cls(initial_lr, gamma, **kwargs)


# Example usage
# Method 1: Direct gamma specification
scheduler = ExponentialDecayLR(initial_lr=0.1, gamma=0.95)

# Method 2: Specify half-life
scheduler = ExponentialDecayLR.from_half_life(
    initial_lr=0.1,
    half_life=20  # LR halves every 20 epochs
)

# Method 3: Specify final LR
scheduler = ExponentialDecayLR.from_final_lr(
    initial_lr=0.1,
    final_lr=1e-4,
    total_epochs=100
)
```

### Per-Step Exponential Decay

For batch-level decay, compute gamma based on total steps:

```python
class PerStepExponentialDecay:
    """
    Exponential decay applied per training step.
    """
    
    def __init__(
        self,
        initial_lr: float,
        total_steps: int,
        final_lr: float = 1e-6
    ):
        """
        Args:
            initial_lr: Starting learning rate
            total_steps: Total training steps
            final_lr: Target final learning rate
        """
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.final_lr = final_lr
        
        # Compute per-step gamma
        self.gamma = (final_lr / initial_lr) ** (1 / total_steps)
        self.current_step = 0
    
    def get_lr(self, step: int = None) -> float:
        """Get learning rate for given step."""
        if step is None:
            step = self.current_step
        
        return self.initial_lr * (self.gamma ** step)
    
    def step(self):
        """Advance one training step."""
        self.current_step += 1
        return self.get_lr()
```

## Practical Guidelines

### Choosing Gamma

The choice of $\gamma$ depends on training length and desired decay speed:

| Training Length | Recommended $\gamma$ | Result |
|-----------------|---------------------|--------|
| Short (20-50 epochs) | 0.90-0.95 | Moderate decay |
| Medium (50-100 epochs) | 0.95-0.98 | Gentle decay |
| Long (100+ epochs) | 0.98-0.99 | Very gradual |

**General guidance:**

- $\gamma = 0.95$: Good default, decays to ~0.6% of initial by epoch 100
- $\gamma = 0.99$: Very gentle, decays to ~37% of initial by epoch 100
- $\gamma = 0.90$: Fast decay, may need higher initial LR

### Computing Gamma from Requirements

```python
def compute_gamma(initial_lr, final_lr, epochs):
    """
    Compute gamma to reach final_lr at given epoch.
    
    gamma = (final_lr / initial_lr)^(1/epochs)
    """
    return (final_lr / initial_lr) ** (1 / epochs)

# Example: Decay from 0.1 to 0.001 over 100 epochs
gamma = compute_gamma(0.1, 0.001, 100)
print(f"Required gamma: {gamma:.6f}")  # ~0.9550
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt

def train_with_exponential_decay(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 100,
    initial_lr: float = 0.1,
    gamma: float = 0.95,
    device: str = 'cuda'
):
    """
    Train model with exponential learning rate decay.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Total training epochs
        initial_lr: Starting learning rate
        gamma: Per-epoch decay factor
        device: Training device
    
    Returns:
        Training history dictionary
    """
    model = model.to(device)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }
    
    for epoch in range(epochs):
        # Training
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
        
        # Record metrics
        current_lr = scheduler.get_last_lr()[0]
        history['train_loss'].append(train_loss / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['train_acc'].append(100 * train_correct / train_total)
        history['val_acc'].append(100 * val_correct / val_total)
        history['lr'].append(current_lr)
        
        # Step scheduler
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"LR: {current_lr:.6f} | "
                  f"Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Val Acc: {history['val_acc'][-1]:.2f}%")
    
    return history


def compare_gamma_values(
    model_fn,
    train_loader,
    val_loader,
    gammas: list = [0.90, 0.95, 0.99],
    epochs: int = 50,
    initial_lr: float = 0.1
):
    """
    Compare training with different gamma values.
    """
    results = {}
    
    for gamma in gammas:
        print(f"\nTraining with gamma={gamma}")
        model = model_fn()
        history = train_with_exponential_decay(
            model, train_loader, val_loader,
            epochs=epochs, initial_lr=initial_lr, gamma=gamma
        )
        results[gamma] = history
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for gamma, history in results.items():
        epochs_range = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs_range, history['val_loss'], label=f'γ={gamma}')
        axes[1].plot(epochs_range, history['val_acc'], label=f'γ={gamma}')
        axes[2].plot(epochs_range, history['lr'], label=f'γ={gamma}')
    
    axes[0].set_title('Validation Loss')
    axes[0].legend()
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[2].set_title('Learning Rate')
    axes[2].set_yscale('log')
    axes[2].legend()
    
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return results, fig
```

## Advantages and Disadvantages

### Advantages

✅ **Smooth decay** - No sudden drops that might destabilize training

✅ **Simple parameterization** - Only one hyperparameter ($\gamma$)

✅ **Predictable** - Easy to calculate LR at any future epoch

✅ **Continuous** - Provides gradual transition for fine-tuning

### Disadvantages

❌ **Can decay too fast or slow** - Sensitive to $\gamma$ choice

❌ **No adaptivity** - Doesn't respond to training progress

❌ **May not reach minimum** - Can plateau above useful LR

❌ **Less commonly used** - Less empirical guidance available

## When to Use Exponential Decay

**Good use cases:**

- When smooth, gradual decay is preferred
- Smaller datasets where training is stable
- As baseline for comparison with other methods
- When step decay causes instability

**Avoid when:**

- Training large models (transformers, etc.)
- Short training budgets (prefer OneCycleLR)
- When periodic "restarts" might help
- Maximum performance is critical

## Comparison with Step Decay

| Aspect | Exponential | Step |
|--------|-------------|------|
| Decay pattern | Smooth, continuous | Discrete jumps |
| Parameterization | Single $\gamma$ | Step size + $\gamma$ |
| Stability | More stable | Can cause jumps |
| Final LR | Gradual approach | Sharp drops |
| Popularity | Less common | Very common |

## Variants

### Inverse Time Decay

A related schedule that decays inversely with time:

$$\eta_t = \frac{\eta_0}{1 + \alpha t}$$

```python
class InverseTimeDecay:
    """Inverse time decay schedule."""
    
    def __init__(self, initial_lr: float, decay_rate: float):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
    
    def get_lr(self, step: int) -> float:
        return self.initial_lr / (1 + self.decay_rate * step)
```

### Inverse Square Root Decay

Common in transformer training:

$$\eta_t = \eta_0 \cdot \frac{1}{\sqrt{t + 1}}$$

```python
class InverseSqrtDecay:
    """Inverse square root decay (common for transformers)."""
    
    def __init__(self, initial_lr: float, warmup_steps: int = 0):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
    
    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.initial_lr * step / max(1, self.warmup_steps)
        return self.initial_lr / math.sqrt(step - self.warmup_steps + 1)
```

## Summary

Exponential decay provides a simple, smooth alternative to step decay. While less commonly used in modern deep learning, it offers benefits when gradual, predictable decay is desired. The key is choosing $\gamma$ appropriately based on training length and desired final learning rate.

**Key takeaways:**

1. Choose $\gamma$ based on training length (higher for longer training)
2. Can compute $\gamma$ to reach specific final LR
3. Smoother than step decay but less commonly used
4. Consider alternatives like cosine annealing for modern architectures
