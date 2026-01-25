# Step Decay (StepLR and MultiStepLR)

## Overview

Step decay schedulers reduce the learning rate by a fixed factor at predetermined intervals. This approach is one of the oldest and most widely used learning rate scheduling strategies, appearing in many foundational deep learning papers.

## Mathematical Formulation

### StepLR

StepLR multiplies the learning rate by a decay factor $\gamma$ every `step_size` epochs:

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}$$

where:
- $\eta_t$ is the learning rate at epoch $t$
- $\eta_0$ is the initial learning rate
- $\gamma$ is the decay factor (typically 0.1)
- $k$ is the step size (number of epochs between decays)
- $\lfloor \cdot \rfloor$ is the floor function

**Example Progression** ($\eta_0 = 0.1$, $k = 30$, $\gamma = 0.1$):

| Epoch Range | Calculation | Learning Rate |
|-------------|-------------|---------------|
| 0-29 | $0.1 \times 0.1^0$ | 0.1 |
| 30-59 | $0.1 \times 0.1^1$ | 0.01 |
| 60-89 | $0.1 \times 0.1^2$ | 0.001 |
| 90-119 | $0.1 \times 0.1^3$ | 0.0001 |

### MultiStepLR

MultiStepLR is a generalization that allows decay at arbitrary milestone epochs:

$$\eta_t = \eta_0 \cdot \gamma^{n(t)}$$

where $n(t) = |\{m \in \text{milestones} : m \leq t\}|$ is the number of milestones reached by epoch $t$.

**Example Progression** ($\eta_0 = 0.1$, milestones $= [30, 80]$, $\gamma = 0.1$):

| Epoch Range | Milestones Passed | Learning Rate |
|-------------|-------------------|---------------|
| 0-29 | 0 | 0.1 |
| 30-79 | 1 | 0.01 |
| 80+ | 2 | 0.001 |

## Learning Rate Curve Visualization

```
Learning Rate (log scale)
    │
0.1 ├────────────┐
    │            │
    │            │
0.01├────────────┴────────────────────┐
    │                                  │
    │                                  │
0.001├─────────────────────────────────┴──────
    │
    └──────┬──────┬──────┬──────┬──────┬─────→ Epoch
           0     30     60     90    120
```

## PyTorch Implementation

### StepLR

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Create model and optimizer
model = create_model()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Create StepLR scheduler
scheduler = StepLR(
    optimizer,
    step_size=30,    # Decay every 30 epochs
    gamma=0.1        # Multiply LR by 0.1 at each step
)

# Training loop
for epoch in range(100):
    train_one_epoch(model, optimizer, train_loader)
    validate(model, val_loader)
    
    # Step the scheduler after each epoch
    scheduler.step()
    
    print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]:.6f}")
```

### MultiStepLR

```python
from torch.optim.lr_scheduler import MultiStepLR

# Create MultiStepLR scheduler
scheduler = MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],  # Decay at these epochs
    gamma=0.1                  # Decay factor
)

# Alternative: Different milestones for different training phases
# Early decay for rapid convergence
milestones_aggressive = [10, 20, 30]

# Spread out for longer training
milestones_conservative = [50, 100, 150]
```

### Custom Step Decay Implementation

```python
class CustomStepLR:
    """
    Custom step learning rate scheduler with detailed tracking.
    """
    
    def __init__(
        self,
        initial_lr: float,
        step_size: int,
        gamma: float = 0.1,
        min_lr: float = 1e-7
    ):
        """
        Args:
            initial_lr: Starting learning rate
            step_size: Epochs between LR reductions
            gamma: Multiplicative decay factor
            min_lr: Minimum learning rate floor
        """
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def get_lr(self, epoch: int = None) -> float:
        """Calculate learning rate for given epoch."""
        if epoch is None:
            epoch = self.current_epoch
        
        # Calculate number of decay steps
        n_decays = epoch // self.step_size
        
        # Calculate learning rate with floor
        lr = self.initial_lr * (self.gamma ** n_decays)
        return max(lr, self.min_lr)
    
    def step(self):
        """Advance to next epoch."""
        self.current_epoch += 1
    
    def get_schedule(self, total_epochs: int) -> list:
        """Generate full LR schedule for visualization."""
        return [self.get_lr(epoch) for epoch in range(total_epochs)]


class CustomMultiStepLR:
    """
    Custom multi-step learning rate scheduler.
    """
    
    def __init__(
        self,
        initial_lr: float,
        milestones: list,
        gamma: float = 0.1
    ):
        """
        Args:
            initial_lr: Starting learning rate
            milestones: List of epochs at which to decay LR
            gamma: Multiplicative decay factor
        """
        self.initial_lr = initial_lr
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.current_epoch = 0
    
    def get_lr(self, epoch: int = None) -> float:
        """Calculate learning rate for given epoch."""
        if epoch is None:
            epoch = self.current_epoch
        
        # Count milestones passed
        n_decays = sum(1 for m in self.milestones if epoch >= m)
        
        return self.initial_lr * (self.gamma ** n_decays)
    
    def step(self):
        """Advance to next epoch."""
        self.current_epoch += 1
```

## Practical Guidelines

### Choosing Step Size

The step size determines how frequently the learning rate drops:

| Training Length | Recommended Step Size | Rationale |
|-----------------|----------------------|-----------|
| 50 epochs | 15-20 epochs | 2-3 drops total |
| 100 epochs | 30-40 epochs | 2-3 drops total |
| 200 epochs | 50-70 epochs | 2-3 drops total |
| 300 epochs | 80-100 epochs | 3 drops total |

**Rule of thumb:** Plan for 2-4 learning rate drops during training.

### Choosing Gamma

The decay factor $\gamma$ controls the magnitude of each drop:

| Gamma | Drop Factor | Use Case |
|-------|-------------|----------|
| 0.1 | 10× | Standard choice, aggressive decay |
| 0.2 | 5× | Moderate decay |
| 0.5 | 2× | Gentle decay, more gradual convergence |

**Common choice:** $\gamma = 0.1$ is used in most papers.

### Choosing Milestones (MultiStepLR)

Common milestone patterns:

```python
# Pattern 1: Even spacing
# Good for unknown datasets
milestones = [epochs // 3, 2 * epochs // 3]  # [33, 66] for 100 epochs

# Pattern 2: Back-loaded (more time at high LR)
# Good for exploration
milestones = [epochs * 0.6, epochs * 0.8]  # [60, 80] for 100 epochs

# Pattern 3: Front-loaded (quick convergence)
# Good when dataset is well-understood
milestones = [epochs * 0.3, epochs * 0.6, epochs * 0.8]

# Pattern 4: ImageNet standard
# From ResNet paper
milestones = [30, 60, 90]  # for 100 epochs
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train_with_step_decay(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    initial_lr: float = 0.1,
    milestones: list = None,
    gamma: float = 0.1,
    device: str = 'cuda'
):
    """
    Train model with step decay learning rate schedule.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Total training epochs
        initial_lr: Starting learning rate
        milestones: Epochs for LR decay (default: [30, 60, 90])
        gamma: Decay factor
        device: Training device
    
    Returns:
        Dictionary with training history
    """
    if milestones is None:
        milestones = [30, 60, 90]
    
    model = model.to(device)
    
    # Optimizer with momentum and weight decay
    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # MultiStepLR scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # History tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(epochs):
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
        current_lr = scheduler.get_last_lr()[0]
        history['train_loss'].append(train_loss / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['train_acc'].append(100 * train_correct / train_total)
        history['val_acc'].append(100 * val_correct / val_total)
        history['lr'].append(current_lr)
        
        # Step scheduler
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch in milestones:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"LR: {current_lr:.6f} | "
                  f"Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Val Acc: {history['val_acc'][-1]:.2f}%")
    
    return history


def plot_training_history(history: dict, milestones: list):
    """Plot training curves with milestone markers."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Validation')
    for m in milestones:
        axes[0].axvline(x=m, color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], label='Train')
    axes[1].plot(epochs, history['val_acc'], label='Validation')
    for m in milestones:
        axes[1].axvline(x=m, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Advantages and Disadvantages

### Advantages

✅ **Simple and intuitive** - Easy to understand and implement

✅ **Predictable behavior** - Know exactly when LR will change

✅ **Well-established** - Proven effective in many landmark papers

✅ **Low overhead** - Minimal computational cost

✅ **Reproducible** - No stochastic elements

### Disadvantages

❌ **Sharp transitions** - Sudden LR drops can cause temporary instability

❌ **Requires tuning** - Must choose step_size and milestones

❌ **Not adaptive** - Doesn't respond to training dynamics

❌ **May be suboptimal** - Fixed schedule might not match training needs

## When to Use Step Decay

**Good use cases:**

- Reproducing results from papers that used step decay
- Image classification with standard architectures (ResNet, VGG)
- When you have prior knowledge about optimal decay points
- When simplicity is valued over optimal performance

**Avoid when:**

- Dataset or architecture is unfamiliar
- Training budget is very limited
- Maximum performance is critical
- Training dynamics are unpredictable

## Historical Context

Step decay became popular through its use in influential papers:

1. **AlexNet (2012)** - Manual LR division during training
2. **VGGNet (2014)** - Decay at specific epochs
3. **ResNet (2015)** - Decay at epochs 30, 60 for 90 total

The ImageNet training recipe with milestones [30, 60, 90] and $\gamma=0.1$ became a standard baseline for image classification.

## Comparison with Other Schedulers

| Aspect | StepLR | CosineAnnealing | OneCycleLR |
|--------|--------|-----------------|------------|
| Smoothness | Sharp drops | Smooth curve | Smooth cycle |
| Tuning effort | Medium | Low | Low |
| Adaptivity | None | None | None |
| Typical use | Classical | Modern | Fast training |

## Summary

Step decay schedulers provide a straightforward approach to learning rate scheduling through predetermined decay points. While more sophisticated methods often achieve better results, step decay remains relevant for its simplicity, reproducibility, and proven track record in computer vision tasks.

**Key takeaways:**

1. Plan for 2-4 decay events during training
2. Standard $\gamma = 0.1$ works well in most cases
3. MultiStepLR offers more flexibility than StepLR
4. Consider smoother alternatives for modern architectures
