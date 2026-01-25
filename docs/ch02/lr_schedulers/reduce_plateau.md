# ReduceLROnPlateau

## Overview

ReduceLROnPlateau is an **adaptive learning rate scheduler** that monitors a specified metric and reduces the learning rate when the metric stops improving. Unlike other schedulers with predetermined schedules, this approach responds dynamically to training progress, making it an excellent choice when optimal scheduling is unknown.

## How It Works

The scheduler monitors a validation metric (typically loss or accuracy) and reduces the learning rate when improvement stalls:

```
Logic:
    IF metric has not improved for 'patience' epochs:
        new_lr = current_lr × factor
        reset patience counter
```

This adaptive approach means the scheduler:
- Waits patiently during normal training
- Reduces LR only when truly needed
- Can make multiple reductions as training progresses

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `mode` | 'min' (loss) or 'max' (accuracy) | 'min' for loss |
| `factor` | Multiplication factor | 0.1 to 0.5 |
| `patience` | Epochs to wait | 5 to 20 |
| `threshold` | Minimum change for improvement | 1e-4 |
| `min_lr` | Lower bound on LR | 1e-7 |
| `cooldown` | Epochs after reduction before monitoring | 0 to 5 |

## Mathematical Behavior

The scheduler applies:

$$\eta_{new} = \eta_{current} \times \text{factor}$$

when:

$$\text{best\_metric} - \text{current\_metric} < \text{threshold}$$

for `patience` consecutive epochs.

## PyTorch Implementation

### Basic Usage

```python
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',           # 'min' for loss, 'max' for accuracy
    factor=0.1,           # LR reduction factor
    patience=10,          # Epochs to wait
    threshold=1e-4,       # Minimum improvement
    threshold_mode='rel', # 'rel' or 'abs'
    cooldown=0,           # Epochs after reduction
    min_lr=1e-7,          # Minimum LR
    verbose=True          # Print when LR changes
)

# Training loop - step with metric!
for epoch in range(epochs):
    train_loss = train_one_epoch(model, optimizer, train_loader)
    val_loss = validate(model, val_loader)
    
    # Step scheduler with validation loss
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, "
          f"LR = {optimizer.param_groups[0]['lr']:.6f}")
```

### Using with Accuracy

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',        # Higher is better for accuracy
    factor=0.5,        # Gentler reduction
    patience=5,        # Shorter patience
    threshold=0.001    # 0.1% improvement threshold
)

# Training loop
for epoch in range(epochs):
    train_one_epoch(model, optimizer, train_loader)
    val_acc = validate_accuracy(model, val_loader)
    
    # Step with accuracy metric
    scheduler.step(val_acc)
```

### Custom Implementation

```python
class ReduceLROnPlateau:
    """
    Custom implementation of ReduceLROnPlateau scheduler.
    """
    
    def __init__(
        self,
        initial_lr: float,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: float = 0,
        verbose: bool = False
    ):
        """
        Args:
            initial_lr: Starting learning rate
            mode: 'min' or 'max'
            factor: Factor to multiply LR by
            patience: Epochs without improvement before reduction
            threshold: Minimum change to qualify as improvement
            threshold_mode: 'rel' (relative) or 'abs' (absolute)
            cooldown: Epochs to wait after reduction
            min_lr: Lower bound on LR
            verbose: Print LR changes
        """
        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")
        if factor >= 1.0:
            raise ValueError("factor should be < 1.0")
        
        self.lr = initial_lr
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.verbose = verbose
        
        # State
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_epoch = 0
    
    def is_better(self, current: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min':
            if self.threshold_mode == 'rel':
                return current < self.best * (1 - self.threshold)
            else:
                return current < self.best - self.threshold
        else:  # mode == 'max'
            if self.threshold_mode == 'rel':
                return current > self.best * (1 + self.threshold)
            else:
                return current > self.best + self.threshold
    
    def step(self, metric: float) -> float:
        """
        Update scheduler with new metric value.
        
        Args:
            metric: Current metric value (loss or accuracy)
        
        Returns:
            Current learning rate
        """
        self.last_epoch += 1
        
        # Check cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
            return self.lr
        
        # Check improvement
        if self.is_better(metric):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        # Reduce LR if patience exceeded
        if self.num_bad_epochs > self.patience:
            old_lr = self.lr
            self.lr = max(self.lr * self.factor, self.min_lr)
            
            if self.verbose and self.lr < old_lr:
                print(f"Reducing LR: {old_lr:.2e} -> {self.lr:.2e}")
            
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        
        return self.lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.lr
    
    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            'lr': self.lr,
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'cooldown_counter': self.cooldown_counter,
            'last_epoch': self.last_epoch
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.lr = state_dict['lr']
        self.best = state_dict['best']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.cooldown_counter = state_dict['cooldown_counter']
        self.last_epoch = state_dict['last_epoch']
```

## Practical Guidelines

### Choosing Mode

| Metric Type | Mode | Example |
|-------------|------|---------|
| Loss (lower = better) | 'min' | val_loss, MSE |
| Accuracy (higher = better) | 'max' | accuracy, F1 |
| Error rate | 'min' | top-1 error |

### Choosing Factor

| Factor | Effect | Use Case |
|--------|--------|----------|
| 0.1 | Strong reduction (10×) | Default, aggressive |
| 0.2 | Moderate reduction (5×) | Balanced approach |
| 0.5 | Gentle reduction (2×) | Conservative |

### Choosing Patience

| Training Length | Recommended Patience |
|-----------------|---------------------|
| Short (<50 epochs) | 3-5 |
| Medium (50-100) | 5-10 |
| Long (>100) | 10-20 |

**Rule of thumb:** Patience should be long enough to allow for normal fluctuations but short enough to respond to genuine plateaus.

### Using Cooldown

Cooldown prevents multiple rapid reductions:

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    patience=10,
    cooldown=5  # Wait 5 epochs after each reduction
)
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_with_plateau_scheduler(
    model: nn.Module,
    train_loader,
    val_loader,
    max_epochs: int = 100,
    initial_lr: float = 0.001,
    patience: int = 10,
    factor: float = 0.1,
    min_lr: float = 1e-7,
    early_stop_patience: int = 20,
    device: str = 'cuda'
):
    """
    Train model with ReduceLROnPlateau and early stopping.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        max_epochs: Maximum training epochs
        initial_lr: Starting learning rate
        patience: LR reduction patience
        factor: LR reduction factor
        min_lr: Minimum learning rate
        early_stop_patience: Early stopping patience
        device: Training device
    
    Returns:
        Training history dictionary
    """
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=patience,
        threshold=1e-4,
        min_lr=min_lr,
        verbose=True
    )
    
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
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
        
        # Compute metrics
        train_loss_avg = train_loss / train_total
        val_loss_avg = val_loss / val_total
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Step scheduler with validation loss
        scheduler.step(val_loss_avg)
        
        # Early stopping check
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_without_improvement = 0
            # Save best model
            best_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1
        
        # Print progress
        print(f"Epoch {epoch+1}/{max_epochs} | "
              f"LR: {current_lr:.2e} | "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss_avg:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        # Check if LR is at minimum
        if current_lr <= min_lr:
            print(f"\nLearning rate reached minimum: {min_lr}")
    
    # Restore best model
    model.load_state_dict(best_state)
    
    return history


def plot_plateau_training(history: dict):
    """Plot training curves showing LR reductions."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train')
    axes[1].plot(epochs, history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate (log scale)
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    # Mark LR reduction points
    for i in range(1, len(history['lr'])):
        if history['lr'][i] < history['lr'][i-1]:
            for ax in axes:
                ax.axvline(x=i+1, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Combining with Early Stopping

ReduceLROnPlateau pairs naturally with early stopping:

```python
class EarlyStoppingWithLRScheduler:
    """
    Combined early stopping and LR scheduling.
    """
    
    def __init__(
        self,
        optimizer,
        patience_lr: int = 10,
        patience_stop: int = 20,
        factor: float = 0.1,
        min_delta: float = 1e-4,
        min_lr: float = 1e-7
    ):
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience_lr,
            threshold=min_delta,
            min_lr=min_lr
        )
        self.patience_stop = patience_stop
        self.min_delta = min_delta
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None
    
    def step(self, val_loss, model):
        """
        Update scheduler and check early stopping.
        
        Returns:
            bool: True if should stop
        """
        # Update LR scheduler
        self.scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict().copy()
        else:
            self.counter += 1
        
        return self.counter >= self.patience_stop
    
    def restore_best(self, model):
        """Restore model to best state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
```

## Advantages and Disadvantages

### Advantages

✅ **Adaptive** - Responds to actual training dynamics

✅ **Requires no schedule planning** - Works without knowing optimal decay points

✅ **Safe default** - Hard to get wrong

✅ **Automatic** - Reduces tuning burden

✅ **Pairs with early stopping** - Natural combination

### Disadvantages

❌ **Reactive, not proactive** - May reduce LR too late

❌ **Can be conservative** - May not reduce when beneficial

❌ **Requires validation metric** - Extra computation per epoch

❌ **Sensitive to noise** - Fluctuating metrics can trigger false positives

## When to Use ReduceLROnPlateau

**Ideal use cases:**

- Unknown or new datasets
- Exploratory experiments
- Production systems requiring robustness
- When unsure of optimal schedule
- Combined with early stopping

**Consider alternatives when:**

- Training time is critical (use OneCycleLR)
- You know optimal decay points (use StepLR)
- Training transformers (use warmup + cosine)
- Maximum performance is needed

## Comparison with Other Schedulers

| Aspect | ReduceLROnPlateau | StepLR | CosineAnnealing |
|--------|------------------|--------|-----------------|
| Adaptivity | High | None | None |
| Tuning effort | Low | Medium | Low |
| Schedule planning | None | Required | Required |
| Predictability | Low | High | High |
| Best for | Unknown datasets | Known schedules | Modern training |

## Summary

ReduceLROnPlateau provides an adaptive, safe approach to learning rate scheduling that responds to actual training progress. It's an excellent default choice when optimal scheduling is unknown.

**Key takeaways:**

1. Use 'min' mode for loss, 'max' for accuracy
2. Default factor=0.1 is often good
3. Patience should balance responsiveness and stability
4. Pairs naturally with early stopping
5. Great for exploratory work and production systems
