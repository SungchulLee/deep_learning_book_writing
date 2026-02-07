# Early Stopping

## Overview

Early stopping is a regularization technique that halts training when the model's performance on a validation set stops improving. This prevents overfitting by identifying the point where the model has learned generalizable patterns but hasn't yet begun memorizing training noise.

## Conceptual Foundation

### The Overfitting Trajectory

During training, models typically follow a predictable trajectory:

1. **Initial phase**: Both training and validation loss decrease rapidly
2. **Learning phase**: Training loss continues decreasing; validation loss decreases more slowly
3. **Overfitting phase**: Training loss decreases; validation loss increases

Early stopping identifies the transition point between phases 2 and 3.

### Implicit Regularization

Early stopping acts as an implicit regularizer by:

- Limiting model complexity through restricted optimization iterations
- Preventing weights from reaching extreme values
- Keeping the model in a region of parameter space closer to initialization

For linear models, early stopping with gradient descent is mathematically equivalent to L2 regularization, where the effective regularization strength is inversely proportional to the number of iterations.

## Mathematical Formulation

### Validation-Based Stopping Criterion

Let $\mathcal{L}_{\text{val}}^{(t)}$ denote the validation loss at epoch $t$. The basic stopping criterion:

$$
\text{Stop if } \mathcal{L}_{\text{val}}^{(t)} > \mathcal{L}_{\text{val}}^{(t-1)} \text{ for } k \text{ consecutive epochs}
$$

where $k$ is the **patience** parameter.

### Best Model Selection

Track the best validation performance:

$$
t^* = \arg\min_{t \leq T} \mathcal{L}_{\text{val}}^{(t)}
$$

Return the model parameters $\theta^{(t^*)}$ rather than the final parameters $\theta^{(T)}$.

### Generalization Bound Perspective

Early stopping provides implicit regularization. For linear regression with gradient descent, stopping at iteration $t$ is equivalent to Ridge regression with:

$$
\lambda_{\text{eff}} \approx \frac{1}{\eta t}
$$

where $\eta$ is the learning rate.

## PyTorch Implementation

### Basic Early Stopping

```python
import torch
import numpy as np
from typing import Optional
import copy

class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait after last improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss (lower is better), 'max' for accuracy
        restore_best_weights: Whether to restore best model weights
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta
    
    def __call__(self, score: float, model: torch.nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially save
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False
    
    def get_best_score(self) -> Optional[float]:
        return self.best_score
    
    def get_best_epoch(self) -> int:
        return self.best_epoch
```

### Training Loop with Early Stopping

```python
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    max_epochs: int = 1000,
    patience: int = 20,
    min_delta: float = 1e-4,
    verbose: bool = True
) -> dict:
    """
    Train model with early stopping.
    
    Args:
        model: Neural network
        train_loader: Training data loader
        val_loader: Validation data loader  
        criterion: Loss function
        optimizer: Optimizer
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        min_delta: Minimum improvement threshold
        verbose: Print progress
        
    Returns:
        Training history
    """
    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode='min',
        restore_best_weights=True
    )
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Check early stopping
        if early_stopping(val_loss, model, epoch):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
                print(f"Best epoch: {early_stopping.get_best_epoch()+1} "
                      f"with val_loss: {early_stopping.get_best_score():.4f}")
            break
    
    return history
```

### Advanced Early Stopping with Multiple Criteria

```python
class MultiMetricEarlyStopping:
    """
    Early stopping based on multiple metrics.
    
    Stops when ALL monitored metrics stop improving.
    """
    
    def __init__(
        self,
        metrics_config: dict,
        patience: int = 10,
        restore_best_weights: bool = True
    ):
        """
        Args:
            metrics_config: Dict mapping metric names to 'min' or 'max'
                           e.g., {'val_loss': 'min', 'val_acc': 'max'}
            patience: Patience for each metric
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        
        self.metrics_config = metrics_config
        self.best_scores = {name: None for name in metrics_config}
        self.counters = {name: 0 for name in metrics_config}
        self.best_weights = None
        self.best_epoch = 0
    
    def _is_better(self, name: str, current: float, best: float) -> bool:
        mode = self.metrics_config[name]
        if mode == 'min':
            return current < best
        return current > best
    
    def __call__(self, metrics: dict, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            metrics: Dict of current metric values
            model: Model to save
            epoch: Current epoch
            
        Returns:
            True if should stop
        """
        any_improved = False
        
        for name in self.metrics_config:
            current = metrics[name]
            best = self.best_scores[name]
            
            if best is None or self._is_better(name, current, best):
                self.best_scores[name] = current
                self.counters[name] = 0
                any_improved = True
            else:
                self.counters[name] += 1
        
        if any_improved:
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        
        # Stop if ALL metrics exceeded patience
        all_exceeded = all(c >= self.patience for c in self.counters.values())
        
        if all_exceeded and self.restore_best_weights:
            model.load_state_dict(self.best_weights)
        
        return all_exceeded
```

### Early Stopping with Learning Rate Scheduling

```python
class EarlyStoppingWithLRScheduler:
    """
    Combines early stopping with learning rate reduction on plateau.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        patience: int = 20,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        min_lr: float = 1e-7,
        min_delta: float = 1e-4
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.lr_counter = 0
        self.num_lr_reductions = 0
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            return False
        
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.lr_counter = 0
        else:
            self.counter += 1
            self.lr_counter += 1
            
            # Reduce LR if plateau
            if self.lr_counter >= self.lr_patience:
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * self.lr_factor, self.min_lr)
                
                if new_lr < current_lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"Reducing LR to {new_lr:.2e}")
                    self.num_lr_reductions += 1
                
                self.lr_counter = 0
        
        # Stop if patience exceeded
        if self.counter >= self.patience:
            model.load_state_dict(self.best_weights)
            return True
        
        return False
```

## Hyperparameter Considerations

### Choosing Patience

The patience parameter controls the trade-off between:

- **Too low**: May stop prematurely during temporary fluctuations
- **Too high**: May waste computation and risk overfitting

Guidelines:
- Start with patience = 10-20 epochs
- Increase for noisy validation metrics
- Decrease for smooth, predictable learning curves

```python
def analyze_optimal_patience(history: dict, test_patience_values: list):
    """
    Analyze what stopping point different patience values would yield.
    
    Args:
        history: Training history with 'val_loss'
        test_patience_values: List of patience values to test
        
    Returns:
        Dict mapping patience to stopping epoch and best val_loss
    """
    val_losses = history['val_loss']
    results = {}
    
    for patience in test_patience_values:
        best_loss = float('inf')
        best_epoch = 0
        counter = 0
        
        for epoch, loss in enumerate(val_losses):
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                counter = 0
            else:
                counter += 1
            
            if counter >= patience:
                break
        
        results[patience] = {
            'stop_epoch': epoch,
            'best_epoch': best_epoch,
            'best_val_loss': best_loss
        }
    
    return results
```

### Choosing the Monitored Metric

| Metric | When to Use |
|--------|-------------|
| Validation loss | Default choice; directly measures generalization |
| Validation accuracy | When accuracy is the primary goal |
| F1 score | For imbalanced classification |
| Custom metric | Domain-specific requirements |

### Minimum Delta Selection

The `min_delta` parameter defines what counts as improvement:

```python
# For typical loss values around 0.1-1.0
min_delta = 1e-4  # Default

# For very small loss values (< 0.01)
min_delta = 1e-5

# For noisy validation metrics
min_delta = 1e-3  # More tolerant
```

## Theoretical Analysis

### Connection to L2 Regularization

For gradient descent on linear regression, stopping at iteration $t$ gives:

$$
\hat{w}_t = \sum_{i=1}^{t} (I - \eta X^T X)^{i-1} \eta X^T y
$$

This converges to the Ridge solution as $t \to \infty$:

$$
\hat{w}_\infty = (X^T X)^{-1} X^T y
$$

The effective regularization is approximately:

$$
\hat{w}_t \approx (X^T X + \frac{1}{\eta t} I)^{-1} X^T y
$$

### Bias-Variance Trade-off

Early stopping affects the bias-variance decomposition:

- **Early stopping (small $t$)**: Higher bias, lower variance
- **Late stopping (large $t$)**: Lower bias, higher variance

The optimal stopping point minimizes total generalization error.

## Practical Guidelines

### When to Use Early Stopping

1. **Always**: It's essentially free and often helps
2. **Limited compute**: Saves unnecessary training time
3. **Unclear training length**: When you don't know optimal epochs
4. **Overfitting observed**: Gap between train and validation performance

### Best Practices

```python
def recommended_training_setup(model, train_loader, val_loader):
    """
    Recommended setup combining early stopping with other techniques.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=20,
        min_delta=1e-4,
        restore_best_weights=True
    )
    
    max_epochs = 500  # Upper bound
    
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Check early stopping
        if early_stopping(val_loss, model, epoch):
            print(f"Stopped at epoch {epoch+1}")
            break
    
    return model
```

### Checkpointing

Always save checkpoints alongside early stopping:

```python
class CheckpointingEarlyStopping(EarlyStopping):
    """Early stopping with periodic checkpointing."""
    
    def __init__(self, checkpoint_dir: str = './checkpoints', 
                 checkpoint_freq: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def __call__(self, score, model, epoch):
        # Save periodic checkpoint
        if (epoch + 1) % self.checkpoint_freq == 0:
            path = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'score': score
            }, path)
        
        # Save best model
        should_stop = super().__call__(score, model, epoch)
        
        if self.best_weights is not None:
            path = f"{self.checkpoint_dir}/best_model.pt"
            torch.save({
                'epoch': self.best_epoch,
                'model_state_dict': self.best_weights,
                'score': self.best_score
            }, path)
        
        return should_stop
```

## Visualization

```python
import matplotlib.pyplot as plt

def plot_training_with_early_stopping(history: dict, best_epoch: int):
    """
    Visualize training progress with early stopping point.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0].axvline(best_epoch + 1, color='r', linestyle='--', 
                    label=f'Best Epoch ({best_epoch + 1})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot (if available)
    if 'train_acc' in history:
        axes[1].plot(epochs, history['train_acc'], label='Train Acc')
        axes[1].plot(epochs, history['val_acc'], label='Val Acc')
        axes[1].axvline(best_epoch + 1, color='r', linestyle='--',
                        label=f'Best Epoch ({best_epoch + 1})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## Common Pitfalls

1. **Not using validation set**: Early stopping requires held-out data
2. **Patience too low**: Stopping during normal fluctuations
3. **Not restoring best weights**: Using final weights instead of best
4. **Ignoring the metric**: Using wrong metric for the task
5. **Data leakage**: Validation set contaminated by training data

## References

1. Prechelt, L. (1998). Early Stopping - But When? *Neural Networks: Tricks of the Trade*, 55-69.
2. Yao, Y., Rosasco, L., & Caponnetto, A. (2007). On Early Stopping in Gradient Descent Learning. *Constructive Approximation*, 26(2), 289-315.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 7.
