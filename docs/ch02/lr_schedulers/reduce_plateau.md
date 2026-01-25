# ReduceLROnPlateau

ReduceLROnPlateau is an adaptive learning rate scheduler that monitors a metric (typically validation loss) and reduces the learning rate when improvement stalls. Unlike time-based schedulers, it responds to actual training dynamics.

## Motivation

Time-based schedules (step, cosine, exponential) assume fixed training dynamics:
- They reduce LR at predetermined points
- No adaptation to actual convergence
- May reduce too early or too late

ReduceLROnPlateau adapts to the model's learning:
- Reduces LR only when progress stalls
- Can handle varying convergence speeds
- Naturally accommodates different datasets and architectures

## How It Works

The scheduler monitors a metric over epochs:

1. Track the metric (e.g., validation loss) each epoch
2. If no improvement for `patience` epochs → reduce LR
3. Continue until minimum LR is reached

```
Epoch 1-5:   Loss ↓ (improving)     → Keep LR
Epoch 6-10:  Loss → (plateau)       → Patience counting
Epoch 11:    Still plateau          → Reduce LR
Epoch 12-15: Loss ↓ (improving)     → Reset patience
...
```

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',        # 'min' for loss, 'max' for accuracy
    factor=0.5,        # Multiply LR by this factor
    patience=5,        # Wait this many epochs without improvement
    threshold=1e-4,    # Minimum change to qualify as improvement
    threshold_mode='rel',  # 'rel' or 'abs'
    cooldown=0,        # Wait this many epochs after LR reduction
    min_lr=1e-6,       # Don't reduce below this
    verbose=True       # Print when LR changes
)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | 'min' | 'min' to reduce when metric stops decreasing |
| `factor` | 0.1 | Multiply LR by this factor when reducing |
| `patience` | 10 | Epochs to wait before reducing |
| `threshold` | 1e-4 | Minimum improvement to reset patience |
| `min_lr` | 0 | Lower bound for learning rate |
| `cooldown` | 0 | Epochs to wait after reduction before resuming monitoring |

## Training Loop Integration

Unlike other schedulers, `ReduceLROnPlateau.step()` requires the metric:

```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch['input'])
        loss = criterion(output, batch['target'])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            output = model(batch['input'])
            val_loss += criterion(output, batch['target']).item()
    
    val_loss /= len(val_loader)
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)  # Pass the metric here!
    
    print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
          f"Val Loss = {val_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
```

## Simulated Example

```python
# Simulate training with plateaus
simulated_val_losses = (
    [2.0, 1.8, 1.6, 1.4, 1.2] +      # Improving
    [1.2, 1.2, 1.2, 1.2, 1.2, 1.2] + # Plateau (triggers reduction)
    [1.0, 0.9, 0.8, 0.7] +            # Improving after reduction
    [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]   # Another plateau
)

optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

for epoch, val_loss in enumerate(simulated_val_losses):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}: Val Loss = {val_loss:.2f}, LR = {current_lr:.6f}")
    scheduler.step(val_loss)
```

Output shows LR reductions at plateau points.

## Choosing Parameters

### patience

- **Low (3-5)**: Quick adaptation, may be too aggressive
- **Medium (10)**: Good default for most tasks
- **High (20+)**: For noisy metrics or long training

### factor

- **0.1**: Aggressive reduction (default)
- **0.5**: More gradual reduction
- **0.2-0.5**: Common practical range

### threshold

Controls what counts as "improvement":
- `threshold=1e-4, threshold_mode='rel'`: 0.01% relative improvement
- `threshold=1e-4, threshold_mode='abs'`: Absolute improvement of 0.0001

## Comparison with Time-Based Schedulers

| Aspect | ReduceLROnPlateau | Time-Based |
|--------|-------------------|------------|
| Adaptation | Responds to training | Fixed schedule |
| Configuration | Based on dynamics | Based on epochs |
| Reproducibility | May vary across runs | Deterministic |
| Best for | Unknown training dynamics | Well-characterized tasks |

## Combining with Early Stopping

ReduceLROnPlateau pairs naturally with early stopping:

```python
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 20

for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    # LR scheduling
    scheduler.step(val_loss)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

## When to Use ReduceLROnPlateau

**Good choices:**
- Exploratory training with unknown dynamics
- Fine-tuning pretrained models
- When training time is flexible
- Combined with early stopping

**Consider alternatives when:**
- Training dynamics are well understood
- Need deterministic, reproducible schedules
- Using very short training runs
- Strict computational budgets

## Key Takeaways

ReduceLROnPlateau provides adaptive learning rate scheduling by monitoring validation metrics and reducing LR when improvement stalls. It requires passing the monitored metric to `scheduler.step(metric)`. Key parameters are `patience` (epochs to wait) and `factor` (reduction multiplier). It pairs naturally with early stopping for efficient training. Use it when training dynamics are unknown or highly variable across experiments.
