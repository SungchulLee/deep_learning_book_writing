# ReduceLROnPlateau

## Overview

ReduceLROnPlateau is a reactive scheduler that reduces the learning rate when a monitored metric (typically validation loss) stops improving. Unlike other schedulers, it adapts to the training dynamics rather than following a predetermined schedule.

## PyTorch Implementation

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # 'min' for loss, 'max' for accuracy
    factor=0.1,          # New LR = old LR * factor
    patience=10,         # Wait 10 epochs before reducing
    threshold=1e-4,      # Minimum improvement to qualify
    min_lr=1e-7,         # Lower bound on LR
    verbose=True
)

for epoch in range(num_epochs):
    train_one_epoch(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # Must pass the monitored metric
```

## Parameters

- **`patience`**: Number of epochs with no improvement before reducing LR. Higher patience avoids premature reduction.
- **`factor`**: Multiplicative reduction factor. 0.1 (10× reduction) is aggressive; 0.5 (2× reduction) is gentler.
- **`threshold`**: Minimum change to qualify as an improvement. Prevents LR reduction from tiny fluctuations.
- **`cooldown`**: Number of epochs to wait after a reduction before resuming monitoring.

## Usage Pattern

ReduceLROnPlateau is useful when you don't know the optimal schedule a priori. It is commonly combined with early stopping—if the LR has been reduced multiple times and the metric still doesn't improve, stop training.

## Key Takeaways

- Adapts the learning rate based on actual training progress.
- Requires passing a monitored metric to `scheduler.step()`.
- Combine with early stopping for automated training management.
