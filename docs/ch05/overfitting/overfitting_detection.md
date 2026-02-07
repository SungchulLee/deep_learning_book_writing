# Overfitting Detection

## Overview

Detecting overfitting early prevents wasted computation and guides corrective action. The primary signal is divergence between training and validation performance.

## Training-Validation Gap

```python
for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    gap = val_loss - train_loss
    if gap > threshold:
        print(f"Warning: overfitting detected. Gap = {gap:.4f}")
```

## Visual Diagnostics

Plot training and validation loss curves. Common patterns:

- **Healthy training**: Both curves decrease, gap is small and stable.
- **Overfitting**: Training loss decreases, validation loss increases or plateaus.
- **Underfitting**: Both curves plateau at high values.
- **High variance**: Validation curve is noisy; training curve is smooth.

## Metric Divergence

Beyond loss, track task-specific metrics (accuracy, F1, Sharpe ratio) on both training and validation sets. Overfitting is confirmed when training metrics improve but validation metrics stagnate or degrade.

## Practical Thresholds

There is no universal threshold for the train-val gap. Rules of thumb: if validation loss increases for 5–10 consecutive epochs, overfitting is likely. In finance, if the in-sample Sharpe exceeds the out-of-sample Sharpe by more than 2×, the strategy is likely overfit.

## Key Takeaways

- Monitor the training-validation loss gap continuously.
- Visual inspection of loss curves is the most reliable diagnostic.
- Overfitting detection should trigger corrective actions: early stopping, increased regularization, or reduced model complexity.
