# Metrics Tracking

## Overview

Systematic metrics tracking transforms training from an opaque process into a transparent, diagnosable one. By recording losses, accuracies, learning rates, and gradient statistics over time, practitioners can detect problems early and make informed decisions about hyperparameters.

## Manual Tracking

The simplest approach accumulates metrics in Python data structures:

```python
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
```

## Running Averages

Within an epoch, track running averages rather than per-batch values to smooth noise:

```python
class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


loss_meter = AverageMeter()
for x, y in train_loader:
    loss = loss_fn(model(x), y)
    loss_meter.update(loss.item(), x.size(0))

print(f"Epoch average loss: {loss_meter.avg:.4f}")
```

## Gradient Statistics

Monitoring gradient norms helps diagnose vanishing or exploding gradients:

```python
def compute_gradient_stats(model):
    total_norm = 0.0
    max_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            max_norm = max(max_norm, param_norm)
    total_norm = total_norm ** 0.5
    return {'grad_norm': total_norm, 'grad_max': max_norm}
```

## Learning Rate Tracking

When using learning rate schedulers, track the current learning rate:

```python
current_lr = optimizer.param_groups[0]['lr']
history['learning_rate'].append(current_lr)
```

## Diagnostic Patterns

| Observation | Likely Cause | Action |
|---|---|---|
| Train loss not decreasing | LR too low, bug in loop | Increase LR, check code |
| Train loss oscillating wildly | LR too high | Decrease LR |
| Val loss diverges from train loss | Overfitting | Add regularization, reduce model |
| Gradient norm exploding | Unstable optimization | Add gradient clipping |
| Gradient norm near zero | Vanishing gradients | Check architecture, initialization |

## Key Takeaways

- Track train/validation loss, accuracy, learning rate, and gradient norms at minimum.
- Use `AverageMeter` for stable within-epoch statistics.
- Gradient monitoring detects vanishing/exploding gradients before they cause training failure.
- Systematic logging enables post-hoc analysis and comparison across experiments.
