# Custom Schedulers

## Overview

Custom schedulers enable domain-specific learning rate policies that go beyond built-in options. PyTorch provides `LambdaLR` for simple custom schedules and the `LRScheduler` base class for more complex implementations.

## LambdaLR

The simplest way to define a custom schedule:

```python
# Inverse square root decay (common for transformers)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(1.0, step / warmup_steps) *
                           (warmup_steps / max(step, warmup_steps)) ** 0.5
)
```

The `lr_lambda` function receives the current step/epoch and returns a multiplicative factor applied to the base learning rate.

## Custom Scheduler Class

For complex schedules, subclass `LRScheduler`:

```python
from torch.optim.lr_scheduler import LRScheduler

class WarmupCosineScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = self.last_epoch / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        return [self.min_lr + (base_lr - self.min_lr) * scale
                for base_lr in self.base_lrs]
```

## Financial Application: Regime-Adaptive Schedule

```python
class RegimeAdaptiveScheduler(LRScheduler):
    """Adjust LR based on rolling market volatility."""
    def __init__(self, optimizer, volatility_fn, base_vol=0.15):
        self.volatility_fn = volatility_fn
        self.base_vol = base_vol
        super().__init__(optimizer)

    def get_lr(self):
        current_vol = self.volatility_fn()
        # Reduce LR in high-volatility regimes (noisier gradients)
        vol_ratio = self.base_vol / max(current_vol, 1e-6)
        scale = min(1.0, vol_ratio)
        return [base_lr * scale for base_lr in self.base_lrs]
```

## Key Takeaways

- `LambdaLR` handles most custom schedules with a single function.
- Subclass `LRScheduler` for stateful or complex scheduling logic.
- Custom schedulers can incorporate domain knowledge (e.g., market regime indicators) into the learning rate policy.
