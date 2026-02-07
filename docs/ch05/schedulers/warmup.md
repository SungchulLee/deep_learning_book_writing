# Warmup Strategies

## Overview

Learning rate warmup gradually increases the learning rate from a small initial value to the target value over the first few epochs or steps. Warmup stabilizes early training by preventing large, uninformed parameter updates when the model's weights are randomly initialized.

## Why Warmup?

At initialization, model outputs are nearly random. Large learning rates applied to random gradient directions cause erratic updates that can push parameters into poor regions of the loss landscape from which recovery is difficult. Warmup allows the optimizer's state (momentum, adaptive learning rates) to stabilize before applying the full learning rate.

## Linear Warmup

```python
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=10
)
# LR linearly increases from 0.01 * base_lr to base_lr over 10 epochs
```

## Warmup + Cosine Decay

The most common modern schedule combines linear warmup with cosine decay:

```python
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=95)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[5]
)
```

## Warmup Duration

Typical warmup durations: 5–10% of total training steps for transformers, 1–5 epochs for CNNs. The duration should be long enough for optimizer state to stabilize but short enough to not waste training budget.

## Key Takeaways

- Warmup prevents destabilizing large updates during early training.
- Linear warmup followed by cosine decay is the modern standard.
- RAdam provides implicit warmup, potentially eliminating the need for an explicit warmup schedule.
