# Scheduler Overview

## Overview

Learning rate schedulers adjust the learning rate during training according to a predefined policy. A well-chosen schedule can significantly improve both convergence speed and final model quality.

## Motivation

A fixed learning rate faces a fundamental tension: a large learning rate enables fast early progress but prevents fine-grained convergence near a minimum, while a small learning rate converges precisely but wastes computation in early epochs. Schedulers resolve this by varying the learning rate over training.

## PyTorch Scheduler Interface

All schedulers inherit from `torch.optim.lr_scheduler.LRScheduler`:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train_one_epoch(...)
    val_loss = validate(...)
    scheduler.step()  # Update learning rate
```

The scheduler modifies the learning rate stored in `optimizer.param_groups[i]['lr']`.

## Scheduler Placement

**After epoch** (most common):

```python
for epoch in range(num_epochs):
    train_one_epoch(...)
    scheduler.step()
```

**After batch** (for OneCycleLR, cosine annealing with restarts):

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        train_step(...)
        scheduler.step()
```

**After validation** (for ReduceLROnPlateau):

```python
for epoch in range(num_epochs):
    train_one_epoch(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # Requires metric
```

## Scheduler Taxonomy

| Category | Schedulers | Behavior |
|----------|-----------|----------|
| Decay | StepLR, MultiStepLR, ExponentialLR | Monotonically decreasing |
| Cyclic | CosineAnnealing, OneCycleLR | Non-monotonic, periodic |
| Adaptive | ReduceLROnPlateau | Responds to training dynamics |
| Warmup | LinearWarmup, custom | Increasing from zero |

## Chaining Schedulers

Multiple schedulers can be chained with `SequentialLR` or `ChainedScheduler`:

```python
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=10)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[10])
```

## Key Takeaways

- Learning rate scheduling resolves the fixed-LR tension between exploration and exploitation.
- Most schedulers are called after each epoch; some are called after each batch.
- Schedulers can be chained to combine warmup with decay policies.
- The following sections detail each scheduler with mathematical formulations and usage patterns.
