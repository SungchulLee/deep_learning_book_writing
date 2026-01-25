# Scheduler Comparison and Selection Guide

## Overview

Choosing the right learning rate scheduler can significantly impact training speed and final model performance. This guide provides a comprehensive comparison of all schedulers and practical guidance for selection.

## Quick Selection Decision Tree

```
Start Here
    │
    ├── Training transformers/LLMs?
    │   └── Yes → Warmup + Cosine Decay
    │
    ├── Fast training needed (<20 epochs)?
    │   └── Yes → OneCycleLR
    │
    ├── Unknown optimal schedule?
    │   └── Yes → ReduceLROnPlateau
    │
    ├── Long training (100+ epochs)?
    │   ├── Want ensembles? → SGDR
    │   └── Standard → Cosine Annealing
    │
    ├── Classic/proven approach?
    │   └── Yes → StepLR or MultiStepLR
    │
    └── Default choice → Cosine Annealing
```

## Comprehensive Comparison Table

| Scheduler | Adaptivity | Complexity | Speed | Best For |
|-----------|------------|------------|-------|----------|
| StepLR | None | Low | Medium | Classic training |
| ExponentialLR | None | Low | Medium | Smooth decay |
| CosineAnnealing | None | Low | Good | Modern training |
| OneCycleLR | None | Medium | Excellent | Fast training |
| CyclicLR | None | Medium | Good | Exploration |
| SGDR | None | Medium | Good | Long training |
| ReduceLROnPlateau | High | Low | Variable | Unknown tasks |
| Warmup + Decay | None | Medium | Good | Transformers |

## Detailed Scheduler Profiles

### StepLR / MultiStepLR

**Mathematical form:**
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/k \rfloor}$$

| Aspect | Rating | Notes |
|--------|--------|-------|
| Ease of use | ⭐⭐⭐⭐⭐ | Simplest to understand |
| Tuning effort | ⭐⭐⭐ | Need to choose milestones |
| Training speed | ⭐⭐⭐ | Moderate |
| Final accuracy | ⭐⭐⭐ | Good, not best |
| Robustness | ⭐⭐⭐⭐ | Proven, reliable |

**Recommended parameters:**
- `step_size`: Total epochs / 3-4
- `gamma`: 0.1 (standard), 0.5 (gentle)

---

### ExponentialLR

**Mathematical form:**
$$\eta_t = \eta_0 \cdot \gamma^t$$

| Aspect | Rating | Notes |
|--------|--------|-------|
| Ease of use | ⭐⭐⭐⭐⭐ | Very simple |
| Tuning effort | ⭐⭐⭐⭐ | Only gamma to tune |
| Training speed | ⭐⭐⭐ | Moderate |
| Final accuracy | ⭐⭐⭐ | Good |
| Robustness | ⭐⭐⭐⭐ | Reliable |

**Recommended parameters:**
- `gamma`: 0.95-0.99 depending on training length

---

### Cosine Annealing

**Mathematical form:**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t \cdot \pi}{T_{max}}))$$

| Aspect | Rating | Notes |
|--------|--------|-------|
| Ease of use | ⭐⭐⭐⭐ | Simple |
| Tuning effort | ⭐⭐⭐⭐⭐ | Almost none |
| Training speed | ⭐⭐⭐⭐ | Good |
| Final accuracy | ⭐⭐⭐⭐⭐ | Excellent |
| Robustness | ⭐⭐⭐⭐⭐ | Very reliable |

**Recommended parameters:**
- `T_max`: Total epochs
- `eta_min`: 0 or 1e-6

---

### OneCycleLR

**Mathematical form:** Linear warmup + cosine decay

| Aspect | Rating | Notes |
|--------|--------|-------|
| Ease of use | ⭐⭐⭐ | Need LR range test |
| Tuning effort | ⭐⭐⭐ | max_lr is critical |
| Training speed | ⭐⭐⭐⭐⭐ | Excellent (super-convergence) |
| Final accuracy | ⭐⭐⭐⭐⭐ | Excellent |
| Robustness | ⭐⭐⭐ | Sensitive to max_lr |

**Recommended parameters:**
- `max_lr`: From LR range test
- `pct_start`: 0.3
- `div_factor`: 25

---

### CyclicLR

**Mathematical form:** Oscillating between bounds

| Aspect | Rating | Notes |
|--------|--------|-------|
| Ease of use | ⭐⭐⭐ | Need to find LR bounds |
| Tuning effort | ⭐⭐⭐ | Multiple parameters |
| Training speed | ⭐⭐⭐⭐ | Good |
| Final accuracy | ⭐⭐⭐⭐ | Good |
| Robustness | ⭐⭐⭐ | Variable |

**Recommended parameters:**
- `base_lr`, `max_lr`: From LR range test
- `step_size`: 2-8 epochs
- `mode`: 'triangular2'

---

### SGDR (Warm Restarts)

**Mathematical form:** Cosine with periodic restarts

| Aspect | Rating | Notes |
|--------|--------|-------|
| Ease of use | ⭐⭐⭐ | More parameters |
| Tuning effort | ⭐⭐⭐ | T_0 and T_mult |
| Training speed | ⭐⭐⭐ | Longer training needed |
| Final accuracy | ⭐⭐⭐⭐⭐ | Excellent (with ensembles) |
| Robustness | ⭐⭐⭐⭐ | Good |

**Recommended parameters:**
- `T_0`: 10-20 epochs
- `T_mult`: 2

---

### ReduceLROnPlateau

**Mathematical form:** Reactive reduction

| Aspect | Rating | Notes |
|--------|--------|-------|
| Ease of use | ⭐⭐⭐⭐⭐ | Set and forget |
| Tuning effort | ⭐⭐⭐⭐ | Minimal |
| Training speed | ⭐⭐⭐ | Variable |
| Final accuracy | ⭐⭐⭐⭐ | Good |
| Robustness | ⭐⭐⭐⭐⭐ | Very robust |

**Recommended parameters:**
- `factor`: 0.1
- `patience`: 10
- `mode`: 'min' for loss

---

### Warmup + Decay

**Mathematical form:** Linear warmup + cosine decay

| Aspect | Rating | Notes |
|--------|--------|-------|
| Ease of use | ⭐⭐⭐ | Need warmup steps |
| Tuning effort | ⭐⭐⭐⭐ | Low once warmup set |
| Training speed | ⭐⭐⭐⭐ | Good |
| Final accuracy | ⭐⭐⭐⭐⭐ | Excellent |
| Robustness | ⭐⭐⭐⭐⭐ | Very robust |

**Recommended parameters:**
- `warmup_steps`: 5-10% of total
- Decay: Cosine to 0 or min_lr

## Use Case Recommendations

### By Model Type

| Model Type | Recommended Scheduler | Reason |
|------------|----------------------|--------|
| CNN (ResNet, etc.) | OneCycleLR or Cosine | Fast, effective |
| Transformer | Warmup + Cosine | Stability, standard |
| RNN/LSTM | ReduceLROnPlateau | Adaptive to convergence |
| GAN | StepLR | Stable, predictable |
| Fine-tuning | Cosine (small LR) | Gentle adaptation |

### By Training Duration

| Duration | Recommended Scheduler |
|----------|----------------------|
| Very short (<10 epochs) | OneCycleLR |
| Short (10-30 epochs) | OneCycleLR or Cosine |
| Medium (30-100 epochs) | Cosine Annealing |
| Long (100+ epochs) | SGDR or Cosine |
| Unknown | ReduceLROnPlateau |

### By Objective

| Goal | Recommended Scheduler |
|------|----------------------|
| Fastest training | OneCycleLR |
| Best accuracy | SGDR with ensembles |
| Lowest tuning effort | ReduceLROnPlateau |
| Production reliability | Cosine Annealing |
| Reproducibility | StepLR or Cosine |

## PyTorch Implementation Patterns

### Pattern 1: Standard Per-Epoch Stepping

```python
# For: StepLR, ExponentialLR, CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    train_one_epoch(model, optimizer, train_loader)
    scheduler.step()  # Step after each epoch
```

### Pattern 2: Per-Batch Stepping

```python
# For: OneCycleLR, CyclicLR
scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=epochs*len(loader))

for epoch in range(epochs):
    for batch in train_loader:
        train_step(batch)
        scheduler.step()  # Step after each batch
```

### Pattern 3: Metric-Based Stepping

```python
# For: ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10)

for epoch in range(epochs):
    train_one_epoch(model, optimizer, train_loader)
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)  # Step with metric
```

### Pattern 4: Combined Warmup + Main Scheduler

```python
# For: Transformer training
def get_lr(step, warmup_steps, base_lr, total_steps):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    # Cosine decay after warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

for step, batch in enumerate(dataloader):
    lr = get_lr(step, warmup_steps, base_lr, total_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    train_step(batch)
```

## Common Mistakes to Avoid

### Mistake 1: Wrong Stepping Frequency

```python
# WRONG for OneCycleLR
for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()  # Should be per batch!

# CORRECT
for epoch in range(epochs):
    for batch in train_loader:
        train_step(batch)
        scheduler.step()  # Per batch
```

### Mistake 2: Forgetting to Pass Metric

```python
# WRONG
scheduler = ReduceLROnPlateau(optimizer)
scheduler.step()  # Missing metric!

# CORRECT
scheduler.step(val_loss)
```

### Mistake 3: Not Scaling Warmup with Batch Size

```python
# WRONG - same warmup for all batch sizes
warmup_steps = 1000

# CORRECT - scale with batch size
warmup_steps = int(1000 * (batch_size / 32))
```

## Combining Schedulers

### Sequential Combination

```python
from torch.optim.lr_scheduler import SequentialLR

# Warmup then step decay
scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
scheduler2 = StepLR(optimizer, step_size=30, gamma=0.1)

scheduler = SequentialLR(
    optimizer,
    schedulers=[scheduler1, scheduler2],
    milestones=[warmup]
)
```

### Chained Combination

```python
from torch.optim.lr_scheduler import ChainedScheduler

# Combine multiple schedulers
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = StepLR(optimizer, step_size=30, gamma=0.5)

scheduler = ChainedScheduler([scheduler1, scheduler2])
```

## Quick Reference

### When to Use Each Scheduler

| Situation | Scheduler |
|-----------|-----------|
| "I want the fastest training" | OneCycleLR |
| "I'm training a transformer" | Warmup + Cosine |
| "I don't know what to use" | ReduceLROnPlateau |
| "I want something simple and proven" | CosineAnnealingLR |
| "I'm doing a research paper" | StepLR (reproducible) |
| "I'm training for 200+ epochs" | SGDR |
| "Training seems stuck" | CyclicLR or SGDR |

### Default Recommendations by Framework/Library

| Framework/Library | Default Choice |
|-------------------|----------------|
| fast.ai | OneCycleLR |
| Hugging Face | Warmup + Linear/Cosine |
| PyTorch ImageNet | StepLR or Cosine |
| Detectron2 | Warmup + StepLR |

## Summary

The choice of learning rate scheduler depends on your specific needs:

1. **For fast training**: OneCycleLR enables super-convergence
2. **For transformers**: Warmup + Cosine is the standard
3. **For unknown tasks**: ReduceLROnPlateau adapts automatically
4. **For long training**: SGDR with snapshots can improve results
5. **As a safe default**: Cosine Annealing works well in most cases

Remember:
- Match stepping frequency to scheduler type (batch vs epoch)
- Consider combining warmup with your main scheduler
- Use LR range test for OneCycleLR and CyclicLR
- Monitor validation metrics to validate scheduler choice
