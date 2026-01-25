# Warmup Strategies

Learning rate warmup gradually increases the learning rate from a small initial value to the target learning rate during early training. This technique has become essential for training large models, particularly transformers.

## Why Warmup Matters

At the start of training, several factors create instability:

1. **Random initialization**: Model parameters are randomly initialized, producing unreliable gradients
2. **Adaptive optimizer state**: Adam's moment estimates need time to stabilize
3. **Batch normalization**: Running statistics are inaccurate initially
4. **Large models**: More parameters amplify initialization noise

Starting with a high learning rate can cause:
- Divergence (loss explodes to infinity)
- Poor local minima (bad early updates lock in suboptimal solutions)
- Unstable training dynamics

## Warmup Types

### Linear Warmup

Linearly increase LR from near-zero to target:

$$\eta_t = \eta_{\text{target}} \cdot \frac{t}{T_{\text{warmup}}}$$

```python
from torch.optim.lr_scheduler import LinearLR

optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Warmup from 0.1× to 1× of target LR over 10 epochs
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,    # Start at 10% of LR
    end_factor=1.0,      # End at 100% of LR
    total_iters=10       # Over 10 epochs
)
```

### Exponential Warmup

Exponentially increase LR (less common):

$$\eta_t = \eta_{\text{target}} \cdot \gamma^{T_{\text{warmup}} - t}$$

### Gradual Warmup (Custom)

Manual implementation for fine control:

```python
def get_warmup_lr(epoch, warmup_epochs, base_lr):
    """Linear warmup schedule."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr

# Usage
for epoch in range(num_epochs):
    if epoch < warmup_epochs:
        lr = get_warmup_lr(epoch, warmup_epochs, base_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # ... training code ...
```

## Warmup + Decay Schedules

The standard pattern combines warmup with a decay schedule:

```
LR
↑
│    /‾‾‾‾\
│   /      \
│  /        ╲
│ /          ╲
│/            ╲____
└─────────────────→ Epochs
  Warmup  │  Decay
```

### Implementation with SequentialLR

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Phase 1: Linear warmup for 10 epochs
warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)

# Phase 2: Cosine decay for remaining 90 epochs  
decay = CosineAnnealingLR(optimizer, T_max=90, eta_min=1e-6)

# Combine into single scheduler
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, decay],
    milestones=[10]  # Switch from warmup to decay at epoch 10
)

for epoch in range(100):
    train_epoch(model, train_loader, optimizer)
    scheduler.step()
    print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.6f}")
```

### Manual Implementation

For more control or complex schedules:

```python
def get_lr_with_warmup(epoch, warmup_epochs, total_epochs, base_lr, min_lr):
    """Warmup + cosine annealing."""
    import math
    
    if epoch < warmup_epochs:
        # Linear warmup
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

# Usage
for epoch in range(total_epochs):
    lr = get_lr_with_warmup(epoch, warmup_epochs=10, total_epochs=100,
                            base_lr=1e-3, min_lr=1e-6)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    train_epoch(...)
```

## Warmup Duration Guidelines

| Model Type | Warmup Epochs/Steps | Notes |
|------------|---------------------|-------|
| Small CNN | 0-5 epochs | Often unnecessary |
| Large CNN | 5-10 epochs | Helpful for stability |
| Transformer (small) | 1-2k steps | ~5-10% of training |
| Transformer (large) | 4-10k steps | Critical for convergence |
| BERT fine-tuning | 6-10% of steps | Standard practice |
| GPT pretraining | 1-2k steps | Relatively short |

Rule of thumb: **5-10% of total training** for warmup.

## Warmup with Different Optimizers

### Adam/AdamW

Warmup is especially important because:
- Moment estimates ($m$, $v$) start at zero
- Bias correction helps but isn't perfect
- High initial LR can cause instability

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
```

### SGD with Momentum

Less critical but still beneficial for large models:

```python
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=5)
```

## Large Batch Training

Warmup becomes critical when using large batch sizes:

$$\text{Adjusted LR} = \text{Base LR} \times \frac{\text{Batch Size}}{\text{Reference Batch Size}}$$

With linear scaling, larger batches need proportionally longer warmup:

```python
base_batch_size = 256
actual_batch_size = 4096
scale_factor = actual_batch_size / base_batch_size

# Scale both LR and warmup
base_lr = 0.1
scaled_lr = base_lr * scale_factor

warmup_epochs = 5 * scale_factor  # Also scale warmup
```

## Monitoring Warmup

Track learning rate and loss during warmup to ensure stability:

```python
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    scheduler.step()
    
    lr = optimizer.param_groups[0]['lr']
    
    # Log for monitoring
    print(f"Epoch {epoch}: LR = {lr:.6f}, Loss = {train_loss:.4f}")
    
    # Warning if loss is unstable during warmup
    if epoch < warmup_epochs and train_loss > 10 * initial_loss:
        print("Warning: Loss unstable during warmup, consider longer warmup")
```

## Common Pitfalls

| Mistake | Symptom | Fix |
|---------|---------|-----|
| No warmup with large LR | Training diverges immediately | Add warmup or reduce initial LR |
| Warmup too short | Loss spikes then recovers | Extend warmup duration |
| Warmup too long | Wasted training time | Reduce to 5-10% of training |
| Wrong scheduler order | LR doesn't follow expected curve | Verify SequentialLR milestones |

## Key Takeaways

Warmup gradually increases learning rate from a small initial value, stabilizing early training. It's essential for transformers and large models, helpful for most modern architectures. Linear warmup is the most common strategy, typically lasting 5-10% of total training. Combine warmup with decay schedules (cosine annealing) using `SequentialLR`. For large batch training, scale both learning rate and warmup duration proportionally.
