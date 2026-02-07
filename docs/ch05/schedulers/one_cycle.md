# OneCycleLR

## Overview

OneCycleLR implements the 1cycle policy proposed by Leslie Smith, which ramps the learning rate from a minimum to a maximum (warmup phase) and then decays it back to the minimum (annealing phase) over a single training cycle. It also inversely varies momentum.

## Schedule

The learning rate follows a two-phase schedule:
1. **Phase 1** (warmup): LR increases linearly from $\eta_{\min}$ to $\eta_{\max}$ over the first `pct_start` fraction of training.
2. **Phase 2** (annealing): LR decays from $\eta_{\max}$ back to $\eta_{\min}$ (or lower) via cosine or linear annealing.

Momentum varies inversely: high when LR is low, low when LR is high.

## PyTorch Implementation

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=len(train_loader) * num_epochs,
    pct_start=0.3,           # 30% warmup
    anneal_strategy='cos',   # Cosine annealing
    div_factor=25,           # Initial LR = max_lr / 25
    final_div_factor=1e4     # Final LR = max_lr / (25 * 10000)
)

# Step per batch, not per epoch
for epoch in range(num_epochs):
    for x, y in train_loader:
        train_step(...)
        scheduler.step()
```

## Finding max_lr

Use the learning rate range test (LR finder): train for one epoch while exponentially increasing the LR, and plot loss vs. LR. Choose `max_lr` where loss starts decreasing steeply (typically one order of magnitude below the divergence point).

## Properties

OneCycleLR often achieves superior results in fewer epochs compared to standard schedules. The high learning rate phase acts as a regularizer (similar to SGD noise), while the long annealing phase allows precise convergence.

## Key Takeaways

- OneCycleLR combines warmup and annealing in a single, principled schedule.
- Step per batch, not per epoch.
- Use the LR range test to find `max_lr`.
- Often achieves better results in fewer training epochs.
