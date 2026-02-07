# Exponential LR

## Overview

ExponentialLR decays the learning rate by a constant multiplicative factor every epoch.

## Schedule

$$\eta_t = \eta_0 \cdot \gamma^t$$

## PyTorch Implementation

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# Each epoch: LR *= 0.95
```

## Properties

Exponential decay provides a smooth, monotonic decrease. The decay is rapid—after $t$ epochs, the LR is $\gamma^t$ of the initial value. With $\gamma = 0.95$, the LR halves every $\sim 14$ epochs.

Choose $\gamma$ based on total training duration: $\gamma = (\eta_{\text{final}} / \eta_0)^{1/T}$.

## Key Takeaways

- Smooth multiplicative decay every epoch.
- Decay rate is controlled by a single parameter $\gamma$.
- Can decay too aggressively for long training runs.
