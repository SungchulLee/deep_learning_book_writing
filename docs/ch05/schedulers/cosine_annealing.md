# Cosine Annealing

## Overview

Cosine annealing follows a cosine curve from the initial learning rate to a minimum value, providing a smooth decay that spends more time at low learning rates near the end of training.

## Schedule

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

where $T$ is the total number of epochs.

## PyTorch Implementation

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=200, eta_min=1e-6
)
```

## With Warm Restarts

`CosineAnnealingWarmRestarts` periodically resets the learning rate:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
# First cycle: 10 epochs, second: 20, third: 40, ...
```

Warm restarts can help escape local minima by periodically increasing the learning rate.

## Properties

Cosine annealing is now the default scheduling strategy for most modern training recipes. It requires only one hyperparameter ($T_{\max}$), provides smooth decay, and empirically outperforms step-based schedules.

## Key Takeaways

- Cosine decay spends more training time at low learning rates, enabling fine-grained convergence.
- The modern default schedule for most architectures.
- Warm restarts add periodic exploration phases.
