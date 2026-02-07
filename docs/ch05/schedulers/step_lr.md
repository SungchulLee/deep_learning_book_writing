# Step LR

## Overview

StepLR decays the learning rate by a multiplicative factor at fixed epoch intervals. It is the simplest scheduling policy.

## Schedule

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / S \rfloor}$$

where $S$ is the step size (in epochs) and $\gamma$ is the decay factor.

## PyTorch Implementation

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# LR: 0.1 → 0.01 at epoch 30 → 0.001 at epoch 60 → ...
```

## Usage

StepLR is appropriate when you have prior knowledge about training dynamics (e.g., loss plateaus at roughly known epochs). For most modern applications, cosine annealing or OneCycleLR provide better results without requiring manual milestone selection.

## Key Takeaways

- Decays LR by factor $\gamma$ every $S$ epochs.
- Simple but requires manual selection of step size and decay factor.
