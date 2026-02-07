# Multi-Step LR

## Overview

MultiStepLR decays the learning rate at specified epoch milestones, providing more control than uniform StepLR.

## Schedule

$$\eta_t = \eta_0 \cdot \gamma^{|\{m \in \text{milestones} : m \leq t\}|}$$

## PyTorch Implementation

```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[60, 120, 160], gamma=0.2
)
# LR: 0.1 → 0.02 at epoch 60 → 0.004 at epoch 120 → 0.0008 at epoch 160
```

## Usage

MultiStepLR is the standard scheduler for ResNet training on ImageNet and CIFAR. Milestones are typically set at 60%, 80%, and 90% of total epochs.

## Key Takeaways

- Decays LR at manually specified milestones.
- Standard for CNN training with SGD.
