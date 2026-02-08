#!/usr/bin/env python3
"""
Learning Rate Schedulers - Common patterns
Includes:
  - Warmup + cosine decay
  - Step decay (PyTorch built-in note)
  - ReduceLROnPlateau (PyTorch built-in note)

File: appendix/utils/schedulers.py
Note: Educational scheduler that returns lr multiplier given step.
"""

import math


def warmup_cosine_lr(step, warmup_steps, total_steps, base_lr):
    """
    Warmup + cosine decay schedule.

    - Linearly increase lr from 0 -> base_lr during warmup
    - Then cosine decay from base_lr -> 0

    Returns:
      lr at current step
    """
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))

    # Cosine decay phase
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# Note:
# For production, you typically use PyTorch schedulers:
#   torch.optim.lr_scheduler.StepLR
#   torch.optim.lr_scheduler.CosineAnnealingLR
#   torch.optim.lr_scheduler.ReduceLROnPlateau
