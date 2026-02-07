# LAMB

## Overview

LAMB (Layer-wise Adaptive Moments optimizer for Batch training) extends Adam with layer-wise learning rate adaptation, enabling stable training with very large batch sizes. It was developed for training BERT in 76 minutes using batch sizes up to 65,536.

## Update Rule

LAMB computes the Adam update and then normalizes it per-layer using the ratio of parameter norm to update norm:

$$r_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot \phi(\|\theta_t\|) \cdot \frac{r_t}{\|r_t\|}$$

where $\phi(\|\theta_t\|)$ is a trust ratio function (typically $\|\theta_t\|$) that scales the update based on the parameter magnitude.

## Intuition

In large-batch training, different layers can have vastly different gradient scales. LAMB normalizes updates per layer, ensuring no single layer dominates the parameter update. This trust ratio acts as an automatic per-layer learning rate.

## PyTorch Implementation

LAMB is not included in core PyTorch but is available in third-party libraries:

```python
# Using torch_optimizer
from torch_optimizer import Lamb

optimizer = Lamb(model.parameters(), lr=0.001, weight_decay=0.01)
```

## When to Use

LAMB is specifically designed for large-batch distributed training where standard Adam becomes unstable. For typical single-GPU training with batch sizes under 1024, AdamW is preferred.

## Key Takeaways

- LAMB enables stable training with very large batch sizes via layer-wise trust ratios.
- Essential for distributed training at scale (e.g., BERT, large transformers).
- For standard training, AdamW is simpler and sufficient.
