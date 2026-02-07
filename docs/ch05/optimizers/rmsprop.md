# RMSprop

## Overview

RMSprop (Root Mean Square Propagation) was proposed by Geoffrey Hinton to fix Adagrad's decaying learning rate problem. It uses an exponentially decaying average of squared gradients instead of their cumulative sum.

## Update Rule

$$s_t = \alpha \, s_{t-1} + (1 - \alpha) \, g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon} \, g_t$$

where $\alpha$ (typically 0.99) is the decay rate for the squared gradient moving average.

## PyTorch Implementation

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001,
                                alpha=0.99, eps=1e-8)

# With momentum
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001,
                                alpha=0.99, momentum=0.9)
```

## Comparison with Adagrad

The key difference is the window of gradient history:

- **Adagrad**: $s_t = \sum_{i=1}^t g_i^2$ (entire history, monotonically growing)
- **RMSprop**: $s_t = \alpha s_{t-1} + (1-\alpha) g_t^2$ (exponentially weighted, bounded)

RMSprop's bounded denominator prevents the learning rate from decaying to zero, enabling continued learning over long training runs.

## With Centered Gradients

Centered RMSprop subtracts the mean gradient before squaring, using the variance rather than the second moment:

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001,
                                alpha=0.99, centered=True)
```

$$\bar{g}_t = \alpha \, \bar{g}_{t-1} + (1 - \alpha) \, g_t$$
$$s_t = \alpha \, s_{t-1} + (1 - \alpha) \, g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t - \bar{g}_t^2} + \epsilon} \, g_t$$

## When to Use

RMSprop was the default adaptive optimizer before Adam. It remains useful for recurrent neural networks and reinforcement learning, where Adam can sometimes be unstable.

## Key Takeaways

- RMSprop uses exponential decay of squared gradients, fixing Adagrad's learning rate collapse.
- Standard default: $\alpha = 0.99$, $\eta = 0.001$.
- Largely superseded by Adam but still relevant for RNNs and RL.
