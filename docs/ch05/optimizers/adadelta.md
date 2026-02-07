# Adadelta

## Overview

Adadelta addresses Adagrad's aggressive learning rate decay by replacing the unbounded accumulation of squared gradients with an exponentially decaying average. Uniquely, Adadelta also eliminates the need for an initial learning rate hyperparameter.

## Update Rule

$$s_t = \rho \, s_{t-1} + (1 - \rho) \, g_t^2$$
$$\Delta\theta_t = -\frac{\sqrt{\delta_{t-1} + \epsilon}}{\sqrt{s_t + \epsilon}} \, g_t$$
$$\delta_t = \rho \, \delta_{t-1} + (1 - \rho) \, (\Delta\theta_t)^2$$
$$\theta_{t+1} = \theta_t + \Delta\theta_t$$

where $\rho$ is the decay rate (typically 0.9) and $\delta_t$ accumulates squared parameter updates.

## PyTorch Implementation

```python
optimizer = torch.optim.Adadelta(model.parameters(), rho=0.9, eps=1e-6)
```

## Key Insight

The numerator $\sqrt{\delta_{t-1} + \epsilon}$ uses the RMS of past parameter updates to scale the current update. This provides a form of unit correction—the update has the same "units" as the parameters, unlike SGD or Adagrad where the update has units of gradients divided by the square root of accumulated squared gradients.

## Properties

- No initial learning rate required (though PyTorch defaults to `lr=1.0`).
- Exponential decay in $s_t$ prevents Adagrad's runaway accumulation.
- Rarely used in modern practice; Adam and AdamW have superseded it.

## Key Takeaways

- Adadelta fixes Adagrad's decay problem with exponential moving averages.
- The unit correction via accumulated parameter updates is a unique design choice.
- In practice, Adam is preferred for most applications.
