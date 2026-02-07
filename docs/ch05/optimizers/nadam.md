# NAdam

## Overview

NAdam (Nesterov-accelerated Adaptive Moment Estimation) incorporates Nesterov momentum into the Adam optimizer. Just as Nesterov momentum improves SGD by evaluating the gradient at a look-ahead position, NAdam applies a similar correction to Adam's momentum term.

## Update Rule

NAdam modifies Adam's update by using the look-ahead first moment:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \left( \frac{\beta_1 \hat{m}_t}{1 - \beta_1^{t+1}} + \frac{(1 - \beta_1) g_t}{1 - \beta_1^t} \right)$$

This is equivalent to using the Nesterov-corrected momentum within the adaptive framework.

## PyTorch Implementation

```python
optimizer = torch.optim.NAdam(model.parameters(), lr=0.001,
                              betas=(0.9, 0.999), eps=1e-8)
```

## When to Use

NAdam is a good choice when both Nesterov momentum and adaptive learning rates are desired. It tends to converge slightly faster than Adam, particularly in the early stages of training.

## Key Takeaways

- NAdam combines Nesterov momentum with Adam's adaptive learning rates.
- Provides a modest convergence speed improvement over Adam.
- Same hyperparameters as Adam; a drop-in replacement.
