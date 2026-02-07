# Adam

## Overview

Adam (Adaptive Moment Estimation) combines the benefits of momentum (first moment) and RMSprop (second moment) into a single optimizer with bias correction. It is the most widely used optimizer in deep learning.

## Update Rule

$$m_t = \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t$$
$$v_t = \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where $m_t$ and $v_t$ are the first and second moment estimates, and the hat notation denotes bias-corrected values.

## Bias Correction

Both $m_t$ and $v_t$ are initialized to zero. In early iterations, they are biased toward zero. The correction terms $1/(1 - \beta_1^t)$ and $1/(1 - \beta_2^t)$ compensate for this initialization bias. As $t \to \infty$, the correction approaches 1 and has no effect.

## PyTorch Implementation

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                             betas=(0.9, 0.999), eps=1e-8)
```

## Default Hyperparameters

The original paper's defaults work well across a wide range of problems:

- $\eta = 0.001$ (learning rate)
- $\beta_1 = 0.9$ (first moment decay)
- $\beta_2 = 0.999$ (second moment decay)
- $\epsilon = 10^{-8}$ (numerical stability)

## Strengths

- **Low tuning effort**: Default hyperparameters work well out of the box.
- **Fast convergence**: Combines momentum's acceleration with adaptive learning rates.
- **Robustness**: Handles sparse gradients and noisy objectives well.

## Known Issues

**Weight decay interaction**: L2 regularization in Adam is not equivalent to weight decay (unlike in SGD). The adaptive scaling distorts the regularization effect. This motivated AdamW (next section).

**Generalization gap**: Adam can converge to sharp minima that generalize worse than those found by SGD with momentum, particularly in computer vision tasks.

**Non-convergence**: In certain settings, Adam's adaptive learning rate can fail to converge. AMSGrad addresses this with a monotonic constraint on the second moment.

## Key Takeaways

- Adam combines momentum and adaptive learning rates with bias correction.
- Default hyperparameters ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.001$) work well for most tasks.
- For proper weight decay, use AdamW instead of Adam with L2 regularization.
