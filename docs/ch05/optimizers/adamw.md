# AdamW

## Overview

AdamW fixes Adam's problematic interaction with weight decay by decoupling the weight decay from the gradient-based update. This seemingly minor change has a significant impact on regularization effectiveness and is now the recommended default optimizer for most deep learning tasks.

## The Problem with Adam + L2

In SGD, L2 regularization and weight decay are equivalent:

$$\theta_{t+1} = \theta_t - \eta(g_t + \lambda\theta_t) = (1 - \eta\lambda)\theta_t - \eta g_t$$

In Adam, L2 regularization is applied to the gradient *before* adaptive scaling:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)(g_t + \lambda\theta_t)$$

The adaptive denominator $\sqrt{\hat{v}_t}$ rescales the regularization term alongside the gradient, effectively applying *different* weight decay to each parameter—defeating the purpose of uniform regularization.

## AdamW Update Rule

AdamW applies weight decay directly to the parameters, outside the adaptive update:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

The weight decay $\lambda$ acts uniformly on all parameters regardless of their adaptive learning rates.

## PyTorch Implementation

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                              betas=(0.9, 0.999), weight_decay=0.01)
```

## Typical Hyperparameters

- $\eta = 0.001$ to $0.0001$ (learning rate)
- $\lambda = 0.01$ to $0.1$ (weight decay)
- $\beta_1 = 0.9$, $\beta_2 = 0.999$

## When to Use

AdamW is the recommended default optimizer for:

- Transformer models (BERT, GPT, Vision Transformers)
- Fine-tuning pretrained models
- Any task where you want to combine adaptive optimization with weight decay

## Key Takeaways

- AdamW decouples weight decay from the adaptive gradient update.
- This provides correct, uniform regularization regardless of per-parameter learning rate scaling.
- AdamW is the modern default optimizer, preferred over `Adam(weight_decay=...)`.
