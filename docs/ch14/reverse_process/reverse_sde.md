# Reverse SDE and Denoising Process

The **reverse process** is the generative core of diffusion models. It learns to transform noise back into structured data by reversing the forward diffusion.

## Overview

While the forward process corrupts data: $x_0 \to x_1 \to \cdots \to x_T$

The reverse process generates data: $x_T \to x_{T-1} \to \cdots \to x_0$

## Discrete-Time Reverse Process

### Model Definition

The reverse process is a learned Markov chain:

$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)
$$

where:
- $p(x_T) = \mathcal{N}(0, I)$ is the prior (pure noise)
- $p_\theta(x_{t-1} | x_t)$ is learned

### Gaussian Parameterization

Each reverse transition is modeled as Gaussian:

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

In practice, the variance is often fixed:
$$
\Sigma_\theta(x_t, t) = \sigma_t^2 I
$$

## The True Reverse Posterior

When $x_0$ is known, the true reverse transition has a closed form:

$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

### Posterior Mean

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
$$

### Posterior Variance

$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

## Noise Prediction Parameterization

### From Noise to Mean

Since $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$, we can express $x_0$ as:

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t} \epsilon)
$$

If the network predicts noise $\epsilon_\theta(x_t, t)$, the mean becomes:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)
$$

## Continuous-Time: Reverse SDE

### Forward SDE

The forward process can be written as a stochastic differential equation:

$$
dx = f(x, t) dt + g(t) dW
$$

For variance-preserving (VP) diffusion:
- $f(x, t) = -\frac{1}{2}\beta(t) x$
- $g(t) = \sqrt{\beta(t)}$

### Anderson's Reverse SDE

The time-reversed SDE (Anderson, 1982):

$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t) d\bar{W}
$$

where $d\bar{W}$ is a reverse-time Brownian motion.

### Score-Based Interpretation

The reverse drift depends on the **score function** $\nabla_x \log p_t(x)$.

For diffusion models:
$$
\nabla_x \log p_t(x) \approx -\frac{\epsilon_\theta(x, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

## Probability Flow ODE

### Deterministic Alternative

There exists a deterministic ODE with the same marginal distributions:

$$
dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right] dt
$$

This is the **probability flow ODE**.

### Benefits

1. **Deterministic mapping**: Same noise → same image
2. **Exact likelihood**: Can compute $\log p(x_0)$
3. **Faster solvers**: Use ODE integrators (Euler, Heun, RK45)

## Sampling Algorithms

### DDPM Sampling

```python
def ddpm_sample(model, shape, T, alphas, alpha_bars, betas):
    x = torch.randn(shape)  # x_T ~ N(0, I)
    
    for t in reversed(range(T)):
        # Predict noise
        eps = model(x, t)
        
        # Compute mean
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        
        mu = (1 / sqrt(alpha_t)) * (
            x - (1 - alpha_t) / sqrt(1 - alpha_bar_t) * eps
        )
        
        # Sample (add noise except at t=0)
        if t > 0:
            sigma = sqrt(betas[t])
            x = mu + sigma * torch.randn_like(x)
        else:
            x = mu
    
    return x
```

### DDIM Sampling

```python
def ddim_sample(model, shape, timesteps, alpha_bars, eta=0):
    x = torch.randn(shape)
    
    for i, t in enumerate(timesteps):
        eps = model(x, t)
        
        # Predict x_0
        alpha_bar_t = alpha_bars[t]
        x0_pred = (x - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        
        # Get alpha_bar for previous timestep
        t_prev = timesteps[i+1] if i < len(timesteps)-1 else 0
        alpha_bar_prev = alpha_bars[t_prev]
        
        # Compute sigma (eta=0 for deterministic)
        sigma = eta * sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * \
                sqrt(1 - alpha_bar_t / alpha_bar_prev)
        
        # Direction pointing to x_t
        dir_xt = sqrt(1 - alpha_bar_prev - sigma**2) * eps
        
        # Sample x_{t-1}
        noise = torch.randn_like(x) if sigma > 0 else 0
        x = sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise
    
    return x
```

## Connection to Langevin Dynamics

The reverse SDE can be viewed as **annealed Langevin dynamics**:

$$
x_{t-1} = x_t + \frac{\epsilon}{2} \nabla_x \log p_t(x_t) + \sqrt{\epsilon} z
$$

where the noise level decreases with $t$.

## Variance Choices

Different variance choices for $p_\theta(x_{t-1}|x_t)$:

| Choice | Formula | Properties |
|--------|---------|------------|
| $\tilde{\beta}_t$ | $\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$ | Optimal for data |
| $\beta_t$ | $1 - \alpha_t$ | Upper bound |
| Learned | $\exp(v \log\beta_t + (1-v)\log\tilde{\beta}_t)$ | Interpolated |

## Summary

The reverse process transforms noise into data by learning to predict and remove the noise added during the forward process. It can be formulated as a discrete Markov chain (DDPM), a continuous SDE, or a deterministic ODE. The key insight is that the reverse drift depends on the score function, which the network learns to approximate through noise prediction.
