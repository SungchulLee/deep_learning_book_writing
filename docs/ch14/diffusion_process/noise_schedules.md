# Noise Schedules

The **noise schedule** $\{\beta_t\}_{t=1}^T$ controls how quickly noise is added during the forward diffusion process. This seemingly simple choice has profound effects on model performance and generation quality.

## Why Noise Schedules Matter

The schedule determines:

1. **Information flow**: How quickly data structure is destroyed
2. **Training dynamics**: Which timesteps contribute most to learning
3. **Sample quality**: How well the model can recover fine details
4. **Sampling efficiency**: Trade-offs in accelerated sampling methods

## Key Schedule Parameters

Given $\beta_t$, we derive:

$$
\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$$

The signal-to-noise ratio at time $t$:

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
$$

## Linear Schedule

### Definition

The original DDPM uses a linear schedule:

$$
\beta_t = \beta_{\text{start}} + \frac{t-1}{T-1}(\beta_{\text{end}} - \beta_{\text{start}})
$$

**Typical values**: $\beta_{\text{start}} = 10^{-4}$, $\beta_{\text{end}} = 0.02$, $T = 1000$

### Properties

- $\bar{\alpha}_t$ decays roughly exponentially
- Most corruption happens in early steps
- Later steps (high $t$) add little additional noise

### Limitations

- Inefficient use of model capacity
- Late timesteps are "wasted" as data is already mostly noise
- Can struggle with fine details

## Cosine Schedule

### Definition (Nichol & Dhariwal, 2021)

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2
$$

with offset $s = 0.008$ to prevent $\beta_t$ from being too small.

### Derivation of $\beta_t$

From $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s = \prod_{s=1}^t (1 - \beta_s)$:

$$
\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} = 1 - \frac{f(t)}{f(t-1)}
$$

Clipping: $\beta_t = \min(\beta_t, 0.999)$ for numerical stability.

### Properties

- Smoother decay of $\bar{\alpha}_t$
- More uniform noise distribution across timesteps
- Better preservation of information at early steps
- Improved sample quality in practice

## Quadratic Schedule

### Definition

$$
\beta_t = \beta_{\text{start}} + \frac{(t-1)^2}{(T-1)^2}(\beta_{\text{end}} - \beta_{\text{start}})
$$

### Properties

- Slower initial noise addition
- Faster noise increase toward the end
- Middle ground between linear and cosine

## Sigmoid Schedule

### Definition

$$
\beta_t = \sigma\left(-6 + 12 \cdot \frac{t-1}{T-1}\right) \cdot (\beta_{\text{end}} - \beta_{\text{start}}) + \beta_{\text{start}}
$$

where $\sigma(x) = 1/(1 + e^{-x})$ is the sigmoid function.

### Properties

- S-shaped transition
- Gentle start and end, steeper middle
- Approximates learned schedules in some cases

## Learned Schedules

Instead of fixed schedules, recent work explores learning optimal schedules.

### Variational Diffusion Models (Kingma et al., 2021)

Learn the SNR function $\text{SNR}(t)$ as a monotonic neural network:

$$
\log \text{SNR}(t) = \text{MLP}(t)
$$

with monotonicity enforced via positive weights.

### Schedule Distillation

Train a student model with a compressed schedule (fewer steps) to match a teacher with more steps.

## Continuous-Time Formulation

### SDE Perspective

In continuous time, the forward process is:

$$
dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dW
$$

The discrete schedule $\beta_t$ approximates $\beta(t)$ via:

$$
\beta_t \approx \beta(t/T) \cdot \Delta t
$$

### Common Continuous Schedules

| Name | $\beta(t)$ | Use Case |
|------|------------|----------|
| VP (Variance Preserving) | Linear in $t$ | DDPM-like |
| VE (Variance Exploding) | $\sigma(t)^2$ increasing | NCSN-like |
| Sub-VP | Faster decay | Stable training |

## Schedule Comparison

### Implementation

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def cosine_schedule(T, s=0.008):
    t = torch.arange(T + 1)
    f = torch.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, 0, 0.999)

def quadratic_schedule(T, beta_start=1e-4, beta_end=0.02):
    t = torch.arange(T)
    return beta_start + (t / (T - 1)) ** 2 * (beta_end - beta_start)

def sigmoid_schedule(T, beta_start=1e-4, beta_end=0.02):
    t = torch.linspace(-6, 6, T)
    betas = torch.sigmoid(t) * (beta_end - beta_start) + beta_start
    return betas

# Compute alpha_bar for each schedule
def compute_alpha_bar(betas):
    alphas = 1 - betas
    return torch.cumprod(alphas, dim=0)
```

### Visual Comparison

| Timestep (%) | Linear $\bar{\alpha}$ | Cosine $\bar{\alpha}$ |
|--------------|----------------------|----------------------|
| 10% | 0.95 | 0.98 |
| 25% | 0.80 | 0.90 |
| 50% | 0.50 | 0.70 |
| 75% | 0.15 | 0.35 |
| 100% | 0.00 | 0.00 |

## Practical Recommendations

### For Image Generation

- **Cosine schedule** generally works best
- Use $T = 1000$ for training, can reduce during sampling
- Clamp $\beta_t$ to avoid numerical issues

### For Audio/Video

- Longer schedules ($T = 2000-4000$) may help
- Consider learned schedules for domain-specific optima

### For Fast Sampling

- DDIM/DPM-Solver work better with smooth schedules
- Cosine schedule enables more aggressive step skipping

## Connection to Training Objective

The schedule affects the training loss weighting:

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[ w(t) \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

Different schedules implicitly apply different $w(t)$. The cosine schedule provides more uniform weighting across timesteps.

## Summary

The noise schedule is a crucial hyperparameter that controls the forward diffusion dynamics. While the linear schedule is simple and historically significant, the cosine schedule generally provides better results by distributing information loss more evenly across timesteps. Advanced methods explore learned and continuous-time schedules for further improvements.
