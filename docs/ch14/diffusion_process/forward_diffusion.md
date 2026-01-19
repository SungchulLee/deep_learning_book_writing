# Forward Diffusion Process

The **forward diffusion process** is the foundation of diffusion models. It defines how data is gradually corrupted into noise through a sequence of steps.

## Overview

The forward process transforms structured data $x_0$ (e.g., images) into unstructured noise $x_T$ over $T$ timesteps. This process is:

1. **Fixed** (no learnable parameters)
2. **Markovian** (each step depends only on the previous)
3. **Gaussian** (noise is added via Gaussian perturbations)

## Mathematical Formulation

### The Markov Chain

Starting from data $x_0 \sim q(x_0)$, define the forward process as:

$$
q(x_{1:T} | x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})
$$

Each transition adds Gaussian noise:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1-\alpha_t) I)
$$

### Noise Schedule Parameters

Define the **variance schedule** $\{\beta_t\}_{t=1}^T$ where $\beta_t \in (0, 1)$:

$$
\alpha_t = 1 - \beta_t
$$

$$
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$$

Key properties:
- $\alpha_t$ close to 1: small noise at step $t$
- $\bar{\alpha}_t$ decreases monotonically toward 0
- As $t \to T$: $\bar{\alpha}_T \approx 0$, so $x_T \approx \mathcal{N}(0, I)$

### Sampling Formula

To sample $x_t$ given $x_{t-1}$:

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

## Direct Sampling from Any Timestep

A crucial property: we can sample $x_t$ directly from $x_0$ without iterating through intermediate steps.

### Closed-Form Marginal

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

### Direct Sampling Formula

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

This is derived by recursively composing the per-step transitions:

$$
\begin{aligned}
x_1 &= \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1 \\
x_2 &= \sqrt{\alpha_2} x_1 + \sqrt{1-\alpha_2} \epsilon_2 \\
    &= \sqrt{\alpha_1 \alpha_2} x_0 + \sqrt{1 - \alpha_1 \alpha_2} \bar{\epsilon}_2 \\
    &\vdots \\
x_t &= \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
\end{aligned}
$$

## Noise Schedules

The choice of $\{\beta_t\}$ significantly impacts model performance.

### Linear Schedule (Original DDPM)

$$
\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)
$$

Typical values: $\beta_1 = 10^{-4}$, $\beta_T = 0.02$, $T = 1000$

### Cosine Schedule (Improved DDPM)

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2
$$

where $s$ is a small offset (typically 0.008) to prevent $\beta_t$ from being too small near $t=0$.

### Comparison

| Schedule | Advantages | Disadvantages |
|----------|------------|---------------|
| Linear | Simple, well-studied | Fast noise increase; wastes capacity near $t=T$ |
| Cosine | Smoother transition; better sample quality | Slightly more complex |

## Signal-to-Noise Ratio View

The forward process can be understood through the **signal-to-noise ratio (SNR)**:

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
$$

- At $t=0$: SNR $= \infty$ (pure signal)
- At $t=T$: SNR $\approx 0$ (pure noise)

The log-SNR decreases approximately linearly for the cosine schedule.

## Implementation

```python
import torch
import numpy as np

class ForwardDiffusion:
    def __init__(self, T=1000, schedule='linear'):
        self.T = T
        
        if schedule == 'linear':
            self.betas = self._linear_schedule()
        elif schedule == 'cosine':
            self.betas = self._cosine_schedule()
        
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def _linear_schedule(self, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, self.T)
    
    def _cosine_schedule(self, s=0.008):
        steps = self.T + 1
        t = torch.linspace(0, self.T, steps)
        f_t = torch.cos((t / self.T + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bars = f_t / f_t[0]
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        return torch.clamp(betas, 0, 0.999)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Sample x_t from q(x_t | x_0).
        
        Args:
            x_0: Clean data [batch_size, ...]
            t: Timesteps [batch_size]
            noise: Optional pre-sampled noise
        
        Returns:
            x_t: Noisy samples at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get alpha_bar values for each sample
        alpha_bar_t = self.alpha_bars[t]
        
        # Reshape for broadcasting
        while len(alpha_bar_t.shape) < len(x_0.shape):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        
        # Direct sampling formula
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        return x_t
```

## Visualization of the Forward Process

At different timesteps, the data transforms:

| Timestep | $\bar{\alpha}_t$ | Visual Effect |
|----------|------------------|---------------|
| $t = 0$ | 1.0 | Original image |
| $t = T/4$ | ~0.7 | Slightly blurry, some noise |
| $t = T/2$ | ~0.3 | Significant noise, structure fading |
| $t = 3T/4$ | ~0.05 | Mostly noise, hints of structure |
| $t = T$ | ~0 | Pure Gaussian noise |

## Key Insights

### No Learning Required

The forward process has no trainable parameters. It's a deterministic (given the noise) corruption process.

### Training Efficiency

Direct sampling via $q(x_t | x_0)$ means training doesn't require sequential simulation. We can:
1. Sample a random $t \sim \text{Uniform}(1, T)$
2. Directly compute $x_t$ from $x_0$
3. Train the network on that pair

### Information Destruction

The forward process progressively destroys information about $x_0$. The reverse process must learn to recover this information, which is the core challenge of diffusion models.

## Summary

The forward diffusion process defines a Markov chain that gradually transforms data into noise via Gaussian perturbations. The closed-form marginal $q(x_t|x_0)$ enables efficient training by allowing direct sampling at any timestep. The noise schedule $\{\beta_t\}$ controls the rate of corruption and significantly impacts generation quality.
