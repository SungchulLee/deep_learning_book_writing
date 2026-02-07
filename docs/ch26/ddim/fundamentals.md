# Denoising Diffusion Implicit Models (DDIM)

**DDIM** (Song et al., 2020) provides a faster, often deterministic sampling method for diffusion models without retraining.

## Motivation

DDPM sampling requires iterating through all $T$ timesteps (typically 1000), making generation slow. DDIM addresses this by:

1. Defining a **non-Markovian** forward process with the same marginals
2. Enabling **step skipping** during sampling
3. Allowing **deterministic** generation when desired

## Key Insight

The DDPM training objective only depends on the marginal distributions $q(x_t|x_0)$, not the full joint $q(x_{1:T}|x_0)$. This means many different forward processes share the same training loss.

## DDIM Forward Process

DDIM defines a family of non-Markovian processes indexed by $\sigma$:

$$
q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\cdot\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}, \sigma_t^2 I\right)
$$

### Special Cases

| $\sigma_t$ | Behavior |
|------------|----------|
| $\sigma_t = \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t}$ | Equivalent to DDPM |
| $\sigma_t = 0$ | Deterministic (DDIM) |

## DDIM Sampling Update

With the trained noise predictor $\epsilon_\theta(x_t, t)$:

### Step 1: Predict $x_0$

$$
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

### Step 2: Update to $x_{t-1}$

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\cdot\epsilon_\theta(x_t, t) + \sigma_t z
$$

where $z \sim \mathcal{N}(0, I)$.

### Deterministic Case ($\sigma_t = 0$)

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\cdot\epsilon_\theta(x_t, t)
$$

No randomness! Each initial noise $x_T$ maps to a unique image $x_0$.

## Accelerated Sampling

### Subsequence Sampling

Instead of all timesteps $\{1, 2, \ldots, T\}$, use a subsequence $\tau = \{t_1, t_2, \ldots, t_S\}$ where $S \ll T$.

Example: For $T=1000$, use $\tau = \{1, 21, 41, \ldots, 981\}$ (50 steps).

### Update Rule for Subsequences

For consecutive elements $t_i$ and $t_{i+1}$ in $\tau$:

$$
x_{t_i} = \sqrt{\bar{\alpha}_{t_i}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t_i}}\cdot\epsilon_\theta(x_{t_{i+1}}, t_{i+1})
$$

## Algorithm

```
Algorithm: DDIM Sampling
────────────────────────
Input: Trained ε_θ, subsequence τ = [t_S, ..., t_1], η (stochasticity)

x_T ~ N(0, I)

for i = S, S-1, ..., 1:
    t = τ[i]
    t_prev = τ[i-1] if i > 1 else 0
    
    # Predict x_0
    x̂_0 = (x_t - sqrt(1-ᾱ_t) * ε_θ(x_t, t)) / sqrt(ᾱ_t)
    
    # Compute variance
    σ_t = η * sqrt((1-ᾱ_{t_prev})/(1-ᾱ_t)) * sqrt(1-ᾱ_t/ᾱ_{t_prev})
    
    # Direction pointing to x_t
    dir_xt = sqrt(1 - ᾱ_{t_prev} - σ_t²) * ε_θ(x_t, t)
    
    # Sample x_{t_prev}
    noise = N(0, I) if t > 1 else 0
    x_{t_prev} = sqrt(ᾱ_{t_prev}) * x̂_0 + dir_xt + σ_t * noise

return x_0
```

## Implementation

```python
import torch

class DDIMSampler:
    def __init__(self, model, alphas_bar, T=1000):
        self.model = model
        self.alphas_bar = alphas_bar
        self.T = T
    
    def get_timestep_sequence(self, num_steps):
        """Create evenly spaced timestep subsequence."""
        step_size = self.T // num_steps
        return list(range(0, self.T, step_size))[::-1]  # Descending
    
    @torch.no_grad()
    def sample(self, shape, device, num_steps=50, eta=0.0):
        """
        DDIM sampling.
        
        Args:
            shape: Output shape (batch, channels, height, width)
            device: torch device
            num_steps: Number of sampling steps
            eta: Stochasticity (0 = deterministic, 1 = DDPM-like)
        
        Returns:
            Generated samples
        """
        # Get timestep subsequence
        timesteps = self.get_timestep_sequence(num_steps)
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Get alpha_bar values
            alpha_bar_t = self.alphas_bar[t]
            alpha_bar_prev = self.alphas_bar[timesteps[i+1]] if i < len(timesteps)-1 else torch.tensor(1.0)
            
            # Predict noise
            eps = self.model(x, t_batch)
            
            # Predict x_0
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            
            # Optionally clip x0 prediction
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            # Compute sigma
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
            
            # Sample x_{t-1}
            if i < len(timesteps) - 1:
                noise = torch.randn_like(x) if eta > 0 else 0
                x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise
            else:
                x = x0_pred
        
        return x
```

## Comparison: DDPM vs DDIM

| Aspect | DDPM | DDIM |
|--------|------|------|
| Sampling type | Stochastic | Deterministic (η=0) or stochastic |
| Steps required | T (e.g., 1000) | Any S ≪ T (e.g., 50) |
| Latent space | No structure | Meaningful interpolation |
| Sample quality | Baseline | Similar with fewer steps |
| Speed | Slow | 10-50× faster |

## Benefits of Deterministic Sampling

### Latent Space Structure

With $\eta = 0$, there's a one-to-one mapping between $x_T$ and $x_0$:

$$
x_0 = f_\theta(x_T)
$$

This enables:

1. **Interpolation**: Blend latents to blend images
2. **Inversion**: Find $x_T$ that generates a given image
3. **Editing**: Modify $x_T$ for controlled changes

### Consistency

The same $x_T$ always produces the same $x_0$, useful for:
- Reproducibility
- Debugging
- Ablation studies

## Connection to Neural ODEs

Deterministic DDIM can be viewed as solving an ODE:

$$
\frac{dx}{dt} = f_\theta(x, t)
$$

This probability flow ODE has the same marginals as the diffusion SDE, allowing:
- Use of ODE solvers (Euler, Heun, RK45)
- Adaptive step sizes
- Further acceleration via higher-order methods

## Practical Tips

### Choosing Number of Steps

| Steps | Quality | Speed |
|-------|---------|-------|
| 10-20 | Acceptable | Very fast |
| 50 | Good | Fast |
| 100 | Very good | Moderate |
| 250+ | Excellent | Slow |

### Choosing η

- $\eta = 0$: Deterministic, good for consistency
- $\eta = 1$: DDPM-like stochasticity
- $\eta \in (0, 1)$: Balance between diversity and consistency

### Timestep Spacing

- **Uniform**: Simple, works well
- **Quadratic**: More steps at high noise
- **Learned**: Optimized per model

## Summary

DDIM enables fast sampling from diffusion models by exploiting a non-Markovian formulation that shares the same training objective as DDPM. With $\eta=0$, sampling becomes deterministic, enabling latent space manipulation. The key practical benefit is reducing sampling from 1000 steps to 50 or fewer while maintaining quality.
