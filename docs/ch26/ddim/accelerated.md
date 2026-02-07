# Accelerated Sampling

Standard DDPM requires $T = 1000$ sequential denoising steps, making generation slow. This section covers methods to reduce the number of steps while maintaining sample quality.

## DDIM Step Skipping

The key insight behind DDIM is that the DDPM training objective depends only on the marginals $q(x_t|x_0)$, not the full joint distribution. This means we can define a different (non-Markovian) reverse process that uses a **subsequence** of timesteps.

Given a subsequence $\tau = [\tau_1, \tau_2, \ldots, \tau_S]$ with $S \ll T$:

$$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}}\, \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}, \tau_i) + \sigma_{\tau_i}\, z$$

No retraining is required—the same model trained with $T = 1000$ can be sampled with 50, 20, or even 10 steps.

### Choosing the Subsequence

| Strategy | Description | Example (S=50, T=1000) |
|----------|-------------|------------------------|
| Uniform | Every $T/S$-th step | $[0, 20, 40, \ldots, 980]$ |
| Quadratic | Denser at low $t$ | $[0, 0.4, 1.6, \ldots, 980]$ |
| Trailing | Denser at high $t$ | Focus on coarse structure |

Uniform spacing is the most common default. Denser spacing at low $t$ (where fine details emerge) can improve quality.

### Quality vs Steps

Typical quality degradation (FID on CIFAR-10):

| Steps | DDPM | DDIM ($\eta=0$) | DDIM ($\eta=1$) |
|-------|------|-----------------|-----------------|
| 1000 | 3.17 | 4.04 | 3.17 |
| 100 | — | 4.16 | 6.84 |
| 50 | — | 4.67 | 13.36 |
| 20 | — | 6.84 | 32.55 |
| 10 | — | 13.36 | 72.61 |

Deterministic DDIM degrades more gracefully than stochastic sampling as steps decrease.

## Higher-Order ODE Solvers

Since deterministic DDIM approximates the probability flow ODE, we can use better ODE solvers for fewer function evaluations.

### DPM-Solver (Lu et al., 2022)

A dedicated high-order solver for diffusion ODEs. DPM-Solver++ achieves strong results in 10–20 steps by exploiting the specific structure of the diffusion ODE drift.

### Heun's Method (2nd Order)

The predictor-corrector approach:

1. **Predict**: Take an Euler step to get $\hat{x}_{t-1}$
2. **Correct**: Average the drift at $x_t$ and $\hat{x}_{t-1}$

This doubles the cost per step but halves the error, often yielding better quality at the same total compute.

## Progressive Distillation

Salimans & Ho (2022) introduced **progressive distillation**: train a student model to match two teacher steps in one, then iterate.

### Method

Starting from a $T$-step teacher:

1. **Round 1**: Train student to match $x_{t-2}^{\text{teacher}}$ in one step. Student now uses $T/2$ steps.
2. **Round 2**: The $T/2$-step student becomes the new teacher. Train a new student with $T/4$ steps.
3. **Continue** until 4–8 steps.

The loss at each round is:

$$\mathcal{L}_{\text{distill}} = \|x_{t-2}^{\text{student}} - x_{t-2}^{\text{teacher}}\|^2$$

### Results

Progressive distillation achieves near-teacher quality in 4–8 steps, roughly a 100–250× speedup over the original 1000-step model.

## Consistency Models

Song et al. (2023) introduced **consistency models** that learn to map any point on an ODE trajectory directly to the clean data endpoint.

### The Consistency Property

For any points $x_t$ and $x_s$ on the same probability flow ODE trajectory:

$$f_\theta(x_t, t) = f_\theta(x_s, s) = x_0 \qquad \forall\, t, s$$

The model always outputs the same $x_0$, regardless of where on the trajectory the input lies.

### Training Approaches

**Consistency distillation.** Given a pre-trained diffusion model, enforce consistency by requiring adjacent ODE steps to produce the same output:

$$\mathcal{L}_{\text{CD}} = \|f_\theta(x_{t+\Delta t}, t+\Delta t) - f_{\theta^-}(x_t, t)\|^2$$

where $\theta^-$ is an EMA of $\theta$ and $x_t$ is obtained from $x_{t+\Delta t}$ via one ODE step.

**Consistency training.** Train from scratch without a pre-trained model by using the forward process to generate training pairs.

### Generation

Once trained, consistency models generate samples in a **single step**: sample $x_T \sim \mathcal{N}(0, I)$ and compute $\hat{x}_0 = f_\theta(x_T, T)$. Multi-step refinement is possible but not required.

## Comparison of Acceleration Methods

| Method | Steps | Retraining? | Quality | Speed |
|--------|-------|-------------|---------|-------|
| DDIM | 10–100 | No | Good | Fast |
| DPM-Solver++ | 10–20 | No | Very good | Fast |
| Progressive distillation | 4–8 | Yes (iterative) | Very good | Very fast |
| Consistency models | 1–4 | Yes | Good–very good | Fastest |
| Original DDPM | 1000 | — | Baseline | Slow |

## PyTorch Implementation: Step Scheduling

```python
import torch
import numpy as np


def make_ddim_timesteps(
    T: int, num_steps: int, spacing: str = "uniform"
) -> list[int]:
    """Create DDIM timestep subsequence.

    Args:
        T: Total training timesteps.
        num_steps: Number of sampling steps.
        spacing: 'uniform' or 'quadratic'.

    Returns:
        List of timesteps (descending).
    """
    if spacing == "uniform":
        step_size = T // num_steps
        timesteps = list(range(0, T, step_size))[:num_steps]
    elif spacing == "quadratic":
        timesteps = (
            np.linspace(0, np.sqrt(T * 0.8), num_steps) ** 2
        ).astype(int).tolist()
    else:
        raise ValueError(f"Unknown spacing: {spacing}")

    return sorted(set(timesteps), reverse=True)
```

## Summary

DDIM step skipping provides the simplest acceleration (no retraining), higher-order solvers like DPM-Solver++ push quality further, and distillation or consistency models achieve few-step generation with additional training. The choice depends on the application: DDIM for quick experiments, DPM-Solver++ for production inference without retraining, and consistency models when single-step generation is required.

## References

1. Song, J., Meng, C., & Ermon, S. (2020). "Denoising Diffusion Implicit Models." *ICLR*.
2. Lu, C., et al. (2022). "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling." *NeurIPS*.
3. Salimans, T., & Ho, J. (2022). "Progressive Distillation for Fast Sampling of Diffusion Models." *ICLR*.
4. Song, Y., et al. (2023). "Consistency Models." *ICML*.
