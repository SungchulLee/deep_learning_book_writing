# Reverse Process

The **reverse process** is the generative core of diffusion models. It transforms noise into structured data by learning to reverse the forward diffusion, one step at a time.

## Discrete-Time Formulation

The reverse process is a learned Markov chain running backward from $t=T$ to $t=0$:

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)$$

where $p(x_T) = \mathcal{N}(0, I)$ is the noise prior and each reverse transition is modelled as Gaussian:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}\bigl(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I\bigr)$$

The model learns the mean $\mu_\theta$; the variance $\sigma_t^2$ is typically fixed to either $\beta_t$ or $\tilde{\beta}_t$ (defined below).

## The True Posterior $q(x_{t-1} | x_t, x_0)$

When the clean data $x_0$ is known, the true reverse transition has a closed form. This is the target that the learned reverse process approximates.

$$\boxed{q(x_{t-1} | x_t, x_0) = \mathcal{N}\bigl(x_{t-1};\, \tilde{\mu}_t(x_t, x_0),\, \tilde{\beta}_t I\bigr)}$$

### Posterior Variance

$$\tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)} \beta_t$$

### Posterior Mean

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1 - \bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t$$

### Derivation: Product of Two Gaussians

The posterior arises from Bayes' rule applied to the Markov chain:

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)} \propto q(x_t | x_{t-1}) \cdot q(x_{t-1} | x_0)$$

Both factors are Gaussian in $x_{t-1}$:

**Likelihood**: $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}\, x_{t-1},\, (1-\alpha_t)I)$ contributes precision $A_1 = \alpha_t / (1-\alpha_t)$.

**Prior**: $q(x_{t-1} | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}\, x_0,\, (1-\bar{\alpha}_{t-1})I)$ contributes precision $A_2 = 1 / (1-\bar{\alpha}_{t-1})$.

The product of two Gaussians is another Gaussian with combined precision $A = A_1 + A_2$ and mean $B/A$ where $B = B_1 + B_2$ aggregates the precision-weighted means. Using $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$ and $\beta_t = 1 - \alpha_t$, the algebra simplifies to the formulas above.

The posterior mean is a **weighted average**: it interpolates between what $x_0$ predicts about $x_{t-1}$ (via the prior) and what $x_t$ predicts about $x_{t-1}$ (via the likelihood), with weights determined by relative precisions.

## Noise Prediction and the Reverse Mean

In practice we do not know $x_0$, so we estimate it from the network's noise prediction. Since $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$, the clean data estimate is:

$$\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\bigl(x_t - \sqrt{1-\bar{\alpha}_t}\, \epsilon_\theta(x_t, t)\bigr)$$

Substituting into the posterior mean formula and simplifying:

$$\boxed{\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\, \epsilon_\theta(x_t, t)\right)}$$

This formula says: start with $x_t$, subtract the predicted noise (appropriately scaled), and rescale by $1/\sqrt{\alpha_t}$ to account for signal attenuation. The derivation proceeds by plugging $\hat{x}_0$ into $\tilde{\mu}_t$ and collecting the $x_t$ coefficient (which simplifies to $1/\sqrt{\alpha_t}$) and the $\epsilon_\theta$ coefficient (which simplifies to $-(1-\alpha_t)/(\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t})$).

### Alternative Parameterisations

| Prediction target | Reverse mean formula |
|-------------------|---------------------|
| Noise $\epsilon_\theta$ | $\frac{1}{\sqrt{\alpha_t}}\bigl(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\bigr)$ |
| Clean data $\hat{x}_0$ | $\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} \hat{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$ |
| Velocity $v_\theta$ | Interpolation between $\epsilon$ and $x_0$ parameterisations |

## Variance Choices

The reverse variance $\sigma_t^2$ can be set to:

| Choice | Formula | Properties |
|--------|---------|------------|
| Posterior variance | $\tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)}\beta_t$ | Optimal when $x_0$ is known |
| Forward variance | $\beta_t = 1 - \alpha_t$ | Upper bound; noisier |
| Learned (interpolated) | $\exp(v \log\beta_t + (1-v)\log\tilde{\beta}_t)$ | Improved DDPM |

The two fixed choices bracket the optimal variance. Learning $v$ per timestep (Nichol & Dhariwal, 2021) provides modest improvements, especially for small $T$.

## Continuous-Time: The Reverse SDE

In continuous time, the forward process is described by a stochastic differential equation:

$$dx = f(x, t)\, dt + g(t)\, dW$$

Anderson's time-reversal theorem (1982) gives the reverse process:

$$\boxed{dx = \bigl[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\bigr] dt + g(t)\, d\bar{W}}$$

where $d\bar{W}$ is a reverse-time Wiener process and $\nabla_x \log p_t(x)$ is the **score function** at time $t$.

For the variance-preserving formulation with $f(x,t) = -\frac{1}{2}\beta(t)x$ and $g(t) = \sqrt{\beta(t)}$:

$$dx = \left[-\frac{1}{2}\beta(t)\, x - \beta(t)\, \nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)}\, d\bar{W}$$

The term $-\beta(t) \nabla_x \log p_t(x)$ is a **score-guided drift** that steers the process from noise back toward data.

## Connection to Langevin Dynamics

The reverse diffusion SDE is **time-inhomogeneous Langevin dynamics** with a learned score:

$$dx_t = \underbrace{-\frac{1}{2}\beta(t)\, x_t}_{\text{drift toward origin}} + \underbrace{\beta(t)\, s_\theta(x_t, t)}_{\text{score-guided drift}}\, dt + \sqrt{\beta(t)}\, d\bar{W}_t$$

This unifies the MCMC and generative-model perspectives:

| MCMC (Langevin) | Diffusion model |
|-----------------|-----------------|
| Known target $\pi(x)$ | Unknown $p_{\text{data}}(x)$ |
| Analytical score $\nabla \log \pi$ | Learned score $s_\theta(x, t)$ |
| Stationary distribution | Time-varying marginals $p_t(x)$ |
| Single temperature | Time-varying noise schedule |

Standard Langevin dynamics is the **time-homogeneous special case**: the score does not depend on $t$ and the target distribution is fixed. The discrete analogue—annealed Langevin dynamics with noise levels $\sigma_1 > \cdots > \sigma_L$—predates DDPM and is equivalent up to discretisation.

## Score–Noise Equivalence

The DDPM noise predictor and the score function are related by:

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

Predicting noise is equivalent to estimating the score. This equivalence unifies the DDPM (noise prediction) and score-based (score estimation) formulations—they describe the same mathematical object from different viewpoints.

## PyTorch Implementation

```python
import torch


def compute_posterior_params(
    x_0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
    betas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute parameters of q(x_{t-1} | x_t, x_0).

    Returns:
        (posterior_mean, posterior_variance)
    """
    alpha_bar_t = alpha_bars[t]
    alpha_bar_prev = torch.where(t > 0, alpha_bars[t - 1], torch.ones_like(alpha_bar_t))
    beta_t = betas[t]
    alpha_t = 1.0 - beta_t

    # Posterior variance: β̃_t = β_t (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
    posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)

    # Posterior mean coefficients
    coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
    coef_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)

    # Reshape for broadcasting
    for _ in range(len(x_0.shape) - 1):
        coef_x0 = coef_x0.unsqueeze(-1)
        coef_xt = coef_xt.unsqueeze(-1)

    posterior_mean = coef_x0 * x_0 + coef_xt * x_t
    return posterior_mean, posterior_var


def reverse_mean_from_noise(
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_pred: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """Compute reverse mean from noise prediction.

    μ_θ(x_t, t) = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ)
    """
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]

    for _ in range(len(x_t.shape) - 1):
        alpha_t = alpha_t.unsqueeze(-1)
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)

    return (1.0 / torch.sqrt(alpha_t)) * (
        x_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps_pred
    )
```

## Summary

The reverse process transforms noise into data by learning to approximate the true denoising posterior $q(x_{t-1}|x_t, x_0)$. In the noise-prediction parameterisation, the model estimates $\epsilon$ and plugs it into a closed-form mean formula. In continuous time, the reverse process is an SDE whose drift depends on the score function—the central quantity that connects all diffusion model formulations.
