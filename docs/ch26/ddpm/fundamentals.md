# DDPM Fundamentals

**Denoising Diffusion Probabilistic Models** (Ho et al., 2020) demonstrated that diffusion models can achieve image generation quality competitive with GANs. DDPM combines a fixed Gaussian forward process, a learned Gaussian reverse process, and a simplified noise-prediction training objective into a clean, practical framework.

## The DDPM Framework

DDPM instantiates the general diffusion framework (§25.1) with specific design choices:

**Forward process.** A discrete Markov chain with $T = 1000$ steps and a linear noise schedule $\beta_t \in [10^{-4}, 0.02]$:

$$q(x_t | x_{t-1}) = \mathcal{N}\bigl(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I\bigr)$$

**Reverse process.** Gaussian transitions with learned mean and fixed variance:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}\bigl(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I\bigr)$$

**Training objective.** The simplified noise-prediction loss:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\bigl[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\bigr]$$

## Connection to Score-Based Models

DDPM and Noise Conditional Score Networks (NCSN, Song & Ermon 2019) are two views of the same underlying framework. NCSN learns the score $s_\theta(x, \sigma) \approx \nabla_x \log q_\sigma(x)$ at multiple noise levels and samples via annealed Langevin dynamics. The correspondence is:

| DDPM | NCSN |
|------|------|
| Discrete timesteps $t \in \{1,\ldots,T\}$ | Discrete noise levels $\sigma_1 > \cdots > \sigma_L$ |
| Noise predictor $\epsilon_\theta(x_t, t)$ | Score network $s_\theta(x, \sigma)$ |
| $\epsilon_\theta = -\sqrt{1-\bar{\alpha}_t}\, s_\theta$ | $s_\theta = -\epsilon_\theta / \sqrt{1-\bar{\alpha}_t}$ |
| Ancestral sampling | Annealed Langevin dynamics |
| Markov forward process | Independent noise perturbations |

Both optimise the same denoising score matching objective. The $\sigma^2$-weighted NCSN loss is equivalent to the DDPM noise-prediction loss. The key innovation of DDPM was showing that the noise-prediction parameterisation with uniform timestep sampling produces excellent results with a simple training loop.

## Comparison with Other Generative Models

### vs GANs

DDPM achieves comparable or superior image quality without adversarial training. The regression-style loss eliminates mode collapse and training instability. The trade-off is sampling speed: DDPM requires 1000 sequential denoising steps versus a single forward pass for GANs.

### vs VAEs

DDPM can be viewed as a hierarchical VAE with $T$ latent layers, but with the crucial simplification that the encoder (forward process) is fixed rather than learned. This removes the amortisation gap and posterior collapse issues of standard VAEs, typically producing sharper samples.

### vs Normalizing Flows

DDPM's denoising network has no invertibility or Jacobian constraints, allowing more expressive architectures. Flows provide exact likelihoods and fast sampling; DDPM provides better sample quality with slower generation.

## The ELBO Perspective

The DDPM training objective arises from the variational lower bound. The ELBO decomposes as:

$$\mathcal{L}_{\text{VLB}} = \underbrace{D_{\text{KL}}(q(x_T|x_0) \| p(x_T))}_{\text{constant}} + \sum_{t=2}^T \underbrace{D_{\text{KL}}\bigl(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t)\bigr)}_{\text{denoising loss at step } t} - \underbrace{\log p_\theta(x_0|x_1)}_{\text{reconstruction}}$$

Each denoising term compares the true posterior (a Gaussian with known mean and variance) to the learned reverse transition. With fixed variance $\sigma_t^2$, this reduces to a squared distance between means, which in the noise-prediction parameterisation becomes $\|\epsilon - \epsilon_\theta\|^2$ up to a timestep-dependent constant.

The simplified loss drops this constant, making all timesteps equally weighted. This is not a valid ELBO but produces better samples because it up-weights fine-grained denoising steps (low $t$) where perceptual quality is most sensitive.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import numpy as np


class DDPM:
    """Complete DDPM framework: schedule, training, and sampling."""

    def __init__(
        self,
        model: nn.Module,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.model = model
        self.T = T
        self.device = device

        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precompute posterior variance
        alpha_bars_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_bars[:-1]]
        )
        self.posterior_var = (
            self.betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)
        )

    def training_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Compute simplified DDPM loss."""
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        eps = torch.randn_like(x_0)

        # Forward process: create x_t
        a_bar = self.alpha_bars[t]
        while a_bar.dim() < x_0.dim():
            a_bar = a_bar.unsqueeze(-1)

        x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1.0 - a_bar) * eps

        # Predict noise
        eps_pred = self.model(x_t, t)
        return ((eps - eps_pred) ** 2).mean()

    @torch.no_grad()
    def sample(self, shape: tuple) -> torch.Tensor:
        """Generate samples via ancestral sampling."""
        x = torch.randn(shape, device=self.device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps_pred = self.model(x, t_batch)

            # Compute reverse mean
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            mu = (1.0 / torch.sqrt(alpha_t)) * (
                x - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps_pred
            )

            # Add noise (except at t=0)
            if t > 0:
                sigma = torch.sqrt(self.betas[t])
                x = mu + sigma * torch.randn_like(x)
            else:
                x = mu

        return x
```

## Key Design Decisions

**$T = 1000$ steps.** Large $T$ ensures each step is a small perturbation, making the Gaussian reverse approximation accurate. Fewer steps can work with better schedules or DDIM sampling (§25.4).

**Linear schedule.** Simple but not optimal. The cosine schedule (Improved DDPM) provides better results by distributing information loss more evenly.

**Fixed variance.** The original DDPM fixes $\sigma_t^2 = \beta_t$. Learning the variance (Improved DDPM) provides modest gains, particularly for smaller $T$.

**EMA weights.** Using exponential moving average weights for sampling is critical—without EMA, sample quality degrades significantly.

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
2. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
3. Nichol, A., & Dhariwal, P. (2021). "Improved Denoising Diffusion Probabilistic Models." *ICML*.
4. Sohl-Dickstein, J., et al. (2015). "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." *ICML*.
