# Forward Diffusion Process

The **forward diffusion process** defines how data is gradually corrupted into noise. It is the foundation upon which the entire diffusion model framework is built—fixed, parameter-free, and analytically tractable.

## The Markov Chain

Starting from data $x_0 \sim q(x_0)$, the forward process defines a Markov chain:

$$q(x_{1:T} | x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})$$

Each transition adds Gaussian noise:

$$q(x_t | x_{t-1}) = \mathcal{N}\bigl(x_t;\, \sqrt{\alpha_t}\, x_{t-1},\, (1-\alpha_t)\, I\bigr)$$

where $\alpha_t = 1 - \beta_t$ and $\{\beta_t\}_{t=1}^T$ is the **variance schedule** with $\beta_t \in (0, 1)$.

The cumulative product $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ decreases monotonically from $\bar{\alpha}_0 = 1$ (pure signal) toward $\bar{\alpha}_T \approx 0$ (pure noise).

## Direct Sampling: The Closed-Form Marginal

A crucial property enables efficient training: we can sample $x_t$ directly from $x_0$ without iterating through intermediate steps.

$$\boxed{q(x_t | x_0) = \mathcal{N}\bigl(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t)\, I\bigr)}$$

Equivalently:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

**Derivation.** Compose the per-step transitions using the fact that sums of independent Gaussians are Gaussian:

$$\begin{aligned}
x_1 &= \sqrt{\alpha_1}\, x_0 + \sqrt{1-\alpha_1}\, \epsilon_1 \\
x_2 &= \sqrt{\alpha_2}\, x_1 + \sqrt{1-\alpha_2}\, \epsilon_2 = \sqrt{\alpha_1 \alpha_2}\, x_0 + \sqrt{1 - \alpha_1\alpha_2}\, \bar{\epsilon}_2 \\
&\;\vdots \\
x_t &= \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon
\end{aligned}$$

This closed form means training does not require sequential simulation: sample a random $t \sim \text{Uniform}(1, T)$, compute $x_t$ directly from $x_0$, and train the network on that pair.

## Signal-to-Noise Ratio

The forward process is characterised by the **signal-to-noise ratio (SNR)**:

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

At $t=0$ the SNR is infinite (pure signal); at $t=T$ it approaches zero (pure noise). The noise schedule determines how the SNR decreases, which in turn determines how the model allocates capacity across noise levels.

## Noise Schedules

The choice of $\{\beta_t\}$ significantly impacts model performance. It controls information flow (how quickly structure is destroyed), training dynamics (which timesteps dominate the gradient), and sample quality (how well fine details are recovered).

### Linear Schedule

The original DDPM (Ho et al., 2020) uses:

$$\beta_t = \beta_{\text{start}} + \frac{t-1}{T-1}(\beta_{\text{end}} - \beta_{\text{start}})$$

with typical values $\beta_{\text{start}} = 10^{-4}$, $\beta_{\text{end}} = 0.02$, $T = 1000$. The cumulative signal $\bar{\alpha}_t$ decays roughly exponentially, concentrating most corruption in early steps and wasting capacity at high $t$ where data is already mostly noise.

### Cosine Schedule

Nichol & Dhariwal (2021) proposed a smoother schedule defined directly on $\bar{\alpha}_t$:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^{\!2}$$

with offset $s = 0.008$ to prevent $\beta_t$ from being too small near $t=0$. The corresponding $\beta_t$ is recovered as:

$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, \qquad \text{clipped to } [0, 0.999]$$

The cosine schedule distributes information loss more evenly across timesteps, yielding better sample quality in practice. Its log-SNR decreases approximately linearly with $t$.

### Other Schedules

**Quadratic**: $\beta_t = \beta_{\text{start}} + \frac{(t-1)^2}{(T-1)^2}(\beta_{\text{end}} - \beta_{\text{start}})$. Slower initial noise, faster increase toward the end.

**Sigmoid**: $\beta_t = \sigma(-6 + 12(t-1)/(T-1)) \cdot (\beta_{\text{end}} - \beta_{\text{start}}) + \beta_{\text{start}}$. S-shaped transition with gentle start and end.

**Learned**: Variational Diffusion Models (Kingma et al., 2021) parameterise $\log \text{SNR}(t)$ as a monotonic neural network, learning the optimal schedule end-to-end.

### Comparison

| Schedule | $\bar{\alpha}$ at 25% | $\bar{\alpha}$ at 50% | $\bar{\alpha}$ at 75% | Key property |
|----------|----------------------|----------------------|----------------------|-------------|
| Linear | ~0.80 | ~0.50 | ~0.15 | Fast early decay |
| Cosine | ~0.90 | ~0.70 | ~0.35 | Uniform log-SNR decay |

## Connection to the Training Objective

The schedule affects the implicit loss weighting:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\!\left[w(t)\, \|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

Different schedules implicitly apply different $w(t)$. The cosine schedule provides more uniform weighting, preventing any single noise level from dominating the gradient.

## PyTorch Implementation

```python
import torch
import numpy as np


class ForwardDiffusion:
    """Forward diffusion process with configurable noise schedule."""

    def __init__(self, T: int = 1000, schedule: str = 'cosine'):
        self.T = T

        if schedule == 'linear':
            self.betas = self._linear_schedule()
        elif schedule == 'cosine':
            self.betas = self._cosine_schedule()
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.snr = self.alpha_bars / (1.0 - self.alpha_bars)

    def _linear_schedule(
        self, beta_start: float = 1e-4, beta_end: float = 0.02
    ) -> torch.Tensor:
        return torch.linspace(beta_start, beta_end, self.T)

    def _cosine_schedule(self, s: float = 0.008) -> torch.Tensor:
        steps = self.T + 1
        t = torch.linspace(0, self.T, steps)
        f_t = torch.cos((t / self.T + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bars = f_t / f_t[0]
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        return torch.clamp(betas, 0, 0.999)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from q(x_t | x_0).

        Args:
            x_0: Clean data [batch_size, ...].
            t: Timesteps [batch_size], values in {0, ..., T-1}.
            noise: Optional pre-sampled noise.

        Returns:
            (x_t, noise): Noisy samples and the noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar_t = self.alpha_bars[t]

        # Reshape for broadcasting with arbitrary data dimensions
        while len(alpha_bar_t.shape) < len(x_0.shape):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise
```

## Key Properties

**No learning required.** The forward process has no trainable parameters. It is a fixed corruption procedure.

**Training efficiency.** Direct sampling via $q(x_t|x_0)$ means each training iteration requires only a single forward pass through the network, regardless of $T$.

**Information destruction.** The forward process progressively destroys mutual information between $x_t$ and $x_0$. The reverse process must learn to recover this information—the core challenge.

**Continuous-time limit.** As $T \to \infty$ with appropriate scaling, the discrete chain converges to the SDE $dx = -\frac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dW$, connecting to the continuous framework of §25.5.
