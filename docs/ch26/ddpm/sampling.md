# DDPM Sampling

## Overview

Sampling from a trained DDPM generates data by iteratively denoising pure Gaussian noise through the learned reverse process. Starting from $x_T \sim \mathcal{N}(0, I)$, each step applies the model's noise prediction $\epsilon_\theta(x_t, t)$ to compute a less noisy version $x_{t-1}$, progressively recovering structure until a clean sample $x_0$ emerges after $T$ steps. This section derives the reverse sampling equations from the posterior, details the two standard variance parameterizations, presents a complete PyTorch implementation with classifier and classifier-free guidance, and analyzes the computational cost that motivates the accelerated sampling methods of [DDIM](../ddim/fundamentals.md).

---

## 1. Reverse Process Derivation

### 1.1 The Reverse Conditional

The [forward process](../foundations/forward_process.md) adds noise according to $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)$. To generate samples, we need the reverse:

$$q(x_{t-1} \mid x_t) = \int q(x_{t-1} \mid x_t, x_0)\, q(x_0 \mid x_t)\, dx_0$$

This is intractable because $q(x_0 \mid x_t)$ depends on the unknown data distribution. However, the **posterior conditioned on $x_0$** is tractable:

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1};\, \tilde{\mu}_t(x_t, x_0),\, \tilde{\beta}_t I)$$

where, applying Bayes' theorem to Gaussian conditionals:

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1 - \bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t$$

$$\tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)}\, \beta_t$$

Here $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$.

### 1.2 From $x_0$-Prediction to $\epsilon$-Prediction

Since the [forward process](../foundations/forward_process.md) gives $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$, we can express $x_0$ in terms of $x_t$ and $\epsilon$:

$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon}{\sqrt{\bar{\alpha}_t}}$$

Substituting the model's noise prediction $\epsilon_\theta(x_t, t)$ for $\epsilon$ and inserting into $\tilde{\mu}_t$:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t) \right)$$

This is the mean of the learned reverse step $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I)$.

### 1.3 The $x_0$-Prediction View

Equivalently, the model implicitly predicts $x_0$:

$$\hat{x}_0(x_t, t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

and the mean is:

$$\mu_\theta(x_t, t) = \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1 - \bar{\alpha}_t}\, \hat{x}_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t$$

Clipping $\hat{x}_0 \in [-1, 1]$ (for data normalized to this range) before computing $\mu_\theta$ stabilizes sampling and improves sample quality, especially for early timesteps where the $x_0$ prediction is noisy.

---

## 2. Variance Parameterization

### 2.1 Fixed Variance Options

The original DDPM paper uses a fixed reverse variance $\sigma_t^2$. Two natural choices correspond to different assumptions:

**Lower bound** — posterior variance assuming $x_0$ is known:

$$\sigma_t^2 = \tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)}\, \beta_t$$

**Upper bound** — forward process variance:

$$\sigma_t^2 = \beta_t$$

Both produce similar sample quality. The lower bound $\tilde{\beta}_t$ yields slightly better log-likelihoods, while $\beta_t$ can produce slightly sharper samples.

### 2.2 Learned Variance (Improved DDPM)

Nichol & Dhariwal (2021) parameterize the variance as a learned interpolation in log-space:

$$\log \sigma_t^2 = v_t \log \beta_t + (1 - v_t) \log \tilde{\beta}_t$$

where $v_t$ is an additional scalar output of the network for each timestep. This is trained with a hybrid objective combining the standard [training loss](training.md) with a variational bound term:

$$\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{\text{simple}} + \lambda\, \mathcal{L}_{\text{vlb}}$$

with $\lambda = 0.001$ to prevent the VLB term from dominating early training.

---

## 3. The Sampling Algorithm

### 3.1 Pseudocode

```
Algorithm: DDPM Ancestral Sampling
───────────────────────────────────
Input:  Trained noise predictor ε_θ, noise schedule {β_t, ᾱ_t}_{t=1}^T
Output: Generated sample x_0

 1. Sample x_T ~ N(0, I)
 2. for t = T, T-1, ..., 1 do
 3.     ε = ε_θ(x_t, t)                                    ▷ Predict noise
 4.     x̂_0 = (x_t − √(1−ᾱ_t) · ε) / √ᾱ_t              ▷ Predict clean image
 5.     x̂_0 = clip(x̂_0, −1, 1)                            ▷ Optional: stabilize
 6.     μ = (√ᾱ_{t-1} · β_t)/(1−ᾱ_t) · x̂_0
            + (√α_t · (1−ᾱ_{t-1}))/(1−ᾱ_t) · x_t         ▷ Posterior mean
 7.     if t > 1 then
 8.         z ~ N(0, I)
 9.         x_{t-1} = μ + σ_t · z                           ▷ Stochastic step
10.     else
11.         x_{t-1} = μ                                     ▷ Final step: no noise
12.     end if
13. end for
14. return x_0
```

The final step ($t = 1 \to t = 0$) is deterministic because adding noise at $t = 0$ would corrupt the generated sample.

### 3.2 Why Stochasticity Matters

The noise $z$ added at each step (line 8) is not an artifact—it is essential. The reverse process $p_\theta(x_{t-1} \mid x_t)$ is a Gaussian distribution, and sampling from it (rather than just taking the mean) ensures the generated distribution matches the data distribution in the limit of perfect learning. Removing this noise corresponds to [DDIM](../ddim/deterministic.md) deterministic sampling, which trades diversity for consistency.

---

## 4. Implementation

```python
"""
DDPM Sampling
=============
Complete implementation of DDPM ancestral sampling with support for
unconditional generation, classifier guidance, and classifier-free guidance.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from tqdm import tqdm


class DDPMSampler:
    """DDPM ancestral sampler with precomputed schedule coefficients."""

    def __init__(
        self,
        model: nn.Module,
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        variance_type: str = "fixed_lower",
        clip_denoised: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Parameters
        ----------
        model : nn.Module
            Trained noise prediction network ε_θ(x_t, t) or
            ε_θ(x_t, t, c) for conditional models.
        n_timesteps : int
            Number of diffusion steps T.
        beta_start, beta_end : float
            Endpoints of the noise schedule.
        beta_schedule : str
            Schedule type: 'linear' or 'cosine'.
        variance_type : str
            'fixed_lower' (β̃_t), 'fixed_upper' (β_t), or 'learned'.
        clip_denoised : bool
            Whether to clip x̂_0 predictions to [-1, 1].
        device : torch.device
            Computation device.
        """
        self.model = model
        self.n_timesteps = n_timesteps
        self.variance_type = variance_type
        self.clip_denoised = clip_denoised
        self.device = device

        # --- Build noise schedule ---
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, n_timesteps)
        elif beta_schedule == "cosine":
            # Nichol & Dhariwal (2021) cosine schedule
            steps = torch.arange(n_timesteps + 1, dtype=torch.float64)
            f = torch.cos((steps / n_timesteps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = betas.clamp(max=0.999).float()
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # --- Precompute coefficients (all shape [T]) ---
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        self.sqrt_alphas = torch.sqrt(alphas).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

        # Posterior mean coefficients: μ = coef1 * x̂_0 + coef2 * x_t
        self.posterior_mean_coef1 = (
            torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod)
        ).to(device)
        self.posterior_mean_coef2 = (
            torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)

        # Posterior variance: β̃_t
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        # Clamp log for numerical stability at t=0
        self.posterior_log_variance = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        ).to(device)

    # ------------------------------------------------------------------
    # Core sampling step
    # ------------------------------------------------------------------
    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: int,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        x0 = (
            x_t - self.sqrt_one_minus_alphas_cumprod[t] * eps_pred
        ) / self.sqrt_alphas_cumprod[t]

        if self.clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        return x0

    def posterior_mean(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Compute posterior mean μ_θ(x_t, t)."""
        return self.posterior_mean_coef1[t] * x0_pred + self.posterior_mean_coef2[t] * x_t

    def get_variance(self, t: int) -> torch.Tensor:
        """Return σ_t² for the reverse step."""
        if self.variance_type == "fixed_lower":
            return self.posterior_variance[t]
        elif self.variance_type == "fixed_upper":
            return self.betas[t]
        else:
            raise ValueError(f"Variance type '{self.variance_type}' not supported here. "
                             "Use 'learned' variance via the model output.")

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single reverse step: sample x_{t-1} ~ p_θ(x_{t-1} | x_t).

        Parameters
        ----------
        x_t : Tensor of shape (B, C, H, W) or (B, D)
            Current noisy sample.
        t : int
            Current timestep.
        condition : Tensor, optional
            Conditioning information (class labels, text embeddings, etc.).

        Returns
        -------
        x_{t-1} : Tensor, same shape as x_t.
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Predict noise
        if condition is not None:
            eps_pred = self.model(x_t, t_tensor, condition)
        else:
            eps_pred = self.model(x_t, t_tensor)

        # Predict x_0 and compute posterior mean
        x0_pred = self.predict_x0(x_t, t, eps_pred)
        mean = self.posterior_mean(x_t, x0_pred, t)

        # Sample
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = self.get_variance(t).sqrt()
            return mean + sigma * noise
        else:
            return mean  # No noise at final step

    # ------------------------------------------------------------------
    # Full sampling loops
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        condition: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        trajectory_interval: int = 100,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples via ancestral sampling.

        Parameters
        ----------
        shape : tuple
            Shape of samples to generate, e.g. (B, C, H, W).
        condition : Tensor, optional
            Conditioning information.
        return_trajectory : bool
            If True, also return intermediate x_t at regular intervals.
        trajectory_interval : int
            Save trajectory every this many steps.
        show_progress : bool
            Display tqdm progress bar.

        Returns
        -------
        samples : Tensor of shape `shape`.
        trajectory : list of Tensors (optional).
        """
        self.model.eval()

        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        trajectory = [x.cpu().clone()] if return_trajectory else None

        timesteps = range(self.n_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling", total=self.n_timesteps)

        for t in timesteps:
            x = self.p_sample(x, t, condition=condition)

            if return_trajectory and t % trajectory_interval == 0 and t > 0:
                trajectory.append(x.cpu().clone())

        if return_trajectory:
            trajectory.append(x.cpu().clone())  # Final sample
            return x, trajectory
        return x

    @torch.no_grad()
    def sample_with_classifier_guidance(
        self,
        shape: tuple,
        classifier: nn.Module,
        class_label: int,
        guidance_scale: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Sample with classifier guidance (Dhariwal & Nichol, 2021).

        Modifies the score: ∇ log p(x_t | y) = ∇ log p(x_t) + s · ∇ log p(y | x_t)

        Parameters
        ----------
        shape : tuple
            Shape of samples to generate.
        classifier : nn.Module
            Noise-aware classifier p(y | x_t, t).
        class_label : int
            Target class index.
        guidance_scale : float
            Strength of classifier guidance (s).
        show_progress : bool
            Display progress bar.

        Returns
        -------
        samples : Tensor of shape `shape`.
        """
        self.model.eval()
        classifier.eval()

        x = torch.randn(shape, device=self.device)
        labels = torch.full((shape[0],), class_label, device=self.device, dtype=torch.long)

        timesteps = range(self.n_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Classifier-Guided Sampling")

        for t in timesteps:
            batch_size = x.shape[0]
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Compute classifier gradient
            x_in = x.detach().requires_grad_(True)
            with torch.enable_grad():
                log_probs = torch.log_softmax(classifier(x_in, t_tensor), dim=-1)
                selected = log_probs[range(batch_size), labels]
                grad = torch.autograd.grad(selected.sum(), x_in)[0]

            # Predict noise and shift by guidance
            eps_pred = self.model(x, t_tensor)
            eps_guided = eps_pred - guidance_scale * self.sqrt_one_minus_alphas_cumprod[t] * grad

            # Denoise with guided noise prediction
            x0_pred = self.predict_x0(x, t, eps_guided)
            mean = self.posterior_mean(x, x0_pred, t)

            if t > 0:
                sigma = self.get_variance(t).sqrt()
                x = mean + sigma * torch.randn_like(x)
            else:
                x = mean

        return x

    @torch.no_grad()
    def sample_classifier_free(
        self,
        shape: tuple,
        condition: torch.Tensor,
        guidance_scale: float = 7.5,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Sample with classifier-free guidance (Ho & Salimans, 2022).

        Combines conditional and unconditional predictions:
            ε̃ = ε_θ(x_t, t, ∅) + s · (ε_θ(x_t, t, c) − ε_θ(x_t, t, ∅))

        Parameters
        ----------
        shape : tuple
            Shape of samples to generate.
        condition : Tensor
            Conditioning signal (class labels, text embeddings, etc.).
        guidance_scale : float
            Guidance strength s. Values > 1 sharpen the conditional distribution.
        show_progress : bool
            Display progress bar.

        Returns
        -------
        samples : Tensor of shape `shape`.
        """
        self.model.eval()

        x = torch.randn(shape, device=self.device)
        # Null condition for unconditional prediction (zeros or learned null token)
        null_condition = torch.zeros_like(condition)

        timesteps = range(self.n_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Classifier-Free Sampling")

        for t in timesteps:
            batch_size = x.shape[0]
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Unconditional and conditional predictions
            eps_uncond = self.model(x, t_tensor, null_condition)
            eps_cond = self.model(x, t_tensor, condition)

            # Classifier-free guidance combination
            eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # Denoise
            x0_pred = self.predict_x0(x, t, eps_guided)
            mean = self.posterior_mean(x, x0_pred, t)

            if t > 0:
                sigma = self.get_variance(t).sqrt()
                x = mean + sigma * torch.randn_like(x)
            else:
                x = mean

        return x
```

---

## 5. Sampling Trajectories and Intermediate Predictions

### 5.1 Visualizing the Denoising Process

Saving intermediate states reveals how structure emerges:

```python
def visualize_trajectory(
    sampler: DDPMSampler,
    shape: tuple,
    save_steps: list = None,
) -> list:
    """
    Generate samples and save intermediate x_t and x̂_0 predictions.

    Parameters
    ----------
    sampler : DDPMSampler
    shape : tuple
        Sample shape, e.g. (4, 3, 64, 64).
    save_steps : list of int
        Timesteps at which to save. Defaults to evenly spaced.

    Returns
    -------
    snapshots : list of dicts with keys 'timestep', 'x_t', 'x0_pred'.
    """
    if save_steps is None:
        save_steps = list(range(0, sampler.n_timesteps, sampler.n_timesteps // 10))

    sampler.model.eval()
    x = torch.randn(shape, device=sampler.device)
    snapshots = []

    for t in range(sampler.n_timesteps - 1, -1, -1):
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, device=sampler.device, dtype=torch.long)

        with torch.no_grad():
            eps_pred = sampler.model(x, t_tensor)
            x0_pred = sampler.predict_x0(x, t, eps_pred)

        if t in save_steps:
            snapshots.append({
                "timestep": t,
                "x_t": x.cpu().clone(),
                "x0_pred": x0_pred.cpu().clone(),
            })

        # Take reverse step
        with torch.no_grad():
            x = sampler.p_sample(x, t)

    # Final sample
    snapshots.append({"timestep": 0, "x_t": x.cpu().clone(), "x0_pred": x.cpu().clone()})
    return snapshots
```

### 5.2 What the Trajectory Reveals

| Stage | Timestep Range | Observation |
|-------|---------------|-------------|
| Early denoising | $t \in [T, 0.7T]$ | Global structure emerges: layout, large shapes, dominant colors |
| Middle denoising | $t \in [0.7T, 0.3T]$ | Medium-scale features: object boundaries, texture regions |
| Late denoising | $t \in [0.3T, 0]$ | Fine details: edges, textures, high-frequency content |

The $\hat{x}_0$ predictions at early timesteps are blurry but structurally coherent, progressively sharpening as $t$ decreases. This coarse-to-fine generation is a hallmark of diffusion models.

---

## 6. Computational Analysis

### 6.1 Cost per Sample

Each sampling step requires one forward pass through the model $\epsilon_\theta$. For a U-Net with $P$ parameters processing an image of size $C \times H \times W$:

| Component | Cost per Step | Total ($T$ steps) |
|-----------|--------------|-------------------|
| Model forward pass | $O(P \cdot C \cdot H \cdot W)$ | $T \times O(P \cdot C \cdot H \cdot W)$ |
| Noise generation | $O(C \cdot H \cdot W)$ | Negligible |
| Coefficient lookup | $O(1)$ | Negligible |

For a typical U-Net (~100M parameters) at 256×256 resolution with $T = 1000$:

| Metric | Approximate Value |
|--------|------------------|
| Time per step | ~15 ms (A100 GPU) |
| Total sampling time | ~15 s per image |
| Memory (batch=1) | ~4 GB |
| Memory (batch=16) | ~20 GB |

### 6.2 The Speed Problem

Standard DDPM requires $T = 1000$ sequential model evaluations per sample. This is orders of magnitude slower than single-pass generators like GANs (~20 ms per image). This fundamental limitation motivates:

| Acceleration Method | Steps Required | Quality | Section |
|--------------------|---------------|---------|---------|
| **DDPM** (baseline) | 1000 | Best | This page |
| **DDIM** | 50–100 | Near-DDPM | [DDIM Fundamentals](../ddim/fundamentals.md) |
| **DDIM (deterministic)** | 20–50 | Good | [Deterministic Sampling](../ddim/deterministic.md) |
| **Probability Flow ODE** | 20–100 | Good | [Probability Flow](../sde/probability_flow.md) |
| **Progressive Distillation** | 4–8 | Good | Architecture-dependent |
| **Consistency Models** | 1–2 | Moderate | Single-step generation |

---

## 7. Practical Considerations

### 7.1 Numerical Precision

Use `float32` for sampling even if training used mixed precision. Half-precision accumulation errors compound over 1000 steps and produce artifacts:

```python
# Ensure float32 for sampling
model = model.float()
x = torch.randn(shape, device=device, dtype=torch.float32)
```

### 7.2 Clipping $\hat{x}_0$

Clipping the predicted $x_0$ to $[-1, 1]$ (or the data range) prevents the posterior mean from drifting due to inaccurate predictions at high noise levels. This is especially important for:

- Early timesteps ($t$ close to $T$) where predictions are unreliable
- Models trained without extensive noise schedule tuning
- Conditional generation where guidance can push predictions out of range

### 7.3 Batch Sampling

Generate multiple samples in parallel by increasing the batch dimension. Memory permitting, this is far more efficient than sequential generation:

```python
# Generate 64 samples in parallel
samples = sampler.sample(shape=(64, 3, 64, 64), show_progress=True)
```

### 7.4 Reproducibility

Fix the initial noise for reproducible generation. This enables controlled comparisons across models or guidance scales:

```python
# Reproducible generation
generator = torch.Generator(device=device).manual_seed(42)
x_T = torch.randn(shape, device=device, generator=generator)
```

---

## 8. Finance Applications

DDPM sampling extends naturally to financial data generation:

| Application | Sampling Approach | Notes |
|-------------|------------------|-------|
| Synthetic return paths | Unconditional sampling | Generate realistic multivariate return distributions |
| Regime-conditional scenarios | Classifier-free guidance | Condition on market regime labels |
| Stress testing | Classifier guidance | Guide toward tail events using a risk classifier |
| Portfolio scenario generation | Conditional sampling | Condition on macro variables for forward-looking scenarios |
| Missing data imputation | Inpainting-style sampling | Fix observed values, sample missing entries |

For details on financial applications, see [Time Series Generation](../finance/time_series.md) and [Scenario Generation](../finance/scenarios.md).

---

## 9. Key Takeaways

1. **DDPM sampling reverses the forward process** by iteratively applying the learned noise predictor $\epsilon_\theta(x_t, t)$ to compute the posterior mean $\mu_\theta$ and sampling from the Gaussian reverse step.

2. **The $\hat{x}_0$-prediction view** provides intuition: at each step, the model predicts a clean image, and the posterior mean interpolates between this prediction and the current noisy state.

3. **Variance choices** ($\tilde{\beta}_t$ vs $\beta_t$ vs learned) have modest impact on sample quality but affect log-likelihood. Learned variance with cosine schedule gives the best likelihood bounds.

4. **The final step is deterministic** — noise is only added for $t > 0$ to maintain the correct reverse distribution.

5. **Classifier guidance** modifies the score with gradients from an external classifier; **classifier-free guidance** avoids a separate classifier by interpolating between conditional and unconditional model predictions.

6. **The 1000-step requirement is the primary limitation**, motivating [DDIM](../ddim/fundamentals.md) and other accelerated sampling methods.

---

## Exercises

### Exercise 1: Variance Comparison

Implement sampling with both $\sigma_t = \sqrt{\tilde{\beta}_t}$ and $\sigma_t = \sqrt{\beta_t}$. Generate 1000 samples with each and compare FID scores. Is the difference significant?

### Exercise 2: Clipping Ablation

Generate samples with and without $\hat{x}_0$ clipping. Visualize the $\hat{x}_0$ predictions at $t = 900, 500, 100$ for both settings. At which timesteps does clipping have the largest effect?

### Exercise 3: Trajectory Visualization

Use `visualize_trajectory` to save snapshots every 100 steps. Create a figure showing both $x_t$ and $\hat{x}_0$ at each saved step. Describe the coarse-to-fine generation process.

### Exercise 4: Guidance Scale Sweep

Using classifier-free guidance, generate samples with $s \in \{1.0, 3.0, 5.0, 7.5, 10.0, 15.0\}$. Plot the diversity-quality tradeoff (FID vs. number of unique modes). At what scale does mode collapse become visible?

### Exercise 5: Sampling Budget

If your application requires generating 10,000 samples with a 100M-parameter U-Net at 256×256, estimate the total GPU-hours needed for DDPM ($T = 1000$) vs DDIM ($T = 50$). At \$2/GPU-hour, what is the cost difference?

---

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems (NeurIPS)*.
2. Nichol, A. Q. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *Proceedings of the 38th International Conference on Machine Learning (ICML)*.
3. Dhariwal, P. & Nichol, A. Q. (2021). Diffusion Models Beat GANs on Image Synthesis. *Advances in Neural Information Processing Systems (NeurIPS)*.
4. Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*.
5. Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. *Proceedings of the 9th International Conference on Learning Representations (ICLR)*.
