# DDPM Training

## Overview

Training a DDPM amounts to learning a noise prediction network $\epsilon_\theta(x_t, t)$ that can reverse the [forward diffusion process](../foundations/forward_process.md). The [training objective](../foundations/training_objective.md) simplifies to a reweighted variational bound that reduces to a remarkably simple mean squared error loss between predicted and actual noise. This section develops the complete training pipeline—from the loss derivation and its connection to score matching, through practical components like EMA, gradient clipping, and mixed-precision training, to a production-ready PyTorch implementation with logging and checkpointing.

Despite the simplicity of the per-step loss, training high-quality diffusion models requires careful attention to noise schedule design (covered in [Noise Schedule](noise_schedule.md)), architecture choices (covered in [U-Net for Diffusion](../architectures/unet.md)), and the training practices detailed here.

---

## 1. The Training Objective

### 1.1 From Variational Bound to Simple Loss

The [training objective derivation](../foundations/training_objective.md) shows that maximizing the variational lower bound on $\log p_\theta(x_0)$ yields a sum of KL divergences between the true reverse posterior $q(x_{t-1} \mid x_t, x_0)$ and the learned reverse $p_\theta(x_{t-1} \mid x_t)$. Since both are Gaussian, each KL reduces to a weighted MSE on the means.

Ho et al. (2020) showed that reparameterizing the mean in terms of the noise $\epsilon$ and dropping the weighting gives the **simplified objective**:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\, x_0,\, \epsilon}\left[\left\| \epsilon - \epsilon_\theta\bigl(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\; t\bigr) \right\|^2\right]$$

where $t \sim \text{Uniform}\{1, \ldots, T\}$, $x_0 \sim q(x_0)$, and $\epsilon \sim \mathcal{N}(0, I)$.

### 1.2 Prediction Parameterizations

The model can equivalently be trained to predict different quantities:

| Parameterization | Target | Loss | Connection |
|-----------------|--------|------|------------|
| $\epsilon$-prediction | $\epsilon$ (noise) | $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ | Original DDPM; predicts the noise added at step $t$ |
| $x_0$-prediction | $x_0$ (clean data) | $\|x_0 - x_{0,\theta}(x_t, t)\|^2$ | Equivalent via $x_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon)/\sqrt{\bar{\alpha}_t}$ |
| $v$-prediction | $v = \sqrt{\bar{\alpha}_t}\,\epsilon - \sqrt{1-\bar{\alpha}_t}\,x_0$ | $\|v - v_\theta(x_t, t)\|^2$ | Salimans & Ho (2022); better for high-resolution and latent diffusion |

The $\epsilon$-prediction parameterization is standard for DDPM. The $v$-prediction parameterization improves training stability when using the cosine [noise schedule](noise_schedule.md) and is adopted by several modern architectures including [Latent Diffusion](../architectures/latent_diffusion.md).

### 1.3 Connection to Score Matching

The DDPM objective is equivalent to [denoising score matching](../score_based/denoising_score_matching.md). Since the score of the noisy distribution is:

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

training $\epsilon_\theta$ to predict $\epsilon$ is equivalent to training a score network $s_\theta(x_t, t) = -\epsilon_\theta(x_t, t) / \sqrt{1 - \bar{\alpha}_t}$ to match $\nabla_{x_t} \log q(x_t)$. This connection unifies DDPM with the [score-based framework](../score_based/score_function.md).

### 1.4 Loss Weighting

The simplified loss assigns equal weight to all timesteps, which effectively upweights high-noise timesteps (large $t$) relative to the variational bound. Alternative weightings offer different tradeoffs:

$$\mathcal{L}_{\text{weighted}} = \mathbb{E}_{t,\, x_0,\, \epsilon}\left[w(t)\,\left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2\right]$$

| Weighting | $w(t)$ | Effect |
|-----------|--------|--------|
| Simple (DDPM) | $1$ | Emphasizes global structure; best FID |
| VLB | $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}$ | Tight variational bound; best log-likelihood |
| SNR | $\bar{\alpha}_t / (1 - \bar{\alpha}_t)$ | Signal-to-noise ratio weighting |
| Min-SNR-$\gamma$ | $\min(\text{SNR}(t),\, \gamma) / \text{SNR}(t)$ | Clips large weights at low noise; stable training |
| $P2$ (perception) | $(1 + \text{SNR}(t))^{-1}$ | Emphasizes perceptually important timesteps |

The simplified loss ($w(t) = 1$) consistently produces the best sample quality (FID) despite not optimizing the variational bound tightly. This is because it allocates more gradient signal to high-noise timesteps where global structure is determined.

---

## 2. Training Algorithm

### 2.1 Pseudocode

```
Algorithm: DDPM Training
────────────────────────
Input:  Dataset D, noise schedule {β_t, ᾱ_t}_{t=1}^T, model ε_θ
Output: Trained parameters θ, EMA parameters θ_ema

 1. Initialize θ randomly, θ_ema ← θ
 2. repeat
 3.     Sample x_0 ~ D                                       ▷ Mini-batch from data
 4.     Sample t ~ Uniform{1, ..., T}                        ▷ Random timestep per sample
 5.     Sample ε ~ N(0, I)                                   ▷ Random noise
 6.     x_t ← √ᾱ_t · x_0 + √(1−ᾱ_t) · ε                  ▷ Forward diffusion (one step)
 7.     L ← || ε − ε_θ(x_t, t) ||²                         ▷ Simple MSE loss
 8.     θ ← θ − η · ∇_θ L                                   ▷ Gradient step (with clipping)
 9.     θ_ema ← μ · θ_ema + (1−μ) · θ                      ▷ EMA update (μ = 0.9999)
10. until converged
```

Key observations:

- **No sequential diffusion**: Training does not run the full $T$-step chain. Each iteration independently samples a random $(t, \epsilon)$ pair and computes the noisy input $x_t$ in closed form via the marginal $q(x_t \mid x_0)$.
- **One model for all $t$**: The same network $\epsilon_\theta$ handles all noise levels, conditioned on $t$ via sinusoidal embeddings.
- **EMA for sampling**: The exponential moving average of weights produces smoother, higher-quality samples than the raw training weights.

---

## 3. Implementation

```python
"""
DDPM Training Pipeline
======================
Complete training implementation with EMA, gradient clipping, mixed-precision
training, loss weighting options, logging, and checkpointing.
"""

import copy
import math
from pathlib import Path
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────────
# Noise Schedule
# ──────────────────────────────────────────────────────────────────────

def build_schedule(
    n_timesteps: int = 1000,
    schedule: str = "linear",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Build diffusion noise schedule and precompute all derived quantities.

    Parameters
    ----------
    n_timesteps : int
        Total number of diffusion steps T.
    schedule : str
        'linear' or 'cosine'.
    beta_start, beta_end : float
        Endpoints for linear schedule.
    device : torch.device

    Returns
    -------
    dict of precomputed Tensors, all of shape [T].
    """
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, n_timesteps)
    elif schedule == "cosine":
        steps = torch.arange(n_timesteps + 1, dtype=torch.float64)
        f = torch.cos((steps / n_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = (f / f[0]).float()
        betas = (1 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(max=0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas.to(device),
        "alphas_cumprod": alphas_cumprod.to(device),
        "sqrt_alphas_cumprod": alphas_cumprod.sqrt().to(device),
        "sqrt_one_minus_alphas_cumprod": (1.0 - alphas_cumprod).sqrt().to(device),
        "snr": (alphas_cumprod / (1.0 - alphas_cumprod)).to(device),
    }


# ──────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────

class EMAModel:
    """
    Exponential Moving Average of model parameters.

    EMA parameters produce smoother, higher-quality samples than
    raw training parameters. Standard decay is 0.9999.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters: θ_ema ← μ·θ_ema + (1−μ)·θ."""
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ──────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────

class DDPMTrainer:
    """
    Complete DDPM training pipeline.

    Parameters
    ----------
    model : nn.Module
        Noise prediction network ε_θ(x_t, t) -> ε.
        Must accept (x_t: Tensor, t: LongTensor) and return Tensor same shape as x_t.
    n_timesteps : int
        Number of diffusion steps T.
    beta_schedule : str
        'linear' or 'cosine'.
    beta_start, beta_end : float
        Endpoints for linear schedule.
    prediction : str
        'epsilon' (noise), 'x0' (clean data), or 'v' (velocity).
    loss_weighting : str
        'simple' (uniform), 'snr', or 'min_snr' with gamma clipping.
    min_snr_gamma : float
        Clipping value for min-SNR weighting.
    lr : float
        Learning rate for AdamW.
    weight_decay : float
        AdamW weight decay.
    ema_decay : float
        EMA decay rate (0.9999 standard).
    grad_clip : float
        Maximum gradient norm (0 to disable).
    use_amp : bool
        Enable automatic mixed-precision training.
    device : torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        n_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction: Literal["epsilon", "x0", "v"] = "epsilon",
        loss_weighting: Literal["simple", "snr", "min_snr"] = "simple",
        min_snr_gamma: float = 5.0,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        ema_decay: float = 0.9999,
        grad_clip: float = 1.0,
        use_amp: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.to(device)
        self.device = device
        self.n_timesteps = n_timesteps
        self.prediction = prediction
        self.loss_weighting = loss_weighting
        self.min_snr_gamma = min_snr_gamma
        self.grad_clip = grad_clip
        self.use_amp = use_amp

        # Noise schedule
        self.schedule = build_schedule(
            n_timesteps, beta_schedule, beta_start, beta_end, device
        )

        # EMA
        self.ema = EMAModel(model, decay=ema_decay)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Mixed-precision scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

        self.global_step = 0

    # ── Forward diffusion (closed-form) ──────────────────────────────

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I).

        Parameters
        ----------
        x_0 : Tensor of shape (B, ...)
            Clean data samples.
        t : LongTensor of shape (B,)
            Timestep indices.
        noise : Tensor, optional
            Pre-sampled noise (for reproducibility).

        Returns
        -------
        x_t : Tensor, same shape as x_0.
        noise : Tensor, the noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.schedule["sqrt_alphas_cumprod"][t]
        sqrt_one_minus = self.schedule["sqrt_one_minus_alphas_cumprod"][t]

        # Reshape coefficients for broadcasting: (B,) -> (B, 1, 1, ...) 
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    # ── Compute training target ──────────────────────────────────────

    def _get_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return the training target for the chosen parameterization."""
        if self.prediction == "epsilon":
            return noise
        elif self.prediction == "x0":
            return x_0
        elif self.prediction == "v":
            sqrt_alpha = self.schedule["sqrt_alphas_cumprod"][t]
            sqrt_one_minus = self.schedule["sqrt_one_minus_alphas_cumprod"][t]
            while sqrt_alpha.dim() < x_0.dim():
                sqrt_alpha = sqrt_alpha.unsqueeze(-1)
                sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
            return sqrt_alpha * noise - sqrt_one_minus * x_0
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction}")

    # ── Loss weighting ───────────────────────────────────────────────

    def _get_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Return per-sample loss weight w(t)."""
        if self.loss_weighting == "simple":
            return torch.ones(t.shape[0], device=self.device)
        elif self.loss_weighting == "snr":
            return self.schedule["snr"][t]
        elif self.loss_weighting == "min_snr":
            snr = self.schedule["snr"][t]
            return torch.clamp(snr, max=self.min_snr_gamma) / snr
        else:
            raise ValueError(f"Unknown loss weighting: {self.loss_weighting}")

    # ── Training loss ────────────────────────────────────────────────

    def compute_loss(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute the DDPM training loss.

        Parameters
        ----------
        x_0 : Tensor of shape (B, ...)
            Clean data samples.
        condition : Tensor, optional
            Conditioning signal for conditional models.

        Returns
        -------
        loss : scalar Tensor
            Weighted MSE loss.
        metrics : dict
            Per-step diagnostics.
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device)

        # Forward diffusion
        x_t, noise = self.q_sample(x_0, t)

        # Model prediction
        if condition is not None:
            pred = self.model(x_t, t, condition)
        else:
            pred = self.model(x_t, t)

        # Target
        target = self._get_target(x_0, noise, t)

        # Per-sample MSE
        per_sample_loss = F.mse_loss(pred, target, reduction="none")
        per_sample_loss = per_sample_loss.flatten(start_dim=1).mean(dim=1)  # (B,)

        # Apply weighting
        weights = self._get_loss_weight(t)
        loss = (weights * per_sample_loss).mean()

        metrics = {
            "loss": loss.item(),
            "loss_unweighted": per_sample_loss.mean().item(),
            "mean_t": t.float().mean().item(),
        }
        return loss, metrics

    # ── Single training step ─────────────────────────────────────────

    def train_step(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Execute one training step: forward, backward, optimize, EMA update.

        Parameters
        ----------
        x_0 : Tensor
            Clean data batch.
        condition : Tensor, optional
            Conditioning signal.

        Returns
        -------
        metrics : dict with 'loss', 'grad_norm', 'lr', 'step'.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Forward + loss (with optional mixed precision)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            loss, metrics = self.compute_loss(x_0, condition)

        # Backward
        self.scaler.scale(loss).backward()

        # Unscale for gradient clipping
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        grad_norm = 0.0
        if self.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            ).item()

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # EMA update
        self.ema.update(self.model)

        self.global_step += 1
        metrics.update({
            "grad_norm": grad_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
            "step": self.global_step,
        })
        return metrics

    # ── Epoch loop ───────────────────────────────────────────────────

    def train_epoch(
        self,
        dataloader: DataLoader,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        log_interval: int = 100,
    ) -> dict:
        """
        Train for one epoch.

        Parameters
        ----------
        dataloader : DataLoader
            Yields batches of (x_0,) or (x_0, condition).
        lr_scheduler : LRScheduler, optional
            Stepped after each batch.
        log_interval : int
            Print metrics every this many steps.

        Returns
        -------
        epoch_metrics : dict with averaged loss.
        """
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            # Unpack batch
            if isinstance(batch, (list, tuple)):
                x_0 = batch[0].to(self.device)
                condition = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                x_0 = batch.to(self.device)
                condition = None

            metrics = self.train_step(x_0, condition)
            total_loss += metrics["loss"]
            n_batches += 1

            if lr_scheduler is not None:
                lr_scheduler.step()

            if self.global_step % log_interval == 0:
                print(
                    f"  step {self.global_step:>7d} | "
                    f"loss {metrics['loss']:.4f} | "
                    f"grad_norm {metrics['grad_norm']:.3f} | "
                    f"lr {metrics['lr']:.2e}"
                )

        return {"epoch_loss": total_loss / max(n_batches, 1)}

    # ── Checkpointing ────────────────────────────────────────────────

    def save_checkpoint(self, path: str):
        """Save full training state for resumption."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "global_step": self.global_step,
                "model_state": self.model.state_dict(),
                "ema_state": self.ema.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_state": self.scaler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Restore training state from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.global_step = ckpt["global_step"]
        self.model.load_state_dict(ckpt["model_state"])
        self.ema.load_state_dict(ckpt["ema_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
```

---

## 4. Training Components in Detail

### 4.1 Exponential Moving Average (EMA)

EMA maintains a smoothed copy of the model weights:

$$\theta_{\text{ema}}^{(k)} = \mu\, \theta_{\text{ema}}^{(k-1)} + (1 - \mu)\, \theta^{(k)}$$

with decay $\mu = 0.9999$ (standard). This is critical for DDPM because:

- Sampling involves 1000 sequential applications of $\epsilon_\theta$, so small weight fluctuations from training noise compound into visible artifacts
- EMA parameters are smoother, producing consistently higher-quality samples
- The gap between EMA and raw-weight sample quality is larger for diffusion models than for most other generative models

**Always use EMA weights for sampling and evaluation; use raw weights only for training.**

Warm-up consideration: EMA is biased toward the initial random weights early in training. Some implementations use a warm-up period with $\mu = 0$ for the first few thousand steps, or apply bias correction:

$$\hat{\theta}_{\text{ema}}^{(k)} = \frac{\theta_{\text{ema}}^{(k)}}{1 - \mu^k}$$

### 4.2 Gradient Clipping

Gradient norm clipping at 1.0 is standard and essential:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Without clipping, occasional large gradients (often from high-noise timesteps near $t = T$) cause training instabilities that manifest as sudden loss spikes and degraded sample quality. Monitoring the gradient norm is a useful diagnostic: sustained values near the clip threshold suggest the learning rate may be too high.

### 4.3 Mixed-Precision Training

FP16 mixed precision provides ~2× speedup and ~50% memory reduction on modern GPUs with minimal impact on training quality:

```python
scaler = torch.amp.GradScaler("cuda")

with torch.amp.autocast("cuda"):
    loss, metrics = trainer.compute_loss(x_0)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

Important: gradient clipping must happen **after** `scaler.unscale_()` and **before** `scaler.step()` to clip in the correct scale.

### 4.4 Timestep Sampling Strategies

The standard approach samples $t \sim \text{Uniform}\{1, \ldots, T\}$, but alternative strategies can improve training efficiency:

| Strategy | Description | Benefit |
|----------|------------|---------|
| Uniform | $t \sim \text{Uniform}\{1, \ldots, T\}$ | Standard, simple |
| Importance sampling | Oversample timesteps with high loss | Faster convergence |
| Low-discrepancy | Stratified sampling across $[0, T]$ | Reduced variance |
| Loss-aware | Adaptively weight by recent per-$t$ loss | Focuses on hard timesteps |

In practice, uniform sampling works well and is the standard choice. More sophisticated strategies provide modest speedups (10–20% fewer steps to convergence).

---

## 5. Conditional Training

### 5.1 Class-Conditional DDPM

To train a class-conditional model, embed the class label and add it to the timestep embedding:

```python
class ConditionalDDPM(nn.Module):
    """Noise predictor conditioned on class labels."""

    def __init__(self, base_model: nn.Module, num_classes: int, embed_dim: int):
        super().__init__()
        self.base_model = base_model
        self.class_embed = nn.Embedding(num_classes, embed_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Class embedding is added to timestep embedding inside base_model
        class_emb = self.class_embed(y) if y is not None else None
        return self.base_model(x_t, t, context=class_emb)
```

### 5.2 Classifier-Free Guidance Training

[Classifier-free guidance](../conditional/classifier_free.md) requires training the same model both conditionally and unconditionally. During training, randomly drop the conditioning signal with probability $p_{\text{uncond}}$ (typically 10–20%):

```python
def compute_loss_cfg(
    self,
    x_0: torch.Tensor,
    condition: torch.Tensor,
    p_uncond: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Training loss with random conditioning dropout for classifier-free guidance."""
    batch_size = x_0.shape[0]
    t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device)
    x_t, noise = self.q_sample(x_0, t)

    # Randomly drop conditioning
    mask = torch.rand(batch_size, device=self.device) > p_uncond
    # Replace dropped conditions with null token (e.g., zeros)
    null_condition = torch.zeros_like(condition)
    condition_masked = torch.where(
        mask.unsqueeze(-1).expand_as(condition), condition, null_condition
    )

    pred = self.model(x_t, t, condition_masked)
    target = self._get_target(x_0, noise, t)

    loss = F.mse_loss(pred, target)
    return loss, {"loss": loss.item()}
```

At [sampling time](sampling.md), the unconditional and conditional predictions are combined:

$$\tilde{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + s \cdot \bigl(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\bigr)$$

where $s > 1$ sharpens the conditional distribution.

---

## 6. Training Hyperparameters

### 6.1 Recommended Defaults

| Hyperparameter | Value | Notes |
|---------------|-------|-------|
| Optimizer | AdamW | $\beta_1 = 0.9$, $\beta_2 = 0.999$ |
| Learning rate | $1 \times 10^{-4}$ to $2 \times 10^{-4}$ | Constant or cosine decay |
| Weight decay | $0.0$ | Some works use $0.01$ |
| Batch size | 64–256 | Larger improves stability |
| EMA decay | $0.9999$ | Higher for longer training |
| Gradient clip | $1.0$ | Essential |
| Timesteps $T$ | $1000$ | Standard |
| Noise schedule | Linear or cosine | Cosine for improved log-likelihood |
| Training iterations | 500K–1M+ | Diffusion models benefit from long training |
| Mixed precision | FP16 | ~2× speedup, minimal quality loss |

### 6.2 Scaling Considerations

| Resolution | Batch Size | Training Steps | GPU-Hours (A100) |
|-----------|-----------|---------------|------------------|
| 32×32 (CIFAR) | 128 | 500K | ~24 |
| 64×64 | 128 | 500K | ~72 |
| 256×256 | 64 | 800K | ~500 |
| 512×512 | 32 | 1M+ | ~2000+ |

Training diffusion models is expensive. For higher resolutions, [Latent Diffusion](../architectures/latent_diffusion.md) reduces cost by operating in a compressed latent space.

---

## 7. Monitoring and Diagnostics

### 7.1 Training Loss

The loss curve for DDPM training is characteristically noisy because each batch samples random timesteps. Smooth with a large window (1000+ steps) for meaningful trends. A well-training model shows:

- Rapid initial decrease (first 10K steps)
- Slow, steady improvement thereafter
- No sudden spikes (suggests gradient or data issues)

### 7.2 Per-Timestep Loss

Decomposing the loss by timestep reveals whether the model struggles with particular noise levels:

```python
def compute_per_timestep_loss(
    trainer: DDPMTrainer,
    dataloader: DataLoader,
    n_bins: int = 20,
) -> dict:
    """Compute average loss per timestep bin."""
    bins = torch.linspace(0, trainer.n_timesteps, n_bins + 1, dtype=torch.long)
    bin_losses = torch.zeros(n_bins)
    bin_counts = torch.zeros(n_bins)

    trainer.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x_0 = batch[0].to(trainer.device) if isinstance(batch, (list, tuple)) else batch.to(trainer.device)
            batch_size = x_0.shape[0]

            for bin_idx in range(n_bins):
                t = torch.randint(
                    bins[bin_idx], bins[bin_idx + 1], (batch_size,), device=trainer.device
                )
                x_t, noise = trainer.q_sample(x_0, t)
                pred = trainer.model(x_t, t)
                loss = F.mse_loss(pred, noise, reduction="none").flatten(1).mean(1)
                bin_losses[bin_idx] += loss.sum().item()
                bin_counts[bin_idx] += batch_size

    return {
        "bin_centers": ((bins[:-1] + bins[1:]) / 2).tolist(),
        "mean_loss": (bin_losses / bin_counts.clamp(min=1)).tolist(),
    }
```

Typical pattern: loss is highest at intermediate timesteps ($t \approx T/2$) where the model must handle partially structured inputs, and lower at extremes (near-clean or near-noise).

### 7.3 Sample Quality During Training

Generate samples periodically (every 10K–50K steps) using the EMA model and track visual quality. For quantitative evaluation:

- **FID** ([Fréchet Inception Distance](../evaluation/fid.md)): Primary metric; compute on 50K samples
- **IS** ([Inception Score](../evaluation/inception_score.md)): Complementary quality/diversity metric
- FID typically improves monotonically with training, though improvements slow dramatically after 500K+ steps

---

## 8. Common Training Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss spikes | Large gradients at high-$t$ steps | Reduce LR, ensure grad clipping at 1.0 |
| Blurry samples | Undertrained, wrong EMA | Train longer; verify using EMA weights for sampling |
| Color artifacts | Numerical issues in schedule | Use float32; check $\bar{\alpha}_T$ not too close to 0 |
| Mode dropping | Rare with DDPM; dataset issue | Check data diversity and augmentation |
| NaN loss | Overflow in noise computation | Enable AMP properly; check schedule endpoints |
| Flat loss after initial drop | LR too low or model too small | Increase LR or model capacity |

---

## 9. Finance Applications

| Application | Training Consideration | Notes |
|-------------|----------------------|-------|
| Return distribution modeling | Normalize returns to $[-1, 1]$; multivariate | See [Time Series Generation](../finance/time_series.md) |
| Scenario generation | Condition on macro variables | Use classifier-free guidance training |
| Synthetic data augmentation | Match marginal and cross-sectional statistics | Validate with distribution tests; see [Synthetic Data](../finance/synthetic.md) |
| Covariance matrix generation | Ensure positive definiteness in output | Post-process or use Cholesky parameterization |

---

## 10. Key Takeaways

1. **The training loss is remarkably simple**: MSE between predicted and actual noise, with random timestep sampling and closed-form forward diffusion. No adversarial training, no mode-seeking objectives.

2. **Three prediction parameterizations** ($\epsilon$, $x_0$, $v$) are mathematically equivalent but differ in optimization dynamics. $\epsilon$-prediction is standard; $v$-prediction is preferred for cosine schedules and latent diffusion.

3. **Loss weighting matters**: the simplified loss ($w(t) = 1$) gives best FID despite not being a tight variational bound, because it allocates gradient signal to high-noise timesteps that determine global structure.

4. **EMA is essential, not optional**: sampling quality from raw training weights is substantially worse due to compounding of weight noise over 1000 denoising steps.

5. **Classifier-free guidance** requires training-time conditioning dropout—the model must learn both conditional and unconditional generation from a single set of weights.

6. **Training is expensive but stable**: unlike GANs, DDPM training does not suffer from mode collapse, training oscillation, or generator-discriminator imbalance. The primary cost is wall-clock time.

---

## Exercises

### Exercise 1: Prediction Parameterization Comparison

Train three models on CIFAR-10 using $\epsilon$-prediction, $x_0$-prediction, and $v$-prediction with the same architecture and hyperparameters. Compare loss curves, final FID, and generated sample quality. Which converges fastest?

### Exercise 2: EMA Ablation

Train a DDPM and generate samples using (a) EMA weights ($\mu = 0.9999$), (b) EMA weights ($\mu = 0.999$), (c) raw training weights. Compute FID for each. How large is the EMA benefit?

### Exercise 3: Loss Weighting

Implement simple, SNR, and min-SNR-$\gamma$ loss weighting. Train each for 200K steps on CIFAR-10. Compare FID (sample quality) and bits-per-dimension (log-likelihood). Does the theoretical prediction hold?

### Exercise 4: Per-Timestep Loss Analysis

Use `compute_per_timestep_loss` to plot the loss profile at initialization, 50K, 200K, and 500K steps. How does the loss profile evolve? Which timestep range improves most over training?

### Exercise 5: Classifier-Free Guidance Training

Train a class-conditional DDPM on CIFAR-10 with conditioning dropout $p_{\text{uncond}} \in \{0.0, 0.1, 0.2, 0.5\}$. At sampling time, sweep guidance scale $s \in [1, 10]$. What dropout rate gives the best quality-diversity tradeoff?

---

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems (NeurIPS)*.
2. Nichol, A. Q. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *Proceedings of the 38th International Conference on Machine Learning (ICML)*.
3. Ho, J. & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*.
4. Salimans, T. & Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models. *Proceedings of the 10th International Conference on Learning Representations (ICLR)*.
5. Hang, T., Gu, S., Li, C., Bao, J., Chen, D., Hu, H., ... & Guo, B. (2023). Efficient Diffusion Training via Min-SNR Weighting Strategy. *Proceedings of ICCV*.
6. Kingma, D. P. & Gao, R. (2023). Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation. *Advances in Neural Information Processing Systems (NeurIPS)*.
