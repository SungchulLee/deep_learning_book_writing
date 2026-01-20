# Connection to Diffusion Models

## Learning Objectives

By the end of this section, you will be able to:

- Understand the deep connection between Langevin MCMC and diffusion generative models
- Explain how score-based generative models learn and use the score function
- Derive the reverse diffusion process from time-reversed SDEs
- Connect denoising score matching to the MCMC framework
- Implement simple diffusion-based sampling using Langevin dynamics

## The Big Picture: MCMC Meets Generative Modeling

### Historical Context

Langevin dynamics has been used for MCMC sampling since the 1980s. In the 2010s, researchers realized the same principles could generate **new data** from learned distributions. This led to **score-based generative models** and **diffusion models**, which now power state-of-the-art image generation (DALL-E, Stable Diffusion, Midjourney).

**The key insight**: Both MCMC and generative diffusion use the score function to guide dynamics toward a target distribution.

### The Unified Framework

| MCMC Perspective | Generative Model Perspective |
|------------------|------------------------------|
| Known target $\pi(x)$ | Unknown data distribution $p_{\text{data}}(x)$ |
| Analytical score $s(x) = \nabla \log \pi(x)$ | Learned score network $s_\theta(x)$ |
| Sample via Langevin dynamics | Generate via reverse diffusion |
| Temperature for exploration | Noise levels for multi-scale |
| Single distribution | Time-varying distributions $p_t(x)$ |

## Score-Based Generative Models

### The Core Idea

If we could compute the score $s(x) = \nabla_x \log p_{\text{data}}(x)$ for the data distribution, we could:

1. Start from random noise $x_0 \sim \mathcal{N}(0, I)$
2. Run Langevin dynamics with $s(x)$
3. Converge to samples from $p_{\text{data}}$

**Problem**: We don't know $p_{\text{data}}(x)$—we only have samples from it!

**Solution**: **Learn** the score from data using **score matching**.

### Score Matching

**Naive objective** (intractable):

$$
\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\frac{1}{2}\|s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|^2\right]
$$

This requires knowing $\nabla_x \log p_{\text{data}}(x)$, which we don't have.

**Hyvärinen's trick** (2005): The objective can be rewritten as:

$$
\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\frac{1}{2}\|s_\theta(x)\|^2 + \text{tr}(\nabla_x s_\theta(x))\right] + \text{const}
$$

This only requires evaluating $s_\theta$ and its Jacobian on data samples—no need for the true score!

### Denoising Score Matching

An even simpler approach: perturb data with noise and learn to "denoise".

**Setup**: Given data $x$, add Gaussian noise:

$$
\tilde{x} = x + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

The score of the **noised** conditional distribution $q(\tilde{x} | x) = \mathcal{N}(\tilde{x} | x, \sigma^2 I)$ is:

$$
\nabla_{\tilde{x}} \log q(\tilde{x} | x) = -\frac{\tilde{x} - x}{\sigma^2} = -\frac{\epsilon}{\sigma}
$$

**Denoising score matching objective**:

$$
\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0,I)}\left[\left\|s_\theta(\tilde{x}, \sigma) + \frac{\epsilon}{\sigma}\right\|^2\right]
$$

!!! success "Key Insight"
    Training a network to predict the noise $\epsilon$ is equivalent to learning the score! This is exactly what **DDPM** does: the "noise prediction" network is actually a score network.

### Multi-Scale Denoising

A single noise level $\sigma$ has problems:

- **Low noise**: Score is accurate but chain mixes slowly (stuck in modes)
- **High noise**: Chain explores but score is inaccurate

**Solution**: Learn scores at **multiple noise levels** $\sigma_1 > \sigma_2 > \cdots > \sigma_L$ and anneal from high to low during sampling.

This is **Annealed Langevin Dynamics** for score-based models:

```
Algorithm: Annealed Langevin Sampling
--------------------------------------
Input: Learned score s_θ(x, σ), noise levels [σ₁, ..., σₗ], steps per level T
Output: Sample x

x ~ N(0, σ₁²I)  # Start from pure noise

for l = 1 to L:
    ε = α × σₗ²  # Step size proportional to noise level
    for t = 1 to T:
        z ~ N(0, I)
        x = x + (ε/2) × s_θ(x, σₗ) + √ε × z
    
return x
```

## Diffusion Models: The Continuous View

### The Forward Diffusion Process

**Denoising Diffusion Probabilistic Models (DDPMs)** define a forward process that gradually adds noise to data:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t | \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

where $\beta_t$ is a **noise schedule** (typically $\beta_t \in [10^{-4}, 0.02]$).

After $T$ steps, $x_T \approx \mathcal{N}(0, I)$ regardless of $x_0$.

**In continuous time**, this becomes the **forward SDE**:

$$
dx_t = -\frac{1}{2}\beta(t) x_t \, dt + \sqrt{\beta(t)} \, dW_t
$$

This is an **Ornstein-Uhlenbeck process**—a drift toward zero plus diffusion.

### The Reverse Process: Time-Reversed SDE

**Anderson's Theorem (1982)**: Any SDE can be time-reversed. For the forward SDE:

$$
dx_t = f(x_t, t) \, dt + g(t) \, dW_t
$$

The reverse-time SDE is:

$$
dx_t = \left[f(x_t, t) - g(t)^2 \nabla_x \log p_t(x_t)\right] dt + g(t) \, d\bar{W}_t
$$

where $\bar{W}_t$ is a backward Brownian motion and $p_t(x)$ is the marginal distribution at time $t$.

For the VP-SDE (variance preserving) forward process:

$$
\text{Reverse: } dx_t = \left[-\frac{1}{2}\beta(t) x_t - \beta(t) \nabla_x \log p_t(x_t)\right] dt + \sqrt{\beta(t)} \, d\bar{W}_t
$$

**The score** $\nabla_x \log p_t(x_t)$ is the key quantity—it tells us how to reverse the diffusion!

### Connection to Langevin Dynamics

The reverse diffusion can be seen as **time-varying Langevin dynamics**:

$$
dx_t = \underbrace{-\frac{1}{2}\beta(t) x_t}_{\text{drift toward origin}} \underbrace{- \beta(t) s_t(x_t)}_{\text{score-guided drift}} \, dt + \sqrt{\beta(t)} \, d\bar{W}_t
$$

where $s_t(x) = \nabla_x \log p_t(x)$ is the time-dependent score.

This is Langevin dynamics with:

- **Time-varying temperature**: $\sqrt{\beta(t)}$ changes over time
- **Time-varying target**: $p_t(x)$ evolves from noise to data
- **Additional drift**: $-\frac{1}{2}\beta(t) x_t$ keeps the scale controlled

## The DDPM ↔ Score Matching Equivalence

### DDPM Loss Function

DDPM trains a network $\epsilon_\theta(x_t, t)$ to predict the noise $\epsilon$ added to create $x_t$:

$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon_\theta(x_t, t) - \epsilon\|^2\right]
$$

where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ and $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$.

### Score Parameterization

The score at time $t$ is related to the noise by:

$$
\nabla_{x_t} \log p_t(x_t | x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

Therefore:

$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}
$$

!!! info "The Deep Connection"
    **Noise prediction** (DDPM) and **score estimation** (score-based models) are equivalent parameterizations:
    
    $$
    \epsilon_\theta(x_t, t) = -\sqrt{1-\bar{\alpha}_t} \cdot s_\theta(x_t, t)
    $$
    
    DDPM training is denoising score matching!

## PyTorch Implementation

### Simple Score-Based Generation

```python
import torch
import torch.nn as nn
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class SimpleScoreNet(nn.Module):
    """
    Simple MLP for score estimation.
    """
    
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for noise level
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, sigma):
        """
        Estimate score at noise level sigma.
        
        Args:
            x: Noisy samples (batch, dim)
            sigma: Noise level (scalar or (batch, 1))
            
        Returns:
            score: Estimated score (batch, dim)
        """
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0).expand(x.shape[0], 1)
        elif sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        
        # Normalize sigma for network input
        sigma_normalized = torch.log(sigma + 1e-5)
        
        x_input = torch.cat([x, sigma_normalized], dim=-1)
        return self.net(x_input)


def train_score_network(data, n_epochs=1000, lr=1e-3, sigma_min=0.01, sigma_max=1.0):
    """
    Train a score network using denoising score matching.
    
    Args:
        data: Training data (n_samples, dim)
        n_epochs: Number of training epochs
        lr: Learning rate
        sigma_min, sigma_max: Range of noise levels
        
    Returns:
        model: Trained score network
    """
    dim = data.shape[1]
    model = SimpleScoreNet(dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        # Sample batch
        idx = torch.randint(0, len(data), (128,))
        x = data[idx]
        
        # Sample noise level (log-uniform)
        log_sigma = torch.rand(x.shape[0], 1) * (np.log(sigma_max) - np.log(sigma_min)) + np.log(sigma_min)
        sigma = torch.exp(log_sigma)
        
        # Add noise
        epsilon = torch.randn_like(x)
        x_noisy = x + sigma * epsilon
        
        # Target: -epsilon / sigma (the true score of the conditional)
        target = -epsilon / sigma
        
        # Predict score
        score_pred = model(x_noisy, sigma)
        
        # Loss (weighted by sigma^2 for numerical stability)
        loss = ((sigma ** 2) * (score_pred - target) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return model, losses


def annealed_langevin_sampling(model, dim, n_samples, sigmas, steps_per_sigma=100):
    """
    Generate samples using annealed Langevin dynamics.
    
    Args:
        model: Trained score network
        dim: Dimension of samples
        n_samples: Number of samples to generate
        sigmas: List of noise levels (decreasing)
        steps_per_sigma: Langevin steps per noise level
        
    Returns:
        samples: Generated samples
    """
    # Start from noise
    x = torch.randn(n_samples, dim) * sigmas[0]
    
    for sigma in tqdm(sigmas, desc="Sampling"):
        # Step size (Langevin discretization)
        epsilon = 0.5 * (sigma ** 2)
        
        for _ in range(steps_per_sigma):
            # Get score
            score = model(x, torch.tensor(sigma))
            
            # Langevin update
            noise = torch.randn_like(x)
            x = x + epsilon * score + np.sqrt(2 * epsilon) * noise
    
    return x


def demo_score_based_generation():
    """
    Demonstrate score-based generative modeling on 2D data.
    """
    torch.manual_seed(42)
    
    # Create 2D mixture data
    n_data = 5000
    
    # Three Gaussian clusters
    means = torch.tensor([
        [-2.0, -2.0],
        [2.0, -2.0],
        [0.0, 2.5]
    ])
    
    data = []
    for i in range(3):
        cluster = means[i] + 0.5 * torch.randn(n_data // 3, 2)
        data.append(cluster)
    
    data = torch.cat(data, dim=0)
    
    # Train score network
    print("Training score network...")
    model, losses = train_score_network(data, n_epochs=2000)
    
    # Generate samples
    print("\nGenerating samples...")
    sigmas = np.geomspace(1.0, 0.01, 50)  # Log-spaced noise levels
    generated = annealed_langevin_sampling(model, dim=2, n_samples=1000, 
                                           sigmas=sigmas, steps_per_sigma=50)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training loss
    ax = axes[0]
    ax.semilogy(losses, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Original data
    ax = axes[1]
    ax.scatter(data[:, 0].numpy(), data[:, 1].numpy(), alpha=0.3, s=5, c='blue')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Training Data', fontsize=13, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Generated samples
    ax = axes[2]
    ax.scatter(generated[:, 0].numpy(), generated[:, 1].numpy(), 
               alpha=0.3, s=5, c='red')
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Generated Samples', fontsize=13, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diffusion_connection_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: diffusion_connection_demo.png")


if __name__ == "__main__":
    demo_score_based_generation()
```

### Visualizing the Diffusion Process

```python
def visualize_diffusion_process():
    """
    Visualize forward and reverse diffusion.
    """
    torch.manual_seed(42)
    
    # Simple 2D data: ring
    n_points = 500
    theta = 2 * np.pi * torch.rand(n_points)
    r = 2.0 + 0.2 * torch.randn(n_points)
    data = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
    
    # Forward diffusion: add noise progressively
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    
    # Noise schedule
    T = 100
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # Forward process
    time_points = [0, 10, 30, 60, 99]
    
    for idx, t in enumerate(time_points):
        ax = axes[0, idx]
        
        if t == 0:
            x_t = data
        else:
            # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
            alpha_bar_t = alpha_bars[t]
            epsilon = torch.randn_like(data)
            x_t = torch.sqrt(alpha_bar_t) * data + torch.sqrt(1 - alpha_bar_t) * epsilon
        
        ax.scatter(x_t[:, 0].numpy(), x_t[:, 1].numpy(), alpha=0.5, s=10, c='blue')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_title(f't = {t}\n' + (r'$\bar{\alpha}_t$' + f' = {alpha_bars[t]:.3f}' if t > 0 else 'Original data'),
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.set_ylabel('Forward\nDiffusion', fontsize=12, fontweight='bold')
    
    # Reverse process (simulated using known score)
    # For demonstration, we use analytical score of Gaussian mixture
    
    # Start from noise
    x_T = torch.randn(n_points, 2)
    x_reverse = x_T.clone()
    
    # Store intermediate states
    reverse_states = {99: x_T.clone()}
    
    for t in reversed(range(T)):
        # Simple reverse step (DDPM sampling)
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        
        # Approximate score (for demo, just use gradient toward data mean)
        # In practice, this would be the learned score network
        score_approx = -x_reverse / (1 - alpha_bar_t + 0.1)
        
        # Reverse step
        if t > 0:
            noise = torch.randn_like(x_reverse)
            x_reverse = (1 / torch.sqrt(alpha_t)) * (
                x_reverse + betas[t] * score_approx
            ) + torch.sqrt(betas[t]) * noise
        else:
            x_reverse = (1 / torch.sqrt(alpha_t)) * (
                x_reverse + betas[t] * score_approx
            )
        
        if t in time_points:
            reverse_states[t] = x_reverse.clone()
    
    reverse_states[0] = x_reverse.clone()
    
    # Plot reverse process
    for idx, t in enumerate(reversed(time_points)):
        ax = axes[1, 4 - idx]
        x_t = reverse_states.get(t, x_reverse)
        
        ax.scatter(x_t[:, 0].numpy(), x_t[:, 1].numpy(), alpha=0.5, s=10, c='red')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_title(f't = {t}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if idx == 4:
            ax.set_ylabel('Reverse\nDiffusion', fontsize=12, fontweight='bold')
    
    plt.suptitle('Forward and Reverse Diffusion Process\n' + 
                 '(Forward: Data → Noise, Reverse: Noise → Data)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('diffusion_process_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated: diffusion_process_visualization.png")


if __name__ == "__main__":
    visualize_diffusion_process()
```

## The Probability Flow ODE

### From SDE to ODE

Song et al. (2021) showed that every diffusion SDE has an associated **deterministic ODE** with the same marginal distributions:

$$
dx_t = \left[f(x_t, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x_t)\right] dt
$$

For standard diffusion:

$$
dx_t = \left[-\frac{1}{2}\beta(t) x_t - \frac{1}{2}\beta(t) s_t(x_t)\right] dt
$$

### Advantages of the ODE Formulation

- **Deterministic**: Same initial condition → same trajectory
- **Fast solvers**: Can use high-order ODE solvers (Runge-Kutta, etc.)
- **Fewer steps**: Often 10-50 steps suffice (vs. 1000 for SDE)
- **Interpolation**: Meaningful trajectories in latent space

**DDIM** (Denoising Diffusion Implicit Models) uses this ODE for efficient sampling.

### The Trade-off

| SDE Sampling | ODE Sampling |
|--------------|--------------|
| Stochastic | Deterministic |
| More exploration | Less exploration |
| Robust to errors | Sensitive to score errors |
| Typically slower | Can be faster |

## Practical Implications

### Why Diffusion Models Work

The success of diffusion models can be understood through the MCMC lens:

1. **Multi-scale scores**: Avoids mode collapse via annealing
2. **Denoising objective**: Simple, stable training
3. **Invertible dynamics**: Exact likelihood computation possible
4. **Flexible architecture**: Score network can be any neural net

### From MCMC to Generation

| MCMC Application | Generative Application |
|------------------|------------------------|
| Bayesian posterior sampling | Unconditional generation |
| Simulated annealing | Noise level annealing |
| Temperature scheduling | Noise scheduling |
| Preconditioning | Adaptive diffusion coefficients |
| Multiple chains | Parallel sampling |

### What Diffusion Models Teach Us About MCMC

1. **Learned scores work**: Neural networks can approximate complex score functions
2. **Multi-scale annealing is powerful**: Start coarse, refine gradually
3. **ODE alternatives**: Deterministic flows can replace stochastic dynamics
4. **Score is the key quantity**: Everything flows from $\nabla \log p$

## Summary

The connection between Langevin MCMC and diffusion models reveals a unified framework:

| Concept | MCMC View | Diffusion Model View |
|---------|-----------|---------------------|
| Score function | Known analytically | Learned from data |
| Sampling | Langevin dynamics | Reverse diffusion |
| Temperature | Controls exploration | Noise level |
| Annealing | Simulated annealing | Noise schedule |
| Exactness | MH correction (MALA) | Variance preserving SDE |

**The deep insight**: Score-based diffusion models are **annealed Langevin MCMC with learned energy functions**. The breakthrough was realizing that scores can be learned via denoising, eliminating the need for explicit density models.

Modern image generators (DALL-E, Stable Diffusion, Midjourney) are running annealed Langevin dynamics with scores learned from billions of images. The theory developed for Bayesian inference now powers the generative AI revolution.

## Exercises

### Exercise 1: Score Network Training

Train a score network on a 2D "swiss roll" dataset. Visualize the learned score field at different noise levels.

### Exercise 2: DDPM Implementation

Implement a simple DDPM for 1D data. Show the equivalence between noise prediction and score estimation.

### Exercise 3: Probability Flow ODE

Implement sampling using the probability flow ODE with a Runge-Kutta solver. Compare speed and quality with SDE sampling.

### Exercise 4: Conditional Generation

Extend the score-based model to conditional generation: given class labels, generate samples from each class.

## References

1. Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS*.

2. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.

3. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *ICLR*.

4. Anderson, B. D. (1982). Reverse-time diffusion equation models. *Stochastic Processes and their Applications*, 12(3), 313-326.

5. Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. *JMLR*.

6. Vincent, P. (2011). A connection between score matching and denoising autoencoders. *Neural Computation*.
