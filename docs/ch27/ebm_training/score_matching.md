# Score Matching for EBMs

## Learning Objectives

After completing this section, you will be able to:

1. Derive the score matching objective and understand why it avoids the partition function
2. Implement explicit score matching, denoising score matching, and sliced score matching
3. Train neural energy functions using score matching variants
4. Compare score matching against contrastive divergence for EBM training

## The Key Insight

Recall from Section 26.1 that the score function of a Boltzmann distribution depends only on the energy gradient:

$$s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$$

(setting $T=1$ for simplicity). The partition function $Z(\theta)$, which makes likelihood-based training intractable, disappears entirely when we work with the score. Score matching exploits this by training the model to match the score of the data distribution rather than the density itself.

## Explicit (Fisher) Score Matching

### Objective

The score matching objective, introduced by Hyvärinen (2005), minimizes the expected squared distance between model and data score functions:

$$J_{\text{SM}}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}} \left[\|\nabla_x \log p_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|^2\right]$$

Since $\nabla_x \log p_{\text{data}}$ is unknown, Hyvärinen showed this is equivalent (up to a constant independent of $\theta$) to:

$$J_{\text{SM}}(\theta) = \mathbb{E}_{p_{\text{data}}} \left[\frac{1}{2}\|\nabla_x \log p_\theta(x)\|^2 + \text{tr}(\nabla_x^2 \log p_\theta(x))\right]$$

This form requires only the model score and its Jacobian—no data score, no partition function.

### Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class EnergyNetwork(nn.Module):
    """Neural energy function for score matching."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def compute_score(energy_net, x):
    """Compute score s(x) = -∇_x E(x)."""
    x = x.requires_grad_(True)
    energy = energy_net(x).sum()
    grad = torch.autograd.grad(energy, x, create_graph=True)[0]
    return -grad

def explicit_score_matching_loss(energy_net, x):
    """
    Explicit score matching loss (Hyvärinen, 2005).
    
    L = (1/2)||s(x)||² + tr(∇s(x))
    
    The trace term requires computing diagonal elements of the 
    Jacobian of the score, which is expensive in high dimensions.
    """
    x = x.requires_grad_(True)
    energy = energy_net(x).sum()
    
    # Score = -∇E
    score = torch.autograd.grad(energy, x, create_graph=True)[0]
    score = -score  # s(x) = -∇E(x)
    
    # Term 1: ||s(x)||²
    score_norm = (score ** 2).sum(dim=1)
    
    # Term 2: tr(∇s(x)) = Σ_i ∂s_i/∂x_i
    trace = torch.zeros(x.shape[0], device=x.device)
    for i in range(x.shape[1]):
        grad_si = torch.autograd.grad(
            score[:, i].sum(), x, create_graph=True
        )[0]
        trace += grad_si[:, i]
    
    loss = 0.5 * score_norm + trace
    return loss.mean()
```

The explicit score matching loss involves computing the trace of the score Jacobian, which requires $d$ backward passes for $d$-dimensional data. This $O(d)$ cost motivates the scalable alternatives below.

## Denoising Score Matching

### Objective

Vincent (2011) showed that the score matching objective can be reformulated by adding noise to the data. Given a noise distribution $q_\sigma(\tilde{x}|x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$, the denoising score matching (DSM) objective is:

$$J_{\text{DSM}}(\theta) = \mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{q_\sigma(\tilde{x}|x)} \left[\left\|s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x)\right\|^2\right]$$

Since $\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) = -({\tilde{x} - x})/{\sigma^2}$, this simplifies to:

$$J_{\text{DSM}}(\theta) = \mathbb{E} \left[\left\|s_\theta(\tilde{x}) + \frac{\tilde{x} - x}{\sigma^2}\right\|^2\right]$$

This is equivalent to training the score function to predict the direction back to the clean data from the noisy version—a denoising task.

### Implementation

```python
def denoising_score_matching_loss(energy_net, x, sigma=0.1):
    """
    Denoising score matching (Vincent, 2011).
    
    1. Add Gaussian noise to data: x̃ = x + σε
    2. Train score to point from x̃ back to x
    
    No Jacobian computation needed — O(1) backward passes.
    """
    # Add noise
    noise = torch.randn_like(x) * sigma
    x_noisy = x + noise
    
    # Compute model score at noisy point
    x_noisy = x_noisy.requires_grad_(True)
    energy = energy_net(x_noisy).sum()
    score = -torch.autograd.grad(energy, x_noisy, create_graph=True)[0]
    
    # Target: direction from noisy to clean
    target = -(x_noisy.detach() - x) / (sigma ** 2)
    
    # MSE loss
    loss = ((score - target) ** 2).sum(dim=1).mean()
    return loss
```

### Advantages of DSM

**Computational efficiency**: No Jacobian trace computation, making it $O(1)$ per sample regardless of dimensionality.

**Robustness**: The noise injection acts as regularization, preventing the score from overfitting to individual data points.

**Connection to diffusion models**: DSM at multiple noise levels is exactly the training objective of score-based diffusion models (Song & Ermon, 2019). This connection places EBMs and diffusion models within a unified framework.

## Sliced Score Matching

### Objective

Song et al. (2020) proposed sliced score matching (SSM), which estimates the trace of the Jacobian through random projections:

$$J_{\text{SSM}}(\theta) = \mathbb{E}_{p_{\text{data}}} \mathbb{E}_{p_v} \left[\frac{1}{2}(v^T s_\theta(x))^2 + v^T \nabla_x (v^T s_\theta(x))\right]$$

where $v$ is a random direction vector (typically $v \sim \mathcal{N}(0, I)$). This requires only a single directional derivative rather than the full Jacobian trace.

### Implementation

```python
def sliced_score_matching_loss(energy_net, x, n_projections=1):
    """
    Sliced score matching (Song et al., 2020).
    
    Uses random projections to estimate the Jacobian trace.
    Cost: O(n_projections) backward passes, independent of dimension.
    """
    x = x.requires_grad_(True)
    energy = energy_net(x).sum()
    score = -torch.autograd.grad(energy, x, create_graph=True)[0]
    
    loss = torch.zeros(x.shape[0], device=x.device)
    
    for _ in range(n_projections):
        # Random projection direction
        v = torch.randn_like(x)
        
        # Projected score
        sv = (score * v).sum(dim=1)
        
        # Term 1: (v^T s)^2
        loss += 0.5 * sv ** 2
        
        # Term 2: v^T ∇(v^T s)  (directional derivative)
        grad_sv = torch.autograd.grad(
            sv.sum(), x, create_graph=True
        )[0]
        loss += (grad_sv * v).sum(dim=1)
    
    return loss.mean() / n_projections
```

## Complete Training Example

```python
def train_ebm_with_score_matching():
    """
    Train a neural EBM on 2D data using different score matching variants.
    """
    # Generate 2D data (three clusters)
    n_samples = 2000
    data = torch.cat([
        torch.randn(n_samples // 3, 2) * 0.3 + torch.tensor([-2.0, 0.0]),
        torch.randn(n_samples // 3, 2) * 0.3 + torch.tensor([2.0, 0.0]),
        torch.randn(n_samples // 3, 2) * 0.3 + torch.tensor([0.0, 2.0]),
    ])
    
    methods = {
        'DSM (σ=0.1)': lambda net, x: denoising_score_matching_loss(net, x, 0.1),
        'SSM': lambda net, x: sliced_score_matching_loss(net, x, n_projections=2),
    }
    
    results = {}
    
    for name, loss_fn in methods.items():
        print(f"\nTraining with {name}...")
        energy_net = EnergyNetwork(input_dim=2, hidden_dim=64)
        optimizer = torch.optim.Adam(energy_net.parameters(), lr=1e-3)
        
        losses = []
        for epoch in range(200):
            perm = torch.randperm(len(data))
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, len(data), 128):
                batch = data[perm[i:i+128]]
                
                loss = loss_fn(energy_net, batch)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(energy_net.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            losses.append(epoch_loss / n_batches)
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: loss = {losses[-1]:.4f}")
        
        results[name] = {'net': energy_net, 'losses': losses}
    
    # Visualize learned energy landscapes
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(5 * (len(methods) + 1), 4))
    
    # Data
    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), 
                    s=1, alpha=0.3, c='blue')
    axes[0].set_title('Training Data')
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-2, 4)
    
    # Learned energy landscapes
    grid_x = torch.linspace(-4, 4, 100)
    grid_y = torch.linspace(-2, 4, 100)
    X, Y = torch.meshgrid(grid_x, grid_y, indexing='ij')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    for idx, (name, res) in enumerate(results.items()):
        with torch.no_grad():
            energies = res['net'](grid).reshape(100, 100)
        
        axes[idx + 1].contourf(X.numpy(), Y.numpy(), energies.numpy(),
                               levels=30, cmap='viridis')
        axes[idx + 1].scatter(data[:, 0].numpy(), data[:, 1].numpy(),
                             s=1, alpha=0.1, c='white')
        axes[idx + 1].set_title(f'{name}\n(low energy = high prob)')
        axes[idx + 1].set_xlim(-4, 4)
        axes[idx + 1].set_ylim(-2, 4)
    
    plt.tight_layout()
    plt.show()

train_ebm_with_score_matching()
```

## Method Comparison

| Method | Cost per Sample | Needs Jacobian | Noise Needed | Scalability |
|--------|----------------|----------------|--------------|-------------|
| Explicit SM | $O(d)$ backward passes | Full trace | No | Poor for high-$d$ |
| Denoising SM | $O(1)$ backward passes | No | Yes ($\sigma$) | Excellent |
| Sliced SM | $O(k)$ backward passes | Directional only | No | Good |

**Recommendation**: For most practical applications, denoising score matching is the method of choice due to its simplicity, efficiency, and connection to diffusion models. Sliced score matching is useful when the noise level $\sigma$ in DSM is difficult to tune.

## Connection to Other Training Methods

### Score Matching vs. Contrastive Divergence

Both avoid computing the partition function, but through different mechanisms:

- **CD**: Approximates the model expectation in the MLE gradient via short-run MCMC
- **Score matching**: Bypasses the partition function entirely by matching score functions

Score matching avoids MCMC altogether, eliminating concerns about mixing and chain management. However, CD naturally handles discrete data (binary visible units in RBMs), whereas score matching requires continuous inputs.

### Score Matching and Diffusion Models

Denoising score matching at multiple noise levels is precisely the training objective of score-based diffusion models. Training an EBM with DSM at noise level $\sigma$ is equivalent to training a single step of a diffusion model's denoising process. This deep connection means that insights from EBM training (energy landscapes, mode coverage) transfer directly to understanding diffusion model behavior.

## Key Takeaways

!!! success "Core Concepts"
    1. Score matching trains EBMs by matching $\nabla_x \log p_\theta(x)$ to $\nabla_x \log p_{\text{data}}(x)$, completely avoiding the partition function
    2. Denoising score matching replaces the intractable Jacobian trace with a simple denoising objective
    3. Sliced score matching uses random projections for a scalable unbiased estimator
    4. Score matching requires continuous data; for discrete data, CD-based methods are necessary
    5. The connection between DSM and diffusion models unifies two major generative modeling paradigms

!!! warning "Practical Notes"
    - The noise level $\sigma$ in DSM is a critical hyperparameter: too small gives noisy gradients, too large blurs the target distribution
    - Gradient clipping is important for stable training with score matching objectives
    - Score matching does not provide a direct estimate of the log-likelihood, making model evaluation harder

## Exercises

1. **Explicit vs. denoising**: For a 2D energy function, compare the gradients of explicit and denoising score matching. How does the noise level $\sigma$ affect the quality of the DSM approximation?

2. **Multi-scale DSM**: Implement DSM with multiple noise levels $\sigma \in \{0.01, 0.1, 0.5, 1.0\}$ and a weighted combination of losses. This is the training objective of NCSN (Song & Ermon, 2019).

3. **Financial application**: Train a score-matching EBM on 2D financial return data (e.g., two correlated assets). Visualize the learned energy landscape and compare it to a fitted Gaussian copula.

## References

- Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. *JMLR*.
- Vincent, P. (2011). A connection between score matching and denoising autoencoders. *Neural Computation*.
- Song, Y., et al. (2020). Sliced Score Matching: A Scalable Approach to Density and Score Estimation. *UAI*.
- Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *NeurIPS*.
