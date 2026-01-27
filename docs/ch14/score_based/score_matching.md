# Score Matching

**Score matching** is a technique for estimating probability distributions by learning their score functions without requiring the intractable normalization constant. It is the theoretical foundation underlying modern diffusion models.

## Learning Objectives

By the end of this section, you will be able to:

1. Explain why direct score estimation is challenging and how score matching solves this
2. Derive the explicit (implicit) score matching objective using Stein's identity
3. Understand the computational challenges and solutions for high-dimensional data
4. Implement ESM, DSM, and SSM in PyTorch
5. Compare score matching variants and their trade-offs
6. Connect score matching to related methods (MLE, NCE, contrastive divergence)

## Prerequisites

- Score function definition (previous section)
- Basic optimization theory
- PyTorch autograd mechanics

---

## 1. The Score Matching Problem

### 1.1 Problem Statement

Given samples $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$ from an unknown distribution $p_{\text{data}}(\mathbf{x})$, we want to train a neural network $\mathbf{s}_\theta(\mathbf{x})$ to approximate the true score function:

$$
\mathbf{s}_\theta(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})
$$

### 1.2 The Naive Approach (Doesn't Work)

A natural loss function would be the **Fisher divergence** (expected squared error):

$$
\mathcal{L}_{\text{naive}}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}}\left[\|\mathbf{s}_\theta(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})\|^2\right]
$$

**Problem**: We don't know $p_{\text{data}}(\mathbf{x})$! If we knew it, we wouldn't need to learn it.

!!! question "Key Question"
    How can we train a score model when we can't compute the target score?

---

## 2. Explicit Score Matching (ESM)

### 2.1 The Fundamental Identity

Hyvärinen (2005) showed that the naive loss can be rewritten **without requiring the true score**:

$$
\mathcal{L}_{\text{ESM}}(\theta) = \mathbb{E}_{p_{\text{data}}}\left[\frac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2 + \text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}))\right] + \text{const}
$$

where $\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x})) = \sum_i \frac{\partial s_{\theta,i}(\mathbf{x})}{\partial x_i}$ is the trace of the Jacobian.

!!! info "Also Called Implicit Score Matching (ISM)"
    Some literature calls this "implicit" score matching because the true score is implicitly present through the Jacobian trace, rather than appearing explicitly in the loss.

### 2.2 Derivation

Starting from the Fisher divergence:

$$
\mathcal{L}_{\text{naive}}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}}\left[\|\mathbf{s}_\theta(\mathbf{x}) - \mathbf{s}(\mathbf{x})\|^2\right]
$$

Expand the squared norm:

$$
= \frac{1}{2} \mathbb{E}\left[\|\mathbf{s}_\theta(\mathbf{x})\|^2 - 2\mathbf{s}_\theta(\mathbf{x})^\top \mathbf{s}(\mathbf{x}) + \|\mathbf{s}(\mathbf{x})\|^2\right]
$$

The last term $\|\mathbf{s}(\mathbf{x})\|^2$ is constant with respect to $\theta$. The key is transforming the cross term:

$$
\mathbb{E}_{p_{\text{data}}}\left[\mathbf{s}_\theta(\mathbf{x})^\top \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})\right]
$$

Using **integration by parts** (Stein's identity):

$$
= -\mathbb{E}_{p_{\text{data}}}\left[\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}))\right]
$$

This remarkable identity replaces the unknown true score with a computable Jacobian trace!

### 2.3 Stein's Identity

The derivation relies on **Stein's identity**: for smooth functions $f$ and distributions $p$ that decay at infinity,

$$
\mathbb{E}_p\left[f(\mathbf{x})^\top \nabla_{\mathbf{x}} \log p(\mathbf{x})\right] = -\mathbb{E}_p\left[\nabla_{\mathbf{x}} \cdot f(\mathbf{x})\right]
$$

where $\nabla_{\mathbf{x}} \cdot f = \sum_i \frac{\partial f_i}{\partial x_i}$ is the divergence.

**Proof sketch**: Integration by parts, assuming boundary terms vanish:

$$
\int p(\mathbf{x}) f(\mathbf{x})^\top \frac{\nabla p(\mathbf{x})}{p(\mathbf{x})} d\mathbf{x} = \int f(\mathbf{x})^\top \nabla p(\mathbf{x}) d\mathbf{x} = -\int p(\mathbf{x}) \nabla \cdot f(\mathbf{x}) d\mathbf{x}
$$

### 2.4 Final ESM Objective

$$
\boxed{\mathcal{L}_{\text{ESM}}(\theta) = \mathbb{E}_{p_{\text{data}}}\left[\frac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2 + \sum_{i=1}^D \frac{\partial s_{\theta,i}(\mathbf{x})}{\partial x_i}\right]}
$$

### 2.5 Computational Challenge

The trace term requires computing **diagonal elements of the Jacobian**:

$$
\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x})) = \sum_{i=1}^D \frac{\partial s_{\theta,i}(\mathbf{x})}{\partial x_i}
$$

This requires $D$ backward passes through the network—**prohibitively expensive** for high-dimensional data!

| Data Type | Dimension $D$ | Backward Passes/Sample |
|-----------|---------------|------------------------|
| 2D toy | 2 | 2 |
| MNIST | 784 | 784 |
| CIFAR-10 | 3,072 | 3,072 |
| ImageNet | 150,528 | 150,528 |

!!! warning "Scalability Issue"
    ESM is only practical for low-dimensional problems. For images, we need alternatives.

---

## 3. Denoising Score Matching (DSM)

### 3.1 The Key Idea

Instead of learning the score of $p_{\text{data}}$, learn the score of a **noise-perturbed distribution**:

$$
\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
$$

The perturbed distribution is:

$$
q_\sigma(\tilde{\mathbf{x}}) = \int p_{\text{data}}(\mathbf{x}) \, q(\tilde{\mathbf{x}}|\mathbf{x}) \, d\mathbf{x}
$$

where $q(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}|\mathbf{x}, \sigma^2 \mathbf{I})$ is the noise kernel.

### 3.2 The DSM Objective

$$
\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}} \mathbb{E}_{q(\tilde{\mathbf{x}}|\mathbf{x})}\left[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}}|\mathbf{x})\|^2\right]
$$

### 3.3 The Target Score is Known!

For Gaussian noise, the conditional score has a simple closed form:

$$
\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}}|\mathbf{x}) = \nabla_{\tilde{\mathbf{x}}} \left[-\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|^2}{2\sigma^2}\right] = -\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2} = -\frac{\boldsymbol{\epsilon}}{\sigma^2}
$$

This is simply the **negative noise divided by variance**!

### 3.4 Simplified DSM Loss

$$
\boxed{\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2} \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}}\left[\left\|\mathbf{s}_\theta(\mathbf{x} + \sigma\boldsymbol{\epsilon}) + \frac{\boldsymbol{\epsilon}}{\sigma}\right\|^2\right]}
$$

where $\mathbf{x} \sim p_{\text{data}}$ and $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

!!! success "Key Advantages of DSM"
    1. **No Jacobian computation**: Just a simple MSE loss
    2. **Parallelizable**: Standard neural network training
    3. **Scalable**: Works for any dimension
    4. **Implicit regularization**: Noise acts as data augmentation

### 3.5 Connection to True Score

Vincent (2011) proved that as $\sigma \to 0$:

$$
\mathcal{L}_{\text{DSM}}(\theta) \to \mathcal{L}_{\text{ESM}}(\theta) + \text{const}
$$

In practice, small but non-zero $\sigma$ works well and provides regularization.

---

## 4. Sliced Score Matching (SSM)

### 4.1 Motivation

SSM provides another way to avoid the expensive Jacobian trace by using **random projections** instead of computing all $D$ diagonal elements.

### 4.2 The SSM Objective

$$
\mathcal{L}_{\text{SSM}}(\theta) = \mathbb{E}_{p_{\text{data}}} \mathbb{E}_{\mathbf{v}}\left[\mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}) \mathbf{v} + \frac{1}{2}(\mathbf{v}^\top \mathbf{s}_\theta(\mathbf{x}))^2\right]
$$

where $\mathbf{v}$ is drawn from a distribution satisfying $\mathbb{E}[\mathbf{v}\mathbf{v}^\top] = \mathbf{I}$:
- Gaussian: $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- Rademacher: $v_i \in \{-1, +1\}$ with equal probability
- Uniform on sphere

### 4.3 Why It Works: Hutchinson's Trace Estimator

For any matrix $\mathbf{A}$ and random vector $\mathbf{v}$ with $\mathbb{E}[\mathbf{v}\mathbf{v}^\top] = \mathbf{I}$:

$$
\mathbb{E}_{\mathbf{v}}[\mathbf{v}^\top \mathbf{A} \mathbf{v}] = \text{tr}(\mathbf{A})
$$

So SSM estimates the Jacobian trace using random projections, requiring only **one backward pass per random vector** (typically 1-4 vectors suffice).

### 4.4 Comparison of Methods

| Method | Jacobian | Backward Passes | Bias | Best For |
|--------|----------|-----------------|------|----------|
| **ESM** | Full diagonal | $D$ per sample | None | Low-dim only |
| **DSM** | None | 1 per sample | $O(\sigma^2)$ | General use ✅ |
| **SSM** | Random projection | $M$ per sample | None | When noise is problematic |

---

## 5. Limitations of Vanilla Score Matching

### 5.1 The Low-Density Problem

In regions where $p_{\text{data}}(\mathbf{x}) \approx 0$:
- Few or no training samples exist
- Score estimates are unreliable
- Langevin sampling can get stuck or diverge

### 5.2 The Multimodality Problem

For distributions with well-separated modes:
- Gradients between modes may not point toward either mode
- Score field can be nearly zero in "no-man's land"
- Sampling struggles to traverse between modes

### 5.3 The Manifold Problem

Real data often lies on low-dimensional manifolds:
- $p(\mathbf{x}) = 0$ off the manifold
- Score is **undefined** in ambient space
- Need full support for score matching to work

!!! tip "Solution"
    These limitations motivate **noise perturbation** (DSM) and **multi-scale noise** (NCSN), which smooth the distribution and provide full support.

---

## 6. Connection to Other Methods

### 6.1 Maximum Likelihood Estimation

Score matching can be viewed as an alternative to MLE that avoids computing normalizing constants:

| Aspect | MLE | Score Matching |
|--------|-----|----------------|
| Objective | $\max \mathbb{E}[\log p_\theta(\mathbf{x})]$ | $\min \mathbb{E}[\|\mathbf{s}_\theta - \mathbf{s}\|^2]$ |
| Requires $Z$ | Yes | No |
| For exponential families | Equivalent | Equivalent |

### 6.2 Contrastive Divergence

Like contrastive divergence, score matching avoids partition functions:

| Aspect | Contrastive Divergence | Score Matching |
|--------|------------------------|----------------|
| Approach | MCMC samples | Explicit gradients |
| Requires sampling | Yes | No |
| Gradient estimation | Approximate | Exact (for ESM) |

### 6.3 Noise Contrastive Estimation (NCE)

| Aspect | NCE | Score Matching |
|--------|-----|----------------|
| Learning target | Density ratio | Score function |
| Requires noise distribution | Yes | No (except DSM) |
| Output | Unnormalized density | Score vectors |

---

## 7. PyTorch Implementation

### 7.1 Explicit Score Matching

```python
import torch
import torch.nn as nn

def esm_loss(score_model: nn.Module, samples: torch.Tensor) -> torch.Tensor:
    """
    Explicit Score Matching loss.
    
    L_ESM = E[||s_θ(x)||²/2 + tr(∇s_θ(x))]
    
    WARNING: Requires D backward passes - only for low-dimensional data!
    
    Args:
        score_model: Neural network outputting score
        samples: Data samples, shape (N, D)
    
    Returns:
        ESM loss value
    """
    samples = samples.clone().detach().requires_grad_(True)
    N, D = samples.shape
    
    # Compute model scores
    scores = score_model(samples)  # (N, D)
    
    # Term 1: ||s_θ(x)||²/2
    norm_term = 0.5 * torch.sum(scores ** 2, dim=1)  # (N,)
    
    # Term 2: tr(∇s_θ(x)) - requires D backward passes!
    trace_term = torch.zeros(N, device=samples.device)
    
    for i in range(D):
        # Gradient of i-th score output w.r.t. i-th input
        grad_i = torch.autograd.grad(
            outputs=scores[:, i].sum(),
            inputs=samples,
            create_graph=True,
            retain_graph=True
        )[0]
        trace_term += grad_i[:, i]  # Diagonal element
    
    loss = torch.mean(norm_term + trace_term)
    return loss
```

### 7.2 Denoising Score Matching

```python
def dsm_loss(
    score_model: nn.Module,
    samples: torch.Tensor,
    noise_std: float = 0.1
) -> torch.Tensor:
    """
    Denoising Score Matching loss.
    
    L_DSM = E[||s_θ(x̃) + ε/σ||²] / 2
    
    Args:
        score_model: Neural network outputting score
        samples: Clean data samples, shape (N, D)
        noise_std: Standard deviation σ of Gaussian noise
    
    Returns:
        DSM loss value
    """
    # Sample noise
    noise = torch.randn_like(samples)
    
    # Perturb data: x̃ = x + σε
    noisy_samples = samples + noise_std * noise
    
    # Predict score at noisy samples
    predicted_score = score_model(noisy_samples)
    
    # Target: ∇log q(x̃|x) = -ε/σ
    target_score = -noise / noise_std
    
    # MSE loss
    loss = 0.5 * torch.mean(torch.sum((predicted_score - target_score) ** 2, dim=1))
    
    return loss
```

!!! warning "Common Pitfall"
    The target score for DSM is $-\boldsymbol{\epsilon}/\sigma$, not $-\boldsymbol{\epsilon}/\sigma^2$. The DSM loss as written predicts the **scaled score** $\sigma \cdot \mathbf{s}_\theta$, which simplifies to predicting $-\boldsymbol{\epsilon}/\sigma$.

### 7.3 Sliced Score Matching

```python
def ssm_loss(
    score_model: nn.Module,
    samples: torch.Tensor,
    n_projections: int = 1
) -> torch.Tensor:
    """
    Sliced Score Matching loss using Hutchinson's trace estimator.
    
    L_SSM = E[v^T ∇s_θ(x) v + (v^T s_θ(x))²/2]
    
    Args:
        score_model: Neural network outputting score
        samples: Data samples, shape (N, D)
        n_projections: Number of random projections M
    
    Returns:
        SSM loss value
    """
    samples = samples.clone().detach().requires_grad_(True)
    N, D = samples.shape
    
    # Compute scores
    scores = score_model(samples)  # (N, D)
    
    total_loss = 0.0
    
    for _ in range(n_projections):
        # Random projection vector (Rademacher for lower variance)
        v = torch.randint(0, 2, samples.shape, device=samples.device).float() * 2 - 1
        
        # Term 1: (v^T s_θ(x))² / 2
        vs = torch.sum(v * scores, dim=1)  # (N,)
        squared_term = 0.5 * vs ** 2
        
        # Term 2: v^T ∇s_θ(x) v via vector-Jacobian product
        vjp = torch.autograd.grad(
            outputs=vs.sum(),
            inputs=samples,
            create_graph=True
        )[0]  # (N, D)
        
        trace_term = torch.sum(vjp * v, dim=1)  # (N,)
        
        total_loss += torch.mean(squared_term + trace_term)
    
    return total_loss / n_projections
```

### 7.4 Simple Score Network

```python
class ScoreNetwork(nn.Module):
    """
    Simple MLP for score estimation.
    
    Input: x ∈ R^D
    Output: s_θ(x) ∈ R^D (score vector, same dimension as input)
    """
    
    def __init__(self, input_dim: int, hidden_dims: list[int] = [128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),  # Smooth activation for better gradients
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation - scores can be any real value)
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

---

## 8. Training Example

### 8.1 Complete Training Loop

```python
def train_score_model(
    samples: torch.Tensor,
    noise_std: float = 0.1,
    hidden_dims: list[int] = [128, 128],
    num_epochs: int = 1000,
    lr: float = 1e-3,
    batch_size: int = 256
):
    """
    Train a score model using DSM.
    
    Args:
        samples: Training data, shape (N, D)
        noise_std: Noise level σ for DSM
        hidden_dims: Hidden layer sizes
        num_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Mini-batch size
    
    Returns:
        model: Trained score network
        losses: Training loss history
    """
    N, D = samples.shape
    model = ScoreNetwork(D, hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(num_epochs):
        # Random mini-batch
        idx = torch.randperm(N)[:batch_size]
        batch = samples[idx]
        
        # Compute DSM loss
        loss = dsm_loss(model, batch, noise_std)
        
        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    return model, losses
```

### 8.2 Example: Learning Gaussian Score

```python
# Generate data from 2D Gaussian
torch.manual_seed(42)
mu = torch.tensor([0.0, 0.0])
sigma = 1.0
samples = torch.randn(10000, 2) * sigma + mu

# Train score model
model, losses = train_score_model(samples, noise_std=0.5, num_epochs=2000)

# Compare with analytical true score
test_points = torch.randn(100, 2) * 2
true_score = -(test_points - mu) / (sigma ** 2)
pred_score = model(test_points).detach()

# Compute error
mse = torch.mean((pred_score - true_score) ** 2).item()
print(f"MSE between predicted and true score: {mse:.6f}")
```

---

## 9. Choosing Noise Level in DSM

### 9.1 The Bias-Variance Trade-off

| $\sigma$ | Bias | Variance | Optimization | Detail |
|----------|------|----------|--------------|--------|
| Small | Low | High | Harder | Preserves fine structure |
| Large | High | Low | Easier | Blurs details |

### 9.2 Practical Guidelines

1. **Start with $\sigma \approx 0.5$** for normalized data (zero mean, unit variance)
2. **Larger $\sigma$ for high-dimensional data** (more regularization needed)
3. **Multiple noise levels** for best results (see NCSN)
4. **Data-dependent**: $\sigma \approx 0.5 \times$ median nearest-neighbor distance

### 9.3 When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| Low-dimensional ($D < 10$) | ESM or DSM |
| High-dimensional (images) | DSM ✅ |
| Noise perturbation problematic | SSM |
| Multi-modal distributions | DSM with multiple $\sigma$ (NCSN) |
| Production / diffusion models | DSM ✅ |

---

## 10. Summary

| Method | Loss Function | Complexity | Bias | Practical |
|--------|--------------|------------|------|-----------|
| **ESM** | $\frac{1}{2}\|\mathbf{s}_\theta\|^2 + \text{tr}(\nabla\mathbf{s}_\theta)$ | $O(D)$ backward | None | Low-dim only |
| **DSM** | $\frac{1}{2}\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) + \boldsymbol{\epsilon}/\sigma\|^2$ | 1 backward | $O(\sigma^2)$ | ✅ Recommended |
| **SSM** | $\mathbf{v}^\top\nabla\mathbf{s}_\theta\mathbf{v} + \frac{1}{2}(\mathbf{v}^\top\mathbf{s}_\theta)^2$ | $M$ backward | None | Noise-sensitive |

!!! tip "Key Takeaways"
    1. **Score matching enables learning without knowing the true score** (Stein's identity)
    2. **ESM is theoretically elegant but computationally intractable** for high dimensions
    3. **DSM is the practical choice**—simple MSE loss, scales to any dimension
    4. **SSM provides an unbiased alternative** when noise perturbation is problematic
    5. **DSM is the foundation** for modern diffusion models (DDPM, score SDE)

---

## Exercises

1. **Implement ESM vs DSM**: Train both methods on a 2D Gaussian mixture and compare training time, final Fisher divergence, and score field quality.

2. **Noise Level Study**: For a fixed dataset, plot the final loss vs. noise level $\sigma \in \{0.1, 0.3, 0.5, 1.0, 2.0\}$. Find the optimal $\sigma$.

3. **SSM Projections**: Implement SSM with $M \in \{1, 2, 4, 8\}$ projections. How does $M$ affect accuracy and training time?

4. **Derive Stein's Identity**: Prove that $\mathbb{E}_p[f(\mathbf{x})^\top \nabla \log p(\mathbf{x})] = -\mathbb{E}_p[\nabla \cdot f(\mathbf{x})]$ using integration by parts.

5. **DSM-ESM Equivalence**: Empirically verify that DSM loss approaches ESM loss as $\sigma \to 0$ on a simple 2D distribution.

---

## References

1. Hyvärinen, A. (2005). "Estimation of Non-Normalized Statistical Models by Score Matching." *JMLR*.
2. Vincent, P. (2011). "A Connection Between Score Matching and Denoising Autoencoders." *Neural Computation*.
3. Song, Y., et al. (2019). "Sliced Score Matching: A Scalable Approach to Density and Score Estimation." *UAI*.
4. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
