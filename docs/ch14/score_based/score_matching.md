# Score Matching

**Score matching** is a technique for estimating probability distributions by learning their score functions without requiring the normalization constant.

## The Score Matching Problem

Given samples $\{x_i\}_{i=1}^N$ from an unknown distribution $p_{\text{data}}(x)$, we want to learn a score network $s_\theta(x)$ that approximates:

$$
s_\theta(x) \approx \nabla_x \log p_{\text{data}}(x)
$$

## Explicit Score Matching

The naive approach would minimize the **Fisher divergence**:

$$
\mathcal{L}_{\text{ESM}}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}} \left[ \| s_\theta(x) - \nabla_x \log p_{\text{data}}(x) \|^2 \right]
$$

**Problem**: This requires knowing $\nabla_x \log p_{\text{data}}(x)$, which is exactly what we're trying to learn!

## Implicit Score Matching (Hyvärinen, 2005)

The key insight is that we can rewrite the objective without the unknown true score. Using integration by parts:

$$
\mathcal{L}_{\text{ISM}}(\theta) = \mathbb{E}_{p_{\text{data}}} \left[ \frac{1}{2} \| s_\theta(x) \|^2 + \text{tr}(\nabla_x s_\theta(x)) \right]
$$

where $\text{tr}(\nabla_x s_\theta(x)) = \sum_i \frac{\partial s_\theta^{(i)}(x)}{\partial x_i}$ is the trace of the Jacobian.

### Derivation

Starting from the Fisher divergence:

$$
\begin{aligned}
\mathcal{L}_{\text{ESM}} &= \frac{1}{2} \mathbb{E}_{p_{\text{data}}} \left[ \| s_\theta(x) \|^2 - 2 s_\theta(x)^T \nabla_x \log p_{\text{data}}(x) + \| \nabla_x \log p_{\text{data}}(x) \|^2 \right]
\end{aligned}
$$

The last term is constant w.r.t. $\theta$. For the middle term, integration by parts gives:

$$
\mathbb{E}_{p_{\text{data}}} [s_\theta(x)^T \nabla_x \log p_{\text{data}}(x)] = -\mathbb{E}_{p_{\text{data}}} [\text{tr}(\nabla_x s_\theta(x))]
$$

under boundary conditions (density vanishes at infinity).

## Computational Challenge

The Jacobian trace $\text{tr}(\nabla_x s_\theta(x))$ requires computing $d$ partial derivatives, where $d$ is the data dimension. For images with millions of pixels, this is prohibitively expensive.

### Solutions

1. **Sliced Score Matching**: Project onto random directions (see dedicated section)
2. **Denoising Score Matching**: Add noise and use an explicit target (see next section)
3. **Finite Differences**: Approximate derivatives numerically

## Sliced Score Matching

Project the score onto random directions $v$:

$$
\mathcal{L}_{\text{SSM}}(\theta) = \mathbb{E}_{v \sim p_v} \mathbb{E}_{p_{\text{data}}} \left[ \frac{1}{2} (v^T s_\theta(x))^2 + v^T \nabla_x s_\theta(x) v \right]
$$

where $v$ is typically drawn from $\mathcal{N}(0, I)$ or uniform on the sphere.

**Advantage**: Only requires $O(1)$ Jacobian-vector products instead of $O(d)$ scalar derivatives.

## Practical Implementation

```python
import torch
import torch.nn as nn

def sliced_score_matching_loss(score_net, x, n_slices=1):
    """
    Sliced score matching loss.
    
    Args:
        score_net: Network that outputs score s(x)
        x: Input data [batch_size, dim]
        n_slices: Number of random projections
    
    Returns:
        Loss value
    """
    x = x.requires_grad_(True)
    score = score_net(x)  # [batch_size, dim]
    
    loss = 0.0
    for _ in range(n_slices):
        # Random projection direction
        v = torch.randn_like(x)
        v = v / v.norm(dim=-1, keepdim=True)
        
        # Score projection: v^T s(x)
        sv = (score * v).sum(dim=-1)  # [batch_size]
        
        # Jacobian-vector product: v^T (∂s/∂x) v
        grad_sv = torch.autograd.grad(
            sv.sum(), x, create_graph=True
        )[0]
        jvp = (grad_sv * v).sum(dim=-1)  # [batch_size]
        
        # Sliced score matching objective
        loss += 0.5 * sv.pow(2) + jvp
    
    return loss.mean() / n_slices
```

## Limitations of Vanilla Score Matching

1. **Low-density regions**: Score estimation is unreliable where few samples exist
2. **Multimodality**: Gradients between modes may not point toward either mode
3. **High dimensions**: Even sliced score matching struggles in very high dimensions

These limitations motivate **denoising score matching** and **noise conditional score networks**, which add noise to create smoother training targets.

## Connection to Other Methods

### Maximum Likelihood Estimation

Score matching can be viewed as an alternative to MLE that avoids computing normalizing constants. For exponential families, they are equivalent.

### Contrastive Divergence

Like contrastive divergence, score matching avoids partition functions. However, score matching uses explicit gradient information rather than MCMC samples.

### Noise Contrastive Estimation (NCE)

Both NCE and score matching learn unnormalized models, but NCE requires a noise distribution while score matching does not.

## Summary

Score matching enables learning score functions without knowing the data density. The implicit score matching objective avoids needing the true score, while sliced score matching makes high-dimensional problems tractable. Despite these advances, vanilla score matching struggles with multimodal distributions and low-density regions, motivating denoising approaches used in modern diffusion models.
