# Posterior Collapse

Understanding, diagnosing, and mitigating the posterior collapse problem in VAEs.

---

## Learning Objectives

By the end of this section, you will be able to:

- Define posterior collapse and explain why it occurs
- Diagnose collapse through quantitative metrics
- Implement mitigation strategies including KL annealing and free bits
- Understand the information-theoretic perspective on collapse

---

## What Is Posterior Collapse?

### Definition

**Posterior collapse** occurs when the approximate posterior matches the prior for all inputs:

$$q_\phi(z|x) \approx p(z) = \mathcal{N}(0, I) \quad \text{for all } x$$

The encoder ignores its input, outputting $\mu \approx 0$ and $\sigma^2 \approx 1$ regardless of $x$. The latent code carries no information about the data, and all generation burden falls on the decoder, which effectively becomes an unconditional generative model.

### Symptoms

| Symptom | Observation |
|---------|-------------|
| **KL → 0** | KL divergence collapses to near zero |
| **μ → 0** | Encoder means are near-zero for all inputs |
| **σ² → 1** | Encoder variances are near-one for all inputs |
| **I(X; Z) → 0** | Mutual information between data and latent codes vanishes |
| **Poor latent structure** | t-SNE of latent codes shows no class separation |

---

## Why Does Collapse Happen?

### The Optimization Landscape

At the start of training, the decoder is random and produces poor reconstructions regardless of $z$. The reconstruction term provides weak gradients. Meanwhile, the KL term provides a clear gradient toward $q(z|x) = p(z)$, which achieves $D_{KL} = 0$. The KL term is "easier" to optimize, so the model can get trapped in a local optimum where KL = 0 and the decoder learns to generate average-looking outputs.

### Formal Analysis

The ELBO is:

$$\mathcal{L} = \mathbb{E}_q[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

If the decoder is powerful enough to model $p(x)$ without using $z$ (e.g., an autoregressive decoder), it can achieve reasonable reconstruction loss even when $z$ carries no information. In this case, the optimal solution has $D_{KL} = 0$ because reducing KL improves the ELBO without hurting reconstruction.

### Decoder Capacity

Powerful decoders exacerbate collapse. An autoregressive decoder (like PixelCNN) can model complex distributions without latent information, making collapse the optimal solution. Simpler decoders (MLP, small CNN) are forced to rely on latent codes for reconstruction.

---

## Diagnosing Posterior Collapse

### Quantitative Metrics

```python
import torch

def diagnose_collapse(model, data_loader, device, threshold=0.01):
    """
    Comprehensive posterior collapse diagnosis.
    
    Args:
        model: Trained VAE
        data_loader: Data loader
        device: Device
        threshold: KL threshold for "active" dimension
    
    Returns:
        Dictionary of diagnostic metrics
    """
    model.eval()
    
    all_mu = []
    all_logvar = []
    all_kl = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(data.size(0), -1).to(device)
            mu, logvar = model.encode(data)
            
            # Per-dimension KL
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            all_kl.append(kl_per_dim.cpu())
    
    mu = torch.cat(all_mu, dim=0)
    logvar = torch.cat(all_logvar, dim=0)
    kl = torch.cat(all_kl, dim=0)
    
    # Metrics
    mean_kl_per_dim = kl.mean(dim=0)
    active_dims = (mean_kl_per_dim > threshold).sum().item()
    total_dims = mean_kl_per_dim.shape[0]
    
    mu_variance = mu.var(dim=0).mean().item()
    logvar_mean = logvar.mean().item()
    
    diagnostics = {
        'active_dimensions': active_dims,
        'total_dimensions': total_dims,
        'utilization': active_dims / total_dims,
        'total_kl': mean_kl_per_dim.sum().item(),
        'mu_variance': mu_variance,
        'mean_logvar': logvar_mean,
        'kl_per_dim': mean_kl_per_dim,
    }
    
    # Assessment
    if active_dims == 0:
        diagnostics['status'] = 'FULL COLLAPSE'
    elif active_dims < total_dims * 0.1:
        diagnostics['status'] = 'SEVERE COLLAPSE'
    elif active_dims < total_dims * 0.5:
        diagnostics['status'] = 'PARTIAL COLLAPSE'
    else:
        diagnostics['status'] = 'HEALTHY'
    
    return diagnostics
```

### Visual Diagnosis

Monitor these quantities during training:

1. **KL divergence over epochs:** A KL that drops to near-zero early and stays there indicates collapse
2. **Active dimensions over time:** Count of dimensions with mean KL above a threshold
3. **Latent variance across samples:** If $\text{Var}_{x}[\mu_\phi(x)]$ is small, the encoder is ignoring input variation

---

## Mitigation Strategies

### Strategy 1: KL Annealing

Gradually increase the KL weight during training, allowing the encoder to learn useful representations before regularization kicks in:

$$\mathcal{L} = \mathbb{E}_q[\log p(x|z)] - \beta(t) \cdot D_{KL}(q(z|x) \| p(z))$$

where $\beta(t)$ increases from 0 to 1 over training.

### Strategy 2: Free Bits

Guarantee each latent dimension contributes at least $\lambda$ nats of KL:

$$D_{KL}^{\text{free}} = \sum_{j=1}^{d} \max(\lambda, D_{KL,j})$$

This prevents individual dimensions from collapsing while allowing others to be used heavily.

### Strategy 3: Reduce Decoder Capacity

Use simpler decoders that cannot model data without latent information. For image data, use MLPs instead of autoregressive models. For sequential data, limit the decoder's receptive field.

### Strategy 4: δ-VAE

Constrain the minimum KL divergence:

$$\mathcal{L} = \mathbb{E}_q[\log p(x|z)] - \max(\delta, D_{KL}(q(z|x) \| p(z)))$$

This ensures the latent space always carries at least $\delta$ nats of information.

### Strategy 5: Aggressive Training Schedule

Train the encoder for multiple steps per decoder step, or use a larger learning rate for the encoder. This gives the encoder time to find useful representations before the decoder learns to ignore $z$.

---

## Information-Theoretic View

Posterior collapse can be understood through the lens of mutual information. The KL term in the ELBO penalizes $I(X; Z)$. When the penalty is too strong relative to the reconstruction benefit, the optimal solution has $I(X; Z) = 0$, meaning $z$ carries zero information about $x$.

The β-VAE framework makes this explicit: with $\beta > 1$, collapse is more likely because the information penalty is amplified. The critical $\beta$ value depends on data complexity and decoder capacity.

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Cause** | KL optimization easier than reconstruction; powerful decoders |
| **Detection** | KL → 0, inactive dimensions, no class separation |
| **KL annealing** | Gradually increase β from 0 to 1 |
| **Free bits** | Minimum KL per dimension |
| **Decoder control** | Limit decoder capacity to force latent usage |

---

## Exercises

### Exercise 1: Inducing Collapse

Train a VAE with β = 10 and observe posterior collapse. Plot KL per dimension over training.

### Exercise 2: Annealing Comparison

Compare linear annealing (10 epochs), cyclical annealing, and no annealing. Which prevents collapse best?

### Exercise 3: Active Dimension Count

Train VAEs with latent dimensions $d \in \{10, 32, 64, 128\}$ on MNIST. How many dimensions are active in each case?

---

## What's Next

The next section covers VAE variants, starting with [β-VAE](../variants/beta_vae.md) for learning disentangled representations.
