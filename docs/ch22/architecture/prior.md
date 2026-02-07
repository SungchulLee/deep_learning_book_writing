# Prior Selection

Choosing and designing the prior distribution $p(z)$ for the latent space.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain the role of the prior in VAE generation and regularization
- Understand why the standard Gaussian prior is the default choice
- Describe alternative priors and their trade-offs
- Implement VAEs with non-standard priors

---

## The Role of the Prior

The prior $p(z)$ serves two critical roles in VAEs. During **training**, the KL divergence $D_{KL}(q_\phi(z|x) \| p(z))$ regularizes the encoder's output to match the prior, shaping the geometry of the latent space. During **generation**, we sample $z \sim p(z)$ and decode, so the prior defines the distribution from which we draw latent codes.

A mismatch between the prior and the aggregated posterior $q_\phi(z) = \mathbb{E}_{p_{\text{data}}}[q_\phi(z|x)]$ leads to poor generation: the decoder sees latent codes during generation that it never encountered during training.

---

## The Standard Gaussian Prior

### Why $\mathcal{N}(0, I)$?

The isotropic Gaussian prior $p(z) = \mathcal{N}(0, I)$ is the standard choice for several reasons:

| Property | Benefit |
|----------|---------|
| **Closed-form KL** | Analytic KL divergence with Gaussian $q_\phi(z\|x)$ |
| **Easy sampling** | Trivial to sample from |
| **Isotropy** | No preferred direction in latent space |
| **Maximum entropy** | Among distributions with given mean and variance, Gaussian has maximum entropy |

### Limitations

The standard Gaussian prior can be overly restrictive. It assumes the aggregated posterior should be unimodal and spherically symmetric, which may not match the true structure of the data. For complex data with distinct clusters or manifold structure, the Gaussian prior forces a suboptimal latent organization.

---

## The Prior Hole Problem

### What Is It?

The **prior hole** arises when the aggregated posterior $q_\phi(z)$ doesn't match the prior $p(z)$. Regions where $p(z)$ has mass but $q_\phi(z)$ does not are "holes" — the decoder hasn't learned to produce meaningful outputs for these latent codes, leading to poor generation quality.

```
Prior p(z):              Aggregated q(z):        Gap:
   ┌─────────┐             ┌─────────┐          ┌─────────┐
   │  ░░░░░  │             │         │          │  ░░░░░  │
   │ ░░░░░░░ │             │  ▓▓ ▓▓  │          │ ░░   ░░ │
   │ ░░░░░░░ │      -      │  ▓▓ ▓▓  │    =     │ ░░   ░░ │
   │ ░░░░░░░ │             │         │          │ ░░░░░░░ │
   │  ░░░░░  │             │         │          │  ░░░░░  │
   └─────────┘             └─────────┘          └─────────┘
   Smooth bell          Clustered data          Decoder unseen
```

### Consequence

Sampling from the prior may land in these holes, producing low-quality or unrealistic outputs. This is a fundamental limitation when the data has complex structure.

---

## Alternative Priors

### Mixture of Gaussians Prior

A mixture prior can capture multi-modal structure:

$$p(z) = \sum_{k=1}^{K} \pi_k \mathcal{N}(z; \mu_k, \Sigma_k)$$

```python
import torch
import torch.nn as nn

class GaussianMixturePrior(nn.Module):
    """Learnable Gaussian mixture prior."""
    
    def __init__(self, latent_dim, num_components=10):
        super().__init__()
        self.num_components = num_components
        
        # Learnable mixture parameters
        self.logits = nn.Parameter(torch.zeros(num_components))
        self.means = nn.Parameter(torch.randn(num_components, latent_dim) * 0.5)
        self.logvars = nn.Parameter(torch.zeros(num_components, latent_dim))
    
    def log_prob(self, z):
        """Compute log p(z) under the mixture."""
        # z: [batch, latent_dim]
        # Expand for broadcasting: [batch, K, latent_dim]
        z_exp = z.unsqueeze(1)
        
        # Log probabilities of each component
        log_var = self.logvars.unsqueeze(0)
        means = self.means.unsqueeze(0)
        
        log_p_per_component = -0.5 * (log_var + (z_exp - means).pow(2) / log_var.exp())
        log_p_per_component = log_p_per_component.sum(dim=2)  # [batch, K]
        
        # Mix with weights
        log_weights = torch.log_softmax(self.logits, dim=0)
        log_p = torch.logsumexp(log_p_per_component + log_weights, dim=1)
        
        return log_p
    
    def sample(self, num_samples):
        """Sample from the mixture prior."""
        # Choose components
        weights = torch.softmax(self.logits, dim=0)
        indices = torch.multinomial(weights, num_samples, replacement=True)
        
        # Sample from chosen components
        means = self.means[indices]
        stds = torch.exp(0.5 * self.logvars[indices])
        
        return means + stds * torch.randn_like(means)
```

### VampPrior (Variational Mixture of Posteriors)

The VampPrior (Tomczak & Welling, 2018) defines the prior as a mixture of encoder outputs evaluated at learned pseudo-inputs:

$$p(z) = \frac{1}{K}\sum_{k=1}^{K} q_\phi(z | u_k)$$

where $u_1, \ldots, u_K$ are learnable pseudo-inputs in data space. This ensures the prior automatically matches the aggregated posterior's structure.

```python
class VampPrior(nn.Module):
    """VampPrior: Variational Mixture of Posteriors."""
    
    def __init__(self, encoder, input_dim, num_pseudoinputs=100):
        super().__init__()
        self.encoder = encoder
        self.pseudoinputs = nn.Parameter(torch.randn(num_pseudoinputs, input_dim) * 0.05)
    
    def get_prior_params(self):
        """Get mixture component parameters from pseudo-inputs."""
        with torch.no_grad():
            mu, logvar = self.encoder(torch.sigmoid(self.pseudoinputs))
        return mu, logvar
```

### Learnable Prior via Normalizing Flows

A normalizing flow can transform the standard Gaussian into a more expressive prior:

$$z_0 \sim \mathcal{N}(0, I), \quad z = f(z_0), \quad p(z) = p(z_0)|det \frac{\partial f^{-1}}{\partial z}|$$

This allows the prior to capture complex, multi-modal structure while maintaining tractable density evaluation.

---

## Conditional Priors

In Conditional VAEs, the prior can depend on conditioning information:

$$p(z|y) = \mathcal{N}(z; \mu_{\text{prior}}(y), \sigma^2_{\text{prior}}(y))$$

A learned conditional prior allows different classes or conditions to occupy different regions of latent space naturally, rather than forcing all conditions to share the same $\mathcal{N}(0, I)$ prior.

---

## Practical Recommendations

For most applications, start with the standard Gaussian $\mathcal{N}(0, I)$. It is simple, well-understood, and works well for moderate-complexity data. Consider alternative priors when generation quality is poor despite good reconstruction, the data has clear cluster structure (use mixture prior), you need the prior to adapt to the data (use VampPrior), or you're working with complex, multi-modal distributions (use flow-based prior).

---

## Summary

| Prior | KL Computation | Expressiveness | Complexity |
|-------|---------------|----------------|------------|
| **Standard Gaussian** | Closed-form | Low | Minimal |
| **Mixture of Gaussians** | Requires sampling | Medium | Moderate |
| **VampPrior** | Requires sampling | High | Moderate |
| **Flow-based** | Exact (change of variables) | High | High |
| **Conditional** | Depends on form | Task-specific | Moderate |

---

## Exercises

### Exercise 1: Prior Visualization

Train VAEs with standard Gaussian and 10-component mixture prior on MNIST. Visualize the aggregated posterior and prior in 2D.

### Exercise 2: Generation Quality

Compare FID scores (or visual quality) of samples generated from standard vs mixture priors.

### Exercise 3: VampPrior

Implement VampPrior with 50 pseudo-inputs. Do the learned pseudo-inputs resemble training data?

---

## What's Next

The next section covers [Posterior Collapse](posterior_collapse.md), a common training failure where the encoder ignores the input data.
