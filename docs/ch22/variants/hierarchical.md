# Hierarchical VAE

Multi-scale latent representations with continuous variables at multiple levels.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain the motivation for hierarchical latent spaces
- Derive the ELBO for hierarchical VAEs
- Understand top-down and bottom-up inference pathways
- Describe the connection between hierarchical VAEs and diffusion models

---

## Motivation

A standard VAE uses a single latent layer $z$, forcing one representation to capture all levels of abstraction simultaneously. Hierarchical VAEs introduce multiple layers of latent variables $z_1, z_2, \ldots, z_L$ organized from bottom (closest to data) to top (most abstract), enabling richer representations and tighter ELBOs.

---

## Mathematical Framework

### Generative Model

$$p_\theta(x, z_1, \ldots, z_L) = p(z_L) \prod_{l=1}^{L-1} p_\theta(z_l | z_{l+1}) \cdot p_\theta(x | z_1)$$

### Inference Model

$$q_\phi(z_1, \ldots, z_L | x) = q_\phi(z_L | x) \prod_{l=1}^{L-1} q_\phi(z_l | z_{l+1}, x)$$

### Hierarchical ELBO

$$\mathcal{L} = \mathbb{E}_q[\log p_\theta(x|z_1)] - D_{KL}(q_\phi(z_L|x) \| p(z_L)) - \sum_{l=1}^{L-1} \mathbb{E}_q[D_{KL}(q_\phi(z_l|z_{l+1}, x) \| p_\theta(z_l|z_{l+1}))]$$

Each KL term regularizes one layer's approximate posterior against its conditional prior from the layer above.

---

## Ladder VAE Architecture

The Ladder VAE (Sønderby et al., 2016) combines **bottom-up** feature extraction with **top-down** generative conditioning:

```
Bottom-Up (Encoder)         Top-Down (Generative + Inference)

x ──► h_1 ──────────────────────────► q(z_1 | h_1, z_2) ──► z_1
       │                                      ▲
       ▼                                      │
      h_2 ──────────────────────────► q(z_2 | h_2, z_3) ──► z_2
       │                                      ▲
       ▼                                      │
      h_L ────────────────────────────► q(z_L | h_L) ──────► z_L
```

The bottom-up pass extracts features $h_l$ at each scale. The top-down pass combines these features with generative model information to form the approximate posterior at each level.

### Implementation

```python
import torch
import torch.nn as nn

class HierarchicalVAE(nn.Module):
    """
    Simplified hierarchical VAE with L latent layers.
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, 
                 latent_dims=[32, 16, 8], num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        self.latent_dims = latent_dims
        
        # Bottom-up encoder layers
        self.bu_layers = nn.ModuleList()
        in_dim = input_dim
        for l in range(num_layers):
            self.bu_layers.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU()
            ))
            in_dim = hidden_dim
        
        # Posterior parameters at each level
        self.post_mu = nn.ModuleList()
        self.post_logvar = nn.ModuleList()
        for l in range(num_layers):
            cond_dim = hidden_dim + (latent_dims[l+1] if l < num_layers - 1 else 0)
            self.post_mu.append(nn.Linear(cond_dim, latent_dims[l]))
            self.post_logvar.append(nn.Linear(cond_dim, latent_dims[l]))
        
        # Prior parameters (conditional on layer above)
        self.prior_mu = nn.ModuleList()
        self.prior_logvar = nn.ModuleList()
        for l in range(num_layers - 1):
            self.prior_mu.append(nn.Linear(latent_dims[l+1], latent_dims[l]))
            self.prior_logvar.append(nn.Linear(latent_dims[l+1], latent_dims[l]))
        
        # Decoder from z_1
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Bottom-up pass
        h = []
        current = x
        for layer in self.bu_layers:
            current = layer(current)
            h.append(current)
        
        # Top-down pass (from top to bottom)
        z_samples = [None] * self.num_layers
        total_kl = 0
        
        for l in reversed(range(self.num_layers)):
            if l == self.num_layers - 1:
                # Top layer: posterior from BU features only
                post_mu = self.post_mu[l](h[l])
                post_logvar = self.post_logvar[l](h[l])
                prior_mu = torch.zeros_like(post_mu)
                prior_logvar = torch.zeros_like(post_logvar)
            else:
                # Lower layers: combine BU features with z from above
                combined = torch.cat([h[l], z_samples[l+1]], dim=1)
                post_mu = self.post_mu[l](combined)
                post_logvar = self.post_logvar[l](combined)
                prior_mu = self.prior_mu[l](z_samples[l+1])
                prior_logvar = self.prior_logvar[l](z_samples[l+1])
            
            # Sample
            std = torch.exp(0.5 * post_logvar)
            z_samples[l] = post_mu + std * torch.randn_like(std)
            
            # KL divergence at this layer
            kl_l = 0.5 * torch.sum(
                prior_logvar - post_logvar
                + (post_logvar.exp() + (post_mu - prior_mu).pow(2)) / prior_logvar.exp()
                - 1, dim=1
            )
            total_kl = total_kl + kl_l.mean()
        
        # Decode from bottom latent
        recon = self.decoder(z_samples[0])
        
        return recon, total_kl, z_samples
```

---

## Benefits

Hierarchical VAEs achieve **tighter ELBOs** because conditional priors are more flexible than fixed $\mathcal{N}(0, I)$. They enable **specialization** where each layer captures different abstraction levels. They also reduce **posterior collapse** because lower layers have learned conditional priors rather than a fixed prior, giving them a natural role.

---

## Connection to Diffusion Models

Hierarchical VAEs with many layers can be viewed as a discrete-step approximation to continuous diffusion processes. As the number of layers $L \to \infty$ and each layer makes small changes, the hierarchical VAE approaches the formulation of denoising diffusion probabilistic models (DDPMs). This connection was formalized by Kingma et al. (2021) in the Variational Diffusion Model.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Multiple latent layers** | Different layers capture different abstraction levels |
| **Top-down + bottom-up** | Combines data features with generative conditioning |
| **Conditional priors** | $p(z_l \| z_{l+1})$ is more flexible than fixed Gaussian |
| **Tighter ELBO** | Hierarchical structure reduces approximation gap |

---

## Exercises

### Exercise 1: Hierarchy Depth

Train hierarchical VAEs with 1, 2, 3, and 5 layers on MNIST. Compare ELBO and reconstruction quality.

### Exercise 2: Layer Analysis

For a trained 3-layer hierarchical VAE, examine what each layer encodes by manipulating latents at each level independently.

---

## What's Next

The next section covers [NVAE](nvae.md), a state-of-the-art hierarchical VAE architecture.
