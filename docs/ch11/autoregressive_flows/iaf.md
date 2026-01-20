# Inverse Autoregressive Flow (IAF)

## Introduction

Inverse Autoregressive Flow (IAF) is a normalizing flow architecture that achieves **fast sampling** at the cost of slow density evaluation. By reversing the autoregressive direction compared to MAF, IAF enables parallel sample generation while requiring sequential computation for likelihood evaluation. This makes IAF particularly useful in variational autoencoders where fast sampling from the approximate posterior is crucial.

## Motivation: The Sampling Bottleneck

### Standard Autoregressive Flows

In standard autoregressive flows (like MAF), sampling requires sequential computation:

```python
# MAF sampling: SLOW (sequential)
def maf_sample(model, n_samples):
    x = torch.zeros(n_samples, dim)
    for d in range(dim):
        mu, sigma = model.get_params(x)  # Depends on x[:, :d]
        x[:, d] = mu[:, d] + sigma[:, d] * torch.randn(n_samples)
    return x
```

Each dimension depends on all previous dimensions, forcing O(D) sequential steps.

### IAF's Key Insight

IAF inverts the dependency structure:

```python
# IAF sampling: FAST (parallel)
def iaf_sample(model, n_samples):
    z = torch.randn(n_samples, dim)  # Sample all noise at once
    x = model.forward(z)  # Single parallel pass
    return x
```

The transformation from noise to data is **parallel**, while the inverse (data to noise) is **sequential**.

## Mathematical Formulation

### Autoregressive Transform

Standard autoregressive transform from $\mathbf{z}$ to $\mathbf{x}$:

$$x_d = \mu_d(\mathbf{x}_{<d}) + \sigma_d(\mathbf{x}_{<d}) \cdot z_d$$

**Problem**: Computing $x_d$ requires $x_1, \ldots, x_{d-1}$, which requires $x_1, \ldots, x_{d-2}$, etc.

### Inverse Autoregressive Transform

IAF reverses the dependency:

$$x_d = \mu_d(\mathbf{z}_{<d}) + \sigma_d(\mathbf{z}_{<d}) \cdot z_d$$

**Key difference**: Parameters depend on **previous noise values** $\mathbf{z}_{<d}$, not previous outputs.

Since all $z_d$ are sampled independently upfront, all transformations can be computed in parallel!

### Forward Pass (Sampling)

Given base samples $\mathbf{z} \sim \mathcal{N}(0, I)$:

$$x_d = \mu_d(z_1, \ldots, z_{d-1}) + \exp(s_d(z_1, \ldots, z_{d-1})) \cdot z_d$$

This can be computed for all $d$ simultaneously using MADE-style masking.

### Inverse Pass (Encoding)

Given data $\mathbf{x}$, recover $\mathbf{z}$:

$$z_d = \frac{x_d - \mu_d(z_1, \ldots, z_{d-1})}{\exp(s_d(z_1, \ldots, z_{d-1}))}$$

**Problem**: Computing $z_d$ requires $z_1, \ldots, z_{d-1}$, which must be computed first.

This is **sequential** - O(D) steps.

## Jacobian and Log-Determinant

### Triangular Jacobian

The Jacobian $\partial \mathbf{x} / \partial \mathbf{z}$ is **lower triangular**:

$$\frac{\partial x_i}{\partial z_j} = \begin{cases}
\exp(s_i(z_{<i})) & \text{if } i = j \\
\frac{\partial \mu_i}{\partial z_j} + z_i \frac{\partial \exp(s_i)}{\partial z_j} & \text{if } i > j \\
0 & \text{if } i < j
\end{cases}$$

### Log-Determinant

For triangular matrices, determinant is the product of diagonal elements:

$$\log |\det J| = \sum_{d=1}^{D} s_d(z_1, \ldots, z_{d-1})$$

This is computed during the forward pass at no extra cost!

## Implementation

### Basic IAF Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IAFLayer(nn.Module):
    """
    Single Inverse Autoregressive Flow layer.
    
    Forward: z -> x (parallel, used for sampling)
    Inverse: x -> z (sequential, used for density evaluation)
    """
    
    def __init__(self, dim, hidden_dim=64, context_dim=0):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        
        # MADE-style network for computing mu and s
        # Input: z (and optional context)
        # Output: mu and s for each dimension
        self.made = self._build_made(dim, hidden_dim, context_dim)
    
    def _build_made(self, dim, hidden_dim, context_dim):
        """Build MADE network with proper masking."""
        input_dim = dim + context_dim
        
        # Simple 2-layer MADE
        layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, dim * 2)  # mu and s
        ])
        
        # Create masks
        m_input = np.arange(1, dim + 1)
        if context_dim > 0:
            m_input = np.concatenate([m_input, np.zeros(context_dim)])
        
        m_hidden = np.random.randint(1, dim, size=hidden_dim)
        m_output = np.tile(np.arange(dim), 2)  # For mu and s
        
        # Mask for input -> hidden
        mask1 = (m_hidden[:, None] >= m_input[None, :]).astype(np.float32)
        
        # Mask for hidden -> hidden
        mask2 = (m_hidden[:, None] >= m_hidden[None, :]).astype(np.float32)
        
        # Mask for hidden -> output (strict inequality)
        mask3 = (m_output[:, None] > m_hidden[None, :]).astype(np.float32)
        
        self.register_buffer('mask1', torch.from_numpy(mask1))
        self.register_buffer('mask2', torch.from_numpy(mask2))
        self.register_buffer('mask3', torch.from_numpy(mask3))
        
        return layers
    
    def _masked_forward(self, x):
        """Forward through MADE with masks."""
        h = F.relu(F.linear(x, self.made[0].weight * self.mask1, self.made[0].bias))
        h = F.relu(F.linear(h, self.made[1].weight * self.mask2, self.made[1].bias))
        out = F.linear(h, self.made[2].weight * self.mask3, self.made[2].bias)
        return out
    
    def forward(self, z, context=None):
        """
        Forward pass: z -> x (parallel).
        
        Args:
            z: Base samples (batch, dim)
            context: Optional context (batch, context_dim)
        
        Returns:
            x: Transformed samples (batch, dim)
            log_det: Log determinant (batch,)
        """
        if context is not None:
            input_z = torch.cat([z, context], dim=-1)
        else:
            input_z = z
        
        # Get transformation parameters
        params = self._masked_forward(input_z)
        mu = params[:, :self.dim]
        s = params[:, self.dim:]
        
        # Transform: x = mu + exp(s) * z
        x = mu + torch.exp(s) * z
        
        # Log determinant
        log_det = s.sum(dim=-1)
        
        return x, log_det
    
    def inverse(self, x, context=None):
        """
        Inverse pass: x -> z (sequential).
        
        Args:
            x: Data samples (batch, dim)
            context: Optional context (batch, context_dim)
        
        Returns:
            z: Base samples (batch, dim)
            log_det: Log determinant (batch,)
        """
        batch_size = x.shape[0]
        z = torch.zeros_like(x)
        log_det = torch.zeros(batch_size, device=x.device)
        
        for d in range(self.dim):
            # Build input: z[:, :d] is known, rest is zeros
            if context is not None:
                input_z = torch.cat([z, context], dim=-1)
            else:
                input_z = z
            
            # Get parameters
            params = self._masked_forward(input_z)
            mu_d = params[:, d]
            s_d = params[:, self.dim + d]
            
            # Invert: z_d = (x_d - mu_d) / exp(s_d)
            z[:, d] = (x[:, d] - mu_d) * torch.exp(-s_d)
            log_det -= s_d
        
        return z, log_det
```

### Stacking IAF Layers

```python
class IAF(nn.Module):
    """
    Full IAF model with multiple layers.
    """
    
    def __init__(self, dim, hidden_dim=64, n_layers=4, context_dim=0):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([
            IAFLayer(dim, hidden_dim, context_dim)
            for _ in range(n_layers)
        ])
        
        # Permutations between layers
        self.permutations = nn.ModuleList()
        for _ in range(n_layers - 1):
            perm = torch.randperm(dim)
            self.register_buffer(f'perm_{len(self.permutations)}', perm)
            self.permutations.append(perm)
    
    def forward(self, z, context=None):
        """
        Forward pass: z -> x (fast sampling).
        """
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        x = z
        
        for i, layer in enumerate(self.layers):
            x, log_det = layer(x, context)
            log_det_total += log_det
            
            # Permute between layers (except last)
            if i < len(self.permutations):
                perm = getattr(self, f'perm_{i}')
                x = x[:, perm]
        
        return x, log_det_total
    
    def inverse(self, x, context=None):
        """
        Inverse pass: x -> z (slow density evaluation).
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for i in reversed(range(self.n_layers)):
            # Inverse permutation
            if i < len(self.permutations):
                perm = getattr(self, f'perm_{i}')
                inv_perm = torch.argsort(perm)
                z = z[:, inv_perm]
            
            z, log_det = self.layers[i].inverse(z, context)
            log_det_total += log_det
        
        return z, log_det_total
    
    def sample(self, n_samples, context=None):
        """Generate samples."""
        z = torch.randn(n_samples, self.dim)
        if context is not None and context.device != z.device:
            z = z.to(context.device)
        x, _ = self.forward(z, context)
        return x
    
    def log_prob(self, x, context=None):
        """Compute log probability."""
        z, log_det = self.inverse(x, context)
        
        # Log prob under base distribution
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        
        return log_pz + log_det
```

## IAF for Variational Autoencoders

### The VAE Posterior Problem

Standard VAE uses simple Gaussian posterior:

$$q(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mu(\mathbf{x}), \sigma^2(\mathbf{x}))$$

This is often too restrictive. IAF enables richer posteriors.

### IAF-Enhanced VAE

```python
class IAFVAE(nn.Module):
    """VAE with IAF-enhanced posterior."""
    
    def __init__(self, input_dim, latent_dim, hidden_dim=256, n_iaf_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: outputs initial mu, sigma and context for IAF
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_log_sigma = nn.Linear(hidden_dim, latent_dim)
        self.enc_context = nn.Linear(hidden_dim, hidden_dim)
        
        # IAF layers (context-conditioned)
        self.iaf = IAF(latent_dim, hidden_dim=64, n_layers=n_iaf_layers, 
                       context_dim=hidden_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        """Encode and sample from IAF posterior."""
        h = self.encoder(x)
        
        # Initial Gaussian parameters
        mu = self.enc_mu(h)
        log_sigma = self.enc_log_sigma(h)
        
        # Context for IAF
        context = self.enc_context(h)
        
        # Sample from initial Gaussian
        eps = torch.randn_like(mu)
        z0 = mu + torch.exp(log_sigma) * eps
        
        # Transform through IAF
        z, log_det_iaf = self.iaf(z0, context)
        
        # KL divergence computation
        # log q(z|x) = log q(z0|x) - log_det_iaf
        log_qz0 = -0.5 * (eps ** 2 + 2 * log_sigma + np.log(2 * np.pi)).sum(dim=-1)
        log_qz = log_qz0 - log_det_iaf
        
        # log p(z) under standard normal prior
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        
        # KL = E[log q(z|x) - log p(z)]
        kl = log_qz - log_pz
        
        return z, kl
    
    def decode(self, z):
        """Decode latent to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass."""
        z, kl = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, kl
    
    def loss(self, x):
        """ELBO loss."""
        x_recon, kl = self.forward(x)
        
        # Reconstruction loss (Gaussian)
        recon_loss = 0.5 * ((x - x_recon) ** 2).sum(dim=-1)
        
        # ELBO = -recon_loss - KL
        elbo = -recon_loss - kl
        
        return -elbo.mean()  # Negative ELBO as loss
    
    def sample(self, n_samples):
        """Generate samples from prior."""
        z = torch.randn(n_samples, self.latent_dim)
        return self.decode(z)
```

## Comparison: IAF vs MAF

### Computational Trade-off

| Operation | MAF | IAF |
|-----------|-----|-----|
| **Sampling** | O(D) sequential | O(1) parallel |
| **Density** | O(1) parallel | O(D) sequential |
| **Training** | Fast (density-based) | Slow (sequential inverse) |
| **Use case** | Density estimation | VAE posterior |

### When to Use Each

**Use MAF when:**
- Primary goal is density estimation
- Need to evaluate many data points
- Can tolerate slow sampling

**Use IAF when:**
- Primary goal is fast sampling
- Building VAE with flexible posterior
- Training uses reparameterization (forward direction)

## Architectural Improvements

### LSTM-based IAF

For high-dimensional data, use LSTM to model dependencies:

```python
class LSTMIAF(nn.Module):
    """IAF with LSTM for sequential dependencies."""
    
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # LSTM processes z sequentially
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        
        # Output mu and s from LSTM hidden state
        self.output = nn.Linear(hidden_dim, 2)
    
    def forward(self, z, context=None):
        """Fast forward using LSTM."""
        batch_size = z.shape[0]
        
        # Reshape for LSTM: (batch, seq_len, 1)
        z_seq = z.unsqueeze(-1)
        
        # Initialize hidden state (optionally from context)
        if context is not None:
            h0 = context.unsqueeze(0)
            c0 = torch.zeros_like(h0)
        else:
            h0 = c0 = None
        
        # Process through LSTM
        outputs, _ = self.lstm(z_seq, (h0, c0) if h0 is not None else None)
        
        # Get mu, s from each timestep
        params = self.output(outputs)  # (batch, dim, 2)
        mu = params[:, :, 0]
        s = params[:, :, 1]
        
        # Shift mu, s so they depend on previous z values
        mu_shifted = torch.cat([torch.zeros(batch_size, 1, device=z.device), 
                                 mu[:, :-1]], dim=1)
        s_shifted = torch.cat([torch.zeros(batch_size, 1, device=z.device), 
                                s[:, :-1]], dim=1)
        
        # Transform
        x = mu_shifted + torch.exp(s_shifted) * z
        log_det = s_shifted.sum(dim=-1)
        
        return x, log_det
```

### Attention-based IAF

Using self-attention for global dependencies:

```python
class AttentionIAF(nn.Module):
    """IAF with causal self-attention."""
    
    def __init__(self, dim, n_heads=4, hidden_dim=64):
        super().__init__()
        self.dim = dim
        
        # Causal self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Project z to hidden dim
        self.input_proj = nn.Linear(1, hidden_dim)
        
        # Output mu and s
        self.output_proj = nn.Linear(hidden_dim, 2)
        
        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(dim, dim), diagonal=1).bool()
        )
    
    def forward(self, z, context=None):
        batch_size = z.shape[0]
        
        # Project to hidden dim
        z_embed = self.input_proj(z.unsqueeze(-1))  # (batch, dim, hidden)
        
        # Causal self-attention
        attended, _ = self.attention(
            z_embed, z_embed, z_embed,
            attn_mask=self.causal_mask
        )
        
        # Get parameters
        params = self.output_proj(attended)
        mu = params[:, :, 0]
        s = params[:, :, 1]
        
        # Transform
        x = mu + torch.exp(s) * z
        log_det = s.sum(dim=-1)
        
        return x, log_det
```

## Training Considerations

### Gradient Flow

IAF benefits from deep architectures, but gradients must flow through many sequential operations during inverse. Solutions:

1. **Residual connections**: Add skip connections between IAF layers
2. **Gradient checkpointing**: Trade compute for memory
3. **Layer normalization**: Stabilize activations

### Initialization

Good initialization is crucial:

```python
def init_iaf_weights(module):
    """Initialize IAF for identity-like initial transform."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=0.1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

## Summary

Inverse Autoregressive Flow:

1. **Fast sampling** through parallel forward transformation
2. **Slow density** due to sequential inverse computation
3. **Ideal for VAEs** where sampling speed matters during training
4. **Trade-off** with MAF - opposite computational characteristics
5. **Context conditioning** enables flexible posterior distributions

The key insight is that autoregressive structure can be inverted: instead of dependencies flowing through the output, they flow through the input noise. This simple reversal has profound implications for when each architecture is most useful.

## References

1. Kingma, D. P., et al. (2016). Improving Variational Inference with Inverse Autoregressive Flow. *NeurIPS*.
2. Papamakarios, G., et al. (2017). Masked Autoregressive Flow for Density Estimation. *NeurIPS*.
3. Chen, X., et al. (2017). Variational Lossy Autoencoder. *ICLR*.
4. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
