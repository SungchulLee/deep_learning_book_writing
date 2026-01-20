# VAE Loss Function

Putting it all together: the complete VAE training objective.

---

## Learning Objectives

By the end of this section, you will be able to:

- Implement the complete VAE loss function
- Understand each component and its role
- Handle different data types with appropriate losses
- Apply common loss modifications (β-VAE, annealing)

---

## The Complete VAE Loss

### Mathematical Form

The VAE training objective is to minimize the negative ELBO:

$$\mathcal{L}_{\text{VAE}} = \underbrace{-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction Loss}} + \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL Divergence}}$$

### Component Breakdown

| Component | Formula | Encourages |
|-----------|---------|------------|
| **Reconstruction** | $-\mathbb{E}_q[\log p(x\|z)]$ | Accurate data reconstruction |
| **KL Divergence** | $D_{KL}(q(z\|x) \| p(z))$ | Latent codes close to prior |

---

## Reconstruction Loss Options

### Binary Cross-Entropy (Bernoulli Decoder)

For data in $[0, 1]$:

$$\mathcal{L}_{\text{recon}} = -\sum_{i=1}^{d}[x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)]$$

```python
def bce_reconstruction(recon_x, x):
    """BCE for binary/normalized data."""
    return F.binary_cross_entropy(recon_x, x, reduction='sum')
```

### Mean Squared Error (Gaussian Decoder)

For continuous data:

$$\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|^2$$

```python
def mse_reconstruction(recon_x, x):
    """MSE for continuous data."""
    return F.mse_loss(recon_x, x, reduction='sum')
```

---

## KL Divergence (Closed Form)

### For Diagonal Gaussian

When $q(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ and $p(z) = \mathcal{N}(0, I)$:

$$D_{KL} = -\frac{1}{2}\sum_{j=1}^{k}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

### Implementation

```python
def kl_divergence(mu, logvar):
    """
    KL divergence from q(z|x) = N(mu, exp(logvar)) to p(z) = N(0, I).
    
    Args:
        mu: Mean [batch_size, latent_dim]
        logvar: Log variance [batch_size, latent_dim]
    
    Returns:
        KL divergence summed over batch and latent dimensions
    """
    # -0.5 * sum(1 + log(σ²) - μ² - σ²)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

---

## Complete Loss Implementation

### Standard VAE Loss

```python
import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar, reduction='sum'):
    """
    Standard VAE loss = Reconstruction + KL Divergence.
    
    Args:
        recon_x: Reconstructed data [batch_size, data_dim]
        x: Original data [batch_size, data_dim]
        mu: Encoder mean [batch_size, latent_dim]
        logvar: Encoder log-variance [batch_size, latent_dim]
        reduction: 'sum' or 'mean'
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (BCE for MNIST-like data)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if reduction == 'mean':
        kl_loss = kl_loss / mu.size(0)
    
    total_loss = recon_loss + kl_loss
    
    return total_loss, recon_loss, kl_loss
```

### β-VAE Loss

```python
def beta_vae_loss(recon_x, x, mu, logvar, beta=4.0):
    """
    β-VAE loss with weighted KL term.
    
    Higher beta encourages more disentangled representations
    at the cost of reconstruction quality.
    
    Args:
        beta: Weight for KL term (β > 1 for disentanglement)
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss
```

---

## Loss Modifications

### KL Annealing

Gradually increase KL weight during training:

```python
def get_kl_weight(epoch, warmup_epochs=10, max_weight=1.0):
    """Linear KL annealing schedule."""
    if epoch < warmup_epochs:
        return max_weight * epoch / warmup_epochs
    return max_weight

# In training loop
kl_weight = get_kl_weight(epoch)
loss = recon_loss + kl_weight * kl_loss
```

### Cyclical Annealing

Repeat annealing cycles:

```python
def cyclical_kl_weight(step, cycle_length=10000, ratio=0.5):
    """Cyclical annealing (Fu et al., 2019)."""
    cycle_position = step % cycle_length
    if cycle_position < cycle_length * ratio:
        return cycle_position / (cycle_length * ratio)
    return 1.0
```

### Free Bits

Prevent posterior collapse with minimum KL per dimension:

```python
def vae_loss_free_bits(recon_x, x, mu, logvar, free_bits=0.1):
    """VAE loss with free bits constraint."""
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL per dimension
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # Apply free bits: each dimension contributes at least 'free_bits'
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss = kl_per_dim.sum()
    
    return recon_loss + kl_loss, recon_loss, kl_loss
```

---

## Flexible Loss Class

```python
class VAELoss(nn.Module):
    """
    Flexible VAE loss with configurable options.
    """
    
    def __init__(self, 
                 reconstruction_type='bce',
                 beta=1.0,
                 free_bits=None,
                 reduction='sum'):
        super().__init__()
        self.reconstruction_type = reconstruction_type
        self.beta = beta
        self.free_bits = free_bits
        self.reduction = reduction
    
    def reconstruction_loss(self, recon_x, x):
        """Compute reconstruction loss based on type."""
        if self.reconstruction_type == 'bce':
            return F.binary_cross_entropy(recon_x, x, reduction=self.reduction)
        elif self.reconstruction_type == 'mse':
            return F.mse_loss(recon_x, x, reduction=self.reduction)
        else:
            raise ValueError(f"Unknown reconstruction type: {self.reconstruction_type}")
    
    def kl_loss(self, mu, logvar):
        """Compute KL divergence with optional free bits."""
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.free_bits is not None:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        
        if self.reduction == 'sum':
            return kl_per_dim.sum()
        else:
            return kl_per_dim.mean()
    
    def forward(self, recon_x, x, mu, logvar):
        """
        Compute total VAE loss.
        
        Returns:
            total_loss, recon_loss, kl_loss
        """
        recon = self.reconstruction_loss(recon_x, x)
        kl = self.kl_loss(mu, logvar)
        total = recon + self.beta * kl
        
        return total, recon, kl


# Usage
loss_fn = VAELoss(reconstruction_type='bce', beta=4.0, free_bits=0.1)
total_loss, recon_loss, kl_loss = loss_fn(recon_x, x, mu, logvar)
```

---

## Training Loop Example

```python
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train VAE for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.view(-1, 784).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Compute loss
        loss, recon, kl = loss_fn(recon_batch, data, mu, logvar)
        
        # Backward pass
        loss.backward()
        
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
    
    n_samples = len(dataloader.dataset)
    return {
        'loss': total_loss / n_samples,
        'recon': total_recon / n_samples,
        'kl': total_kl / n_samples
    }
```

---

## Summary

| Loss Component | Formula | Purpose |
|----------------|---------|---------|
| **BCE** | $-\sum[x\log\hat{x} + (1-x)\log(1-\hat{x})]$ | Binary/[0,1] data |
| **MSE** | $\|x - \hat{x}\|^2$ | Continuous data |
| **KL** | $-\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$ | Regularization |
| **β-VAE** | $\text{recon} + \beta \cdot \text{KL}$ | Disentanglement |

---

## Exercises

### Exercise 1: Loss Comparison

Train a VAE with MSE vs. BCE loss on MNIST. Compare reconstruction quality.

### Exercise 2: β Sweep

Train models with β ∈ {0.1, 1, 4, 10}. Plot:
- Reconstruction quality vs. β
- Latent space organization vs. β

### Exercise 3: Annealing Schedules

Implement and compare:
- Linear annealing
- Cyclical annealing
- No annealing

Measure final ELBO and reconstruction quality.

---

## What's Next

The next section covers β-VAE in detail, explaining how the β parameter controls disentanglement.
