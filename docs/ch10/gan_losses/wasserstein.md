# Wasserstein Loss

The Wasserstein GAN (WGAN) replaces the Jensen-Shannon divergence with the Wasserstein distance (Earth Mover's Distance), providing more stable training and meaningful loss values.

## Motivation

### Problems with JS Divergence

The original GAN minimizes Jensen-Shannon divergence:

$$\text{JSD}(p_{\text{data}} \| p_g) = \frac{1}{2} D_{KL}(p_{\text{data}} \| m) + \frac{1}{2} D_{KL}(p_g \| m)$$

where $m = \frac{1}{2}(p_{\text{data}} + p_g)$.

**Problem**: When $p_{\text{data}}$ and $p_g$ have non-overlapping supports:

$$\text{JSD}(p_{\text{data}} \| p_g) = \log 2 \quad \text{(constant)}$$

This provides **zero gradient** to the generator—it can't learn!

### Wasserstein Distance

The Wasserstein-1 distance (Earth Mover's Distance):

$$W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$

where $\Pi(p_{\text{data}}, p_g)$ is the set of all joint distributions with marginals $p_{\text{data}}$ and $p_g$.

**Advantage**: Wasserstein distance provides meaningful gradients even when distributions don't overlap.

## Mathematical Formulation

### Kantorovich-Rubinstein Duality

The Wasserstein distance has a dual formulation:

$$W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \leq 1} \left[ \mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)] \right]$$

where the supremum is over all 1-Lipschitz functions $f$.

### WGAN Objective

The discriminator (called **critic** in WGAN) approximates this:

$$\max_D \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

subject to $D$ being 1-Lipschitz.

The generator minimizes:

$$\min_G -\mathbb{E}_{z \sim p_z}[D(G(z))]$$

## Key Differences from Standard GAN

| Aspect | Standard GAN | WGAN |
|--------|--------------|------|
| D output | Probability [0,1] | Unbounded score |
| D activation | Sigmoid | None (linear) |
| D name | Discriminator | Critic |
| Objective | max E[log D(x)] + E[log(1-D(G(z)))] | max E[D(x)] - E[D(G(z))] |
| Divergence | Jensen-Shannon | Wasserstein |
| D constraint | None | 1-Lipschitz |

## Enforcing Lipschitz Constraint

### Method 1: Weight Clipping (Original WGAN)

```python
def clip_weights(model, clip_value=0.01):
    """Clip weights to enforce Lipschitz constraint (crude method)."""
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)
```

**Issues with weight clipping**:
- Reduces model capacity
- Can cause vanishing/exploding gradients
- Biases network toward simple functions

### Method 2: Gradient Penalty (WGAN-GP)

A better approach is to penalize the gradient norm:

$$\mathcal{L}_{\text{GP}} = \lambda \mathbb{E}_{\hat{x}}\left[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2\right]$$

where $\hat{x}$ is sampled uniformly along lines between real and fake samples.

```python
def gradient_penalty(D, real_data, fake_data, device, lambda_gp=10):
    """
    Compute gradient penalty for WGAN-GP.
    
    Args:
        D: Critic network
        real_data: Real samples
        fake_data: Generated samples
        device: Computation device
        lambda_gp: Gradient penalty coefficient
    
    Returns:
        Gradient penalty loss
    """
    batch_size = real_data.size(0)
    
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_data)
    
    # Interpolated samples
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    # Critic output
    d_interpolated = D(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(batch_size, -1)
    
    # Gradient norm
    gradient_norm = gradients.norm(2, dim=1)
    
    # Penalty: (||grad|| - 1)^2
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return penalty
```

## Implementation

### WGAN Loss Class

```python
import torch
import torch.nn as nn

class WassersteinGANLoss:
    """
    Wasserstein GAN loss with gradient penalty (WGAN-GP).
    
    Critic objective: max E[D(x)] - E[D(G(z))] - λ * GP
    Generator objective: max E[D(G(z))]
    """
    
    def __init__(self, lambda_gp=10):
        self.lambda_gp = lambda_gp
    
    def critic_loss(self, D, real_data, fake_data, device):
        """
        Critic loss with gradient penalty.
        
        We minimize: -E[D(x)] + E[D(G(z))] + λ * GP
        """
        # Wasserstein distance estimate
        d_real = D(real_data)
        d_fake = D(fake_data)
        
        wasserstein_distance = d_real.mean() - d_fake.mean()
        
        # Gradient penalty
        gp = gradient_penalty(D, real_data, fake_data, device, self.lambda_gp)
        
        # Total loss (minimize negative Wasserstein + penalty)
        loss = -wasserstein_distance + gp
        
        return loss, {
            'wasserstein': wasserstein_distance.item(),
            'gp': gp.item(),
            'd_real': d_real.mean().item(),
            'd_fake': d_fake.mean().item()
        }
    
    def generator_loss(self, d_fake):
        """
        Generator loss: minimize -E[D(G(z))]
        """
        return -d_fake.mean()
```

### Critic Architecture (No Sigmoid!)

```python
class WGANCritic(nn.Module):
    """
    WGAN Critic - outputs unbounded score, not probability.
    
    Key difference: NO sigmoid activation at output.
    """
    
    def __init__(self, image_channels=1, feature_maps=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: C x 32 x 32
            nn.Conv2d(image_channels, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.LayerNorm([feature_maps * 2, 8, 8]),  # Use LayerNorm, not BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            nn.LayerNorm([feature_maps * 4, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0),
            # NO sigmoid - output is unbounded score
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)
```

**Note**: WGAN-GP uses **LayerNorm** instead of BatchNorm because BatchNorm creates dependencies between samples in a batch, which can interfere with the gradient penalty computation.

### Complete Training Loop

```python
def train_wgan_gp(G, D, dataloader, config, device):
    """Train WGAN with gradient penalty."""
    
    loss_fn = WassersteinGANLoss(lambda_gp=config['lambda_gp'])
    
    # Optimizers (Adam works well with WGAN-GP)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=config['lr'], betas=(0.0, 0.9))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=config['lr'], betas=(0.0, 0.9))
    
    n_critic = config.get('n_critic', 5)  # Critic updates per generator update
    
    for epoch in range(config['n_epochs']):
        for i, (real_data, _) in enumerate(dataloader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # ============
            # Train Critic
            # ============
            for _ in range(n_critic):
                d_optimizer.zero_grad()
                
                # Generate fake data
                z = torch.randn(batch_size, config['latent_dim'], device=device)
                fake_data = G(z).detach()
                
                # Critic loss
                d_loss, d_info = loss_fn.critic_loss(D, real_data, fake_data, device)
                d_loss.backward()
                d_optimizer.step()
            
            # ===============
            # Train Generator
            # ===============
            g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, config['latent_dim'], device=device)
            fake_data = G(z)
            d_fake = D(fake_data)
            
            g_loss = loss_fn.generator_loss(d_fake)
            g_loss.backward()
            g_optimizer.step()
            
            # Log
            if i % 100 == 0:
                print(f"Epoch [{epoch}] Batch [{i}] "
                      f"W-dist: {d_info['wasserstein']:.4f} "
                      f"GP: {d_info['gp']:.4f} "
                      f"G_loss: {g_loss.item():.4f}")
```

## Theoretical Properties

### Wasserstein Distance Properties

1. **Metric**: $W$ is a true distance metric
2. **Continuous**: Differentiable almost everywhere
3. **Weak convergence**: $W(p_n, p) \to 0$ iff $p_n \to p$ weakly

### Gradient Behavior

Unlike JSD, Wasserstein distance provides useful gradients:

$$\nabla_\theta W(p_{\text{data}}, p_{g_\theta}) = -\mathbb{E}_{z \sim p_z}[\nabla_\theta D^*(G_\theta(z))]$$

where $D^*$ is the optimal critic.

### Meaningful Loss Value

The critic loss approximates the Wasserstein distance, so:
- **Loss decreases** → distributions getting closer
- Loss correlates with sample quality (unlike standard GAN)

## Advantages

### 1. Stable Training

- No mode collapse (empirically)
- Less sensitive to architecture choices
- No need for careful balancing of G and D

### 2. Meaningful Loss

```python
def plot_wgan_loss(wasserstein_distances):
    """WGAN loss correlates with sample quality."""
    plt.figure(figsize=(10, 5))
    plt.plot(wasserstein_distances)
    plt.xlabel('Iteration')
    plt.ylabel('Wasserstein Distance Estimate')
    plt.title('WGAN Training - Loss Correlates with Quality')
    plt.grid(True, alpha=0.3)
```

### 3. No Vanishing Gradients

Even when distributions don't overlap, critic provides gradients.

## Disadvantages

### 1. Slower Training

Requires multiple critic updates per generator update (typically 5).

### 2. Gradient Penalty Overhead

Computing GP requires second-order gradients—computationally expensive.

### 3. Hyperparameter Sensitivity

$\lambda_{GP}$ needs tuning (typically 10 works).

## WGAN Variants

### Spectral Normalization (Alternative to GP)

```python
from torch.nn.utils import spectral_norm

class SNWGANCritic(nn.Module):
    """WGAN critic with spectral normalization."""
    
    def __init__(self, image_channels=1, feature_maps=64):
        super().__init__()
        
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(image_channels, feature_maps, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(feature_maps * 4, 1, 4, 1, 0)),
        )
```

## Summary

| Aspect | Value |
|--------|-------|
| **Divergence** | Wasserstein-1 (Earth Mover's) |
| **Critic Output** | Unbounded score |
| **Lipschitz Constraint** | GP or Spectral Norm |
| **n_critic** | 5 (typical) |
| **λ_GP** | 10 (typical) |
| **Optimizer** | Adam with β₁=0 |
| **Main Advantage** | Stable training, meaningful loss |

WGAN and WGAN-GP represent a significant improvement in GAN stability and have become foundational techniques in modern GAN architectures.
