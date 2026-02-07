# WGAN-GP: Wasserstein GAN with Gradient Penalty

WGAN-GP (Gulrajani et al., 2017) improves upon the original WGAN by replacing weight clipping with a gradient penalty to enforce the Lipschitz constraint, providing more stable training and better sample quality.

## Motivation

The original WGAN enforced the Lipschitz constraint via weight clipping:

```python
def clip_weights(model, clip_value=0.01):
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)
```

**Problems with weight clipping:**
- Reduces model capacity (weights concentrated at boundaries)
- Can cause vanishing or exploding gradients
- Biases network toward simple functions
- Sensitive to clip value hyperparameter

## Gradient Penalty Formulation

Instead of clipping, WGAN-GP penalizes the gradient norm of the critic:

$$\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$

where $\hat{x}$ is sampled uniformly along straight lines between real and fake samples:

$$\hat{x} = \alpha x_{\text{real}} + (1 - \alpha) x_{\text{fake}}, \quad \alpha \sim U(0, 1)$$

## Implementation

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



### Complete WGAN-GP Loss

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



### Critic Architecture

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



**Important**: WGAN-GP uses **LayerNorm** instead of BatchNorm because BatchNorm creates dependencies between samples in a batch, which interferes with the per-sample gradient penalty computation.

### Complete Training Loop

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



## R1 and R2 Regularization

## R1 Regularization

Simpler penalty on real samples only (used in StyleGAN):

$$\mathcal{L}_{R1} = \frac{\gamma}{2} \mathbb{E}_{x \sim p_{data}}[\|\nabla_x D(x)\|^2]$$

```python
def r1_regularization(D, real_samples, gamma=10):
    """R1 gradient penalty on real samples."""
    real_samples.requires_grad_(True)
    
    d_output = D(real_samples)
    
    gradients = torch.autograd.grad(
        outputs=d_output.sum(),
        inputs=real_samples,
        create_graph=True
    )[0]
    
    penalty = gamma / 2 * gradients.pow(2).sum([1, 2, 3]).mean()
    return penalty
```

## R2 Regularization

Penalty on fake samples:

```python
def r2_regularization(D, fake_samples, gamma=10):
    """R2 gradient penalty on fake samples."""
    fake_samples.requires_grad_(True)
    
    d_output = D(fake_samples)
    
    gradients = torch.autograd.grad(
        outputs=d_output.sum(),
        inputs=fake_samples,
        create_graph=True
    )[0]
    
    penalty = gamma / 2 * gradients.pow(2).sum([1, 2, 3]).mean()
    return penalty
```

## Comparison

| Method | Where Applied | Target |
|--------|---------------|--------|
| WGAN-GP | Interpolated | ||grad|| = 1 |
| R1 | Real samples | ||grad|| → 0 |
| R2 | Fake samples | ||grad|| → 0 |



## Lazy Regularization

## Lazy Regularization

Apply penalty every N steps for efficiency:

```python
def train_step(D, G, real, z, step, reg_interval=16, lambda_gp=10):
    # Standard loss every step
    fake = G(z)
    d_loss = d_loss_fn(D(real), D(fake))
    
    # Gradient penalty every reg_interval steps
    if step % reg_interval == 0:
        gp = gradient_penalty(D, real, fake) * reg_interval
        d_loss += gp
    
    return d_loss
```



## Summary

| Aspect | WGAN-GP |
|--------|---------|
| **Divergence** | Wasserstein-1 (Earth Mover's) |
| **Critic Output** | Unbounded score (no sigmoid) |
| **Lipschitz Method** | Gradient penalty on interpolated samples |
| **n_critic** | 5 (typical) |
| **λ_GP** | 10 (typical) |
| **Normalization** | LayerNorm (not BatchNorm) |
| **Optimizer** | Adam with β₁=0, β₂=0.9 |
| **Main Advantage** | Stable training, meaningful loss, no weight clipping |
