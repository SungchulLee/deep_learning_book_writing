# Two-Timescale Update Rule (TTUR)

TTUR uses different learning rates for the generator and discriminator, providing theoretical convergence guarantees and practical training stability.

## Motivation

Standard GAN training updates G and D with the same learning rate, but:
- D needs to track the changing G distribution
- G needs stable gradients from D
- Equal learning rates can cause oscillation

## The Solution

Use higher learning rate for D than G:

$$\alpha_D > \alpha_G$$

### Common Configuration

```python
# TTUR learning rates
g_lr = 0.0001  # Generator: slower
d_lr = 0.0004  # Discriminator: 4x faster

g_optimizer = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.0, 0.9))
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.0, 0.9))
```

## Theoretical Justification

### Convergence Guarantee

Heusel et al. (2017) showed that with TTUR and certain conditions:
- GAN training converges to a local Nash equilibrium
- D converges faster, providing stable gradients to G
- Reduces oscillation around equilibrium

### Intuition

```
Without TTUR:
D and G chase each other → oscillation

With TTUR:
D quickly adapts to G → G receives consistent gradients → stable progress
```

## Implementation

### Basic TTUR

```python
class TTURTrainer:
    def __init__(self, G, D, g_lr=1e-4, d_lr=4e-4):
        self.G = G
        self.D = D
        
        # Different learning rates
        self.g_optimizer = torch.optim.Adam(
            G.parameters(), lr=g_lr, betas=(0.0, 0.9)
        )
        self.d_optimizer = torch.optim.Adam(
            D.parameters(), lr=d_lr, betas=(0.0, 0.9)
        )
    
    def train_step(self, real_data, z):
        # Train D (with higher lr, it adapts faster)
        self.d_optimizer.zero_grad()
        d_loss = self.discriminator_loss(real_data, z)
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train G (with lower lr, more stable)
        self.g_optimizer.zero_grad()
        g_loss = self.generator_loss(z)
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), d_loss.item()
```

### With Learning Rate Scheduling

```python
def create_ttur_schedulers(g_optimizer, d_optimizer, total_steps):
    """Create learning rate schedulers maintaining TTUR ratio."""
    
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        g_optimizer, T_max=total_steps
    )
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        d_optimizer, T_max=total_steps
    )
    
    return g_scheduler, d_scheduler
```

## Recommended Configurations

### Standard TTUR

| Parameter | Value |
|-----------|-------|
| G learning rate | 1e-4 |
| D learning rate | 4e-4 |
| β₁ | 0.0 |
| β₂ | 0.9 |

### BigGAN Style

| Parameter | Value |
|-----------|-------|
| G learning rate | 1e-4 |
| D learning rate | 4e-4 |
| D steps per G step | 2 |

### StyleGAN Style

| Parameter | Value |
|-----------|-------|
| G learning rate | 2e-3 |
| D learning rate | 2e-3 |
| Note | Uses R1 regularization instead of TTUR ratio |

## Combining with Other Techniques

### TTUR + Spectral Normalization

```python
# Apply spectral norm to D
D = apply_spectral_norm(D)

# Use TTUR learning rates
g_optimizer = Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
d_optimizer = Adam(D.parameters(), lr=4e-4, betas=(0.0, 0.9))
```

### TTUR + Gradient Penalty

```python
def train_step_ttur_gp(G, D, real, z, g_opt, d_opt, lambda_gp=10):
    # D step with GP
    d_opt.zero_grad()
    fake = G(z).detach()
    d_loss = d_loss_fn(D, real, fake)
    gp = gradient_penalty(D, real, fake, lambda_gp)
    (d_loss + gp).backward()
    d_opt.step()  # Higher lr
    
    # G step
    g_opt.zero_grad()
    fake = G(z)
    g_loss = g_loss_fn(D, fake)
    g_loss.backward()
    g_opt.step()  # Lower lr
```

## Summary

| Aspect | TTUR |
|--------|------|
| G learning rate | Lower (e.g., 1e-4) |
| D learning rate | Higher (e.g., 4e-4) |
| Ratio | Typically 1:4 |
| Benefit | Convergence guarantee |
| Used with | Adam (β₁=0, β₂=0.9) |

TTUR is a simple but effective technique that provides both theoretical convergence guarantees and practical training improvements.
