# Least Squares GAN

Least Squares GAN (LSGAN) replaces the cross-entropy loss with mean squared error, providing more stable gradients and often higher quality samples.

## Motivation

### Problems with BCE Loss

The standard GAN uses binary cross-entropy:

$$\mathcal{L}_D^{BCE} = -\mathbb{E}_x[\log D(x)] - \mathbb{E}_z[\log(1 - D(G(z)))]$$

**Issue**: When $D(G(z)) \approx 0$ (D is confident fake is fake), the generator gradient:

$$\frac{\partial}{\partial D} \log(1 - D) = \frac{-1}{1-D} \approx -1$$

is bounded, even though the sample might be far from real data.

### The LSGAN Solution

Replace log-likelihood with squared error:

$$\mathcal{L}_D = \frac{1}{2}\mathbb{E}_x[(D(x) - 1)^2] + \frac{1}{2}\mathbb{E}_z[D(G(z))^2]$$

Now when $D(G(z))$ is far from target (e.g., $D(G(z)) = -5$), the gradient is proportional to the distance!

## Mathematical Formulation

### Discriminator Loss

$$\mathcal{L}_D = \frac{1}{2}\mathbb{E}_{x \sim p_{\text{data}}}[(D(x) - b)^2] + \frac{1}{2}\mathbb{E}_{z \sim p_z}[(D(G(z)) - a)^2]$$

where:
- $b$ = label for real samples (typically 1)
- $a$ = label for fake samples (typically 0)

### Generator Loss

$$\mathcal{L}_G = \frac{1}{2}\mathbb{E}_{z \sim p_z}[(D(G(z)) - c)^2]$$

where $c$ = target label for generator (typically 1, wanting D to think fakes are real).

### Common Label Choices

| Configuration | a (fake) | b (real) | c (G target) |
|---------------|----------|----------|--------------|
| Standard | 0 | 1 | 1 |
| Pearson χ² | -1 | 1 | 0 |

## Theoretical Foundation

### Connection to Pearson χ² Divergence

With $a = -1$, $b = 1$, $c = 0$, LSGAN minimizes the Pearson χ² divergence:

$$\chi^2_{\text{Pearson}}(p_{\text{data}} \| p_g) = \int \frac{(p_{\text{data}}(x) - p_g(x))^2}{p_g(x)} dx$$

**Proof Sketch**: The optimal discriminator for LSGAN is:

$$D^*(x) = \frac{b \cdot p_{\text{data}}(x) + a \cdot p_g(x)}{p_{\text{data}}(x) + p_g(x)}$$

With $a = -1$, $b = 1$:

$$D^*(x) = \frac{p_{\text{data}}(x) - p_g(x)}{p_{\text{data}}(x) + p_g(x)}$$

Substituting into the generator loss leads to the Pearson χ² divergence.

## Implementation

### LSGAN Loss Class

```python
import torch
import torch.nn as nn

class LSGANLoss:
    """
    Least Squares GAN Loss.
    
    Uses MSE instead of BCE for more stable gradients.
    
    Args:
        real_label: Target for real samples (default: 1)
        fake_label: Target for fake samples (default: 0)
    """
    
    def __init__(self, real_label=1.0, fake_label=0.0):
        self.criterion = nn.MSELoss()
        self.real_label = real_label
        self.fake_label = fake_label
    
    def discriminator_loss(self, d_real, d_fake):
        """
        Discriminator loss:
        L_D = 0.5 * E[(D(x) - real_label)²] + 0.5 * E[(D(G(z)) - fake_label)²]
        """
        batch_real = d_real.size(0)
        batch_fake = d_fake.size(0)
        
        # Target labels
        real_targets = torch.full((batch_real, 1), self.real_label, 
                                  device=d_real.device, dtype=d_real.dtype)
        fake_targets = torch.full((batch_fake, 1), self.fake_label,
                                  device=d_fake.device, dtype=d_fake.dtype)
        
        # MSE losses
        real_loss = self.criterion(d_real, real_targets)
        fake_loss = self.criterion(d_fake, fake_targets)
        
        total_loss = 0.5 * (real_loss + fake_loss)
        
        return total_loss, {
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'd_real': d_real.mean().item(),
            'd_fake': d_fake.mean().item()
        }
    
    def generator_loss(self, d_fake):
        """
        Generator loss:
        L_G = 0.5 * E[(D(G(z)) - real_label)²]
        
        Generator wants D to output real_label for fake samples.
        """
        batch_size = d_fake.size(0)
        
        # Target: we want D(G(z)) = real_label
        real_targets = torch.full((batch_size, 1), self.real_label,
                                  device=d_fake.device, dtype=d_fake.dtype)
        
        loss = 0.5 * self.criterion(d_fake, real_targets)
        
        return loss
```

### Pearson χ² Configuration

```python
class PearsonLSGANLoss(LSGANLoss):
    """
    LSGAN with Pearson χ² configuration.
    
    Uses a=-1, b=1, c=0 to minimize Pearson χ² divergence.
    """
    
    def __init__(self):
        super().__init__(real_label=1.0, fake_label=-1.0)
        self.g_target = 0.0
    
    def generator_loss(self, d_fake):
        """Generator targets 0 for Pearson χ² configuration."""
        batch_size = d_fake.size(0)
        targets = torch.zeros(batch_size, 1, device=d_fake.device, dtype=d_fake.dtype)
        return 0.5 * self.criterion(d_fake, targets)
```

## Discriminator Architecture

LSGAN discriminator outputs unbounded values (no sigmoid):

```python
class LSGANDiscriminator(nn.Module):
    """
    LSGAN Discriminator - outputs unbounded score.
    
    No sigmoid activation at output layer.
    """
    
    def __init__(self, image_channels=1, feature_maps=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(image_channels, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 1x1
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0),
            # NO SIGMOID - output is unbounded for MSE loss
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)
```

## Gradient Analysis

### Gradient Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_gradients_lsgan():
    """Compare BCE vs MSE gradients."""
    
    d_gz = np.linspace(0.01, 0.99, 100)  # D(G(z)) values
    
    # BCE non-saturating: -log(D)
    # Gradient: -1/D
    bce_grad = -1 / d_gz
    
    # LSGAN: 0.5 * (D - 1)²
    # Gradient: (D - 1)
    lsgan_grad = d_gz - 1
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gradient magnitude
    axes[0].plot(d_gz, np.abs(bce_grad), label='BCE: |−1/D|')
    axes[0].plot(d_gz, np.abs(lsgan_grad), label='LSGAN: |D−1|')
    axes[0].set_xlabel('D(G(z))')
    axes[0].set_ylabel('Gradient Magnitude')
    axes[0].set_title('Generator Gradient Magnitude')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Gradient direction (both negative, want to increase D(G(z)))
    axes[1].plot(d_gz, bce_grad, label='BCE: −1/D')
    axes[1].plot(d_gz, lsgan_grad, label='LSGAN: D−1')
    axes[1].set_xlabel('D(G(z))')
    axes[1].set_ylabel('Gradient')
    axes[1].set_title('Generator Gradient (Raw)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig
```

### Key Insight

For LSGAN, gradient magnitude is **linear** in the distance from target:
- Far from target ($D(G(z)) \approx 0$): gradient = -1 (proportional to error)
- Close to target ($D(G(z)) \approx 1$): gradient ≈ 0

This provides stable, proportional gradients throughout training.

## Training Loop

```python
def train_lsgan(G, D, dataloader, config, device):
    """Train LSGAN."""
    
    loss_fn = LSGANLoss()
    
    g_optimizer = torch.optim.Adam(G.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    
    for epoch in range(config['n_epochs']):
        for i, (real_data, _) in enumerate(dataloader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # ==================
            # Train Discriminator
            # ==================
            d_optimizer.zero_grad()
            
            d_real = D(real_data)
            
            z = torch.randn(batch_size, config['latent_dim'], device=device)
            fake_data = G(z)
            d_fake = D(fake_data.detach())
            
            d_loss, d_info = loss_fn.discriminator_loss(d_real, d_fake)
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
            
            if i % 100 == 0:
                print(f"Epoch [{epoch}] Batch [{i}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} "
                      f"D(real): {d_info['d_real']:.4f} D(fake): {d_info['d_fake']:.4f}")
```

## Advantages

### 1. More Stable Training

- Linear gradients prevent vanishing/exploding issues
- No saturation when D is confident
- Smoother optimization landscape

### 2. Higher Quality Samples

- Penalizes samples far from decision boundary
- Encourages samples to stay near real data manifold
- Empirically produces higher quality images

### 3. Simple Implementation

- Just replace BCE with MSE
- No special constraints needed
- Works with standard architectures

## Disadvantages

### 1. Outlier Sensitivity

- MSE squares the error
- Outliers have disproportionate influence
- May be unstable with extreme values

### 2. Scale Sensitivity

- Loss scale depends on label values
- May need learning rate adjustment compared to BCE
- Different optimal hyperparameters

## Comparison Table

| Aspect | BCE (Vanilla) | LSGAN |
|--------|--------------|-------|
| **Loss Function** | Cross-entropy | Mean Squared Error |
| **Gradients** | Can saturate | Linear |
| **D Output** | Probability [0,1] | Unbounded score |
| **D Activation** | Sigmoid | None (linear) |
| **Divergence** | JS Divergence | Pearson χ² |
| **Stability** | Can be unstable | More stable |
| **Sample Quality** | Good | Often better |

## Variants

### Soft Labels LSGAN

Use soft labels for regularization:

```python
class SoftLSGANLoss:
    """LSGAN with soft labels for regularization."""
    
    def __init__(self, real_range=(0.9, 1.0), fake_range=(0.0, 0.1)):
        self.real_range = real_range
        self.fake_range = fake_range
        self.criterion = nn.MSELoss()
    
    def discriminator_loss(self, d_real, d_fake):
        batch_real = d_real.size(0)
        batch_fake = d_fake.size(0)
        device = d_real.device
        
        # Random soft labels
        real_labels = torch.empty(batch_real, 1, device=device).uniform_(*self.real_range)
        fake_labels = torch.empty(batch_fake, 1, device=device).uniform_(*self.fake_range)
        
        real_loss = self.criterion(d_real, real_labels)
        fake_loss = self.criterion(d_fake, fake_labels)
        
        return 0.5 * (real_loss + fake_loss), {}
```

### Relativistic LSGAN

Combine LSGAN with relativistic discriminator:

```python
class RelativisticLSGANLoss:
    """
    Relativistic average LSGAN.
    
    D judges how much more real the real data is compared to fake.
    """
    
    def __init__(self):
        self.criterion = nn.MSELoss()
    
    def discriminator_loss(self, d_real, d_fake):
        # Relativistic: D(x) - E[D(G(z))]
        diff_real = d_real - d_fake.mean()
        diff_fake = d_fake - d_real.mean()
        
        real_loss = self.criterion(diff_real, torch.ones_like(diff_real))
        fake_loss = self.criterion(diff_fake, -torch.ones_like(diff_fake))
        
        return 0.5 * (real_loss + fake_loss), {}
    
    def generator_loss(self, d_real, d_fake):
        # G wants D(G(z)) - E[D(x)] to be positive
        diff_fake = d_fake - d_real.mean()
        diff_real = d_real - d_fake.mean()
        
        fake_loss = self.criterion(diff_fake, torch.ones_like(diff_fake))
        real_loss = self.criterion(diff_real, -torch.ones_like(diff_real))
        
        return 0.5 * (fake_loss + real_loss)
```

## Summary

| Aspect | Value |
|--------|-------|
| **D Loss** | 0.5 × E[(D(x)−1)²] + 0.5 × E[D(G(z))²] |
| **G Loss** | 0.5 × E[(D(G(z))−1)²] |
| **D Output** | Unbounded score |
| **Gradient Type** | Linear |
| **Divergence** | Pearson χ² (with a=-1, b=1, c=0) |
| **Main Advantage** | Stable, linear gradients |

LSGAN provides a simple yet effective alternative to standard GAN losses, offering more stable training and often improved sample quality through its linear gradient behavior.
