# Minimax Objective

The GAN objective is formulated as a **minimax game** between the generator and discriminator. Understanding this objective deeply is essential for grasping GAN theory and debugging training issues.

## The Value Function

The GAN objective, known as the **value function**, is:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

This single equation encapsulates the entire adversarial training framework.

## Decomposing the Objective

Let's break down each component:

### Term 1: Real Data Classification

$$\mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)]$$

This term measures how well D recognizes real data:

- $x \sim p_{\text{data}}(x)$: Sample from the true data distribution
- $D(x)$: Discriminator's probability estimate that $x$ is real
- $\log D(x)$: Log-probability of correct classification

**Discriminator's perspective**: Maximize this term by pushing $D(x) \to 1$

**Generator's perspective**: This term doesn't depend on G (no gradient flow)

### Term 2: Fake Data Classification

$$\mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

This term measures how well D rejects fake data:

- $z \sim p_z(z)$: Sample from the prior (typically Gaussian)
- $G(z)$: Generated sample
- $D(G(z))$: Discriminator's probability estimate that $G(z)$ is real
- $\log(1 - D(G(z)))$: Log-probability of correct rejection

**Discriminator's perspective**: Maximize by pushing $D(G(z)) \to 0$

**Generator's perspective**: Minimize by pushing $D(G(z)) \to 1$

## Game-Theoretic Interpretation

The minimax formulation defines a **two-player zero-sum game**:

### Discriminator (Maximizing Player)

D wants to maximize $V(D, G)$:

$$\max_D V(D, G) = \max_D \left[ \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))] \right]$$

This is equivalent to maximizing binary cross-entropy classification accuracy.

### Generator (Minimizing Player)

G wants to minimize $V(D, G)$:

$$\min_G V(D, G) = \min_G \mathbb{E}_{z}[\log(1 - D(G(z)))]$$

Note: Only the second term depends on G.

## Mathematical Analysis

### Optimal Discriminator

For a fixed generator G, the optimal discriminator $D^*_G(x)$ can be derived analytically.

**Theorem**: For fixed G, the optimal discriminator is:

$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

where $p_g(x)$ is the distribution induced by the generator.

**Proof**:

The discriminator's objective for a fixed G is:

$$V(D, G) = \int_x p_{\text{data}}(x) \log D(x) \, dx + \int_x p_g(x) \log(1 - D(x)) \, dx$$

For any $x$, we maximize:

$$f(D) = a \log D + b \log(1 - D)$$

where $a = p_{\text{data}}(x)$ and $b = p_g(x)$.

Taking the derivative and setting to zero:

$$\frac{df}{dD} = \frac{a}{D} - \frac{b}{1-D} = 0$$

Solving:

$$a(1-D) = bD \implies D^* = \frac{a}{a+b} = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

### Value at Optimal Discriminator

Substituting $D^*_G$ into the value function:

$$C(G) = \max_D V(D, G) = V(D^*_G, G)$$

$$C(G) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)}\right]$$

### Connection to Jensen-Shannon Divergence

The value function can be rewritten as:

$$C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)$$

where the Jensen-Shannon Divergence is:

$$\text{JSD}(P \| Q) = \frac{1}{2} D_{KL}\left(P \| \frac{P+Q}{2}\right) + \frac{1}{2} D_{KL}\left(Q \| \frac{P+Q}{2}\right)$$

**Implications**:

- JSD is symmetric and bounded: $0 \leq \text{JSD} \leq \log 2$
- $C(G) \geq -\log 4$
- Minimum achieved when $p_g = p_{\text{data}}$
- GANs minimize a divergence between distributions!

### Global Optimum

**Theorem**: The global minimum of $C(G)$ is achieved if and only if $p_g = p_{\text{data}}$. At this point:

- $C(G) = -\log 4$
- $D^*_G(x) = \frac{1}{2}$ for all $x$

## Practical Implementation

### Discriminator Loss

```python
import torch
import torch.nn as nn

def discriminator_loss(d_real, d_fake):
    """
    Standard GAN discriminator loss.
    
    Maximizing V(D,G) is equivalent to minimizing:
    L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
    
    Args:
        d_real: D(x) - discriminator output on real data
        d_fake: D(G(z)) - discriminator output on fake data
    
    Returns:
        Discriminator loss to minimize
    """
    criterion = nn.BCELoss()
    
    real_labels = torch.ones_like(d_real)
    fake_labels = torch.zeros_like(d_fake)
    
    real_loss = criterion(d_real, real_labels)  # -E[log D(x)]
    fake_loss = criterion(d_fake, fake_labels)  # -E[log(1 - D(G(z)))]
    
    return real_loss + fake_loss
```

### Generator Loss (Original)

```python
def generator_loss_original(d_fake):
    """
    Original GAN generator loss: min E[log(1 - D(G(z)))]
    
    This is the theoretically motivated loss but has saturation issues.
    """
    criterion = nn.BCELoss()
    real_labels = torch.ones_like(d_fake)
    
    # Minimize log(1 - D(G(z))) equivalent to maximize -log(1 - D(G(z)))
    # BCE with label 1 gives: -log(D(G(z)))
    # We want: log(1 - D(G(z)))
    
    fake_labels = torch.zeros_like(d_fake)
    return -criterion(d_fake, fake_labels)  # Note the negative sign
```

## Gradient Analysis

### Discriminator Gradients

For the discriminator, gradients are well-behaved:

$$\nabla_D V = \mathbb{E}_x\left[\frac{1}{D(x)} \nabla_D D(x)\right] - \mathbb{E}_z\left[\frac{1}{1-D(G(z))} \nabla_D D(G(z))\right]$$

### Generator Gradients (Original Objective)

The original generator gradient:

$$\nabla_G V = -\mathbb{E}_z\left[\frac{1}{1-D(G(z))} \nabla_G D(G(z))\right]$$

**Problem**: When D is confident (early training):
- $D(G(z)) \approx 0$
- $1 - D(G(z)) \approx 1$
- Gradient magnitude is small: $\frac{1}{1-D(G(z))} \approx 1$

This is the **saturation problem**—the generator receives weak gradients when it needs them most!

### Visualization of Gradient Saturation

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_gradient_saturation():
    """Visualize gradient saturation in original GAN loss."""
    d_gz = np.linspace(0.001, 0.999, 100)
    
    # Original loss: log(1 - D(G(z)))
    # Gradient w.r.t. D(G(z)): -1 / (1 - D(G(z)))
    original_grad = -1 / (1 - d_gz)
    
    # Non-saturating loss: -log(D(G(z)))
    # Gradient w.r.t. D(G(z)): -1 / D(G(z))
    ns_grad = -1 / d_gz
    
    plt.figure(figsize=(10, 5))
    plt.plot(d_gz, np.abs(original_grad), label='Original: |∂log(1-D)/∂D|')
    plt.plot(d_gz, np.abs(ns_grad), label='Non-saturating: |∂(-log D)/∂D|')
    plt.xlabel('D(G(z))')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Magnitude vs Discriminator Confidence')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
```

## Alternative Formulation: Non-Saturating Loss

To address saturation, the generator maximizes $\log D(G(z))$ instead of minimizing $\log(1 - D(G(z)))$:

$$\max_G \mathbb{E}_z[\log D(G(z))]$$

**Benefits**:
- Stronger gradients when G is poor
- Same fixed point as original objective
- Empirically more stable

```python
def generator_loss_nonsaturating(d_fake):
    """
    Non-saturating generator loss: max E[log D(G(z))]
    
    Equivalent to: min -E[log D(G(z))]
    Implemented as: BCE(D(G(z)), 1)
    """
    criterion = nn.BCELoss()
    real_labels = torch.ones_like(d_fake)
    return criterion(d_fake, real_labels)  # -E[log D(G(z))]
```

## Training Dynamics Under Minimax

### Equilibrium Behavior

At the Nash equilibrium:

| Condition | Value |
|-----------|-------|
| $p_g$ | $= p_{\text{data}}$ |
| $D^*(x)$ | $= 0.5$ for all $x$ |
| $V(D^*, G^*)$ | $= -\log 4 \approx -1.386$ |
| JSD$(p_{\text{data}} \| p_g)$ | $= 0$ |

### Convergence Issues

The minimax game doesn't always converge:

1. **Oscillation**: D and G chase each other without converging
2. **Mode collapse**: G finds a "winning" strategy that ignores parts of data
3. **Vanishing gradients**: Perfect D provides no learning signal

## Complete Implementation

```python
class GANTrainer:
    """Complete GAN training with minimax objective."""
    
    def __init__(self, generator, discriminator, device='cpu'):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        
    def discriminator_step(self, real_data, latent_dim):
        """Single discriminator training step."""
        batch_size = real_data.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Real data
        d_real = self.D(real_data)
        loss_real = self.criterion(d_real, real_labels)
        
        # Fake data
        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake_data = self.G(z)
        d_fake = self.D(fake_data.detach())
        loss_fake = self.criterion(d_fake, fake_labels)
        
        # Total loss: -E[log D(x)] - E[log(1 - D(G(z)))]
        d_loss = loss_real + loss_fake
        
        return d_loss, d_real.mean(), d_fake.mean()
    
    def generator_step(self, batch_size, latent_dim, use_nonsaturating=True):
        """Single generator training step."""
        real_labels = torch.ones(batch_size, 1, device=self.device)
        
        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake_data = self.G(z)
        d_fake = self.D(fake_data)
        
        if use_nonsaturating:
            # Non-saturating: max E[log D(G(z))]
            g_loss = self.criterion(d_fake, real_labels)
        else:
            # Original: min E[log(1 - D(G(z)))]
            fake_labels = torch.zeros(batch_size, 1, device=self.device)
            g_loss = -self.criterion(d_fake, fake_labels)
        
        return g_loss
```

## Summary

The minimax objective $\min_G \max_D V(D, G)$ is the theoretical foundation of GANs:

| Aspect | Description |
|--------|-------------|
| **Structure** | Two-player zero-sum game |
| **D Objective** | Maximize binary classification accuracy |
| **G Objective** | Fool discriminator |
| **Optimal D** | $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$ |
| **Divergence** | Minimizes Jensen-Shannon divergence |
| **Equilibrium** | $p_g = p_{\text{data}}$, $D^* = 0.5$ |
| **Practical Issue** | Gradient saturation |
| **Solution** | Non-saturating loss |

Understanding the minimax objective provides insight into both the power and limitations of GAN training, guiding architectural and algorithmic choices.
