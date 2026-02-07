# Training Dynamics

Understanding GAN training dynamics is crucial for successful model development. This section explores how the generator and discriminator evolve during training, common failure modes, and practical monitoring strategies.

## Training Phases

### Phase 1: Early Training

In the beginning, both networks are randomly initialized:

**Generator**:
- Produces random noise-like outputs
- Samples clearly distinguishable from real data
- High gradient signal from discriminator

**Discriminator**:
- Easily separates real from fake
- $D(x_{\text{real}}) \approx 1$
- $D(G(z)) \approx 0$
- Binary cross-entropy loss is low

```python
def analyze_early_training(D, G, real_data, latent_dim, device):
    """Analyze early training phase characteristics."""
    
    # Generate fake samples
    z = torch.randn(real_data.size(0), latent_dim, device=device)
    fake_data = G(z)
    
    with torch.no_grad():
        d_real = D(real_data).mean().item()
        d_fake = D(fake_data).mean().item()
    
    print(f"Early Training Analysis:")
    print(f"  D(real): {d_real:.4f} (expected ~1.0)")
    print(f"  D(fake): {d_fake:.4f} (expected ~0.0)")
    print(f"  D accuracy: {((d_real > 0.5).sum() + (d_fake < 0.5).sum()) / (2 * real_data.size(0)):.2%}")
    
    if d_real > 0.9 and d_fake < 0.1:
        print("  Status: Normal early training (D easily distinguishes)")
```

### Phase 2: Competitive Learning

As training progresses, both networks improve:

**Generator**:
- Learns to produce more realistic features
- $D(G(z))$ increases from 0 toward 0.5
- May discover different "modes" of the data

**Discriminator**:
- Must learn more subtle distinguishing features
- Becomes more sophisticated
- $D(x_{\text{real}})$ may decrease slightly from 1

```python
def analyze_competitive_phase(D, G, real_data, latent_dim, device):
    """Analyze competitive learning phase."""
    
    z = torch.randn(real_data.size(0), latent_dim, device=device)
    fake_data = G(z)
    
    with torch.no_grad():
        d_real = D(real_data).mean().item()
        d_fake = D(fake_data).mean().item()
    
    print(f"Competitive Phase Analysis:")
    print(f"  D(real): {d_real:.4f}")
    print(f"  D(fake): {d_fake:.4f}")
    
    # Healthy signs
    if 0.6 < d_real < 0.95 and 0.05 < d_fake < 0.4:
        print("  Status: Healthy competitive learning")
    elif d_real < 0.6 and d_fake > 0.4:
        print("  Status: G is winning (D struggling)")
    elif d_real > 0.95 and d_fake < 0.05:
        print("  Status: D is winning (G may have vanishing gradients)")
```

### Phase 3: Convergence (Ideal)

At convergence:

- Generator produces samples indistinguishable from real data
- $D(x) \approx 0.5$ for all $x$
- Nash equilibrium achieved

```python
def check_convergence(D, G, real_data, latent_dim, device, tolerance=0.1):
    """Check if training has converged."""
    
    z = torch.randn(real_data.size(0), latent_dim, device=device)
    fake_data = G(z)
    
    with torch.no_grad():
        d_real = D(real_data).mean().item()
        d_fake = D(fake_data).mean().item()
    
    # At Nash equilibrium: D(x) ≈ 0.5 for all x
    real_converged = abs(d_real - 0.5) < tolerance
    fake_converged = abs(d_fake - 0.5) < tolerance
    
    print(f"Convergence Check (tolerance={tolerance}):")
    print(f"  D(real): {d_real:.4f} {'✓' if real_converged else '✗'}")
    print(f"  D(fake): {d_fake:.4f} {'✓' if fake_converged else '✗'}")
    
    return real_converged and fake_converged
```

## Loss Dynamics

### Understanding GAN Losses

GAN losses behave differently from supervised learning:

```python
import matplotlib.pyplot as plt

def plot_loss_dynamics(g_losses, d_losses, d_real_history, d_fake_history):
    """Visualize GAN training dynamics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Raw losses
    axes[0, 0].plot(g_losses, label='Generator Loss', alpha=0.7)
    axes[0, 0].plot(d_losses, label='Discriminator Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Smoothed losses
    window = 100
    g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
    d_smooth = np.convolve(d_losses, np.ones(window)/window, mode='valid')
    axes[0, 1].plot(g_smooth, label='G Loss (smoothed)')
    axes[0, 1].plot(d_smooth, label='D Loss (smoothed)')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Smoothed Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Discriminator outputs
    axes[1, 0].plot(d_real_history, label='D(real)', alpha=0.7)
    axes[1, 0].plot(d_fake_history, label='D(fake)', alpha=0.7)
    axes[1, 0].axhline(y=0.5, color='black', linestyle='--', label='Equilibrium')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('D Output')
    axes[1, 0].set_title('Discriminator Outputs')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: D accuracy over time
    d_acc = [(r > 0.5).float().mean() + (1 - f > 0.5).float().mean() 
             for r, f in zip(d_real_history, d_fake_history)]
    axes[1, 1].plot(d_acc, alpha=0.7)
    axes[1, 1].axhline(y=0.5, color='black', linestyle='--', label='Random')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Discriminator Accuracy')
    axes[1, 1].set_ylim(0.4, 1.0)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

### Interpreting Loss Curves

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| D loss → 0, G loss → ∞ | D too strong | Reduce D learning rate |
| Both losses oscillate | Normal | Continue training |
| G loss decreasing steadily | G improving | Continue training |
| D loss near log(2) ≈ 0.693 | D is random guessing | May need more D capacity |
| Both losses flat | Training stuck | Check for mode collapse |

## Gradient Dynamics

### Gradient Flow Analysis

```python
def analyze_gradients(model, loss):
    """Analyze gradient magnitudes in a model."""
    loss.backward(retain_graph=True)
    
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    return grad_norms

def plot_gradient_norms(g_grad_norms, d_grad_norms, iteration):
    """Visualize gradient norms for G and D."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Generator gradients
    axes[0].bar(range(len(g_grad_norms)), list(g_grad_norms.values()))
    axes[0].set_xticks(range(len(g_grad_norms)))
    axes[0].set_xticklabels(list(g_grad_norms.keys()), rotation=45, ha='right')
    axes[0].set_ylabel('Gradient Norm')
    axes[0].set_title(f'Generator Gradients (iter {iteration})')
    axes[0].set_yscale('log')
    
    # Discriminator gradients
    axes[1].bar(range(len(d_grad_norms)), list(d_grad_norms.values()))
    axes[1].set_xticks(range(len(d_grad_norms)))
    axes[1].set_xticklabels(list(d_grad_norms.keys()), rotation=45, ha='right')
    axes[1].set_ylabel('Gradient Norm')
    axes[1].set_title(f'Discriminator Gradients (iter {iteration})')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    return fig
```

### Vanishing Gradients

When D is too confident:

```python
def detect_vanishing_gradients(D, G, z, threshold=1e-6):
    """Detect if generator is receiving vanishing gradients."""
    
    fake_data = G(z)
    d_fake = D(fake_data)
    
    # Compute generator gradient
    g_loss = -torch.log(d_fake + 1e-8).mean()  # Non-saturating loss
    g_loss.backward()
    
    total_grad_norm = 0
    for param in G.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    if total_grad_norm < threshold:
        print(f"WARNING: Vanishing gradients detected!")
        print(f"  Total gradient norm: {total_grad_norm:.2e}")
        print(f"  D(fake) mean: {d_fake.mean().item():.6f}")
        return True
    
    return False
```

## Mode Collapse Detection

### What is Mode Collapse?

Mode collapse occurs when the generator produces limited variety:

```python
def detect_mode_collapse(G, latent_dim, n_samples=1000, device='cpu'):
    """
    Detect mode collapse by analyzing sample diversity.
    
    Methods:
    1. Visual inspection
    2. Pairwise distance analysis
    3. Cluster analysis
    """
    G.eval()
    
    with torch.no_grad():
        # Generate many samples
        z = torch.randn(n_samples, latent_dim, device=device)
        samples = G(z)
        
        # Flatten samples
        flat_samples = samples.view(n_samples, -1)
        
        # Compute pairwise distances
        distances = torch.cdist(flat_samples, flat_samples)
        
        # Remove diagonal (self-distances)
        mask = ~torch.eye(n_samples, dtype=bool, device=device)
        pairwise_distances = distances[mask]
        
        # Statistics
        mean_dist = pairwise_distances.mean().item()
        min_dist = pairwise_distances.min().item()
        std_dist = pairwise_distances.std().item()
    
    G.train()
    
    print(f"Mode Collapse Analysis:")
    print(f"  Mean pairwise distance: {mean_dist:.4f}")
    print(f"  Min pairwise distance: {min_dist:.4f}")
    print(f"  Std of distances: {std_dist:.4f}")
    
    # Warning signs
    if min_dist < 0.01:
        print("  WARNING: Very similar samples detected (possible collapse)")
    if std_dist < mean_dist * 0.1:
        print("  WARNING: Low diversity (possible partial collapse)")
    
    return {
        'mean_distance': mean_dist,
        'min_distance': min_dist,
        'std_distance': std_dist
    }
```

### Visual Mode Collapse Check

```python
def visualize_mode_collapse_check(G, latent_dim, n_samples=64, device='cpu'):
    """Generate grid of samples for visual mode collapse inspection."""
    
    G.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        samples = G(z)
        
        # Denormalize
        samples = (samples + 1) / 2
    
    # Create grid
    from torchvision.utils import make_grid
    grid = make_grid(samples, nrow=8, padding=2)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title('Generated Samples - Check for Mode Collapse')
    
    G.train()
    return samples
```

## Training Stability Monitoring

### Comprehensive Monitoring Class

```python
class GANMonitor:
    """Monitor GAN training dynamics."""
    
    def __init__(self):
        self.g_losses = []
        self.d_losses = []
        self.d_real_outputs = []
        self.d_fake_outputs = []
        self.g_grad_norms = []
        self.d_grad_norms = []
    
    def log_iteration(self, g_loss, d_loss, d_real, d_fake, G, D):
        """Log metrics for one iteration."""
        self.g_losses.append(g_loss.item())
        self.d_losses.append(d_loss.item())
        self.d_real_outputs.append(d_real.mean().item())
        self.d_fake_outputs.append(d_fake.mean().item())
        
        # Compute gradient norms
        g_grad = sum(p.grad.norm().item() ** 2 for p in G.parameters() 
                    if p.grad is not None) ** 0.5
        d_grad = sum(p.grad.norm().item() ** 2 for p in D.parameters() 
                    if p.grad is not None) ** 0.5
        
        self.g_grad_norms.append(g_grad)
        self.d_grad_norms.append(d_grad)
    
    def get_health_status(self, window=100):
        """Get current training health status."""
        
        if len(self.g_losses) < window:
            return "Insufficient data"
        
        recent_d_real = np.mean(self.d_real_outputs[-window:])
        recent_d_fake = np.mean(self.d_fake_outputs[-window:])
        recent_g_grad = np.mean(self.g_grad_norms[-window:])
        
        issues = []
        
        # Check for vanishing gradients
        if recent_g_grad < 1e-6:
            issues.append("Vanishing G gradients")
        
        # Check for D dominance
        if recent_d_real > 0.95 and recent_d_fake < 0.05:
            issues.append("D too strong")
        
        # Check for G dominance
        if recent_d_real < 0.6 and recent_d_fake > 0.4:
            issues.append("D too weak")
        
        # Check for mode collapse (low variance in D outputs)
        d_fake_std = np.std(self.d_fake_outputs[-window:])
        if d_fake_std < 0.01:
            issues.append("Possible mode collapse")
        
        if not issues:
            return "Healthy"
        else:
            return f"Issues: {', '.join(issues)}"
    
    def print_summary(self, window=100):
        """Print training summary."""
        print("\n" + "="*50)
        print("GAN Training Summary")
        print("="*50)
        
        if len(self.g_losses) < window:
            print(f"Iterations: {len(self.g_losses)} (need {window} for analysis)")
            return
        
        print(f"Total iterations: {len(self.g_losses)}")
        print(f"\nRecent {window} iterations:")
        print(f"  G loss: {np.mean(self.g_losses[-window:]):.4f} ± {np.std(self.g_losses[-window:]):.4f}")
        print(f"  D loss: {np.mean(self.d_losses[-window:]):.4f} ± {np.std(self.d_losses[-window:]):.4f}")
        print(f"  D(real): {np.mean(self.d_real_outputs[-window:]):.4f}")
        print(f"  D(fake): {np.mean(self.d_fake_outputs[-window:]):.4f}")
        print(f"\nHealth status: {self.get_health_status(window)}")
```

## Practical Training Recipe

```python
def train_gan_with_monitoring(G, D, dataloader, config, device='cpu'):
    """Complete GAN training with monitoring."""
    
    # Optimizers
    g_optimizer = torch.optim.Adam(G.parameters(), lr=config['g_lr'], betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=config['d_lr'], betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    monitor = GANMonitor()
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, config['latent_dim'], device=device)
    
    for epoch in range(config['n_epochs']):
        for i, (real_data, _) in enumerate(dataloader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # Labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # ==================
            # Train Discriminator
            # ==================
            for _ in range(config['d_steps']):
                d_optimizer.zero_grad()
                
                d_real = D(real_data)
                d_loss_real = criterion(d_real, real_labels)
                
                z = torch.randn(batch_size, config['latent_dim'], device=device)
                fake_data = G(z)
                d_fake = D(fake_data.detach())
                d_loss_fake = criterion(d_fake, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
            
            # ===============
            # Train Generator
            # ===============
            g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, config['latent_dim'], device=device)
            fake_data = G(z)
            d_fake = D(fake_data)
            
            g_loss = criterion(d_fake, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            # Monitor
            monitor.log_iteration(g_loss, d_loss, d_real, d_fake, G, D)
            
            # Print progress
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{config['n_epochs']}] "
                      f"Batch [{i}/{len(dataloader)}] "
                      f"D: {d_loss.item():.4f} G: {g_loss.item():.4f} "
                      f"D(real): {d_real.mean():.4f} D(fake): {d_fake.mean():.4f}")
        
        # End of epoch
        monitor.print_summary()
        
        # Save samples
        if (epoch + 1) % config['save_interval'] == 0:
            visualize_mode_collapse_check(G, config['latent_dim'], device=device)
    
    return G, D, monitor
```

## Summary

| Phase | D(real) | D(fake) | Characteristics |
|-------|---------|---------|-----------------|
| Early | ~1.0 | ~0.0 | D easily separates |
| Competitive | 0.6-0.9 | 0.1-0.4 | Both learning |
| Convergence | ~0.5 | ~0.5 | Nash equilibrium |
| Mode Collapse | varies | ~const | Low sample diversity |
| D Dominance | ~1.0 | ~0.0 | G vanishing gradients |

Understanding training dynamics helps diagnose problems and guides interventions for successful GAN training.

## Visualization and Monitoring Utilities

The following utilities provide essential tools for monitoring GAN training progress, generating sample visualizations, and creating training progress videos.

## Visualization Utilities

### Sample Grid Generation

```python
"""
visualization.py

Utilities for visualizing GAN training progress.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image


def denormalize(tensor, mean=0.5, std=0.5):
    """
    Denormalize tensor from [-1, 1] to [0, 1] range.
    
    Args:
        tensor: Normalized tensor
        mean: Normalization mean (default 0.5)
        std: Normalization std (default 0.5)
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    return tensor * std + mean


def tensor_to_image(tensor):
    """
    Convert PyTorch tensor to numpy image array.
    
    Args:
        tensor: Image tensor (C, H, W) or (H, W)
        
    Returns:
        Numpy array suitable for matplotlib display
    """
    image = tensor.cpu().detach().numpy()
    
    if image.ndim == 3:
        # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))
        
        # Squeeze if single channel
        if image.shape[-1] == 1:
            image = image.squeeze(-1)
    
    return image


def show_sample_grid(generator, latent_dim, n_samples=64, nrow=8, device='cpu'):
    """
    Generate and display a grid of samples from the generator.
    
    Args:
        generator: Generator model
        latent_dim: Dimension of latent space
        n_samples: Number of samples to generate
        nrow: Number of images per row in grid
        device: Device to generate on
    """
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        
        # Handle 4D latent input (for DCGANs)
        if hasattr(generator, 'main') and isinstance(generator.main[0], torch.nn.ConvTranspose2d):
            z = z.view(n_samples, latent_dim, 1, 1)
        
        samples = generator(z)
        
        # Denormalize
        samples = denormalize(samples)
        
        # Create grid
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(tensor_to_image(grid), cmap='gray' if samples.size(1) == 1 else None)
    plt.axis('off')
    plt.title(f'Generated Samples ({n_samples} images)')
    plt.tight_layout()
    plt.show()
    
    generator.train()


def show_interpolation(generator, latent_dim, n_steps=10, device='cpu'):
    """
    Show interpolation between two random latent vectors.
    
    Args:
        generator: Generator model
        latent_dim: Dimension of latent space
        n_steps: Number of interpolation steps
        device: Device to generate on
    """
    generator.eval()
    
    with torch.no_grad():
        # Sample two random latent vectors
        z1 = torch.randn(1, latent_dim, device=device)
        z2 = torch.randn(1, latent_dim, device=device)
        
        # Linear interpolation
        alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, 1)
        z_interp = (1 - alphas) * z1 + alphas * z2
        
        # Handle 4D latent input
        if hasattr(generator, 'main') and isinstance(generator.main[0], torch.nn.ConvTranspose2d):
            z_interp = z_interp.view(n_steps, latent_dim, 1, 1)
        
        samples = generator(z_interp)
        samples = denormalize(samples)
    
    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2))
    
    for i, ax in enumerate(axes):
        img = tensor_to_image(samples[i])
        ax.imshow(img, cmap='gray' if samples.size(1) == 1 else None)
        ax.axis('off')
        ax.set_title(f'α={alphas[i].item():.2f}')
    
    plt.suptitle('Latent Space Interpolation')
    plt.tight_layout()
    plt.show()
    
    generator.train()


def show_attribute_manipulation(generator, latent_dim, direction, scales, device='cpu'):
    """
    Show manipulation of generated images along a learned direction.
    
    Args:
        generator: Generator model
        latent_dim: Dimension of latent space
        direction: Learned direction vector in latent space
        scales: List of scale factors to apply
        device: Device to generate on
    """
    generator.eval()
    
    with torch.no_grad():
        # Base latent vector
        z_base = torch.randn(1, latent_dim, device=device)
        direction = direction.to(device)
        
        samples = []
        for scale in scales:
            z = z_base + scale * direction
            
            if hasattr(generator, 'main') and isinstance(generator.main[0], torch.nn.ConvTranspose2d):
                z = z.view(1, latent_dim, 1, 1)
            
            sample = generator(z)
            samples.append(sample)
        
        samples = torch.cat(samples, dim=0)
        samples = denormalize(samples)
    
    fig, axes = plt.subplots(1, len(scales), figsize=(2 * len(scales), 2))
    
    for i, (ax, scale) in enumerate(zip(axes, scales)):
        img = tensor_to_image(samples[i])
        ax.imshow(img, cmap='gray' if samples.size(1) == 1 else None)
        ax.axis('off')
        ax.set_title(f'scale={scale:.1f}')
    
    plt.suptitle('Attribute Manipulation')
    plt.tight_layout()
    plt.show()
    
    generator.train()
```

### Training Progress Monitoring

```python
"""
monitoring.py

Real-time monitoring of GAN training metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class GANTrainingMonitor:
    """
    Monitor and visualize GAN training progress.
    
    Tracks:
    - Generator and Discriminator losses
    - D outputs on real and fake images
    - Gradient norms
    - Training health indicators
    """
    
    def __init__(self, window_size=1000):
        """
        Args:
            window_size: Number of recent iterations to keep in history
        """
        self.window_size = window_size
        
        # Loss histories
        self.g_losses = deque(maxlen=window_size)
        self.d_losses = deque(maxlen=window_size)
        
        # Discriminator output histories
        self.d_real = deque(maxlen=window_size)
        self.d_fake = deque(maxlen=window_size)
        
        # Gradient norm histories
        self.g_grad_norms = deque(maxlen=window_size)
        self.d_grad_norms = deque(maxlen=window_size)
        
        # Iteration counter
        self.iteration = 0
    
    def log(self, g_loss, d_loss, d_real_out, d_fake_out, G=None, D=None):
        """
        Log metrics for one training iteration.
        
        Args:
            g_loss: Generator loss
            d_loss: Discriminator loss
            d_real_out: D output on real images (batch)
            d_fake_out: D output on fake images (batch)
            G: Generator model (optional, for gradient tracking)
            D: Discriminator model (optional, for gradient tracking)
        """
        self.g_losses.append(g_loss.item() if hasattr(g_loss, 'item') else g_loss)
        self.d_losses.append(d_loss.item() if hasattr(d_loss, 'item') else d_loss)
        
        self.d_real.append(d_real_out.mean().item() if hasattr(d_real_out, 'mean') else d_real_out)
        self.d_fake.append(d_fake_out.mean().item() if hasattr(d_fake_out, 'mean') else d_fake_out)
        
        # Compute gradient norms if models provided
        if G is not None:
            g_grad = self._compute_grad_norm(G)
            self.g_grad_norms.append(g_grad)
        
        if D is not None:
            d_grad = self._compute_grad_norm(D)
            self.d_grad_norms.append(d_grad)
        
        self.iteration += 1
    
    def _compute_grad_norm(self, model):
        """Compute total gradient norm for a model."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def get_health_status(self):
        """
        Assess training health based on recent metrics.
        
        Returns:
            Dictionary with health indicators
        """
        if len(self.g_losses) < 100:
            return {'status': 'initializing', 'issues': []}
        
        recent_d_real = np.mean(list(self.d_real)[-100:])
        recent_d_fake = np.mean(list(self.d_fake)[-100:])
        recent_g_loss = np.mean(list(self.g_losses)[-100:])
        
        issues = []
        
        # Check for D dominance
        if recent_d_real > 0.95 and recent_d_fake < 0.05:
            issues.append('D_DOMINANCE: Discriminator too strong')
        
        # Check for G dominance
        if recent_d_real < 0.55 and recent_d_fake > 0.45:
            issues.append('G_DOMINANCE: Discriminator too weak')
        
        # Check for mode collapse (low variance in D(fake))
        d_fake_std = np.std(list(self.d_fake)[-100:])
        if d_fake_std < 0.01:
            issues.append('MODE_COLLAPSE: Possible mode collapse detected')
        
        # Check for vanishing gradients
        if len(self.g_grad_norms) >= 100:
            recent_g_grad = np.mean(list(self.g_grad_norms)[-100:])
            if recent_g_grad < 1e-6:
                issues.append('VANISHING_GRAD: Generator gradients near zero')
        
        # Check for equilibrium
        near_equilibrium = abs(recent_d_real - 0.5) < 0.15 and abs(recent_d_fake - 0.5) < 0.15
        
        status = 'healthy' if not issues else 'issues_detected'
        if near_equilibrium and not issues:
            status = 'near_equilibrium'
        
        return {
            'status': status,
            'issues': issues,
            'd_real': recent_d_real,
            'd_fake': recent_d_fake,
            'g_loss': recent_g_loss
        }
    
    def plot_training_curves(self, save_path=None):
        """
        Plot comprehensive training curves.
        
        Args:
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = range(len(self.g_losses))
        
        # Plot 1: Raw losses
        axes[0, 0].plot(iterations, list(self.g_losses), label='G Loss', alpha=0.7)
        axes[0, 0].plot(iterations, list(self.d_losses), label='D Loss', alpha=0.7)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Smoothed losses
        window = min(100, len(self.g_losses) // 4)
        if window > 0:
            g_smooth = np.convolve(list(self.g_losses), np.ones(window)/window, mode='valid')
            d_smooth = np.convolve(list(self.d_losses), np.ones(window)/window, mode='valid')
            axes[0, 1].plot(g_smooth, label='G Loss (smoothed)')
            axes[0, 1].plot(d_smooth, label='D Loss (smoothed)')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Smoothed Losses')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Discriminator outputs
        axes[1, 0].plot(iterations, list(self.d_real), label='D(real)', alpha=0.7)
        axes[1, 0].plot(iterations, list(self.d_fake), label='D(fake)', alpha=0.7)
        axes[1, 0].axhline(y=0.5, color='black', linestyle='--', label='Equilibrium')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('D Output')
        axes[1, 0].set_title('Discriminator Outputs')
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Gradient norms
        if len(self.g_grad_norms) > 0:
            axes[1, 1].plot(list(self.g_grad_norms), label='G Gradients', alpha=0.7)
        if len(self.d_grad_norms) > 0:
            axes[1, 1].plot(list(self.d_grad_norms), label='D Gradients', alpha=0.7)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_title('Gradient Norms')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def print_summary(self):
        """Print training summary."""
        health = self.get_health_status()
        
        print("\n" + "="*60)
        print("GAN TRAINING SUMMARY")
        print("="*60)
        print(f"Total iterations: {self.iteration}")
        print(f"Status: {health['status'].upper()}")
        
        if health.get('issues'):
            print("\nIssues detected:")
            for issue in health['issues']:
                print(f"  ⚠️  {issue}")
        
        print(f"\nRecent metrics (last 100 iterations):")
        print(f"  D(real): {health.get('d_real', 'N/A'):.4f}")
        print(f"  D(fake): {health.get('d_fake', 'N/A'):.4f}")
        print(f"  G loss:  {health.get('g_loss', 'N/A'):.4f}")
        print("="*60)
```

### Training Video Generation

```python
"""
video_utils.py

Utilities for creating training progress videos.
"""

import os
import cv2
import imageio.v2 as imageio
import numpy as np


def create_training_video(image_folder, output_path, fps=24, 
                          target_size=(512, 512), codec='mp4v'):
    """
    Create a video from training progress images.
    
    Args:
        image_folder: Folder containing PNG images
        output_path: Output video file path
        fps: Frames per second
        target_size: (width, height) for video frames
        codec: Video codec (mp4v, XVID, etc.)
    """
    # Get sorted list of image files
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))],
        key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0])) or 0)
    )
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    print(f"Creating video from {len(image_files)} images...")
    
    # Read first image to get dimensions
    first_img = cv2.imread(os.path.join(image_folder, image_files[0]))
    
    # Create video writer
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    for i, filename in enumerate(image_files):
        filepath = os.path.join(image_folder, filename)
        img = cv2.imread(filepath)
        
        if img is None:
            print(f"Warning: Could not read {filepath}")
            continue
        
        # Resize to target size
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        out.write(img_resized)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images")
    
    out.release()
    print(f"Video saved to {output_path}")


def create_gif(image_folder, output_path, fps=10, target_size=(256, 256)):
    """
    Create an animated GIF from training progress images.
    
    Args:
        image_folder: Folder containing images
        output_path: Output GIF file path
        fps: Frames per second
        target_size: (width, height) for GIF frames
    """
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))],
        key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0])) or 0)
    )
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return
    
    print(f"Creating GIF from {len(image_files)} images...")
    
    images = []
    for filename in image_files:
        filepath = os.path.join(image_folder, filename)
        img = imageio.imread(filepath)
        
        # Resize
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        images.append(img_resized)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    imageio.mimsave(output_path, images, fps=fps)
    print(f"GIF saved to {output_path}")


def create_interpolation_video(generator, latent_dim, output_path, 
                               n_frames=120, fps=24, image_size=256, device='cpu'):
    """
    Create a video showing smooth interpolation through latent space.
    
    Args:
        generator: Trained generator model
        latent_dim: Dimension of latent space
        output_path: Output video path
        n_frames: Number of frames
        fps: Frames per second
        image_size: Size of output frames
        device: Device to generate on
    """
    import torch
    from torchvision.utils import make_grid
    
    generator.eval()
    
    # Create circular path through latent space
    n_waypoints = 5
    waypoints = [torch.randn(1, latent_dim, device=device) for _ in range(n_waypoints)]
    waypoints.append(waypoints[0])  # Close the loop
    
    frames = []
    frames_per_segment = n_frames // n_waypoints
    
    with torch.no_grad():
        for i in range(n_waypoints):
            z_start = waypoints[i]
            z_end = waypoints[i + 1]
            
            for j in range(frames_per_segment):
                alpha = j / frames_per_segment
                z = (1 - alpha) * z_start + alpha * z_end
                
                # Handle 4D input for DCGANs
                if hasattr(generator, 'main') and isinstance(generator.main[0], torch.nn.ConvTranspose2d):
                    z = z.view(1, latent_dim, 1, 1)
                
                sample = generator(z)
                sample = (sample + 1) / 2  # Denormalize
                
                # Convert to numpy image
                img = sample.squeeze().cpu().numpy()
                if img.ndim == 3:
                    img = np.transpose(img, (1, 2, 0))
                
                img = (img * 255).astype(np.uint8)
                
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                img = cv2.resize(img, (image_size, image_size))
                frames.append(img)
    
    # Write video
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (image_size, image_size))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Interpolation video saved to {output_path}")
    
    generator.train()
```

## Usage Example

```python
"""
Example usage of training utilities.
"""

import torch
from model import Generator, Discriminator
from monitoring import GANTrainingMonitor
from visualization import show_sample_grid, show_interpolation
from video_utils import create_training_video, create_interpolation_video

# Initialize models
latent_dim = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

# Initialize monitor
monitor = GANTrainingMonitor(window_size=1000)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (real_imgs, _) in enumerate(dataloader):
        # ... training code ...
        
        # Log metrics
        monitor.log(
            g_loss=g_loss,
            d_loss=d_loss,
            d_real_out=d_real,
            d_fake_out=d_fake,
            G=G,
            D=D
        )
        
        # Print status periodically
        if batch_idx % 100 == 0:
            monitor.print_summary()

# After training
monitor.plot_training_curves(save_path='training_curves.png')

# Visualize results
show_sample_grid(G, latent_dim, n_samples=64, device=device)
show_interpolation(G, latent_dim, n_steps=10, device=device)

# Create videos
create_training_video('images/', 'videos/training.mp4')
create_interpolation_video(G, latent_dim, 'videos/interpolation.mp4', device=device)
```

## Summary

| Utility | Purpose |
|---------|---------|
| `show_sample_grid` | Display generator samples in a grid |
| `show_interpolation` | Visualize latent space smoothness |
| `GANTrainingMonitor` | Track losses, D outputs, gradients |
| `plot_training_curves` | Visualize training progress |
| `create_training_video` | Compile training images into video |
| `create_interpolation_video` | Generate latent traversal video |

These utilities provide essential tools for understanding and debugging GAN training, making it easier to identify issues and produce compelling visualizations.
