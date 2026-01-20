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
