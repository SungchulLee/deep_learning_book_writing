# Density Estimation and Evaluation

## Introduction

A key advantage of normalizing flows is their ability to compute **exact likelihoods**. This enables rigorous evaluation and comparison with other density estimation methods. This document covers evaluation metrics, benchmarking practices, and interpretation of results.

## Likelihood-Based Evaluation

### Log-Likelihood

The fundamental metric for density estimation:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i)$$

Higher is better (less negative). For normalizing flows:

$$\log p_\theta(x) = \log p_Z(f_\theta^{-1}(x)) + \log \left| \det \frac{\partial f_\theta^{-1}}{\partial x} \right|$$

### Computing Test Log-Likelihood

```python
def compute_test_log_likelihood(
    flow_model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """
    Compute average log-likelihood on test set.
    
    Args:
        flow_model: Trained normalizing flow
        test_loader: DataLoader for test data
        device: Computation device
    
    Returns:
        mean_ll: Mean log-likelihood
        std_ll: Standard deviation of log-likelihood
    """
    flow_model.eval()
    all_log_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            
            batch = batch.to(device)
            
            # Flatten if needed
            if batch.dim() > 2:
                batch = batch.view(batch.size(0), -1)
            
            # Compute log probability
            log_prob = flow_model.log_prob(batch)
            all_log_probs.append(log_prob.cpu())
    
    # Concatenate all batches
    all_log_probs = torch.cat(all_log_probs)
    
    mean_ll = all_log_probs.mean().item()
    std_ll = all_log_probs.std().item()
    
    return mean_ll, std_ll
```

## Bits Per Dimension (BPD)

### Definition

BPD normalizes log-likelihood by dimensionality and converts to bits:

$$\text{BPD} = -\frac{\log_2 p(x)}{d} = -\frac{\log p(x)}{d \cdot \ln 2}$$

where $d$ is the data dimensionality.

### Why BPD?

1. **Comparable Across Dimensions**: MNIST (784D) vs CIFAR (3072D)
2. **Intuitive Interpretation**: Bits needed to encode each dimension
3. **Standard Metric**: Widely used in generative modeling literature

### Computing BPD

```python
def compute_bpd(
    flow_model: nn.Module,
    test_loader: DataLoader,
    data_dim: int,
    device: str = 'cpu'
) -> float:
    """
    Compute bits per dimension on test set.
    
    Args:
        flow_model: Trained normalizing flow
        test_loader: DataLoader for test data
        data_dim: Dimensionality of data (e.g., 784 for MNIST)
        device: Computation device
    
    Returns:
        Average BPD on test set
    """
    mean_ll, _ = compute_test_log_likelihood(flow_model, test_loader, device)
    
    # Convert to bits per dimension
    # log_2(x) = log(x) / log(2)
    bpd = -mean_ll / (data_dim * np.log(2))
    
    return bpd
```

### Accounting for Dequantization

For discrete data with uniform dequantization:

$$\text{BPD}_{\text{adjusted}} = \text{BPD} - \log_2(256) = \text{BPD} - 8$$

Wait, this isn't quite right. The proper formula:

$$\text{BPD} = -\frac{\mathbb{E}[\log p(\tilde{x})]}{d \cdot \ln 2}$$

where $\tilde{x}$ is the dequantized (continuous) data. No adjustment needed if you compute BPD on the dequantized data directly.

### Benchmark BPD Values

| Dataset | Dimensions | Good BPD | State-of-the-Art |
|---------|------------|----------|------------------|
| MNIST | 784 | < 1.5 | ~0.99 |
| CIFAR-10 | 3072 | < 4.0 | ~3.28 |
| ImageNet 32×32 | 3072 | < 4.5 | ~3.98 |
| ImageNet 64×64 | 12288 | < 4.0 | ~3.81 |

## Sampling Quality Evaluation

### Visual Inspection

The most intuitive evaluation: look at samples!

```python
def visualize_samples(
    flow_model: nn.Module,
    n_samples: int = 64,
    img_shape: Tuple[int, int, int] = (1, 28, 28),
    device: str = 'cpu',
    save_path: str = 'samples.png'
):
    """
    Generate and visualize samples from the flow.
    
    Args:
        flow_model: Trained normalizing flow
        n_samples: Number of samples to generate
        img_shape: (channels, height, width)
        device: Computation device
        save_path: Where to save visualization
    """
    flow_model.eval()
    
    with torch.no_grad():
        # Sample from flow
        samples = flow_model.sample(n_samples, device=device)
        
        # Reshape to images
        samples = samples.view(n_samples, *img_shape)
        
        # Clip to valid range
        samples = torch.clamp(samples, 0, 1)
    
    # Create grid
    from torchvision.utils import make_grid
    grid = make_grid(samples, nrow=8, padding=2)
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray' if img_shape[0] == 1 else None)
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved samples to {save_path}")
```

### Temperature Scaling

Sample quality can be controlled by scaling the base distribution:

$$z \sim \mathcal{N}(0, T \cdot I)$$

- $T < 1$: Sharper, less diverse samples
- $T = 1$: Standard sampling
- $T > 1$: More diverse but potentially lower quality

```python
def sample_with_temperature(
    flow_model: nn.Module,
    n_samples: int,
    temperature: float = 1.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample with temperature scaling.
    
    Args:
        flow_model: Trained flow
        n_samples: Number of samples
        temperature: Temperature (< 1 for sharper samples)
        device: Computation device
    
    Returns:
        Generated samples
    """
    # Sample from scaled Gaussian
    z = torch.randn(n_samples, flow_model.dim, device=device) * temperature
    
    # Transform through flow
    samples, _ = flow_model.forward(z)
    
    return samples
```

## Reconstruction Quality

### Invertibility Check

Flows should perfectly reconstruct inputs:

$$x \approx f(f^{-1}(x))$$

```python
def check_invertibility(
    flow_model: nn.Module,
    test_data: torch.Tensor,
    tolerance: float = 1e-5
) -> Tuple[float, float]:
    """
    Verify flow invertibility on test data.
    
    Args:
        flow_model: Normalizing flow
        test_data: Data to test
        tolerance: Acceptable reconstruction error
    
    Returns:
        mean_error: Mean reconstruction error
        max_error: Maximum reconstruction error
    """
    flow_model.eval()
    
    with torch.no_grad():
        # Forward then inverse (or inverse then forward)
        z, _ = flow_model.inverse(test_data)
        x_reconstructed, _ = flow_model.forward(z)
        
        # Compute errors
        errors = (test_data - x_reconstructed).abs()
        mean_error = errors.mean().item()
        max_error = errors.max().item()
    
    print(f"Mean reconstruction error: {mean_error:.2e}")
    print(f"Max reconstruction error: {max_error:.2e}")
    
    if max_error > tolerance:
        print(f"WARNING: Max error exceeds tolerance {tolerance}")
    else:
        print("✓ Invertibility check passed")
    
    return mean_error, max_error
```

## Latent Space Analysis

### Latent Distribution

Check if latent codes match the base distribution:

```python
def analyze_latent_distribution(
    flow_model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
):
    """
    Analyze the distribution of latent codes.
    
    Args:
        flow_model: Trained flow
        test_loader: Test data loader
        device: Computation device
    """
    flow_model.eval()
    all_z = []
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            
            if batch.dim() > 2:
                batch = batch.view(batch.size(0), -1)
            
            z, _ = flow_model.inverse(batch)
            all_z.append(z.cpu())
    
    z = torch.cat(all_z, dim=0)
    
    # Statistics
    mean = z.mean(dim=0)
    std = z.std(dim=0)
    
    print("Latent Distribution Statistics:")
    print(f"  Mean of means: {mean.mean().item():.4f} (should be ~0)")
    print(f"  Std of means: {mean.std().item():.4f} (should be small)")
    print(f"  Mean of stds: {std.mean().item():.4f} (should be ~1)")
    print(f"  Std of stds: {std.std().item():.4f} (should be small)")
    
    # Plot histogram of a few dimensions
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    for i, ax in enumerate(axes.flat):
        if i < z.shape[1]:
            ax.hist(z[:, i].numpy(), bins=50, density=True, alpha=0.7)
            
            # Overlay standard Gaussian
            x_range = np.linspace(-4, 4, 100)
            ax.plot(x_range, np.exp(-x_range**2/2) / np.sqrt(2*np.pi), 
                   'r-', linewidth=2, label='N(0,1)')
            
            ax.set_title(f'Dimension {i}')
            ax.legend()
    
    plt.suptitle('Latent Space Distribution (vs Standard Gaussian)')
    plt.tight_layout()
    plt.savefig('latent_analysis.png', dpi=150)
    plt.close()
```

### Latent Interpolation

Interpolate between two data points in latent space:

```python
def latent_interpolation(
    flow_model: nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
    n_steps: int = 10,
    img_shape: Tuple[int, int, int] = (1, 28, 28)
) -> torch.Tensor:
    """
    Interpolate between two points in latent space.
    
    Args:
        flow_model: Trained flow
        x1, x2: Two data points
        n_steps: Number of interpolation steps
        img_shape: Shape of images
    
    Returns:
        Interpolated samples
    """
    flow_model.eval()
    
    with torch.no_grad():
        # Encode to latent space
        z1, _ = flow_model.inverse(x1.unsqueeze(0))
        z2, _ = flow_model.inverse(x2.unsqueeze(0))
        
        # Linear interpolation
        alphas = torch.linspace(0, 1, n_steps)
        z_interp = torch.stack([
            (1 - a) * z1 + a * z2 for a in alphas
        ]).squeeze(1)
        
        # Decode
        x_interp, _ = flow_model.forward(z_interp)
        x_interp = x_interp.view(n_steps, *img_shape)
    
    return x_interp
```

## Comparison with Other Models

### Model Comparison Framework

```python
def compare_models(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    data_dim: int,
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    Compare multiple models on standard metrics.
    
    Args:
        models: Dictionary of {name: model}
        test_loader: Test data loader
        data_dim: Data dimensionality
        device: Computation device
    
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Log-likelihood
        mean_ll, std_ll = compute_test_log_likelihood(model, test_loader, device)
        
        # BPD
        bpd = -mean_ll / (data_dim * np.log(2))
        
        # Invertibility
        sample_batch = next(iter(test_loader))
        if isinstance(sample_batch, (tuple, list)):
            sample_batch = sample_batch[0]
        sample_batch = sample_batch[:100].to(device)
        if sample_batch.dim() > 2:
            sample_batch = sample_batch.view(sample_batch.size(0), -1)
        
        mean_err, max_err = check_invertibility(model, sample_batch)
        
        results.append({
            'Model': name,
            'Log-Likelihood': f"{mean_ll:.2f} ± {std_ll:.2f}",
            'BPD': f"{bpd:.3f}",
            'Reconstruction Error': f"{mean_err:.2e}"
        })
    
    return pd.DataFrame(results)
```

### Typical Comparisons

| Model Type | Exact Likelihood | Efficient Sampling | Latent Space |
|------------|-----------------|-------------------|--------------|
| Normalizing Flow | ✓ | ✓ | ✓ (bijective) |
| VAE | ✗ (ELBO) | ✓ | ✓ (stochastic) |
| GAN | ✗ | ✓ | ✓ |
| Diffusion | ✗ (approx) | ✗ (slow) | ✓ |
| Autoregressive | ✓ | ✗ (slow) | ✗ |

## Diagnostic Tools

### Training Diagnostics

```python
class FlowDiagnostics:
    """Collection of diagnostic tools for flow training."""
    
    @staticmethod
    def plot_log_det_histogram(
        flow_model: nn.Module,
        data_loader: DataLoader,
        device: str = 'cpu'
    ):
        """Plot histogram of log-determinants."""
        flow_model.eval()
        all_log_dets = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(device)
                if batch.dim() > 2:
                    batch = batch.view(batch.size(0), -1)
                
                _, log_det = flow_model.inverse(batch)
                all_log_dets.append(log_det.cpu())
        
        log_dets = torch.cat(all_log_dets)
        
        plt.figure(figsize=(10, 5))
        plt.hist(log_dets.numpy(), bins=50, density=True, alpha=0.7)
        plt.axvline(log_dets.mean(), color='r', linestyle='--', 
                   label=f'Mean: {log_dets.mean():.2f}')
        plt.xlabel('Log Determinant')
        plt.ylabel('Density')
        plt.title('Distribution of Log-Determinants')
        plt.legend()
        plt.savefig('log_det_histogram.png', dpi=150)
        plt.close()
    
    @staticmethod
    def plot_gradient_norms(training_history: List[Dict]):
        """Plot gradient norms during training."""
        grad_norms = [h['grad_norm'] for h in training_history]
        
        plt.figure(figsize=(10, 5))
        plt.plot(grad_norms)
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm During Training')
        plt.yscale('log')
        plt.savefig('gradient_norms.png', dpi=150)
        plt.close()
    
    @staticmethod
    def check_numerical_stability(
        flow_model: nn.Module,
        n_samples: int = 1000,
        device: str = 'cpu'
    ):
        """Check for numerical issues in sampling."""
        flow_model.eval()
        
        with torch.no_grad():
            samples = flow_model.sample(n_samples, device=device)
        
        # Check for NaN/Inf
        nan_count = torch.isnan(samples).sum().item()
        inf_count = torch.isinf(samples).sum().item()
        
        print("Numerical Stability Check:")
        print(f"  NaN values: {nan_count}")
        print(f"  Inf values: {inf_count}")
        print(f"  Sample range: [{samples.min():.2f}, {samples.max():.2f}]")
        
        if nan_count == 0 and inf_count == 0:
            print("✓ No numerical issues detected")
        else:
            print("⚠ Numerical issues found!")
```

## Best Practices

### Evaluation Checklist

1. **Report BPD** for comparability across papers
2. **Include standard deviation** from multiple runs
3. **Use held-out test set** never seen during training
4. **Check invertibility** to verify implementation
5. **Visualize samples** for qualitative assessment
6. **Analyze latent space** to understand learned representation

### Common Mistakes

1. **Evaluating on training data**: Always use separate test set
2. **Forgetting dequantization**: Must dequantize for evaluation too
3. **Wrong dimensionality**: BPD denominator must match actual dimensions
4. **Ignoring variance**: Single run results can be misleading

## Summary

Proper evaluation of normalizing flows involves:

1. **Log-Likelihood**: The fundamental metric (exact for flows)
2. **Bits Per Dimension**: Normalized, comparable metric
3. **Sample Quality**: Visual inspection and temperature scaling
4. **Invertibility**: Verify implementation correctness
5. **Latent Analysis**: Understand learned representations

Unlike other generative models, flows provide exact likelihoods, making rigorous quantitative evaluation possible.

## References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
2. Theis, L., et al. (2016). A Note on the Evaluation of Generative Models. *ICLR*.
3. Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. *NeurIPS*.
