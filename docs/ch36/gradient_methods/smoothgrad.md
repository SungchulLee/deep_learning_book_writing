# SmoothGrad

## Introduction

**SmoothGrad** is a simple yet effective technique for reducing the visual noise inherent in gradient-based saliency maps. The core idea is counterintuitive: by **adding noise** to the input and averaging the resulting gradients, we obtain **sharper, cleaner** visualizations.

Introduced by Smilkov et al. (2017), SmoothGrad addresses one of the main practical limitations of vanilla gradient saliency—the noisy, speckled appearance that makes interpretation difficult.

## Mathematical Foundation

### The SmoothGrad Formula

Given an input $\mathbf{x}$, model $f$, and target class $c$, the SmoothGrad saliency is:

$$
\text{SG}(\mathbf{x}) = \frac{1}{n} \sum_{k=1}^{n} \frac{\partial f_c(\mathbf{x} + \boldsymbol{\epsilon}_k)}{\partial \mathbf{x}}
$$

where:

- $n$ is the number of noisy samples
- $\boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ is Gaussian noise
- $\sigma$ controls the noise level (standard deviation)

### Why Adding Noise Reduces Noise

The apparent paradox resolves when we consider what the noise does:

**Observation 1: Gradients are locally unstable**

Neural networks have high-frequency components in their loss landscape. Small input perturbations can dramatically change gradients at specific pixels.

**Observation 2: True importance is stable**

If a pixel is truly important for the prediction, it will consistently produce high gradients across slight input variations. Unimportant pixels with spuriously high gradients will average out.

**Intuition:** SmoothGrad performs a form of **local averaging in gradient space**, preserving consistent signals while canceling noise.

### Connection to Gradient Smoothing

SmoothGrad can be viewed as approximating the expected gradient:

$$
\text{SG}(\mathbf{x}) \approx \mathbb{E}_{\boldsymbol{\epsilon}}[\nabla_{\mathbf{x}} f_c(\mathbf{x} + \boldsymbol{\epsilon})]
$$

This is equivalent to computing the gradient of a **smoothed version** of the model:

$$
\tilde{f}_c(\mathbf{x}) = \mathbb{E}_{\boldsymbol{\epsilon}}[f_c(\mathbf{x} + \boldsymbol{\epsilon})] = \int f_c(\mathbf{x} + \boldsymbol{\epsilon}) p(\boldsymbol{\epsilon}) d\boldsymbol{\epsilon}
$$

Under certain conditions:

$$
\nabla_{\mathbf{x}} \tilde{f}_c(\mathbf{x}) = \mathbb{E}_{\boldsymbol{\epsilon}}[\nabla_{\mathbf{x}} f_c(\mathbf{x} + \boldsymbol{\epsilon})] = \text{SG}(\mathbf{x})
$$

This is the gradient of the model convolved with a Gaussian kernel—hence the smoothing effect.

## PyTorch Implementation

### Basic Implementation

```python
import torch
import torch.nn as nn
import numpy as np

def compute_smoothgrad(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    n_samples: int = 50,
    noise_level: float = 0.15
) -> torch.Tensor:
    """
    Compute SmoothGrad saliency map.
    
    Args:
        model: Neural network in eval mode
        image_tensor: Input image [1, C, H, W]
        target_class: Target class index
        device: Computation device
        n_samples: Number of noisy samples to average
        noise_level: Standard deviation of Gaussian noise
                    (as fraction of input range, typically 0.1-0.2)
    
    Returns:
        Saliency map [1, H, W]
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # Standard deviation in input space
    # Input is typically normalized, so noise_level ~0.15 is reasonable
    stdev = noise_level * (image_tensor.max() - image_tensor.min())
    
    # Accumulate gradients
    accumulated_gradients = torch.zeros_like(image_tensor)
    
    for _ in range(n_samples):
        # Add Gaussian noise
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = image_tensor + noise
        noisy_input.requires_grad_(True)
        
        # Forward pass
        output = model(noisy_input)
        target_score = output[0, target_class]
        
        # Backward pass
        model.zero_grad()
        target_score.backward()
        
        # Accumulate
        accumulated_gradients += noisy_input.grad
    
    # Average gradients
    avg_gradients = accumulated_gradients / n_samples
    
    # Take absolute value and aggregate across channels
    saliency = torch.abs(avg_gradients).max(dim=1)[0]
    
    return saliency
```

### Batched Implementation (Efficient)

```python
def compute_smoothgrad_batched(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    n_samples: int = 50,
    noise_level: float = 0.15,
    batch_size: int = 10
) -> torch.Tensor:
    """
    Batch-optimized SmoothGrad computation.
    
    Processes multiple noisy samples in parallel for efficiency.
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    stdev = noise_level * (image_tensor.max() - image_tensor.min())
    accumulated_gradients = torch.zeros_like(image_tensor)
    
    for i in range(0, n_samples, batch_size):
        current_batch_size = min(batch_size, n_samples - i)
        
        # Create batch of noisy inputs
        # Expand image to batch: [batch_size, C, H, W]
        batch = image_tensor.expand(current_batch_size, -1, -1, -1).clone()
        noise = torch.randn_like(batch) * stdev
        noisy_batch = batch + noise
        noisy_batch.requires_grad_(True)
        
        # Forward pass
        outputs = model(noisy_batch)  # [batch_size, num_classes]
        
        # Sum target scores for batch gradient
        target_scores = outputs[:, target_class].sum()
        
        # Backward pass
        model.zero_grad()
        target_scores.backward()
        
        # Accumulate (sum over batch dimension)
        accumulated_gradients += noisy_batch.grad.sum(dim=0, keepdim=True)
    
    # Average
    avg_gradients = accumulated_gradients / n_samples
    saliency = torch.abs(avg_gradients).max(dim=1)[0]
    
    return saliency
```

## Hyperparameters

### Noise Level (σ)

The noise level controls the trade-off between smoothing and accuracy:

| Noise Level | Effect |
|-------------|--------|
| Too low (< 0.05) | Minimal smoothing, still noisy |
| Optimal (0.10-0.20) | Good smoothing, preserves details |
| Too high (> 0.30) | Over-smoothed, loses fine details |

```python
def analyze_noise_levels(
    model, image_tensor, target_class, device,
    noise_levels=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
):
    """Compare SmoothGrad results across noise levels."""
    results = {}
    
    for noise in noise_levels:
        saliency = compute_smoothgrad(
            model, image_tensor, target_class, device,
            n_samples=50, noise_level=noise
        )
        results[noise] = saliency.cpu().numpy()
    
    return results
```

**Guidelines for noise level selection:**

- **Start with 0.15** as a default
- **Increase** if saliency is still noisy
- **Decrease** if important details are being smoothed away
- Consider the **input normalization**: if input is in [0, 1], use noise ~0.1-0.2

### Number of Samples (n)

More samples → smoother, more stable results, but higher computation cost:

| Samples | Quality | Use Case |
|---------|---------|----------|
| 10-20 | Rough | Quick exploration |
| 50 | Good | Standard usage |
| 100+ | Excellent | Publication quality |

```python
def analyze_sample_convergence(
    model, image_tensor, target_class, device,
    sample_counts=[10, 20, 30, 50, 75, 100]
):
    """Analyze how SmoothGrad converges with sample count."""
    results = {}
    
    for n in sample_counts:
        saliency = compute_smoothgrad(
            model, image_tensor, target_class, device,
            n_samples=n, noise_level=0.15
        )
        results[n] = saliency.cpu().numpy()
    
    # Compute stability (correlation with highest sample count)
    reference = results[max(sample_counts)]
    
    correlations = {}
    for n, sal in results.items():
        corr = np.corrcoef(sal.flatten(), reference.flatten())[0, 1]
        correlations[n] = corr
    
    return results, correlations
```

## Variants and Extensions

### SmoothGrad-Squared

Uses squared gradients before averaging (emphasizes high-magnitude gradients):

$$
\text{SG}^2(\mathbf{x}) = \frac{1}{n} \sum_{k=1}^{n} \left( \frac{\partial f_c(\mathbf{x} + \boldsymbol{\epsilon}_k)}{\partial \mathbf{x}} \right)^2
$$

```python
def compute_smoothgrad_squared(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    n_samples: int = 50,
    noise_level: float = 0.15
) -> torch.Tensor:
    """SmoothGrad with squared gradients."""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    stdev = noise_level * (image_tensor.max() - image_tensor.min())
    accumulated_squared = torch.zeros_like(image_tensor)
    
    for _ in range(n_samples):
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = (image_tensor + noise).requires_grad_(True)
        
        output = model(noisy_input)
        output[0, target_class].backward()
        
        # Square before accumulating
        accumulated_squared += noisy_input.grad ** 2
        model.zero_grad()
    
    avg_squared = accumulated_squared / n_samples
    saliency = torch.sqrt(avg_squared).max(dim=1)[0]
    
    return saliency
```

### VarGrad (Variance of Gradients)

Measures gradient variability rather than average:

$$
\text{VarGrad}(\mathbf{x}) = \text{Var}_{\boldsymbol{\epsilon}}\left[ \frac{\partial f_c(\mathbf{x} + \boldsymbol{\epsilon})}{\partial \mathbf{x}} \right]
$$

```python
def compute_vargrad(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    n_samples: int = 50,
    noise_level: float = 0.15
) -> torch.Tensor:
    """VarGrad: Variance of gradients over noisy samples."""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    stdev = noise_level * (image_tensor.max() - image_tensor.min())
    
    gradients_list = []
    
    for _ in range(n_samples):
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = (image_tensor + noise).requires_grad_(True)
        
        output = model(noisy_input)
        output[0, target_class].backward()
        
        gradients_list.append(noisy_input.grad.clone())
        model.zero_grad()
    
    # Stack and compute variance
    all_gradients = torch.stack(gradients_list)  # [n, 1, C, H, W]
    variance = all_gradients.var(dim=0)  # [1, C, H, W]
    
    saliency = variance.max(dim=1)[0]
    
    return saliency
```

**Interpretation:** High variance indicates pixels where the model is **uncertain** or **sensitive to small changes**.

### Combining SmoothGrad with Other Methods

SmoothGrad can be applied to enhance other attribution methods:

```python
def smooth_integrated_gradients(
    model, image_tensor, target_class, device,
    n_smooth_samples: int = 20,
    n_ig_steps: int = 50,
    noise_level: float = 0.1
):
    """
    Combine SmoothGrad with Integrated Gradients.
    
    Average IG attributions computed on noisy inputs.
    """
    accumulated = torch.zeros_like(image_tensor)
    stdev = noise_level * (image_tensor.max() - image_tensor.min())
    
    for _ in range(n_smooth_samples):
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = image_tensor + noise
        
        # Compute IG for noisy input
        ig_attr = compute_integrated_gradients(
            model, noisy_input, target_class, device,
            steps=n_ig_steps
        )
        accumulated += ig_attr
    
    return accumulated / n_smooth_samples


def smooth_gradcam(
    model, target_layer, image_tensor, target_class, device,
    n_samples: int = 20,
    noise_level: float = 0.1
):
    """
    Apply SmoothGrad principle to Grad-CAM.
    """
    gradcam = GradCAM(model, target_layer)
    
    accumulated = None
    stdev = noise_level * (image_tensor.max() - image_tensor.min())
    
    for _ in range(n_samples):
        noise = torch.randn_like(image_tensor) * stdev
        noisy_input = image_tensor + noise
        
        heatmap = gradcam(noisy_input, target_class, device)
        
        if accumulated is None:
            accumulated = heatmap
        else:
            accumulated += heatmap
    
    return accumulated / n_samples
```

## Visualization

### Comparing SmoothGrad with Vanilla Gradient

```python
import matplotlib.pyplot as plt

def visualize_smoothgrad_comparison(
    model, image_tensor, target_class, device,
    noise_level: float = 0.15,
    n_samples: int = 50
):
    """Side-by-side comparison of vanilla gradient and SmoothGrad."""
    
    # Vanilla gradient
    img = image_tensor.clone().requires_grad_(True)
    output = model(img.to(device))
    output[0, target_class].backward()
    vanilla = torch.abs(img.grad).max(dim=1)[0]
    
    # SmoothGrad
    smoothgrad = compute_smoothgrad(
        model, image_tensor, target_class, device,
        n_samples=n_samples, noise_level=noise_level
    )
    
    # Denormalize image
    image_np = denormalize_image(image_tensor)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(vanilla.squeeze().cpu().numpy(), cmap='hot')
    axes[1].set_title('Vanilla Gradient\n(noisy)')
    axes[1].axis('off')
    
    axes[2].imshow(smoothgrad.squeeze().cpu().numpy(), cmap='hot')
    axes[2].set_title(f'SmoothGrad\n(n={n_samples}, σ={noise_level})')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig
```

### Hyperparameter Sensitivity Visualization

```python
def visualize_hyperparameter_effects(
    model, image_tensor, target_class, device
):
    """Visualize effects of noise level and sample count."""
    
    noise_levels = [0.05, 0.15, 0.25]
    sample_counts = [10, 50, 100]
    
    fig, axes = plt.subplots(
        len(noise_levels), len(sample_counts) + 1,
        figsize=(4 * (len(sample_counts) + 1), 4 * len(noise_levels))
    )
    
    image_np = denormalize_image(image_tensor)
    
    for i, noise in enumerate(noise_levels):
        # Show noise level label
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'σ = {noise}' if i == 0 else '')
        axes[i, 0].set_ylabel(f'Noise: {noise}', fontsize=12)
        axes[i, 0].axis('off')
        
        for j, n_samples in enumerate(sample_counts):
            saliency = compute_smoothgrad(
                model, image_tensor, target_class, device,
                n_samples=n_samples, noise_level=noise
            )
            
            axes[i, j + 1].imshow(saliency.squeeze().cpu().numpy(), cmap='hot')
            if i == 0:
                axes[i, j + 1].set_title(f'n = {n_samples}')
            axes[i, j + 1].axis('off')
    
    plt.suptitle('SmoothGrad: Noise Level vs. Sample Count', fontsize=14)
    plt.tight_layout()
    return fig
```

## Advantages and Limitations

### Advantages

1. **Simple to implement**: Only requires sampling and averaging
2. **Works with any gradient method**: Can smooth vanilla gradients, gradient×input, etc.
3. **Computationally parallelizable**: Batched implementation is efficient
4. **Visually cleaner**: Produces more interpretable visualizations
5. **Model-agnostic**: No architecture-specific modifications needed

### Limitations

1. **No theoretical guarantees**: Unlike Integrated Gradients, doesn't satisfy formal axioms
2. **Hyperparameter sensitivity**: Results depend on noise level and sample count
3. **Computational overhead**: Requires multiple forward-backward passes
4. **May over-smooth**: Important fine-grained details can be lost
5. **Doesn't address fundamental issues**: Only improves visualization, not underlying attribution quality

## Practical Recommendations

### When to Use SmoothGrad

**Recommended:**

- Quick visualization and debugging
- When vanilla gradients are too noisy to interpret
- As a post-processing step for other gradient methods
- Presentation and demonstration purposes

**Consider alternatives:**

- When theoretical guarantees matter → Integrated Gradients
- For regional localization → Grad-CAM
- When computation budget is limited → Vanilla gradient

### Implementation Checklist

1. **Noise level**: Start with 0.15, adjust based on results
2. **Sample count**: Use 50 for standard analysis, 100+ for final results
3. **Batch processing**: Use batched implementation for efficiency
4. **Visual inspection**: Compare with vanilla gradient to verify smoothing

### Integration with Existing Pipelines

```python
class SaliencyPipeline:
    """Complete pipeline for saliency computation."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def compute(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
        method: str = 'smoothgrad',
        **kwargs
    ) -> torch.Tensor:
        """
        Compute saliency using specified method.
        
        Args:
            method: 'vanilla', 'smoothgrad', 'smoothgrad_squared', 'vargrad'
        """
        if method == 'vanilla':
            return self._vanilla_gradient(image_tensor, target_class)
        elif method == 'smoothgrad':
            return compute_smoothgrad(
                self.model, image_tensor, target_class, self.device,
                **kwargs
            )
        elif method == 'smoothgrad_squared':
            return compute_smoothgrad_squared(
                self.model, image_tensor, target_class, self.device,
                **kwargs
            )
        elif method == 'vargrad':
            return compute_vargrad(
                self.model, image_tensor, target_class, self.device,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")
```

## Summary

SmoothGrad provides a practical solution to the visual noise problem in gradient-based saliency maps.

**Key equation:**

$$
\text{SG}(\mathbf{x}) = \frac{1}{n} \sum_{k=1}^{n} \frac{\partial f_c(\mathbf{x} + \boldsymbol{\epsilon}_k)}{\partial \mathbf{x}}, \quad \boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
$$

**Key insights:**

- Adding noise cancels spurious high-frequency gradient signals
- True importance is consistent across slight input variations
- Equivalent to computing gradient of Gaussian-smoothed model

**Recommended defaults:**

- Noise level: σ = 0.15
- Sample count: n = 50

## References

1. Smilkov, D., et al. (2017). *SmoothGrad: Removing Noise by Adding Noise*. arXiv:1706.03825.

2. Adebayo, J., et al. (2018). *Sanity Checks for Saliency Maps*. NeurIPS.

3. Hooker, S., et al. (2019). *A Benchmark for Interpretability Methods in Deep Neural Networks*. NeurIPS.
