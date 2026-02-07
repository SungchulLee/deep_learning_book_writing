# Integrated Gradients

## Introduction

**Integrated Gradients (IG)** is a principled attribution method that addresses fundamental limitations of vanilla gradient-based approaches. Unlike simple gradients that provide local sensitivity, Integrated Gradients computes attributions by **accumulating gradients along a path from a baseline to the input**.

The method was introduced by Sundararajan, Taly, and Yan (2017) with the explicit goal of creating an attribution method that satisfies **sensitivity** and **implementation invariance**—two properties that vanilla gradients fail to satisfy. It is notable for being the only widely-used method that satisfies several desirable theoretical axioms simultaneously.

## Motivation

### Problems with Vanilla Gradients

Consider a simple ReLU network: $f(x) = \max(0, x - 1)$

- For $x = 2$: $f(2) = 1$, $\nabla f(2) = 1$ (gradient exists)
- For $x = 0$: $f(0) = 0$, $\nabla f(0) = 0$ (gradient is zero)

Now consider the attribution question: **"Why is $f(2) = 1$?"**

The vanilla gradient says $x$ has importance 1. But what if we used a baseline of $x' = 0$?

The difference $f(2) - f(0) = 1 - 0 = 1$ should be fully attributed to the input, yet the gradient alone doesn't capture the full contribution—it only measures **local sensitivity** at the input point.

**More critically:** For $x = 0.5$ (just below the threshold), we have $f(0.5) = 0$ and $\nabla f(0.5) = 0$. The gradient claims this input has zero importance, even though the input value directly determines whether the output is zero!

### The Saturation Problem

Neural networks with ReLU, sigmoid, or tanh activations have regions where gradients are zero or nearly zero (saturation). In these regions:

- Vanilla gradients claim features have zero importance
- Yet these features clearly affect the output

### The Path Integration Solution

Integrated Gradients solves this by integrating gradients along the **entire path** from baseline to input:

$$
\text{IG}_i(\mathbf{x}) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} \, d\alpha
$$

This accumulates all gradient information along the interpolation path, capturing contributions even through saturated regions.

## Mathematical Foundation

### The Path Integral Formulation

For an input $\mathbf{x} \in \mathbb{R}^n$, a baseline $\mathbf{x}' \in \mathbb{R}^n$, and a model $f: \mathbb{R}^n \rightarrow \mathbb{R}$, the Integrated Gradients attribution for feature $i$ is:

$$
\text{IG}_i(\mathbf{x}) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} \, d\alpha
$$

where:

- $\mathbf{x}'$ is the baseline (reference point, often zeros or a blurred image)
- $\alpha \in [0, 1]$ parameterizes the straight-line path from $\mathbf{x}'$ to $\mathbf{x}$
- $\frac{\partial f}{\partial x_i}$ is the gradient with respect to the $i$-th input feature
- $(x_i - x'_i)$ scales the integrated gradient by how far feature $i$ traveled

### Path Parameterization

The path from baseline to input is:

$$
\gamma(\alpha) = \mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'), \quad \alpha \in [0, 1]
$$

- At $\alpha = 0$: $\gamma(0) = \mathbf{x}'$ (baseline)
- At $\alpha = 1$: $\gamma(1) = \mathbf{x}$ (input)

At each point along this path, we compute gradients. The integral accumulates these gradients, weighted by how far each feature travels from baseline to input.

### Riemann Sum Approximation

In practice, the integral is approximated using a Riemann sum with $m$ steps:

$$
\text{IG}_i(\mathbf{x}) \approx (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial f\left(\mathbf{x}' + \frac{k}{m}(\mathbf{x} - \mathbf{x}')\right)}{\partial x_i}
$$

This requires $m$ forward-backward passes through the network (though these can be batched for efficiency).

## Axiomatic Properties

Integrated Gradients satisfies fundamental axioms that Sundararajan et al. argue any attribution method should satisfy. **It is the unique method satisfying both sensitivity and implementation invariance** (among path-based methods with the straight-line path).

### Axiom 1: Sensitivity

**Statement:** If the input and baseline differ in exactly one feature and the model outputs differ, then that feature must receive non-zero attribution.

**Formally:** If $f(\mathbf{x}) \neq f(\mathbf{x}')$ and $x_i \neq x'_i$ while $x_j = x'_j$ for all $j \neq i$, then $\text{IG}_i(\mathbf{x}) \neq 0$.

**Why vanilla gradients fail:** Consider $f(x) = \text{ReLU}(x - 1)$ with $x = 2$ and $x' = 0$. If we evaluate at $x = 0.5$, the gradient is zero (below threshold), yet $x$ clearly matters for the output.

**Why IG satisfies this:** By integrating along the path, we capture the gradient at points where it *is* non-zero, even if it's zero at the endpoints.

### Axiom 2: Implementation Invariance

**Statement:** Two networks that produce identical outputs for all inputs should have identical attributions, regardless of internal architecture.

**Formally:** If $f(\mathbf{x}) = g(\mathbf{x})$ for all $\mathbf{x}$, then $\text{IG}^f_i(\mathbf{x}) = \text{IG}^g_i(\mathbf{x})$.

**Why this matters:** Different implementations (e.g., using $\sigma(x)$ vs. $1 - \sigma(-x)$ for sigmoid) should not change attributions since the function is mathematically identical.

**Why DeepLIFT violates this:** DeepLIFT propagates contributions through the computational graph, so different graph structures (even for identical functions) can produce different attributions.

### Derived Property: Completeness

An important consequence of the definition is the **Completeness** (or **Efficiency**) property:

$$
\sum_{i=1}^{n} \text{IG}_i(\mathbf{x}) = f(\mathbf{x}) - f(\mathbf{x}')
$$

The attributions **exactly account for** the difference between the model's output on the input versus the baseline. No importance is "lost" or artificially created.

**Proof:**

Using the fundamental theorem of calculus and chain rule:

$$
\sum_i \text{IG}_i = \sum_i (x_i - x'_i) \int_0^1 \frac{\partial f}{\partial x_i} d\alpha = \int_0^1 \nabla f \cdot (\mathbf{x} - \mathbf{x}') \, d\alpha = \int_0^1 \frac{d}{d\alpha} f(\gamma(\alpha)) \, d\alpha = f(\mathbf{x}) - f(\mathbf{x}')
$$

This property is extremely valuable for interpretability—it means attributions have a **concrete meaning**: they partition the prediction difference.

## PyTorch Implementation

### Functional Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

def compute_integrated_gradients(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    baseline_type: Literal['zeros', 'blur', 'random', 'mean'] = 'zeros',
    steps: int = 50
) -> torch.Tensor:
    """
    Compute Integrated Gradients attribution.
    
    Args:
        model: Neural network in eval mode
        image_tensor: Input image [1, C, H, W]
        target_class: Target class index
        device: Computation device
        baseline_type: Type of baseline ('zeros', 'blur', 'random', 'mean')
        steps: Number of interpolation steps (more = more accurate)
        
    Returns:
        Attribution map [1, H, W]
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # Create baseline
    baseline = create_baseline(image_tensor, baseline_type, device)
    
    # Difference between input and baseline
    delta = image_tensor - baseline  # (x - x')
    
    # Accumulate gradients along path
    accumulated_gradients = torch.zeros_like(image_tensor)
    
    for step in range(1, steps + 1):
        # Interpolation coefficient: α = k/m
        alpha = step / steps
        
        # Interpolated input: x' + α(x - x')
        interpolated = baseline + alpha * delta
        interpolated = interpolated.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = model(interpolated)
        target_score = output[0, target_class]
        
        # Backward pass
        model.zero_grad()
        target_score.backward()
        
        # Accumulate: ∂f/∂x at this interpolation point
        accumulated_gradients += interpolated.grad
    
    # Average gradients (Riemann sum approximation)
    avg_gradients = accumulated_gradients / steps  # (1/m) Σ_k ∇f
    
    # Scale by input difference: (x - x') * avg_gradients
    integrated_grads = delta * avg_gradients
    
    # Aggregate across channels (take absolute value for visualization)
    attribution = torch.abs(integrated_grads)
    saliency = attribution.max(dim=1)[0]  # [1, H, W]
    
    return saliency


def create_baseline(
    image_tensor: torch.Tensor,
    baseline_type: str,
    device: torch.device
) -> torch.Tensor:
    """
    Create baseline for Integrated Gradients.
    
    Args:
        image_tensor: Input image
        baseline_type: 'zeros', 'blur', 'random', or 'mean'
        device: Computation device
        
    Returns:
        Baseline tensor with same shape as input
    """
    if baseline_type == 'zeros':
        # Black image (most common choice for images)
        baseline = torch.zeros_like(image_tensor)
        
    elif baseline_type == 'blur':
        # Heavily blurred version of input (preserves low-frequency structure)
        from torchvision.transforms.functional import gaussian_blur
        baseline = gaussian_blur(image_tensor, kernel_size=51, sigma=20)
        
    elif baseline_type == 'random':
        # Random noise (uniform in [0, 0.1])
        baseline = torch.rand_like(image_tensor) * 0.1
        
    elif baseline_type == 'mean':
        # Dataset mean (for ImageNet-normalized images)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        mean = mean.view(1, 3, 1, 1).expand_as(image_tensor)
        baseline = mean
        
    elif baseline_type == 'max_entropy':
        # Maximum entropy baseline (gray for images)
        baseline = torch.ones_like(image_tensor) * 0.5
        
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    return baseline.to(device)
```

### Class-Based Implementation

```python
class IntegratedGradients:
    """
    Integrated Gradients attribution method.
    
    Reference: Sundararajan et al., "Axiomatic Attribution for 
    Deep Networks" (ICML 2017)
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: PyTorch model
        """
        self.model = model
    
    def attribute(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor = None,
        target_class: int = None,
        n_steps: int = 50,
        return_convergence_delta: bool = False
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients attributions.
        
        Args:
            input_tensor: Input tensor of shape (1, *input_dims)
            baseline: Baseline tensor of same shape. If None, uses zeros.
            target_class: Target class for attribution. If None, uses argmax.
            n_steps: Number of integration steps (higher = more accurate)
            return_convergence_delta: If True, also return approximation error
            
        Returns:
            Attribution tensor of same shape as input
        """
        self.model.eval()
        device = input_tensor.device
        
        # Default baseline: zeros (black image for images)
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Determine target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Compute difference
        diff = input_tensor - baseline
        
        # Generate interpolation steps: α_k = k/m for k = 1, ..., m
        scaled_inputs = [
            baseline + (float(i) / n_steps) * diff 
            for i in range(1, n_steps + 1)
        ]
        
        # Stack all interpolated inputs for batched computation
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad_(True)
        
        # Forward pass for all steps at once (batched)
        outputs = self.model(scaled_inputs)
        
        # Extract target class scores
        target_scores = outputs[:, target_class]
        
        # Backward pass
        self.model.zero_grad()
        
        # Compute gradients
        grads = torch.autograd.grad(
            outputs=target_scores.sum(),
            inputs=scaled_inputs,
            create_graph=False
        )[0]
        
        # Average gradients across steps
        avg_grads = grads.mean(dim=0, keepdim=True)
        
        # Integrated Gradients = (input - baseline) * avg_gradients
        attributions = diff * avg_grads
        
        if return_convergence_delta:
            # Check completeness: sum of attributions ≈ f(x) - f(x')
            with torch.no_grad():
                f_x = self.model(input_tensor)[0, target_class]
                f_baseline = self.model(baseline)[0, target_class]
                expected_diff = f_x - f_baseline
                actual_sum = attributions.sum()
                delta = (expected_diff - actual_sum).abs().item()
            return attributions, delta
        
        return attributions
    
    def attribute_with_noise(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor = None,
        target_class: int = None,
        n_steps: int = 50,
        n_samples: int = 5,
        noise_level: float = 0.1
    ) -> torch.Tensor:
        """
        Compute noise-robust Integrated Gradients (Expected Gradients).
        
        Averages IG over multiple noisy baselines for more stable attributions.
        This is related to SHAP's expected gradients formulation.
        """
        attributions = torch.zeros_like(input_tensor)
        
        for _ in range(n_samples):
            if baseline is None:
                noisy_baseline = noise_level * torch.randn_like(input_tensor)
            else:
                noisy_baseline = baseline + noise_level * torch.randn_like(baseline)
            
            attr = self.attribute(
                input_tensor, 
                noisy_baseline, 
                target_class, 
                n_steps
            )
            attributions += attr
        
        return attributions / n_samples
```

### Optimized Batched Implementation

For efficiency, process multiple interpolation steps in a single batch:

```python
def compute_integrated_gradients_batched(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    baseline_type: str = 'zeros',
    steps: int = 50,
    batch_size: int = 10
) -> torch.Tensor:
    """
    Batch-optimized Integrated Gradients computation.
    
    Processes multiple interpolation steps per batch for efficiency.
    Reduces GPU memory usage compared to processing all steps at once.
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    baseline = create_baseline(image_tensor, baseline_type, device)
    delta = image_tensor - baseline
    
    accumulated_gradients = torch.zeros_like(image_tensor)
    
    # Create alpha values: [1/m, 2/m, ..., 1]
    alphas = torch.linspace(1/steps, 1, steps, device=device)
    
    # Process in batches
    for i in range(0, steps, batch_size):
        batch_alphas = alphas[i:i+batch_size]
        current_batch_size = len(batch_alphas)
        
        # Create batch of interpolated inputs
        # Shape: [batch_size, C, H, W]
        batch_alphas = batch_alphas.view(-1, 1, 1, 1)
        interpolated_batch = baseline + batch_alphas * delta
        interpolated_batch.requires_grad_(True)
        
        # Forward pass for batch
        outputs = model(interpolated_batch)  # [batch_size, num_classes]
        
        # Sum target scores (for gradient computation)
        target_scores = outputs[:, target_class].sum()
        
        # Backward pass
        model.zero_grad()
        target_scores.backward()
        
        # Accumulate gradients (sum over batch)
        accumulated_gradients += interpolated_batch.grad.sum(dim=0, keepdim=True)
    
    # Average and scale
    avg_gradients = accumulated_gradients / steps
    integrated_grads = delta * avg_gradients
    
    # Aggregate across channels
    saliency = torch.abs(integrated_grads).max(dim=1)[0]
    
    return saliency
```

## Baseline Selection

The choice of baseline significantly affects attributions. The baseline represents a "neutral" or "absence of information" input.

### Common Baseline Choices

| Baseline | Description | Best For |
|----------|-------------|----------|
| **Zeros (Black)** | All-zero tensor | Images (most common) |
| **Blur** | Heavily blurred input | Preserving structure |
| **Random** | Uniform random noise | Ensemble averaging |
| **Mean** | Dataset mean values | Normalized inputs |
| **Max Entropy** | 0.5 (gray for images) | Maximum uncertainty |

### Baseline Comparison

```python
def compare_baselines(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    baselines: list = ['zeros', 'blur', 'random', 'mean']
):
    """Compare Integrated Gradients with different baselines."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, len(baselines) + 1, figsize=(4 * (len(baselines) + 1), 8))
    
    # Original image
    image_np = denormalize_image(image_tensor)
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    for idx, baseline_type in enumerate(baselines):
        # Compute IG
        attr = compute_integrated_gradients(
            model, image_tensor, target_class, device,
            baseline_type=baseline_type, steps=50
        )
        attr_np = attr.squeeze().cpu().numpy()
        
        # Show baseline
        baseline = create_baseline(image_tensor, baseline_type, device)
        baseline_np = denormalize_image(baseline)
        axes[0, idx + 1].imshow(baseline_np)
        axes[0, idx + 1].set_title(f'{baseline_type.capitalize()} Baseline')
        axes[0, idx + 1].axis('off')
        
        # Show attribution
        axes[1, idx + 1].imshow(attr_np, cmap='hot')
        axes[1, idx + 1].set_title('Attribution')
        axes[1, idx + 1].axis('off')
    
    plt.tight_layout()
    return fig
```

### Domain-Specific Baseline Guidelines

| Domain | Recommended Baseline | Rationale |
|--------|---------------------|-----------|
| Images (RGB) | Zeros (black) | Absence of visual information |
| Text (embeddings) | Padding token embedding | Neutral token |
| Tabular | Training set mean | Average feature values |
| Time series | Zeros or historical mean | Baseline activity level |
| Audio | Silence (zeros) | Absence of sound |

## Verifying Completeness

A critical quality check for IG is verifying the completeness property:

```python
def verify_completeness(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    baseline_type: str = 'zeros',
    steps: int = 50
) -> dict:
    """
    Verify that IG attributions sum to f(x) - f(x').
    
    Returns:
        Dictionary with completeness verification results
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    baseline = create_baseline(image_tensor, baseline_type, device)
    
    # Compute model outputs
    with torch.no_grad():
        output_input = model(image_tensor)[0, target_class].item()
        output_baseline = model(baseline)[0, target_class].item()
    
    output_difference = output_input - output_baseline
    
    # Compute IG (keep channel dimension for sum)
    ig = IntegratedGradients(model)
    attributions = ig.attribute(image_tensor, baseline, target_class, steps)
    
    attribution_sum = attributions.sum().item()
    
    # Calculate error
    absolute_error = abs(attribution_sum - output_difference)
    relative_error = absolute_error / (abs(output_difference) + 1e-8)
    
    return {
        'f(x)': output_input,
        'f(x\')': output_baseline,
        'f(x) - f(x\')': output_difference,
        'sum(IG)': attribution_sum,
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'completeness_satisfied': relative_error < 0.05  # 5% tolerance
    }
```

## Number of Steps Analysis

The number of integration steps controls approximation accuracy:

```python
def analyze_steps_convergence(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    step_counts: list = [5, 10, 20, 50, 100, 200, 500]
):
    """Analyze how attributions converge with more steps."""
    import matplotlib.pyplot as plt
    
    errors = []
    
    for steps in step_counts:
        result = verify_completeness(
            model, image_tensor, target_class, device, steps=steps
        )
        errors.append(result['relative_error'])
        print(f"Steps: {steps:4d}, Relative Error: {result['relative_error']:.6f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(step_counts, errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Steps', fontsize=12)
    plt.ylabel('Relative Completeness Error', fontsize=12)
    plt.title('Integrated Gradients Convergence', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return dict(zip(step_counts, errors))
```

### Recommendations

| Steps | Use Case | Typical Error |
|-------|----------|---------------|
| 20-30 | Quick exploration, debugging | ~5-10% |
| 50 | Standard usage, good accuracy | ~1-5% |
| 100-200 | Publication, rigorous analysis | <1% |
| 300+ | When completeness error is high | <0.5% |

## Visualization

### Standard Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_integrated_gradients(
    image: np.ndarray,
    attribution: np.ndarray,
    title: str = "Integrated Gradients"
):
    """
    Visualize Integrated Gradients attribution.
    
    Args:
        image: Original image [H, W, 3] in [0, 1]
        attribution: Attribution map [H, W]
        title: Plot title
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Signed attribution (diverging colormap)
    vmax = np.abs(attribution).max()
    im = axes[1].imshow(attribution, cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Signed Attribution')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Absolute attribution
    axes[2].imshow(np.abs(attribution), cmap='hot')
    axes[2].set_title('Absolute Attribution')
    axes[2].axis('off')
    
    # Overlay
    overlay = image.copy()
    mask = np.abs(attribution)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    
    axes[3].imshow(image)
    axes[3].imshow(mask, cmap='jet', alpha=0.5)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig
```

## Applications in Finance

### Feature Attribution for Tabular Data

```python
def ig_for_tabular(
    model: nn.Module,
    features: torch.Tensor,
    feature_names: list,
    target_class: int = None,
    baseline_type: str = 'zero',
    device: torch.device = None
):
    """
    Apply Integrated Gradients to tabular financial data.
    
    Args:
        model: Credit risk or return prediction model
        features: Input features tensor [1, n_features]
        feature_names: List of feature names
        target_class: Target class (for classification)
        baseline_type: 'zero' or 'mean'
        
    Returns:
        Attribution values for each feature
    """
    if device is None:
        device = features.device
        
    ig = IntegratedGradients(model)
    
    # Baseline
    if baseline_type == 'zero':
        baseline = torch.zeros_like(features)
    else:
        # Use training set mean (should be passed in practice)
        baseline = features.mean(dim=0, keepdim=True)
    
    # Compute attributions
    attributions = ig.attribute(features, baseline, target_class, n_steps=100)
    
    # Extract values
    attr_values = attributions.squeeze().cpu().numpy()
    
    # Sort by absolute attribution
    sorted_idx = np.argsort(np.abs(attr_values))[::-1]
    
    print("Feature Attributions (sorted by |attribution|):")
    print("-" * 50)
    for i in sorted_idx[:15]:
        print(f"{feature_names[i]:30s}: {attr_values[i]:+.4f}")
    
    return attr_values, feature_names


# Example: Credit risk model
feature_names = [
    'credit_score', 'debt_to_income', 'loan_amount', 
    'employment_years', 'num_credit_lines', 'payment_history',
    'total_debt', 'income', 'age', 'months_since_delinquent'
]

# For a credit default prediction model:
# attributions, names = ig_for_tabular(credit_model, applicant_features, feature_names)
```

### Time Series Attribution

```python
def ig_for_time_series(
    model: nn.Module,
    sequence: torch.Tensor,
    baseline: torch.Tensor = None,
    target_class: int = None,
    n_steps: int = 100
):
    """
    Apply Integrated Gradients to financial time series.
    
    Args:
        model: Sequence model (LSTM, Transformer, etc.)
        sequence: Input sequence (batch, seq_len, features) or (batch, seq_len)
        baseline: Baseline sequence (default: zeros)
        target_class: Target class for classification models
        
    Returns:
        Temporal attribution showing which time steps matter
    """
    import matplotlib.pyplot as plt
    
    ig = IntegratedGradients(model)
    
    # Default baseline: zeros
    if baseline is None:
        baseline = torch.zeros_like(sequence)
    
    # Compute attributions
    attributions = ig.attribute(sequence, baseline, target_class, n_steps=n_steps)
    
    # Sum across features if multiple features per timestep
    if attributions.dim() == 3:
        temporal_attr = attributions.abs().sum(dim=-1).squeeze()
    else:
        temporal_attr = attributions.abs().squeeze()
    
    # Visualize
    temporal_attr_np = temporal_attr.cpu().numpy()
    
    plt.figure(figsize=(14, 4))
    plt.bar(range(len(temporal_attr_np)), temporal_attr_np, color='steelblue')
    plt.xlabel('Time Step (oldest → most recent)', fontsize=12)
    plt.ylabel('Attribution', fontsize=12)
    plt.title('Temporal Attribution: Which time steps influence the prediction?', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return temporal_attr_np
```

## Comparison with Other Methods

### Theoretical Comparison

| Method | Completeness | Sensitivity | Impl. Invariance | Computation |
|--------|:------------:|:-----------:|:----------------:|:-----------:|
| Vanilla Gradient | ✗ | ✗ | ✓ | Fast |
| Gradient × Input | ✗ | ✗ | ✓ | Fast |
| Integrated Gradients | ✓ | ✓ | ✓ | Moderate |
| DeepLIFT | ✓ | ✓ | ✗ | Fast |
| SHAP (exact) | ✓ | ✓ | ✓ | Slow |
| LRP | ✓ | Partial | ✗ | Moderate |

### Practical Comparison

| Aspect | Vanilla Gradient | Integrated Gradients |
|--------|-----------------|---------------------|
| Computation | Single forward-backward | Multiple (m steps) |
| Sensitivity axiom | ❌ Violated | ✅ Satisfied |
| Completeness | ❌ No guarantee | ✅ Exact (up to approx.) |
| Saturation handling | ❌ Poor | ✅ Good |
| Interpretation | Local sensitivity | Path-accumulated contribution |
| Baseline required | No | Yes |

### Combining with Other Methods

**IG + SmoothGrad:** Average IG over noisy inputs for smoother visualizations
```python
# Smooth Integrated Gradients
smooth_ig = ig.attribute_with_noise(input_tensor, n_samples=10, noise_level=0.1)
```

**IG + Grad-CAM:** Use IG for pixel-level detail, Grad-CAM for regional understanding

## Limitations

### 1. Baseline Dependence

Attributions depend on baseline choice. Different baselines can produce meaningfully different attributions for the same input-output pair.

**Mitigation:**
- Use multiple baselines and average (Expected Gradients / SHAP)
- Use domain-appropriate baselines
- Report baseline choice in publications

### 2. Computational Cost

Requires $m$ forward-backward passes (typically 50-200), much slower than vanilla gradients.

**Mitigation:**
- Use batched computation
- Start with fewer steps for exploration
- Use early stopping based on convergence

### 3. Path Choice

The straight-line path is a specific choice. Other paths could also satisfy the axioms:

$$
\gamma: [0, 1] \rightarrow \mathbb{R}^n, \quad \gamma(0) = \mathbf{x}', \gamma(1) = \mathbf{x}
$$

**Note:** Straight line is the most natural, commonly used, and uniquely satisfies a symmetry preservation property.

### 4. Local Linearity Assumption

IG works best when the model is approximately linear in each local region. Highly nonlinear regions may produce counterintuitive attributions.

## Practical Recommendations

1. **Start with zero baseline** for images; it's simple and usually works well

2. **Use 50 steps** for standard analysis; increase to 100+ for rigorous work

3. **Verify completeness** to ensure numerical accuracy (error <5%)

4. **Compare with Grad-CAM** for complementary insights (IG for pixel-level, Grad-CAM for regional)

5. **Consider batched computation** for efficiency with large images or many explanations

6. **Average over baselines** when baseline choice is uncertain (Expected Gradients)

7. **Report your baseline and step count** for reproducibility

## Summary

Integrated Gradients provides **principled, axiomatically-grounded attributions** by integrating gradients along a path from baseline to input.

### Key Equation

$$
\text{IG}_i(\mathbf{x}) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} \, d\alpha
$$

### Key Properties

| Property | Description |
|----------|-------------|
| **Sensitivity** | Non-zero attribution for features that affect output |
| **Implementation Invariance** | Same function → same attributions |
| **Completeness** | $\sum_i \text{IG}_i = f(\mathbf{x}) - f(\mathbf{x}')$ |

### Best Used For

- Rigorous, theoretically-grounded attribution
- Cases where vanilla gradients fail (saturation, ReLU networks)
- When the completeness property is important
- Regulatory or audit contexts requiring principled explanations
- Comparing attributions across different models fairly

## References

1. Sundararajan, M., Taly, A., & Yan, Q. (2017). "Axiomatic Attribution for Deep Networks." *ICML 2017*.

2. Sturmfels, P., Lundberg, S., & Lee, S. I. (2020). "Visualizing the Impact of Feature Attribution Baselines." *Distill*.

3. Kapishnikov, A., et al. (2019). "XRAI: Better Attributions Through Regions." *ICCV 2019*.

4. Mudrakarta, P. K., et al. (2018). "Did the Model Understand the Question?" *ACL 2018*.

5. Erion, G., et al. (2021). "Improving Performance of Deep Learning Models with Axiomatic Attribution Priors and Expected Gradients." *Nature Machine Intelligence*.
