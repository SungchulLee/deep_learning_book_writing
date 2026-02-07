# Saliency Maps and Vanilla Gradients

## Introduction

**Saliency maps** are visualization techniques that reveal which parts of an input are most important for a neural network's prediction. At their core, saliency methods answer a fundamental question: *"Which input features does the model rely on to make this decision?"*

The simplest and most foundational approach uses **vanilla gradients**—computing the gradient of the model's output with respect to its input. The fundamental idea is straightforward: **if changing a pixel value significantly changes the prediction, that pixel is important.**

This chapter introduces the mathematical foundations, implementation details, and practical considerations for gradient-based saliency methods. While simple and intuitive, understanding vanilla gradients is essential as they form the conceptual foundation for more sophisticated interpretability methods.

## Mathematical Foundation

### Definition

Given a classification function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{C}$ (mapping $n$-dimensional inputs to $C$ class scores), the saliency map for class $c$ and input $\mathbf{x}$ is:

$$
S_c(\mathbf{x}) = \left| \frac{\partial f_c(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

where $|\cdot|$ denotes element-wise absolute value.

### The Gradient as Sensitivity Measure

The gradient $\frac{\partial f_c}{\partial x_i}$ measures the **sensitivity** of class score $f_c$ to infinitesimal changes in input feature $x_i$:

- **Large gradient magnitude** → Small changes to this feature significantly affect the prediction
- **Small gradient magnitude** → This feature has little local influence on the prediction
- **Positive gradient** → Increasing this feature increases the class score
- **Negative gradient** → Increasing this feature decreases the class score

### Interpretation via Taylor Expansion

The gradient-based saliency has a natural interpretation through first-order Taylor expansion. For a small perturbation $\boldsymbol{\epsilon}$:

$$
f_c(\mathbf{x} + \boldsymbol{\epsilon}) \approx f_c(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla_{\mathbf{x}} f_c(\mathbf{x})
$$

This reveals that:

1. **High gradient magnitude** at pixel $i$ means small changes to $x_i$ cause large changes in $y_c$
2. **The gradient direction** indicates whether increasing the pixel value increases or decreases the class score
3. **The absolute gradient** captures sensitivity regardless of direction
4. The gradient tells us which **direction of perturbation** maximally changes the output

### Multi-Channel Aggregation

For RGB images with shape $(3, H, W)$, we compute gradients with respect to all channels and aggregate:

$$
\mathbf{G} = \frac{\partial y_c}{\partial \mathbf{x}} \in \mathbb{R}^{3 \times H \times W}
$$

Common aggregation strategies include:

| Method | Formula | Characteristics |
|--------|---------|-----------------|
| **Maximum** | $S_{i,j} = \max_{k \in \{R,G,B\}} \|G_{k,i,j}\|$ | Highlights pixels important in any channel |
| **Mean** | $S_{i,j} = \frac{1}{3} \sum_{k} \|G_{k,i,j}\|$ | Averages importance across channels |
| **L2 Norm** | $S_{i,j} = \sqrt{\sum_{k} G_{k,i,j}^2}$ | Euclidean magnitude of gradient vector |
| **Sum** | $S_{i,j} = \sum_{k} \|G_{k,i,j}\|$ | Total absolute sensitivity |

## PyTorch Implementation

### Basic Vanilla Gradient Saliency

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def compute_saliency_map(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = None,
    device: torch.device = None
) -> np.ndarray:
    """
    Compute vanilla gradient saliency map.
    
    Args:
        model: Neural network model in eval mode
        input_tensor: Input tensor of shape (1, C, H, W)
        target_class: Target class index (uses predicted class if None)
        device: Computation device
        
    Returns:
        Saliency map as numpy array of shape (H, W)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Enable gradient computation for input
    input_tensor = input_tensor.clone().to(device).requires_grad_(True)
    
    # Forward pass
    model.eval()
    output = model(input_tensor)
    
    # Determine target class
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass: compute ∂y_c/∂x
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get gradient with respect to input
    saliency = input_tensor.grad.data.abs()
    
    # Take maximum across color channels
    saliency, _ = saliency.max(dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    
    return saliency


def compute_vanilla_gradient_saliency(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute vanilla gradient saliency map (returns tensor).
    
    Args:
        model: Pretrained neural network in eval mode
        image_tensor: Input image [1, 3, H, W]
        target_class: Class index to compute saliency for
        device: Computation device (CPU/GPU)
        
    Returns:
        Saliency map tensor [1, H, W]
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # Ensure gradients can flow to input
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()
    
    # Forward pass
    output = model(image_tensor)  # [1, num_classes]
    
    # Select target class score
    target_score = output[0, target_class]
    
    # Backward pass
    target_score.backward()
    
    # Get gradients and take absolute value
    gradients = image_tensor.grad  # [1, 3, H, W]
    abs_gradients = torch.abs(gradients)
    
    # Aggregate across color channels (max pooling)
    saliency = torch.max(abs_gradients, dim=1)[0]  # [1, H, W]
    
    return saliency
```

### Signed Saliency (Positive and Negative Influences)

```python
def compute_signed_saliency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = None
) -> tuple:
    """
    Compute signed saliency showing positive and negative influences.
    
    Returns:
        positive_saliency: Features that INCREASE class score when increased
        negative_saliency: Features that DECREASE class score when increased
    """
    input_tensor = input_tensor.clone().requires_grad_(True)
    
    model.eval()
    output = model(input_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, target_class].backward()
    
    gradient = input_tensor.grad.data
    
    # Separate positive and negative gradients
    positive = gradient.clamp(min=0)
    negative = gradient.clamp(max=0).abs()
    
    # Max across channels
    positive_saliency = positive.max(dim=1)[0].squeeze().cpu().numpy()
    negative_saliency = negative.max(dim=1)[0].squeeze().cpu().numpy()
    
    return positive_saliency, negative_saliency
```

### Gradient × Input

Multiplying gradients by input values can sharpen the attribution by highlighting features that both **exist** and are **important**:

$$
\text{Gradient} \times \text{Input} = \mathbf{x} \odot \frac{\partial f_c}{\partial \mathbf{x}}
$$

```python
def gradient_times_input(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = None
) -> np.ndarray:
    """
    Compute Gradient × Input saliency.
    
    This method weights the gradient by the input value,
    showing which features both exist and are important.
    """
    input_tensor = input_tensor.clone().requires_grad_(True)
    
    model.eval()
    output = model(input_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    output[0, target_class].backward()
    
    # Gradient × Input
    grad_input = input_tensor.grad.data * input_tensor.data
    
    # Take absolute value and max across channels
    saliency = grad_input.abs().max(dim=1)[0].squeeze().cpu().numpy()
    
    return saliency
```

### All Aggregation Variants

```python
def compute_all_aggregations(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    device: torch.device
) -> dict:
    """Compare different gradient aggregation methods."""
    model.eval()
    input_tensor = input_tensor.clone().to(device).requires_grad_(True)
    
    output = model(input_tensor)
    model.zero_grad()
    output[0, target_class].backward()
    
    gradients = input_tensor.grad
    abs_gradients = torch.abs(gradients)
    
    return {
        'max': torch.max(abs_gradients, dim=1)[0].squeeze().cpu().numpy(),
        'mean': torch.mean(abs_gradients, dim=1).squeeze().cpu().numpy(),
        'l2': torch.sqrt(torch.sum(abs_gradients ** 2, dim=1)).squeeze().cpu().numpy(),
        'sum': torch.sum(abs_gradients, dim=1).squeeze().cpu().numpy(),
        'squared': (gradients ** 2).max(dim=1)[0].squeeze().cpu().numpy(),
        'positive_only': gradients.clamp(min=0).max(dim=1)[0].squeeze().cpu().numpy(),
    }
```

## Complete Working Example

```python
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = model.to(device)
model.eval()

# Preprocessing transform (ImageNet normalization)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load and preprocess image
image = Image.open('dog.jpg').convert('RGB')
image_tensor = preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]
image_tensor.requires_grad = True

# Get model prediction
with torch.no_grad():
    output = model(image_tensor.to(device))
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0, pred_class].item()

print(f"Predicted class: {pred_class}, Confidence: {confidence:.2%}")

# Compute different saliency maps
saliency = compute_saliency_map(model, image_tensor, pred_class, device)
pos_saliency, neg_saliency = compute_signed_saliency(model, image_tensor, pred_class)
grad_input = gradient_times_input(model, image_tensor, pred_class)

# Prepare original image for visualization
original = np.array(image.resize((224, 224))) / 255.0

print(f"Saliency shape: {saliency.shape}")
print(f"Value range: [{saliency.min():.6f}, {saliency.max():.6f}]")
```

## Visualization

### Proper Normalization

For meaningful visualization, saliency maps should be normalized to $[0, 1]$:

```python
def normalize_saliency(saliency: np.ndarray) -> np.ndarray:
    """
    Normalize saliency map to [0, 1] for visualization.
    """
    if isinstance(saliency, torch.Tensor):
        saliency = saliency.detach().cpu().numpy()
    
    if saliency.ndim == 3:
        saliency = saliency.squeeze(0)
    
    s_min, s_max = saliency.min(), saliency.max()
    
    if s_max - s_min > 1e-10:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)
    
    return saliency
```

### Standard Visualization

```python
def visualize_saliency(
    original_image: np.ndarray,
    saliency_map: np.ndarray,
    title: str = "Saliency Map"
) -> plt.Figure:
    """
    Visualize saliency map alongside original image.
    
    Args:
        original_image: Original image (H, W, 3) in [0, 1]
        saliency_map: Saliency values (H, W)
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Saliency map
    saliency_norm = normalize_saliency(saliency_map)
    im = axes[1].imshow(saliency_norm, cmap='hot')
    axes[1].set_title(title)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    if original_image.ndim == 3:
        gray_img = np.mean(original_image, axis=2)
    else:
        gray_img = original_image
    
    axes[2].imshow(gray_img, cmap='gray')
    axes[2].imshow(saliency_norm, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig
```

### Signed Saliency Visualization

```python
def visualize_signed_saliency(
    original_image: np.ndarray,
    positive_saliency: np.ndarray,
    negative_saliency: np.ndarray
) -> plt.Figure:
    """
    Visualize positive and negative saliency separately.
    
    - Red: Features that INCREASE class score when increased
    - Blue: Features that DECREASE class score when increased
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(positive_saliency, cmap='Reds')
    axes[1].set_title('Positive (increases score)')
    axes[1].axis('off')
    
    axes[2].imshow(negative_saliency, cmap='Blues')
    axes[2].set_title('Negative (decreases score)')
    axes[2].axis('off')
    
    # Combined view: red = positive, blue = negative
    combined = np.zeros((*positive_saliency.shape, 3))
    combined[:, :, 0] = positive_saliency / (positive_saliency.max() + 1e-8)
    combined[:, :, 2] = negative_saliency / (negative_saliency.max() + 1e-8)
    
    axes[3].imshow(combined)
    axes[3].set_title('Combined (Red+, Blue-)')
    axes[3].axis('off')
    
    plt.tight_layout()
    return fig
```

### Comparing Multiple Classes

A key insight is that saliency maps are **class-specific**. Different target classes can highlight different image regions:

```python
def compare_class_saliencies(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_indices: list,
    class_names: list = None,
    device: torch.device = None
) -> dict:
    """
    Compute and compare saliency maps for multiple target classes.
    
    Demonstrates that different classes highlight different regions.
    """
    if device is None:
        device = next(model.parameters()).device
    
    saliencies = {}
    
    for i, class_idx in enumerate(class_indices):
        # Create fresh tensor for each backward pass
        img_copy = image_tensor.clone().detach().requires_grad_(True)
        
        saliency = compute_saliency_map(model, img_copy, class_idx, device)
        
        name = class_names[i] if class_names else f"Class {class_idx}"
        saliencies[name] = saliency
    
    return saliencies


# Example: Compare top-3 predicted classes
with torch.no_grad():
    output = model(image_tensor.to(device))
    top_classes = output[0].topk(3).indices.tolist()

saliencies = compare_class_saliencies(model, image_tensor, top_classes, device=device)
```

## Statistical Analysis

Understanding the distribution of saliency values provides diagnostic insights:

```python
def analyze_saliency_statistics(saliency: np.ndarray) -> dict:
    """Compute comprehensive statistics about saliency distribution."""
    s = saliency.flatten()
    
    stats = {
        'min': s.min(),
        'max': s.max(),
        'mean': s.mean(),
        'std': s.std(),
        'median': np.median(s),
        'p75': np.percentile(s, 75),
        'p90': np.percentile(s, 90),
        'p95': np.percentile(s, 95),
        'p99': np.percentile(s, 99),
        'sparsity': (s < s.mean()).mean(),  # Fraction below mean
        'max_mean_ratio': s.max() / (s.mean() + 1e-8)
    }
    
    return stats
```

**Interpretation Guidelines:**

| Metric | Interpretation |
|--------|----------------|
| High sparsity (>80%) | Few pixels dominate, possibly meaningful localization |
| Low max/mean ratio | Diffuse importance, model uses global information |
| High variance | Strong localization with background suppression |
| High p99/p95 gap | Few extreme outliers, check for artifacts |

## Limitations and Challenges

### 1. Visual Noise

Vanilla gradient saliency maps are notoriously **noisy**. This occurs because:

1. **High-frequency sensitivity**: Neural networks have high Lipschitz constants, making gradients sensitive to small input variations
2. **Saturation**: ReLU activations create discontinuities in the gradient landscape
3. **Lack of smoothness**: The gradient captures local sensitivity, not global importance

```python
def demonstrate_noise():
    """Show that saliency maps can be noisy even for random inputs."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    
    # Random input
    input_tensor = torch.randn(1, 3, 224, 224)
    
    saliency = compute_saliency_map(model, input_tensor)
    
    # Saliency often appears scattered even for random inputs
    print(f"Non-zero pixels: {(saliency > 0.01 * saliency.max()).sum()}")
    print(f"This indicates noise in gradient-based saliency")
```

### 2. Gradient Saturation

For saturating nonlinearities (sigmoid, tanh) or with ReLU networks:

$$
\frac{\partial \text{ReLU}(x)}{\partial x} = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

When activations are in the "dead" region, gradients become zero regardless of input importance. For highly confident predictions, softmax gradients also become very small.

**Solution:** Use raw logits (pre-softmax scores) instead of post-softmax probabilities for computing gradients.

### 3. Lack of Class Discrimination

While gradients are computed with respect to a specific class, vanilla saliency often highlights similar regions for different classes:

```python
def class_discrimination_test(model, input_tensor, device):
    """Compare saliency maps for different classes."""
    with torch.no_grad():
        output = model(input_tensor.to(device))
        _, top_classes = output.topk(5)
    
    saliencies = {}
    for cls in top_classes[0]:
        saliencies[cls.item()] = compute_saliency_map(
            model, input_tensor, cls.item(), device
        )
    
    # Often, saliencies for different classes look quite similar
    # because early layer gradients are shared across all classes
    return saliencies
```

### 4. Sensitivity to Input Perturbations

Small input perturbations can significantly change saliency maps:

```python
def sensitivity_test(model, input_tensor, device, noise_scale=0.01):
    """Test sensitivity of saliency to input noise."""
    saliency_original = compute_saliency_map(model, input_tensor, device=device)
    
    # Add small noise
    noisy_input = input_tensor + noise_scale * torch.randn_like(input_tensor)
    saliency_noisy = compute_saliency_map(model, noisy_input, device=device)
    
    # Compute difference
    diff = np.abs(saliency_original - saliency_noisy)
    
    print(f"Mean absolute difference: {diff.mean():.4f}")
    print(f"Max difference: {diff.max():.4f}")
    print(f"Correlation: {np.corrcoef(saliency_original.flatten(), saliency_noisy.flatten())[0,1]:.4f}")
    
    return diff
```

## SmoothGrad: Noise Reduction

SmoothGrad addresses the noise problem by averaging gradients over noisy copies of the input:

$$
\hat{S}_c(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\partial f_c(\mathbf{x} + \mathcal{N}(0, \sigma^2))}{\partial \mathbf{x}} \right|
$$

```python
def smoothgrad(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int = None,
    n_samples: int = 50,
    noise_level: float = 0.1,
    device: torch.device = None
) -> np.ndarray:
    """
    Compute SmoothGrad saliency map.
    
    Args:
        model: Neural network
        input_tensor: Input tensor
        target_class: Target class
        n_samples: Number of noisy samples
        noise_level: Standard deviation of noise (as fraction of input range)
        device: Computation device
        
    Returns:
        Smoothed saliency map
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    input_tensor = input_tensor.to(device)
    
    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()
    
    # Compute input statistics for noise scaling
    stdev = noise_level * (input_tensor.max() - input_tensor.min())
    
    accumulated_grad = torch.zeros_like(input_tensor)
    
    for _ in range(n_samples):
        # Add Gaussian noise
        noise = torch.randn_like(input_tensor) * stdev
        noisy_input = (input_tensor + noise).requires_grad_(True)
        
        # Forward and backward
        output = model(noisy_input)
        model.zero_grad()
        output[0, target_class].backward()
        
        # Accumulate gradients
        accumulated_grad += noisy_input.grad.data.abs()
    
    # Average
    smoothed_grad = accumulated_grad / n_samples
    
    # Max across channels
    saliency = smoothed_grad.max(dim=1)[0].squeeze().cpu().numpy()
    
    return saliency
```

## Applications in Finance

### Time Series Saliency

For sequential financial data, saliency reveals which time steps matter:

```python
def time_series_saliency(
    model: nn.Module,
    sequence: torch.Tensor,
    target_class: int = None
) -> np.ndarray:
    """
    Compute saliency for time series predictions.
    
    Shows which historical time points most influence the prediction.
    
    Args:
        model: Sequence model (LSTM, Transformer, etc.)
        sequence: Input sequence [1, seq_len, features] or [1, seq_len]
        target_class: Target class for classification models
        
    Returns:
        Saliency over time dimension
    """
    sequence = sequence.clone().requires_grad_(True)
    
    model.eval()
    output = model(sequence)
    
    if target_class is not None:
        output = output[0, target_class]
    else:
        output = output.squeeze()
    
    model.zero_grad()
    output.backward()
    
    # Saliency over time dimension
    saliency = sequence.grad.abs().squeeze().cpu().numpy()
    
    # If multiple features, aggregate
    if saliency.ndim == 2:
        saliency = saliency.sum(axis=1)  # Sum across features
    
    return saliency
```

### Tabular Feature Importance

For tabular financial data (credit risk, trading signals, etc.):

```python
def tabular_saliency(
    model: nn.Module,
    features: torch.Tensor,
    feature_names: list,
    target_class: int = None
) -> np.ndarray:
    """
    Compute and visualize feature importance via gradients.
    
    Args:
        model: Classification or regression model
        features: Feature tensor [1, n_features]
        feature_names: List of feature names
        target_class: Target class (for classification)
        
    Returns:
        Importance scores for each feature
    """
    features = features.clone().requires_grad_(True)
    
    model.eval()
    output = model(features)
    
    if target_class is not None:
        output = output[0, target_class]
    else:
        output = output.squeeze()
    
    model.zero_grad()
    output.backward()
    
    importance = features.grad.abs().squeeze().cpu().numpy()
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    
    print("Feature Importance (by gradient magnitude):")
    print("-" * 50)
    for i in sorted_idx[:10]:
        print(f"{feature_names[i]:30s}: {importance[i]:.6f}")
    
    return importance
```

## Relationship to Other Methods

Vanilla gradients serve as the foundation for more sophisticated methods:

| Method | Modification | Addresses |
|--------|--------------|-----------|
| **Gradient × Input** | Multiply by input values | Sharpens features |
| **SmoothGrad** | Average over noisy samples | Reduces noise |
| **Integrated Gradients** | Integrate along path from baseline | Satisfies axioms |
| **Guided Backpropagation** | Modify ReLU backward pass | Cleaner visuals |
| **Grad-CAM** | Use feature map gradients | Class discrimination |

### Comparison Table

| Method | Pros | Cons |
|--------|------|------|
| Vanilla Gradient | Simple, fast, foundational | Noisy, low discriminability |
| Gradient × Input | Sharper features | Still noisy |
| SmoothGrad | Reduced noise | Computationally expensive |
| Grad-CAM | Clean, class-discriminative | Lower resolution |
| Integrated Gradients | Theoretical guarantees | Slow, baseline-dependent |

## Practical Recommendations

### When to Use Vanilla Gradients

**Suitable for:**
- Quick debugging and sanity checks
- Understanding gradient flow
- Baseline comparisons with other methods
- Educational purposes
- Initial exploration

**Not recommended for:**
- Publication-quality visualizations (too noisy)
- Production explainability (use smoother methods)
- High-stakes decisions (combine with other evidence)
- Class-discriminative explanations (use Grad-CAM)

### Implementation Checklist

1. ✅ **Model in eval mode**: `model.eval()` ensures deterministic behavior
2. ✅ **Enable gradients on input**: `image_tensor.requires_grad = True`
3. ✅ **Clear existing gradients**: `model.zero_grad()` before backward
4. ✅ **Use appropriate class**: Predicted class or class of interest
5. ✅ **Use logits, not softmax**: Avoid gradient saturation
6. ✅ **Normalize for visualization**: Scale to $[0, 1]$
7. ✅ **Clone tensors**: Fresh tensor for each backward pass when comparing

## Summary

Vanilla gradient saliency provides the conceptual foundation for understanding gradient-based interpretability methods. While simple and intuitive, its practical utility is limited by visual noise and lack of class discrimination. Subsequent methods build upon this foundation to produce cleaner, more meaningful attributions.

### Key Equations

**Basic Saliency:**
$$
S(\mathbf{x}) = \left| \frac{\partial f_c(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

**Aggregated Saliency (for multi-channel inputs):**
$$
S_{i,j} = \max_{k} |G_{k,i,j}| \quad \text{or} \quad \sqrt{\sum_k G_{k,i,j}^2}
$$

**Gradient × Input:**
$$
S(\mathbf{x}) = \left| \mathbf{x} \odot \frac{\partial f_c(\mathbf{x})}{\partial \mathbf{x}} \right|
$$

**SmoothGrad:**
$$
\hat{S}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\partial f_c(\mathbf{x} + \epsilon_i)}{\partial \mathbf{x}} \right|, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

### Key Takeaways

1. **Gradients measure local sensitivity**, not global importance
2. **Noise is inherent** due to network non-smoothness
3. **Class discrimination is limited** because early layers are shared
4. **Multiple aggregation methods** are possible; max is most common
5. **Foundation for advanced methods** like Integrated Gradients, SmoothGrad

## References

1. Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps." *ICLR Workshop*.

2. Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). "SmoothGrad: removing noise by adding noise." *ICML Workshop*.

3. Shrikumar, A., Greenside, P., & Kundaje, A. (2017). "Learning Important Features Through Propagating Activation Differences." *ICML*.

4. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). "Sanity Checks for Saliency Maps." *NeurIPS*.

5. Baehrens, D., Schroeter, T., Harmeling, S., Kawanabe, M., Hansen, K., & Müller, K. R. (2010). "How to Explain Individual Classification Decisions." *Journal of Machine Learning Research*.
