# Grad-CAM: Gradient-weighted Class Activation Mapping

## Introduction

**Gradient-weighted Class Activation Mapping (Grad-CAM)** is a technique for producing visual explanations of decisions from CNN-based models. Unlike earlier Class Activation Mapping (CAM) methods that require specific architectures, Grad-CAM works with any CNN architecture without requiring architectural modifications or retraining.

Grad-CAM answers the fundamental question: **Which regions of the input image are most important for predicting a particular class?**

Unlike pixel-level gradient methods, Grad-CAM generates **coarse localization maps** highlighting discriminative image regions. It addresses two key limitations of vanilla gradients:

1. **Class discrimination**: Different classes produce distinctly different heatmaps
2. **Visual interpretability**: Produces smooth, human-understandable localizations

## Mathematical Foundation

### Problem Setup

Consider a CNN classifier with:
- Input image $I \in \mathbb{R}^{H \times W \times 3}$
- Convolutional feature maps $A^k \in \mathbb{R}^{u \times v}$ at a target layer (typically the last convolutional layer)
- Class score $y^c$ for class $c$ (before softmax)

We seek a heatmap $L^c_{\text{Grad-CAM}} \in \mathbb{R}^{u \times v}$ highlighting image regions important for predicting class $c$.

### Core Formulation

Grad-CAM leverages the spatial information preserved in convolutional feature maps. The key insight is that **later convolutional layers contain semantic information** while **gradients indicate importance for the target class**.

For a target class $c$, Grad-CAM computes:

$$
L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_k \alpha^c_k A^k \right)
$$

where:

- $A^k \in \mathbb{R}^{H' \times W'}$ is the $k$-th feature map of the target convolutional layer
- $\alpha^c_k$ is the importance weight of feature map $k$ for class $c$
- ReLU ensures we focus on features with positive influence

### Computing Importance Weights

The importance weights $\alpha^c_k$ are computed via **global average pooling** of gradients:

$$
\alpha^c_k = \underbrace{\frac{1}{Z} \sum_i \sum_j}_{\text{global average pooling}} \underbrace{\frac{\partial y^c}{\partial A^k_{ij}}}_{\text{gradients}}
$$

where:

- $y^c$ is the class score for class $c$ (before softmax)
- $Z = H' \times W'$ is the number of spatial positions
- $\frac{\partial y^c}{\partial A^k_{ij}}$ is the gradient of the class score with respect to activation $A^k$ at position $(i, j)$

This global average pooling of gradients gives the **neuron importance weight** $\alpha^c_k$ for feature map $k$ and class $c$.

### Intuition Behind the Formula

**Why global average pooling of gradients?**

The gradient $\frac{\partial y^c}{\partial A^k_{ij}}$ tells us how important a specific spatial location in feature map $k$ is. By averaging across all spatial positions, we obtain a measure of the **overall importance** of feature map $k$ for class $c$.

**Why ReLU?**

Features with negative contribution (decreasing the class score) are suppressed. We only want to visualize regions that **increase** the likelihood of the target class. Features that need to be suppressed (negative gradients) are not relevant for explaining the class prediction.

**Why the last convolutional layer?**

Later layers contain:
- Higher-level semantic information
- Class-discriminative features
- Sufficient spatial resolution for localization

### Mathematical Derivation

The Grad-CAM formula can be derived from a **first-order Taylor expansion**. Starting from first principles, consider the class score $y^c$ as a function of all feature maps:

$$
y^c = f(A^1, A^2, \ldots, A^K)
$$

The class score change due to a small perturbation in feature maps is approximately:

$$
y^c \approx \sum_k \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k} A_{ij}^k
$$

Rearranging with the global average pooling assumption:

$$
y^c \approx \sum_k \alpha_k^c \sum_i \sum_j A_{ij}^k
$$

This shows that $\alpha_k^c$ measures how much feature map $k$ contributes to the class score.

The global average pooling aggregates spatial information:

$$
\alpha^c_k = \frac{1}{H' \cdot W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \frac{\partial y^c}{\partial A^k_{ij}}
$$

The weighted sum combines feature maps by importance:

$$
L^c(i,j) = \sum_{k=1}^{K} \alpha^c_k A^k_{ij}
$$

Finally, ReLU removes negative influences:

$$
L^c_{\text{Grad-CAM}} = \text{ReLU}(L^c)
$$

## Algorithm

```
Algorithm: Grad-CAM
Input: Image I, CNN model f, target class c, target layer l
Output: Heatmap L_GradCAM

1. Forward pass: Compute feature maps A^k at layer l
2. Forward pass: Compute class score y^c
3. Backward pass: Compute gradients ∂y^c/∂A^k
4. For each feature map k:
   α_k^c = GlobalAveragePool(∂y^c/∂A^k)
5. Compute weighted combination:
   L_GradCAM = ReLU(Σ_k α_k^c * A^k)
6. Upsample L_GradCAM to input resolution
7. Return L_GradCAM
```

## PyTorch Implementation

### Complete GradCAM Class

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """
    Grad-CAM implementation for any CNN architecture.
    
    Args:
        model: PyTorch CNN model
        target_layer: Convolutional layer to visualize
        
    Usage:
        gradcam = GradCAM(model, model.layer4[-1])  # For ResNet
        heatmap = gradcam(image_tensor, target_class, device)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to capture forward pass activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to capture backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def __call__(
        self,
        image_tensor: torch.Tensor,
        target_class: int = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            image_tensor: Input image [1, C, H, W]
            target_class: Target class index. If None, uses predicted class.
            device: Computation device
            
        Returns:
            Heatmap tensor [H, W] in range [0, 1]
        """
        if device is None:
            device = next(self.model.parameters()).device
            
        self.model.eval()
        image_tensor = image_tensor.to(device)
        
        # Forward pass - triggers forward hook
        output = self.model(image_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Get target class score
        target_score = output[0, target_class]
        
        # Backward pass - triggers backward hook
        self.model.zero_grad()
        target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]    # Shape: [K, H', W']
        activations = self.activations[0] # Shape: [K, H', W']
        
        # Compute importance weights: α_k = GAP(gradients)
        # α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)
        weights = gradients.mean(dim=(1, 2))  # Shape: [K]
        
        # Weighted combination: L = Σ_k α_k^c * A^k
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for k, w in enumerate(weights):
            cam += w * activations[k]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Upsample to input resolution
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=image_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return cam.squeeze()  # [H, W]
    
    def generate_visualization(
        self,
        image_tensor: torch.Tensor,
        original_image: np.ndarray = None,
        target_class: int = None,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Generate Grad-CAM visualization overlaid on original image.
        
        Args:
            image_tensor: Input image tensor
            original_image: Original image as numpy array (H, W, 3) in [0, 255]
            target_class: Target class index
            alpha: Overlay transparency (0 = only image, 1 = only heatmap)
            
        Returns:
            Visualization as numpy array (H, W, 3) in [0, 255]
        """
        import cv2
        
        # Generate CAM
        cam = self(image_tensor, target_class)
        cam_np = cam.cpu().numpy()
        
        # Resize to input dimensions
        if original_image is not None:
            h, w = original_image.shape[:2]
        else:
            h, w = image_tensor.shape[2:]
        
        cam_resized = cv2.resize(cam_np, (w, h))
        
        # Convert to heatmap (BGR for OpenCV)
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        if original_image is not None:
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            visualization = cv2.addWeighted(
                original_image, 1 - alpha, heatmap, alpha, 0
            )
        else:
            visualization = heatmap
        
        return visualization
```

### Getting Target Layers for Different Architectures

```python
def get_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """
    Get the appropriate target layer for Grad-CAM.
    
    Args:
        model: The pretrained model
        architecture: Model architecture name
        
    Returns:
        The target convolutional layer
    """
    architecture = architecture.lower()
    
    if 'resnet' in architecture:
        # ResNet: layer4[-1] is the last bottleneck/basicblock
        return model.layer4[-1]
    
    elif 'vgg' in architecture:
        # VGG: Last conv layer before classifier
        return model.features[-1]
    
    elif 'densenet' in architecture:
        # DenseNet: Last dense block
        return model.features.denseblock4
    
    elif 'efficientnet' in architecture:
        # EfficientNet: Last conv layer
        return model.features[-1]
    
    elif 'mobilenet' in architecture:
        # MobileNet: Last conv layer
        return model.features[-1]
    
    elif 'inception' in architecture:
        # Inception: Mixed layer
        return model.Mixed_7c
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
```

### Complete Usage Example

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load pre-trained ResNet
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Target the last convolutional layer
target_layer = model.layer4[-1]

# Initialize Grad-CAM
grad_cam = GradCAM(model, target_layer)

# Prepare input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open('cat.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# Generate Grad-CAM for predicted class
cam = grad_cam(input_tensor)
print(f"CAM shape: {cam.shape}")  # torch.Size([224, 224])

# Generate Grad-CAM for specific class (e.g., class 281 = tabby cat)
cam_tabby = grad_cam(input_tensor, target_class=281)

# Create visualization
original_image = np.array(image.resize((224, 224)))
visualization = grad_cam.generate_visualization(
    input_tensor, 
    original_image,
    target_class=281
)

# Display
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cam_tabby.cpu().numpy(), cmap='jet')
plt.title('Grad-CAM Heatmap')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(visualization)
plt.title('Overlay')
plt.axis('off')

plt.tight_layout()
plt.show()
```

## Choosing the Target Layer

The choice of target layer significantly affects Grad-CAM visualizations.

### Last Convolutional Layer (Recommended)

The last convolutional layer captures the most semantically meaningful features:

| Aspect | Pro | Con |
|--------|-----|-----|
| Semantic content | High-level concepts | - |
| Localization | Good object coverage | Lower spatial resolution |
| Class discrimination | Strong | - |

```python
# Architecture-specific target layers
target_layers = {
    'resnet': model.layer4[-1],
    'vgg': model.features[-1],
    'mobilenet': model.features[-1],
    'densenet': model.features.denseblock4,
    'efficientnet': model.features[-1],
}
```

### Earlier Layers

Earlier layers capture lower-level features:

| Aspect | Pro | Con |
|--------|-----|-----|
| Spatial resolution | Higher | - |
| Features | Texture, edges | Less semantic meaning |
| Interpretability | - | Noisier attributions |

### Multi-Layer Grad-CAM

Combining Grad-CAM from multiple layers provides richer explanations:

```python
def multi_layer_gradcam(model, image_tensor, layers, target_class=None, device=None):
    """Generate Grad-CAM from multiple layers and combine."""
    import cv2
    
    cams = []
    for layer in layers:
        gc = GradCAM(model, layer)
        cam = gc(image_tensor, target_class, device)
        cams.append(cam.cpu().numpy())
    
    # Resize all CAMs to same size and average
    target_size = (224, 224)
    combined = np.zeros(target_size)
    for cam in cams:
        resized = cv2.resize(cam, target_size)
        combined += resized
    
    combined /= len(cams)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    
    return combined

# Example: Combine layer3 and layer4
layers = [model.layer3[-1], model.layer4[-1]]
multi_cam = multi_layer_gradcam(model, input_tensor, layers, target_class=281)
```

## Advanced Techniques

### Comparing Multiple Classes

A key property of Grad-CAM is its **class-discriminative** nature:

```python
def compare_gradcam_classes(
    model: nn.Module,
    gradcam: GradCAM,
    image_tensor: torch.Tensor,
    class_indices: list,
    class_names: list,
    device: torch.device
):
    """
    Compare Grad-CAM heatmaps for multiple target classes.
    
    Demonstrates that different classes highlight different regions.
    """
    n_classes = len(class_indices)
    fig, axes = plt.subplots(2, n_classes + 1, figsize=(4 * (n_classes + 1), 8))
    
    # Denormalize image for display
    image_np = denormalize_image(image_tensor)
    
    # Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original', fontsize=11)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    for idx, (class_idx, class_name) in enumerate(zip(class_indices, class_names)):
        # Compute Grad-CAM for this class
        heatmap = gradcam(image_tensor, class_idx, device)
        heatmap_np = heatmap.cpu().numpy()
        
        # Heatmap
        axes[0, idx + 1].imshow(heatmap_np, cmap='jet')
        axes[0, idx + 1].set_title(f'{class_name}\n(class {class_idx})', fontsize=10)
        axes[0, idx + 1].axis('off')
        
        # Overlay
        overlay = create_overlay(image_np, heatmap_np, alpha=0.5)
        axes[1, idx + 1].imshow(overlay)
        axes[1, idx + 1].set_title('Overlay', fontsize=10)
        axes[1, idx + 1].axis('off')
    
    plt.tight_layout()
    return fig

# Example: Image with cat and dog
# cam_cat highlights the cat region
# cam_dog highlights the dog region
```

### Negative Grad-CAM

To visualize features that **decrease** class probability (counterfactual regions):

```python
def negative_gradcam(gradcam, image_tensor, target_class, device):
    """
    Compute regions that DECREASE target class probability.
    
    Useful for understanding what the model thinks is NOT the target class.
    """
    model = gradcam.model
    model.eval()
    
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    target_score = output[0, target_class]
    
    model.zero_grad()
    target_score.backward()
    
    # Use NEGATIVE weights (regions that decrease class score)
    weights = -gradcam.gradients.mean(dim=(2, 3), keepdim=True)
    
    weighted = weights * gradcam.activations
    heatmap = weighted.sum(dim=1, keepdim=True)
    heatmap = F.relu(heatmap)
    
    # Normalize
    heatmap = heatmap / (heatmap.max() + 1e-8)
    
    return F.interpolate(
        heatmap, size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze()
```

### Grad-CAM for Different Layer Analysis

```python
def analyze_layer_gradcam(model, image_tensor, target_class, device):
    """
    Compare Grad-CAM from different layers to understand feature hierarchy.
    """
    # For ResNet, compare layer2, layer3, layer4
    layers = {
        'layer2 (mid-level)': model.layer2[-1],
        'layer3 (high-level)': model.layer3[-1],
        'layer4 (semantic)': model.layer4[-1]
    }
    
    results = {}
    for name, layer in layers.items():
        gradcam = GradCAM(model, layer)
        heatmap = gradcam(image_tensor, target_class, device)
        results[name] = heatmap.cpu().numpy()
    
    return results
```

## Properties of Grad-CAM

### Class Discriminativeness

A crucial property of Grad-CAM is its **class-discriminative** nature. For different target classes:

- The importance weights $\alpha^c_k$ change with $c$
- Different feature maps become dominant
- The resulting heatmaps highlight different spatial regions

This allows Grad-CAM to answer: *"Which regions are important for THIS specific class?"*

### Relationship to CAM

Grad-CAM is a generalization of **Class Activation Mapping (CAM)** by Zhou et al. (2016).

| Aspect | CAM | Grad-CAM |
|--------|-----|----------|
| Architecture | Requires GAP + FC | Any CNN |
| Retraining | Required for non-GAP models | Not required |
| Target layer | Fixed (last conv) | Any conv layer |
| Weight computation | FC layer weights | Gradient-based |
| Explanation quality | High | High |
| Computational cost | Lower | Higher (backprop) |

**Key insight**: For architectures with GAP→FC, Grad-CAM and CAM produce identical results. Grad-CAM generalizes CAM to arbitrary architectures.

### Resolution Trade-off

Grad-CAM produces **coarse localization** because:

1. Feature maps at deep layers have reduced spatial resolution
   - ResNet-50 layer4: 7×7 for 224×224 input
   - VGG-16: 14×14 for 224×224 input
2. Upsampling to input resolution introduces interpolation artifacts
3. Semantic information is captured at the expense of fine details

This is a **complementary** property to pixel-level methods:
- **Grad-CAM** shows *where* (coarse localization)
- **Gradient methods** show *what* (fine-grained features)
- **Guided Grad-CAM** combines both

## Limitations

### 1. Coarse Spatial Resolution

The main limitation is spatial resolution. Feature maps at deep layers are typically much smaller than input:

| Architecture | Input Size | Layer4 Size | Reduction |
|--------------|------------|-------------|-----------|
| ResNet-50 | 224×224 | 7×7 | 32× |
| VGG-16 | 224×224 | 14×14 | 16× |
| EfficientNet-B0 | 224×224 | 7×7 | 32× |

This makes Grad-CAM unsuitable for:
- Precise boundary detection
- Fine-grained localization
- Small object detection

### 2. Class Confusion with Multiple Objects

When an image contains multiple objects, Grad-CAM highlights regions for the specified class but may show spurious activations:

```python
# Image contains both cat and dog
cam_cat = grad_cam(input_tensor, target_class=281)  # cat class
cam_dog = grad_cam(input_tensor, target_class=235)  # dog class

# Comparing CAMs reveals which regions each class focuses on
# But there may be some overlap or spurious activations
```

### 3. Global Average Pooling Assumption

Grad-CAM assumes feature map importance is spatially uniform. This may not hold when:
- Different spatial regions of a feature map encode different semantics
- The model uses attention mechanisms internally
- Features have spatially varying importance

### 4. Gradient Saturation

For highly confident predictions, gradients may saturate, leading to weak explanations:

```python
# Very confident predictions (softmax ≈ 1.0) have near-zero gradients
# Solution: Use pre-softmax scores (logits) instead of post-softmax
# This is already the default in our implementation
```

### 5. Adversarial Vulnerability

Grad-CAM explanations can be manipulated by adversarial perturbations:
- Adversarially perturbed images can have misleading Grad-CAMs
- The model may use imperceptible adversarial features
- Explanations may highlight "correct-looking" regions while actual decision features are hidden

## Applications

### Computer Vision

```python
# Standard image classification explanation
cam = grad_cam(image_tensor, target_class=predicted_class)
```

### Document Classification (Finance)

Highlight which regions of financial documents (charts, tables, text) drive classification:

```python
# For a document image classifier
cam = grad_cam(document_image, target_class=class_map["quarterly_report"])
# Visualize which sections (tables, charts, signatures) the model focuses on
```

### Technical Analysis

Visualize which chart patterns a CNN-based trading model focuses on:

```python
# For a candlestick pattern recognizer
cam = grad_cam(chart_image, target_class=class_map["bullish_engulfing"])
# Verify the model looks at the correct candlesticks forming the pattern
```

### Satellite/Alternative Data

Understand what a model sees in satellite imagery for economic indicators:

```python
# For retail traffic prediction from parking lot imagery
cam = grad_cam(parking_lot_image, target_class=0)  # high traffic class
# Verify model focuses on parking spaces, not irrelevant features like shadows
```

### Medical Imaging

```python
# For X-ray diagnosis
cam = grad_cam(xray_image, target_class=class_map["pneumonia"])
# Verify model focuses on lung regions, not artifacts or labels
```

## Comparison with Other Methods

| Method | Resolution | Class-Discriminative | Speed | Theoretical Basis |
|--------|------------|---------------------|-------|-------------------|
| Vanilla Gradient | High | Moderate | Fast | Local sensitivity |
| Grad-CAM | Low | Strong | Fast | Feature importance |
| Guided Backprop | High | Weak | Fast | Modified gradients |
| Guided Grad-CAM | High | Strong | Medium | Combined |
| Integrated Gradients | High | Moderate | Slow | Path integral |
| SHAP | High | Strong | Very slow | Shapley values |

## Summary

Grad-CAM provides interpretable, class-discriminative visualizations for CNN decisions.

### Key Equations

**Importance weights:**
$$
\alpha^c_k = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}
$$

**Heatmap:**
$$
L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_k \alpha^c_k A^k \right)
$$

### Key Properties

- **Class-discriminative**: Different classes → different heatmaps
- **Architecture-agnostic**: Works with any CNN
- **Coarse localization**: Shows "where" not "what"
- **Fast computation**: Single forward-backward pass
- **No retraining**: Works with pre-trained models

### Best Practices

1. **Target the last convolutional layer** for semantic, class-discriminative explanations
2. **Use pre-softmax scores** (logits) to avoid gradient saturation
3. **Compare across classes** to verify discrimination
4. **Combine with pixel-level methods** (Guided Grad-CAM) for detailed insights
5. **Validate with domain expertise** - verify highlighted regions make sense
6. **Report target class and layer** for reproducibility

### When to Use Grad-CAM

**Recommended for:**
- Understanding CNN decisions at a regional level
- Debugging misclassifications
- Verifying models use appropriate image regions
- Quick visualization (single forward-backward pass)

**Consider alternatives when:**
- Fine-grained localization needed → Guided Grad-CAM
- Theoretical guarantees required → Integrated Gradients
- Non-CNN architectures → Attention visualization, SHAP

## References

1. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV 2017*.

2. Zhou, B., et al. (2016). "Learning Deep Features for Discriminative Localization." *CVPR 2016*. (Original CAM paper)

3. Chattopadhyay, A., et al. (2018). "Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks." *WACV 2018*.

4. Adebayo, J., et al. (2018). "Sanity Checks for Saliency Maps." *NeurIPS 2018*.
