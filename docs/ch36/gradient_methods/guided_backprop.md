# Guided Backpropagation

## Introduction

**Guided Backpropagation** is a visualization technique that produces high-resolution, sharp saliency maps by modifying how gradients flow backward through ReLU activation functions. Unlike standard backpropagation, Guided Backpropagation only propagates positive gradients through neurons that were activated during the forward pass.

Introduced by Springenberg et al. (2015), this method generates visually appealing, detailed visualizations that highlight fine-grained features relevant to the prediction.

## Mathematical Foundation

### Standard ReLU Backward Pass

In standard backpropagation through a ReLU layer:

**Forward pass:**
$$
y = \text{ReLU}(x) = \max(0, x)
$$

**Standard backward pass:**
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \mathbf{1}[x > 0]
$$

### Guided Backpropagation Modification

Guided Backpropagation adds an additional constraint - only propagate positive gradients:

$$
\frac{\partial L}{\partial x}\bigg|_{\text{guided}} = \frac{\partial L}{\partial y} \cdot \mathbf{1}[x > 0] \cdot \mathbf{1}\left[\frac{\partial L}{\partial y} > 0\right]
$$

This combines:
1. **Forward mask**: Only where neuron was active
2. **Backward mask**: Only positive gradients

## PyTorch Implementation

### Custom ReLU with Guided Backward

```python
import torch
import torch.nn as nn
from torch.autograd import Function

class GuidedReLU(Function):
    """ReLU with guided backpropagation."""
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Forward mask: input > 0
        # Backward mask: grad_output > 0
        return grad_output * (input > 0).float() * (grad_output > 0).float()


class GuidedBackpropagation:
    """Guided Backpropagation implementation."""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Replace ReLU forward with guided version."""
        def guided_relu_hook(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0),)
        
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_backward_hook(guided_relu_hook)
                self.hooks.append(hook)
    
    def __call__(self, image_tensor, target_class, device):
        """Compute guided backpropagation saliency."""
        self.model.eval()
        image_tensor = image_tensor.to(device).requires_grad_(True)
        
        output = self.model(image_tensor)
        self.model.zero_grad()
        output[0, target_class].backward()
        
        saliency = image_tensor.grad.abs().max(dim=1)[0]
        return saliency
    
    def remove_hooks(self):
        """Clean up registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
```

### Usage Example

```python
# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True).to(device).eval()

# Create guided backprop instance
guided_bp = GuidedBackpropagation(model)

# Load and preprocess image
image_tensor = preprocess_image('dog.jpg').unsqueeze(0)

# Get prediction
with torch.no_grad():
    pred_class = model(image_tensor.to(device)).argmax(dim=1).item()

# Compute guided backpropagation
saliency = guided_bp(image_tensor, pred_class, device)

# Clean up
guided_bp.remove_hooks()
```

## Guided Grad-CAM

Combining Guided Backpropagation with Grad-CAM produces the best of both worlds:

- **Grad-CAM**: Coarse, class-discriminative localization
- **Guided Backprop**: High-resolution, detailed features

$$
\text{Guided Grad-CAM} = \text{Guided Backprop} \odot \text{Upsample}(\text{Grad-CAM})
$$

```python
def compute_guided_gradcam(
    model, target_layer, image_tensor, target_class, device
):
    """Compute Guided Grad-CAM."""
    # Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(image_tensor, target_class, device)
    
    # Upsample to input size
    cam_upsampled = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=image_tensor.shape[2:],
        mode='bilinear'
    ).squeeze()
    
    # Guided Backpropagation
    guided_bp = GuidedBackpropagation(model)
    guided = guided_bp(image_tensor, target_class, device)
    guided_bp.remove_hooks()
    
    # Element-wise multiplication
    guided_gradcam = guided.squeeze() * cam_upsampled
    
    return guided_gradcam, guided.squeeze(), cam
```

## Comparison with Other Methods

| Method | Resolution | Class-Discriminative | Visual Quality |
|--------|------------|---------------------|----------------|
| Vanilla Gradient | High | Low | Noisy |
| Grad-CAM | Low | High | Smooth but coarse |
| Guided Backprop | High | Low | Sharp, detailed |
| Guided Grad-CAM | High | High | Best overall |

## Limitations

1. **Not class-discriminative alone**: Similar patterns for different classes
2. **Sanity check concerns**: May act as edge detector rather than true attribution
3. **ReLU-specific**: Only works with ReLU activations

## Summary

Guided Backpropagation produces sharp, detailed visualizations by masking negative gradients during backpropagation. Combined with Grad-CAM, it creates Guided Grad-CAM - providing both high resolution and class discrimination.

**Key equation:**
$$
\frac{\partial L}{\partial x}\bigg|_{\text{guided}} = \frac{\partial L}{\partial y} \cdot \mathbf{1}[x > 0] \cdot \mathbf{1}\left[\frac{\partial L}{\partial y} > 0\right]
$$

## References

1. Springenberg, J.T., et al. (2015). *Striving for Simplicity: The All Convolutional Net*. ICLR Workshop.

2. Selvaraju, R.R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV.
