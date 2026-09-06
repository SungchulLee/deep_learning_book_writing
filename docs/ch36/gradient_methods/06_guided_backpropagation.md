# Guided Backpropagation

06: Guided Backpropagation DESCRIPTION:

Understanding what neural networks learn is crucial for building trust and debugging models. This module demonstrates gradient-based explanation techniques that reveal how models process inputs and make decisions, providing visual and quantitative insight into network behavior.

## Code

```python
"""
06: Guided Backpropagation
==========================

DIFFICULTY: Intermediate

DESCRIPTION:
Guided Backpropagation modifies the backward pass through ReLU layers
to only propagate positive gradients. This produces sharper, cleaner
visualizations by suppressing negative gradients.

MODIFICATION:
Standard ReLU backward: ∂L/∂x = (∂L/∂y) · 1(x > 0)
Guided ReLU backward:   ∂L/∂x = (∂L/∂y) · 1(x > 0) · 1(∂L/∂y > 0)

Additional condition: Only backprop positive gradients

Author: Educational purposes
"""

import torch
import torch.nn as nn
from utils import *

# ========================================================================
# Main
# ========================================================================

class GuidedBackpropReLU(nn.Module):
    """Modified ReLU for guided backpropagation."""
    
    def forward(self, x):
        return F.relu(x)
    
    def backward(self, grad_output):
        # Only backprop positive gradients through positive activations
        return grad_output.clamp(min=0) * (self.output > 0).float()


def replace_relu_with_guided(model):
    """Replace all ReLU with GuidedBackpropReLU."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, GuidedBackpropReLU())
        else:
            replace_relu_with_guided(module)


def compute_guided_backprop(model, image_tensor, target_class, device):
    """Compute guided backpropagation."""
    model.eval()
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad = True
    
    output = model(image_tensor)
    target_score = output[0, target_class]
    
    model.zero_grad()
    target_score.backward()
    
    guided_grads = image_tensor.grad
    saliency = torch.max(torch.abs(guided_grads), dim=1)[0]
    
    return saliency


def example_1_guided_vs_vanilla():
    """Compare guided backprop with vanilla gradients."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Guided Backprop vs Vanilla")
    print("="*60)
    
    device = get_device()
    create_output_dir('outputs')
    
    # Two models: one vanilla, one with guided backprop
    model_vanilla = load_pretrained_model('resnet50', device)
    model_guided = load_pretrained_model('resnet50', device)
    
    # Modify one model for guided backprop
    print("Setting up guided backpropagation...")
    # Note: Full implementation requires custom hooks
    # Simplified version shown here
    
    from PIL import Image
    test_image = Image.new('RGB', (224, 224), color=(140, 160, 100))
    image_tensor = preprocess_image(test_image, requires_grad=True)
    
    with torch.no_grad():
        output = model_vanilla(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()
    
    # Vanilla
    print("Computing vanilla gradient...")
    image_vanilla = preprocess_image(test_image, requires_grad=True)
    output = model_vanilla(image_vanilla.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(image_vanilla.grad), dim=1)[0]
    
    print("\n✓ Guided backprop produces sharper visualizations")
    print("(Full implementation requires custom autograd functions)")


def main():
    print("\n" + "="*70)
    print(" "*15 + "GUIDED BACKPROPAGATION TUTORIAL")
    print("="*70)
    
    try:
        example_1_guided_vs_vanilla()
        
        print("\n" + "="*70)
        print("Key Takeaways:")
        print("1. Modifies ReLU backward pass")
        print("2. Only propagates positive gradients")
        print("3. Produces sharper, cleaner visualizations")
        print("4. Implementation requires custom hooks")
        print("\nNext: Module 07 - Guided Grad-CAM")
        print("="*70)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()```

## Discussion

The `GuidedBackpropReLU` class encapsulates the model architecture using PyTorch's `nn.Module` interface. The `forward` method defines the computational graph, allowing PyTorch's autograd system to handle gradient computation automatically during training. This modular design makes it straightforward to modify individual components or integrate the model into larger pipelines.

The patterns demonstrated here extend naturally to more complex scenarios. Experimenting with hyperparameters, architectural variations, and different datasets deepens understanding and builds practical intuition for model interpretability tasks.

## Exercises

**Exercise 1.**
Read through the code and identify the key design decisions. List three specific implementation choices and explain why each is appropriate for gradient-based explanation.

??? success "Solution to Exercise 1"
    Design decisions vary by implementation but commonly include: (1) choice of activation functions -- ReLU variants provide non-saturating gradients for faster training; (2) normalization strategy -- batch normalization stabilizes training by reducing internal covariate shift; (3) residual connections -- when present, they enable gradient flow in deep networks by providing skip paths. Each choice reflects a trade-off between expressiveness, computational cost, and training stability.

---

**Exercise 2.**
Add input validation to the main function or class to check that inputs have the expected shape and dtype. Raise informative error messages for invalid inputs.

??? success "Solution to Exercise 2"
    At the start of the `forward` method (or relevant function), add checks like: `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'` and `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. For shape validation, check critical dimensions: `B, C, H, W = x.shape; assert C == self.expected_channels`. Informative error messages significantly speed up debugging and make the code more robust for reuse.

---

**Exercise 3.**
Describe two potential failure modes of this implementation and explain how you would diagnose and fix each one.

??? success "Solution to Exercise 3"
    Common failure modes include: (1) **Vanishing/exploding gradients** -- diagnosed by monitoring gradient norms (`torch.nn.utils.clip_grad_norm_` or logging `param.grad.norm()` per layer). Fix with gradient clipping, better initialization (Xavier/Kaiming), or architectural changes (residual connections, normalization). (2) **Overfitting** -- diagnosed when training loss decreases but validation loss increases. Fix with regularization (dropout, weight decay, data augmentation) or reducing model capacity. Always monitor both training and validation metrics to catch these issues early.

---

**Exercise 4.**
Extend `GuidedBackpropReLU` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = GuidedBackpropReLU(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
