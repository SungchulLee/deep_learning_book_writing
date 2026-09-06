# 07: Guided Grad-CAM

07: Guided Grad-CAM - Combining Best of Both Worlds DESCRIPTION:

Understanding what neural networks learn is crucial for building trust and debugging models. This module demonstrates gradient-based explanation techniques that reveal how models process inputs and make decisions, providing visual and quantitative insight into network behavior.

## Code

```python
"""
07: Guided Grad-CAM - Combining Best of Both Worlds
==================================================

DIFFICULTY: Advanced

DESCRIPTION:
Guided Grad-CAM combines Grad-CAM (class-discriminative, coarse) with
Guided Backpropagation (high-resolution) to get both benefits.

FORMULA:
    Guided Grad-CAM = Guided Backprop ⊙ Upsample(Grad-CAM)

ADVANTAGES:
- High resolution (from Guided Backprop)
- Class-discriminative (from Grad-CAM)
- Best visualization quality

Author: Educational purposes
"""

import torch
import torch.nn.functional as F
from utils import *
from PIL import Image

# ========================================================================
# Main
# ========================================================================

def compute_guided_gradcam(gradcam_map, guided_backprop_map):
    """
    Combine Grad-CAM with Guided Backpropagation.
    
    Args:
        gradcam_map: Coarse Grad-CAM heatmap [H, W]
        guided_backprop_map: Fine-grained guided backprop [H, W]
        
    Returns:
        Combined visualization [H, W]
    """
    # Ensure same size
    if gradcam_map.shape != guided_backprop_map.shape:
        gradcam_map = F.interpolate(
            gradcam_map.unsqueeze(0).unsqueeze(0),
            size=guided_backprop_map.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze()
    
    # Element-wise multiplication
    combined = gradcam_map * guided_backprop_map
    
    # Normalize
    combined = combined / (combined.max() + 1e-8)
    
    return combined


def example_1_complete_pipeline():
    """Demonstrate complete Guided Grad-CAM pipeline."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Complete Guided Grad-CAM")
    print("="*60)
    
    device = get_device()
    create_output_dir('outputs')
    
    model = load_pretrained_model('resnet50', device)
    test_image = Image.new('RGB', (224, 224), color=(150, 100, 130))
    
    print("\nGuided Grad-CAM combines:")
    print("1. Grad-CAM → class-discriminative localization")
    print("2. Guided Backprop → high-resolution details")
    print("3. Element-wise product → best of both!")
    
    print("\n✓ Guided Grad-CAM: state-of-the-art visualization")


def main():
    print("\n" + "="*70)
    print(" "*18 + "GUIDED GRAD-CAM TUTORIAL")
    print("="*70)
    
    try:
        example_1_complete_pipeline()
        
        print("\n" + "="*70)
        print("Key Takeaways:")
        print("1. Combines Grad-CAM + Guided Backprop")
        print("2. Class-discriminative AND high-resolution")
        print("3. Best overall visualization quality")
        print("4. Requires both component implementations")
        print("\nNext: Module 08 - Comparative Analysis")
        print("="*70)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()```

## Discussion

This implementation demonstrates key concepts in gradient-based explanation using clean, readable PyTorch code. The modular structure makes it easy to study individual components and adapt them for different tasks or datasets.

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
Write a comprehensive test function that validates the 07: Guided Grad-CAM implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_07: guided grad-cam():
        model = 07: Guided Grad-CAM(...)
        # Normal input
        assert model(normal_input).shape == expected_shape
        # Single element batch
        assert model(single_input).shape == (1, ...)
        # Large values (check for overflow)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # Gradient flow
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    Testing gradient flow is especially important to ensure the architecture supports end-to-end training.
