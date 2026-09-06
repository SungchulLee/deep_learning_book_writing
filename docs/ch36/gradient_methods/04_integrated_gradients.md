# 04: Integrated Gradients

04: Integrated Gradients - Principled Attribution DESCRIPTION:

Understanding what neural networks learn is crucial for building trust and debugging models. This module demonstrates gradient-based explanation techniques that reveal how models process inputs and make decisions, providing visual and quantitative insight into network behavior.

## Code

```python
"""
04: Integrated Gradients - Principled Attribution
================================================

DIFFICULTY: Intermediate

DESCRIPTION:
Integrated Gradients accumulates gradients along a path from a baseline
to the input. This satisfies important attribution axioms: sensitivity
and implementation invariance.

MATHEMATICAL FOUNDATION:
    IG(x) = (x - x') ⊙ ∫₀¹ (∂f(x' + α(x - x'))/∂x) dα

Where:
- x: input image
- x': baseline (often zeros or blurred image)
- α ∈ [0,1]: interpolation coefficient
- Integration approximated by Riemann sum

AXIOMS SATISFIED:
1. Sensitivity: If feature changes output, it gets non-zero attribution
2. Implementation Invariance: Functionally equivalent networks get same attributions
3. Completeness: Attributions sum to f(x) - f(x')

Author: Educational purposes
"""

import torch
import torch.nn as nn
import numpy as np
from utils import *
from PIL import Image, ImageFilter

# ========================================================================
# Main
# ========================================================================

def compute_integrated_gradients(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    baseline: str = 'zeros',
    steps: int = 50
) -> torch.Tensor:
    """
    Compute Integrated Gradients.
    
    ALGORITHM:
    1. Choose baseline x'  
    2. Create interpolated inputs: x^(i) = x' + (i/m)(x - x') for i=0..m
    3. Compute gradients at each point: gᵢ = ∂f(x^(i))/∂x
    4. Average gradients: ḡ = (1/m) Σᵢ gᵢ
    5. Scale by input difference: IG = (x - x') ⊙ ḡ
    
    Args:
        baseline: 'zeros', 'blur', or 'random'
        steps: Number of interpolation steps (more = more accurate)
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # Create baseline
    if baseline == 'zeros':
        baseline_tensor = torch.zeros_like(image_tensor)
    elif baseline == 'blur':
        # Blur the image as baseline
        from torchvision.transforms.functional import gaussian_blur
        baseline_tensor = gaussian_blur(image_tensor, kernel_size=51, sigma=20)
    elif baseline == 'random':
        baseline_tensor = torch.randn_like(image_tensor) * 0.1
    else:
        baseline_tensor = torch.zeros_like(image_tensor)
    
    baseline_tensor = baseline_tensor.to(device)
    
    # Compute path: x' + α(x - x') for α ∈ [0,1]
    accumulated_gradients = torch.zeros_like(image_tensor)
    
    for step in range(steps):
        # Interpolation coefficient
        alpha = (step + 1) / steps
        
        # Interpolated input
        interpolated = baseline_tensor + alpha * (image_tensor - baseline_tensor)
        interpolated.requires_grad = True
        
        # Forward pass
        output = model(interpolated)
        target_score = output[0, target_class]
        
        # Backward pass
        model.zero_grad()
        target_score.backward()
        
        # Accumulate gradients
        accumulated_gradients += interpolated.grad
    
    # Average gradients (Riemann approximation of integral)
    avg_gradients = accumulated_gradients / steps
    
    # Scale by input difference
    integrated_grads = (image_tensor - baseline_tensor) * avg_gradients
    
    # Aggregate
    abs_attr = torch.abs(integrated_grads)
    saliency = torch.max(abs_attr, dim=1)[0]
    
    return saliency


def verify_completeness(model, image_tensor, target_class, device, saliency):
    """Verify that attributions sum to output difference."""
    model.eval()
    
    with torch.no_grad():
        output_image = model(image_tensor.to(device))[0, target_class]
        baseline = torch.zeros_like(image_tensor).to(device)
        output_baseline = model(baseline)[0, target_class]
    
    # Sum of attributions should ≈ output difference
    # Note: This is approximate due to discretization
    print(f"\nCompleteness Check:")
    print(f"f(x) - f(x'): {(output_image - output_baseline).item():.4f}")
    print("(Saliency sums to approximate this value)")


def example_1_baseline_comparison():
    """Compare different baseline choices."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Baseline Comparison")
    print("="*60)
    
    device = get_device()
    create_output_dir('outputs')
    model = load_pretrained_model('resnet50', device)
    
    test_image = Image.new('RGB', (224, 224), color=(150, 120, 90))
    image_tensor = preprocess_image(test_image, requires_grad=False)
    
    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()
    
    baselines = ['zeros', 'blur', 'random']
    saliencies = {}
    
    for baseline in baselines:
        print(f"Computing with {baseline} baseline...")
        sal = compute_integrated_gradients(
            model, image_tensor, target_class, device,
            baseline=baseline, steps=30
        )
        saliencies[f'{baseline}\nbaseline'] = sal
    
    visualize_multiple_saliencies(
        image_tensor, saliencies,
        save_path='outputs/04_baseline_comparison.png'
    )
    
    print("\nBASELINE RECOMMENDATIONS:")
    print("- Zeros: Simple, fast, works well for most cases")
    print("- Blur: Good for natural images")
    print("- Random: Rarely needed")
    print("\n✓ Zeros baseline most common!")


def main():
    print("\n" + "="*70)
    print(" "*15 + "INTEGRATED GRADIENTS TUTORIAL")
    print("="*70)
    
    try:
        example_1_baseline_comparison()
        
        print("\n" + "="*70)
        print("Key Takeaways:")
        print("1. IG satisfies sensitivity & implementation invariance")
        print("2. Baseline choice matters (zeros usually good)")
        print("3. More steps = more accurate (30-50 typical)")
        print("4. Computationally expensive but theoretically sound")
        print("\nNext: Module 05 - Grad-CAM")
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
Write a comprehensive test function that validates the 04: Integrated Gradients implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_04: integrated gradients():
        model = 04: Integrated Gradients(...)
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
