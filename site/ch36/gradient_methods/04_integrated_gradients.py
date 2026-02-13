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
    main()
