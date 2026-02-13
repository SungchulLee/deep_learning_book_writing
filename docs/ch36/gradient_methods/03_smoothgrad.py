"""
03: SmoothGrad - Noise Reduction for Saliency Maps
=================================================

DIFFICULTY: Beginner

DESCRIPTION:
SmoothGrad reduces visual noise in saliency maps by averaging gradients
computed on noisy versions of the input. Adding noise and averaging
cancels out sharp, unstable gradients while preserving meaningful signals.

MATHEMATICAL FOUNDATION:
    SmoothGrad(x) = (1/n) Σᵢ₌₁ⁿ |∂y_c/∂(x + N(0, σ²))|

Where:
- n: number of noisy samples
- N(0, σ²): Gaussian noise with standard deviation σ
- Averaging smooths out gradient noise

KEY PARAMETERS:
- n (num_samples): More samples → smoother but slower (typical: 20-50)
- σ (noise_level): Controls noise magnitude (typical: 0.1-0.2 of input scale)

Author: Educational purposes
"""

import torch
import torch.nn as nn
import numpy as np
from utils import *
from PIL import Image

def compute_smoothgrad(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    num_samples: int = 25,
    noise_level: float = 0.15
) -> torch.Tensor:
    """
    Compute SmoothGrad saliency map.
    
    ALGORITHM:
    1. For i = 1 to n:
       a. Add Gaussian noise to input: x_noisy = x + N(0, σ²)
       b. Compute gradient: gᵢ = |∂y_c/∂x_noisy|
    2. Average gradients: SmoothGrad = (1/n) Σᵢ gᵢ
    
    WHY IT WORKS:
    - Sharp, unstable gradients vary randomly with noise
    - Meaningful gradients remain consistent
    - Averaging cancels noise, preserves signal
    - Similar to ensemble methods in ML
    
    Args:
        model: Pretrained model
        image_tensor: Input image [1, 3, H, W]
        target_class: Target class index  
        device: Computation device
        num_samples: Number of noisy samples
        noise_level: Std of Gaussian noise (fraction of input scale)
        
    Returns:
        torch.Tensor: Smoothed saliency map [1, H, W]
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # Accumulator for gradients
    accumulated_gradients = torch.zeros_like(image_tensor)
    
    # Compute noise standard deviation
    # Scale by typical input range (after normalization)
    noise_std = noise_level
    
    for i in range(num_samples):
        # Create noisy version of input
        noise = torch.randn_like(image_tensor) * noise_std
        noisy_image = image_tensor + noise
        noisy_image.requires_grad = True
        
        # Forward pass
        output = model(noisy_image)
        target_score = output[0, target_class]
        
        # Backward pass
        model.zero_grad()
        target_score.backward()
        
        # Accumulate gradients
        accumulated_gradients += noisy_image.grad
    
    # Average the gradients
    mean_gradients = accumulated_gradients / num_samples
    
    # Take absolute value and aggregate across channels
    abs_gradients = torch.abs(mean_gradients)
    saliency = torch.max(abs_gradients, dim=1)[0]
    
    return saliency


def example_1_smoothgrad_vs_vanilla():
    """Compare SmoothGrad with vanilla gradients."""
    print("\n" + "="*60)
    print("EXAMPLE 1: SmoothGrad vs Vanilla Gradients")
    print("="*60)
    
    device = get_device()
    create_output_dir('outputs')
    model = load_pretrained_model('resnet50', device)
    
    # Create noisy test image
    img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    image_tensor = preprocess_image(test_image, requires_grad=True)
    
    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()
    
    print(f"\nTarget class: {target_class}")
    print("Computing vanilla gradient...")
    
    # Vanilla gradient
    image_vanilla = preprocess_image(test_image, requires_grad=True)
    output = model(image_vanilla.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(image_vanilla.grad), dim=1)[0]
    
    print("Computing SmoothGrad (this takes longer)...")
    # SmoothGrad
    image_smooth = preprocess_image(test_image, requires_grad=False)
    smoothgrad = compute_smoothgrad(
        model, image_smooth, target_class, device,
        num_samples=25, noise_level=0.15
    )
    
    # Visualize
    visualize_multiple_saliencies(
        image_tensor,
        {
            'Vanilla\n(Noisy)': vanilla,
            'SmoothGrad\n(Clean)': smoothgrad
        },
        save_path='outputs/03_smoothgrad_comparison.png'
    )
    
    print("\n✓ SmoothGrad produces cleaner visualizations!")


def example_2_parameter_sensitivity():
    """Explore effect of num_samples and noise_level."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Parameter Sensitivity")
    print("="*60)
    
    device = get_device()
    model = load_pretrained_model('resnet50', device)
    
    test_image = Image.new('RGB', (224, 224), color=(120, 150, 180))
    image_tensor = preprocess_image(test_image, requires_grad=False)
    
    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()
    
    # Test different parameters
    configs = [
        (10, 0.10, "n=10, σ=0.10"),
        (25, 0.15, "n=25, σ=0.15"),
        (50, 0.20, "n=50, σ=0.20"),
    ]
    
    saliencies = {}
    for num_samples, noise_level, label in configs:
        print(f"Computing {label}...")
        sal = compute_smoothgrad(
            model, image_tensor, target_class, device,
            num_samples, noise_level
        )
        saliencies[label] = sal
    
    visualize_multiple_saliencies(
        image_tensor, saliencies,
        save_path='outputs/03_parameter_comparison.png'
    )
    
    print("\nOBSERVATIONS:")
    print("- More samples → smoother but slower")
    print("- Higher noise → more smoothing")
    print("- Trade-off between detail and cleanliness")
    print("\n✓ Typical settings: n=25, σ=0.15")


def main():
    print("\n" + "="*70)
    print(" "*20 + "SMOOTHGRAD TUTORIAL")
    print("="*70)
    
    try:
        example_1_smoothgrad_vs_vanilla()
        example_2_parameter_sensitivity()
        
        print("\n" + "="*70)
        print("Key Takeaways:")
        print("1. SmoothGrad reduces visual noise via averaging")
        print("2. Adding noise paradoxically reduces noise!")
        print("3. Parameters: n=20-50, σ=0.10-0.20")
        print("4. Computationally expensive (n forward/backward passes)")
        print("\nNext: Module 04 - Integrated Gradients")
        print("="*70)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
