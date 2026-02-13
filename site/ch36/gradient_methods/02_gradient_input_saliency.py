"""
02: Gradient × Input Saliency Maps
==================================

DIFFICULTY: Beginner

DESCRIPTION:
This module introduces an improved saliency method that multiplies gradients
by the input values. This addresses a limitation of vanilla gradients: they
ignore the magnitude of input pixels. A pixel with high gradient but zero
input value shouldn't be considered important.

MATHEMATICAL FOUNDATION:
For input x and output y_c:
    Saliency(x) = x ⊙ (∂y_c/∂x)
    
Where:
- ⊙ denotes element-wise multiplication
- x: input pixel values
- ∂y_c/∂x: gradient with respect to input

INTUITION:
- Gradient alone: "How sensitive is output to this pixel?"
- Gradient × Input: "How much does this pixel's actual value contribute?"

ADVANTAGES OVER VANILLA GRADIENTS:
1. Accounts for input magnitude
2. Better attribution to actual features present in image
3. Satisfies certain desirable axioms (sensitivity)

LEARNING OBJECTIVES:
1. Understand why input magnitude matters
2. Implement Gradient × Input method
3. Compare with vanilla gradients
4. Understand attribution vs sensitivity

Author: Educational purposes
Date: 2025
"""

import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    preprocess_image,
    visualize_saliency,
    visualize_multiple_saliencies,
    load_pretrained_model,
    get_device,
    create_output_dir
)
from PIL import Image


# ============================================================================
# GRADIENT × INPUT IMPLEMENTATION
# ============================================================================

def compute_gradient_times_input_saliency(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute Gradient × Input saliency map.
    
    ALGORITHM:
    1. Forward pass: Get model predictions
    2. Backward pass: Compute gradients ∂y_c/∂x
    3. Multiply gradients by input: x ⊙ ∂y_c/∂x
    4. Aggregate across color channels
    
    Args:
        model: Pretrained neural network
        image_tensor: Input image [1, 3, H, W] with requires_grad=True
        target_class: Target class index
        device: Computation device
        
    Returns:
        torch.Tensor: Saliency map [1, H, W]
        
    MATHEMATICAL INSIGHT:
    Consider a linear model: y = w^T x
    - Gradient: ∂y/∂x = w (just the weights)
    - Gradient × Input: w ⊙ x (weighted contribution)
    
    For linear models, Gradient × Input gives exact attribution!
    For neural networks, it provides a better approximation than gradients alone.
    
    AXIOM: Sensitivity
    If changing a pixel from baseline x' to x changes the output,
    then that pixel should have non-zero attribution.
    Gradient × Input satisfies this when x' = 0.
    
    Example:
        >>> model = models.resnet50(pretrained=True)
        >>> image = preprocess_image('dog.jpg', requires_grad=True)
        >>> saliency = compute_gradient_times_input_saliency(
        ...     model, image, target_class=207, device=device
        ... )
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # Zero existing gradients
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()
    
    # Forward pass
    output = model(image_tensor)
    target_score = output[0, target_class]
    
    # Backward pass to get gradients
    target_score.backward()
    gradients = image_tensor.grad  # ∂y_c/∂x
    
    # KEY STEP: Multiply gradient by input
    # This gives attribution: how much does each pixel value contribute?
    attribution = image_tensor * gradients  # x ⊙ ∂y_c/∂x
    
    # Take absolute value (we care about magnitude of contribution)
    abs_attribution = torch.abs(attribution)
    
    # Aggregate across color channels (max)
    saliency = torch.max(abs_attribution, dim=1)[0]  # [1, H, W]
    
    return saliency


def compare_vanilla_vs_gradient_input(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device
) -> tuple:
    """
    Compute both vanilla gradient and gradient × input for comparison.
    
    Returns:
        tuple: (vanilla_saliency, gradient_input_saliency, gradients)
        
    This allows side-by-side comparison to understand the difference.
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # Zero gradients
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()
    
    # Forward and backward pass
    output = model(image_tensor)
    target_score = output[0, target_class]
    target_score.backward()
    
    # Get gradients
    gradients = image_tensor.grad
    
    # Vanilla gradient saliency
    abs_gradients = torch.abs(gradients)
    vanilla_saliency = torch.max(abs_gradients, dim=1)[0]
    
    # Gradient × Input saliency
    attribution = image_tensor * gradients
    abs_attribution = torch.abs(attribution)
    gradient_input_saliency = torch.max(abs_attribution, dim=1)[0]
    
    return vanilla_saliency, gradient_input_saliency, gradients


# ============================================================================
# THEORETICAL ANALYSIS
# ============================================================================

def demonstrate_linear_case():
    """
    THEORETICAL DEMONSTRATION: Linear model case
    
    Shows that for linear models, Gradient × Input gives exact attribution.
    
    SETUP:
    - Linear model: y = w^T x + b
    - Input: x ∈ ℝ^d
    - Weights: w ∈ ℝ^d
    - Output: y ∈ ℝ
    
    ANALYSIS:
    - Gradient: ∂y/∂x = w
    - Gradient × Input: w ⊙ x
    - Sum: Σᵢ (wᵢ · xᵢ) = w^T x = y - b (exact!)
    
    This shows Gradient × Input is the "correct" attribution for linear models.
    """
    print("\n" + "="*60)
    print("THEORETICAL DEMONSTRATION: Linear Model")
    print("="*60)
    
    # Create simple linear model
    torch.manual_seed(42)
    d = 5  # dimension
    
    # Linear model parameters
    w = torch.randn(d)  # weights
    b = torch.randn(1)  # bias
    
    # Input
    x = torch.randn(d, requires_grad=True)
    
    # Forward pass: y = w^T x + b
    y = torch.dot(w, x) + b
    
    # Backward pass: get gradient
    y.backward()
    gradient = x.grad  # Should equal w
    
    # Compute attributions
    gradient_times_input = gradient * x
    
    print("\nLinear model: y = w^T x + b")
    print(f"\nWeights w: {w.numpy()}")
    print(f"Input x: {x.detach().numpy()}")
    print(f"Bias b: {b.item():.4f}")
    print(f"\nOutput y: {y.item():.4f}")
    
    print(f"\nGradient ∂y/∂x: {gradient.numpy()}")
    print(f"(Should equal w: {np.allclose(gradient.numpy(), w.numpy())})")
    
    print(f"\nGradient × Input: {gradient_times_input.detach().numpy()}")
    print(f"Sum of attributions: {gradient_times_input.sum().item():.4f}")
    print(f"y - b (expected): {(y - b).item():.4f}")
    print(f"Match: {np.isclose(gradient_times_input.sum().item(), (y - b).item())}")
    
    print("\n" + "-"*60)
    print("INSIGHT: For linear models, Gradient × Input attributions")
    print("sum exactly to the output (minus bias)!")
    print("This is the 'completeness' property.")
    print("="*60)


def demonstrate_saturation_issue():
    """
    DEMONSTRATION: Why vanilla gradients can fail
    
    Shows a case where vanilla gradients give misleading results
    but Gradient × Input works correctly.
    
    SCENARIO:
    - ReLU activation: y = max(0, w^T x)
    - If w^T x >> 0 (deep in saturation), gradient ≈ 0
    - But input x is clearly important!
    
    Gradient × Input handles this better by incorporating x itself.
    """
    print("\n" + "="*60)
    print("DEMONSTRATION: Saturation Issue")
    print("="*60)
    
    torch.manual_seed(42)
    
    # Simple model with ReLU
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    w = torch.tensor([1.0, 1.0])
    
    # Forward: y = ReLU(w^T x) = ReLU(5)
    z = torch.dot(w, x)  # z = 5
    y = torch.relu(z)
    
    # Backward
    y.backward()
    gradient = x.grad
    
    # Compute saliency
    vanilla = torch.abs(gradient)
    grad_times_input = torch.abs(gradient * x)
    
    print("\nSetup:")
    print(f"Input x: {x.detach().numpy()}")
    print(f"Weights w: {w.numpy()}")
    print(f"Pre-activation z: {z.item():.2f}")
    print(f"Output y = ReLU(z): {y.item():.2f}")
    
    print(f"\nVanilla gradient |∂y/∂x|: {vanilla.detach().numpy()}")
    print(f"Gradient × Input |x ⊙ ∂y/∂x|: {grad_times_input.detach().numpy()}")
    
    print("\nOBSERVATION:")
    print("Both methods give same result here (ReLU not saturated).")
    print("In deep saturation, vanilla gradients → 0, but")
    print("Gradient × Input still reflects input importance.")
    print("="*60)


# ============================================================================
# VISUAL COMPARISONS
# ============================================================================

def example_1_basic_comparison():
    """
    EXAMPLE 1: Basic comparison of methods
    
    Shows vanilla gradient vs Gradient × Input on a real image.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Vanilla vs Gradient × Input")
    print("="*60)
    
    device = get_device()
    create_output_dir('outputs')
    
    # Load model
    model = load_pretrained_model('resnet50', device)
    
    # Create test image
    test_image = Image.new('RGB', (224, 224))
    # Add some structure
    import numpy as np
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    
    image_tensor = preprocess_image(test_image, requires_grad=True)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()
    
    print(f"\nTarget class: {target_class}")
    print("\nComputing saliency maps...")
    
    # Compute both methods
    vanilla, grad_input, _ = compare_vanilla_vs_gradient_input(
        model, image_tensor, target_class, device
    )
    
    # Visualize comparison
    saliencies = {
        'Vanilla Gradient\n|∂y/∂x|': vanilla,
        'Gradient × Input\n|x ⊙ ∂y/∂x|': grad_input
    }
    
    visualize_multiple_saliencies(
        image_tensor,
        saliencies,
        save_path='outputs/02_vanilla_vs_grad_input.png'
    )
    
    # Compare statistics
    print("\nStatistics Comparison:")
    print("-" * 40)
    
    vanilla_np = vanilla.detach().cpu().numpy().flatten()
    grad_input_np = grad_input.detach().cpu().numpy().flatten()
    
    print(f"{'Method':<20} {'Mean':<12} {'Max':<12} {'Std':<12}")
    print("-" * 40)
    print(f"{'Vanilla Gradient':<20} {vanilla_np.mean():<12.6f} {vanilla_np.max():<12.6f} {vanilla_np.std():<12.6f}")
    print(f"{'Gradient × Input':<20} {grad_input_np.mean():<12.6f} {grad_input_np.max():<12.6f} {grad_input_np.std():<12.6f}")
    
    print("\n✓ Example 1 completed!")


def example_2_normalized_image_effect():
    """
    EXAMPLE 2: Effect of image normalization
    
    Shows how ImageNet normalization affects the two methods differently.
    
    KEY INSIGHT:
    Since inputs are normalized (mean ≈ 0), multiplying by gradients
    can give very different results than gradients alone.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Effect of Normalization")
    print("="*60)
    
    device = get_device()
    model = load_pretrained_model('resnet50', device)
    
    # Create simple colored image
    test_image = Image.new('RGB', (224, 224), color=(180, 100, 50))
    image_tensor = preprocess_image(test_image, requires_grad=True)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()
    
    print(f"\nTarget class: {target_class}")
    
    # Compute both methods
    vanilla, grad_input, gradients = compare_vanilla_vs_gradient_input(
        model, image_tensor, target_class, device
    )
    
    # Analyze normalized input values
    print("\nNormalized Input Statistics:")
    img_np = image_tensor.detach().cpu().numpy()[0]
    print(f"Mean per channel: {img_np.mean(axis=(1,2))}")
    print(f"Std per channel: {img_np.std(axis=(1,2))}")
    print(f"Min: {img_np.min():.3f}, Max: {img_np.max():.3f}")
    
    print("\nGradient Statistics:")
    grad_np = gradients.detach().cpu().numpy()[0]
    print(f"Mean per channel: {grad_np.mean(axis=(1,2))}")
    print(f"Std per channel: {grad_np.std(axis=(1,2))}")
    
    print("\nNote: Normalized inputs can be negative!")
    print("This affects Gradient × Input differently than vanilla gradients.")
    
    print("\n✓ Example 2 completed!")


def example_3_bright_vs_dark_regions():
    """
    EXAMPLE 3: Bright vs dark regions
    
    Creates an image with bright and dark regions to show how
    Gradient × Input accounts for pixel intensity.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Bright vs Dark Regions")
    print("="*60)
    
    device = get_device()
    model = load_pretrained_model('resnet50', device)
    
    # Create image with bright and dark regions
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    # Bright region (top half)
    img_array[:112, :, :] = 200
    # Dark region (bottom half)
    img_array[112:, :, :] = 50
    
    test_image = Image.fromarray(img_array)
    image_tensor = preprocess_image(test_image, requires_grad=True)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()
    
    # Compute both methods
    vanilla, grad_input, _ = compare_vanilla_vs_gradient_input(
        model, image_tensor, target_class, device
    )
    
    # Visualize
    saliencies = {
        'Vanilla\n(ignores intensity)': vanilla,
        'Grad × Input\n(accounts for intensity)': grad_input
    }
    
    visualize_multiple_saliencies(
        image_tensor,
        saliencies,
        save_path='outputs/02_bright_vs_dark.png'
    )
    
    print("\nOBSERVATION:")
    print("Gradient × Input may show different patterns in bright vs dark")
    print("regions, accounting for actual pixel intensities.")
    
    print("\n✓ Example 3 completed!")


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1():
    """
    EXERCISE 1: Implement completeness check
    
    Task: Verify that gradient × input attributions sum to approximately
    the output value for a linear layer.
    
    HINT: For a linear layer y = Wx + b, the sum of attributions
    should equal y - b.
    """
    print("\n" + "="*60)
    print("EXERCISE 1: Completeness Check")
    print("="*60)
    
    # Create simple linear layer
    torch.manual_seed(42)
    linear = nn.Linear(10, 1, bias=True)
    x = torch.randn(1, 10, requires_grad=True)
    
    # Forward
    y = linear(x)
    
    # Backward
    y.backward()
    
    # YOUR CODE HERE:
    # 1. Get the gradient
    # gradient = ___
    
    # 2. Compute gradient × input
    # attribution = ___
    
    # 3. Sum the attributions
    # attribution_sum = ___
    
    # 4. Compute y - bias
    # output_minus_bias = y.item() - linear.bias.item()
    
    # 5. Check if they match
    # print(f"Attribution sum: {attribution_sum:.6f}")
    # print(f"Output - bias: {output_minus_bias:.6f}")
    # print(f"Match: {np.isclose(attribution_sum, output_minus_bias)}")
    
    print("\nComplete the code above to verify completeness property!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function demonstrating Gradient × Input saliency.
    """
    print("\n" + "="*70)
    print(" "*15 + "GRADIENT × INPUT SALIENCY TUTORIAL")
    print("="*70)
    
    try:
        # Theoretical demonstrations
        demonstrate_linear_case()
        demonstrate_saturation_issue()
        
        # Visual examples
        example_1_basic_comparison()
        example_2_normalized_image_effect()
        example_3_bright_vs_dark_regions()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nKey Takeaways:")
        print("1. Gradient × Input accounts for actual pixel values")
        print("2. Satisfies completeness for linear models")
        print("3. Better attribution than vanilla gradients")
        print("4. Still can be noisy (addressed in next modules)")
        print("\nNext Steps:")
        print("→ Module 03: SmoothGrad for noise reduction")
        print("→ Module 04: Integrated Gradients for better axioms")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
