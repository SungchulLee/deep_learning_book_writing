"""
01: Vanilla Gradient Saliency Maps
==================================

DIFFICULTY: Beginner

DESCRIPTION:
This module introduces the most basic form of saliency maps using vanilla gradients.
The core idea is simple: compute how much the output changes when we slightly 
perturb each input pixel. Pixels with high gradient magnitude have strong influence
on the model's prediction.

MATHEMATICAL FOUNDATION:
For an input image x and model output y_c for class c:
    Saliency(x) = |∂y_c/∂x|
    
Where:
- ∂y_c/∂x: Gradient of class score with respect to input
- |·|: Absolute value or magnitude
- Higher values indicate more important pixels

LEARNING OBJECTIVES:
1. Understand gradient-based attribution
2. Compute gradients with respect to input
3. Visualize saliency maps
4. Interpret what the model "sees"

PREREQUISITES:
- Basic PyTorch
- Autograd and backpropagation
- Pretrained CNN models

Author: Educational purposes
Date: 2025
"""

import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    load_image, 
    preprocess_image, 
    visualize_saliency,
    load_pretrained_model,
    print_model_prediction,
    get_device,
    create_output_dir
)


# ============================================================================
# VANILLA GRADIENT SALIENCY IMPLEMENTATION
# ============================================================================

def compute_vanilla_gradient_saliency(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute vanilla gradient saliency map.
    
    ALGORITHM:
    1. Forward pass: Get model predictions
    2. Select target class output
    3. Backward pass: Compute gradients w.r.t. input
    4. Take absolute value of gradients
    5. Aggregate across color channels (max or mean)
    
    Args:
        model: Pretrained neural network
        image_tensor: Input image [1, 3, H, W] with requires_grad=True
        target_class: Class index to compute saliency for
        device: Computation device (CPU/GPU)
        
    Returns:
        torch.Tensor: Saliency map [1, H, W]
        
    MATHEMATICAL DETAILS:
    Given:
        - Input: x ∈ ℝ^(3×H×W)  (RGB image)
        - Model: f(x) → ℝ^C  (C classes)
        - Target class: c
        
    Compute:
        Gradient: G = ∂f_c(x)/∂x ∈ ℝ^(3×H×W)
        Saliency: S = max_channel(|G|) ∈ ℝ^(H×W)
        
    WHY IT WORKS:
    - High gradient magnitude → small change in input causes large change in output
    - Indicates pixel importance for classification decision
    - First-order Taylor approximation of model behavior
    
    Example:
        >>> model = models.resnet50(pretrained=True).eval()
        >>> image = preprocess_image('dog.jpg', requires_grad=True)
        >>> saliency = compute_vanilla_gradient_saliency(
        ...     model, image, target_class=281, device=device
        ... )
    """
    # Step 1: Ensure model is in evaluation mode
    model.eval()
    
    # Step 2: Move image to device
    image_tensor = image_tensor.to(device)
    
    # Step 3: Zero out any existing gradients
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()
    
    # Step 4: Forward pass
    # Get model output (logits before softmax)
    output = model(image_tensor)  # Shape: [1, num_classes]
    
    # Step 5: Select the score for target class
    # We compute gradients with respect to THIS specific output
    target_score = output[0, target_class]
    
    # Step 6: Backward pass
    # Compute ∂target_score/∂image_tensor
    target_score.backward()
    
    # Step 7: Get gradients
    # Gradients have same shape as input: [1, 3, H, W]
    gradients = image_tensor.grad  # ∂y_c/∂x
    
    # Step 8: Take absolute value
    # We care about magnitude, not direction
    abs_gradients = torch.abs(gradients)  # |∂y_c/∂x|
    
    # Step 9: Aggregate across color channels
    # Option A: Take maximum across channels (commonly used)
    saliency_max = torch.max(abs_gradients, dim=1)[0]  # [1, H, W]
    
    # Option B: Take mean across channels (alternative)
    # saliency_mean = torch.mean(abs_gradients, dim=1)
    
    # Option C: Take L2 norm across channels (also common)
    # saliency_l2 = torch.sqrt(torch.sum(abs_gradients**2, dim=1))
    
    return saliency_max


def compute_saliency_for_predicted_class(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device
) -> tuple:
    """
    Compute saliency for the top predicted class.
    
    This is useful when we want to see what the model focuses on
    for its own prediction (rather than a specific target class).
    
    Args:
        model: Pretrained model
        image_tensor: Input image with requires_grad=True
        device: Device for computation
        
    Returns:
        tuple: (saliency_map, predicted_class, confidence)
        
    Example:
        >>> saliency, pred_class, conf = compute_saliency_for_predicted_class(
        ...     model, image, device
        ... )
        >>> print(f"Predicted class {pred_class} with {conf:.2%} confidence")
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = probabilities.max(dim=1)
    
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    
    # Compute saliency for predicted class
    saliency = compute_vanilla_gradient_saliency(
        model, image_tensor, predicted_class, device
    )
    
    return saliency, predicted_class, confidence


# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def analyze_saliency_statistics(saliency: torch.Tensor):
    """
    Print statistics about the saliency map.
    
    Helps understand the distribution of importance scores.
    
    Args:
        saliency: Saliency map tensor
        
    Output:
        Prints min, max, mean, std, and percentiles
        
    INTERPRETATION:
    - High max value: Some pixels very important
    - High mean: Many pixels contribute
    - High std: Uneven distribution of importance
    """
    saliency_np = saliency.detach().cpu().numpy().flatten()
    
    print("\n" + "="*50)
    print("SALIENCY MAP STATISTICS")
    print("="*50)
    print(f"Shape: {saliency.shape}")
    print(f"Min value: {saliency_np.min():.6f}")
    print(f"Max value: {saliency_np.max():.6f}")
    print(f"Mean value: {saliency_np.mean():.6f}")
    print(f"Std deviation: {saliency_np.std():.6f}")
    print(f"\nPercentiles:")
    print(f"  25th: {np.percentile(saliency_np, 25):.6f}")
    print(f"  50th (median): {np.percentile(saliency_np, 50):.6f}")
    print(f"  75th: {np.percentile(saliency_np, 75):.6f}")
    print(f"  90th: {np.percentile(saliency_np, 90):.6f}")
    print(f"  95th: {np.percentile(saliency_np, 95):.6f}")
    print(f"  99th: {np.percentile(saliency_np, 99):.6f}")
    print("="*50)


# ============================================================================
# EXAMPLE USAGE AND EXPERIMENTS
# ============================================================================

def example_1_basic_saliency():
    """
    EXAMPLE 1: Basic saliency map computation
    
    Demonstrates the simplest use case: compute and visualize
    saliency for a single image and class.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Vanilla Gradient Saliency")
    print("="*60)
    
    # Setup
    device = get_device()
    create_output_dir('outputs')
    
    # Load model
    print("\n[1/4] Loading pretrained ResNet-50...")
    model = load_pretrained_model('resnet50', device)
    
    # Load and preprocess image
    print("\n[2/4] Loading and preprocessing image...")
    # Create a simple test image (you can replace with actual image path)
    from PIL import Image
    test_image = Image.new('RGB', (224, 224), color=(100, 150, 200))
    image_tensor = preprocess_image(test_image, requires_grad=True)
    
    print(f"Image shape: {image_tensor.shape}")
    print(f"Requires gradient: {image_tensor.requires_grad}")
    
    # Show model prediction
    print("\n[3/4] Model prediction:")
    print_model_prediction(model, image_tensor.to(device), top_k=3)
    
    # Compute saliency for top predicted class
    print("\n[4/4] Computing saliency map...")
    saliency, pred_class, confidence = compute_saliency_for_predicted_class(
        model, image_tensor, device
    )
    
    print(f"\nPredicted class: {pred_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Saliency shape: {saliency.shape}")
    
    # Analyze saliency
    analyze_saliency_statistics(saliency)
    
    # Visualize
    print("\nVisualizing saliency map...")
    visualize_saliency(
        image_tensor,
        saliency,
        title=f"Vanilla Gradient Saliency (Class {pred_class})",
        save_path='outputs/01_basic_saliency.png'
    )
    
    print("\n✓ Example 1 completed!")


def example_2_compare_different_classes():
    """
    EXAMPLE 2: Compare saliency for different target classes
    
    Shows how saliency changes when we ask "Why did the model
    classify this as class A vs class B?"
    
    KEY INSIGHT:
    Different target classes highlight different image regions,
    revealing what features distinguish between classes.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Saliency for Different Target Classes")
    print("="*60)
    
    device = get_device()
    model = load_pretrained_model('resnet50', device)
    
    # Load image
    test_image = Image.new('RGB', (224, 224), color=(150, 100, 50))
    image_tensor = preprocess_image(test_image, requires_grad=True)
    
    # Get top 3 predictions
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probs = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_classes = probs[0].topk(3)
    
    print("\nComputing saliency for top 3 predicted classes...")
    
    # Compute saliency for each top class
    saliencies = {}
    for i, (prob, class_idx) in enumerate(zip(top_probs, top_classes)):
        class_idx = class_idx.item()
        prob = prob.item()
        
        print(f"\nClass {i+1}: {class_idx} (prob: {prob:.2%})")
        
        # Need fresh tensor with gradients for each backward pass
        image_fresh = preprocess_image(test_image, requires_grad=True)
        
        saliency = compute_vanilla_gradient_saliency(
            model, image_fresh, class_idx, device
        )
        
        saliencies[f"Class {class_idx}\n({prob:.1%})"] = saliency
    
    # Visualize comparison
    print("\nVisualizing comparison...")
    from utils import visualize_multiple_saliencies
    visualize_multiple_saliencies(
        image_tensor,
        saliencies,
        save_path='outputs/01_class_comparison.png'
    )
    
    print("\n✓ Example 2 completed!")


def example_3_gradient_aggregation_methods():
    """
    EXAMPLE 3: Different ways to aggregate gradients
    
    Compares different methods for combining RGB channel gradients:
    - Maximum across channels
    - Mean across channels
    - L2 norm across channels
    
    PEDAGOGICAL NOTE:
    Shows that aggregation method affects visualization,
    but core insights remain similar.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Gradient Aggregation Methods")
    print("="*60)
    
    device = get_device()
    model = load_pretrained_model('resnet50', device)
    
    # Load image
    test_image = Image.new('RGB', (224, 224), color=(200, 100, 150))
    image_tensor = preprocess_image(test_image, requires_grad=True)
    
    # Get prediction
    output = model(image_tensor.to(device))
    target_class = output.argmax(dim=1).item()
    
    print(f"\nTarget class: {target_class}")
    print("\nComputing saliency with different aggregation methods...")
    
    # Compute gradients once
    image_tensor = image_tensor.to(device)
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()
    
    output = model(image_tensor)
    output[0, target_class].backward()
    gradients = image_tensor.grad
    abs_gradients = torch.abs(gradients)
    
    # Method 1: Maximum across channels
    saliency_max = torch.max(abs_gradients, dim=1)[0]
    
    # Method 2: Mean across channels
    saliency_mean = torch.mean(abs_gradients, dim=1)
    
    # Method 3: L2 norm across channels
    saliency_l2 = torch.sqrt(torch.sum(abs_gradients**2, dim=1))
    
    # Compare
    saliencies = {
        'Max': saliency_max,
        'Mean': saliency_mean,
        'L2 Norm': saliency_l2
    }
    
    from utils import visualize_multiple_saliencies
    visualize_multiple_saliencies(
        image_tensor,
        saliencies,
        save_path='outputs/01_aggregation_comparison.png'
    )
    
    print("\n✓ Example 3 completed!")


# ============================================================================
# EXERCISES FOR STUDENTS
# ============================================================================

def exercise_1():
    """
    EXERCISE 1: Implement your own saliency function
    
    Task: Fill in the missing code to compute vanilla gradient saliency
    
    HINTS:
    1. Set model to eval mode
    2. Forward pass to get output
    3. Select target class score
    4. Call .backward() on that score
    5. Extract and process gradients
    """
    device = get_device()
    model = load_pretrained_model('resnet18', device)  # Smaller model
    
    # Create test image
    test_image = Image.new('RGB', (224, 224), color=(120, 180, 90))
    image_tensor = preprocess_image(test_image, requires_grad=True)
    image_tensor = image_tensor.to(device)
    
    target_class = 281  # Example class
    
    # YOUR CODE HERE:
    # 1. Set model to evaluation mode
    # model.___()
    
    # 2. Forward pass
    # output = ___
    
    # 3. Get target score
    # target_score = ___
    
    # 4. Backward pass
    # target_score.___()
    
    # 5. Get gradients
    # gradients = ___
    
    # 6. Process gradients (absolute value + max over channels)
    # saliency = ___
    
    # Uncomment to test your solution:
    # visualize_saliency(image_tensor, saliency, title="Exercise 1 Solution")
    
    print("Complete the code above and uncomment the visualization!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all examples.
    
    Demonstrates the complete workflow of vanilla gradient saliency:
    1. Basic computation and visualization
    2. Class comparison
    3. Aggregation method comparison
    """
    print("\n" + "="*70)
    print(" "*15 + "VANILLA GRADIENT SALIENCY TUTORIAL")
    print("="*70)
    
    try:
        # Run examples
        example_1_basic_saliency()
        example_2_compare_different_classes()
        example_3_gradient_aggregation_methods()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nKey Takeaways:")
        print("1. Vanilla gradients show pixel importance via ∂y_c/∂x")
        print("2. Different classes focus on different image regions")
        print("3. Saliency maps can be noisy (addressed in next modules)")
        print("4. Multiple aggregation methods are valid")
        print("\nNext Steps:")
        print("→ Module 02: Gradient × Input for better attribution")
        print("→ Module 03: SmoothGrad for noise reduction")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
