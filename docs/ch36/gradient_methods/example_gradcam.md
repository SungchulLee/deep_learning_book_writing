# Example Usage

Example Usage: Grad-CAM with PyTorch Models This script demonstrates how to use Grad-CAM to visualize CNN predictions.

Understanding what neural networks learn is crucial for building trust and debugging models. This module demonstrates gradient-based explanation techniques that reveal how models process inputs and make decisions, providing visual and quantitative insight into network behavior.

## Code

```python
"""
Example Usage: Grad-CAM with PyTorch Models

This script demonstrates how to use Grad-CAM to visualize CNN predictions.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from gradcam import GradCAM, GradCAMPlusPlus, get_target_layer

# ========================================================================
# Main
# ========================================================================


def load_and_preprocess_image(image_path: str, image_size: int = 224):
    """
    Load and preprocess an image for model input.
    
    Args:
        image_path: Path to the image file
        image_size: Target image size
        
    Returns:
        Preprocessed image tensor and original image array
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess
    input_tensor = preprocess(image).unsqueeze(0)
    
    # Resize original image for overlay
    original_image = np.array(Image.fromarray(original_image).resize((image_size, image_size)))
    
    return input_tensor, original_image


def example_resnet_gradcam():
    """
    Example: Using Grad-CAM with ResNet50
    """
    print("=" * 60)
    print("Example 1: Grad-CAM with ResNet50")
    print("=" * 60)
    
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Get the last convolutional layer
    target_layer = model.layer4[-1]
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Example: Create a random image (replace with real image path)
    print("\nCreating example input (replace with actual image)...")
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # Generate CAM
    print("Generating Grad-CAM heatmap...")
    cam = gradcam.generate_cam(input_tensor, target_class=None)
    
    print(f"CAM shape: {cam.shape}")
    print(f"CAM min/max: {cam.min():.4f} / {cam.max():.4f}")
    
    # Visualize
    print("Creating visualization...")
    visualization = gradcam.visualize_cam(input_tensor)
    
    # Display
    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    plt.title('Grad-CAM Visualization')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('gradcam_example.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'gradcam_example.png'")
    plt.close()


def example_with_real_image(image_path: str):
    """
    Example: Grad-CAM with a real image
    
    Args:
        image_path: Path to input image
    """
    print("\n" + "=" * 60)
    print("Example 2: Grad-CAM with Real Image")
    print("=" * 60)
    
    # Load model
    print("Loading ResNet50...")
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Load and preprocess image
    print(f"Loading image from {image_path}...")
    input_tensor, original_image = load_and_preprocess_image(image_path)
    
    # Get target layer
    target_layer = model.layer4[-1]
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Forward pass to get prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()
    
    print(f"Predicted class: {pred_class} (confidence: {pred_prob:.2%})")
    
    # Generate Grad-CAM
    print("Generating Grad-CAM...")
    visualization = gradcam.visualize_cam(input_tensor, 
                                         target_class=pred_class,
                                         original_image=original_image)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    cam = gradcam.generate_cam(input_tensor, target_class=pred_class)
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(visualization)
    axes[2].set_title(f'Overlay (Class: {pred_class})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison to 'gradcam_comparison.png'")
    plt.close()


def example_gradcam_plusplus():
    """
    Example: Comparing Grad-CAM and Grad-CAM++
    """
    print("\n" + "=" * 60)
    print("Example 3: Grad-CAM vs Grad-CAM++")
    print("=" * 60)
    
    # Load model
    model = models.resnet50(pretrained=True)
    model.eval()
    target_layer = model.layer4[-1]
    
    # Create random input
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # Standard Grad-CAM
    print("Generating standard Grad-CAM...")
    gradcam = GradCAM(model, target_layer)
    cam1 = gradcam.generate_cam(input_tensor)
    vis1 = gradcam.visualize_cam(input_tensor)
    
    # Grad-CAM++
    print("Generating Grad-CAM++...")
    gradcam_pp = GradCAMPlusPlus(model, target_layer)
    cam2 = gradcam_pp.generate_cam(input_tensor)
    vis2 = gradcam_pp.visualize_cam(input_tensor)
    
    # Compare
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(cam1, cmap='jet')
    axes[0, 0].set_title('Grad-CAM Heatmap')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(vis1)
    axes[0, 1].set_title('Grad-CAM Visualization')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(cam2, cmap='jet')
    axes[1, 0].set_title('Grad-CAM++ Heatmap')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(vis2)
    axes[1, 1].set_title('Grad-CAM++ Visualization')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_comparison_methods.png', dpi=150, bbox_inches='tight')
    print("Saved method comparison to 'gradcam_comparison_methods.png'")
    plt.close()


def example_multiple_classes():
    """
    Example: Visualizing Grad-CAM for multiple classes
    """
    print("\n" + "=" * 60)
    print("Example 4: Grad-CAM for Multiple Classes")
    print("=" * 60)
    
    # Load model
    model = models.resnet50(pretrained=True)
    model.eval()
    target_layer = model.layer4[-1]
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Create input
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # Get top-5 predictions
    with torch.no_grad():
        output = model(input_tensor)
        probs, classes = torch.softmax(output, dim=1).topk(5)
    
    # Generate CAM for each top class
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, (prob, cls) in enumerate(zip(probs[0], classes[0])):
        cam = gradcam.generate_cam(input_tensor, target_class=cls.item())
        axes[idx].imshow(cam, cmap='jet')
        axes[idx].set_title(f'Class {cls.item()}\n({prob.item():.2%})')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_multiple_classes.png', dpi=150, bbox_inches='tight')
    print("Saved multi-class visualization to 'gradcam_multiple_classes.png'")
    plt.close()


def example_different_architectures():
    """
    Example: Using Grad-CAM with different architectures
    """
    print("\n" + "=" * 60)
    print("Example 5: Grad-CAM with Different Architectures")
    print("=" * 60)
    
    architectures = {
        'ResNet50': (models.resnet50(pretrained=True), 'layer4'),
        'VGG16': (models.vgg16(pretrained=True), 'features'),
        'MobileNetV2': (models.mobilenet_v2(pretrained=True), 'features'),
    }
    
    input_tensor = torch.randn(1, 3, 224, 224)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, (model, layer_name)) in enumerate(architectures.items()):
        print(f"\nProcessing {name}...")
        model.eval()
        
        # Get target layer
        if layer_name == 'features':
            target_layer = list(model.features.children())[-1]
        else:
            target_layer = get_target_layer(model, layer_name)
        
        # Generate Grad-CAM
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(input_tensor)
        
        axes[idx].imshow(cam, cmap='jet')
        axes[idx].set_title(name)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_different_architectures.png', dpi=150, bbox_inches='tight')
    print("\nSaved architecture comparison to 'gradcam_different_architectures.png'")
    plt.close()


if __name__ == "__main__":
    print("Grad-CAM Examples\n")
    
    # Run examples
    example_resnet_gradcam()
    example_gradcam_plusplus()
    example_multiple_classes()
    example_different_architectures()
    
    # Uncomment to use with real image
    # example_with_real_image('path/to/your/image.jpg')
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)```

## Discussion

Visualization plays an important role in understanding model behavior and diagnosing training issues. The plotting code provides insight into the learned representations, convergence dynamics, or evaluation metrics, making abstract computations tangible.

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
Write a comprehensive test function that validates the Example Usage implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_example usage():
        model = Example Usage(...)
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
