"""
05: Grad-CAM - Gradient-weighted Class Activation Mapping
========================================================

DIFFICULTY: Intermediate

DESCRIPTION:
Grad-CAM uses gradients flowing into the final convolutional layer
to produce coarse localization maps highlighting important regions.
Unlike pixel-level methods, Grad-CAM shows which regions matter.

MATHEMATICAL FOUNDATION:
    α_k = (1/Z) Σᵢⱼ (∂y_c/∂A_k^(i,j))    [importance weights]
    L_Grad-CAM = ReLU(Σ_k α_k A_k)         [weighted sum]

Where:
- A_k: k-th feature map of last conv layer
- α_k: global average pooled gradients
- ReLU: only positive influence

ADVANTAGES:
- Class-discriminative (different classes show different regions)
- Coarse localization (shows "where" not "what")
- Works for any CNN architecture
- Visual interpretability

Author: Educational purposes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class GradCAM:
    """Grad-CAM implementation for any CNN."""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: CNN model
            target_layer: Last convolutional layer (e.g., model.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save forward pass activations."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def __call__(self, image_tensor, target_class, device):
        """
        Compute Grad-CAM.
        
        Returns:
            torch.Tensor: Heatmap [1, H, W]
        """
        self.model.eval()
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        output = self.model(image_tensor)
        target_score = output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward()
        
        # Get activations and gradients
        activations = self.activations  # [1, C, H', W']
        gradients = self.gradients       # [1, C, H', W']
        
        # Global average pool gradients: α_k = (1/Z) Σᵢⱼ ∂y_c/∂A_k^(i,j)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination: Σ_k α_k A_k
        weighted_activations = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        
        # Apply ReLU: only positive influence
        heatmap = F.relu(weighted_activations)  # [1, 1, H', W']
        
        # Normalize to [0, 1]
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        # Upsample to input size
        heatmap = F.interpolate(
            heatmap,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        
        return heatmap.squeeze()  # [H, W]


def get_last_conv_layer(model, model_name='resnet50'):
    """Get last convolutional layer for different architectures."""
    if 'resnet' in model_name:
        return model.layer4[-1]
    elif 'vgg' in model_name:
        return model.features[-1]
    elif 'densenet' in model_name:
        return model.features[-1]
    else:
        raise ValueError(f"Unknown architecture: {model_name}")


def example_1_basic_gradcam():
    """Basic Grad-CAM usage."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Grad-CAM")
    print("="*60)
    
    device = get_device()
    create_output_dir('outputs')
    
    # Load model
    model_name = 'resnet50'
    model = load_pretrained_model(model_name, device)
    
    # Get last conv layer
    target_layer = get_last_conv_layer(model, model_name)
    print(f"Target layer: {target_layer.__class__.__name__}")
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Test image
    from PIL import Image
    test_image = Image.new('RGB', (224, 224), color=(100, 150, 200))
    image_tensor = preprocess_image(test_image, requires_grad=False)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor.to(device))
        target_class = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1)[0, target_class].item()
    
    print(f"\nPredicted class: {target_class}")
    print(f"Confidence: {confidence:.2%}")
    
    # Compute Grad-CAM
    print("\nComputing Grad-CAM...")
    heatmap = gradcam(image_tensor, target_class, device)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Value range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Visualize
    visualize_saliency(
        image_tensor,
        heatmap,
        title=f"Grad-CAM (Class {target_class})",
        colormap='jet',
        alpha=0.4,
        save_path='outputs/05_gradcam_basic.png'
    )
    
    print("\n✓ Grad-CAM shows coarse localization!")


def example_2_compare_with_gradients():
    """Compare Grad-CAM with gradient methods."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Grad-CAM vs Gradient Methods")
    print("="*60)
    
    device = get_device()
    model = load_pretrained_model('resnet50', device)
    
    test_image = Image.new('RGB', (224, 224), color=(180, 120, 80))
    image_tensor_grad = preprocess_image(test_image, requires_grad=True)
    image_tensor_cam = preprocess_image(test_image, requires_grad=False)
    
    with torch.no_grad():
        output = model(image_tensor_cam.to(device))
        target_class = output.argmax(dim=1).item()
    
    # Vanilla gradient
    print("Computing vanilla gradient...")
    output = model(image_tensor_grad.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(image_tensor_grad.grad), dim=1)[0]
    
    # Grad-CAM
    print("Computing Grad-CAM...")
    target_layer = get_last_conv_layer(model, 'resnet50')
    gradcam = GradCAM(model, target_layer)
    gradcam_map = gradcam(image_tensor_cam, target_class, device)
    
    # Compare
    saliencies = {
        'Vanilla Gradient\n(pixel-level)': vanilla,
        'Grad-CAM\n(region-level)': gradcam_map
    }
    
    visualize_multiple_saliencies(
        image_tensor_cam, saliencies,
        save_path='outputs/05_gradcam_vs_gradient.png'
    )
    
    print("\nKEY DIFFERENCES:")
    print("- Vanilla: Pixel-level, noisy, high-res")
    print("- Grad-CAM: Region-level, clean, coarse")
    print("- Grad-CAM better for 'where', gradients for 'what'")
    print("\n✓ Complementary methods!")


def main():
    print("\n" + "="*70)
    print(" "*20 + "GRAD-CAM TUTORIAL")
    print("="*70)
    
    try:
        example_1_basic_gradcam()
        example_2_compare_with_gradients()
        
        print("\n" + "="*70)
        print("Key Takeaways:")
        print("1. Grad-CAM: coarse, class-discriminative localization")
        print("2. Uses last conv layer activations + gradients")
        print("3. Clean visualization, easy interpretation")
        print("4. Limited to CNNs with spatial structure")
        print("\nNext: Module 06 - Guided Backpropagation")
        print("="*70)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
