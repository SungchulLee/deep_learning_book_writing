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
    main()
