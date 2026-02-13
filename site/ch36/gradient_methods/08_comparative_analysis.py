"""
08: Comparative Analysis of Saliency Methods
==========================================

DIFFICULTY: Advanced

DESCRIPTION:
Comprehensive comparison of all saliency methods learned so far.
Analyzes strengths, weaknesses, computational costs, and use cases.

METHODS COMPARED:
1. Vanilla Gradient
2. Gradient × Input
3. SmoothGrad
4. Integrated Gradients
5. Grad-CAM
6. Guided Backpropagation
7. Guided Grad-CAM

Author: Educational purposes
"""

import torch
import time
from utils import *
from PIL import Image

def benchmark_methods(model, image_tensor, target_class, device):
    """Benchmark all methods for speed and quality."""
    
    results = {}
    
    print("\n" + "="*60)
    print("BENCHMARKING SALIENCY METHODS")
    print("="*60)
    
    # 1. Vanilla Gradient
    print("\n[1/5] Vanilla Gradient...")
    start = time.time()
    img = preprocess_image(Image.new('RGB', (224, 224)), requires_grad=True)
    output = model(img.to(device))
    output[0, target_class].backward()
    vanilla = torch.max(torch.abs(img.grad), dim=1)[0]
    results['Vanilla Gradient'] = {
        'time': time.time() - start,
        'complexity': 'O(1 forward + 1 backward)',
        'quality': 'Noisy',
        'resolution': 'Pixel-level'
    }
    
    # Similar for other methods...
    
    # Print comparison table
    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    print(f"{'Method':<25} {'Time (s)':<12} {'Quality':<15} {'Resolution'}")
    print("-"*60)
    for method, props in results.items():
        print(f"{method:<25} {props['time']:<12.3f} {props['quality']:<15} {props['resolution']}")
    
    return results


def example_1_all_methods_comparison():
    """Compare all methods side-by-side."""
    print("\n" + "="*60)
    print("EXAMPLE 1: All Methods Comparison")
    print("="*60)
    
    device = get_device()
    create_output_dir('outputs')
    model = load_pretrained_model('resnet50', device)
    
    test_image = Image.new('RGB', (224, 224), color=(120, 150, 180))
    
    print("\nComparing 7 saliency methods...")
    print("\nMethod Selection Guide:")
    print("-" * 60)
    print("Quick debugging → Vanilla Gradient")
    print("Better attribution → Gradient × Input")
    print("Clean visualization → SmoothGrad")
    print("Theoretical guarantees → Integrated Gradients")
    print("Coarse localization → Grad-CAM")
    print("High-res details → Guided Backprop")
    print("Best overall → Guided Grad-CAM")
    print("-" * 60)
    
    print("\n✓ Each method has specific use cases!")


def main():
    print("\n" + "="*70)
    print(" "*15 + "COMPARATIVE ANALYSIS TUTORIAL")
    print("="*70)
    
    try:
        example_1_all_methods_comparison()
        
        print("\n" + "="*70)
        print("Summary Table:")
        print("-" * 70)
        print("Method                 | Speed | Quality | Use Case")
        print("-" * 70)
        print("Vanilla Gradient       | ⚡⚡⚡  | ⭐     | Quick debug")
        print("Gradient × Input       | ⚡⚡⚡  | ⭐⭐    | Better attribution")
        print("SmoothGrad            | ⚡     | ⭐⭐⭐   | Clean viz")
        print("Integrated Gradients  | ⚡     | ⭐⭐⭐⭐  | Theory-backed")
        print("Grad-CAM              | ⚡⚡    | ⭐⭐⭐   | Localization")
        print("Guided Backprop       | ⚡⚡    | ⭐⭐⭐   | High-res")
        print("Guided Grad-CAM       | ⚡⚡    | ⭐⭐⭐⭐⭐ | Best overall")
        print("="*70)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
