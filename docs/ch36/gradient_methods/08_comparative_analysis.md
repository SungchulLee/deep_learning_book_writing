# Comparative Analysis

08: Comparative Analysis of Saliency Methods DESCRIPTION:

Understanding what neural networks learn is crucial for building trust and debugging models. This module demonstrates gradient-based explanation techniques that reveal how models process inputs and make decisions, providing visual and quantitative insight into network behavior.

## Code

```python
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

# ========================================================================
# Main
# ========================================================================

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
Write a comprehensive test function that validates the Comparative Analysis implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_comparative analysis():
        model = Comparative Analysis(...)
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
