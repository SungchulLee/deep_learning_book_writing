# Advanced Techniques

09: Advanced Saliency Techniques DESCRIPTION:

Understanding what neural networks learn is crucial for building trust and debugging models. This module demonstrates gradient-based explanation techniques that reveal how models process inputs and make decisions, providing visual and quantitative insight into network behavior.

## Code

```python
"""
09: Advanced Saliency Techniques
==============================

DIFFICULTY: Advanced

DESCRIPTION:
Introduction to cutting-edge saliency methods:
- Layer-wise Relevance Propagation (LRP)
- Attention Rollout (for Transformers)
- DeepLIFT
- SHAP for deep learning

These methods address limitations of gradient-based approaches.

Author: Educational purposes
"""

import torch
from utils import *

# ========================================================================
# Main
# ========================================================================

def overview_advanced_methods():
    """Overview of advanced techniques."""
    
    print("\n" + "="*70)
    print(" "*15 + "ADVANCED SALIENCY TECHNIQUES")
    print("="*70)
    
    print("\n1. LAYER-WISE RELEVANCE PROPAGATION (LRP)")
    print("-" * 70)
    print("Concept: Decompose prediction by propagating relevance backward")
    print("Formula: R_i = Σ_j (z_ij / Σ_k z_kj) R_j")
    print("Advantage: Satisfies conservation property")
    print("Use case: When you need exact relevance decomposition")
    
    print("\n2. ATTENTION ROLLOUT (Transformers)")
    print("-" * 70)
    print("Concept: Aggregate attention maps across layers")
    print("Formula: Att = Π_l Att^(l) where Att^(l) are layer attentions")
    print("Advantage: Visualizes what transformers attend to")
    print("Use case: Vision Transformers, BERT, etc.")
    
    print("\n3. DEEPLIFT")
    print("-" * 70)
    print("Concept: Compare activations to reference activations")
    print("Formula: Attribution based on difference from baseline")
    print("Advantage: Handles saturated gradients better")
    print("Use case: When gradients vanish/explode")
    
    print("\n4. SHAP (SHapley Additive exPlanations)")
    print("-" * 70)
    print("Concept: Game-theoretic approach to attribution")
    print("Formula: Shapley values from cooperative game theory")
    print("Advantage: Theoretically optimal, fair attribution")
    print("Use case: When you need provably fair explanations")
    
    print("\n5. DECONVNET")
    print("-" * 70)
    print("Concept: Similar to guided backprop, different ReLU handling")
    print("Use case: Alternative to guided backpropagation")
    
    print("\n" + "="*70)
    print("IMPLEMENTATION RESOURCES:")
    print("-" * 70)
    print("• Captum (PyTorch): https://captum.ai/")
    print("• SHAP: https://github.com/slundberg/shap")
    print("• LRP Toolbox: https://github.com/sebastian-lapuschkin/lrp_toolbox")
    print("• Transformer Explainability: Attention rollout papers")
    print("="*70)


def example_1_when_to_use_what():
    """Guide for selecting the right method."""
    
    print("\n" + "="*60)
    print("DECISION TREE: Which Method to Use?")
    print("="*60)
    
    print("\nQuestion 1: What's your goal?")
    print("  A) Quick debugging → Vanilla Gradient")
    print("  B) Publication-quality visualization → Guided Grad-CAM")
    print("  C) Theoretical guarantees → Integrated Gradients or SHAP")
    print("  D) Understanding transformers → Attention Rollout")
    
    print("\nQuestion 2: What's your model type?")
    print("  A) CNN → Grad-CAM, Guided Grad-CAM")
    print("  B) Transformer → Attention Rollout")
    print("  C) Any → Integrated Gradients, SHAP")
    
    print("\nQuestion 3: What's your constraint?")
    print("  A) Speed → Vanilla Gradient")
    print("  B) Quality → SmoothGrad, Integrated Gradients")
    print("  C) Resolution → Guided methods")
    
    print("\nQuestion 4: What's your use case?")
    print("  A) Scientific paper → Integrated Gradients (citations)")
    print("  B) Demo/presentation → Guided Grad-CAM (visual appeal)")
    print("  C) Production deployment → Grad-CAM (speed)")
    print("  D) Debugging → Vanilla Gradient (fast iteration)")


def main():
    print("\n" + "="*70)
    print(" "*15 + "ADVANCED TECHNIQUES OVERVIEW")
    print("="*70)
    
    try:
        overview_advanced_methods()
        example_1_when_to_use_what()
        
        print("\n" + "="*70)
        print("CONGRATULATIONS!")
        print("="*70)
        print("\nYou've completed the Saliency Maps tutorial series!")
        print("\nWhat you've learned:")
        print("✓ Gradient-based methods (Vanilla, Gradient×Input)")
        print("✓ Noise reduction (SmoothGrad)")
        print("✓ Path integration (Integrated Gradients)")
        print("✓ Localization (Grad-CAM)")
        print("✓ High-resolution (Guided Backprop)")
        print("✓ Combined methods (Guided Grad-CAM)")
        print("✓ Method selection and comparison")
        print("✓ Advanced techniques overview")
        
        print("\nNext steps:")
        print("1. Apply these methods to your own models")
        print("2. Explore Captum library for more techniques")
        print("3. Read original papers for deeper understanding")
        print("4. Experiment with different architectures")
        
        print("\nHappy interpreting! 🎉")
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
Add a dropout layer after the attention weights (before multiplying with values). Use a dropout rate of 0.1 during training. Explain why attention dropout helps with regularization.

??? success "Solution to Exercise 2"
    Add `self.attn_dropout = nn.Dropout(0.1)` in `__init__` and apply it after the softmax: `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. Attention dropout randomly zeroes some attention weights during training, preventing the model from relying too heavily on specific token-to-token relationships. This encourages the model to distribute attention more broadly and learn more robust representations, similar to how standard dropout prevents co-adaptation of neurons.

---

**Exercise 3.**
Explain the computational complexity of self-attention as a function of sequence length $n$ and model dimension $d$. Why does this motivate architectures like Longformer or Linformer for long sequences?

??? success "Solution to Exercise 3"
    Standard self-attention computes an $n \times n$ attention matrix, giving $O(n^2 d)$ time complexity and $O(n^2)$ memory for the attention weights. For long sequences (e.g., $n = 4096$), this becomes prohibitive. Longformer uses a combination of local sliding-window attention ($O(n \cdot w \cdot d)$ where $w$ is window size) and sparse global attention for selected tokens. Linformer projects keys and values to a lower dimension $k \ll n$, reducing complexity to $O(n \cdot k \cdot d)$. Both trade some expressiveness for practical efficiency on long inputs.

---

**Exercise 4.**
Write a comprehensive test function that validates the Advanced Techniques implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_advanced techniques():
        model = Advanced Techniques(...)
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
