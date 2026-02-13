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
    main()
