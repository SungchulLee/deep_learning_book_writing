#!/usr/bin/env python3
"""
==============================================================================
Script 04: Modern Activation Functions (GELU, Swish, Mish)
==============================================================================

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 8 minutes
PREREQUISITES: Scripts 01-03

LEARNING OBJECTIVES:
--------------------
1. Understand modern activation functions used in state-of-the-art models
2. Learn GELU (used in BERT, GPT, transformers)
3. Master Swish/SiLU (used in EfficientNet, mobile models)
4. Explore Mish and Hardswish
5. Know when to use each modern activation

KEY CONCEPTS:
-------------
- GELU: Gaussian Error Linear Unit (smooth, stochastic interpretation)
- Swish/SiLU: Self-gated activation (x * sigmoid(x))
- Mish: Similar to Swish but smoother
- Hardswish: Efficient approximation of Swish
- Performance vs. computation tradeoff

RUN THIS SCRIPT:
----------------
    python 04_modern_activations.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# SECTION 1: GELU (Gaussian Error Linear Unit)
# ==============================================================================

def section1_gelu():
    """
    GELU: Used in BERT, GPT, and most modern transformers.
    Formula: GELU(x) = x * Φ(x), where Φ is the cumulative distribution 
             function of standard normal distribution
    
    Properties:
    - Smooth everywhere (differentiable)
    - Non-monotonic (dips slightly below zero for negative values)
    - Probabilistic interpretation: "expected transformation under random dropout"
    """
    print("=" * 70)
    print("SECTION 1: GELU (Gaussian Error Linear Unit)")
    print("=" * 70)
    
    print("\nGELU is the default activation in modern transformers!")
    print("Used in: BERT, GPT, RoBERTa, ViT (Vision Transformer)")
    
    # Create input range
    x = torch.linspace(-3, 3, 1000)
    
    # GELU activation
    y_gelu = F.gelu(x)
    
    # For comparison, also compute ReLU
    y_relu = F.relu(x)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y_gelu.numpy(), linewidth=2.5, 
             label='GELU', color='blue')
    plt.plot(x.numpy(), y_relu.numpy(), linewidth=2, linestyle='--',
             label='ReLU (for comparison)', color='red', alpha=0.6)
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('GELU vs ReLU', fontsize=14, fontweight='bold')
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Output', fontsize=12)
    plt.legend(fontsize=11)
    plt.xlim(-3, 3)
    
    # Highlight the key difference
    plt.fill_between(x.numpy(), 0, y_gelu.numpy(), 
                      where=(x.numpy() < 0) & (x.numpy() > -1),
                      alpha=0.3, color='blue', 
                      label='GELU allows some negative values')
    
    plt.tight_layout()
    
    print("\n✅ Key GELU Properties:")
    print("   • Smooth approximation to ReLU")
    print("   • Non-zero for small negative inputs (better gradient flow)")
    print("   • Acts like dropout regularization stochastically")
    print("   • Empirically better than ReLU in transformers")
    
    # Practical usage
    print("\n💻 PyTorch Usage:")
    print("   Functional: F.gelu(x)")
    print("   Module:     nn.GELU()")
    
    # Example
    sample = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"\n   Input:  {sample.tolist()}")
    print(f"   GELU:   {F.gelu(sample).tolist()}")
    print(f"   ReLU:   {F.relu(sample).tolist()}")
    
    return plt.gcf()


# ==============================================================================
# SECTION 2: Swish / SiLU (Sigmoid Linear Unit)
# ==============================================================================

def section2_swish():
    """
    Swish (also called SiLU): x * sigmoid(x)
    
    Discovered through neural architecture search.
    Used in: EfficientNet, MobileNet V3, many mobile/edge models
    
    Properties:
    - Self-gated (uses its own value to gate itself)
    - Smooth and non-monotonic
    - Unbounded above, bounded below
    """
    print("\n" + "=" * 70)
    print("SECTION 2: Swish / SiLU (Sigmoid Linear Unit)")
    print("=" * 70)
    
    print("\nSwish: Discovered by Google through automated search")
    print("Used in: EfficientNet, MobileNetV3, many modern CNNs")
    
    # Input range
    x = torch.linspace(-4, 4, 1000)
    
    # Swish/SiLU: x * sigmoid(x)
    y_swish = F.silu(x)  # PyTorch calls it SiLU
    y_relu = F.relu(x)
    
    # Also show the sigmoid component
    sigmoid_x = torch.sigmoid(x)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Swish vs ReLU
    ax1.plot(x.numpy(), y_swish.numpy(), linewidth=2.5, 
             label='Swish/SiLU', color='purple')
    ax1.plot(x.numpy(), y_relu.numpy(), linewidth=2, linestyle='--',
             label='ReLU', color='red', alpha=0.6)
    ax1.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle=':', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Swish/SiLU vs ReLU', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Input (x)', fontsize=11)
    ax1.set_ylabel('Output', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_xlim(-4, 4)
    
    # Plot 2: Decomposition
    ax2.plot(x.numpy(), x.numpy(), linewidth=2, 
             label='x (identity)', color='green', alpha=0.7)
    ax2.plot(x.numpy(), sigmoid_x.numpy(), linewidth=2,
             label='sigmoid(x)', color='orange', alpha=0.7)
    ax2.plot(x.numpy(), y_swish.numpy(), linewidth=2.5,
             label='Swish = x * sigmoid(x)', color='purple')
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle=':', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Swish Decomposition', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Input (x)', fontsize=11)
    ax2.set_ylabel('Output', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.set_xlim(-4, 4)
    
    plt.tight_layout()
    
    print("\n✅ Key Swish Properties:")
    print("   • Self-gated: x * sigmoid(x)")
    print("   • Smooth and non-monotonic")
    print("   • Outperforms ReLU in many image tasks")
    print("   • Slightly more expensive than ReLU")
    
    print("\n💻 PyTorch Usage:")
    print("   Functional: F.silu(x)  ← PyTorch uses SiLU name")
    print("   Module:     nn.SiLU()")
    print("   (Swish and SiLU are the same thing!)")
    
    # Example
    sample = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"\n   Input:  {sample.tolist()}")
    print(f"   Swish:  {F.silu(sample).tolist()}")
    
    return fig


# ==============================================================================
# SECTION 3: Mish
# ==============================================================================

def section3_mish():
    """
    Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    
    Properties:
    - Smoother than Swish
    - Self-regularizing
    - Non-monotonic
    - Unbounded above, bounded below (~-0.31)
    """
    print("\n" + "=" * 70)
    print("SECTION 3: Mish")
    print("=" * 70)
    
    print("\nMish: A newer activation inspired by Swish")
    print("Formula: Mish(x) = x * tanh(softplus(x))")
    
    # Input range
    x = torch.linspace(-4, 4, 1000)
    
    # Mish: x * tanh(softplus(x))
    y_mish = F.mish(x)
    y_swish = F.silu(x)
    y_relu = F.relu(x)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y_mish.numpy(), linewidth=2.5,
             label='Mish', color='darkgreen')
    plt.plot(x.numpy(), y_swish.numpy(), linewidth=2,
             label='Swish/SiLU', color='purple', alpha=0.7)
    plt.plot(x.numpy(), y_relu.numpy(), linewidth=2, linestyle='--',
             label='ReLU', color='red', alpha=0.5)
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('Mish vs Swish vs ReLU', fontsize=14, fontweight='bold')
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Output', fontsize=12)
    plt.legend(fontsize=11)
    plt.xlim(-4, 4)
    plt.tight_layout()
    
    print("\n✅ Key Mish Properties:")
    print("   • Smoother than Swish (especially for negative values)")
    print("   • Self-regularizing properties")
    print("   • Can outperform both ReLU and Swish in some tasks")
    print("   • More computationally expensive")
    
    print("\n💻 PyTorch Usage:")
    print("   Functional: F.mish(x)")
    print("   Module:     nn.Mish()")
    
    # Example
    sample = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"\n   Input:  {sample.tolist()}")
    print(f"   Mish:   {[f'{v:.4f}' for v in F.mish(sample).tolist()]}")
    print(f"   Swish:  {[f'{v:.4f}' for v in F.silu(sample).tolist()]}")
    
    return plt.gcf()


# ==============================================================================
# SECTION 4: Hardswish (Efficient Swish)
# ==============================================================================

def section4_hardswish():
    """
    Hardswish: Efficient approximation of Swish for mobile/edge devices.
    Used in MobileNetV3.
    
    Formula: Hardswish(x) = x * ReLU6(x + 3) / 6
    """
    print("\n" + "=" * 70)
    print("SECTION 4: Hardswish (Efficient Approximation)")
    print("=" * 70)
    
    print("\nHardswish: Faster approximation of Swish")
    print("Used in: MobileNetV3 (optimized for mobile devices)")
    
    # Input range
    x = torch.linspace(-4, 4, 1000)
    
    # Hardswish and Swish
    y_hardswish = F.hardswish(x)
    y_swish = F.silu(x)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y_swish.numpy(), linewidth=3,
             label='Swish (original)', color='purple', alpha=0.5)
    plt.plot(x.numpy(), y_hardswish.numpy(), linewidth=2,
             label='Hardswish (efficient)', color='darkblue', linestyle='--')
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('Hardswish vs Swish', fontsize=14, fontweight='bold')
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Output', fontsize=12)
    plt.legend(fontsize=11)
    plt.xlim(-4, 4)
    plt.tight_layout()
    
    print("\n✅ Key Hardswish Properties:")
    print("   • Piecewise linear approximation of Swish")
    print("   • Much faster computation (no exp or division)")
    print("   • Nearly identical performance to Swish")
    print("   • Ideal for mobile and edge deployment")
    
    print("\n💻 PyTorch Usage:")
    print("   Functional: F.hardswish(x)")
    print("   Module:     nn.Hardswish()")
    
    # Show the computational difference
    print("\n⚡ Computational Efficiency:")
    print("   Swish:     x * sigmoid(x)  → requires exp() operation")
    print("   Hardswish: x * ReLU6(x+3)/6 → only ReLU and multiplication")
    
    return plt.gcf()


# ==============================================================================
# SECTION 5: Comparison and When to Use
# ==============================================================================

def section5_comparison():
    """
    Compare all modern activations and provide guidance on when to use each.
    """
    print("\n" + "=" * 70)
    print("SECTION 5: Comparison and Usage Guide")
    print("=" * 70)
    
    # Create comparison plot
    x = torch.linspace(-3, 3, 1000)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: All modern activations
    activations = {
        'ReLU': F.relu(x),
        'GELU': F.gelu(x),
        'Swish/SiLU': F.silu(x),
        'Mish': F.mish(x),
        'Hardswish': F.hardswish(x)
    }
    
    colors = ['red', 'blue', 'purple', 'darkgreen', 'navy']
    for (name, y), color in zip(activations.items(), colors):
        ax1.plot(x.numpy(), y.numpy(), linewidth=2.5, 
                label=name, color=color, alpha=0.8)
    
    ax1.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle=':', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Modern Activations Comparison', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Input (x)', fontsize=11)
    ax1.set_ylabel('Output', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-1, 3)
    
    # Plot 2: Zoomed in around zero
    for (name, y), color in zip(activations.items(), colors):
        ax2.plot(x.numpy(), y.numpy(), linewidth=2.5,
                label=name, color=color, alpha=0.8)
    
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle=':', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Behavior Near Zero (Zoomed)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Input (x)', fontsize=11)
    ax2.set_ylabel('Output', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-0.3, 1)
    
    plt.tight_layout()
    
    # Usage guide
    print("\n📖 When to Use Modern Activations:")
    print("-" * 70)
    
    guide = {
        "GELU": {
            "Best for": "Transformers, NLP models, Vision Transformers",
            "Examples": "BERT, GPT, RoBERTa, ViT",
            "Pros": "Smooth, works extremely well in transformers",
            "Cons": "More expensive than ReLU",
        },
        "Swish/SiLU": {
            "Best for": "CNNs, image classification, object detection",
            "Examples": "EfficientNet, MobileNetV3",
            "Pros": "Better than ReLU in many vision tasks",
            "Cons": "Slightly more expensive",
        },
        "Mish": {
            "Best for": "When you want to try something beyond Swish",
            "Examples": "YOLOv4, some research models",
            "Pros": "Smoother, can outperform Swish",
            "Cons": "Most expensive of modern activations",
        },
        "Hardswish": {
            "Best for": "Mobile/edge deployment, efficiency critical",
            "Examples": "MobileNetV3, quantized models",
            "Pros": "Nearly as good as Swish, much faster",
            "Cons": "Not as smooth as Swish",
        }
    }
    
    for activation, info in guide.items():
        print(f"\n🔹 {activation}:")
        for key, value in info.items():
            print(f"   {key:12s}: {value}")
    
    print("\n" + "=" * 70)
    print("💡 GENERAL RECOMMENDATIONS:")
    print("=" * 70)
    print("• Starting a new project? Try ReLU first (baseline)")
    print("• Building a transformer? Use GELU (industry standard)")
    print("• Training a CNN? Try Swish/SiLU (often better than ReLU)")
    print("• Need speed for mobile? Use Hardswish")
    print("• Want to experiment? Try Mish")
    print("• When in doubt? Stick with ReLU (it's still very good!)")
    
    return fig


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Run all sections.
    """
    print("\n" + "█" * 70)
    print("   PYTORCH ACTIVATION FUNCTIONS TUTORIAL")
    print("   Script 04: Modern Activation Functions")
    print("█" * 70)
    
    fig1 = section1_gelu()
    fig2 = section2_swish()
    fig3 = section3_mish()
    fig4 = section4_hardswish()
    fig5 = section5_comparison()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("Run '05_binary_classification.py' for complete training examples!")
    print("=" * 70)
    
    print("\n✅ Script completed successfully!")
    print("\n📊 Showing plots... (Close windows to continue)")
    
    plt.show()


if __name__ == "__main__":
    main()
