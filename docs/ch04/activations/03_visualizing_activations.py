#!/usr/bin/env python3
"""
==============================================================================
Script 03: Visualizing Activation Functions
==============================================================================

DIFFICULTY: ⭐⭐ Easy-Medium
ESTIMATED TIME: 10 minutes
PREREQUISITES: Scripts 01, 02

LEARNING OBJECTIVES:
--------------------
1. Visualize activation function curves
2. Understand the shape and behavior of different activations
3. Compare activation functions side-by-side
4. Understand gradient behavior (derivatives)
5. Learn which activations are suitable for which scenarios

KEY CONCEPTS:
-------------
- Activation function outputs (forward pass)
- Gradients/derivatives (backward pass)
- Saturation regions (where gradients vanish)
- Non-monotonic activations
- Output ranges

WHAT YOU'LL SEE:
----------------
- Beautiful plots of 8 different activation functions
- Derivative plots showing gradient behavior
- Comparison grid for easy reference
- Analysis of saturation and gradient flow

RUN THIS SCRIPT:
----------------
    python 03_visualizing_activations.py

EXPECTED OUTPUT:
----------------
- Multiple matplotlib figures with activation curves
- Detailed annotations and explanations
- Comparison plots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

# ==============================================================================
# SECTION 1: Define Activation Functions
# ==============================================================================

def get_activation_functions():
    """
    Returns dictionary of activation functions and their properties.
    """
    activations = {
        'Sigmoid': {
            'func': lambda x: torch.sigmoid(x),
            'range': '(0, 1)',
            'use_case': 'Binary classification output, gates in LSTM',
            'pros': 'Smooth, bounded output',
            'cons': 'Vanishing gradients in deep networks'
        },
        'Tanh': {
            'func': lambda x: torch.tanh(x),
            'range': '(-1, 1)',
            'use_case': 'Hidden layers (better than sigmoid)',
            'pros': 'Zero-centered, stronger gradients than sigmoid',
            'cons': 'Still suffers from vanishing gradients'
        },
        'ReLU': {
            'func': lambda x: torch.relu(x),
            'range': '[0, ∞)',
            'use_case': 'Default choice for hidden layers',
            'pros': 'Simple, fast, avoids vanishing gradients',
            'cons': 'Dead neurons (negative values always 0)'
        },
        'Leaky ReLU': {
            'func': lambda x: F.leaky_relu(x, negative_slope=0.1),
            'range': '(-∞, ∞)',
            'use_case': 'Alternative to ReLU, prevents dead neurons',
            'pros': 'Allows gradient flow for negative values',
            'cons': 'Negative slope is hyperparameter'
        },
        'ELU': {
            'func': lambda x: F.elu(x),
            'range': '(-1, ∞)',
            'use_case': 'Better than ReLU in some tasks',
            'pros': 'Smooth, negative values, closer to zero mean',
            'cons': 'More computationally expensive'
        },
        'GELU': {
            'func': lambda x: F.gelu(x),
            'range': '(-∞, ∞)',
            'use_case': 'Modern: Transformers, BERT, GPT',
            'pros': 'Smooth, probabilistic interpretation',
            'cons': 'More expensive than ReLU'
        },
        'Swish (SiLU)': {
            'func': lambda x: F.silu(x),
            'range': '(-∞, ∞)',
            'use_case': 'Modern deep networks, mobile networks',
            'pros': 'Smooth, self-gated, performs well',
            'cons': 'Non-monotonic near zero'
        },
        'Softplus': {
            'func': lambda x: F.softplus(x),
            'range': '(0, ∞)',
            'use_case': 'Smooth alternative to ReLU',
            'pros': 'Smooth everywhere, no dead neurons',
            'cons': 'More expensive, can saturate for large inputs'
        }
    }
    return activations


# ==============================================================================
# SECTION 2: Plot Individual Activation Functions
# ==============================================================================

def section2_plot_individual_activations():
    """
    Plot each activation function individually with detailed information.
    """
    print("=" * 70)
    print("SECTION 2: Plotting Individual Activation Functions")
    print("=" * 70)
    
    # Get activation functions
    activations = get_activation_functions()
    
    # Create input range
    x = torch.linspace(-5, 5, 1000)
    
    # Create figure with subplots (2 rows x 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Activation Functions: Forward Pass', fontsize=16, fontweight='bold')
    axes = axes.ravel()
    
    for idx, (name, props) in enumerate(activations.items()):
        ax = axes[idx]
        
        # Compute activation
        y = props['func'](x).numpy()
        
        # Plot
        ax.plot(x.numpy(), y, linewidth=2, color='royalblue', label=name)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.grid(True, alpha=0.3)
        
        # Formatting
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Input (x)', fontsize=10)
        ax.set_ylabel('Output f(x)', fontsize=10)
        ax.set_xlim(-5, 5)
        
        # Add range information
        ax.text(0.02, 0.98, f"Range: {props['range']}", 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    print("\n✅ Generated plot: Activation Functions (Forward Pass)")
    print("   → Showing output behavior for different inputs")
    
    return fig


# ==============================================================================
# SECTION 3: Plot Derivatives (Gradients)
# ==============================================================================

def section3_plot_derivatives():
    """
    Plot derivatives of activation functions to understand gradient flow.
    """
    print("\n" + "=" * 70)
    print("SECTION 3: Plotting Activation Derivatives (Gradients)")
    print("=" * 70)
    
    # Input with gradient tracking
    x = torch.linspace(-5, 5, 1000, requires_grad=True)
    
    activations = get_activation_functions()
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Activation Functions: Gradients (Derivatives)', 
                 fontsize=16, fontweight='bold')
    axes = axes.ravel()
    
    for idx, (name, props) in enumerate(activations.items()):
        ax = axes[idx]
        
        # Compute activation
        y = props['func'](x)
        
        # Compute gradient for each point
        gradients = []
        for i in range(len(x)):
            if x.grad is not None:
                x.grad.zero_()
            y_single = props['func'](x[i].unsqueeze(0))
            y_single.backward()
            gradients.append(x.grad[i].item())
        
        # Plot derivative
        ax.plot(x.detach().numpy(), gradients, linewidth=2, 
                color='darkred', label=f"{name} derivative")
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.grid(True, alpha=0.3)
        
        # Formatting
        ax.set_title(f"{name} Gradient", fontsize=12, fontweight='bold')
        ax.set_xlabel('Input (x)', fontsize=10)
        ax.set_ylabel("f'(x)", fontsize=10)
        ax.set_xlim(-5, 5)
        
        # Highlight saturation regions for sigmoid/tanh
        if name in ['Sigmoid', 'Tanh']:
            ax.fill_between([-5, -2], -1, 2, alpha=0.2, color='red', 
                          label='Saturation (vanishing gradient)')
            ax.fill_between([2, 5], -1, 2, alpha=0.2, color='red')
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    
    print("\n✅ Generated plot: Activation Gradients")
    print("   → Red regions show where gradients vanish (bad for deep networks)")
    print("   → ReLU has constant gradient=1 for positive values (good!)")
    
    return fig


# ==============================================================================
# SECTION 4: Comparison Plot
# ==============================================================================

def section4_comparison_plot():
    """
    Compare all activations on a single plot for direct comparison.
    """
    print("\n" + "=" * 70)
    print("SECTION 4: Side-by-Side Comparison")
    print("=" * 70)
    
    x = torch.linspace(-5, 5, 1000)
    activations = get_activation_functions()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Activation Functions Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Classic activations
    classic = ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU']
    for name in classic:
        y = activations[name]['func'](x).numpy()
        ax1.plot(x.numpy(), y, linewidth=2, label=name, alpha=0.8)
    
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Classic Activations', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Input (x)', fontsize=11)
    ax1.set_ylabel('Output f(x)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-2, 5)
    
    # Plot 2: Modern activations
    modern = ['ReLU', 'ELU', 'GELU', 'Swish (SiLU)']
    for name in modern:
        y = activations[name]['func'](x).numpy()
        ax2.plot(x.numpy(), y, linewidth=2, label=name, alpha=0.8)
    
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Modern Activations', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Input (x)', fontsize=11)
    ax2.set_ylabel('Output f(x)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-2, 5)
    
    plt.tight_layout()
    
    print("\n✅ Generated comparison plot")
    print("   → Notice how modern activations (GELU, Swish) are smoother")
    print("   → ReLU is included in both for reference")
    
    return fig


# ==============================================================================
# SECTION 5: Key Insights and Analysis
# ==============================================================================

def section5_key_insights():
    """
    Print key insights about activation functions based on visualizations.
    """
    print("\n" + "=" * 70)
    print("SECTION 5: Key Insights from Visualizations 🎓")
    print("=" * 70)
    
    insights = [
        "\n1. SIGMOID & TANH:",
        "   ✗ Saturate (flat regions) for large |x| → vanishing gradients",
        "   ✗ Not recommended for deep networks",
        "   ✓ Still useful: sigmoid for binary output, tanh for LSTM gates",
        
        "\n2. ReLU:",
        "   ✓ Simple: max(0, x)",
        "   ✓ No saturation for positive values",
        "   ✓ Fast computation",
        "   ✗ 'Dead neurons': negative inputs always produce 0",
        "   → Most common choice for hidden layers",
        
        "\n3. LEAKY ReLU & ELU:",
        "   ✓ Allow negative values (gradient flows everywhere)",
        "   ✓ Solve dead neuron problem",
        "   ✓ ELU is smoother, closer to zero-mean",
        "   → Good alternatives when ReLU causes issues",
        
        "\n4. GELU & SWISH (Modern):",
        "   ✓ Smooth, non-monotonic",
        "   ✓ Used in state-of-the-art models (BERT, GPT, EfficientNet)",
        "   ✓ Better performance in many tasks",
        "   ✗ More computationally expensive",
        "   → Recommended for transformers and large models",
        
        "\n5. GRADIENT BEHAVIOR:",
        "   • Sigmoid/Tanh: Gradients → 0 at extremes (bad for deep networks)",
        "   • ReLU: Constant gradient=1 for x>0 (good for deep networks)",
        "   • Modern activations: Balanced gradient behavior",
        
        "\n6. CHOOSING ACTIVATIONS:",
        "   • Default for hidden layers: ReLU",
        "   • For transformers: GELU",
        "   • Having dead neurons: Leaky ReLU or ELU",
        "   • Output layer (binary): Sigmoid with BCEWithLogitsLoss",
        "   • Output layer (multiclass): No activation, use CrossEntropyLoss",
        "   • Output layer (regression): No activation or use appropriate bounds"
    ]
    
    for insight in insights:
        print(insight)


# ==============================================================================
# SECTION 6: Practical Demonstration
# ==============================================================================

def section6_practical_demo():
    """
    Show how different activations affect a simple network's output.
    """
    print("\n" + "=" * 70)
    print("SECTION 6: Practical Demonstration")
    print("=" * 70)
    
    # Simple test network
    class TestNetwork(nn.Module):
        def __init__(self, activation):
            super().__init__()
            self.fc1 = nn.Linear(1, 10)
            self.fc2 = nn.Linear(10, 1)
            self.activation = activation
        
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Test input
    x = torch.linspace(-3, 3, 100).unsqueeze(1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Test different activations
    activations_to_test = {
        'ReLU': nn.ReLU(),
        'Tanh': nn.Tanh(),
        'GELU': nn.GELU(),
        'Sigmoid': nn.Sigmoid()
    }
    
    for name, activation in activations_to_test.items():
        # Create and run network
        torch.manual_seed(42)  # For reproducibility
        net = TestNetwork(activation)
        with torch.no_grad():
            output = net(x)
        
        # Plot
        ax.plot(x.numpy(), output.numpy(), linewidth=2, label=name, alpha=0.7)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_title('Network Output with Different Activations', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Input', fontsize=11)
    ax.set_ylabel('Output', fontsize=11)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    print("\n✅ Generated practical demonstration plot")
    print("   → Same network architecture, different activations")
    print("   → Notice how activation choice affects output characteristics")
    
    return fig


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Run all sections and display plots.
    """
    print("\n" + "█" * 70)
    print("   PYTORCH ACTIVATION FUNCTIONS TUTORIAL")
    print("   Script 03: Visualizing Activation Functions")
    print("█" * 70)
    
    # Generate all plots
    fig1 = section2_plot_individual_activations()
    fig2 = section3_plot_derivatives()
    fig3 = section4_comparison_plot()
    section5_key_insights()
    fig4 = section6_practical_demo()
    
    # Print summary
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("Run '04_modern_activations.py' to explore GELU, Swish, and Mish!")
    print("=" * 70)
    
    print("\n✅ Script completed successfully!")
    print("\n📊 Showing plots... (Close plot windows to continue)")
    
    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
