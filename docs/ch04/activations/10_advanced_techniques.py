#!/usr/bin/env python3
"""
Script 10: Advanced Activation Function Techniques
DIFFICULTY: ⭐⭐⭐⭐⭐ Very Hard | TIME: 15 min | PREREQ: All previous scripts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ==============================================================================
# SECTION 1: PReLU (Parametric ReLU) - Learnable Activation
# ==============================================================================

class PReLUNetwork(nn.Module):
    """
    PReLU: f(x) = x if x > 0 else α * x
    where α is LEARNED during training (vs Leaky ReLU where α is fixed)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.prelu1 = nn.PReLU(num_parameters=20)  # One α per channel
        self.fc2 = nn.Linear(20, 20)
        self.prelu2 = nn.PReLU(num_parameters=20)
        self.fc3 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))
        x = self.fc3(x)
        return x

def demonstrate_prelu():
    """Demonstrate PReLU learning"""
    print("=" * 70)
    print("SECTION 1: PReLU (Parametric ReLU)")
    print("=" * 70)
    
    model = PReLUNetwork()
    
    print("\nPReLU parameters BEFORE training:")
    print(f"  Layer 1 α: {model.prelu1.weight.data[:5].tolist()}")  # Show first 5
    print(f"  Layer 2 α: {model.prelu2.weight.data[:5].tolist()}")
    
    # Simple training
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("\nTraining PReLU network...")
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    print("\nPReLU parameters AFTER training:")
    print(f"  Layer 1 α: {model.prelu1.weight.data[:5].tolist()}")
    print(f"  Layer 2 α: {model.prelu2.weight.data[:5].tolist()}")
    print("\n✅ PReLU learned optimal negative slopes!")
    
    # Visualize learned activation
    x = torch.linspace(-3, 3, 1000).unsqueeze(1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Use first channel's α
    alpha1 = model.prelu1.weight.data[0].item()
    alpha2 = model.prelu2.weight.data[0].item()
    
    # Plot learned activations
    y1 = torch.where(x > 0, x, alpha1 * x)
    y2 = torch.where(x > 0, x, alpha2 * x)
    y_relu = F.relu(x)
    
    axes[0].plot(x.numpy(), y1.numpy(), linewidth=2.5, 
                 label=f'PReLU Layer 1 (α={alpha1:.3f})', color='darkblue')
    axes[0].plot(x.numpy(), y_relu.numpy(), linewidth=2, linestyle='--',
                 label='ReLU', color='red', alpha=0.5)
    axes[0].set_title('Learned PReLU - Layer 1', fontweight='bold')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='k', linestyle=':', alpha=0.3)
    axes[0].axvline(0, color='k', linestyle=':', alpha=0.3)
    
    axes[1].plot(x.numpy(), y2.numpy(), linewidth=2.5,
                 label=f'PReLU Layer 2 (α={alpha2:.3f})', color='darkgreen')
    axes[1].plot(x.numpy(), y_relu.numpy(), linewidth=2, linestyle='--',
                 label='ReLU', color='red', alpha=0.5)
    axes[1].set_title('Learned PReLU - Layer 2', fontweight='bold')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('f(x)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='k', linestyle=':', alpha=0.3)
    axes[1].axvline(0, color='k', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    
    return fig

# ==============================================================================
# SECTION 2: Adaptive Activation Functions
# ==============================================================================

class AdaptiveActivation(nn.Module):
    """
    Adaptive activation that changes shape during training.
    Combines multiple activations with learnable weights.
    """
    def __init__(self, num_activations=3):
        super().__init__()
        # Learnable weights for combining activations
        self.weights = nn.Parameter(torch.ones(num_activations) / num_activations)
    
    def forward(self, x):
        # Combine: ReLU, tanh, and x (identity)
        activations = torch.stack([
            F.relu(x),
            torch.tanh(x),
            x  # identity
        ], dim=0)
        
        # Softmax to ensure weights sum to 1
        weights = F.softmax(self.weights, dim=0).view(-1, 1, 1)
        
        # Weighted combination
        return (activations * weights).sum(dim=0)

def demonstrate_adaptive():
    """Demonstrate adaptive activation"""
    print("\n" + "=" * 70)
    print("SECTION 2: Adaptive Activation Functions")
    print("=" * 70)
    
    activation = AdaptiveActivation()
    
    print("\nInitial weights:")
    weights = F.softmax(activation.weights, dim=0)
    print(f"  ReLU:  {weights[0].item():.3f}")
    print(f"  Tanh:  {weights[1].item():.3f}")
    print(f"  Identity: {weights[2].item():.3f}")
    
    # Simple training
    X = torch.randn(50, 5)
    y = torch.randn(50, 5)
    
    fc = nn.Linear(5, 5)
    optimizer = torch.optim.Adam(list(activation.parameters()) + list(fc.parameters()), lr=0.01)
    
    print("\nTraining adaptive activation...")
    for epoch in range(100):
        optimizer.zero_grad()
        hidden = fc(X)
        output = activation(hidden)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
    
    print("\nFinal weights:")
    weights = F.softmax(activation.weights, dim=0)
    print(f"  ReLU:  {weights[0].item():.3f}")
    print(f"  Tanh:  {weights[1].item():.3f}")
    print(f"  Identity: {weights[2].item():.3f}")
    print("\n✅ Activation learned optimal combination!")
    
    # Visualize
    x = torch.linspace(-3, 3, 1000).unsqueeze(1)
    y_adaptive = activation(x.expand(-1, 5))[:, 0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y_adaptive.detach().numpy(), 
             linewidth=3, color='purple', label='Learned Adaptive')
    plt.plot(x.numpy(), F.relu(x).numpy(), linewidth=2,
             linestyle='--', alpha=0.5, label='ReLU', color='red')
    plt.plot(x.numpy(), torch.tanh(x).numpy(), linewidth=2,
             linestyle='--', alpha=0.5, label='Tanh', color='blue')
    plt.axhline(0, color='k', linestyle=':', alpha=0.3)
    plt.axvline(0, color='k', linestyle=':', alpha=0.3)
    plt.title('Adaptive Activation (Learned Combination)', fontweight='bold', fontsize=14)
    plt.xlabel('x', fontsize=12); plt.ylabel('f(x)', fontsize=12)
    plt.legend(fontsize=11); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

# ==============================================================================
# SECTION 3: Gradient Flow Analysis
# ==============================================================================

def analyze_gradient_flow():
    """Analyze gradient flow through different activations"""
    print("\n" + "=" * 70)
    print("SECTION 3: Gradient Flow Analysis")
    print("=" * 70)
    
    class DeepNetwork(nn.Module):
        def __init__(self, activation_fn, num_layers=10):
            super().__init__()
            layers = []
            for i in range(num_layers):
                layers.append(nn.Linear(64, 64))
                layers.append(activation_fn())
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    # Test different activations
    activations = {
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'GELU': nn.GELU
    }
    
    gradient_norms = {}
    
    print("\nTesting gradient flow in 10-layer networks...")
    print("-" * 70)
    
    for name, activation_fn in activations.items():
        model = DeepNetwork(activation_fn, num_layers=10)
        
        # Forward pass
        x = torch.randn(32, 64, requires_grad=True)
        output = model(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Measure gradient norm
        grad_norm = x.grad.norm().item()
        gradient_norms[name] = grad_norm
        
        print(f"{name:10s}: Gradient norm = {grad_norm:.6f}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green', 'red', 'orange', 'blue']
    bars = ax.bar(gradient_norms.keys(), gradient_norms.values(), 
                   color=colors, alpha=0.7)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Flow in 10-Layer Networks', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, (name, value) in zip(bars, gradient_norms.items()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    print("\n💡 Interpretation:")
    print("  • Higher gradient norm = Better gradient flow")
    print("  • Sigmoid/Tanh suffer from vanishing gradients in deep networks")
    print("  • ReLU and GELU maintain better gradient flow")
    
    return fig

# ==============================================================================
# SECTION 4: Dead ReLU Problem and Solutions
# ==============================================================================

def demonstrate_dead_relu():
    """Demonstrate dead ReLU problem and solutions"""
    print("\n" + "=" * 70)
    print("SECTION 4: Dead ReLU Problem and Solutions")
    print("=" * 70)
    
    print("\nThe Dead ReLU Problem:")
    print("  • ReLU neurons can 'die' during training")
    print("  • Happens when neuron outputs 0 for all inputs")
    print("  • Gradient is 0, so weights never update")
    print("  • Caused by: large negative bias or large negative gradients")
    
    # Simulate dead ReLU
    print("\n📊 Simulation:")
    
    # Create data where some neurons will die
    X = torch.randn(100, 10)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Intentionally push some neurons negative
    with torch.no_grad():
        model[0].bias[:10] = -100  # Force half the neurons to be dead
    
    # Forward pass
    activations = model[1](model[0](X))
    
    # Count dead neurons
    dead_neurons = (activations.max(dim=0)[0] == 0).sum().item()
    active_neurons = 20 - dead_neurons
    
    print(f"  Total neurons: 20")
    print(f"  Dead neurons:  {dead_neurons}")
    print(f"  Active neurons: {active_neurons}")
    
    # Solutions
    print("\n✅ Solutions to Dead ReLU:")
    print("  1. Use Leaky ReLU: Allows small gradient for negative values")
    print("  2. Use ELU: Smooth, allows negative values")
    print("  3. Use PReLU: Learn optimal negative slope")
    print("  4. Lower learning rate: Prevent large weight updates")
    print("  5. Proper initialization: Use He initialization for ReLU")
    print("  6. Batch Normalization: Helps maintain healthy activations")
    
    # Compare activations
    x = torch.linspace(-3, 3, 1000)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x.numpy(), F.relu(x).numpy(), linewidth=2.5,
            label='ReLU (dies for x<0)', color='red')
    ax.plot(x.numpy(), F.leaky_relu(x, 0.1).numpy(), linewidth=2.5,
            label='Leaky ReLU (survives!)', color='green')
    ax.plot(x.numpy(), F.elu(x).numpy(), linewidth=2.5,
            label='ELU (survives!)', color='blue')
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax.fill_between([-3, 0], -2, 0, alpha=0.2, color='red',
                     label='Dead zone for ReLU')
    ax.set_title('Solutions to Dead ReLU Problem', fontweight='bold', fontsize=14)
    ax.set_xlabel('x', fontsize=12); ax.set_ylabel('f(x)', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

# ==============================================================================
# SECTION 5: Best Practices Summary
# ==============================================================================

def print_best_practices():
    """Print comprehensive best practices"""
    print("\n" + "=" * 70)
    print("SECTION 5: Advanced Best Practices Summary 🎓")
    print("=" * 70)
    
    practices = [
        "\n1. ACTIVATION CHOICE BY TASK:",
        "   • Hidden layers (CNN): ReLU, Leaky ReLU, or PReLU",
        "   • Hidden layers (Transformer): GELU",
        "   • Hidden layers (RNN): Tanh or custom gates",
        "   • Output (binary classification): None (use BCEWithLogitsLoss)",
        "   • Output (multiclass): None (use CrossEntropyLoss)",
        "   • Output (regression): None or bounded activation",
        
        "\n2. DEALING WITH DYING NEURONS:",
        "   ✅ Use Leaky ReLU or PReLU instead of ReLU",
        "   ✅ Lower learning rate",
        "   ✅ Use proper weight initialization (He for ReLU)",
        "   ✅ Add Batch Normalization",
        "   ✅ Monitor activation statistics during training",
        
        "\n3. DEEP NETWORKS:",
        "   • Avoid sigmoid/tanh in deep networks (vanishing gradients)",
        "   • ReLU family works well for very deep networks",
        "   • Consider residual connections (ResNet)",
        "   • Use Batch Normalization or Layer Normalization",
        
        "\n4. LEARNABLE ACTIVATIONS:",
        "   • PReLU: Good for CNNs, learns negative slope",
        "   • Adaptive: Can learn task-specific activation",
        "   • Requires more parameters and training time",
        "   • Monitor for overfitting",
        
        "\n5. INITIALIZATION:",
        "   • ReLU family: Use He initialization",
        "   • Tanh: Use Xavier/Glorot initialization",
        "   • PyTorch default: Kaiming uniform (good for ReLU)",
        
        "\n6. DEBUGGING ACTIVATIONS:",
        "   • Plot activation distributions during training",
        "   • Check for saturated neurons (always 0 or max)",
        "   • Monitor gradient norms",
        "   • Visualize activation functions",
        
        "\n7. MODERN RECOMMENDATIONS:",
        "   • Default: Start with ReLU",
        "   • Transformers: Use GELU",
        "   • Mobile/Edge: Use Hardswish",
        "   • Research: Try Swish/Mish",
        "   • Problems with ReLU: Try Leaky ReLU or ELU"
    ]
    
    for practice in practices:
        print(practice)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("\n" + "█" * 70)
    print("   PYTORCH ACTIVATION FUNCTIONS TUTORIAL")
    print("   Script 10: Advanced Techniques")
    print("   (FINAL SCRIPT)")
    print("█" * 70)
    
    fig1 = demonstrate_prelu()
    fig2 = demonstrate_adaptive()
    fig3 = analyze_gradient_flow()
    fig4 = demonstrate_dead_relu()
    print_best_practices()
    
    print("\n" + "=" * 70)
    print("🎉 CONGRATULATIONS!")
    print("=" * 70)
    print("You have completed the PyTorch Activation Functions Tutorial!")
    print()
    print("You now understand:")
    print("  ✅ What activation functions are and why they're needed")
    print("  ✅ How to use PyTorch's Functional and Module APIs")
    print("  ✅ Classical activations (Sigmoid, Tanh, ReLU)")
    print("  ✅ Modern activations (GELU, Swish, Mish)")
    print("  ✅ Binary and multiclass classification")
    print("  ✅ Regression tasks")
    print("  ✅ Creating custom activations")
    print("  ✅ Comparing activation performance")
    print("  ✅ Advanced techniques (PReLU, adaptive, gradient flow)")
    print()
    print("🚀 You're ready to build your own deep learning models!")
    print("=" * 70)
    
    print("\n📊 Showing plots... (Close windows to exit)")
    plt.show()
    
    print("\n✅ Tutorial completed successfully! Happy coding! 🎓🔥\n")

if __name__ == "__main__":
    main()
