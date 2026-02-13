#!/usr/bin/env python3
"""
Script 08: Creating Custom Activation Functions
DIFFICULTY: ⭐⭐⭐⭐ Hard | TIME: 12 min | PREREQ: Scripts 01-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Custom Activation Functions
# ==============================================================================

class CustomSmooth(nn.Module):
    """
    Custom smooth activation: f(x) = x * tanh(sqrt(x^2 + 1))
    Smooth alternative to ReLU with better gradient properties
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.tanh(torch.sqrt(x**2 + 1))

class ParametricActivation(nn.Module):
    """
    Learnable activation with trainable parameter.
    f(x) = x if x > 0 else alpha * x
    Similar to PReLU but demonstrates learnable parameters.
    """
    def __init__(self, init_alpha=0.1):
        super().__init__()
        # Learnable parameter
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
    
    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)

class SoftClip(nn.Module):
    """
    Soft clipping activation: bounds output smoothly.
    f(x) = tanh(x/scale) * scale
    """
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        return torch.tanh(x / self.scale) * self.scale

def plot_custom_activations():
    """Visualize custom activations"""
    x = torch.linspace(-5, 5, 1000)
    
    # Create instances
    custom_smooth = CustomSmooth()
    parametric = ParametricActivation(init_alpha=0.2)
    soft_clip = SoftClip(scale=2.0)
    
    # Compute outputs
    y_custom = custom_smooth(x)
    y_param = parametric(x)
    y_clip = soft_clip(x)
    y_relu = F.relu(x)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Custom Smooth
    axes[0].plot(x.numpy(), y_custom.detach().numpy(), 
                 linewidth=2.5, color='purple', label='Custom Smooth')
    axes[0].plot(x.numpy(), y_relu.numpy(), 
                 linewidth=2, linestyle='--', alpha=0.5, color='red', label='ReLU')
    axes[0].axhline(0, color='k', linestyle=':', alpha=0.3)
    axes[0].axvline(0, color='k', linestyle=':', alpha=0.3)
    axes[0].set_title('Custom Smooth Activation', fontweight='bold')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Parametric
    axes[1].plot(x.numpy(), y_param.detach().numpy(), 
                 linewidth=2.5, color='darkgreen', label=f'Parametric (α={parametric.alpha.item():.2f})')
    axes[1].plot(x.numpy(), y_relu.numpy(), 
                 linewidth=2, linestyle='--', alpha=0.5, color='red', label='ReLU')
    axes[1].axhline(0, color='k', linestyle=':', alpha=0.3)
    axes[1].axvline(0, color='k', linestyle=':', alpha=0.3)
    axes[1].set_title('Parametric Activation (Learnable)', fontweight='bold')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('f(x)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Soft Clip
    axes[2].plot(x.numpy(), y_clip.detach().numpy(), 
                 linewidth=2.5, color='navy', label='Soft Clip')
    axes[2].axhline(2, color='r', linestyle='--', alpha=0.5, label='Upper bound')
    axes[2].axhline(-2, color='r', linestyle='--', alpha=0.5, label='Lower bound')
    axes[2].axhline(0, color='k', linestyle=':', alpha=0.3)
    axes[2].axvline(0, color='k', linestyle=':', alpha=0.3)
    axes[2].set_title('Soft Clip Activation', fontweight='bold')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('f(x)')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

class CustomActivationNetwork(nn.Module):
    """Network using custom activation"""
    def __init__(self, activation_fn):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.activation = activation_fn
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

def test_custom_activation():
    """Test custom activation in a simple classification task"""
    from sklearn.datasets import make_moons
    
    # Data
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze(1)
    
    # Train with custom activation
    model = CustomActivationNetwork(CustomSmooth())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("\nTraining with Custom Smooth Activation...")
    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                acc = ((probs > 0.5).float() == y).float().mean()
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")
    
    return model

def demonstrate_learnable_parameters():
    """Show how parametric activation learns during training"""
    print("\n" + "=" * 70)
    print("Demonstrating Learnable Parameters")
    print("=" * 70)
    
    # Simple data
    X = torch.randn(100, 2)
    y = (X.sum(dim=1) > 0).float().unsqueeze(1)
    
    # Model with parametric activation
    activation = ParametricActivation(init_alpha=0.1)
    print(f"\nInitial alpha parameter: {activation.alpha.item():.4f}")
    
    model = CustomActivationNetwork(activation)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Track alpha over training
    alphas = [activation.alpha.item()]
    
    for epoch in range(50):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        alphas.append(activation.alpha.item())
    
    print(f"Final alpha parameter: {activation.alpha.item():.4f}")
    print(f"Change: {activation.alpha.item() - alphas[0]:.4f}")
    
    # Plot alpha evolution
    plt.figure(figsize=(10, 5))
    plt.plot(alphas, linewidth=2, color='darkgreen')
    plt.xlabel('Training Step'); plt.ylabel('Alpha Value')
    plt.title('Evolution of Learnable Parameter α', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print("\n→ The activation function learned its optimal parameter!")
    
    return plt.gcf()

def main():
    print("\n" + "█" * 70)
    print("   Script 08: Creating Custom Activation Functions")
    print("█" * 70)
    
    print("\n" + "=" * 70)
    print("SECTION 1: Visualizing Custom Activations")
    print("=" * 70)
    fig1 = plot_custom_activations()
    
    print("\n" + "=" * 70)
    print("SECTION 2: Testing Custom Activation in Network")
    print("=" * 70)
    model = test_custom_activation()
    print("✅ Custom activation works in real network!")
    
    print("\n" + "=" * 70)
    print("SECTION 3: Learnable Parameters")
    print("=" * 70)
    fig2 = demonstrate_learnable_parameters()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Custom activations: Inherit from nn.Module")
    print("2. Implement forward(self, x) method")
    print("3. Can have learnable parameters using nn.Parameter()")
    print("4. PyTorch autograd handles gradients automatically")
    print("5. Test custom activations on simple tasks first")
    
    print("\n" + "=" * 70)
    print("Creating custom activations allows you to:")
    print("• Experiment with novel designs")
    print("• Add domain-specific inductive biases")
    print("• Learn activation shapes during training")
    print("=" * 70)
    
    print("\n✅ Next: Run '09_activation_comparison.py'")
    plt.show()

if __name__ == "__main__":
    main()
