"""
06_activation_functions.py - Understanding Activation Functions

Compare different activation functions and their effects on training.
Visualize their behaviors and learn when to use each one.

TIME: 20-25 minutes | DIFFICULTY: ⭐⭐☆☆☆
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("Activation Functions Comparison")
print("="*70)

# Create input range for visualization
x = torch.linspace(-5, 5, 200)

# Dictionary of activation functions
activations = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'LeakyReLU': nn.LeakyReLU(0.1),
    'ELU': nn.ELU(),
    'Softplus': nn.Softplus()
}

# Plot activations
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, (name, activation) in enumerate(activations.items()):
    y = activation(x)
    axes[idx].plot(x.numpy(), y.numpy(), linewidth=2)
    axes[idx].set_title(name, fontsize=14, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axhline(y=0, color='k', linewidth=0.5)
    axes[idx].axvline(x=0, color='k', linewidth=0.5)
    axes[idx].set_xlabel('Input')
    axes[idx].set_ylabel('Output')

plt.tight_layout()
plt.savefig('06_activations.png', dpi=150)
print("Activation functions plotted and saved!")

print("\n" + "="*70)
print("ACTIVATION FUNCTION GUIDE")
print("="*70)
print("""
ReLU (Rectified Linear Unit)
  Formula: f(x) = max(0, x)
  Pros: Simple, fast, works well
  Cons: Dead neurons (neurons that output 0 for all inputs)
  Use: Default choice for hidden layers

Sigmoid
  Formula: f(x) = 1 / (1 + e^(-x))
  Pros: Smooth, bounded [0,1]
  Cons: Vanishing gradients, slow convergence
  Use: Binary classification output, gates in LSTMs

Tanh
  Formula: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
  Pros: Zero-centered, bounded [-1,1]
  Cons: Vanishing gradients
  Use: Sometimes in RNNs, better than sigmoid for hidden layers

LeakyReLU
  Formula: f(x) = max(0.1x, x)
  Pros: Fixes dead ReLU problem, small gradient when x < 0
  Cons: Inconsistent benefits
  Use: When facing dead ReLU problems

ELU (Exponential Linear Unit)
  Formula: f(x) = x if x>0 else α(e^x - 1)
  Pros: Smooth, mean activations closer to zero
  Cons: More expensive to compute
  Use: When ReLU doesn't work well

Softplus
  Formula: f(x) = log(1 + e^x)
  Pros: Smooth approximation of ReLU
  Cons: More expensive
  Use: Rarely in practice, theoretical interest

RECOMMENDATIONS:
  - Hidden layers: ReLU (default), LeakyReLU if needed
  - Binary output: Sigmoid
  - Multi-class output: Softmax (via CrossEntropyLoss)
  - Regression output: None (linear)
""")

print("\nEXERCISES:")
print("1. Train MNIST with each activation - compare results")
print("2. Visualize gradient flow for each activation")
print("3. Implement custom activation function")
plt.show()
