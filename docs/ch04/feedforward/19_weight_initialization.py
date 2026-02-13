"""
11_weight_initialization.py - Starting with Good Weights

Proper initialization is crucial for successful training.
Bad init → vanishing/exploding gradients → slow/failed training

TIME: 25-30 minutes | DIFFICULTY: ⭐⭐⭐☆☆
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt

print("="*70)
print("Weight Initialization Strategies")
print("="*70)

class DemoNet(nn.Module):
    def __init__(self, init_method='xavier'):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Apply initialization
        self._initialize_weights(init_method)
    
    def _initialize_weights(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier':
                    init.xavier_uniform_(m.weight)
                elif method == 'he':
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == 'normal':
                    init.normal_(m.weight, mean=0, std=0.01)
                elif method == 'zeros':
                    init.zeros_(m.weight)
                
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

print("INITIALIZATION METHODS:")
print("-"*70)

methods = ['xavier', 'he', 'normal', 'zeros']
for method in methods:
    model = DemoNet(method)
    print(f"\n{method.upper()}:")
    w = model.fc1.weight.data
    print(f"  Mean: {w.mean():.6f}")
    print(f"  Std: {w.std():.6f}")
    print(f"  Min: {w.min():.6f}, Max: {w.max():.6f}")

print("\n" + "="*70)
print("INITIALIZATION GUIDE")
print("="*70)
print("""
XAVIER (GLOROT) INITIALIZATION:
  Formula: U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
  Use for: Sigmoid, Tanh activations
  Goal: Maintain variance across layers

HE (KAIMING) INITIALIZATION:
  Formula: U(-√(6/n_in), √(6/n_in))
  Use for: ReLU and variants
  Goal: Account for ReLU's non-linearity

NORMAL INITIALIZATION:
  Formula: N(0, 0.01)
  Use: Rarely, can be too small or large

ZEROS:
  Never use for weights! (breaks symmetry)
  OK for biases

RECOMMENDATIONS:
  ReLU networks → He initialization (default in PyTorch)
  Sigmoid/Tanh → Xavier initialization
  Biases → Zeros or small constant
  
PyTorch default: Kaiming uniform for Linear layers
""")

# Visualize weight distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, method in enumerate(methods):
    model = DemoNet(method)
    weights = model.fc1.weight.data.flatten().numpy()
    
    axes[idx].hist(weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[idx].set_title(f'{method.upper()} Initialization', fontweight='bold')
    axes[idx].set_xlabel('Weight Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].axvline(weights.mean(), color='r', linestyle='--', 
                     label=f'Mean={weights.mean():.3f}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('11_weight_init.png', dpi=150)
print("\nWeight distribution plots saved!")

print("\nEXERCISES:")
print("1. Train models with different initializations")
print("2. Visualize gradient flow with different inits")
print("3. Compare Xavier vs He for ReLU networks")
print("4. Implement custom initialization schemes")
plt.show()
