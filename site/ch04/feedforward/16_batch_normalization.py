"""
09_batch_normalization.py - Stable and Fast Training

Batch Normalization normalizes layer inputs, leading to:
- Faster training
- Higher learning rates possible  
- Less sensitive to initialization
- Acts as regularization

TIME: 30-35 minutes | DIFFICULTY: ⭐⭐⭐☆☆
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("Batch Normalization")
print("="*70)

# Model WITHOUT Batch Norm
class NoBNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Model WITH Batch Norm (recommended placement)
class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.bn1(x)  # Batch norm BEFORE activation
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return self.fc3(x)

print("Models created!")
print(f"Without BN: {sum(p.numel() for p in NoBNNet().parameters())} params")
print(f"With BN: {sum(p.numel() for p in BNNet().parameters())} params")

print("\n" + "="*70)
print("HOW BATCH NORMALIZATION WORKS")
print("="*70)
print("""
For each mini-batch:
1. Compute mean and variance of layer inputs
2. Normalize: x_norm = (x - mean) / sqrt(var + ε)
3. Scale and shift: y = γ * x_norm + β
   where γ and β are learnable parameters

BENEFITS:
✓ Reduces internal covariate shift
✓ Allows higher learning rates (10x faster training)
✓ Less sensitive to initialization
✓ Acts as regularization (slight noise from batch statistics)
✓ Can sometimes replace dropout

USAGE:
- During training: Use batch statistics
- During evaluation: Use running averages

PLACEMENT:
- Typically: Linear → BatchNorm → Activation
- Alternative: Linear → Activation → BatchNorm
  (both work, first is more common)

PARAMETERS:
- Input: Number of features to normalize
- For fully connected: num_features = output_dim
- For conv layers: num_features = num_channels

IMPORTANT NOTES:
⚠ Requires batch_size > 1 (needs multiple samples)
⚠ Different behavior in train vs eval mode
⚠ Remember to call model.train() and model.eval()!
""")

print("\n" + "="*70)
print("BATCH NORMALIZATION VARIANTS")
print("="*70)
print("""
BatchNorm1d: For fully connected layers (batch, features)
BatchNorm2d: For conv layers (batch, channels, height, width)
BatchNorm3d: For 3D data (batch, channels, depth, height, width)

LayerNorm: Normalizes across features (used in Transformers)
InstanceNorm: Normalizes each sample independently
GroupNorm: Hybrid between Layer and Instance norm
""")

# Visualize batch norm effect
def visualize_batch_norm():
    # Simulate layer outputs before and after batch norm
    data = torch.randn(1000, 1) * 5 + 10  # Mean=10, Std=5
    
    bn = nn.BatchNorm1d(1)
    bn.eval()  # Use running stats
    data_normalized = bn(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(data.numpy(), bins=50, alpha=0.7, color='blue')
    ax1.set_title('Before Batch Norm', fontweight='bold')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.axvline(data.mean(), color='r', linestyle='--', label=f'Mean={data.mean():.2f}')
    ax1.legend()
    
    ax2.hist(data_normalized.detach().numpy(), bins=50, alpha=0.7, color='green')
    ax2.set_title('After Batch Norm', fontweight='bold')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.axvline(data_normalized.mean(), color='r', linestyle='--', 
                label=f'Mean={data_normalized.mean():.2f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('09_batch_norm_effect.png', dpi=150)
    print("Visualization saved!")
    plt.show()

visualize_batch_norm()

print("\nEXERCISES:")
print("1. Compare training speed with/without BatchNorm")
print("2. Try different placement: before vs after activation")
print("3. Visualize internal activations with/without BN")
print("4. Experiment with different network depths")
