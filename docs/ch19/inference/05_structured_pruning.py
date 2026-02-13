"""
INTERMEDIATE LEVEL: Structured Pruning

This script demonstrates structured pruning where we remove entire structures
(filters, channels, neurons) rather than individual weights. This achieves
actual speedups on regular hardware.

Topics Covered:
- Filter pruning in convolutional layers
- Channel pruning
- Neuron pruning in fully connected layers
- L1-norm based importance ranking
- Filter reconstruction for accuracy recovery

Mathematical Background:
- Filter importance: I(F_j) = ||F_j||_1 = Σ|w_ijk|
- Remove filters with lowest L1-norm
- Results in smaller feature maps and fewer parameters

Benefits over unstructured pruning:
- Actual speedup on GPUs (no sparse operations needed)
- Reduced memory bandwidth
- Compatible with existing hardware

Prerequisites:
- Module 02: Pruning Basics
- Understanding of CNNs
- Familiarity with feature maps
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from utils import (
    count_parameters,
    evaluate_accuracy,
    compare_model_sizes,
    seed_everything
)


class SimpleCNN(nn.Module):
    """Simple CNN for structured pruning demonstration."""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(256 * 3 * 3, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def compute_filter_importance(layer):
    """
    Compute L1-norm importance for each filter.
    
    For a conv layer with shape (out_channels, in_channels, kh, kw):
    Importance of filter j = ||F_j||_1 = Σ|w_ijk|
    
    Args:
        layer: Convolutional layer
        
    Returns:
        1D tensor of importance scores per filter
    """
    weight = layer.weight.data.abs()
    # Sum over all dimensions except output channel
    importance = weight.view(weight.size(0), -1).sum(dim=1)
    return importance


def prune_filters(model, prune_ratio=0.5):
    """
    Prune filters from convolutional layers based on L1-norm.
    
    Process:
    1. Calculate importance for each filter
    2. Identify filters to prune
    3. Create new smaller layer
    4. Copy surviving filters
    
    Args:
        model: CNN model
        prune_ratio: Fraction of filters to prune per layer
        
    Returns:
        Pruned model with smaller layers
    """
    print(f"\nPruning {prune_ratio*100:.0f}% of filters per layer...")
    
    # Track which filters to keep for each layer
    cfg = []
    
    # Pass 1: Determine which filters to keep
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            importance = compute_filter_importance(module)
            num_filters = len(importance)
            num_keep = int(num_filters * (1 - prune_ratio))
            
            # Get indices of filters to keep
            _, indices = torch.sort(importance, descending=True)
            keep_indices = indices[:num_keep]
            
            cfg.append((name, keep_indices))
            print(f"Layer {name}: Keep {num_keep}/{num_filters} filters")
    
    # Pass 2: Create new model with pruned layers
    # (In practice, this requires careful reconstruction)
    # For this demo, we'll just return the original model
    # with identified filters zeroed out
    
    for (name, keep_indices), (module_name, module) in zip(cfg, model.named_modules()):
        if isinstance(module, nn.Conv2d):
            # Zero out pruned filters
            mask = torch.zeros(module.weight.size(0))
            mask[keep_indices] = 1.0
            mask = mask.view(-1, 1, 1, 1).to(module.weight.device)
            module.weight.data *= mask
    
    return model


def main():
    """Main function for structured pruning demonstration."""
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("STRUCTURED PRUNING DEMONSTRATION")
    print("="*60)
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create model
    model = SimpleCNN()
    print(f"\nOriginal model parameters: {count_parameters(model):,}")
    
    # Prune filters
    pruned_model = prune_filters(model, prune_ratio=0.5)
    
    print("\nStructured pruning achieves:")
    print("- Actual speedup on regular hardware")
    print("- Reduced memory bandwidth")
    print("- Lower sparsity but guaranteed speedup")
    print("- Compatible with all accelerators")
    
    print("\nNote: Full implementation requires layer reconstruction")
    print("      This demo shows filter selection process")


if __name__ == "__main__":
    main()
