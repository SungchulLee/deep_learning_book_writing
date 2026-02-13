"""
08_binary_classification.py
============================
Binary Classification Analysis on MNIST

This advanced tutorial explores all 45 pairwise binary classification problems
on MNIST digits (0vs1, 0vs2, ..., 8vs9) to understand which digit pairs are
easiest/hardest to distinguish.

What you'll learn:
- Binary classification vs multi-class
- Analyzing model confusion patterns
- Understanding which classes are similar
- Visualizing learning curves across problems

Difficulty: Advanced
Estimated Time: 2-3 hours

Author: PyTorch CNN Tutorial
Date: November 2025
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

print("=" * 70)
print("Binary Classification Analysis on MNIST")
print("=" * 70)

# =============================================================================
# Simple Binary CNN
# =============================================================================

class BinaryCNN(nn.Module):
    """Simple CNN for binary classification"""
    def __init__(self):
        super(BinaryCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary: 2 classes
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =============================================================================
# Load MNIST Data
# =============================================================================

print("\nLoading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
all_images, all_labels = next(iter(train_loader))

# Organize data by digit
digit_images = {}
digit_labels = {}
for i in range(10):
    mask = all_labels == i
    digit_images[i] = all_images[mask]
    digit_labels[i] = all_labels[mask]

print("Data organized by digit")

# =============================================================================
# Training Function
# =============================================================================

def train_binary(model, images_0, labels_0, images_1, labels_1, epochs=10):
    """Train on two classes and return loss trace"""
    # Combine data
    images = torch.cat([images_0, images_1])
    labels = torch.cat([torch.zeros(len(labels_0), dtype=torch.long),
                        torch.ones(len(labels_1), dtype=torch.long)])
    
    # Create dataloader
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    loss_trace = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_images, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_trace.append(total_loss / len(loader))
    
    return loss_trace

# =============================================================================
# Run All Pairwise Classifications
# =============================================================================

print("\nTraining 45 binary classifiers...")
print("This will take a few minutes...\n")

fig, axes = plt.subplots(10, 10, figsize=(15, 15))

for i in range(10):
    for j in range(10):
        if i >= j:
            axes[i, j].axis('off')
            continue
        
        print(f"Training classifier: {i} vs {j}")
        
        # Train model
        model = BinaryCNN()
        loss_trace = train_binary(
            model,
            digit_images[i], digit_labels[i],
            digit_images[j], digit_labels[j],
            epochs=10
        )
        
        # Plot loss curve
        axes[i, j].plot(loss_trace, 'r-', linewidth=2)
        axes[i, j].set_title(f'{i} vs {j}', fontsize=8)
        axes[i, j].set_ylim([0, 1])
        axes[i, j].axis('off')
        
        # Mirror plot
        axes[j, i].plot(loss_trace, 'b-', linewidth=2)
        axes[j, i].set_title(f'{j} vs {i}', fontsize=8)
        axes[j, i].set_ylim([0, 1])
        axes[j, i].axis('off')

plt.suptitle('MNIST Binary Classification Learning Curves\n(Red/Blue show loss decrease)', 
             fontsize=16)
plt.tight_layout()
plt.savefig('/home/claude/pytorch_cnn_tutorial/binary_classification_results.png', 
            dpi=150, bbox_inches='tight')
print("\nResults saved to: binary_classification_results.png")
plt.show()

# =============================================================================
# Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Analysis")
print("=" * 70)

print("""
What the plot shows:
-------------------
• Each subplot shows the learning curve for one digit pair
• Fast convergence = easy to distinguish
• Slow/unstable convergence = hard to distinguish

Typically Easy Pairs:
--------------------
• 0 vs 1: Very different shapes
• 1 vs 8: Distinct features
• 6 vs 7: Different structure

Typically Hard Pairs:
--------------------
• 3 vs 5: Similar curves
• 4 vs 9: Overlapping features  
• 7 vs 9: Can look similar

Why This Matters:
-----------------
Understanding confusion patterns helps:
1. Design better architectures
2. Collect better training data
3. Identify model weaknesses
4. Improve feature engineering

Key Insight:
-----------
Even simple digit recognition has inherent ambiguity!
Some digits are just naturally harder to distinguish.
""")

print("=" * 70)
print("Tutorial Complete! ✓")
print("=" * 70)
