"""
06_cifar10_basic.py
===================
Basic CNN for CIFAR-10 (Simpler Understanding Version)

This script provides a simpler, easy-to-understand CNN for CIFAR-10.
It won't achieve the best accuracy, but it's great for learning!

Based on the classic "spaghetti code" example but cleaned up and commented.

What you'll learn:
- Adapting CNNs for RGB images
- Why CIFAR-10 is much harder
- Basic CNN architecture for color images
- Understanding the challenges of natural images

Difficulty: Intermediate-Challenging
Estimated Time: 1-2 hours

Author: PyTorch CNN Tutorial
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: Configuration
# =============================================================================

print("=" * 70)
print("Basic CNN for CIFAR-10")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
num_epochs = 5  # Start with few epochs to see results quickly
batch_size = 64
learning_rate = 0.001

print(f"\nConfiguration:")
print(f"  Epochs: {num_epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")

# =============================================================================
# SECTION 2: Data Loading
# =============================================================================

print("\n" + "=" * 70)
print("Loading CIFAR-10 Dataset")
print("=" * 70)

# Transform: Convert to tensor and normalize
# For CIFAR-10, we normalize all 3 channels (R, G, B)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Dataset loaded!")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")
print(f"  Training batches: {len(train_loader)}")

# Class names for visualization
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# =============================================================================
# SECTION 3: Simple CNN Architecture
# =============================================================================

print("\n" + "=" * 70)
print("Defining CNN Architecture")
print("=" * 70)

class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10
    
    Architecture:
    - Conv1: 3→6 channels, 5×5 kernel
    - Pool: 2×2 max pooling
    - Conv2: 6→16 channels, 5×5 kernel
    - Pool: 2×2 max pooling
    - FC1: 16*5*5 → 120
    - FC2: 120 → 84
    - FC3: 84 → 10
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)    # 3 input channels (RGB)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # Input: (N, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))  # → (N, 6, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # → (N, 16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)            # → (N, 400)
        x = F.relu(self.fc1(x))               # → (N, 120)
        x = F.relu(self.fc2(x))               # → (N, 84)
        x = self.fc3(x)                       # → (N, 10)
        return x

model = SimpleCNN().to(device)

print("Model architecture:")
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")
print("\nNote: This is a SIMPLE model for learning purposes")
print("      It won't achieve state-of-the-art accuracy")

# =============================================================================
# SECTION 4: Training Setup
# =============================================================================

print("\n" + "=" * 70)
print("Training Setup")
print("=" * 70)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print("Loss: CrossEntropyLoss")
print(f"Optimizer: SGD with lr={learning_rate}")

# =============================================================================
# SECTION 5: Training Loop
# =============================================================================

print("\n" + "=" * 70)
print("Training")
print("=" * 70)

print(f"\nTraining for {num_epochs} epochs...")
print("This will take longer than MNIST due to:")
print("  • More complex images (3 channels)")
print("  • Harder classification task")
print("  • More data to process\n")

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print progress
        if (i + 1) % 200 == 0:
            avg_loss = running_loss / 200
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{i+1}/{n_total_steps}], '
                  f'Loss: {avg_loss:.4f}')
            running_loss = 0.0

print('\nTraining finished!')

# =============================================================================
# SECTION 6: Evaluation
# =============================================================================

print("\n" + "=" * 70)
print("Evaluation")
print("=" * 70)

model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(10)]
    n_class_samples = [0 for _ in range(10)]
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    
    acc = 100.0 * n_correct / n_samples
    print(f'\nOverall Accuracy: {acc:.2f}%')
    
    print('\nPer-class Accuracy:')
    print(f"{'Class':<15} | {'Accuracy':<10}")
    print("-" * 30)
    for i in range(10):
        class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'{classes[i]:<15} | {class_acc:>9.2f}%')

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(f"""
Results with Simple CNN:
-----------------------
• Overall Accuracy: {acc:.2f}%
• Total Parameters: {total_params:,}
• Training Epochs: {num_epochs}

Why accuracy is lower than MNIST:
---------------------------------
1. Natural images are much more complex
2. High intra-class variation
3. Backgrounds add noise
4. Small image size (32×32) limits detail
5. This model is intentionally simple

Performance Expectations:
------------------------
• This simple model: 60-70%
• Deeper CNN (next tutorial): 75-85%
• State-of-the-art: 95%+
• Human performance: ~94%

What's Next:
-----------
Move to 07_cifar10_advanced.py for better performance!
We'll use:
• Deeper architecture
• More filters
• Batch normalization
• Better training strategies

This simple model taught us the basics.
Now let's build something more powerful! 🚀
""")

print("=" * 70)
print("Tutorial Complete! ✓")
print("=" * 70)
