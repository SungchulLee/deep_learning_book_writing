"""
================================================================================
Level 4 - Project 1: Neural Network for MNIST Digit Classification
================================================================================

LEARNING OBJECTIVES:
- Build a complete neural network from scratch
- Train on a real-world dataset (MNIST)
- Implement proper train/validation/test splits
- Apply all gradient descent concepts learned
- Achieve >95% accuracy

DIFFICULTY: ⭐⭐⭐⭐ Project

TIME: 60-90 minutes

PREREQUISITES:
- Completed Levels 1-3
- Understanding of neural networks
- Comfortable with PyTorch

PROJECT DESCRIPTION:
--------------------
MNIST is a dataset of 70,000 handwritten digits (0-9):
- Training set: 60,000 images
- Test set: 10,000 images
- Image size: 28x28 grayscale
- Task: Classify each image into one of 10 classes (0-9)

This is a classic benchmark problem in machine learning!

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time

print("="*80)
print("MNIST DIGIT CLASSIFICATION WITH NEURAL NETWORKS")
print("="*80)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# PART 1: Data Loading and Preprocessing
# ============================================================================
print("\n" + "="*80)
print("PART 1: LOADING MNIST DATASET")
print("="*80)

# Define transformations
# Normalize with mean=0.1307, std=0.3081 (MNIST statistics)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize
])

# Download and load training data
print("Downloading MNIST dataset (this may take a minute)...")
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

# Download and load test data
test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# Create validation split from training data
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # 20% for validation

train_dataset, val_dataset = random_split(
    train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"\n✓ Dataset loaded successfully!")
print(f"  Training samples:   {len(train_dataset):,}")
print(f"  Validation samples: {len(val_dataset):,}")
print(f"  Test samples:       {len(test_dataset):,}")

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nBatch size: {batch_size}")
print(f"Batches per epoch: {len(train_loader)}")

# ============================================================================
# PART 2: Visualize Sample Data
# ============================================================================
print("\n" + "="*80)
print("PART 2: VISUALIZING SAMPLE DATA")
print("="*80)

# Get a batch of training data
examples = iter(train_loader)
example_data, example_targets = next(examples)

print(f"\nBatch shape: {example_data.shape}")  # (batch_size, 1, 28, 28)
print(f"Labels shape: {example_targets.shape}")  # (batch_size,)

# Plot first 10 images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for i in range(10):
    img = example_data[i].squeeze()  # Remove channel dimension
    label = example_targets[i].item()
    
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_4_projects/mnist_samples.png', dpi=150)
print("\n✓ Sample images saved as 'mnist_samples.png'")

# ============================================================================
# PART 3: Define Neural Network Architecture
# ============================================================================
print("\n" + "="*80)
print("PART 3: NEURAL NETWORK ARCHITECTURE")
print("="*80)

class MNISTNet(nn.Module):
    """
    Neural Network for MNIST Classification
    
    Architecture:
    - Input: 28x28 = 784 features
    - Hidden Layer 1: 128 neurons + ReLU
    - Hidden Layer 2: 64 neurons + ReLU
    - Output Layer: 10 neurons (one per digit)
    """
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # Input layer: 784 (28x28) → 128
        self.fc1 = nn.Linear(28 * 28, 128)
        
        # Hidden layer: 128 → 64
        self.fc2 = nn.Linear(128, 64)
        
        # Output layer: 64 → 10 (classes 0-9)
        self.fc3 = nn.Linear(64, 10)
        
        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            output: logits of shape (batch_size, 10)
        """
        # Flatten image: (batch, 1, 28, 28) → (batch, 784)
        x = x.view(-1, 28 * 28)
        
        # Hidden layer 1 with ReLU activation
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Hidden layer 2 with ReLU activation
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer (no activation - will use CrossEntropyLoss)
        x = self.fc3(x)
        
        return x

# Create model
model = MNISTNet()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nModel Architecture:")
print(model)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# PART 4: Training Setup
# ============================================================================
print("\n" + "="*80)
print("PART 4: TRAINING CONFIGURATION")
print("="*80)

# Loss function: CrossEntropyLoss for classification
# Combines LogSoftmax and NLLLoss
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam (our favorite from Level 3!)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler: reduce LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=3, 
    verbose=True
)

print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: Adam")
print(f"Learning rate: {learning_rate}")
print(f"Scheduler: ReduceLROnPlateau")

# ============================================================================
# PART 5: Training and Validation Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch"""
    model.train()  # Set model to training mode
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion):
    """Validate the model"""
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradients needed
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

# ============================================================================
# PART 6: Training Loop
# ============================================================================
print("\n" + "="*80)
print("PART 6: TRAINING")
print("="*80)

n_epochs = 15

# Track history
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print(f"\nTraining for {n_epochs} epochs...")
print("-" * 80)
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>10} | {'Time':>7}")
print("-" * 80)

start_time = time.time()

for epoch in range(n_epochs):
    epoch_start = time.time()
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Store history
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    epoch_time = time.time() - epoch_start
    
    print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.2f}% | {val_loss:10.4f} | {val_acc:9.2f}% | {epoch_time:6.1f}s")

total_time = time.time() - start_time
print("-" * 80)
print(f"Training completed in {total_time:.1f}s ({total_time/n_epochs:.1f}s per epoch)")

# ============================================================================
# PART 7: Test Set Evaluation
# ============================================================================
print("\n" + "="*80)
print("PART 7: FINAL EVALUATION ON TEST SET")
print("="*80)

test_loss, test_acc = validate(model, test_loader, criterion)

print(f"\nTest Set Results:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.2f}%")

if test_acc > 95:
    print("\n🎉 Congratulations! You achieved >95% accuracy!")
elif test_acc > 90:
    print("\n✓ Good job! Try tuning hyperparameters to reach 95%")
else:
    print("\n→ Try training longer or adjusting the architecture")

# ============================================================================
# PART 8: Visualization
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Loss curves
axes[0, 0].plot(train_losses, label='Train', linewidth=2)
axes[0, 0].plot(val_losses, label='Validation', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Accuracy curves
axes[0, 1].plot(train_accuracies, label='Train', linewidth=2)
axes[0, 1].plot(val_accuracies, label='Validation', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('Training and Validation Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Sample predictions
model.eval()
examples = iter(test_loader)
example_data, example_targets = next(examples)

with torch.no_grad():
    output = model(example_data)
    _, predictions = torch.max(output, 1)

# Show first 6 test examples
for i in range(6):
    ax = axes[1, i//3]
    
    if i < 3:
        idx = i
    else:
        idx = i + 3
    
    img = example_data[idx].squeeze()
    true_label = example_targets[idx].item()
    pred_label = predictions[idx].item()
    
    color = 'green' if true_label == pred_label else 'red'
    
    if i < 3:
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        ax.axis('off')

# Plot 4: Confusion visualization
axes[1, 1].text(0.5, 0.5, 
                f'Test Accuracy\n{test_acc:.2f}%\n\nTest Loss\n{test_loss:.4f}',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_4_projects/mnist_results.png', dpi=150)
print("\n✓ Results saved as 'mnist_results.png'")
print("\nClose the plot window to continue...")
plt.show()

# ============================================================================
# PART 9: Save Model
# ============================================================================
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

# Save model weights
torch.save(model.state_dict(), '/home/claude/pytorch_gradient_descent_tutorial/level_4_projects/mnist_model.pth')
print("\n✓ Model saved as 'mnist_model.pth'")

print("\nTo load the model later:")
print("  model = MNISTNet()")
print("  model.load_state_dict(torch.load('mnist_model.pth'))")
print("  model.eval()")

# ============================================================================
# PART 10: Key Takeaways
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. COMPLETE ML PIPELINE:
   ✓ Data loading and preprocessing
   ✓ Train/validation/test splits
   ✓ Model definition
   ✓ Training loop with proper evaluation
   ✓ Visualization and model saving

2. BEST PRACTICES:
   • Always use validation set for hyperparameter tuning
   • Never touch test set until final evaluation
   • Use dropout for regularization
   • Implement learning rate scheduling
   • Save models for later use

3. GRADIENT DESCENT IN ACTION:
   • Adam optimizer converges quickly
   • Batch training with DataLoader
   • Automatic differentiation handles complex network
   • All concepts from Levels 1-3 applied here!

4. ACHIEVING GOOD PERFORMANCE:
   • >95% accuracy on MNIST is achievable
   • Proper architecture design matters
   • Hyperparameter tuning improves results
   • More training typically helps (to a point)

5. NEXT STEPS:
   • Try different architectures (more/fewer layers)
   • Experiment with hyperparameters
   • Add convolutional layers for better performance
   • Try other datasets (Fashion-MNIST, CIFAR-10)
""")

print("="*80)
print("🎉 CONGRATULATIONS!")
print("="*80)
print("""
You've completed a full neural network project!

You now know how to:
✓ Load and preprocess real datasets
✓ Build neural networks with PyTorch
✓ Train models using gradient descent
✓ Evaluate and visualize results
✓ Save and load trained models

This is the foundation for deep learning!
""")
print("="*80)
