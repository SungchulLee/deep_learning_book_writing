"""
==============================================================================
Tutorial 07: Regularization Techniques
==============================================================================
DIFFICULTY: ⭐⭐⭐ Intermediate-Advanced

WHAT YOU'LL LEARN:
- Overfitting and underfitting
- Dropout regularization
- L2 regularization (weight decay)
- Validation set usage
- Early stopping

PREREQUISITES:
- Tutorial 06 (MNIST classification)

KEY CONCEPTS:
- Overfitting prevention
- nn.Dropout
- Weight decay
- Model validation
- Training monitoring
==============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np

torch.manual_seed(42)

# ==============================================================================
# INTRODUCTION: The Overfitting Problem
# ==============================================================================
print("=" * 70)
print("Understanding Overfitting and Regularization")
print("=" * 70)
print("""
What is Overfitting?
  - Model learns training data TOO well
  - Performs great on training set
  - Performs poorly on unseen data (test set)
  - Memorizes rather than generalizes

How to Prevent Overfitting?
  1. More training data
  2. Simpler model (fewer parameters)
  3. Regularization techniques:
     - Dropout
     - L2 regularization (weight decay)
     - Data augmentation
  4. Early stopping
""")

# ==============================================================================
# STEP 1: Prepare Data with Validation Split
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 1: Creating Train/Validation/Test Split")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load MNIST
transform = transforms.Compose([transforms.ToTensor()])

train_val_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

# Split training data into train and validation (80/20)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(
    train_val_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Dataset split:")
print(f"  Training:   {len(train_dataset):,} samples (80%)")
print(f"  Validation: {len(val_dataset):,} samples (20%)")
print(f"  Test:       {len(test_dataset):,} samples")

print("\nWhy use validation set?")
print("  - Monitor overfitting during training")
print("  - Tune hyperparameters")
print("  - Test set remains untouched until final evaluation")

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==============================================================================
# STEP 2: Define Models (With and Without Regularization)
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Defining Models")
print("=" * 70)

class SimpleNet(nn.Module):
    """
    Simple network WITHOUT regularization
    Prone to overfitting!
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

class RegularizedNet(nn.Module):
    """
    Network WITH regularization techniques
    
    Regularization methods used:
      1. Dropout: Randomly zeros some neurons during training
      2. Weight decay: Added via optimizer (L2 regularization)
    """
    def __init__(self, dropout_rate=0.5):
        super(RegularizedNet, self).__init__()
        
        self.network = nn.Sequential(
            # First layer
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout after activation
            
            # Second layer
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer (no dropout here)
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

print("Two models defined:")
print("  1. SimpleNet: No regularization")
print("  2. RegularizedNet: Dropout + Weight decay")
print(f"\nDropout explanation:")
print("  - Randomly sets neurons to 0 during training")
print("  - Forces network to learn robust features")
print("  - Prevents co-adaptation of neurons")
print("  - Automatically disabled during eval mode")

# ==============================================================================
# STEP 3: Training Function
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Defining Training and Evaluation Functions")
print("=" * 70)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()  # Enable dropout
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset"""
    model.eval()  # Disable dropout
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

print("Functions defined:")
print("  - train_epoch(): Trains for one epoch")
print("  - evaluate(): Evaluates model (no gradient computation)")

# ==============================================================================
# STEP 4: Train Both Models
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Training Both Models")
print("=" * 70)

n_epochs = 15
learning_rate = 0.001

# Model 1: Without regularization
print("\n" + "-" * 70)
print("Training Model 1: SimpleNet (No Regularization)")
print("-" * 70)

model1 = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
# No weight decay for model 1

history1 = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

for epoch in range(n_epochs):
    train_loss, train_acc = train_epoch(model1, train_loader, criterion, optimizer1, device)
    val_loss, val_acc = evaluate(model1, val_loader, criterion, device)
    
    history1['train_loss'].append(train_loss)
    history1['train_acc'].append(train_acc)
    history1['val_loss'].append(val_loss)
    history1['val_acc'].append(val_acc)
    
    if (epoch + 1) % 3 == 0:
        print(f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Model 2: With regularization
print("\n" + "-" * 70)
print("Training Model 2: RegularizedNet (With Dropout + Weight Decay)")
print("-" * 70)

model2 = RegularizedNet(dropout_rate=0.3).to(device)
# Add weight decay (L2 regularization)
optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=1e-4)

print(f"Regularization parameters:")
print(f"  - Dropout rate: 0.3 (30% of neurons dropped)")
print(f"  - Weight decay: 1e-4 (L2 penalty)\n")

history2 = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

for epoch in range(n_epochs):
    train_loss, train_acc = train_epoch(model2, train_loader, criterion, optimizer2, device)
    val_loss, val_acc = evaluate(model2, val_loader, criterion, device)
    
    history2['train_loss'].append(train_loss)
    history2['train_acc'].append(train_acc)
    history2['val_loss'].append(val_loss)
    history2['val_acc'].append(val_acc)
    
    if (epoch + 1) % 3 == 0:
        print(f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# ==============================================================================
# STEP 5: Compare Results
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Comparing Models on Test Set")
print("=" * 70)

# Evaluate both models on test set
test_loss1, test_acc1 = evaluate(model1, test_loader, criterion, device)
test_loss2, test_acc2 = evaluate(model2, test_loader, criterion, device)

print("\nFinal Test Results:")
print("-" * 70)
print(f"Model 1 (No Regularization):")
print(f"  Test Accuracy: {test_acc1:.2f}%")
print(f"  Test Loss: {test_loss1:.4f}")
print(f"\nModel 2 (With Regularization):")
print(f"  Test Accuracy: {test_acc2:.2f}%")
print(f"  Test Loss: {test_loss2:.4f}")

# Calculate overfitting metric (train-val gap)
overfit1 = history1['train_acc'][-1] - history1['val_acc'][-1]
overfit2 = history2['train_acc'][-1] - history2['val_acc'][-1]
print(f"\nOverfitting Analysis (Train-Val Accuracy Gap):")
print(f"  Model 1: {overfit1:.2f}% gap")
print(f"  Model 2: {overfit2:.2f}% gap")
print(f"  {'Model 2 has less overfitting! ✓' if overfit2 < overfit1 else 'Unexpected result'}")

# ==============================================================================
# STEP 6: Visualization
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Visualizing Training Progress")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

epochs = range(1, n_epochs + 1)

# Model 1: Loss
axes[0, 0].plot(epochs, history1['train_loss'], 'b-o', label='Train Loss', alpha=0.7)
axes[0, 0].plot(epochs, history1['val_loss'], 'r-s', label='Val Loss', alpha=0.7)
axes[0, 0].set_title('Model 1 (No Regularization): Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Model 1: Accuracy
axes[0, 1].plot(epochs, history1['train_acc'], 'b-o', label='Train Acc', alpha=0.7)
axes[0, 1].plot(epochs, history1['val_acc'], 'r-s', label='Val Acc', alpha=0.7)
axes[0, 1].set_title('Model 1 (No Regularization): Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Model 2: Loss
axes[1, 0].plot(epochs, history2['train_loss'], 'b-o', label='Train Loss', alpha=0.7)
axes[1, 0].plot(epochs, history2['val_loss'], 'r-s', label='Val Loss', alpha=0.7)
axes[1, 0].set_title('Model 2 (With Regularization): Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Model 2: Accuracy
axes[1, 1].plot(epochs, history2['train_acc'], 'b-o', label='Train Acc', alpha=0.7)
axes[1, 1].plot(epochs, history2['val_acc'], 'r-s', label='Val Acc', alpha=0.7)
axes[1, 1].set_title('Model 2 (With Regularization): Accuracy')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/07_regularization_comparison.png', dpi=100)
print("Comparison saved as '07_regularization_comparison.png'")

# ==============================================================================
# KEY TAKEAWAYS:
# ==============================================================================
print("\n" + "=" * 70)
print("Key Takeaways")
print("=" * 70)
print("""
1. Signs of Overfitting:
   - Large gap between training and validation accuracy
   - Training accuracy keeps improving, validation plateaus
   - Model performs poorly on unseen data

2. Dropout (nn.Dropout):
   - Randomly zeros neurons during training
   - dropout_rate: Probability of dropping (typical: 0.2-0.5)
   - Automatically disabled with model.eval()
   - Prevents co-adaptation of neurons

3. Weight Decay (L2 Regularization):
   - Adds penalty for large weights: Loss = Data_Loss + λ * Σ(weights²)
   - Set via optimizer: weight_decay=1e-4
   - Encourages smaller weights
   - Improves generalization

4. Validation Set:
   - Essential for monitoring overfitting
   - Used for hyperparameter tuning
   - Not used for gradient updates
   - Separate from test set

5. Other regularization techniques (not covered here):
   - Data augmentation
   - Batch normalization
   - Early stopping
   - L1 regularization

6. Best Practices:
   - Always use validation set
   - Monitor train-val gap
   - Start with light regularization, increase if needed
   - Use dropout after ReLU, not before

NEXT STEPS:
- Tutorial 08: Batch Normalization
- Tutorial 09: Learning Rate Scheduling
- Tutorial 10: Advanced Architectures
""")

print("\nTraining completed successfully! ✓")
# ==============================================================================
