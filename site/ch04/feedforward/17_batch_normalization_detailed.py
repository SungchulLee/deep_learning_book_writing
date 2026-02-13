"""
==============================================================================
Tutorial 08: Batch Normalization
==============================================================================
DIFFICULTY: ⭐⭐⭐ Advanced

WHAT YOU'LL LEARN:
- Internal covariate shift problem
- Batch normalization layer
- How BatchNorm improves training
- When and where to use BatchNorm

PREREQUISITES:
- Tutorial 07 (regularization techniques)

KEY CONCEPTS:
- nn.BatchNorm1d
- Training stability
- Faster convergence
- Batch vs layer normalization
==============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import time

torch.manual_seed(42)

# ==============================================================================
# INTRODUCTION: The Internal Covariate Shift Problem
# ==============================================================================
print("=" * 70)
print("Understanding Batch Normalization")
print("=" * 70)
print("""
What Problem Does BatchNorm Solve?

Internal Covariate Shift:
  - As network trains, layer inputs change distribution
  - Each layer must adapt to these changes
  - Slows down training
  - Requires careful weight initialization
  - Sensitive to learning rate

Batch Normalization Solution:
  - Normalizes layer inputs for each mini-batch
  - Reduces internal covariate shift
  - Allows higher learning rates
  - Reduces sensitivity to initialization
  - Acts as regularization (like dropout)

Formula (for each feature in batch):
  1. μ = mean(batch)
  2. σ² = variance(batch)
  3. x̂ = (x - μ) / √(σ² + ε)     # Normalize
  4. y = γ * x̂ + β                # Scale and shift

Where γ (gamma) and β (beta) are learnable parameters!
""")

# ==============================================================================
# STEP 1: Load Data
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 1: Loading MNIST Data")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

transform = transforms.Compose([transforms.ToTensor()])

train_val_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

# 80/20 train/val split
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(
    train_val_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

batch_size = 128  # Larger batch size for stable BatchNorm statistics
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Dataset loaded (batch size: {batch_size})")

# ==============================================================================
# STEP 2: Define Models
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Defining Models (With and Without BatchNorm)")
print("=" * 70)

class NetWithoutBN(nn.Module):
    """
    Deep network WITHOUT Batch Normalization
    May be harder to train, especially with deeper architectures
    """
    def __init__(self):
        super(NetWithoutBN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

class NetWithBN(nn.Module):
    """
    Deep network WITH Batch Normalization
    
    BatchNorm placement:
    - Typically after linear layer, before activation
    - Some research suggests after activation also works
    - We use the standard: Linear -> BatchNorm -> Activation
    """
    def __init__(self):
        super(NetWithBN, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(28*28, 256),
            nn.BatchNorm1d(256),  # BatchNorm after linear, before ReLU
            nn.ReLU(),
            
            # Layer 2
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Layer 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Output layer (no BatchNorm here)
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

print("Two models defined:")
print("  1. NetWithoutBN: Standard network")
print("  2. NetWithBN: Network with BatchNorm layers")
print("\nBatchNorm1d:")
print("  - For fully connected layers (1D data)")
print("  - BatchNorm2d exists for CNNs (2D data)")
print("  - BatchNorm3d for 3D data (videos, medical images)")

# ==============================================================================
# STEP 3: Understanding BatchNorm Behavior
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Understanding BatchNorm Behavior")
print("=" * 70)

# Create a simple BatchNorm layer to demonstrate
demo_bn = nn.BatchNorm1d(3)
print("\nBatchNorm1d layer parameters:")
print(f"  gamma (weight): {demo_bn.weight.data}")
print(f"  beta (bias):    {demo_bn.bias.data}")
print(f"  running_mean:   {demo_bn.running_mean}")
print(f"  running_var:    {demo_bn.running_var}")

print("\nKey points:")
print("  - gamma and beta are learnable (updated by optimizer)")
print("  - running_mean and running_var are NOT learnable")
print("  - During training: uses batch statistics")
print("  - During evaluation: uses running statistics")
print("  - This is why model.train() and model.eval() matter!")

# ==============================================================================
# STEP 4: Training Function
# ==============================================================================

def train_and_evaluate(model, train_loader, val_loader, optimizer, 
                       criterion, n_epochs, device):
    """Train model and track metrics"""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'epoch_time': []
    }
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)
        
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:2d}/{n_epochs}: "
                  f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                  f"Time: {epoch_time:.2f}s")
    
    return history

# ==============================================================================
# STEP 5: Train Both Models
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Training Both Models")
print("=" * 70)

n_epochs = 15
learning_rate = 0.01  # Note: Higher LR works better with BatchNorm!

criterion = nn.CrossEntropyLoss()

# Model 1: Without BatchNorm
print("\n" + "-" * 70)
print("Training Model WITHOUT BatchNorm")
print("-" * 70)

model1 = NetWithoutBN().to(device)
optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate, momentum=0.9)
history1 = train_and_evaluate(model1, train_loader, val_loader, 
                               optimizer1, criterion, n_epochs, device)

# Model 2: With BatchNorm
print("\n" + "-" * 70)
print("Training Model WITH BatchNorm")
print("-" * 70)

model2 = NetWithBN().to(device)
optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate, momentum=0.9)
history2 = train_and_evaluate(model2, train_loader, val_loader, 
                               optimizer2, criterion, n_epochs, device)

# ==============================================================================
# STEP 6: Compare Results
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Comparison")
print("=" * 70)

# Test set evaluation
model1.eval()
model2.eval()

def test_accuracy(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

test_acc1 = test_accuracy(model1, test_loader, device)
test_acc2 = test_accuracy(model2, test_loader, device)

print("\nFinal Results:")
print("-" * 70)
print(f"Without BatchNorm:")
print(f"  Best Val Accuracy: {max(history1['val_acc']):.2f}%")
print(f"  Test Accuracy: {test_acc1:.2f}%")
print(f"  Avg Time/Epoch: {sum(history1['epoch_time'])/len(history1['epoch_time']):.2f}s")

print(f"\nWith BatchNorm:")
print(f"  Best Val Accuracy: {max(history2['val_acc']):.2f}%")
print(f"  Test Accuracy: {test_acc2:.2f}%")
print(f"  Avg Time/Epoch: {sum(history2['epoch_time'])/len(history2['epoch_time']):.2f}s")

improvement = test_acc2 - test_acc1
print(f"\nImprovement with BatchNorm: {improvement:+.2f}% {'✓' if improvement > 0 else ''}")

# ==============================================================================
# STEP 7: Visualization
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Visualizing Training Progress")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
epochs = range(1, n_epochs + 1)

# Training Loss
axes[0, 0].plot(epochs, history1['train_loss'], 'b-', label='Without BN', linewidth=2)
axes[0, 0].plot(epochs, history2['train_loss'], 'r-', label='With BN', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Validation Loss
axes[0, 1].plot(epochs, history1['val_loss'], 'b-', label='Without BN', linewidth=2)
axes[0, 1].plot(epochs, history2['val_loss'], 'r-', label='With BN', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Validation Loss Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Training Accuracy
axes[1, 0].plot(epochs, history1['train_acc'], 'b-', label='Without BN', linewidth=2)
axes[1, 0].plot(epochs, history2['train_acc'], 'r-', label='With BN', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].set_title('Training Accuracy Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Validation Accuracy
axes[1, 1].plot(epochs, history1['val_acc'], 'b-', label='Without BN', linewidth=2)
axes[1, 1].plot(epochs, history2['val_acc'], 'r-', label='With BN', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].set_title('Validation Accuracy Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/08_batchnorm_comparison.png', dpi=100)
print("Comparison saved as '08_batchnorm_comparison.png'")

# ==============================================================================
# KEY TAKEAWAYS:
# ==============================================================================
print("\n" + "=" * 70)
print("Key Takeaways")
print("=" * 70)
print("""
1. What BatchNorm Does:
   - Normalizes layer inputs per mini-batch
   - Learns optimal scale (γ) and shift (β)
   - Maintains running statistics for inference

2. Benefits:
   ✓ Faster convergence (can use higher learning rates)
   ✓ Reduces sensitivity to initialization
   ✓ Acts as regularization (slight generalization improvement)
   ✓ More stable training (especially for deep networks)

3. Where to Place BatchNorm:
   - Typical: Linear -> BatchNorm -> Activation
   - After linear layer, before activation
   - Not needed in output layer
   - Use BatchNorm1d for fully connected layers
   - Use BatchNorm2d for convolutional layers

4. Training vs Evaluation:
   - Training: Uses batch statistics (mean, var from current batch)
   - Evaluation: Uses running statistics (accumulated during training)
   - Always call model.train() and model.eval() appropriately!

5. BatchNorm Variants:
   - BatchNorm1d: For fully connected layers
   - BatchNorm2d: For 2D convolutional layers
   - BatchNorm3d: For 3D convolutional layers
   - LayerNorm: Alternative (normalizes across features, not batch)
   - GroupNorm: Works with small batch sizes

6. When NOT to Use BatchNorm:
   - Very small batch sizes (statistics unreliable)
   - Recurrent networks (use LayerNorm instead)
   - When batch independence is important

7. Common Pitfalls:
   ✗ Forgetting to call model.eval() during inference
   ✗ Using batch size = 1 (statistics don't make sense)
   ✗ Mixing up training and evaluation modes

NEXT STEPS:
- Tutorial 09: Learning Rate Scheduling
- Tutorial 10: Advanced Architectures
- Tutorial 11: CIFAR-10 Challenge
""")

print("\nTraining completed successfully! ✓")
# ==============================================================================
