"""
12_cifar10_classifier.py - Full-Color Image Classification

Build a complete CIFAR-10 classifier with proper evaluation.
CIFAR-10: 60,000 32x32 color images in 10 classes.

CLASSES: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

This demonstrates a complete ML pipeline from data loading to deployment.

TIME: 40-50 minutes | DIFFICULTY: ⭐⭐⭐⭐☆
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

print("="*80)
print("CIFAR-10 Image Classification Pipeline")
print("="*80)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck')

print("\n" + "="*80)
print("STEP 1: Data Loading and Augmentation")
print("="*80)

# Data augmentation for training (improves generalization)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # Random crop with padding
    transforms.RandomHorizontalFlip(),         # 50% chance of flip
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),     # Normalize to [-1, 1]
                        (0.5, 0.5, 0.5))
])

# No augmentation for test (just normalize)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

# Create data loaders
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {batch_size}")

print("\n" + "="*80)
print("STEP 2: Model Architecture")
print("="*80)

class CIFAR10Net(nn.Module):
    """
    Deep feedforward network for CIFAR-10.
    
    Architecture: 5 layers with batch normalization and dropout.
    Input: 32x32x3 = 3072 features
    Output: 10 classes
    """
    
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        
        self.network = nn.Sequential(
            # Input: 3072 features (32*32*3)
            nn.Linear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 10)  # 10 classes
        )
    
    def forward(self, x):
        # Flatten image: (B, 3, 32, 32) → (B, 3072)
        x = x.view(x.size(0), -1)
        return self.network(x)

model = CIFAR10Net().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "="*80)
print("STEP 3: Training Setup")
print("="*80)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print(f"Loss: CrossEntropyLoss")
print(f"Optimizer: Adam (lr=0.001, weight_decay=1e-4)")
print(f"Scheduler: StepLR (step_size=20, gamma=0.5)")

print("\n" + "="*80)
print("STEP 4: Training")
print("="*80)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# Training loop
num_epochs = 50
best_acc = 0
train_losses, test_losses = [], []
train_accs, test_accs = [], []

print(f"Training for {num_epochs} epochs...")
print("-"*80)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    scheduler.step()
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_cifar10_model.pth')
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

print(f"\nBest Test Accuracy: {best_acc:.2f}%")

print("\n" + "="*80)
print("STEP 5: Evaluation and Analysis")
print("="*80)

# Load best model
model.load_state_dict(torch.load('best_cifar10_model.pth'))

# Final evaluation
_, final_acc, all_preds, all_labels = evaluate(model, test_loader, criterion)

# Per-class accuracy
class_correct = [0] * 10
class_total = [0] * 10
for pred, label in zip(all_preds, all_labels):
    class_correct[label] += (pred == label)
    class_total[label] += 1

print("\nPer-Class Accuracy:")
print("-"*60)
for i in range(10):
    acc = 100 * class_correct[i] / class_total[i]
    print(f"{classes[i]:10s}: {acc:5.2f}% ({class_correct[i]}/{class_total[i]})")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

print("\n" + "="*80)
print("STEP 6: Visualization")
print("="*80)

# Create visualizations
fig = plt.figure(figsize=(18, 5))

# Plot 1: Training curves
ax1 = plt.subplot(1, 3, 1)
ax1.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
ax1.plot(test_losses, 'r-', label='Test Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(1, 3, 2)
ax2.plot(train_accs, 'b-', label='Train Acc', linewidth=2)
ax2.plot(test_accs, 'r-', label='Test Acc', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Confusion matrix
ax3 = plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes, ax=ax3)
ax3.set_xlabel('Predicted')
ax3.set_ylabel('True')
ax3.set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('12_cifar10_results.png', dpi=150)
print("Results saved as '12_cifar10_results.png'")

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print(f"""
ACHIEVED: {final_acc:.2f}% accuracy on CIFAR-10
(State-of-the-art CNNs achieve ~99%)

COMPLETE PIPELINE DEMONSTRATED:
✓ Data augmentation for better generalization
✓ Proper train/test split
✓ Deep architecture with regularization
✓ Learning rate scheduling
✓ Model checkpointing (save best model)
✓ Comprehensive evaluation (per-class, confusion matrix)
✓ Visualization of results

PRODUCTION CONSIDERATIONS:
- Always use validation set for hyperparameter tuning
- Monitor multiple metrics, not just accuracy
- Save checkpoints regularly
- Analyze errors (confusion matrix)
- Test final model only once on test set
""")
plt.show()
