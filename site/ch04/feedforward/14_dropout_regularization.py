"""
08_dropout_regularization.py - Preventing Overfitting

Learn how to use dropout and other regularization techniques
to prevent your model from memorizing the training data.

OVERFITTING: Model performs well on training but poor on test data
SOLUTION: Regularization techniques that encourage generalization

TIME: 25-30 minutes | DIFFICULTY: ⭐⭐⭐☆☆
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

print("="*70)
print("Dropout and Regularization")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

# Model WITHOUT dropout (prone to overfitting)
class NoDropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Model WITH dropout (better generalization)
class DropoutNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout after activation
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def train_and_evaluate(model, name, epochs=10):
    """Train model and track train/test accuracy."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  # L2 regularization
    
    train_accs, test_accs = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()  # Enable dropout
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        
        train_acc = 100 * correct / total
        train_accs.append(train_acc)
        
        # Testing
        model.eval()  # Disable dropout
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        
        print(f"{name} - Epoch {epoch+1}: Train={train_acc:.1f}%, Test={test_acc:.1f}%")
    
    return train_accs, test_accs

print("\nTraining WITHOUT Dropout:")
no_drop_train, no_drop_test = train_and_evaluate(NoDropoutNet(), "No Dropout")

print("\nTraining WITH Dropout:")
drop_train, drop_test = train_and_evaluate(DropoutNet(0.5), "Dropout 0.5")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(no_drop_train, 'b-', label='No Dropout', linewidth=2)
ax1.plot(drop_train, 'r-', label='With Dropout', linewidth=2)
ax1.set_title('Training Accuracy', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(no_drop_test, 'b-', label='No Dropout', linewidth=2)
ax2.plot(drop_test, 'r-', label='With Dropout', linewidth=2)
ax2.set_title('Test Accuracy', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('08_dropout_comparison.png', dpi=150)
print("\nPlot saved as '08_dropout_comparison.png'")

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
DROPOUT:
  - Randomly drops neurons during training
  - Prevents co-adaptation of features
  - Acts as ensemble of many networks
  - Typical rates: 0.2-0.5 for hidden layers
  
USAGE:
  model.train()  # Enable dropout during training
  model.eval()   # Disable dropout during evaluation
  
OTHER REGULARIZATION:
  - L2 (weight_decay in optimizer): Penalizes large weights
  - L1: Encourages sparsity
  - Early stopping: Stop when validation loss increases
  - Data augmentation: Artificially expand dataset
  
WHEN TO USE:
  ✓ Large models on small datasets
  ✓ Model overfitting (train acc >> test acc)
  ✓ Deep networks
  
DROPOUT RATES:
  - Input layer: 0.1-0.2 (lower)
  - Hidden layers: 0.3-0.5 (higher)
  - Output layer: Never!
""")

print("\nEXERCISES:")
print("1. Try different dropout rates (0.1, 0.3, 0.7)")
print("2. Compare L1 vs L2 regularization")
print("3. Implement early stopping")
print("4. Add dropout to different positions in network")
plt.show()
