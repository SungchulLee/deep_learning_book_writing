"""
15_deep_network.py - Building Very Deep Networks

Learn to build and train very deep feedforward networks using:
- Residual connections (skip connections)
- Careful initialization
- Gradient clipping
- Advanced techniques

Going deep: More layers = more capacity, but harder to train!

TIME: 45-60 minutes | DIFFICULTY: ⭐⭐⭐⭐⭐
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

print("="*80)
print("Building Very Deep Networks")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

print("\n" + "="*80)
print("Architecture Design")
print("="*80)

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    Output = ReLU(Block(x) + x)
    """
    
    def __init__(self, size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection!
        out = self.relu(out)
        return out

class DeepNet(nn.Module):
    """Very deep network with residual connections."""
    
    def __init__(self, input_size=784, hidden_size=256, num_blocks=10, num_classes=10):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # Stack of residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_proj(x)
        
        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.output(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Create models with different depths
models = {
    'Shallow (3 blocks)': DeepNet(num_blocks=3),
    'Medium (7 blocks)': DeepNet(num_blocks=7),
    'Deep (15 blocks)': DeepNet(num_blocks=15)
}

print("Model Comparison:")
print("-"*80)
for name, model in models.items():
    params = model.count_parameters()
    layers = len(list(model.modules()))
    print(f"{name:20s} | Parameters: {params:,} | Modules: {layers}")

print("\n" + "="*80)
print("Training Deep Network")
print("="*80)

# Use the deep model
model = models['Deep (15 blocks)'].to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Learning rate scheduler with warmup
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Gradient clipping for stability
max_grad_norm = 1.0

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# Training
epochs = 20
train_losses, train_accs, test_accs = [], [], []

print(f"Training deep network ({model.count_parameters():,} parameters)...")
print("-"*80)

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader)
    test_acc = evaluate(model, test_loader)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    scheduler.step()
    
    print(f"Epoch [{epoch+1:2d}/{epochs}] | "
          f"Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Test Acc: {test_acc:.2f}%")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, linewidth=2, label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss (Deep Network)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, linewidth=2, label='Train Accuracy')
ax2.plot(test_accs, linewidth=2, label='Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Progress', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('15_deep_network_results.png', dpi=150)
print("\nResults saved!")

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
BUILDING DEEP NETWORKS:
✓ Use residual connections (skip connections)
✓ Batch normalization after linear layers
✓ Proper weight initialization (Kaiming for ReLU)
✓ Gradient clipping to prevent explosion
✓ Learning rate scheduling

RESIDUAL CONNECTIONS:
  H(x) = F(x) + x
  - Easier to optimize (gradients flow directly)
  - Enable training of 100+ layer networks
  - Used in ResNet, DenseNet, Transformers

TRAINING STABILITY:
- Gradient clipping: Limit gradient magnitude
- Batch normalization: Stabilize activations
- Skip connections: Direct gradient flow
- Proper initialization: Good starting point

DEPTH vs WIDTH:
- Deeper: More hierarchical features
- Wider: More capacity per layer
- Trade-off depends on problem

CHALLENGES WITH DEPTH:
⚠ Vanishing/exploding gradients
⚠ Degradation (accuracy saturates)
⚠ More memory and compute
⚠ Longer training time

SOLUTIONS:
✓ Residual connections
✓ Normalization layers
✓ Careful initialization
✓ Gradient clipping
✓ Learning rate warmup

CONGRATULATIONS!
You've completed the full tutorial! 🎉
You now know how to build production-ready neural networks!
""")
plt.show()
