"""
================================================================================
04_using_sequential.py - Quick Model Building with nn.Sequential
================================================================================

nn.Sequential is PyTorch's way of building models quickly without writing
custom classes. Perfect for simple feed-forward architectures!

WHEN TO USE nn.Sequential:
    ✓ Simple feed-forward networks
    ✓ Rapid prototyping
    ✓ Linear layer stacks
    ✗ Complex branching architectures
    ✗ Multiple inputs/outputs
    ✗ Custom forward logic

LEARNING OBJECTIVES:
    1. Build models using nn.Sequential
    2. Understand the trade-offs vs custom nn.Module
    3. Learn to compose Sequential blocks
    4. Practice with different architectural patterns

DIFFICULTY: ⭐⭐☆☆☆ (Beginner-Intermediate)
TIME: 15-20 minutes
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ================================================================================
# PART 1: SIMPLE SEQUENTIAL MODEL
# ================================================================================
print("=" * 80)
print("PART 1: Building a Simple Sequential Model")
print("=" * 80)

# The simplest way to build a neural network in PyTorch!
# Each layer is applied sequentially in the order specified
simple_model = nn.Sequential(
    nn.Linear(784, 256),    # Input layer: 784 → 256
    nn.ReLU(),              # Activation
    nn.Linear(256, 128),    # Hidden layer: 256 → 128
    nn.ReLU(),              # Activation
    nn.Linear(128, 10)      # Output layer: 128 → 10
)

print("Simple Sequential Model:")
print(simple_model)
print(f"\nTotal parameters: {sum(p.numel() for p in simple_model.parameters()):,}")

# ================================================================================
# PART 2: SEQUENTIAL WITH NAMED LAYERS
# ================================================================================
print("\n" + "=" * 80)
print("PART 2: Sequential with Named Layers")
print("=" * 80)

# You can name layers for better debugging and understanding
# This is helpful when you want to access specific layers later
named_model = nn.Sequential(
    # Use OrderedDict or direct naming via key-value pairs
    ('flatten', nn.Flatten()),              # Flatten input
    ('fc1', nn.Linear(784, 256)),           # First fully connected
    ('relu1', nn.ReLU()),                   # First activation
    ('dropout1', nn.Dropout(0.2)),          # Dropout for regularization
    ('fc2', nn.Linear(256, 128)),           # Second fully connected
    ('relu2', nn.ReLU()),                   # Second activation
    ('dropout2', nn.Dropout(0.2)),          # More dropout
    ('fc3', nn.Linear(128, 10))             # Output layer
)

print("Named Sequential Model:")
for name, module in named_model.named_children():
    print(f"  {name}: {module}")

# ================================================================================
# PART 3: MODULAR SEQUENTIAL (COMPOSING BLOCKS)
# ================================================================================
print("\n" + "=" * 80)
print("PART 3: Composing Sequential Blocks")
print("=" * 80)

# You can create reusable building blocks!
def make_fc_block(in_features, out_features, dropout=0.2):
    """
    Create a fully connected block: Linear → ReLU → Dropout
    
    This is a common pattern - encapsulating it in a function
    makes your code cleaner and more maintainable.
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Dropout(dropout)
    )

# Build model from blocks
modular_model = nn.Sequential(
    nn.Flatten(),                          # Flatten 28×28 to 784
    make_fc_block(784, 512, dropout=0.3),  # Block 1
    make_fc_block(512, 256, dropout=0.3),  # Block 2
    make_fc_block(256, 128, dropout=0.2),  # Block 3
    nn.Linear(128, 10)                     # Output (no activation for logits)
)

print("Modular Sequential Model:")
print(modular_model)
print(f"\nNumber of layers: {len(list(modular_model.children()))}")

# ================================================================================
# PART 4: TRAIN ON MNIST
# ================================================================================
print("\n" + "=" * 80)
print("PART 4: Training Sequential Model on MNIST")
print("=" * 80)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load MNIST
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=2
)

# We'll use the simple model for training
model = simple_model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        # Flatten images from (batch, 1, 28, 28) to (batch, 784)
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

# Evaluation function
def evaluate(model, loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

# Training loop
print("\nTraining...")
num_epochs = 5
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

# ================================================================================
# PART 5: COMPARING ARCHITECTURES
# ================================================================================
print("\n" + "=" * 80)
print("PART 5: Architecture Comparison")
print("=" * 80)

# Let's compare our different models
models_dict = {
    'Simple (3 layers)': simple_model,
    'Named (with dropout)': named_model,
    'Modular (4 layers)': modular_model
}

print("\nModel Comparison:")
print("-" * 80)
print(f"{'Model':<25} {'Parameters':<15} {'Layers':<10}")
print("-" * 80)
for name, model in models_dict.items():
    params = sum(p.numel() for p in model.parameters())
    layers = len([m for m in model.modules() if not isinstance(m, nn.Sequential)])
    print(f"{name:<25} {params:<15,} {layers:<10}")
print("-" * 80)

# ================================================================================
# PART 6: ACCESSING AND MODIFYING SEQUENTIAL MODELS
# ================================================================================
print("\n" + "=" * 80)
print("PART 6: Accessing Sequential Model Components")
print("=" * 80)

# You can access layers by index
print("First layer of simple_model:")
print(simple_model[0])

print("\nThird layer (second ReLU):")
print(simple_model[3])

# You can iterate over layers
print("\nAll layers:")
for idx, layer in enumerate(simple_model):
    print(f"  Layer {idx}: {layer.__class__.__name__}")

# You can slice Sequential models
print("\nFirst 3 layers:")
feature_extractor = simple_model[:3]  # Gets first 3 layers
print(feature_extractor)

# You can modify Sequential models
print("\nModifying model by adding a new layer:")
extended_model = nn.Sequential(
    *simple_model,  # Unpack existing layers
    nn.ReLU(),      # Add another activation
    nn.Linear(10, 5)  # Add final layer
)
print(f"Original output size: 10")
print(f"Extended output size: 5")

# ================================================================================
# PART 7: VISUALIZATION
# ================================================================================
print("\n" + "=" * 80)
print("PART 7: Visualizing Training Progress")
print("=" * 80)

fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 5))

# Plot losses
ax1.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
ax1.plot(test_losses, 'r-', label='Test Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot accuracies
ax2.plot(train_accs, 'b-', label='Train Accuracy', linewidth=2)
ax2.plot(test_accs, 'r-', label='Test Accuracy', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([90, 100])

plt.tight_layout()
plt.savefig('04_sequential_training.png', dpi=150, bbox_inches='tight')
print("Training progress saved as '04_sequential_training.png'")
plt.show()

# ================================================================================
# KEY TAKEAWAYS
# ================================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. nn.Sequential is perfect for simple feed-forward architectures
   - Clean, readable code
   - Quick prototyping
   - Less boilerplate than custom nn.Module

2. Three ways to use Sequential:
   - Simple: Just pass layers in order
   - Named: Use tuples for better debugging
   - Modular: Compose reusable blocks

3. Limitations of Sequential:
   ✗ Can't handle multiple inputs/outputs
   ✗ No custom forward logic
   ✗ No conditional execution
   → Use custom nn.Module for these cases

4. Sequential models are fully compatible with:
   ✓ All PyTorch training APIs
   ✓ Model saving/loading
   ✓ Transfer learning
   ✓ Model inspection tools

5. You can access, slice, and modify Sequential models easily
   - Access by index: model[0]
   - Slice: model[:3]
   - Iterate: for layer in model

WHEN TO USE:
  - Use Sequential for simple stacks of layers
  - Use custom Module for complex architectures
""")

# ================================================================================
# EXERCISES FOR STUDENTS
# ================================================================================
print("=" * 80)
print("EXERCISES TO TRY")
print("=" * 80)
print("""
1. Create a deeper model with 5-6 layers using Sequential
2. Add batch normalization between layers
3. Experiment with different dropout rates
4. Build a "wide" network (more neurons per layer) vs "deep" (more layers)
5. Create a function that generates Sequential models from a configuration list
6. Extract intermediate features using slicing
7. Try different activation functions (LeakyReLU, ELU, etc.)
8. Build an ensemble of Sequential models
9. Visualize the weights of the first layer
10. Implement model pruning by removing layers
""")
