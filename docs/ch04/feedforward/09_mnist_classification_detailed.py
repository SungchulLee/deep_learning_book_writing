"""
==============================================================================
Tutorial 06: MNIST Digit Classification
==============================================================================
DIFFICULTY: ⭐⭐ Intermediate

WHAT YOU'LL LEARN:
- Working with real-world datasets
- Data loading with DataLoader
- Training and validation splits
- Batch processing
- Model evaluation metrics

PREREQUISITES:
- Tutorial 05 (nn.Module and optimizers)

KEY CONCEPTS:
- torchvision.datasets
- torch.utils.data.DataLoader
- Batch training
- Train/test split
- Accuracy computation
==============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

torch.manual_seed(42)

# ==============================================================================
# INTRODUCTION: Real-World Dataset
# ==============================================================================
print("=" * 70)
print("Welcome to MNIST Classification!")
print("=" * 70)
print("\nMNIST Dataset:")
print("  - 70,000 handwritten digit images (0-9)")
print("  - 28x28 grayscale images")
print("  - Classic machine learning benchmark")
print("  - Real-world computer vision task!")

# ==============================================================================
# STEP 1: Load and Explore MNIST Dataset
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 1: Loading MNIST Dataset")
print("=" * 70)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Transform: Convert PIL Image to tensor
# ToTensor() automatically normalizes to [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download and load training dataset
print("Downloading MNIST dataset (if not already present)...")
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# Load test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

print(f"\nDataset loaded successfully!")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

# Explore a single sample
sample_image, sample_label = train_dataset[0]
print(f"\nSample exploration:")
print(f"  Image shape: {sample_image.shape}")  # [channels, height, width]
print(f"  Label: {sample_label}")
print(f"  Image value range: [{sample_image.min():.2f}, {sample_image.max():.2f}]")

# ==============================================================================
# STEP 2: Create Data Loaders
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Creating Data Loaders")
print("=" * 70)

print("\nWhat is a DataLoader?")
print("  - Batches data for efficient training")
print("  - Shuffles data each epoch")
print("  - Handles parallel data loading")
print("  - Essential for large datasets!")

batch_size = 64

# Create data loaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,  # Shuffle for better training
    num_workers=2  # Parallel data loading
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False  # No need to shuffle test data
)

print(f"\nData loaders created:")
print(f"  Batch size: {batch_size}")
print(f"  Training batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# Example: Iterate through one batch
images, labels = next(iter(train_loader))
print(f"\nExample batch:")
print(f"  Images shape: {images.shape}")  # [batch_size, channels, height, width]
print(f"  Labels shape: {labels.shape}")  # [batch_size]

# ==============================================================================
# STEP 3: Visualize Sample Images
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Visualizing Sample Images")
print("=" * 70)

# Get a batch for visualization
examples = iter(test_loader)
example_images, example_labels = next(examples)

# Plot 12 samples
fig, axes = plt.subplots(2, 6, figsize=(12, 4))
axes = axes.ravel()

for i in range(12):
    axes[i].imshow(example_images[i].squeeze(), cmap='gray')
    axes[i].set_title(f'Label: {example_labels[i].item()}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/06_mnist_samples.png', dpi=100)
print("Sample images saved as '06_mnist_samples.png'")

# ==============================================================================
# STEP 4: Define Neural Network
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Defining the Neural Network")
print("=" * 70)

class MNISTNet(nn.Module):
    """
    Feedforward Neural Network for MNIST Classification
    
    Architecture:
      Input (784) -> Hidden1 (128) -> Hidden2 (64) -> Output (10)
    
    Why this architecture?
      - 784 inputs: 28x28 pixels flattened
      - 128, 64: Hidden layers for learning features
      - 10 outputs: One per digit (0-9)
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            
            # Second hidden layer
            nn.Linear(128, 64),
            nn.ReLU(),
            
            # Output layer (no activation - we'll use CrossEntropyLoss)
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        # Flatten images: [batch, 1, 28, 28] -> [batch, 784]
        x = x.view(x.size(0), -1)
        return self.network(x)

# Create model
model = MNISTNet().to(device)
print("Model architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# ==============================================================================
# STEP 5: Define Loss and Optimizer
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Loss Function and Optimizer")
print("=" * 70)

# CrossEntropyLoss for multi-class classification
# Combines LogSoftmax and NLLLoss
criterion = nn.CrossEntropyLoss()
print("Loss function: CrossEntropyLoss")
print("  - Perfect for multi-class classification")
print("  - Expects raw logits (no softmax needed in model)")
print("  - Numerically stable")

# Adam optimizer
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f"\nOptimizer: Adam (lr={learning_rate})")

# ==============================================================================
# STEP 6: Training Loop
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Training the Model")
print("=" * 70)

n_epochs = 5
train_losses = []
train_accuracies = []

print(f"Training for {n_epochs} epochs...\n")

for epoch in range(n_epochs):
    model.train()  # Set model to training mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress tracking
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], '
                  f'Step [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    # Epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    print(f'\nEpoch [{epoch+1}/{n_epochs}] Summary:')
    print(f'  Average Loss: {epoch_loss:.4f}')
    print(f'  Training Accuracy: {epoch_acc:.2f}%\n')

# ==============================================================================
# STEP 7: Evaluate on Test Set
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Testing the Model")
print("=" * 70)

model.eval()  # Set model to evaluation mode

test_correct = 0
test_total = 0
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

test_accuracy = 100 * test_correct / test_total
print(f'\nOverall Test Accuracy: {test_accuracy:.2f}%')

print('\nPer-class accuracy:')
for i in range(10):
    acc = 100 * class_correct[i] / class_total[i]
    print(f'  Digit {i}: {acc:.2f}%')

# ==============================================================================
# STEP 8: Visualize Predictions
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 8: Visualizing Predictions")
print("=" * 70)

# Get some test images
model.eval()
with torch.no_grad():
    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)
    outputs = model(test_images)
    _, predictions = torch.max(outputs, 1)

# Plot predictions
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.ravel()

for i in range(12):
    img = test_images[i].cpu().squeeze()
    true_label = test_labels[i].item()
    pred_label = predictions[i].item()
    
    axes[i].imshow(img, cmap='gray')
    color = 'green' if true_label == pred_label else 'red'
    axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/06_predictions.png', dpi=100)
print("Predictions saved as '06_predictions.png'")

# ==============================================================================
# STEP 9: Training Visualization
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 9: Training Progress Visualization")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot training loss
ax1.plot(range(1, n_epochs + 1), train_losses, 'b-o', linewidth=2, markersize=8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

# Plot training accuracy
ax2.plot(range(1, n_epochs + 1), train_accuracies, 'g-o', linewidth=2, markersize=8)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Accuracy')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/06_training_progress.png', dpi=100)
print("Training progress saved as '06_training_progress.png'")

# ==============================================================================
# STEP 10: Save Model
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 10: Saving the Model")
print("=" * 70)

model_path = '/home/claude/pytorch_feedforward_tutorial/mnist_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# ==============================================================================
# KEY TAKEAWAYS:
# ==============================================================================
print("\n" + "=" * 70)
print("Key Takeaways")
print("=" * 70)
print("""
1. Real-world data workflow:
   a) Load dataset (torchvision.datasets)
   b) Create data loaders (batching, shuffling)
   c) Define model architecture
   d) Train with batches
   e) Evaluate on test set

2. DataLoader benefits:
   - Automatic batching
   - Data shuffling
   - Parallel loading
   - Memory efficiency

3. CrossEntropyLoss for classification:
   - Combines LogSoftmax + NLLLoss
   - Expects raw logits (no softmax in model)
   - Numerically stable

4. Training best practices:
   - model.train() before training
   - model.eval() before evaluation
   - torch.no_grad() during inference
   - Track both loss and accuracy

5. Typical accuracy for MNIST:
   - Simple feedforward: 95-97%
   - CNN (covered later): 99%+
   - Our model: ~{test_accuracy:.1f}%

NEXT STEPS:
- Tutorial 07: Add validation set and regularization
- Tutorial 08: Batch normalization
- Tutorial 09: Deeper networks
- Tutorial 10: Advanced techniques
""")

print("Training completed successfully! ✓")
# ==============================================================================
