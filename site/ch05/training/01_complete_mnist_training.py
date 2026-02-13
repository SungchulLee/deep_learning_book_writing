"""
================================================================================
REAL-WORLD EXAMPLE: Complete MNIST Digit Classification
================================================================================

WHAT YOU'LL LEARN:
- Complete end-to-end training pipeline
- Data loading and preprocessing
- Model definition
- Training with loss and optimizer
- Validation and testing
- Saving and loading models
- Best practices for production code

PREREQUISITES:
- Complete beginner tutorials
- Basic understanding of CNNs

TIME TO COMPLETE: ~30 minutes
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

print("=" * 80)
print("COMPLETE MNIST DIGIT CLASSIFICATION")
print("=" * 80)

# ============================================================================
# SECTION 1: Configuration and Setup
# ============================================================================
print("\n" + "-" * 80)
print("CONFIGURATION")
print("-" * 80)

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
config = {
    'batch_size': 64,
    'test_batch_size': 1000,
    'epochs': 5,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'log_interval': 100,  # Print every N batches
    'save_model': True,
    'model_path': '/home/claude/mnist_model.pt'
}

print("\nHyperparameters:")
for key, value in config.items():
    print(f"  {key}: {value}")

# ============================================================================
# SECTION 2: Data Loading and Preprocessing
# ============================================================================
print("\n" + "-" * 80)
print("DATA LOADING")
print("-" * 80)

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean/std
])

print("Downloading MNIST dataset...")

# Download and load training data
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

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,  # Shuffle training data
    num_workers=0  # Number of processes for data loading
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config['test_batch_size'],
    shuffle=False,  # Don't shuffle test data
    num_workers=0
)

print(f"\nDataset Statistics:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")
print(f"  Number of classes: 10 (digits 0-9)")
print(f"  Image size: 28x28 pixels")
print(f"  Training batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ============================================================================
# SECTION 3: Model Definition
# ============================================================================
print("\n" + "-" * 80)
print("MODEL ARCHITECTURE")
print("-" * 80)

class ConvNet(nn.Module):
    """
    Convolutional Neural Network for MNIST
    
    Architecture:
    - Conv Layer 1: 1 → 32 channels, 3x3 kernel
    - Conv Layer 2: 32 → 64 channels, 3x3 kernel
    - Max Pooling: 2x2
    - Fully Connected 1: 9216 → 128
    - Fully Connected 2: 128 → 10 (classes)
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # After conv1, conv2, and 2 poolings: 28 → 14 → 7
        # Feature map size: 7 * 7 * 64 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))  # 28x28x32
        x = self.pool(x)            # 14x14x32
        
        # Conv block 2
        x = F.relu(self.conv2(x))  # 14x14x64
        x = self.pool(x)            # 7x7x64
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)  # Flatten to vector
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)  # No activation (CrossEntropyLoss applies softmax)
        
        return x

# Create model and move to device
model = ConvNet().to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model Architecture:")
print(model)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# SECTION 4: Loss Function and Optimizer
# ============================================================================
print("\n" + "-" * 80)
print("LOSS FUNCTION AND OPTIMIZER")
print("-" * 80)

# Loss function
criterion = nn.CrossEntropyLoss()
print(f"Loss function: {criterion}")

# Optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=config['learning_rate'],
    momentum=config['momentum']
)
print(f"Optimizer: {optimizer}")

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1,  # Decay every epoch
    gamma=0.7     # Multiply LR by 0.7
)
print(f"LR Scheduler: StepLR(step_size=1, gamma=0.7)")

# ============================================================================
# SECTION 5: Training Function
# ============================================================================

def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch
    
    Args:
        model: Neural network model
        device: Device to train on (CPU/GPU)
        train_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        epoch: Current epoch number
    """
    model.train()  # Set model to training mode
    
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Print progress
        if batch_idx % config['log_interval'] == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100. * correct / total:.2f}%')
    
    # Epoch statistics
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    print(f'\n  Epoch {epoch} Summary:')
    print(f'    Avg Loss: {avg_loss:.4f}')
    print(f'    Accuracy: {accuracy:.2f}%')
    print(f'    Time: {epoch_time:.2f}s')
    
    return avg_loss, accuracy

# ============================================================================
# SECTION 6: Validation/Test Function
# ============================================================================

def test(model, device, test_loader, criterion):
    """
    Evaluate the model on test data
    
    Args:
        model: Neural network model
        device: Device to test on (CPU/GPU)
        test_loader: DataLoader for test data
        criterion: Loss function
    
    Returns:
        Average loss and accuracy
    """
    model.eval()  # Set model to evaluation mode
    
    test_loss = 0
    correct = 0
    total = 0
    
    # No gradient computation during testing
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            test_loss += criterion(output, target).item()
            
            # Get predictions
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # Calculate statistics
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'\n  Test Results:')
    print(f'    Avg Loss: {avg_loss:.4f}')
    print(f'    Accuracy: {accuracy:.2f}% ({correct}/{total})')
    
    return avg_loss, accuracy

# ============================================================================
# SECTION 7: Training Loop
# ============================================================================
print("\n" + "-" * 80)
print("TRAINING")
print("-" * 80)

# Track history
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

print(f"\nTraining for {config['epochs']} epochs...\n")

for epoch in range(1, config['epochs'] + 1):
    print(f"{'=' * 80}")
    print(f"Epoch {epoch}/{config['epochs']}")
    print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"{'=' * 80}")
    
    # Train
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Test
    test_loss, test_acc = test(model, device, test_loader, criterion)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    # Update learning rate
    scheduler.step()
    
    print()

# ============================================================================
# SECTION 8: Save Model
# ============================================================================
print("\n" + "-" * 80)
print("SAVING MODEL")
print("-" * 80)

if config['save_model']:
    # Save complete model
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1],
        'test_accuracy': test_accuracies[-1]
    }, config['model_path'])
    
    print(f"Model saved to: {config['model_path']}")
    
    # Also save just the model weights (smaller file)
    weights_path = config['model_path'].replace('.pt', '_weights.pt')
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to: {weights_path}")

# ============================================================================
# SECTION 9: Load Model and Test
# ============================================================================
print("\n" + "-" * 80)
print("LOADING MODEL")
print("-" * 80)

# Create new model instance
loaded_model = ConvNet().to(device)

# Load checkpoint
checkpoint = torch.load(config['model_path'])
loaded_model.load_state_dict(checkpoint['model_state_dict'])

print("Model loaded successfully!")
print(f"  Trained for: {checkpoint['epoch']} epochs")
print(f"  Final test accuracy: {checkpoint['test_accuracy']:.2f}%")

# Test loaded model
print("\nVerifying loaded model:")
test_loss, test_acc = test(loaded_model, device, test_loader, criterion)

# ============================================================================
# SECTION 10: Inference Example
# ============================================================================
print("\n" + "-" * 80)
print("INFERENCE EXAMPLE")
print("-" * 80)

# Get a batch of test images
data_iter = iter(test_loader)
images, labels = next(data_iter)

# Take first 5 images
images = images[:5].to(device)
labels = labels[:5]

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(images)
    probabilities = F.softmax(outputs, dim=1)
    _, predictions = torch.max(outputs, 1)

print("Sample Predictions:")
print(f"{'True':^6} {'Pred':^6} {'Confidence':^12}")
print("-" * 26)

for i in range(5):
    true_label = labels[i].item()
    pred_label = predictions[i].item()
    confidence = probabilities[i, pred_label].item()
    status = "✓" if true_label == pred_label else "✗"
    
    print(f"  {true_label}      {pred_label}      {confidence*100:5.1f}%    {status}")

# ============================================================================
# SECTION 11: Training Summary
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)

print(f"\nFinal Results:")
print(f"  Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"  Test Accuracy: {test_accuracies[-1]:.2f}%")
print(f"  Training Loss: {train_losses[-1]:.4f}")
print(f"  Test Loss: {test_losses[-1]:.4f}")

print(f"\nTraining Progress:")
for epoch in range(config['epochs']):
    print(f"  Epoch {epoch+1}: "
          f"Train Acc={train_accuracies[epoch]:.2f}%, "
          f"Test Acc={test_accuracies[epoch]:.2f}%, "
          f"Test Loss={test_losses[epoch]:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. COMPLETE PIPELINE:
   ✓ Data loading and preprocessing
   ✓ Model definition
   ✓ Loss function and optimizer setup
   ✓ Training loop with validation
   ✓ Model saving and loading
   ✓ Inference

2. BEST PRACTICES DEMONSTRATED:
   ✓ Use DataLoader for efficient batching
   ✓ Set model.train() / model.eval() appropriately
   ✓ Use torch.no_grad() for inference
   ✓ Track metrics during training
   ✓ Save checkpoints with metadata
   ✓ Use learning rate scheduling
   ✓ Add dropout for regularization

3. PRODUCTION CONSIDERATIONS:
   ✓ Configuration management
   ✓ Reproducibility (random seeds)
   ✓ Device handling (CPU/GPU)
   ✓ Progress logging
   ✓ Error handling (not shown but important)
   ✓ Version control for models

4. OPTIMIZATION CHOICES:
   • SGD with momentum (reliable, well-tested)
   • CrossEntropyLoss (standard for classification)
   • StepLR scheduler (gradual learning rate decay)
   • Dropout (prevents overfitting)

NEXT STEPS:
→ Experiment with different architectures
→ Try other optimizers (Adam, AdamW)
→ Add data augmentation
→ Implement early stopping
→ Use tensorboard for visualization
→ Try on your own datasets!
""")
print("=" * 80)
