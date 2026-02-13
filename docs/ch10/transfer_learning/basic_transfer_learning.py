"""
Example 1: Basic Transfer Learning (Feature Extraction)
========================================================

This script demonstrates the fundamentals of transfer learning using PyTorch.
We'll use a pre-trained ResNet18 model and adapt it for CIFAR-10 classification.

Author: PyTorch Transfer Learning Tutorial
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import time
import copy

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================================================
# STEP 1: DEVICE CONFIGURATION
# ============================================================================
"""
First, we check if a GPU is available. Training on GPU is much faster.
If no GPU is available, we fall back to CPU.
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# STEP 2: DATA PREPARATION
# ============================================================================
"""
We need to prepare our data with proper transformations:
1. Resize images to 224x224 (ResNet expects this size)
2. Convert to tensor
3. Normalize using ImageNet statistics (important for transfer learning!)

Why ImageNet statistics?
- Our pre-trained model was trained on ImageNet
- It expects images normalized with ImageNet mean and std
- This ensures the input distribution matches what the model saw during training
"""

# ImageNet normalization values
# These are the mean and std of millions of ImageNet images
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB channels
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB channels

# Define transformations for training data
train_transform = transforms.Compose([
    transforms.Resize(256),              # Resize shortest side to 256
    transforms.CenterCrop(224),          # Crop to 224x224 (ResNet input size)
    transforms.ToTensor(),               # Convert PIL Image to tensor [0, 1]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # Normalize
])

# Define transformations for test data (no augmentation needed)
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

print("\nDownloading and loading CIFAR-10 dataset...")
print("(This may take a few minutes on first run)")

# Load CIFAR-10 dataset
# CIFAR-10 consists of 60000 32x32 color images in 10 classes
# - 50000 training images
# - 10000 test images
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)

# Create data loaders
# DataLoader handles batching, shuffling, and parallel loading
BATCH_SIZE = 32  # Process 32 images at a time

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,      # Shuffle training data for better learning
    num_workers=0      # Use 2 parallel workers for data loading
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,     # No need to shuffle test data
    num_workers=0
)

# CIFAR-10 class names
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

print(f"\nDataset loaded successfully!")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: {len(classes)}")
print(f"Classes: {classes}")

# ============================================================================
# STEP 3: LOAD PRE-TRAINED MODEL
# ============================================================================
"""
Now we load a pre-trained ResNet18 model.
ResNet18 is a relatively small and fast convolutional neural network.

The model comes with weights trained on ImageNet, which contains:
- 1.2 million training images
- 1000 classes
- General object recognition

We'll use this knowledge as a starting point!
"""

print("\nLoading pre-trained ResNet18 model...")

# Load ResNet18 with pre-trained weights from ImageNet
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

print("Pre-trained model loaded successfully!")
print(f"\nModel architecture overview:")
print(f"- Input: 224x224 RGB images")
print(f"- Feature extractor: Convolutional layers (frozen)")
print(f"- Classifier: Final fully connected layer (will be trained)")

# ============================================================================
# STEP 4: FREEZE THE FEATURE EXTRACTOR
# ============================================================================
"""
This is the key to transfer learning!

We FREEZE all layers except the final one by setting requires_grad=False.
This means:
- These layers won't update during training
- We save computation time
- We preserve the learned features from ImageNet
"""

print("\nFreezing all layers except the final classifier...")

# Freeze all parameters in the model
for param in model.parameters():
    param.requires_grad = False

# Now all layers are frozen - they won't be updated during training

# ============================================================================
# STEP 5: REPLACE THE FINAL LAYER
# ============================================================================
"""
ResNet18 was trained on ImageNet (1000 classes), but CIFAR-10 has only 10 classes.
We need to replace the final fully connected layer.

The original final layer: fc (512 inputs → 1000 outputs)
Our new final layer: fc (512 inputs → 10 outputs)

This new layer has requires_grad=True by default, so it will be trained!
"""

# Get the number of input features for the final layer
num_features = model.fc.in_features
print(f"\nReplacing final layer:")
print(f"- Input features: {num_features}")
print(f"- Output classes: {len(classes)} (CIFAR-10)")

# Replace the final fully connected layer
# nn.Linear creates a layer: y = xW^T + b
model.fc = nn.Linear(num_features, len(classes))

# Move model to the device (GPU or CPU)
model = model.to(device)

print("Model prepared successfully!")

# Let's verify what parameters will be trained
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# ============================================================================
# STEP 6: DEFINE LOSS FUNCTION AND OPTIMIZER
# ============================================================================
"""
Loss Function (Criterion):
- CrossEntropyLoss for multi-class classification
- Combines LogSoftmax and NLLLoss
- Perfect for classification tasks

Optimizer:
- Adam: Adaptive learning rate optimizer
- Only optimizes the final layer (the only trainable parameters)
- Learning rate of 0.001 is a good starting point
"""

criterion = nn.CrossEntropyLoss()

# Adam optimizer - we only pass parameters that require gradients
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001  # Learning rate
)

print("\nTraining configuration:")
print(f"- Loss function: CrossEntropyLoss")
print(f"- Optimizer: Adam")
print(f"- Learning rate: 0.001")
print(f"- Batch size: {BATCH_SIZE}")

# ============================================================================
# STEP 7: TRAINING FUNCTION
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to train on (CPU or GPU)
    
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()  # Set model to training mode (enables dropout, batchnorm updates)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Iterate over batches
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch {batch_idx + 1}/{len(train_loader)}: '
                  f'Loss: {running_loss / (batch_idx + 1):.3f}, '
                  f'Acc: {100. * correct / total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# ============================================================================
# STEP 8: EVALUATION FUNCTION
# ============================================================================

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on (CPU or GPU)
    
    Returns:
        Average loss and accuracy
    """
    model.eval()  # Set model to evaluation mode (disables dropout, batchnorm uses running stats)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # We don't need gradients for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

# ============================================================================
# STEP 9: TRAINING LOOP
# ============================================================================
"""
Now we put it all together and train the model!

We'll train for 10 epochs, which means the model will see all training data 10 times.
After each epoch, we evaluate on the test set to monitor progress.
"""

NUM_EPOCHS = 10

print(f"\n{'='*70}")
print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"{'='*70}\n")

# Keep track of the best model
best_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

# Start training timer
start_time = time.time()

# Training loop
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 70)
    
    # Train for one epoch
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # Print epoch results
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        print(f"  ✓ New best accuracy! Saving model...")
    
    print()

# Calculate training time
total_time = time.time() - start_time
print(f"{'='*70}")
print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
print(f"Best test accuracy: {best_acc:.2f}%")
print(f"{'='*70}\n")

# Load best model weights
model.load_state_dict(best_model_weights)

# ============================================================================
# STEP 10: FINAL EVALUATION
# ============================================================================
"""
Let's do a final comprehensive evaluation on the test set.
"""

print("Final Evaluation on Test Set:")
print("-" * 70)

final_loss, final_acc = evaluate(model, test_loader, criterion, device)
print(f"Final Test Loss: {final_loss:.4f}")
print(f"Final Test Accuracy: {final_acc:.2f}%")

# ============================================================================
# STEP 11: PER-CLASS ACCURACY
# ============================================================================
"""
Let's see how well the model performs on each individual class.
This helps us understand if the model is biased towards certain classes.
"""

print("\nPer-class accuracy:")
print("-" * 70)

class_correct = [0] * len(classes)
class_total = [0] * len(classes)

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        # Count correct predictions per class
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

# Print per-class accuracies
for i in range(len(classes)):
    accuracy = 100 * class_correct[i] / class_total[i]
    print(f"{classes[i]:>10s}: {accuracy:>6.2f}%")

# ============================================================================
# STEP 12: SAVE THE MODEL (OPTIONAL)
# ============================================================================
"""
You can save the trained model for later use.
"""

print("\nSaving model...")
torch.save(model.state_dict(), 'resnet18_cifar10_transfer.pth')
print("Model saved as 'resnet18_cifar10_transfer.pth'")

# ============================================================================
# SUMMARY AND KEY TAKEAWAYS
# ============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nKey Takeaways from Example 1:")
print("1. We used a pre-trained ResNet18 model as a feature extractor")
print("2. We froze all layers except the final classifier")
print("3. We only trained 5,130 parameters (the final layer)")
print("4. Despite training so few parameters, we achieved good accuracy!")
print("5. Training was fast because we didn't update most of the network")
print("\nNext Steps:")
print("- Try Example 2 to learn about fine-tuning")
print("- Experiment with different learning rates")
print("- Try other pre-trained models (ResNet50, VGG16, etc.)")
print("="*70)
