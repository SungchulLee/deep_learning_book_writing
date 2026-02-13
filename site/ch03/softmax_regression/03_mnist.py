"""
===============================================================================
LEVEL 3: Softmax Regression on MNIST Dataset
===============================================================================
Difficulty: Intermediate
Prerequisites: Level 1 & 2, understanding of image data
Learning Goals:
  - Work with real-world image dataset (MNIST)
  - Handle data loading and batching
  - Implement proper train/validation/test splits
  - Use data loaders and mini-batch training
  - Visualize predictions on images

Time to complete: 45-60 minutes
===============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("LEVEL 3: SOFTMAX REGRESSION ON MNIST")
print("=" * 80)


# =============================================================================
# PART 1: Load and Explore MNIST Dataset
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: Loading MNIST Dataset")
print("=" * 80)

"""
MNIST Dataset:
--------------
- 70,000 handwritten digit images (0-9)
- 28x28 pixels, grayscale
- 60,000 training images
- 10,000 test images
- One of the most famous datasets in machine learning!
"""

# Define transformations for the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor (0-1 range)
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
])

# Download and load the training data
print("Downloading MNIST dataset...")
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Download and load the test data
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"✅ Dataset loaded successfully!")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}")  # (1, 28, 28) - CHW format
print(f"Number of classes: {len(train_dataset.classes)}")


# =============================================================================
# PART 2: Visualize Sample Images
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: Exploring the Data")
print("=" * 80)

def show_images(dataset, num_images=10):
    """
    Display a grid of sample images from the dataset.
    
    Args:
        dataset: PyTorch dataset
        num_images: Number of images to display
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_images):
        img, label = dataset[i]
        # Convert from CHW to HW (remove channel dimension for grayscale)
        img = img.squeeze().numpy()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# Uncomment to show sample images
# show_images(train_dataset)
# plt.show()

# Examine a single sample
sample_img, sample_label = train_dataset[0]
print(f"\nSample image shape: {sample_img.shape}")  # (1, 28, 28)
print(f"Sample label: {sample_label}")
print(f"Pixel value range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
print("(Values are normalized around 0)")


# =============================================================================
# PART 3: Create Data Loaders
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: Setting Up Data Loaders")
print("=" * 80)

"""
Data Loaders:
-------------
Instead of loading all data at once, we use data loaders to:
- Load data in mini-batches (saves memory)
- Shuffle data each epoch (prevents overfitting)
- Enable parallel data loading (faster training)
"""

# Split training data into train and validation sets
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # 20% for validation

train_dataset, val_dataset = random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Training set: {len(train_dataset)} samples")
print(f"Validation set: {len(val_dataset)} samples")
print(f"Test set: {len(test_dataset)} samples")

# Create data loaders
batch_size = 128  # Process 128 images at a time

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,      # Shuffle training data each epoch
    num_workers=0      # Parallel data loading
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,     # Don't shuffle validation data
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

print(f"\nBatch size: {batch_size}")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")
print(f"Number of test batches: {len(test_loader)}")


# =============================================================================
# PART 4: Define the Model
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: Building the Neural Network")
print("=" * 80)

class MNISTClassifier(nn.Module):
    """
    Neural network for MNIST digit classification.
    
    Architecture:
        Flatten (28×28) → 784 features
          ↓
        Linear Layer (784 → 256) + ReLU + Dropout
          ↓
        Linear Layer (256 → 128) + ReLU + Dropout
          ↓
        Output Layer (128 → 10 classes)
    
    Note: Dropout helps prevent overfitting by randomly dropping neurons
          during training (they're always active during inference).
    """
    
    def __init__(self, input_size=784, hidden_size1=256, hidden_size2=128, 
                 num_classes=10, dropout_rate=0.2):
        """
        Initialize the network.
        
        Args:
            input_size: Number of input features (28*28=784 for MNIST)
            hidden_size1: Neurons in first hidden layer
            hidden_size2: Neurons in second hidden layer
            num_classes: Number of output classes (10 digits)
            dropout_rate: Dropout probability (0.2 = drop 20% of neurons)
        """
        super(MNISTClassifier, self).__init__()
        
        # Flatten layer to convert 2D images to 1D vectors
        self.flatten = nn.Flatten()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output layer (no activation - we return logits)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Flatten the image: (batch, 1, 28, 28) → (batch, 784)
        x = self.flatten(x)
        
        # First hidden layer
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Output layer (logits)
        logits = self.fc3(x)
        return logits


# Create the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MNISTClassifier(
    input_size=784,
    hidden_size1=256,
    hidden_size2=128,
    num_classes=10,
    dropout_rate=0.2
).to(device)

print("\nModel Architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# =============================================================================
# PART 5: Training with Mini-Batches
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: Training the Model")
print("=" * 80)

# Set up training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: Adam (lr=0.001)")
print(f"Training epochs: {num_epochs}")


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Training accuracy
    """
    model.train()  # Set to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device (GPU/CPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Evaluate the model on validation set.
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
    """
    model.eval()  # Set to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


# Training loop
print("\nStarting training...\n")
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

start_time = time.time()

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, 
                                            optimizer, device)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Store history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print()

training_time = time.time() - start_time
print(f"✅ Training complete! Time taken: {training_time:.2f} seconds")


# =============================================================================
# PART 6: Evaluate on Test Set
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: Final Evaluation on Test Set")
print("=" * 80)

def test_model(model, test_loader, device):
    """
    Evaluate model on test set and return detailed metrics.
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, np.array(all_predictions), np.array(all_labels)


test_accuracy, predictions, true_labels = test_model(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")


# =============================================================================
# PART 7: Visualize Predictions
# =============================================================================
print("\n" + "=" * 80)
print("PART 7: Visualizing Predictions")
print("=" * 80)

def visualize_predictions(model, test_loader, device, num_images=10):
    """
    Show sample predictions with their probabilities.
    """
    model.eval()
    
    # Get one batch
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
    
    # Move back to CPU for plotting
    images = images.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()
    probabilities = probabilities.cpu()
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.ravel()
    
    for i in range(num_images):
        img = images[i].squeeze().numpy()
        true_label = labels[i].item()
        pred_label = predictions[i].item()
        prob = probabilities[i][pred_label].item()
        
        axes[i].imshow(img, cmap='gray')
        
        # Color code: green if correct, red if wrong
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {prob:.2f}',
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# Uncomment to show predictions
# visualize_predictions(model, test_loader, device)
# plt.show()


# =============================================================================
# PART 8: Confusion Matrix
# =============================================================================
print("\n" + "=" * 80)
print("PART 8: Confusion Matrix")
print("=" * 80)

from sklearn.metrics import confusion_matrix, classification_report

# Compute confusion matrix
cm = confusion_matrix(true_labels, predictions)

print("Confusion Matrix:")
print("-" * 80)
print(cm)
print("\nRow = True digit, Column = Predicted digit")

# Find most commonly confused digits
print("\nMost Common Misclassifications:")
print("-" * 80)
misclassifications = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            misclassifications.append((i, j, cm[i, j]))

# Sort by frequency
misclassifications.sort(key=lambda x: x[2], reverse=True)

for true_digit, pred_digit, count in misclassifications[:5]:
    print(f"Digit {true_digit} predicted as {pred_digit}: {count} times")

# Classification report
print("\n\nDetailed Classification Report:")
print("-" * 80)
print(classification_report(true_labels, predictions))


# =============================================================================
# PART 9: Per-Class Accuracy
# =============================================================================
print("\n" + "=" * 80)
print("PART 9: Per-Class Performance")
print("=" * 80)

# Calculate per-class accuracy
class_correct = [0] * 10
class_total = [0] * 10

for i in range(len(true_labels)):
    label = true_labels[i]
    class_total[label] += 1
    if predictions[i] == label:
        class_correct[label] += 1

print("Accuracy for each digit:")
print("-" * 80)
for i in range(10):
    accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"Digit {i}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")


# =============================================================================
# PART 10: Save the Model
# =============================================================================
print("\n" + "=" * 80)
print("PART 10: Saving the Model")
print("=" * 80)

# Save complete model information
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs,
    'train_acc': history['train_acc'][-1],
    'val_acc': history['val_acc'][-1],
    'test_acc': test_accuracy,
}

model_path = '/home/claude/softmax_regression_tutorial/level_03_mnist_model.pth'
torch.save(checkpoint, model_path)
print(f"✅ Model saved to: {model_path}")

# Load the model (demonstration)
loaded_checkpoint = torch.load(model_path)
loaded_model = MNISTClassifier().to(device)
loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
loaded_model.eval()

print(f"✅ Model loaded successfully")
print(f"   Saved test accuracy: {loaded_checkpoint['test_acc']:.4f}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY - What You Learned")
print("=" * 80)

print(f"""
✅ Loaded and explored MNIST dataset
✅ Used DataLoader for efficient batch processing
✅ Implemented train/validation/test splits
✅ Built a deeper neural network with dropout
✅ Trained on a real-world dataset
✅ Achieved {test_accuracy:.2%} accuracy on test set
✅ Visualized predictions and analyzed errors
✅ Generated confusion matrix and per-class metrics
✅ Saved and loaded trained models

Key Concepts:
-------------
• Mini-batch training: Process data in small batches
• Data loaders: Efficient data handling with shuffling
• Dropout: Regularization to prevent overfitting
• train() vs eval() modes: Different behavior for dropout/batchnorm
• Confusion matrix: Understand which classes are confused

Performance Summary:
--------------------
Training time: {training_time:.2f} seconds
Final train accuracy: {history['train_acc'][-1]:.2%}
Final validation accuracy: {history['val_acc'][-1]:.2%}
Test accuracy: {test_accuracy:.2%}

Next Steps:
-----------
→ Level 4: Implement custom training from scratch
→ Level 5: Advanced techniques (learning rate scheduling, data augmentation)
→ Experiment: Try different architectures, hyperparameters

🎉 Excellent work! You've successfully trained on MNIST!
""")
