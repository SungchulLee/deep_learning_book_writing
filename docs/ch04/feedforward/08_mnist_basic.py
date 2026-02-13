"""
================================================================================
03_mnist_basic.py - Complete MNIST Digit Classifier
================================================================================

This example implements a complete image classification pipeline using the
famous MNIST dataset of handwritten digits (0-9).

DATASET: MNIST
    - 60,000 training images
    - 10,000 test images
    - 28×28 pixel grayscale images
    - 10 classes (digits 0-9)

ARCHITECTURE:
    Input (784) → Hidden (128) with ReLU → Output (10) with Softmax

This is your first real-world deep learning task!

LEARNING OBJECTIVES:
    1. Load and preprocess real datasets
    2. Build a complete training pipeline
    3. Implement proper train/test splits
    4. Evaluate model performance
    5. Use GPU acceleration
    6. Visualize predictions

DIFFICULTY: ⭐⭐⭐☆☆ (Beginner-Intermediate)
TIME: 30-45 minutes
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ================================================================================
# PART 1: CONFIGURATION AND DEVICE SETUP
# ================================================================================
print("=" * 80)
print("STEP 1: Configuration and Device Setup")
print("=" * 80)

# Set random seed for reproducibility
# This ensures consistent results across runs
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
# PyTorch can run on CPU or GPU (CUDA)
# Using GPU dramatically speeds up training (10-100x faster)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Hyperparameters
# These control the learning process and model architecture
config = {
    'input_size': 784,        # 28×28 = 784 flattened pixels
    'hidden_size': 128,       # Number of neurons in hidden layer
    'num_classes': 10,        # Digits 0-9
    'num_epochs': 5,          # How many times to see entire dataset
    'batch_size': 100,        # Samples per training step
    'learning_rate': 0.001,   # Step size for optimizer
}

print(f"\nHyperparameters:")
for key, value in config.items():
    print(f"  {key:15s}: {value}")

# ================================================================================
# PART 2: DATA LOADING AND PREPROCESSING
# ================================================================================
print("\n" + "=" * 80)
print("STEP 2: Loading MNIST Dataset")
print("=" * 80)

# Transform: Convert PIL images to PyTorch tensors
# ToTensor() automatically scales pixel values from [0, 255] to [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
])

# Download and load training data
# The data is automatically downloaded to './data' if not present
print("Loading training data...")
train_dataset = torchvision.datasets.MNIST(
    root='./data',           # Where to store data
    train=True,              # Load training split
    transform=transform,     # Apply transformations
    download=True            # Download if not present
)

# Load test data
print("Loading test data...")
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,             # Load test split
    transform=transform,
    download=True
)

print(f"\nDataset Statistics:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")
print(f"  Image shape: {train_dataset[0][0].shape}")  # (channels, height, width)
print(f"  Number of classes: {len(train_dataset.classes)}")

# Create data loaders
# DataLoader handles batching, shuffling, and parallel loading
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,            # Shuffle training data each epoch
    num_workers=2,           # Use 2 subprocesses for data loading
    pin_memory=True          # Speed up CPU-GPU transfer
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=config['batch_size'],
    shuffle=False,           # Don't shuffle test data
    num_workers=2,
    pin_memory=True
)

print(f"\nDataLoader Info:")
print(f"  Training batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ================================================================================
# PART 3: VISUALIZE SAMPLE DATA
# ================================================================================
print("\n" + "=" * 80)
print("STEP 3: Visualizing Sample Images")
print("=" * 80)

# Get a batch of test images
examples = iter(test_loader)
example_data, example_labels = next(examples)

# Plot 12 sample images
fig, axes = plt.subplots(2, 6, figsize=(12, 4))
for i, ax in enumerate(axes.flat):
    # Convert from (1, 28, 28) to (28, 28) for plotting
    image = example_data[i].squeeze()
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Label: {example_labels[i]}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('03_mnist_samples.png', dpi=150, bbox_inches='tight')
print("Sample images saved as '03_mnist_samples.png'")
plt.close()

# ================================================================================
# PART 4: DEFINE THE NEURAL NETWORK
# ================================================================================
print("\n" + "=" * 80)
print("STEP 4: Building the Neural Network")
print("=" * 80)

class MNISTClassifier(nn.Module):
    """
    Feedforward neural network for MNIST classification.
    
    Architecture:
        Input (784) → Hidden (128) with ReLU → Output (10)
    
    Note: We don't apply softmax here because CrossEntropyLoss
    does it internally (it's more numerically stable that way).
    """
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(MNISTClassifier, self).__init__()
        
        # Layer 1: Input → Hidden
        # 784 → 128 transformation
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # ReLU activation
        # Introduces non-linearity, enables learning complex patterns
        self.relu = nn.ReLU()
        
        # Layer 2: Hidden → Output
        # 128 → 10 transformation (one output per digit class)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            Output logits of shape (batch_size, 10)
        """
        # Flatten the image
        # From (batch_size, 1, 28, 28) to (batch_size, 784)
        # -1 means "infer this dimension"
        x = x.reshape(x.size(0), -1)
        
        # Layer 1: Linear → ReLU
        hidden = self.fc1(x)           # (batch_size, 128)
        hidden = self.relu(hidden)     # (batch_size, 128)
        
        # Layer 2: Linear (no activation - CrossEntropyLoss expects logits)
        output = self.fc2(hidden)      # (batch_size, 10)
        
        return output
    
    def predict(self, x):
        """
        Make predictions (return class labels, not logits).
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            Predicted class labels of shape (batch_size,)
        """
        logits = self.forward(x)
        # torch.max returns (values, indices)
        # We want indices (class with highest probability)
        _, predicted = torch.max(logits, dim=1)
        return predicted

# Instantiate the model and move to device
model = MNISTClassifier(
    config['input_size'],
    config['hidden_size'],
    config['num_classes']
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model: MNISTClassifier")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Parameters breakdown:")
print(f"    Layer 1: {config['input_size']} × {config['hidden_size']} + {config['hidden_size']} = {config['input_size'] * config['hidden_size'] + config['hidden_size']:,}")
print(f"    Layer 2: {config['hidden_size']} × {config['num_classes']} + {config['num_classes']} = {config['hidden_size'] * config['num_classes'] + config['num_classes']:,}")

# ================================================================================
# PART 5: DEFINE LOSS AND OPTIMIZER
# ================================================================================
print("\n" + "=" * 80)
print("STEP 5: Setting Up Training Components")
print("=" * 80)

# Loss function: Cross-Entropy Loss
# Perfect for multi-class classification
# Combines LogSoftmax and NLLLoss in one step
# Expects raw logits (no softmax applied)
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam
# Adaptive learning rate optimizer
# Works well out of the box for most problems
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: Adam")
print(f"Learning rate: {config['learning_rate']}")

# ================================================================================
# PART 6: TRAINING LOOP
# ================================================================================
print("\n" + "=" * 80)
print("STEP 6: Training the Model")
print("=" * 80)

# Training history
train_losses = []
train_accuracies = []

# Total number of steps
total_steps = len(train_loader)

print(f"\nStarting training for {config['num_epochs']} epochs...")
print(f"Steps per epoch: {total_steps}")
print("-" * 80)

for epoch in range(config['num_epochs']):
    # Set model to training mode
    # This affects layers like Dropout and BatchNorm (not used here, but good practice)
    model.train()
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device (GPU/CPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # ----------------------------------------
        # Forward pass
        # ----------------------------------------
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # ----------------------------------------
        # Backward pass and optimization
        # ----------------------------------------
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()         # Compute gradients
        optimizer.step()        # Update weights
        
        # ----------------------------------------
        # Track statistics
        # ----------------------------------------
        epoch_loss += loss.item()
        
        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            current_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{config['num_epochs']}], "
                  f"Step [{batch_idx+1}/{total_steps}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Accuracy: {current_acc:.2f}%")
    
    # Compute epoch statistics
    avg_loss = epoch_loss / total_steps
    epoch_accuracy = 100 * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(epoch_accuracy)
    
    print(f"\nEpoch [{epoch+1}/{config['num_epochs']}] Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Training Accuracy: {epoch_accuracy:.2f}%")
    print("-" * 80)

print("\nTraining completed!")

# ================================================================================
# PART 7: EVALUATION ON TEST SET
# ================================================================================
print("\n" + "=" * 80)
print("STEP 7: Evaluating on Test Set")
print("=" * 80)

# Set model to evaluation mode
# Disables dropout, uses running stats for batchnorm, etc.
model.eval()

# Disable gradient computation for efficiency
# We don't need gradients during inference
with torch.no_grad():
    correct = 0
    total = 0
    
    # Per-class accuracy tracking
    class_correct = [0] * config['num_classes']
    class_total = [0] * config['num_classes']
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Overall accuracy
overall_accuracy = 100 * correct / total
print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
print(f"Correct predictions: {correct}/{total}")

# Per-class accuracy
print("\nPer-Class Accuracy:")
print("-" * 40)
for i in range(config['num_classes']):
    class_acc = 100 * class_correct[i] / class_total[i]
    print(f"  Digit {i}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
print("-" * 40)

# ================================================================================
# PART 8: VISUALIZE PREDICTIONS
# ================================================================================
print("\n" + "=" * 80)
print("STEP 8: Visualizing Predictions")
print("=" * 80)

# Get a batch of test images
model.eval()
examples = iter(test_loader)
example_data, example_labels = next(examples)
example_data = example_data.to(device)
example_labels = example_labels.to(device)

with torch.no_grad():
    outputs = model(example_data)
    _, predictions = torch.max(outputs, 1)
    
    # Get probabilities (softmax of logits)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)

# Move back to CPU for plotting
example_data = example_data.cpu()
example_labels = example_labels.cpu()
predictions = predictions.cpu()
probabilities = probabilities.cpu()

# Plot predictions
fig, axes = plt.subplots(3, 6, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    if i < 18:
        image = example_data[i].squeeze()
        true_label = example_labels[i].item()
        pred_label = predictions[i].item()
        confidence = probabilities[i][pred_label].item() * 100
        
        ax.imshow(image, cmap='gray')
        
        # Color code: green for correct, red for wrong
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.1f}%',
                    color=color, fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.savefig('03_mnist_predictions.png', dpi=150, bbox_inches='tight')
print("Predictions saved as '03_mnist_predictions.png'")
plt.close()

# ================================================================================
# PART 9: TRAINING VISUALIZATION
# ================================================================================
print("\n" + "=" * 80)
print("STEP 9: Training Progress Visualization")
print("=" * 80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot loss
ax1.plot(range(1, config['num_epochs'] + 1), train_losses, 'b-', linewidth=2, marker='o')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Average Loss', fontsize=12)
ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot accuracy
ax2.plot(range(1, config['num_epochs'] + 1), train_accuracies, 'g-', linewidth=2, marker='s')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('03_mnist_training_progress.png', dpi=150, bbox_inches='tight')
print("Training progress saved as '03_mnist_training_progress.png'")
plt.show()

# ================================================================================
# KEY TAKEAWAYS
# ================================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print(f"""
1. Complete ML Pipeline:
   ✓ Data loading and preprocessing
   ✓ Model architecture design
   ✓ Training loop with monitoring
   ✓ Evaluation on held-out test set
   ✓ Visualization of results

2. Achieved ~{overall_accuracy:.1f}% accuracy with simple 2-layer network!
   - State-of-the-art CNNs achieve ~99.7%
   - This baseline is quite respectable

3. CrossEntropyLoss for multi-class classification
   - Combines LogSoftmax + NLLLoss
   - More numerically stable than separate operations

4. GPU acceleration makes training much faster
   - Always move both model AND data to device
   - Use .to(device) for tensors and models

5. Training vs Evaluation mode:
   - model.train(): Enables dropout, batchnorm training
   - model.eval(): Disables them for inference

NEXT: Level 2 will introduce more PyTorch features and better architectures!
""")

# ================================================================================
# EXERCISES FOR STUDENTS
# ================================================================================
print("=" * 80)
print("EXERCISES TO TRY")
print("=" * 80)
print("""
1. Increase hidden_size to 256 or 512 - does accuracy improve?
2. Add another hidden layer - create a 3-layer network
3. Try different optimizers: SGD, RMSprop, AdaGrad
4. Experiment with learning rates: 0.0001, 0.01, 0.1
5. Train for more epochs (10-20) - watch for overfitting
6. Implement early stopping based on validation loss
7. Save the trained model: torch.save(model.state_dict(), 'model.pth')
8. Add data augmentation: random rotations, shifts
9. Visualize what the network learned: plot first layer weights
10. Create a confusion matrix to see which digits are confused
""")
