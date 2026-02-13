# [Code Source](https://github.com/patrickloeber/pytorchTutorial)
# ============================================================================
# MNIST Neural Network Training with TensorBoard Visualization
# ============================================================================
# This script demonstrates:
# - Training a neural network on the MNIST dataset
# - Using TensorBoard for visualization and monitoring
# - Logging training metrics, model graph, and PR curves
# ============================================================================

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ============================================================================
# TENSORBOARD SETUP
# ============================================================================
# TensorBoard is a visualization toolkit for machine learning experimentation
# It helps track and visualize metrics, model graphs, and other data
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Create a SummaryWriter instance to log data to TensorBoard
# The 'runs/mnist1' directory will store all TensorBoard log files
# You can view these logs by running: tensorboard --logdir=runs
writer = SummaryWriter('runs/mnist1')
# ============================================================================

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
# Check if CUDA (GPU) is available, otherwise use CPU
# Training on GPU is significantly faster for neural networks
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
# These values control the model architecture and training process
input_size = 784        # 28x28 pixels flattened into a 1D vector
hidden_size = 500       # Number of neurons in the hidden layer
num_classes = 10        # MNIST has 10 classes (digits 0-9)
num_epochs = 1          # Number of complete passes through the training dataset
batch_size = 64         # Number of samples processed before updating the model
learning_rate = 0.001   # Step size for gradient descent optimization

# ============================================================================
# DATASET LOADING
# ============================================================================
# MNIST dataset contains 60,000 training images and 10,000 test images
# of handwritten digits (0-9), each image is 28x28 pixels

# Load training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',                          # Directory to store/load data
    train=True,                             # Load training set
    transform=transforms.ToTensor(),        # Convert PIL images to tensors
    download=True                           # Download if not already present
)

# Load test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data',                          # Same directory for consistency
    train=False,                            # Load test set
    transform=transforms.ToTensor()         # Apply same transformation
)

# ============================================================================
# DATA LOADERS
# ============================================================================
# DataLoader handles batching, shuffling, and parallel data loading

# Training data loader with shuffling for better generalization
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True                            # Shuffle data each epoch
)

# Test data loader without shuffling (order doesn't matter for evaluation)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# ============================================================================
# VISUALIZE SAMPLE IMAGES
# ============================================================================
# Get a batch of test data for visualization
examples = iter(test_loader)
example_data, example_targets = next(examples)

# Create a 2x3 grid of sample images using matplotlib
for i in range(6):
    plt.subplot(2, 3, i+1)                  # Create subplot in 2x3 grid
    plt.imshow(example_data[i][0], cmap='gray')  # Display grayscale image
    plt.title(f'Label: {example_targets[i]}')    # Show the true label
    plt.axis('off')                         # Hide axes for cleaner look
# plt.show()  # Uncomment to display the plot

# ============================================================================
# TENSORBOARD: LOG SAMPLE IMAGES
# ============================================================================
# Create a grid of images and log them to TensorBoard
# This helps visualize what the model is working with
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)

# Optional: Close writer and exit here if you only want to visualize images
# writer.close()
# sys.exit()
# ============================================================================

# ============================================================================
# MODEL DEFINITION
# ============================================================================
# Define a simple fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    """
    A simple feedforward neural network for MNIST classification.
    
    Architecture:
    - Input layer: 784 neurons (28x28 flattened image)
    - Hidden layer: 500 neurons with ReLU activation
    - Output layer: 10 neurons (one per digit class)
    
    Args:
        input_size (int): Number of input features (784 for MNIST)
        hidden_size (int): Number of neurons in hidden layer
        num_classes (int): Number of output classes (10 for MNIST)
    """
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        
        # First linear layer: input_size -> hidden_size
        self.l1 = nn.Linear(input_size, hidden_size)
        
        # ReLU activation function: max(0, x)
        # Introduces non-linearity to learn complex patterns
        self.relu = nn.ReLU()
        
        # Second linear layer: hidden_size -> num_classes
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Raw output scores (logits) of shape (batch_size, num_classes)
        """
        # Pass through first linear layer
        out = self.l1(x)
        
        # Apply ReLU activation
        out = self.relu(out)
        
        # Pass through second linear layer
        out = self.l2(out)
        
        # Note: No softmax here because CrossEntropyLoss applies it internally
        return out

# ============================================================================
# MODEL INSTANTIATION
# ============================================================================
# Create the model and move it to the appropriate device (CPU or GPU)
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
print(f'\nModel Architecture:\n{model}')

# ============================================================================
# LOSS FUNCTION AND OPTIMIZER
# ============================================================================
# CrossEntropyLoss combines softmax and negative log likelihood
# It's ideal for multi-class classification problems
criterion = nn.CrossEntropyLoss()

# Adam optimizer: adaptive learning rate optimization algorithm
# Generally works well without much hyperparameter tuning
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ============================================================================
# TENSORBOARD: LOG MODEL GRAPH
# ============================================================================
# Log the computational graph of the model to TensorBoard
# This visualizes the network architecture and data flow
writer.add_graph(model, example_data.reshape(-1, 28*28).to(device))

# Optional: Close writer and exit here if you only want to visualize the graph
# writer.close()
# sys.exit()
# ============================================================================

# ============================================================================
# TRAINING LOOP
# ============================================================================
print('\n' + '='*60)
print('STARTING TRAINING')
print('='*60)

# Variables to track running metrics for TensorBoard logging
running_loss = 0.0
running_correct = 0
n_total_steps = len(train_loader)

# Loop over the entire dataset for the specified number of epochs
for epoch in range(num_epochs):
    # Loop over batches in the training set
    for i, (images, labels) in enumerate(train_loader):
        # ====================================================================
        # DATA PREPARATION
        # ====================================================================
        # Original shape: [batch_size, 1, 28, 28] (batch, channels, height, width)
        # Reshaped to: [batch_size, 784] (flatten each image)
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # ====================================================================
        # FORWARD PASS
        # ====================================================================
        # Compute model predictions
        outputs = model(images)
        
        # Compute loss between predictions and true labels
        loss = criterion(outputs, labels)
        
        # ====================================================================
        # BACKWARD PASS AND OPTIMIZATION
        # ====================================================================
        # Clear gradients from previous iteration
        # PyTorch accumulates gradients by default
        optimizer.zero_grad()
        
        # Compute gradients through backpropagation
        loss.backward()
        
        # Update model parameters using computed gradients
        optimizer.step()
        
        # ====================================================================
        # ACCUMULATE METRICS
        # ====================================================================
        # Add current loss to running total
        running_loss += loss.item()
        
        # Get predicted class (index of maximum logit)
        _, predicted = torch.max(outputs.data, 1)
        
        # Count correct predictions
        running_correct += (predicted == labels).sum().item()
        
        # ====================================================================
        # LOGGING TO TENSORBOARD (every 100 steps)
        # ====================================================================
        if (i+1) % 100 == 0:
            # Print progress to console
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{i+1}/{n_total_steps}], '
                  f'Loss: {loss.item():.4f}')
            
            # ----------------------------------------------------------------
            # LOG TRAINING LOSS
            # ----------------------------------------------------------------
            # Average loss over the last 100 batches
            writer.add_scalar('training loss', 
                            running_loss / 100, 
                            epoch * n_total_steps + i)
            
            # ----------------------------------------------------------------
            # LOG TRAINING ACCURACY
            # ----------------------------------------------------------------
            # Calculate accuracy: correct predictions / total predictions
            # predicted.size(0) is the batch size
            running_accuracy = running_correct / 100 / predicted.size(0)
            writer.add_scalar('accuracy', 
                            running_accuracy, 
                            epoch * n_total_steps + i)
            
            # Reset running metrics for next 100 batches
            running_correct = 0
            running_loss = 0.0

print('='*60)
print('TRAINING COMPLETED')
print('='*60 + '\n')

# ============================================================================
# MODEL EVALUATION ON TEST SET
# ============================================================================
print('='*60)
print('EVALUATING MODEL ON TEST SET')
print('='*60)

# Lists to store predictions and labels for all batches
class_labels = []
class_preds = []

# Disable gradient computation for evaluation (saves memory and computation)
with torch.no_grad():
    n_correct = 0    # Total number of correct predictions
    n_samples = 0    # Total number of samples
    
    # Loop over test batches
    for images, labels in test_loader:
        # Prepare data (same as training)
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Get predicted classes
        # torch.max returns both values and indices
        values, predicted = torch.max(outputs.data, 1)
        
        # Accumulate statistics
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        # ====================================================================
        # COMPUTE CLASS PROBABILITIES FOR PR CURVES
        # ====================================================================
        # Apply softmax to convert logits to probabilities
        # Do this for each sample in the batch
        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]
        
        # Store predictions and labels for this batch
        class_preds.append(class_probs_batch)
        class_labels.append(labels)
    
    # ========================================================================
    # CONSOLIDATE PREDICTIONS AND LABELS
    # ========================================================================
    # Stack: concatenates tensors along a NEW dimension
    # Cat: concatenates tensors along an EXISTING dimension
    
    # Convert list of lists to a single tensor
    # Final shape: [10000, 10] (all samples, all class probabilities)
    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    
    # Concatenate all label batches
    # Final shape: [10000] (all labels)
    class_labels = torch.cat(class_labels)
    
    # ========================================================================
    # CALCULATE AND DISPLAY ACCURACY
    # ========================================================================
    acc = 100.0 * n_correct / n_samples
    print(f'\nAccuracy of the network on the 10000 test images: {acc:.2f}%')
    
    # ========================================================================
    # TENSORBOARD: LOG PRECISION-RECALL CURVES
    # ========================================================================
    # PR curves show the trade-off between precision and recall
    # Useful for understanding model performance across different thresholds
    # A separate curve is created for each digit class (0-9)
    
    classes = range(10)
    for i in classes:
        # Create binary labels: True if sample is class i, False otherwise
        labels_i = class_labels == i
        
        # Get predicted probabilities for class i
        preds_i = class_preds[:, i]
        
        # Add PR curve to TensorBoard for this class
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    
    # Close the TensorBoard writer
    writer.close()
    print('\nTensorBoard logs saved to: runs/mnist1')
    print('To view, run: tensorboard --logdir=runs')
    print('='*60)

# ============================================================================
# END OF SCRIPT
# ============================================================================
# To view results in TensorBoard:
# 1. Open a terminal
# 2. Navigate to the directory containing this script
# 3. Run: tensorboard --logdir=runs
# 4. Open your browser and go to: http://localhost:6006
# ============================================================================
