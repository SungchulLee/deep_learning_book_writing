"""
cnn_utils.py
============
Comprehensive utility module for CNN tutorials

This module provides all common functions needed for training CNNs:
- Argument parsing and configuration
- Data loading for MNIST, Fashion-MNIST, and CIFAR-10
- Model architectures
- Training and evaluation loops
- Visualization utilities
- Model saving/loading

Author: PyTorch CNN Tutorial
Date: November 2025
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random


# =============================================================================
# SECTION 1: Configuration and Setup
# =============================================================================

def parse_args():
    """
    Parse command line arguments for training configuration
    
    Returns:
        Namespace object containing all configuration parameters
    """
    parser = argparse.ArgumentParser(description='PyTorch CNN Training')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    
    # System configuration
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging (default: 10)')
    
    # Model persistence
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--path', type=str, default='./model.pth',
                        help='Path to save/load model')
    
    args = parser.parse_args()
    
    # Set up device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    
    if use_cuda:
        args.device = torch.device("cuda")
    elif use_mps:
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")
    
    return args


def set_seed(seed=1):
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and Python
    
    Args:
        seed: Integer seed value (default: 1)
    
    Note:
        This ensures that experiments can be reproduced with the same results
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# SECTION 2: Data Loading
# =============================================================================

def load_data(train_kwargs, test_kwargs, fashion_mnist=False, cifar10=False):
    """
    Load and prepare datasets with appropriate transforms
    
    Args:
        train_kwargs: Dictionary of DataLoader arguments for training
        test_kwargs: Dictionary of DataLoader arguments for testing
        fashion_mnist: If True, load Fashion-MNIST instead of MNIST
        cifar10: If True, load CIFAR-10 instead of MNIST
    
    Returns:
        tuple: (train_loader, test_loader)
    
    Note:
        - MNIST/Fashion-MNIST: 28x28 grayscale images, 10 classes
        - CIFAR-10: 32x32 RGB images, 10 classes
        - All datasets are normalized to have mean 0.5 and std 0.5
    """
    
    if cifar10:
        # CIFAR-10: RGB images, need 3-channel normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Normalize each channel (R, G, B) to [-1, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Download and load CIFAR-10
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        
    else:
        # MNIST or Fashion-MNIST: Grayscale images, single-channel normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Normalize to [-1, 1] for single channel
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Select dataset type
        if fashion_mnist:
            dataset_class = datasets.FashionMNIST
        else:
            dataset_class = datasets.MNIST
        
        # Download and load selected dataset
        train_dataset = dataset_class(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = dataset_class(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
    
    # Create data loaders with specified parameters
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)
    
    return train_loader, test_loader


# =============================================================================
# SECTION 3: Model Architectures
# =============================================================================

class CNN(nn.Module):
    """
    Basic CNN architecture for MNIST and Fashion-MNIST
    
    Architecture:
        Conv1: 1 → 32 channels, 3x3 kernel, ReLU
        MaxPool: 2x2
        Conv2: 32 → 64 channels, 3x3 kernel, ReLU
        MaxPool: 2x2
        FC1: 64*7*7 → 128, ReLU, Dropout(0.5)
        FC2: 128 → 10
    
    Input: (batch_size, 1, 28, 28)
    Output: (batch_size, 10) logits
    """
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # First convolutional layer: 1 input channel, 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Max pooling layer: 2x2 window
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 2 pooling layers: 28 -> 14 -> 7
        # So we have 64 channels * 7 * 7 = 3136 features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            Output tensor of shape (batch_size, 10) - class logits
        """
        # First conv block: Conv -> ReLU -> Pool
        x = self.conv1(x)           # (N, 1, 28, 28) -> (N, 32, 28, 28)
        x = F.relu(x)
        x = self.pool(x)            # (N, 32, 28, 28) -> (N, 32, 14, 14)
        x = self.dropout1(x)
        
        # Second conv block: Conv -> ReLU -> Pool
        x = self.conv2(x)           # (N, 32, 14, 14) -> (N, 64, 14, 14)
        x = F.relu(x)
        x = self.pool(x)            # (N, 64, 14, 14) -> (N, 64, 7, 7)
        x = self.dropout1(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)  # (N, 64, 7, 7) -> (N, 3136)
        
        # First FC layer with ReLU and dropout
        x = self.fc1(x)             # (N, 3136) -> (N, 128)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer (no activation - will use CrossEntropyLoss)
        x = self.fc2(x)             # (N, 128) -> (N, 10)
        
        return x


class CNN_CIFAR10(nn.Module):
    """
    Advanced CNN architecture for CIFAR-10
    
    Architecture:
        Conv1: 3 → 32 channels, 3x3 kernel, ReLU
        Conv2: 32 → 32 channels, 3x3 kernel, ReLU
        MaxPool: 2x2
        Conv3: 32 → 64 channels, 3x3 kernel, ReLU
        Conv4: 64 → 64 channels, 3x3 kernel, ReLU
        MaxPool: 2x2
        FC1: 64*8*8 → 512, ReLU, Dropout(0.5)
        FC2: 512 → 10
    
    Input: (batch_size, 3, 32, 32)
    Output: (batch_size, 10) logits
    """
    
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        
        # First block: two conv layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        
        # Second block: two conv layers with more filters
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 2 pooling: 32 -> 16 -> 8
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
        
        Returns:
            Output tensor of shape (batch_size, 10) - class logits
        """
        # First block: 2 convs + pool
        x = F.relu(self.conv1(x))   # (N, 3, 32, 32) -> (N, 32, 32, 32)
        x = F.relu(self.conv2(x))   # (N, 32, 32, 32) -> (N, 32, 32, 32)
        x = self.pool(x)            # (N, 32, 32, 32) -> (N, 32, 16, 16)
        x = self.dropout1(x)
        
        # Second block: 2 convs + pool
        x = F.relu(self.conv3(x))   # (N, 32, 16, 16) -> (N, 64, 16, 16)
        x = F.relu(self.conv4(x))   # (N, 64, 16, 16) -> (N, 64, 16, 16)
        x = self.pool(x)            # (N, 64, 16, 16) -> (N, 64, 8, 8)
        x = self.dropout1(x)
        
        # Flatten and FC layers
        x = x.view(-1, 64 * 8 * 8)  # (N, 64, 8, 8) -> (N, 4096)
        x = F.relu(self.fc1(x))     # (N, 4096) -> (N, 512)
        x = self.dropout2(x)
        x = self.fc2(x)             # (N, 512) -> (N, 10)
        
        return x


# =============================================================================
# SECTION 4: Training and Evaluation
# =============================================================================

def train(model, train_loader, loss_fn, optimizer, scheduler, device, epochs, log_interval=10, dry_run=False):
    """
    Complete training loop for the model
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer (e.g., SGD, Adam)
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        epochs: Number of training epochs
        log_interval: How often to print training statistics
        dry_run: If True, only run one batch per epoch (for testing)
    
    Returns:
        None (trains model in-place)
    """
    model.train()  # Set model to training mode
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Compute loss
            loss = loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Logging
            if batch_idx % log_interval == 0:
                print(f'Epoch: {epoch}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}')
            
            if dry_run:
                break
        
        # Epoch summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'\nEpoch {epoch} Summary: Avg Loss: {epoch_loss:.4f}, '
              f'Accuracy: {correct}/{total} ({epoch_acc:.2f}%)\n')
        
        # Update learning rate
        scheduler.step()


def compute_accuracy(model, test_loader, device):
    """
    Compute accuracy on test set
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        float: Test accuracy as a percentage
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():  # Don't compute gradients during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get predictions
            _, predicted = output.max(1)
            
            # Update counters
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'\nTest Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    
    return accuracy


# =============================================================================
# SECTION 5: Visualization
# =============================================================================

def show_batch_or_ten_images_with_label_and_predict(test_loader, model, device, 
                                                      classes=None, n=10, cifar10=False):
    """
    Visualize predictions on a batch of images
    
    Args:
        test_loader: DataLoader for test data
        model: Trained model for predictions
        device: Device to run inference on
        classes: Tuple of class names (optional)
        n: Number of images to display
        cifar10: If True, handle CIFAR-10 denormalization
    
    Returns:
        None (displays matplotlib figure)
    """
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predictions = outputs.max(1)
    
    # Move to CPU for visualization
    images = images.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()
    
    # Create figure
    n_display = min(n, len(images))
    cols = min(5, n_display)
    rows = (n_display + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    if n_display == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx in range(n_display):
        ax = axes[idx]
        img = images[idx]
        
        # Denormalize image
        if cifar10:
            img = img / 2 + 0.5  # From [-1, 1] to [0, 1]
            img = img.permute(1, 2, 0)  # CHW to HWC
            ax.imshow(img.numpy())
        else:
            img = img.squeeze()  # Remove channel dimension for grayscale
            img = img / 2 + 0.5  # From [-1, 1] to [0, 1]
            ax.imshow(img.numpy(), cmap='gray')
        
        # Create title with label and prediction
        true_label = labels[idx].item()
        pred_label = predictions[idx].item()
        
        if classes:
            title = f'True: {classes[true_label]}\nPred: {classes[pred_label]}'
        else:
            title = f'True: {true_label}\nPred: {pred_label}'
        
        # Color code: green if correct, red if wrong
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(title, fontsize=8, color=color)
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(n_display, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 6: Model Persistence
# =============================================================================

def save_model(model, path):
    """
    Save model state dictionary to disk
    
    Args:
        model: PyTorch model to save
        path: File path to save to
    
    Returns:
        None
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model_class, device, path):
    """
    Load model state dictionary from disk
    
    Args:
        model_class: Model class (not instance) to instantiate
        device: Device to load model to
        path: File path to load from
    
    Returns:
        Loaded model instance
    """
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
    return model


# =============================================================================
# MODULE INFORMATION
# =============================================================================

__version__ = "1.0.0"
__author__ = "PyTorch CNN Tutorial"
__all__ = [
    'parse_args',
    'set_seed',
    'load_data',
    'CNN',
    'CNN_CIFAR10',
    'train',
    'compute_accuracy',
    'show_batch_or_ten_images_with_label_and_predict',
    'save_model',
    'load_model'
]
