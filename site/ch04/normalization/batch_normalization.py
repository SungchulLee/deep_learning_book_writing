"""
Batch Normalization Implementation and Examples
================================================

Batch Normalization normalizes inputs across the batch dimension.
It helps with faster training and allows higher learning rates.

Paper: "Batch Normalization: Accelerating Deep Network Training by 
       Reducing Internal Covariate Shift" (Ioffe & Szegedy, 2015)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm1dNumPy:
    """
    Batch Normalization implementation from scratch using NumPy.
    Normalizes across the batch dimension.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Args:
            num_features: Number of features/channels
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)  # Scale parameter
        self.beta = np.zeros(num_features)  # Shift parameter
        
        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        self.training = True
        
    def forward(self, x):
        """
        Forward pass of Batch Normalization.
        
        Args:
            x: Input of shape (batch_size, num_features)
            
        Returns:
            Normalized output of same shape as input
        """
        if self.training:
            # Calculate batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # Use running statistics during inference
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_normalized + self.beta
        
        return out


class SimpleNetworkWithBatchNorm(nn.Module):
    """
    Example neural network using PyTorch's BatchNorm layers.
    """
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleNetworkWithBatchNorm, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        return x


class ConvNetWithBatchNorm(nn.Module):
    """
    Convolutional network with Batch Normalization.
    Uses BatchNorm2d for convolutional layers.
    """
    
    def __init__(self, num_classes=10):
        super(ConvNetWithBatchNorm, self).__init__()
        
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Conv block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(256 * 4 * 4, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def demonstrate_batch_norm():
    """
    Demonstrate the effect of Batch Normalization.
    """
    print("=" * 60)
    print("Batch Normalization Demonstration")
    print("=" * 60)
    
    # Create sample data with different scales
    np.random.seed(42)
    batch_size = 32
    num_features = 5
    
    # Feature 1: Small values
    feature1 = np.random.randn(batch_size, 1) * 0.1
    # Feature 2: Large values
    feature2 = np.random.randn(batch_size, 1) * 100
    # Feature 3: Medium values
    feature3 = np.random.randn(batch_size, 1) * 10
    # Feature 4-5: Normal values
    feature4 = np.random.randn(batch_size, 1)
    feature5 = np.random.randn(batch_size, 1)
    
    x = np.concatenate([feature1, feature2, feature3, feature4, feature5], axis=1)
    
    print("\nOriginal data statistics:")
    print(f"Mean per feature: {np.mean(x, axis=0)}")
    print(f"Std per feature:  {np.std(x, axis=0)}")
    
    # Apply batch normalization
    bn = BatchNorm1dNumPy(num_features)
    x_normalized = bn.forward(x)
    
    print("\nAfter Batch Normalization:")
    print(f"Mean per feature: {np.mean(x_normalized, axis=0)}")
    print(f"Std per feature:  {np.std(x_normalized, axis=0)}")
    
    print("\nKey observations:")
    print("- All features now have mean ≈ 0 and std ≈ 1")
    print("- Features are on the same scale")
    print("- Gradients can flow more easily")


def compare_with_without_batchnorm():
    """
    Compare training behavior with and without BatchNorm.
    """
    print("\n" + "=" * 60)
    print("Comparison: With vs Without BatchNorm")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create dummy data
    x = torch.randn(64, 784)
    
    # Network without BatchNorm
    net_without = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Network with BatchNorm
    net_with = nn.Sequential(
        nn.Linear(784, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Forward pass
    with torch.no_grad():
        out_without = net_without(x)
        out_with = net_with(x)
    
    print("\nOutput statistics:")
    print(f"Without BatchNorm - Mean: {out_without.mean():.4f}, Std: {out_without.std():.4f}")
    print(f"With BatchNorm    - Mean: {out_with.mean():.4f}, Std: {out_with.std():.4f}")
    
    print("\nBenefits of Batch Normalization:")
    print("1. Faster convergence during training")
    print("2. Allows higher learning rates")
    print("3. Reduces sensitivity to initialization")
    print("4. Acts as a form of regularization")
    print("5. Reduces internal covariate shift")


if __name__ == "__main__":
    demonstrate_batch_norm()
    compare_with_without_batchnorm()
    
    print("\n" + "=" * 60)
    print("Additional Notes:")
    print("=" * 60)
    print("- BatchNorm normalizes over the batch dimension")
    print("- For Conv layers, use BatchNorm2d (normalizes per channel)")
    print("- For fully connected layers, use BatchNorm1d")
    print("- Remember to call model.eval() during inference!")
