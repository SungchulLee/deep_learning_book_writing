"""
Module 33.1: ResNet - Residual Networks (BEGINNER LEVEL)

This module introduces Residual Networks (ResNet), one of the most influential
architectures in deep learning. ResNet introduced skip connections that enable
training of very deep networks (100+ layers) by solving the vanishing gradient problem.

Key Concepts:
1. Residual blocks with skip connections
2. Identity mapping vs projection shortcuts
3. Deep network training with residuals
4. ResNet-18 and ResNet-34 architectures

Mathematical Foundation:
- Standard mapping: H(x) = F(x)
- Residual mapping: H(x) = F(x) + x
- Easier to learn F(x) = 0 than H(x) = x for identity

Paper: He et al., 2015 - "Deep Residual Learning for Image Recognition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


# ==============================================================================
# PART 1: BASIC RESIDUAL BLOCK
# ==============================================================================

class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet-18 and ResNet-34.
    
    Architecture:
        x --> [Conv 3x3] --> [BN] --> [ReLU] --> [Conv 3x3] --> [BN] --> (+) --> [ReLU]
        |                                                                    |
        |_________________ [identity or projection] ________________________|
    
    This block has two 3x3 convolutions with batch normalization and a skip connection.
    The skip connection can be:
    1. Identity: Direct connection when input/output dimensions match
    2. Projection: 1x1 conv when dimensions change (stride=2 or channels change)
    """
    
    # expansion is 1 for BasicBlock (output channels = input channels * expansion)
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for first conv layer (1 or 2)
            downsample (nn.Module): Projection shortcut when dimensions change
        """
        super(BasicBlock, self).__init__()
        
        # First convolutional layer
        # If stride=2, this reduces spatial dimensions by half
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,  # Padding=1 maintains spatial dimensions when stride=1
            bias=False  # No bias when using BatchNorm
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        # Always has stride=1 to maintain spatial dimensions
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample layer for skip connection when dimensions change
        # This is typically a 1x1 conv + BN to match dimensions
        self.downsample = downsample
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Forward pass through basic block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, H', W')
        """
        # Save input for skip connection
        identity = x
        
        # First conv block: Conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block: Conv -> BN (no ReLU yet)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply projection to identity if dimensions changed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection (this is the key innovation!)
        # Element-wise addition of residual and identity
        out += identity
        
        # Final ReLU after addition
        out = self.relu(out)
        
        return out


# ==============================================================================
# PART 2: BOTTLENECK BLOCK (for deeper ResNets)
# ==============================================================================

class BottleneckBlock(nn.Module):
    """
    Bottleneck Residual Block for ResNet-50, ResNet-101, ResNet-152.
    
    Architecture:
        x --> [Conv 1x1] --> [BN] --> [ReLU] --> [Conv 3x3] --> [BN] --> [ReLU] 
              --> [Conv 1x1] --> [BN] --> (+) --> [ReLU]
        |                                         |
        |_____________ [projection] _____________|
    
    This block uses 1x1 convolutions to reduce dimensions, then 3x3 conv,
    then 1x1 to expand back. This is more parameter-efficient for deep networks.
    
    Example: 256 input channels, 64 bottleneck channels, 256 output channels
    - 1x1 conv: 256 -> 64 (dimension reduction)
    - 3x3 conv: 64 -> 64 (main computation on reduced dimensions)
    - 1x1 conv: 64 -> 256 (dimension expansion)
    """
    
    # expansion factor (output channels = bottleneck channels * 4)
    expansion = 4
    
    def __init__(self, in_channels, bottleneck_channels, stride=1, downsample=None):
        """
        Args:
            in_channels (int): Number of input channels
            bottleneck_channels (int): Number of channels in bottleneck (reduced dimension)
            stride (int): Stride for 3x3 conv layer
            downsample (nn.Module): Projection shortcut when dimensions change
        """
        super(BottleneckBlock, self).__init__()
        
        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        # 3x3 conv on reduced dimensions
        self.conv2 = nn.Conv2d(
            bottleneck_channels, 
            bottleneck_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        # 1x1 conv to expand dimensions back
        out_channels = bottleneck_channels * self.expansion
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """Forward pass through bottleneck block."""
        identity = x
        
        # 1x1 reduce
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # 1x1 expand
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


# ==============================================================================
# PART 3: RESNET ARCHITECTURE
# ==============================================================================

class ResNet(nn.Module):
    """
    ResNet architecture for image classification.
    
    Architecture Overview:
    1. Initial conv layer: 7x7 conv with stride 2, reduces spatial dimensions
    2. Max pooling: Further spatial reduction
    3. Four residual stages: Each stage has multiple residual blocks
       - Stage 1: 64 channels
       - Stage 2: 128 channels (spatial dimension reduced by 2)
       - Stage 3: 256 channels (spatial dimension reduced by 2)
       - Stage 4: 512 channels (spatial dimension reduced by 2)
    4. Global average pooling: Reduces to single feature vector
    5. Fully connected layer: Maps to num_classes
    
    Popular Configurations:
    - ResNet-18: [2, 2, 2, 2] blocks with BasicBlock
    - ResNet-34: [3, 4, 6, 3] blocks with BasicBlock
    - ResNet-50: [3, 4, 6, 3] blocks with BottleneckBlock
    - ResNet-101: [3, 4, 23, 3] blocks with BottleneckBlock
    - ResNet-152: [3, 8, 36, 3] blocks with BottleneckBlock
    """
    
    def __init__(self, block, layers, num_classes=10, in_channels=3):
        """
        Args:
            block (nn.Module): Type of residual block (BasicBlock or BottleneckBlock)
            layers (list): Number of blocks in each stage [stage1, stage2, stage3, stage4]
            num_classes (int): Number of output classes
            in_channels (int): Number of input channels (3 for RGB images)
        """
        super(ResNet, self).__init__()
        
        self.in_channels = 64  # Initial channel size
        
        # Initial convolutional layer
        # For CIFAR-10 (32x32), we use smaller kernel and no maxpool
        # For ImageNet (224x224), use 7x7 kernel with stride 2 and maxpool
        self.conv1 = nn.Conv2d(
            in_channels, 
            64, 
            kernel_size=3,  # Use 3x3 for CIFAR-10 (7x7 for ImageNet)
            stride=1,        # Use stride=1 for CIFAR-10 (stride=2 for ImageNet)
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Maxpool layer (typically used for ImageNet, skip for CIFAR-10)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Four residual stages
        # Each stage has multiple residual blocks
        self.stage1 = self._make_stage(block, 64, layers[0], stride=1)
        self.stage2 = self._make_stage(block, 128, layers[1], stride=2)
        self.stage3 = self._make_stage(block, 256, layers[2], stride=2)
        self.stage4 = self._make_stage(block, 512, layers[3], stride=2)
        
        # Global average pooling: reduces (B, C, H, W) to (B, C, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for classification
        # Input size = 512 * block.expansion
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_stage(self, block, out_channels, num_blocks, stride):
        """
        Create a stage with multiple residual blocks.
        
        Args:
            block (nn.Module): Type of residual block
            out_channels (int): Number of output channels for this stage
            num_blocks (int): Number of blocks in this stage
            stride (int): Stride for first block (1 or 2)
            
        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        downsample = None
        
        # Create downsample layer if dimensions change
        # This happens when:
        # 1. Stride != 1 (spatial dimensions change)
        # 2. in_channels != out_channels * expansion (channel dimensions change)
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels * block.expansion,
                    kernel_size=1, 
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        # Create list of blocks
        layers = []
        
        # First block may have stride=2 and downsample
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks have stride=1 and no downsample
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialize weights using He initialization.
        
        He initialization is designed for ReLU activations and helps
        maintain variance of activations across layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BN weights to 1 and biases to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # He initialization for linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through ResNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 3, 32, 32) for CIFAR-10
            
        Returns:
            torch.Tensor: Logits of shape (batch, num_classes)
        """
        # Initial conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)  # Skip for CIFAR-10
        
        # Four residual stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten to (batch, features)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x


# ==============================================================================
# PART 4: FACTORY FUNCTIONS FOR STANDARD RESNET VARIANTS
# ==============================================================================

def resnet18(num_classes=10):
    """
    ResNet-18: 18 layers total
    - 1 initial conv + 4 stages * 4 layers + 1 FC = 18 layers
    - Architecture: [2, 2, 2, 2] BasicBlocks
    - Parameters: ~11M for ImageNet (or ~11M for CIFAR-10)
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=10):
    """
    ResNet-34: 34 layers total
    - Architecture: [3, 4, 6, 3] BasicBlocks
    - Parameters: ~21M for ImageNet
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=10):
    """
    ResNet-50: 50 layers total
    - Architecture: [3, 4, 6, 3] BottleneckBlocks
    - Parameters: ~25M for ImageNet
    """
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=10):
    """
    ResNet-101: 101 layers total
    - Architecture: [3, 4, 23, 3] BottleneckBlocks
    - Parameters: ~44M for ImageNet
    """
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes=10):
    """
    ResNet-152: 152 layers total
    - Architecture: [3, 8, 36, 3] BottleneckBlocks
    - Parameters: ~60M for ImageNet
    """
    return ResNet(BottleneckBlock, [3, 8, 36, 3], num_classes=num_classes)


# ==============================================================================
# PART 5: TRAINING UTILITIES
# ==============================================================================

def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    """
    Create CIFAR-10 dataloaders with standard augmentation.
    
    Data Augmentation for Training:
    1. Random horizontal flip: 50% chance
    2. Random crop: 32x32 with 4-pixel padding
    3. Normalization: per-channel mean/std from CIFAR-10
    
    Args:
        batch_size (int): Batch size for training and testing
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # CIFAR-10 statistics for normalization
    # Computed from entire training set
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    # Training transformations with data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Random crop with padding
        transforms.RandomHorizontalFlip(),      # Random horizontal flip
        transforms.ToTensor(),                  # Convert to tensor
        transforms.Normalize(mean, std)         # Normalize
    ])
    
    # Test transformations without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()  # Set model to training mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc='Training')
    
    for inputs, targets in pbar:
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test set.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to evaluate on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()  # Set model to evaluation mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # No gradient computation during evaluation
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


# ==============================================================================
# PART 6: MAIN TRAINING SCRIPT
# ==============================================================================

def main():
    """
    Main training script for ResNet on CIFAR-10.
    
    Training Configuration:
    - Model: ResNet-18
    - Dataset: CIFAR-10
    - Optimizer: SGD with momentum
    - Learning rate: 0.1 with cosine annealing
    - Batch size: 128
    - Epochs: 200
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    num_epochs = 200
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    
    # Create dataloaders
    print('Loading CIFAR-10 dataset...')
    train_loader, test_loader = get_cifar10_dataloaders(batch_size)
    
    # Create model
    print('Creating ResNet-18 model...')
    model = resnet18(num_classes=10)
    model = model.to(device)
    
    # Print model information
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params:,}')
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
        momentum=momentum, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler: cosine annealing
    # Gradually decreases learning rate following cosine curve
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training history
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # Training loop
    print('Starting training...')
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Print epoch results
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'resnet18_best.pth')
            print(f'✓ Best model saved with accuracy: {best_acc:.2f}%')
    
    print(f'\nTraining completed! Best test accuracy: {best_acc:.2f}%')
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Test Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('resnet18_training_curves.png')
    print('Training curves saved to resnet18_training_curves.png')


if __name__ == '__main__':
    main()


# ==============================================================================
# EXERCISES
# ==============================================================================

"""
BEGINNER EXERCISES:

1. Basic Training:
   - Run the training script and achieve >90% test accuracy on CIFAR-10
   - Monitor training curves and identify if model is overfitting or underfitting
   
2. Architecture Exploration:
   - Modify the code to train ResNet-34 instead of ResNet-18
   - Compare training time and final accuracy between ResNet-18 and ResNet-34
   
3. Skip Connection Analysis:
   - Remove skip connections (set out += identity to out = out)
   - Train the modified model and compare convergence speed
   - What happens to gradient flow without skip connections?
   
4. Hyperparameter Tuning:
   - Experiment with different batch sizes (64, 128, 256)
   - Try different learning rates (0.01, 0.1, 0.5)
   - Use different optimizers (Adam, RMSprop) instead of SGD
   
5. Visualization:
   - Visualize feature maps from different layers
   - Plot the distribution of gradients during training
   - Visualize which images the model gets wrong

EXPECTED RESULTS:
- ResNet-18 on CIFAR-10 should achieve 93-95% test accuracy
- Training should take 2-4 hours on a single GPU
- Loss should decrease smoothly without large spikes
"""
