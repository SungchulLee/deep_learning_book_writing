"""
Module 33.3: Inception/GoogLeNet - Multi-Scale Feature Extraction (INTERMEDIATE LEVEL)

This module covers Inception networks, which introduced the concept of multi-scale
feature extraction within a single layer using parallel convolution paths.

Key Concepts:
1. Inception modules with parallel multi-scale convolutions
2. 1x1 convolutions for dimensionality reduction
3. Global average pooling instead of fully connected layers
4. Auxiliary classifiers for training deep networks

Mathematical Foundation:
- Parallel paths: 1x1, 3x3, 5x5 convolutions and 3x3 max pooling
- 1x1 convolutions reduce computational cost
- Concatenate outputs along channel dimension

Paper: Szegedy et al., 2015 - "Going Deeper with Convolutions"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ==============================================================================
# PART 1: INCEPTION MODULE
# ==============================================================================

class InceptionModule(nn.Module):
    """
    Basic Inception module with four parallel paths.
    
    Architecture:
    Input
    ├── 1x1 conv (branch1)
    ├── 1x1 conv → 3x3 conv (branch2)
    ├── 1x1 conv → 5x5 conv (branch3)
    └── 3x3 maxpool → 1x1 conv (branch4)
    → Concatenate along channel dimension
    
    The 1x1 convolutions before 3x3 and 5x5 reduce dimensions to save computation.
    
    Example:
        Input: 192 channels
        Branch 1: 1x1 conv → 64 channels
        Branch 2: 1x1(96) → 3x3 conv → 128 channels  
        Branch 3: 1x1(16) → 5x5 conv → 32 channels
        Branch 4: maxpool → 1x1(32) → 32 channels
        Output: 64+128+32+32 = 256 channels
    """
    
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        """
        Args:
            in_channels (int): Number of input channels
            ch1x1 (int): Output channels for 1x1 branch
            ch3x3red (int): Reduction channels before 3x3
            ch3x3 (int): Output channels for 3x3 branch
            ch5x5red (int): Reduction channels before 5x5
            ch5x5 (int): Output channels for 5x5 branch
            pool_proj (int): Output channels for pooling branch
        """
        super(InceptionModule, self).__init__()
        
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x1 → 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 1x1 → 5x5 convolution
        # Note: 5x5 can be replaced with two 3x3 for efficiency
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 max pooling → 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass concatenates outputs from all branches.
        
        Args:
            x (torch.Tensor): Input of shape (B, C_in, H, W)
            
        Returns:
            torch.Tensor: Output of shape (B, C_out, H, W)
                         where C_out = ch1x1 + ch3x3 + ch5x5 + pool_proj
        """
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate along channel dimension
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# ==============================================================================
# PART 2: AUXILIARY CLASSIFIER
# ==============================================================================

class AuxiliaryClassifier(nn.Module):
    """
    Auxiliary classifier attached to intermediate layers.
    
    Purpose:
    - Combat vanishing gradients in very deep networks
    - Add gradient signal at intermediate layers during training
    - Discarded during inference
    
    Architecture:
    Input → AvgPool(5x5) → Conv1x1(128) → FC(1024) → Dropout → FC(num_classes)
    """
    
    def __init__(self, in_channels, num_classes):
        """
        Args:
            in_channels (int): Number of input channels
            num_classes (int): Number of output classes
        """
        super(AuxiliaryClassifier, self).__init__()
        
        # Average pooling to reduce spatial dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Convolution to reduce channels
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        """Forward pass through auxiliary classifier."""
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==============================================================================
# PART 3: GOOGLENET ARCHITECTURE
# ==============================================================================

class GoogLeNet(nn.Module):
    """
    GoogLeNet/Inception v1 architecture.
    
    Architecture Overview:
    1. Initial convolutions and max pooling
    2. Two inception modules (3a, 3b)
    3. Max pooling
    4. Five inception modules (4a-4e) with auxiliary classifier after 4a
    5. Max pooling  
    6. Two inception modules (5a, 5b) with auxiliary classifier after 5a
    7. Global average pooling
    8. Dropout and linear classifier
    
    Key Innovation: No fully connected layers except final classifier!
    This drastically reduces parameters (only 5M vs 60M in AlexNet).
    """
    
    def __init__(self, num_classes=10, aux_logits=True):
        """
        Args:
            num_classes (int): Number of output classes
            aux_logits (bool): Whether to use auxiliary classifiers during training
        """
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        
        # Initial layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception blocks
        # Format: InceptionModule(in_ch, 1x1, 3x3red, 3x3, 5x5red, 5x5, pool_proj)
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)  # Output: 256
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)  # Output: 480
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)  # Output: 512
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)  # Output: 512
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)  # Output: 512
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)  # Output: 528
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)  # Output: 832
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)  # Output: 832
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)  # Output: 1024
        
        # Auxiliary classifiers (used during training)
        if self.aux_logits:
            self.aux1 = AuxiliaryClassifier(512, num_classes)  # After inception4a
            self.aux2 = AuxiliaryClassifier(528, num_classes)  # After inception4d
        
        # Global average pooling and dropout
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Returns:
            If training with aux_logits: (main_output, aux1_output, aux2_output)
            If evaluating: main_output only
        """
        # Initial convolutions
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        # Inception 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        # Inception 4
        x = self.inception4a(x)
        
        # First auxiliary classifier
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        # Second auxiliary classifier
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        # Inception 5
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # Global average pooling and classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        # Return outputs
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x


# ==============================================================================
# PART 4: TRAINING WITH AUXILIARY LOSSES
# ==============================================================================

def train_epoch_with_aux(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch with auxiliary classifiers.
    
    Total loss = main_loss + 0.3 * aux1_loss + 0.3 * aux2_loss
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass returns (main, aux1, aux2)
        outputs, aux1, aux2 = model(inputs)
        
        # Compute losses
        loss_main = criterion(outputs, targets)
        loss_aux1 = criterion(aux1, targets)
        loss_aux2 = criterion(aux2, targets)
        
        # Weighted combination
        loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/len(pbar):.3f}',
                         'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model (no auxiliary outputs during eval)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(test_loader), 100. * correct / total


# ==============================================================================
# PART 5: MAIN TRAINING SCRIPT
# ==============================================================================

def main():
    """Main training script for GoogLeNet on CIFAR-10."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    num_epochs = 100
    batch_size = 128
    learning_rate = 0.01
    
    # Load data
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model
    print('Creating GoogLeNet...')
    model = GoogLeNet(num_classes=10, aux_logits=True)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params:,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss, train_acc = train_epoch_with_aux(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'googlenet_best.pth')
            print(f'✓ Best model saved: {best_acc:.2f}%')
    
    print(f'\nBest test accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()


# ==============================================================================
# EXERCISES
# ==============================================================================

"""
INTERMEDIATE EXERCISES:

1. Inception Module Analysis:
   - Visualize the outputs of each branch in an Inception module
   - Compare computational cost with/without 1x1 bottleneck convolutions
   - Measure the importance of each branch using ablation studies

2. Auxiliary Classifier Study:
   - Train with and without auxiliary classifiers
   - Compare convergence speed and final accuracy
   - Try different weighting schemes (0.1, 0.3, 0.5)

3. Multi-Scale Feature Extraction:
   - Extract features at different scales from Inception modules
   - Visualize what patterns each scale captures
   - Apply on different image types (faces, objects, scenes)

4. Architecture Variants:
   - Implement Inception v2 with factorized convolutions
   - Replace 5x5 convs with two 3x3 convs
   - Compare efficiency and accuracy

5. Comparison with ResNet:
   - Train GoogLeNet and ResNet-34 (similar depth)
   - Compare parameters, FLOPs, accuracy, training time
   - Analyze which is more parameter-efficient

EXPECTED RESULTS:
- GoogLeNet should achieve 92-94% on CIFAR-10
- Training takes 3-4 hours on GPU
- Much fewer parameters than VGG despite similar depth
- Auxiliary classifiers improve convergence in early epochs
"""
