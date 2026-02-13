"""
Module 33.2: VGG - Very Deep Convolutional Networks (INTERMEDIATE LEVEL)

This module covers VGG networks, which demonstrated that depth matters when using
small (3x3) convolutional filters. VGG's simple and uniform architecture made it
very popular despite being computationally expensive.

Key Concepts:
1. Stacking small 3x3 convolutions instead of large filters
2. Deep uniform architecture (repeated patterns)
3. VGG-16 and VGG-19 configurations
4. Feature visualization techniques

Mathematical Foundation:
- Two 3x3 convs have same receptive field as one 5x5 conv
- Three 3x3 convs have same receptive field as one 7x7 conv
- But with fewer parameters and more nonlinearity!

Paper: Simonyan & Zisserman, 2014 - "Very Deep Convolutional Networks for Large Scale Image Recognition"
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
import torch.nn.functional as F


# ==============================================================================
# PART 1: VGG ARCHITECTURE
# ==============================================================================

class VGG(nn.Module):
    """
    VGG network architecture.
    
    Key Design Principles:
    1. All convolutions are 3x3 with stride 1, padding 1
    2. All max pooling is 2x2 with stride 2
    3. Channels double after each max pooling (64 -> 128 -> 256 -> 512 -> 512)
    4. Three fully connected layers at the end (4096 -> 4096 -> num_classes)
    5. ReLU activation after every conv and FC layer
    6. Dropout for regularization in FC layers
    
    Why 3x3 Convolutions?
    - Two 3x3 convs have effective receptive field of 5x5
    - Three 3x3 convs have effective receptive field of 7x7
    - But fewer parameters: 3*(3²*C²) vs 1*(7²*C²) = 27C² vs 49C²
    - More nonlinearity from additional ReLU layers
    
    Architecture (example VGG-16):
    [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    where numbers are conv channels and 'M' is max pooling
    """
    
    def __init__(self, features, num_classes=10, init_weights=True, dropout=0.5):
        """
        Args:
            features (nn.Sequential): Convolutional feature extractor
            num_classes (int): Number of output classes
            init_weights (bool): Whether to initialize weights
            dropout (float): Dropout probability for FC layers
        """
        super(VGG, self).__init__()
        
        # Feature extraction layers (conv + pooling)
        self.features = features
        
        # Adaptive average pooling to handle different input sizes
        # Outputs fixed 7x7 feature maps regardless of input size
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier layers (fully connected)
        # For CIFAR-10, we use smaller FC layers than original ImageNet version
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # First FC layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),         # Dropout for regularization
            
            nn.Linear(4096, 4096),         # Second FC layer
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(4096, num_classes)   # Output layer
        )
        
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights using Xavier initialization for conv layers
        and He initialization for linear layers.
        
        Weight initialization is crucial for training deep networks.
        Poor initialization can lead to vanishing/exploding gradients.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier (Glorot) initialization for conv layers
                # Designed to maintain variance across layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BN parameters
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Normal initialization for linear layers
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through VGG network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 3, H, W)
            
        Returns:
            torch.Tensor: Output logits of shape (batch, num_classes)
        """
        # Feature extraction
        x = self.features(x)
        
        # Adaptive pooling
        x = self.avgpool(x)
        
        # Flatten for FC layers
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def extract_features(self, x, layer_idx=None):
        """
        Extract intermediate features for visualization.
        
        Args:
            x (torch.Tensor): Input tensor
            layer_idx (int): Index of layer to extract features from
                            If None, return all layer outputs
                            
        Returns:
            torch.Tensor or list: Features from specified layer or all layers
        """
        if layer_idx is None:
            # Extract features from all layers
            features = []
            for layer in self.features:
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    features.append(x.detach())
            return features
        else:
            # Extract features from specific layer
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == layer_idx:
                    return x.detach()
            return x.detach()


def make_vgg_layers(cfg, batch_norm=False, in_channels=3):
    """
    Create VGG feature extraction layers from configuration.
    
    This function converts a simple list configuration into a sequence
    of convolutional and pooling layers.
    
    Args:
        cfg (list): Configuration list where integers are output channels
                   and 'M' indicates max pooling
        batch_norm (bool): Whether to add batch normalization layers
        in_channels (int): Number of input channels
        
    Returns:
        nn.Sequential: Sequential container of layers
    """
    layers = []
    
    for v in cfg:
        if v == 'M':
            # Max pooling layer
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            # Convolutional block
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            
            if batch_norm:
                # Conv -> BN -> ReLU
                layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                # Conv -> ReLU
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            
            # Update input channels for next layer
            in_channels = v
    
    return nn.Sequential(*layers)


# ==============================================================================
# PART 2: VGG CONFIGURATIONS
# ==============================================================================

# Configuration dictionary for different VGG variants
# Numbers represent output channels, 'M' represents max pooling
VGG_CONFIGS = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(num_classes=10, batch_norm=False):
    """
    VGG-11 model (8 conv layers + 3 FC layers = 11 weight layers).
    
    Architecture:
    - 1 conv layer with 64 channels
    - 1 conv layer with 128 channels
    - 2 conv layers with 256 channels
    - 2 conv layers with 512 channels
    - 2 conv layers with 512 channels
    
    Parameters: ~132M (for ImageNet)
    """
    features = make_vgg_layers(VGG_CONFIGS['VGG11'], batch_norm=batch_norm)
    return VGG(features, num_classes=num_classes)


def vgg13(num_classes=10, batch_norm=False):
    """
    VGG-13 model (10 conv layers + 3 FC layers = 13 weight layers).
    
    Parameters: ~133M (for ImageNet)
    """
    features = make_vgg_layers(VGG_CONFIGS['VGG13'], batch_norm=batch_norm)
    return VGG(features, num_classes=num_classes)


def vgg16(num_classes=10, batch_norm=False):
    """
    VGG-16 model (13 conv layers + 3 FC layers = 16 weight layers).
    
    This is the most popular VGG variant used for transfer learning.
    
    Architecture:
    - 2 conv layers with 64 channels
    - 2 conv layers with 128 channels
    - 3 conv layers with 256 channels
    - 3 conv layers with 512 channels
    - 3 conv layers with 512 channels
    
    Parameters: ~138M (for ImageNet)
    """
    features = make_vgg_layers(VGG_CONFIGS['VGG16'], batch_norm=batch_norm)
    return VGG(features, num_classes=num_classes)


def vgg19(num_classes=10, batch_norm=False):
    """
    VGG-19 model (16 conv layers + 3 FC layers = 19 weight layers).
    
    Deepest VGG variant, often used for style transfer.
    
    Parameters: ~144M (for ImageNet)
    """
    features = make_vgg_layers(VGG_CONFIGS['VGG19'], batch_norm=batch_norm)
    return VGG(features, num_classes=num_classes)


# ==============================================================================
# PART 3: FEATURE VISUALIZATION
# ==============================================================================

def visualize_filters(model, layer_idx=0, num_filters=16):
    """
    Visualize convolutional filters from a specific layer.
    
    Filters in early layers often detect edges, colors, textures.
    Filters in deeper layers detect more complex patterns.
    
    Args:
        model (VGG): VGG model
        layer_idx (int): Index of conv layer to visualize
        num_filters (int): Number of filters to display
    """
    # Get the conv layer
    conv_layers = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
    conv_layer = conv_layers[layer_idx]
    
    # Get weights: shape (out_channels, in_channels, height, width)
    weights = conv_layer.weight.data.cpu().numpy()
    
    # Normalize weights to [0, 1] for visualization
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    
    # Create subplot grid
    num_cols = 8
    num_rows = (num_filters + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 2*num_rows))
    axes = axes.flatten()
    
    for i in range(min(num_filters, weights.shape[0])):
        # Take first input channel for visualization
        filter_img = weights[i, 0, :, :]
        axes[i].imshow(filter_img, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'F{i}', fontsize=8)
    
    # Hide unused subplots
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Layer {layer_idx} Filters (3x3)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'vgg_filters_layer{layer_idx}.png', dpi=150)
    print(f'Filters saved to vgg_filters_layer{layer_idx}.png')


def visualize_feature_maps(model, image, layer_idx=0, num_maps=16):
    """
    Visualize feature maps (activations) for a given input image.
    
    Feature maps show what patterns the network detects in the image.
    
    Args:
        model (VGG): VGG model
        image (torch.Tensor): Input image (1, C, H, W)
        layer_idx (int): Index of conv layer to visualize
        num_maps (int): Number of feature maps to display
    """
    model.eval()
    
    # Extract features from specified layer
    features = model.extract_features(image, layer_idx=layer_idx)
    features = features.squeeze(0).cpu().numpy()  # Remove batch dimension
    
    # Normalize for visualization
    features = (features - features.min()) / (features.max() - features.min())
    
    # Create subplot grid
    num_cols = 8
    num_rows = (num_maps + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 2*num_rows))
    axes = axes.flatten()
    
    for i in range(min(num_maps, features.shape[0])):
        axes[i].imshow(features[i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Map {i}', fontsize=8)
    
    # Hide unused subplots
    for i in range(num_maps, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Feature Maps from Layer {layer_idx}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'vgg_feature_maps_layer{layer_idx}.png', dpi=150)
    print(f'Feature maps saved to vgg_feature_maps_layer{layer_idx}.png')


# ==============================================================================
# PART 4: TRAINING UTILITIES
# ==============================================================================

def get_dataloaders(batch_size=64, augmentation=True):
    """
    Create CIFAR-10 dataloaders.
    
    Args:
        batch_size (int): Batch size
        augmentation (bool): Whether to use data augmentation
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    if augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform_train = transforms.Compose([
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
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
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
    """Evaluate model."""
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
    """
    Main training script for VGG on CIFAR-10.
    
    Training Configuration:
    - Model: VGG-16 with Batch Normalization
    - Dataset: CIFAR-10
    - Optimizer: Adam (easier to train than SGD for VGG)
    - Learning rate: 0.001 with step decay
    - Batch size: 64 (VGG is memory-intensive)
    - Epochs: 100
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    num_epochs = 100
    batch_size = 64  # Smaller batch size due to large model
    learning_rate = 0.001
    weight_decay = 5e-4
    
    # Create dataloaders
    print('Loading CIFAR-10 dataset...')
    train_loader, test_loader = get_dataloaders(batch_size)
    
    # Create model with batch normalization
    print('Creating VGG-16 with Batch Normalization...')
    model = vgg16(num_classes=10, batch_norm=True)
    model = model.to(device)
    
    # Print model information
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params:,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training history
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    best_acc = 0.0
    
    print('Starting training...')
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'vgg16_best.pth')
            print(f'✓ Best model saved with accuracy: {best_acc:.2f}%')
    
    print(f'\nTraining completed! Best test accuracy: {best_acc:.2f}%')
    
    # Visualize some filters and feature maps
    print('\nGenerating visualizations...')
    
    # Load best model
    model.load_state_dict(torch.load('vgg16_best.pth'))
    
    # Visualize filters from first layer
    visualize_filters(model, layer_idx=0, num_filters=64)
    
    # Visualize feature maps for a sample image
    test_images, _ = next(iter(test_loader))
    sample_image = test_images[0:1].to(device)
    visualize_feature_maps(model, sample_image, layer_idx=5, num_maps=64)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('vgg16_training_curves.png')
    print('Training curves saved to vgg16_training_curves.png')


if __name__ == '__main__':
    main()


# ==============================================================================
# EXERCISES
# ==============================================================================

"""
INTERMEDIATE EXERCISES:

1. Architecture Comparison:
   - Train VGG-11, VGG-16, and VGG-19 on CIFAR-10
   - Compare accuracy, training time, and memory usage
   - Plot the results in a comparative table
   
2. Batch Normalization Impact:
   - Train VGG-16 with and without batch normalization
   - Compare convergence speed and final accuracy
   - Analyze training stability

3. Receptive Field Analysis:
   - Calculate theoretical receptive field for each layer
   - Verify empirically using gradient-based methods
   - Visualize what each layer "sees"

4. Feature Visualization:
   - Extract and visualize features from all conv layers
   - Observe how features become more abstract in deeper layers
   - Apply t-SNE to visualize learned representations

5. Filter Analysis:
   - Visualize filters from first, middle, and last conv layers
   - Identify what patterns different filters detect
   - Analyze filter redundancy

EXPECTED RESULTS:
- VGG-16 with BN should achieve 91-93% on CIFAR-10
- Training takes 3-5 hours on a single GPU
- Model is memory-intensive (>500MB)
- Early layers detect edges/colors, deeper layers detect textures/patterns
"""
