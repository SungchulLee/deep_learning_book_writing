"""
Module 33.8: Transfer Learning with Pretrained Models (ADVANCED)

Transfer learning leverages knowledge from models pretrained on large datasets (like ImageNet)
to improve performance on new tasks with limited data. This is one of the most powerful
techniques in practical deep learning.

Key Concepts:
1. Feature extraction vs fine-tuning
2. Layer freezing strategies
3. Learning rate scheduling for transfer learning
4. Domain adaptation techniques
5. When and how to use pretrained models

Learning Objectives:
- Understand when transfer learning helps
- Master feature extraction and fine-tuning approaches
- Learn optimal hyperparameter settings for transfer
- Apply transfer learning to custom datasets

Paper References:
- Yosinski et al., 2014 - "How transferable are features in deep neural networks?"
- Kornblith et al., 2019 - "Do Better ImageNet Models Transfer Better?"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image


# ==============================================================================
# PART 1: LOADING PRETRAINED MODELS
# ==============================================================================

def load_pretrained_model(model_name='resnet50', num_classes=10, pretrained=True):
    """
    Load a pretrained model and modify it for new task.
    
    Common pretrained models from torchvision:
    - ResNet family: resnet18, resnet34, resnet50, resnet101, resnet152
    - VGG family: vgg11, vgg13, vgg16, vgg19 (with/without batch norm)
    - MobileNet family: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
    - EfficientNet family: efficientnet_b0 through efficientnet_b7
    - Vision Transformers: vit_b_16, vit_b_32, vit_l_16
    
    Args:
        model_name: Name of pretrained model
        num_classes: Number of classes in new task
        pretrained: Whether to load ImageNet pretrained weights
        
    Returns:
        Modified model with new classifier head
    """
    print(f'\nLoading {model_name}...')
    
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        # Replace final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'vgg16':
        model = models.vgg16_bn(pretrained=pretrained)
        # VGG has classifier as Sequential module
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        # MobileNet classifier is in different location
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
    else:
        raise ValueError(f'Unknown model: {model_name}')
    
    if pretrained:
        print(f'✓ Loaded pretrained {model_name} weights from ImageNet')
    else:
        print(f'Initialized {model_name} from scratch')
    
    return model


# ==============================================================================
# PART 2: LAYER FREEZING STRATEGIES
# ==============================================================================

def freeze_layers(model, freeze_strategy='all'):
    """
    Freeze layers of pretrained model.
    
    Freezing prevents parameters from being updated during training.
    This is useful for:
    1. Preserving general features learned on ImageNet
    2. Reducing computation during training
    3. Preventing overfitting with limited data
    
    Common Strategies:
    - 'all': Freeze all layers except final classifier
    - 'partial': Freeze early layers, train later layers
    - 'none': Train all layers (full fine-tuning)
    
    Args:
        model: PyTorch model
        freeze_strategy: Which layers to freeze
        
    Returns:
        model: Model with frozen layers
        num_frozen: Number of frozen parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    if freeze_strategy == 'all':
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier head (last layer)
        if hasattr(model, 'fc'):  # ResNet
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):  # VGG, MobileNet, EfficientNet
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f'Frozen {frozen_params:,} / {total_params:,} parameters ({100*frozen_params/total_params:.1f}%)')
        print('Strategy: Feature extraction (only train classifier)')
    
    elif freeze_strategy == 'partial':
        # Freeze first half of layers
        all_modules = list(model.children())
        num_to_freeze = len(all_modules) // 2
        
        for i, module in enumerate(all_modules):
            if i < num_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
        
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f'Frozen {frozen_params:,} / {total_params:,} parameters ({100*frozen_params/total_params:.1f}%)')
        print('Strategy: Partial fine-tuning (train later layers)')
    
    elif freeze_strategy == 'none':
        # Don't freeze anything
        for param in model.parameters():
            param.requires_grad = True
        print(f'Training all {total_params:,} parameters')
        print('Strategy: Full fine-tuning')
    
    else:
        raise ValueError(f'Unknown freeze strategy: {freeze_strategy}')
    
    return model


def unfreeze_layers_gradually(model, epoch, total_epochs, strategy='linear'):
    """
    Gradually unfreeze layers during training.
    
    Progressive unfreezing can help fine-tuning by:
    1. Starting with stable features in early layers
    2. Gradually adapting to new domain
    3. Reducing risk of catastrophic forgetting
    
    Args:
        model: PyTorch model
        epoch: Current epoch number
        total_epochs: Total training epochs
        strategy: How to unfreeze ('linear', 'exponential')
    """
    if strategy == 'linear':
        # Unfreeze layers linearly over epochs
        fraction_complete = epoch / total_epochs
        
        all_layers = [m for m in model.modules() if len(list(m.children())) == 0]
        num_to_unfreeze = int(len(all_layers) * fraction_complete)
        
        for i, layer in enumerate(all_layers):
            if i < num_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True


# ==============================================================================
# PART 3: LEARNING RATE STRATEGIES
# ==============================================================================

def get_transfer_learning_optimizer(model, strategy='different_lr', base_lr=0.001):
    """
    Create optimizer with differential learning rates for transfer learning.
    
    Key Principle: Earlier layers learn more general features, later layers
    learn more task-specific features. Therefore:
    - Early layers: Small learning rate (preserve general features)
    - Middle layers: Medium learning rate (adapt features)
    - Final classifier: Large learning rate (learn new task)
    
    Args:
        model: PyTorch model
        strategy: Learning rate strategy
        base_lr: Base learning rate
        
    Returns:
        optimizer: Configured optimizer
    """
    if strategy == 'same_lr':
        # Same learning rate for all parameters
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        print(f'Using single learning rate: {base_lr}')
    
    elif strategy == 'different_lr':
        # Different learning rates for different layers
        # This is the recommended approach for transfer learning!
        
        # Separate parameters into groups
        feature_params = []
        classifier_params = []
        
        if hasattr(model, 'fc'):  # ResNet
            feature_params = [p for n, p in model.named_parameters() if 'fc' not in n]
            classifier_params = [p for n, p in model.named_parameters() if 'fc' in n]
        elif hasattr(model, 'classifier'):  # VGG, MobileNet
            feature_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
            classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
        
        # Use smaller learning rate for pretrained features
        optimizer = optim.Adam([
            {'params': feature_params, 'lr': base_lr / 10},  # 10x smaller for features
            {'params': classifier_params, 'lr': base_lr}      # Regular lr for classifier
        ])
        
        print(f'Using differential learning rates:')
        print(f'  - Features: {base_lr/10:.6f}')
        print(f'  - Classifier: {base_lr:.6f}')
    
    elif strategy == 'discriminative':
        # Gradually increasing learning rates from bottom to top
        # Bottom layers: smallest lr, top layers: largest lr
        
        param_groups = []
        num_groups = 5  # Divide model into 5 groups
        all_params = list(model.parameters())
        group_size = len(all_params) // num_groups
        
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < num_groups - 1 else len(all_params)
            group_params = all_params[start_idx:end_idx]
            
            # Learning rate increases linearly with layer depth
            lr = base_lr * (i + 1) / num_groups
            param_groups.append({'params': group_params, 'lr': lr})
        
        optimizer = optim.Adam(param_groups)
        print(f'Using discriminative learning rates (5 groups)')
    
    return optimizer


# ==============================================================================
# PART 4: DATA AUGMENTATION FOR TRANSFER LEARNING
# ==============================================================================

def get_transfer_transforms(input_size=224, augmentation_strength='medium'):
    """
    Get transforms appropriate for transfer learning.
    
    ImageNet pretrained models expect:
    - Input size: 224x224 (or 299x299 for Inception)
    - Normalization: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - RGB color format
    
    Args:
        input_size: Expected input size (224 for most models)
        augmentation_strength: 'light', 'medium', 'strong'
        
    Returns:
        tuple: (train_transform, test_transform)
    """
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if augmentation_strength == 'light':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    elif augmentation_strength == 'medium':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    elif augmentation_strength == 'strong':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return train_transform, test_transform


# ==============================================================================
# PART 5: TRAINING PIPELINE
# ==============================================================================

class TransferLearningTrainer:
    """
    Trainer specifically designed for transfer learning scenarios.
    
    Implements best practices:
    - Differential learning rates
    - Gradual unfreezing
    - Early stopping
    - Model checkpointing
    """
    
    def __init__(self, model, model_name, device='cuda'):
        """
        Args:
            model: PyTorch model
            model_name: String identifier
            device: Device to train on
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self.best_acc = 0.0
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return running_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader, criterion):
        """Validate model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return running_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader, num_epochs, 
             learning_rate=0.001, patience=10):
        """
        Train with transfer learning best practices.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            learning_rate: Learning rate (base)
            patience: Early stopping patience
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = get_transfer_learning_optimizer(self.model, 'different_lr', learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Early stopping
        epochs_no_improve = 0
        
        print(f'\nTraining {self.model_name}...')
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print results
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), f'{self.model_name}_best.pth')
                print(f'✓ Best model saved: {self.best_acc:.2f}%')
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
        
        print(f'\nTraining completed! Best validation accuracy: {self.best_acc:.2f}%')
        return self.history


# ==============================================================================
# PART 6: MAIN TRANSFER LEARNING SCRIPT
# ==============================================================================

def main():
    """
    Main transfer learning demonstration.
    
    This example shows transfer learning on CIFAR-10.
    In practice, you would use this for a custom dataset.
    """
    print('='*80)
    print('TRANSFER LEARNING DEMONSTRATION')
    print('='*80)
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    
    # Hyperparameters
    batch_size = 32  # Smaller batch size due to larger input size
    num_epochs = 50
    learning_rate = 0.001
    
    # Get transforms (ImageNet normalization)
    train_transform, test_transform = get_transfer_transforms(224, 'medium')
    
    # Load CIFAR-10 with transfer learning transforms
    print('\nLoading CIFAR-10...')
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Compare different strategies
    strategies = [
        ('resnet18_feature_extraction', 'all'),
        ('resnet18_fine_tuning', 'none'),
        ('resnet18_partial', 'partial')
    ]
    
    results = {}
    
    for name, freeze_strategy in strategies:
        print(f'\n{"="*80}')
        print(f'Strategy: {name}')
        print(f'{"="*80}')
        
        # Load pretrained model
        model = load_pretrained_model('resnet18', num_classes=10, pretrained=True)
        
        # Apply freezing strategy
        model = freeze_layers(model, freeze_strategy)
        
        # Train
        trainer = TransferLearningTrainer(model, name, device)
        history = trainer.train(train_loader, test_loader, num_epochs, learning_rate)
        
        results[name] = {
            'history': history,
            'best_acc': trainer.best_acc
        }
    
    # Compare results
    print(f'\n{"="*80}')
    print('COMPARISON OF TRANSFER LEARNING STRATEGIES')
    print(f'{"="*80}')
    for name, result in results.items():
        print(f'{name:40s}: {result["best_acc"]:.2f}% best validation accuracy')
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['history']['val_acc'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Transfer Learning Strategy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(result['history']['val_loss'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transfer_learning_comparison.png', dpi=150)
    print('\nComparison plot saved to transfer_learning_comparison.png')


if __name__ == '__main__':
    main()


# ==============================================================================
# EXERCISES
# ==============================================================================

"""
ADVANCED EXERCISES:

1. Custom Dataset Transfer Learning:
   - Create a custom dataset (e.g., cats vs dogs, flowers, food)
   - Apply transfer learning with different pretrained models
   - Compare results with training from scratch
   
2. Domain Adaptation Study:
   - Transfer from ImageNet to CIFAR-10, CIFAR-100, STL-10
   - Measure transferability across different domains
   - Identify which layers transfer best

3. Few-Shot Learning:
   - Use transfer learning with limited data (10, 50, 100 examples per class)
   - Compare different freezing strategies
   - Plot accuracy vs training set size

4. Fine-tuning Schedule Optimization:
   - Implement gradual unfreezing schedules
   - Compare different learning rate strategies
   - Find optimal schedule for your dataset

5. Multi-Source Transfer:
   - Load weights from multiple pretrained models
   - Ensemble predictions
   - Compare with single-model transfer

EXPECTED RESULTS:
- Feature extraction (freeze all): 85-90% on CIFAR-10, trains fast
- Full fine-tuning: 92-95% on CIFAR-10, slower, may overfit with small data
- Partial fine-tuning: 90-93%, good balance
- Transfer learning should significantly outperform training from scratch,
  especially with limited data (>10% accuracy improvement typical)
"""
