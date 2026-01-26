# CNN for Image Classification

## Introduction

Image classification is the canonical application of CNNs: given an input image, predict which class it belongs to. This section provides a comprehensive guide to building, training, and evaluating CNN classifiers, drawing on best practices from both theory and practical implementations.

## The Image Classification Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│   Input     │     │   Feature    │     │    Feature     │     │   Output     │
│   Image     │ ──▶ │  Extraction  │ ──▶ │   Aggregation  │ ──▶ │  (Classes)   │
│  (H×W×C)    │     │  (Conv+Pool) │     │   (FC/GAP)     │     │  (softmax)   │
└─────────────┘     └──────────────┘     └────────────────┘     └──────────────┘
```

## Data Preparation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Validation/test transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# CIFAR-10 example
train_dataset = datasets.CIFAR10('./data', train=True, download=True, 
                                  transform=train_transform)
val_dataset = datasets.CIFAR10('./data', train=False, download=True,
                                transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
```

## Model Architectures

### Classic Architecture: LeNet-5

The original CNN for digit recognition (LeCun, 1998):

```python
class LeNet5(nn.Module):
    """
    LeNet-5: Classic CNN architecture for MNIST.
    
    Architecture:
        Conv(1→6, 5×5) → Pool → Conv(6→16, 5×5) → Pool → FC(400→120) → FC(120→84) → FC(84→10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),      # 28×28 → 24×24
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 24×24 → 12×12
            nn.Conv2d(6, 16, kernel_size=5),     # 12×12 → 8×8
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 8×8 → 4×4
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### Simple CNN for CIFAR-10

```python
class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification.
    Architecture: Conv blocks → Global Average Pooling → Classifier
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 32×32 → 16×16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 16×16 → 8×8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 8×8 → 4×4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=10)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Modern Architecture: ResNet-Style

ResNet introduced skip connections to enable very deep networks:

```python
class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # Skip connection
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture for image classification."""
    
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 
                         1, stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
```

### ResNet for CIFAR (Smaller Input)

```python
class ResNetCIFAR(nn.Module):
    """ResNet variant for CIFAR-10/100 (32×32 images)."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        # No aggressive downsampling for small images
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)   # 32×32
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 16×16
        self.layer3 = self._make_layer(128, 256, 2, stride=2) # 8×8
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [self._residual_block(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(self._residual_block(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    
    def _residual_block(self, in_ch, out_ch, stride):
        return ResidualBlock(in_ch, out_ch, stride)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
```

## Transfer Learning

Using pretrained models for new tasks:

```python
import torchvision.models as models

def create_transfer_model(num_classes, freeze_features=True):
    """
    Create a model with transfer learning.
    
    Args:
        num_classes: Number of output classes
        freeze_features: Freeze feature extractor weights
    """
    # Load pretrained ResNet-50
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # Freeze feature extractor
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model


# Example usage
model = create_transfer_model(num_classes=10, freeze_features=True)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / Total: {total:,}")
```

## Training Loop

```python
def train_epoch(model, loader, criterion, optimizer, device):
    """Single training epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, 100. * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return total_loss / total, 100. * correct / total


def train_classifier(model, train_loader, val_loader, epochs=100, lr=0.1):
    """Complete training loop for image classification."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    return best_acc
```

## Advanced Training Techniques

### Data Augmentation

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# Strong augmentation for better generalization
strong_augment = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    AutoAugment(AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25),
])
```

### Learning Rate Schedule with Warmup

```python
import math

def get_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs):
    """Warmup + Cosine Annealing schedule."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Label Smoothing

```python
class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing for regularization."""
    
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)
        
        one_hot = torch.full_like(pred, smooth_value)
        one_hot.scatter_(1, target.unsqueeze(1), confidence)
        
        return (-one_hot * F.log_softmax(pred, dim=1)).sum(dim=1).mean()
```

## Evaluation Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def detailed_evaluation(model, loader, class_names, device):
    """Comprehensive model evaluation with per-class metrics."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
```

## Common Datasets and Benchmarks

| Dataset | Images | Classes | Image Size | Task |
|---------|--------|---------|------------|------|
| MNIST | 70,000 | 10 | 28×28 | Digit recognition |
| Fashion-MNIST | 70,000 | 10 | 28×28 | Clothing classification |
| CIFAR-10 | 60,000 | 10 | 32×32 | Object classification |
| CIFAR-100 | 60,000 | 100 | 32×32 | Fine-grained |
| ImageNet | 1.2M | 1000 | ~256×256 | General objects |

## Expected Results

| Dataset | Model | Test Accuracy |
|---------|-------|---------------|
| MNIST | LeNet-5 | ~99% |
| Fashion-MNIST | Simple CNN | ~92% |
| CIFAR-10 | ResNet-18 | ~93-95% |
| CIFAR-100 | ResNet-50 | ~78-80% |
| ImageNet | ResNet-50 | ~76% (top-1) |

## Best Practices

### Architecture Selection by Dataset Size

| Dataset Size | Recommended Approach |
|--------------|---------------------|
| < 1,000 | Transfer learning, heavy augmentation |
| 1,000 - 10,000 | Transfer learning, fine-tune last layers |
| 10,000 - 100,000 | Transfer learning or train from scratch |
| > 100,000 | Train from scratch with modern architectures |

### Training Tips

1. **Start with pretrained models** when possible
2. **Use appropriate augmentation** for dataset size
3. **Learning rate warmup** for training from scratch
4. **Label smoothing** reduces overconfidence
5. **Early stopping** based on validation accuracy
6. **Cosine annealing** for learning rate schedule

## Key Takeaways

1. **CNN architecture evolution**: LeNet → VGG → ResNet → EfficientNet
2. **Skip connections** enable very deep networks (100+ layers)
3. **Transfer learning** is the default starting point for limited data
4. **Data augmentation** is critical for generalization
5. **Global Average Pooling** reduces parameters vs FC layers
6. **Batch normalization** stabilizes training and acts as regularization

## References

1. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.

2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *NeurIPS*.

3. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*.

4. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.

5. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
