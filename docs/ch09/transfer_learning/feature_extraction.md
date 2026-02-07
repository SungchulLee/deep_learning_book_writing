# Feature Extraction

## Learning Objectives

By the end of this section, you will:

- Understand feature extraction as a transfer learning strategy
- Implement feature extraction using pre-trained CNNs
- Know how to properly freeze layers and replace classifiers
- Analyze feature representations from different network depths
- Apply feature extraction to classify images with limited data

## 1. Feature Extraction Strategy

Feature extraction is the simplest and often most effective form of transfer learning. The core idea is elegant: use a pre-trained network as a fixed feature extractor and only train a new classifier on top.

### 1.1 Conceptual Overview

```
Pre-trained Network (e.g., ResNet-18 trained on ImageNet)

Input Image (224×224×3)
        │
        ▼
┌───────────────────┐
│   Conv Layers     │ ← FROZEN (weights fixed)
│   (Feature        │   These layers have learned to extract
│    Extractor)     │   universal visual features from ImageNet
└───────────────────┘
        │
        ▼
    Feature Vector
      (512-dim)
        │
        ▼
┌───────────────────┐
│  New Classifier   │ ← TRAINABLE (randomly initialized)
│   (FC Layer)      │   Only this layer learns task-specific
│                   │   classification boundaries
└───────────────────┘
        │
        ▼
   Class Predictions
    (num_classes)
```

### 1.2 Why Feature Extraction Works

The feature extraction approach works because:

1. **Universal Features**: Convolutional layers learn features (edges, textures, shapes) that are useful across many visual tasks

2. **Efficient Optimization**: Training only the classifier (typically <1% of parameters) is computationally cheap

3. **Reduced Overfitting**: Fewer trainable parameters means less risk of overfitting on small datasets

4. **Fast Convergence**: The feature extractor already produces meaningful representations

### 1.3 When to Use Feature Extraction

Feature extraction is the right choice when:

| Condition | Recommendation |
|-----------|----------------|
| Very limited data (<1000 images) | ✅ Feature extraction |
| Target domain similar to source | ✅ Feature extraction |
| Limited computational resources | ✅ Feature extraction |
| Quick prototyping needed | ✅ Feature extraction |
| Large target dataset available | Consider fine-tuning instead |
| Very different target domain | Consider fine-tuning instead |

## 2. Implementation in PyTorch

### 2.1 Loading and Modifying Pre-trained Models

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FeatureExtractor(nn.Module):
    """
    Transfer learning model using feature extraction strategy.
    
    The pre-trained convolutional layers are frozen and only
    the final classification layer is trained.
    """
    
    def __init__(self, num_classes, pretrained=True):
        super(FeatureExtractor, self).__init__()
        
        # Load pre-trained ResNet-18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features  # 512 for ResNet-18
        
        # Replace the classification head
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_num_trainable_params(self):
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


# Create model for 10-class classification
model = FeatureExtractor(num_classes=10)

# Verify parameter counts
trainable = model.get_num_trainable_params()
total = model.get_total_params()
print(f"Trainable parameters: {trainable:,}")
print(f"Total parameters: {total:,}")
print(f"Percentage trainable: {100 * trainable / total:.4f}%")
```

Output:
```
Trainable parameters: 5,130
Total parameters: 11,181,642
Percentage trainable: 0.0459%
```

### 2.2 Alternative: Using Model Surgery

For more control, we can manually construct the feature extractor:

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def create_feature_extractor_v2(num_classes):
    """
    Create feature extractor by removing the original classifier
    and adding a new one.
    """
    # Load pre-trained model
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Extract feature layers (everything except fc)
    features = nn.Sequential(*list(resnet.children())[:-1])
    
    # Freeze feature extractor
    for param in features.parameters():
        param.requires_grad = False
    
    # Create new classifier
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )
    
    # Combine into single model
    model = nn.Sequential(features, classifier)
    
    return model


# Alternative with dropout for regularization
def create_feature_extractor_with_dropout(num_classes, dropout_rate=0.5):
    """
    Feature extractor with dropout regularization in classifier.
    """
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    features = nn.Sequential(*list(resnet.children())[:-1])
    for param in features.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes)
    )
    
    return nn.Sequential(features, classifier)
```

### 2.3 Extracting Features for Downstream Use

Sometimes you want to extract features and use them with other classifiers (SVM, Random Forest, etc.):

```python
import torch
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

def extract_features(model, dataloader, device='cuda'):
    """
    Extract features from all images in a dataloader.
    
    Returns:
        features: numpy array of shape (n_samples, feature_dim)
        labels: numpy array of shape (n_samples,)
    """
    # Remove classifier to get feature extractor
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            # Extract features
            features = feature_extractor(inputs)
            features = features.squeeze()  # Remove spatial dimensions
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.vstack(all_features), np.concatenate(all_labels)


# Usage example
resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
train_features, train_labels = extract_features(resnet, train_loader)
test_features, test_labels = extract_features(resnet, test_loader)

print(f"Training features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")

# Now use with sklearn classifiers
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm = SVC(kernel='rbf', C=1.0)
svm.fit(train_features, train_labels)
predictions = svm.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
print(f"SVM accuracy on extracted features: {accuracy:.4f}")
```

## 3. Data Preparation

### 3.1 ImageNet Normalization

When using ImageNet pre-trained models, proper normalization is essential:

```python
import torchvision.transforms as transforms

# ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transforms (minimal augmentation for feature extraction)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Test transforms
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
```

### 3.2 Why ImageNet Statistics?

The pre-trained model learned to process images with specific statistics. Using different normalization will cause distribution mismatch:

```python
# WRONG: Using dataset-specific statistics
wrong_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # BAD!
])

# CORRECT: Using ImageNet statistics
correct_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # GOOD!
])
```

## 4. Complete Training Pipeline

### 4.1 Full Example with CIFAR-10

```python
"""
Feature Extraction Transfer Learning Example

This script demonstrates the complete workflow for transfer learning
using the feature extraction strategy on CIFAR-10.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import time
import copy

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

print(f"Using device: {DEVICE}")

# ============================================================================
# Data Preparation
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Load CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

CLASSES = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ============================================================================
# Model Setup
# ============================================================================

# Load pre-trained ResNet-18
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(CLASSES))

# Move to device
model = model.to(DEVICE)

# Verify trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.4f}%")

# ============================================================================
# Training Setup
# ============================================================================

criterion = nn.CrossEntropyLoss()

# Only optimize trainable parameters
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# ============================================================================
# Training Loop
# ============================================================================

print(f"\n{'='*60}")
print(f"Starting training for {NUM_EPOCHS} epochs")
print(f"{'='*60}\n")

best_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # Train
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )
    
    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    
    # Print results
    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        best_model_weights = copy.deepcopy(model.state_dict())

# Training summary
total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"Training completed in {total_time:.0f}s")
print(f"Best test accuracy: {best_acc:.2f}%")
print(f"{'='*60}")

# Load best model
model.load_state_dict(best_model_weights)
```

### 4.2 Expected Results

With the feature extraction approach on CIFAR-10:

| Metric | Expected Value |
|--------|---------------|
| Training time | 5-10 minutes (GPU) |
| Best test accuracy | 80-85% |
| Convergence | 3-5 epochs |
| Trainable parameters | ~5,130 |

## 5. Layer Analysis and Feature Visualization

### 5.1 Understanding What Layers Learn

Different depths of the network capture different types of features:

```python
import torch
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

def visualize_intermediate_features(model, image, layer_names):
    """
    Extract and visualize features from intermediate layers.
    """
    # Create feature extractor for specified layers
    feature_extractor = create_feature_extractor(
        model, return_nodes=layer_names
    )
    
    model.eval()
    with torch.no_grad():
        features = feature_extractor(image.unsqueeze(0))
    
    # Visualize
    fig, axes = plt.subplots(1, len(layer_names), figsize=(15, 4))
    
    for ax, name in zip(axes, layer_names):
        feat = features[name][0]  # Remove batch dimension
        
        # Average across channels for visualization
        feat_mean = feat.mean(dim=0).cpu().numpy()
        
        ax.imshow(feat_mean, cmap='viridis')
        ax.set_title(f"{name}\nShape: {tuple(feat.shape)}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('intermediate_features.png', dpi=150)
    plt.show()

# Usage
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

# Load a sample image
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

image = Image.open('sample.jpg')
image_tensor = transform(image)

visualize_intermediate_features(model, image_tensor, layer_names)
```

### 5.2 Feature Space Analysis

We can analyze how well features separate classes using t-SNE:

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_feature_space(features, labels, class_names):
    """
    Visualize high-dimensional features using t-SNE.
    
    Args:
        features: numpy array (n_samples, feature_dim)
        labels: numpy array (n_samples,)
        class_names: list of class name strings
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            label=class_name,
            alpha=0.6,
            s=20
        )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Feature Space Visualization (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig('feature_tsne.png', dpi=150)
    plt.show()

# Extract features and visualize
features, labels = extract_features(model, test_loader)
visualize_feature_space(features[:1000], labels[:1000], CLASSES)
```

## 6. Comparing Different Backbone Architectures

### 6.1 Architecture Comparison for Feature Extraction

```python
from torchvision.models import (
    resnet18, resnet50, vgg16, efficientnet_b0,
    ResNet18_Weights, ResNet50_Weights, VGG16_Weights, 
    EfficientNet_B0_Weights
)

def get_model_info(model_fn, weights, num_classes=10):
    """Get model information for comparison."""
    model = model_fn(weights=weights)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Find and replace classifier
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total,
        'trainable_params': trainable,
        'feature_dim': in_features,
        'model': model
    }

# Compare architectures
architectures = [
    ('ResNet-18', resnet18, ResNet18_Weights.IMAGENET1K_V1),
    ('ResNet-50', resnet50, ResNet50_Weights.IMAGENET1K_V1),
    ('VGG-16', vgg16, VGG16_Weights.IMAGENET1K_V1),
    ('EfficientNet-B0', efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
]

print("Architecture Comparison for Feature Extraction")
print("=" * 70)
print(f"{'Model':<18} {'Total Params':>15} {'Trainable':>12} {'Feature Dim':>12}")
print("-" * 70)

for name, model_fn, weights in architectures:
    info = get_model_info(model_fn, weights)
    print(f"{name:<18} {info['total_params']:>15,} {info['trainable_params']:>12,} "
          f"{info['feature_dim']:>12}")
```

Output:
```
Architecture Comparison for Feature Extraction
======================================================================
Model              Total Params    Trainable  Feature Dim
----------------------------------------------------------------------
ResNet-18            11,181,642        5,130          512
ResNet-50            23,528,522       20,490        2,048
VGG-16              134,285,130       40,970        4,096
EfficientNet-B0       4,016,274       12,810        1,280
```

### 6.2 Speed vs. Accuracy Trade-offs

| Model | Inference Time* | Expected Accuracy** | Best For |
|-------|----------------|---------------------|----------|
| ResNet-18 | 1.0x (baseline) | 80-82% | Fast prototyping |
| ResNet-50 | 2.5x | 83-86% | Balanced choice |
| VGG-16 | 4.0x | 81-84% | Simple architecture |
| EfficientNet-B0 | 1.2x | 85-88% | Best efficiency |

*Relative to ResNet-18 on same hardware  
**On CIFAR-10 with feature extraction

## 7. Practical Tips and Best Practices

### 7.1 Checklist for Feature Extraction

- [ ] Use correct normalization (ImageNet mean/std for ImageNet-pretrained models)
- [ ] Verify all convolutional layers are frozen
- [ ] Check only classifier parameters are trainable
- [ ] Use appropriate learning rate (0.001-0.01 typical)
- [ ] Monitor for overfitting despite fewer parameters

### 7.2 Common Mistakes to Avoid

```python
# MISTAKE 1: Forgetting to freeze layers
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 10)  # BAD: All layers still trainable!

# CORRECT: Freeze first, then modify
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 10)  # GOOD: Only fc is trainable


# MISTAKE 2: Wrong normalization
transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # BAD

# CORRECT: Use ImageNet statistics
transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# MISTAKE 3: Not using model.eval() for batch normalization
model.fc = nn.Linear(512, 10)
outputs = model(inputs)  # BAD: BN uses batch stats

# CORRECT: Set to eval mode
model.eval()  # GOOD: BN uses running stats
model.fc.train()  # Keep classifier in train mode if needed
```

### 7.3 When Feature Extraction Isn't Enough

Consider switching to fine-tuning when:

- Feature extraction accuracy plateaus well below expectations
- Target domain differs significantly from ImageNet
- You have sufficient data (>5,000 images per class)
- Computational resources allow longer training

## 8. Summary

Feature extraction provides a simple yet powerful approach to transfer learning:

1. **Core Idea**: Use pre-trained CNN as fixed feature extractor, train only the classifier
2. **Advantages**: Fast training, works with limited data, reduces overfitting risk
3. **Key Implementation**: Freeze backbone parameters, replace final layer, train with appropriate learning rate
4. **Best Practices**: Use ImageNet normalization, verify frozen layers, monitor for underfitting

Feature extraction is often the best starting point for any transfer learning project. If performance is insufficient, proceed to fine-tuning strategies covered in the next section.

## Exercises

1. **Implement feature extraction** with VGG-16 instead of ResNet-18 and compare results

2. **Extract features** from multiple layers and compare their discriminative power using linear probing

3. **Experiment with classifier complexity**: Compare a single linear layer vs. a 2-layer MLP as the classifier head

4. **Apply to a new dataset**: Use feature extraction on a dataset of your choice (e.g., Flowers102, Food101)
