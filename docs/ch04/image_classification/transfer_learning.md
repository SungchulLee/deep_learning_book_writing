# Transfer Learning: Leveraging Pretrained Models

## Overview

Transfer learning uses knowledge from models pretrained on large datasets (ImageNet) to improve performance on new tasks, especially when training data is limited. This is one of the most powerful techniques in practical deep learning.

!!! info "Key Papers"
    - Yosinski et al., 2014 - "How Transferable are Features in Deep Neural Networks?"
    - Kornblith et al., 2019 - "Do Better ImageNet Models Transfer Better?"

## Learning Objectives

1. Understand when and why transfer learning helps
2. Implement feature extraction and fine-tuning strategies
3. Apply differential learning rates for optimal transfer
4. Adapt pretrained models to custom datasets

## Why Transfer Learning Works

### Hierarchical Feature Learning

CNNs learn hierarchical features from general to specific:

```
Layer 1-2:    Edges, colors, gradients       ← Universal
Layer 3-5:    Textures, patterns             ← Mostly universal
Layer 6-10:   Object parts                   ← Domain-dependent
Layer 11+:    Whole objects, task-specific   ← Task-specific
```

**Key insight**: Early layers transfer well across tasks; later layers need adaptation.

### When to Use Transfer Learning

| Scenario | Data Size | Similarity | Strategy |
|----------|-----------|------------|----------|
| Small data, similar | <1K | High | Feature extraction |
| Small data, different | <1K | Low | Feature extraction + augmentation |
| Large data, similar | >10K | High | Fine-tune all layers |
| Large data, different | >10K | Low | Fine-tune, longer training |

## Loading Pretrained Models

### From torchvision

```python
import torchvision.models as models

# Load pretrained ResNet-50
model = models.resnet50(pretrained=True)  # Or weights='IMAGENET1K_V1'

# Load pretrained EfficientNet
model = models.efficientnet_b0(pretrained=True)

# Load pretrained MobileNet
model = models.mobilenet_v2(pretrained=True)
```

### Modifying the Classifier

```python
def load_pretrained_model(model_name, num_classes, pretrained=True):
    """Load pretrained model and replace classifier."""
    
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'vgg16':
        model = models.vgg16_bn(pretrained=pretrained)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model
```

## Transfer Strategies

### 1. Feature Extraction (Freeze All)

Use pretrained network as fixed feature extractor:

```python
def freeze_backbone(model):
    """Freeze all layers except classifier."""
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier
    if hasattr(model, 'fc'):  # ResNet
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):  # VGG, EfficientNet
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Training {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")
```

**When to use**: Small dataset (<1K images), limited compute, similar domain

### 2. Fine-Tuning (Train All)

Train entire network with lower learning rate:

```python
def fine_tune_all(model, lr=0.001):
    """Fine-tune all layers with uniform learning rate."""
    
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer
```

**When to use**: Large dataset (>10K), different domain, ample compute

### 3. Gradual Fine-Tuning

Start frozen, then unfreeze progressively:

```python
def gradual_unfreeze(model, epoch, total_epochs):
    """Unfreeze layers progressively through training."""
    
    # Get all layer groups
    if hasattr(model, 'layer1'):  # ResNet
        groups = [model.conv1, model.bn1, model.layer1, 
                  model.layer2, model.layer3, model.layer4, model.fc]
    else:
        groups = list(model.children())
    
    # Unfreeze from end based on epoch
    num_to_unfreeze = int(len(groups) * epoch / total_epochs)
    
    for i, group in enumerate(reversed(groups)):
        for param in group.parameters():
            param.requires_grad = (i < num_to_unfreeze)
```

### 4. Differential Learning Rates

Different learning rates for different layers:

```python
def get_differential_optimizer(model, base_lr=0.001):
    """
    Higher LR for classifier, lower for early layers.
    
    Principle:
    - Early layers: Small LR (preserve general features)
    - Late layers: Medium LR (adapt features)
    - Classifier: High LR (learn new task)
    """
    
    if hasattr(model, 'layer1'):  # ResNet
        param_groups = [
            {'params': model.conv1.parameters(), 'lr': base_lr * 0.01},
            {'params': model.bn1.parameters(), 'lr': base_lr * 0.01},
            {'params': model.layer1.parameters(), 'lr': base_lr * 0.1},
            {'params': model.layer2.parameters(), 'lr': base_lr * 0.1},
            {'params': model.layer3.parameters(), 'lr': base_lr * 0.5},
            {'params': model.layer4.parameters(), 'lr': base_lr * 0.5},
            {'params': model.fc.parameters(), 'lr': base_lr},
        ]
    else:
        # Generic: low LR for features, high for classifier
        param_groups = [
            {'params': model.features.parameters(), 'lr': base_lr * 0.1},
            {'params': model.classifier.parameters(), 'lr': base_lr},
        ]
    
    return optim.Adam(param_groups, lr=base_lr)
```

## Data Preparation

### ImageNet Normalization

**Always use ImageNet statistics for pretrained models:**

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

### Data Augmentation for Small Datasets

```python
from torchvision.transforms import autoaugment

train_transform_strong = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.RandomRotation(15),
    autoaugment.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

## Complete Training Pipeline

```python
class TransferLearner:
    def __init__(self, model_name, num_classes, strategy='feature_extract'):
        self.model = load_pretrained_model(model_name, num_classes)
        self.strategy = strategy
        
        if strategy == 'feature_extract':
            freeze_backbone(self.model)
        
    def train(self, train_loader, val_loader, epochs=30, lr=0.001):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        # Optimizer based on strategy
        if self.strategy == 'differential_lr':
            optimizer = get_differential_optimizer(self.model, lr)
        else:
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr
            )
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        history = {'train_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Gradual unfreeze
            if self.strategy == 'gradual':
                gradual_unfreeze(self.model, epoch, epochs)
            
            # Train
            self.model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validate
            val_acc = self.evaluate(val_loader, device)
            scheduler.step(val_acc)
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_acc'].append(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Acc={val_acc:.2f}%")
        
        return history, best_acc
    
    def evaluate(self, loader, device):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = self.model(x).argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        return 100 * correct / total
```

## Strategy Comparison

### Results on Custom Dataset (5K images, 10 classes)

| Strategy | Val Acc | Training Time | Best For |
|----------|---------|---------------|----------|
| Feature extract | 85-88% | Fast (10 min) | Quick baseline |
| Fine-tune all | 91-94% | Slow (2 hr) | Maximum accuracy |
| Differential LR | 90-93% | Medium (1 hr) | Good balance |
| Gradual unfreeze | 89-92% | Medium (1.5 hr) | Avoiding overfitting |

### Comparison Plot

```python
def compare_strategies(train_loader, val_loader, num_classes):
    strategies = ['feature_extract', 'fine_tune', 'differential_lr', 'gradual']
    results = {}
    
    for strategy in strategies:
        learner = TransferLearner('resnet50', num_classes, strategy)
        history, best_acc = learner.train(train_loader, val_loader)
        results[strategy] = {'history': history, 'best': best_acc}
    
    # Plot
    plt.figure(figsize=(10, 4))
    for strategy, data in results.items():
        plt.plot(data['history']['val_acc'], label=f"{strategy}: {data['best']:.1f}%")
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.legend()
    plt.title('Transfer Learning Strategy Comparison')
    plt.savefig('transfer_comparison.png')
```

## Few-Shot Transfer Learning

For extremely limited data (<100 samples per class):

```python
def few_shot_transfer(model, num_classes, num_shots=10):
    """Transfer learning with very few samples."""
    
    # 1. Freeze everything except final layer
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. Replace classifier with simpler one (fewer params)
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Heavy regularization
            nn.Linear(256, num_classes)
        )
    
    # Unfreeze new classifier
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # 3. Use heavy data augmentation
    # 4. Train with early stopping
    # 5. Consider meta-learning approaches for very few shots
    
    return model
```

## Exercises

1. **Strategy Comparison**: Compare all strategies on CIFAR-10 with only 10% of training data
2. **Domain Shift**: Transfer ImageNet model to medical images, measure accuracy drop
3. **Feature Analysis**: Visualize features from different layers, identify where task-specific features emerge
4. **Few-Shot**: Achieve >80% accuracy with only 50 images per class

## Key Takeaways

1. **Always start with pretrained** models when data is limited
2. **Feature extraction** is fast and effective for similar domains
3. **Differential learning rates** balance preservation and adaptation
4. **ImageNet normalization** must be used with ImageNet pretrained models
5. **More data + different domain** → more fine-tuning needed

## References

1. Yosinski et al. (2014). How Transferable are Features in Deep Neural Networks?
2. Kornblith et al. (2019). Do Better ImageNet Models Transfer Better?
3. Howard & Ruder (2018). Universal Language Model Fine-tuning (ULMFiT).

---

**Previous**: [Model Comparison](model_comparison.md) | **Next**: [Ensemble Methods](ensemble.md)
