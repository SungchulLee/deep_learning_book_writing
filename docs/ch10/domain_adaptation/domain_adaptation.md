# Domain Adaptation

## Learning Objectives

By the end of this section, you will:

- Understand the domain adaptation problem and its theoretical foundations
- Distinguish between different types of domain shift
- Implement domain adaptation techniques for transfer learning
- Apply techniques for handling domain mismatch in practice
- Know when and how to use unsupervised domain adaptation

## 1. The Domain Adaptation Problem

Domain adaptation addresses a fundamental challenge: models trained on one data distribution (source domain) often perform poorly on a different distribution (target domain), even when the underlying task is the same.

### 1.1 Formal Definition

Given:
- **Source domain** $\mathcal{D}_S$ with labeled data $\{(\mathbf{x}_i^s, y_i^s)\}_{i=1}^{n_s}$
- **Target domain** $\mathcal{D}_T$ with unlabeled data $\{\mathbf{x}_j^t\}_{j=1}^{n_t}$

The domains differ: $P_S(\mathbf{X}) \neq P_T(\mathbf{X})$

Goal: Learn a model that performs well on $\mathcal{D}_T$ using knowledge from $\mathcal{D}_S$.

### 1.2 Types of Domain Shift

| Type | Definition | Example |
|------|------------|---------|
| Covariate Shift | $P_S(\mathbf{X}) \neq P_T(\mathbf{X})$, $P(Y|\mathbf{X})$ same | Day→Night photos |
| Label Shift | $P_S(Y) \neq P_T(Y)$, $P(\mathbf{X}|Y)$ same | Different class proportions |
| Concept Drift | $P_S(Y|\mathbf{X}) \neq P_T(Y|\mathbf{X})$ | Word meanings change |

### 1.3 The Domain Adaptation Bound

Ben-David et al. (2010) established:

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda^*$$

Target error is bounded by source error plus domain divergence plus irreducible error.

## 2. Simple Domain Adaptation Techniques

### 2.1 Data Augmentation for Domain Robustness

```python
import torchvision.transforms as transforms

def create_domain_robust_transforms(target_type='general'):
    """Create augmentations to improve domain robustness."""
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    if target_type == 'low_light':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=(0.1, 0.5), contrast=0.5),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
```

### 2.2 Batch Normalization Adaptation

```python
import torch
import torch.nn as nn

def adapt_batchnorm(model, target_loader, device='cuda'):
    """
    Update BatchNorm statistics using target domain data.
    Simple but effective domain adaptation technique.
    """
    model.train()  # Enable BN stat updates
    
    with torch.no_grad():
        for inputs, _ in target_loader:
            inputs = inputs.to(device)
            _ = model(inputs)  # Updates running mean/var
    
    model.eval()
    return model
```

## 3. Feature-Level Domain Adaptation

### 3.1 Maximum Mean Discrepancy (MMD)

MMD measures distance between feature distributions:

```python
import torch
import torch.nn as nn

def compute_mmd(source_features, target_features, gamma=1.0):
    """Compute MMD between source and target features."""
    n_s, n_t = source_features.size(0), target_features.size(0)
    
    # RBF kernel
    def rbf_kernel(x, y):
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        yy = torch.sum(y ** 2, dim=1, keepdim=True)
        distances = xx + yy.t() - 2 * torch.mm(x, y.t())
        return torch.exp(-gamma * distances)
    
    ss = rbf_kernel(source_features, source_features)
    tt = rbf_kernel(target_features, target_features)
    st = rbf_kernel(source_features, target_features)
    
    mmd = (ss.sum() / (n_s * n_s) + 
           tt.sum() / (n_t * n_t) - 
           2 * st.sum() / (n_s * n_t))
    
    return torch.sqrt(torch.clamp(mmd, min=0))


class DomainAdaptationModel(nn.Module):
    """Model with MMD regularization for domain adaptation."""
    
    def __init__(self, backbone, num_classes, feature_dim=512):
        super().__init__()
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Freeze early layers
        for param in self.feature_extractor[:5].parameters():
            param.requires_grad = False
    
    def forward(self, x, return_features=False):
        features = self.flatten(self.feature_extractor(x))
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
```

### 3.2 Training with MMD Loss

```python
def train_with_domain_adaptation(model, source_loader, target_loader, 
                                  num_epochs=20, mmd_weight=0.1, device='cuda'):
    """
    Train with both classification and MMD domain adaptation losses.
    """
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Zip source and target loaders
        target_iter = iter(target_loader)
        
        for source_batch in source_loader:
            # Get target batch (cycle if needed)
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
            
            source_inputs, source_labels = source_batch
            target_inputs, _ = target_batch
            
            source_inputs = source_inputs.to(device)
            source_labels = source_labels.to(device)
            target_inputs = target_inputs.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with features
            source_logits, source_features = model(source_inputs, return_features=True)
            _, target_features = model(target_inputs, return_features=True)
            
            # Classification loss (source only)
            cls_loss = criterion(source_logits, source_labels)
            
            # MMD domain adaptation loss
            mmd_loss = compute_mmd(source_features, target_features)
            
            # Combined loss
            loss = cls_loss + mmd_weight * mmd_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(source_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model
```

## 4. Domain Adversarial Neural Networks (DANN)

### 4.1 The Adversarial Approach

DANN learns domain-invariant features by fooling a domain classifier:

```
                    ┌──────────────────┐
                    │  Label Predictor │ → Classification Loss
                    └────────┬─────────┘
                             │
Input ──► Feature Extractor ─┴──► Gradient Reversal ──► Domain Classifier → Domain Loss
                                        (flip gradient)
```

### 4.2 Gradient Reversal Layer

```python
import torch
from torch.autograd import Function

class GradientReversal(Function):
    """Gradient Reversal Layer for domain adversarial training."""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient
        return -ctx.lambda_ * grad_output, None


class GRL(nn.Module):
    """Gradient Reversal Layer module."""
    
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_
```

### 4.3 DANN Architecture

```python
class DANN(nn.Module):
    """Domain Adversarial Neural Network."""
    
    def __init__(self, backbone, num_classes, feature_dim=512):
        super().__init__()
        
        # Feature extractor (shared)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.flatten = nn.Flatten()
        
        # Label predictor
        self.label_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Domain classifier (with GRL)
        self.grl = GRL(lambda_=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Binary: source/target
        )
    
    def forward(self, x, alpha=1.0):
        features = self.flatten(self.feature_extractor(x))
        
        # Label prediction
        class_output = self.label_predictor(features)
        
        # Domain prediction (with gradient reversal)
        self.grl.set_lambda(alpha)
        reversed_features = self.grl(features)
        domain_output = self.domain_classifier(reversed_features)
        
        return class_output, domain_output


def train_dann(model, source_loader, target_loader, num_epochs=50, device='cuda'):
    """Train DANN model."""
    model = model.to(device)
    
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(num_epochs):
        model.train()
        
        # Progressive lambda schedule
        p = epoch / num_epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1  # Grows from 0 to 1
        
        target_iter = iter(target_loader)
        
        for source_inputs, source_labels in source_loader:
            try:
                target_inputs, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_inputs, _ = next(target_iter)
            
            # Combine source and target
            batch_size = source_inputs.size(0)
            inputs = torch.cat([source_inputs, target_inputs[:batch_size]], dim=0)
            inputs = inputs.to(device)
            source_labels = source_labels.to(device)
            
            # Domain labels: 0 for source, 1 for target
            domain_labels = torch.cat([
                torch.zeros(batch_size),
                torch.ones(min(batch_size, target_inputs.size(0)))
            ]).long().to(device)
            
            optimizer.zero_grad()
            
            class_output, domain_output = model(inputs, alpha=alpha)
            
            # Classification loss (source only)
            class_loss = class_criterion(class_output[:batch_size], source_labels)
            
            # Domain loss (both source and target)
            domain_loss = domain_criterion(domain_output, domain_labels)
            
            # Total loss
            loss = class_loss + domain_loss
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Alpha: {alpha:.3f}")
    
    return model
```

## 5. Practical Domain Adaptation Workflow

### 5.1 Complete Pipeline

```python
"""
Domain Adaptation Pipeline

Workflow:
1. Assess domain gap
2. Try simple techniques first (BN adaptation, augmentation)
3. If needed, apply feature-level adaptation (MMD, DANN)
4. Evaluate on target domain
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

# Step 1: Load pre-trained model
def load_pretrained_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Step 2: Assess domain gap (using feature statistics)
def assess_domain_gap(model, source_loader, target_loader, device='cuda'):
    """Compute simple statistics to assess domain gap."""
    model.eval()
    model = model.to(device)
    
    # Extract features
    def get_features(loader):
        features = []
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                feat = feature_extractor(inputs).squeeze()
                features.append(feat.cpu())
        return torch.cat(features, dim=0)
    
    source_feat = get_features(source_loader)
    target_feat = get_features(target_loader)
    
    # Compare statistics
    source_mean = source_feat.mean(dim=0)
    target_mean = target_feat.mean(dim=0)
    
    mean_diff = (source_mean - target_mean).norm().item()
    mmd = compute_mmd(source_feat[:1000], target_feat[:1000]).item()
    
    print(f"Domain Gap Assessment:")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  MMD: {mmd:.4f}")
    
    return mean_diff, mmd

# Step 3: Apply appropriate adaptation
def apply_domain_adaptation(model, source_loader, target_loader, 
                           method='bn_adapt', device='cuda'):
    """Apply domain adaptation based on method."""
    
    if method == 'bn_adapt':
        # Simple BN adaptation
        return adapt_batchnorm(model, target_loader, device)
    
    elif method == 'mmd':
        # MMD-based adaptation
        da_model = DomainAdaptationModel(model, num_classes=10)
        return train_with_domain_adaptation(
            da_model, source_loader, target_loader, 
            num_epochs=20, mmd_weight=0.1, device=device
        )
    
    elif method == 'dann':
        # DANN
        dann = DANN(model, num_classes=10)
        return train_dann(
            dann, source_loader, target_loader,
            num_epochs=50, device=device
        )
    
    return model
```

### 5.2 Method Selection Guide

```
Domain Gap Assessment
        │
        ├─► Gap is small (MMD < 0.1)
        │       │
        │       └─► Use BatchNorm adaptation or standard fine-tuning
        │
        ├─► Gap is moderate (0.1 < MMD < 0.5)
        │       │
        │       ├─► Labeled target data available → Fine-tuning with augmentation
        │       │
        │       └─► No target labels → MMD-based adaptation
        │
        └─► Gap is large (MMD > 0.5)
                │
                ├─► Labeled target data → Full fine-tuning on target
                │
                └─► No target labels → DANN or advanced methods
```

## 6. Evaluating Domain Adaptation

### 6.1 Key Metrics

```python
def evaluate_domain_adaptation(model, source_loader, target_loader, device='cuda'):
    """Comprehensive evaluation of domain adaptation."""
    model.eval()
    model = model.to(device)
    
    def compute_accuracy(loader):
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs) if not isinstance(model, DANN) else model(inputs)[0]
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100.0 * correct / total
    
    source_acc = compute_accuracy(source_loader)
    target_acc = compute_accuracy(target_loader)
    
    print(f"\nDomain Adaptation Results:")
    print(f"  Source accuracy: {source_acc:.2f}%")
    print(f"  Target accuracy: {target_acc:.2f}%")
    print(f"  Adaptation gap: {source_acc - target_acc:.2f}%")
    
    return source_acc, target_acc
```

## 7. Summary

Domain adaptation enables transfer learning when source and target distributions differ:

1. **Assess the gap**: Measure domain divergence before choosing methods
2. **Simple first**: BatchNorm adaptation and augmentation often suffice
3. **Feature alignment**: MMD minimizes distribution distance in feature space
4. **Adversarial training**: DANN learns domain-invariant features via adversarial learning
5. **Evaluate carefully**: Monitor both source and target performance

The choice of method depends on:
- Size of domain gap
- Availability of target labels
- Computational budget
- Amount of target data

## Exercises

1. **Implement A-distance**: Create a classifier to measure domain divergence as proxy A-distance

2. **Compare methods**: Apply BN adaptation, MMD, and DANN to a domain shift benchmark and compare

3. **Style transfer augmentation**: Use neural style transfer to augment source data toward target style

4. **Self-training**: Implement pseudo-labeling for semi-supervised domain adaptation
