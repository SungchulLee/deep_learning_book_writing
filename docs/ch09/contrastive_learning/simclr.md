# SimCLR: A Simple Framework for Contrastive Learning

## Learning Objectives

By the end of this section, you will be able to:

- Understand the SimCLR framework and its key components
- Implement SimCLR from scratch in PyTorch
- Analyze the importance of data augmentation composition
- Apply the NT-Xent loss function correctly
- Train SimCLR models and evaluate learned representations

## Introduction

SimCLR (Simple Framework for Contrastive Learning of Visual Representations) is a landmark self-supervised learning method introduced by Chen et al. (2020). Its elegance lies in achieving state-of-the-art results with a conceptually simple framework: learn representations by maximizing agreement between differently augmented views of the same image.

### Key Contributions

1. **Composition of data augmentations** is crucial for effective representation learning
2. **Learnable nonlinear transformation** (projection head) between representation and contrastive loss substantially improves quality
3. **Larger batch sizes and more training epochs** benefit contrastive learning more than supervised learning

## Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import ImageFilter, ImageOps
import random
import math
import numpy as np


# =============================================================================
# Data Augmentation
# =============================================================================

class GaussianBlur:
    """
    Gaussian blur augmentation as used in SimCLR.
    
    The kernel size is fixed at 10% of the image size.
    Sigma is sampled uniformly from [0.1, 2.0].
    """
    def __init__(self, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class SimCLRAugmentation:
    """
    Data augmentation pipeline for SimCLR.
    
    The paper systematically studied the composition of augmentations and found:
    1. Random crop + resize + color distortion is the most important composition
    2. Color distortion alone is not enough
    3. Random crop alone is not enough
    4. Composition is key to learning good representations
    
    Augmentation strength 's' controls the intensity of color jitter:
    - Brightness: 0.8 * s
    - Contrast: 0.8 * s
    - Saturation: 0.8 * s
    - Hue: 0.2 * s
    
    Args:
        size: Final image size (default: 224)
        s: Strength of color distortion (default: 1.0)
    """
    def __init__(self, size=224, s=1.0):
        # Color jitter with strength s
        color_jitter = transforms.ColorJitter(
            brightness=0.8 * s,
            contrast=0.8 * s,
            saturation=0.8 * s,
            hue=0.2 * s
        )
        
        # Full augmentation pipeline
        self.transform = transforms.Compose([
            # Random resized crop: most important augmentation
            # Scale: (0.08, 1.0) - crops can be very small
            # This forces the model to recognize objects at different scales
            transforms.RandomResizedCrop(size=size, scale=(0.08, 1.0)),
            
            # Random horizontal flip
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Color jitter with probability 0.8
            transforms.RandomApply([color_jitter], p=0.8),
            
            # Random grayscale with probability 0.2
            transforms.RandomGrayscale(p=0.2),
            
            # Gaussian blur with probability 0.5
            transforms.RandomApply([GaussianBlur()], p=0.5),
            
            # Convert to tensor
            transforms.ToTensor(),
            
            # ImageNet normalization
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def __call__(self, x):
        """
        Apply augmentation twice to create two correlated views.
        
        Args:
            x: PIL Image
        
        Returns:
            x1, x2: Two augmented views of the input
        """
        return self.transform(x), self.transform(x)


class SimCLRDataset(Dataset):
    """
    Dataset wrapper for SimCLR training.
    
    Wraps any image dataset and applies SimCLR augmentation
    to create positive pairs.
    """
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original image (ignore label)
        image, _ = self.base_dataset[idx]
        
        # Apply augmentation twice
        x1, x2 = self.transform(image)
        
        return x1, x2


# =============================================================================
# SimCLR Model
# =============================================================================

class ProjectionHead(nn.Module):
    """
    Projection head g(·) for SimCLR.
    
    The paper found that:
    1. A nonlinear projection head helps significantly
    2. 2-layer MLP with ReLU is better than linear projection
    3. Representations BEFORE the projection head transfer better
    
    This is because the projection head can remove information useful
    for downstream tasks but not needed for the contrastive objective.
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layer
        output_dim: Dimension of output projections
    """
    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class SimCLR(nn.Module):
    """
    SimCLR: Simple Framework for Contrastive Learning
    
    Architecture:
    - Encoder f(·): ResNet backbone (returns 2048-dim for ResNet50)
    - Projection head g(·): 2-layer MLP (2048 → 2048 → 128)
    
    The model is trained to maximize agreement between two augmented
    views of the same image using NT-Xent loss.
    
    Args:
        base_encoder: Backbone model name
        projection_dim: Output dimension of projection head
        temperature: Temperature for contrastive loss
    """
    def __init__(
        self,
        base_encoder='resnet50',
        projection_dim=128,
        temperature=0.5
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Build encoder
        if base_encoder == 'resnet18':
            self.encoder = models.resnet18(weights=None)
            encoder_dim = 512
        elif base_encoder == 'resnet50':
            self.encoder = models.resnet50(weights=None)
            encoder_dim = 2048
        elif base_encoder == 'resnet101':
            self.encoder = models.resnet101(weights=None)
            encoder_dim = 2048
        else:
            raise ValueError(f"Unknown encoder: {base_encoder}")
        
        # Remove classification head
        self.encoder.fc = nn.Identity()
        self.encoder_dim = encoder_dim
        
        # Projection head
        self.projection_head = ProjectionHead(
            input_dim=encoder_dim,
            hidden_dim=encoder_dim,  # Same as encoder dim
            output_dim=projection_dim
        )
    
    def forward(self, x1, x2=None):
        """
        Forward pass.
        
        Training: Pass two views, get projections for contrastive loss
        Inference: Pass one image, get representation
        """
        # Encode
        h1 = self.encoder(x1)
        
        if x2 is None:
            return h1
        
        h2 = self.encoder(x2)
        
        # Project
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        
        return z1, z2


# =============================================================================
# NT-Xent Loss
# =============================================================================

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    For a batch of N images, we create 2N augmented samples.
    For sample i, its positive pair is the other augmentation of the same image.
    The remaining 2(N-1) samples serve as negatives.
    
    Loss for positive pair (i, j):
    l(i, j) = -log(exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ))
    
    Total loss: average over all 2N samples
    
    Args:
        temperature: Temperature parameter τ
        base_temperature: Base temperature for BYOL-style normalization (optional)
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        Compute NT-Xent loss.
        
        Args:
            z1: Projections from first augmentation (N, D)
            z2: Projections from second augmentation (N, D)
        
        Returns:
            loss: Scalar loss value
        """
        device = z1.device
        batch_size = z1.shape[0]
        
        # L2 normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate: [z1; z2] -> (2N, D)
        representations = torch.cat([z1, z2], dim=0)
        
        # Cosine similarity matrix: (2N, 2N)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Apply temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels: for sample i in z1, its positive is at index i + N
        # for sample i in z2 (at position N+i), its positive is at index i
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ], dim=0).to(device)
        
        # Mask out self-similarity (diagonal)
        # We can't use the sample itself as a negative
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


# =============================================================================
# Training Functions
# =============================================================================

class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    
    SimCLR uses LARS for large batch training because:
    1. SGD fails to converge with very large batches
    2. LARS adapts learning rate per layer based on weight/gradient norms
    
    trust_ratio = trust_coef * ||w|| / (||g|| + weight_decay * ||w||)
    lr_layer = lr * trust_ratio
    
    Args:
        params: Model parameters
        lr: Base learning rate
        momentum: Momentum coefficient
        weight_decay: Weight decay coefficient
        trust_coefficient: LARS trust coefficient
        exclude_bias_and_bn: Whether to exclude bias and BN from LARS
    """
    def __init__(
        self,
        params,
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-6,
        trust_coefficient=0.001,
        exclude_bias_and_bn=True
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient,
            exclude_bias_and_bn=exclude_bias_and_bn
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Compute LARS trust ratio
                weight_norm = p.norm()
                grad_norm = grad.norm()
                
                trust_ratio = 1.0
                if weight_norm > 0 and grad_norm > 0:
                    # Check if this parameter should be excluded
                    is_excluded = (
                        group['exclude_bias_and_bn'] and
                        (len(p.shape) == 1)  # Bias or BN parameters
                    )
                    
                    if not is_excluded:
                        trust_ratio = group['trust_coefficient'] * weight_norm / grad_norm
                
                # Apply scaled learning rate
                actual_lr = group['lr'] * trust_ratio
                
                # Momentum update
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = param_state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad, alpha=actual_lr)
                
                p.sub_(buf)
        
        return loss


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, 
                                    min_lr=0, last_epoch=-1):
    """
    Cosine annealing schedule with warmup.
    
    SimCLR uses:
    - Linear warmup for first 10 epochs
    - Cosine decay without restarts for remaining epochs
    
    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr: Minimum learning rate at end of training
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Train SimCLR for one epoch.
    
    Args:
        model: SimCLR model
        dataloader: Training data loader
        optimizer: Optimizer (LARS recommended)
        criterion: NT-Xent loss
        device: Device
        epoch: Current epoch number
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (x1, x2) in enumerate(dataloader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        
        # Forward pass
        z1, z2 = model(x1, x2)
        
        # Compute loss
        loss = criterion(z1, z2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def train_simclr(
    model,
    train_dataset,
    epochs=100,
    batch_size=256,
    learning_rate=0.3,
    weight_decay=1e-6,
    temperature=0.5,
    warmup_epochs=10,
    device='cuda'
):
    """
    Full SimCLR training loop.
    
    SimCLR hyperparameters from the paper:
    - Batch size: 4096 (we use smaller for demonstration)
    - Learning rate: 0.3 * batch_size / 256 (linear scaling rule)
    - Weight decay: 1e-6
    - Temperature: 0.5
    - Epochs: 100-1000
    - Warmup: 10 epochs
    
    Args:
        model: SimCLR model
        train_dataset: Training dataset (will be wrapped)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Base learning rate
        weight_decay: Weight decay
        temperature: Temperature for NT-Xent loss
        warmup_epochs: Number of warmup epochs
        device: Device
    
    Returns:
        Trained model, training losses
    """
    model = model.to(device)
    
    # Create dataset with augmentations
    augmentation = SimCLRAugmentation(size=224)
    simclr_dataset = SimCLRDataset(train_dataset, augmentation)
    
    # Data loader
    dataloader = DataLoader(
        simclr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Important for consistent batch sizes
    )
    
    # Learning rate scaling
    scaled_lr = learning_rate * batch_size / 256
    
    # Optimizer (LARS for large batches, SGD for small batches)
    if batch_size >= 256:
        optimizer = LARS(
            model.parameters(),
            lr=scaled_lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=scaled_lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    
    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs
    )
    
    # Loss function
    criterion = NTXentLoss(temperature=temperature)
    
    # Training loop
    losses = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        epoch_loss = train_epoch(
            model, dataloader, optimizer, criterion, device, epoch
        )
        
        losses.append(epoch_loss)
        scheduler.step()
        
        print(f"  Average loss: {epoch_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'simclr_checkpoint_epoch_{epoch+1}.pt')
    
    return model, losses


# =============================================================================
# Evaluation
# =============================================================================

def extract_features(model, dataloader, device):
    """
    Extract features using trained encoder (without projection head).
    
    Args:
        model: Trained SimCLR model
        dataloader: Data loader
        device: Device
    
    Returns:
        features: (N, encoder_dim) numpy array
        labels: (N,) numpy array
    """
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Get representations (before projection head)
            features = model(images)  # model.forward with single input
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    return features, labels


def linear_evaluation(model, train_loader, test_loader, device, 
                      num_classes=10, epochs=100):
    """
    Linear evaluation protocol for SimCLR.
    
    Train a linear classifier on frozen representations.
    This is the standard evaluation method for self-supervised learning.
    
    Args:
        model: Trained SimCLR model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device
        num_classes: Number of classes
        epochs: Training epochs for linear classifier
    
    Returns:
        Test accuracy
    """
    # Extract features
    print("Extracting features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    
    # Create tensors
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Linear classifier
    classifier = nn.Linear(train_features.shape[1], num_classes).to(device)
    
    # Optimizer
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training loop
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader_lin = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    print("Training linear classifier...")
    classifier.train()
    for epoch in range(epochs):
        for features, labels in train_loader_lin:
            features = features.to(device)
            labels = labels.to(device)
            
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    # Evaluate
    classifier.eval()
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    with torch.no_grad():
        logits = classifier(test_features)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == test_labels).float().mean().item() * 100
    
    print(f"Linear evaluation accuracy: {accuracy:.2f}%")
    return accuracy


def knn_evaluation(model, train_loader, test_loader, device, k=200):
    """
    k-NN evaluation for SimCLR.
    
    A simpler alternative to linear evaluation that doesn't require training.
    
    Args:
        model: Trained SimCLR model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device
        k: Number of neighbors
    
    Returns:
        Test accuracy
    """
    # Extract features
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    
    # Convert to tensors
    train_features = torch.tensor(train_features).to(device)
    train_labels = torch.tensor(train_labels).to(device)
    test_features = torch.tensor(test_features).to(device)
    test_labels = torch.tensor(test_labels).to(device)
    
    # Normalize
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)
    
    # Compute similarities
    similarity = torch.mm(test_features, train_features.T)
    
    # Get k nearest neighbors
    _, indices = similarity.topk(k, dim=1)
    
    # Vote
    neighbor_labels = train_labels[indices]
    predictions = torch.mode(neighbor_labels, dim=1).values
    
    accuracy = (predictions == test_labels).float().mean().item() * 100
    
    print(f"k-NN accuracy (k={k}): {accuracy:.2f}%")
    return accuracy
```

## Key Findings from the Paper

### 1. Data Augmentation Analysis

The paper systematically studies augmentation combinations:

| Augmentation Composition | ImageNet Top-1 |
|--------------------------|----------------|
| Random crop only | 56.9% |
| Color distortion only | 47.5% |
| Crop + Color | 64.5% |
| Crop + Color + Blur | 66.6% |
| Crop + Color + Blur + Grayscale | **66.6%** |

```python
def compare_augmentation_effects():
    """
    Demonstrate the importance of augmentation composition.
    """
    # Different augmentation configurations
    configs = {
        'crop_only': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'color_only': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'crop_and_color': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'full_simclr': SimCLRAugmentation(224)
    }
    
    print("SimCLR key insight: Composition of augmentations matters more than individual augmentations")
    print("\nAugmentation configurations for analysis:")
    for name in configs:
        print(f"  - {name}")
```

### 2. Projection Head Importance

The projection head significantly improves representation quality:

| Setting | Linear Eval Top-1 |
|---------|------------------|
| No projection (h) | 60.6% |
| Linear projection | 63.9% |
| 2-layer MLP | **66.8%** |
| 3-layer MLP | 66.5% |

```python
def analyze_projection_head_impact(encoder_dim=2048, projection_dim=128):
    """
    Different projection head configurations.
    
    Key insight: The projection head removes information that is useful
    for downstream tasks but not for the contrastive objective.
    
    This is why we use representations BEFORE the projection head
    for downstream tasks.
    """
    configurations = {
        'no_projection': nn.Identity(),
        'linear': nn.Linear(encoder_dim, projection_dim),
        'mlp_2_layer': nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        ),
        'mlp_3_layer': nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )
    }
    
    print("Projection head analysis:")
    print("- 2-layer MLP is optimal")
    print("- Representations before projection transfer better")
    print("- Projection head 'filters out' task-irrelevant information")
    
    return configurations
```

### 3. Batch Size and Training Duration

SimCLR benefits from large batches and long training:

| Batch Size | Epochs | ImageNet Top-1 |
|------------|--------|----------------|
| 256 | 100 | 61.9% |
| 1024 | 100 | 63.8% |
| 4096 | 100 | 66.6% |
| 4096 | 200 | 68.3% |
| 4096 | 1000 | **69.3%** |

```python
def batch_size_learning_rate_scaling(base_lr=0.3, base_batch_size=256):
    """
    Linear scaling rule for learning rate with batch size.
    
    SimCLR finding: Performance improves with batch size up to 4096-8192.
    
    This is because more negatives provide a tighter bound on mutual information.
    """
    batch_sizes = [256, 512, 1024, 2048, 4096, 8192]
    
    print("Batch size to learning rate mapping:")
    for bs in batch_sizes:
        lr = base_lr * bs / base_batch_size
        print(f"  Batch size {bs}: lr = {lr:.4f}")
    
    print("\nWhy larger batches help:")
    print("1. More negative samples per update")
    print("2. Tighter bound on mutual information")
    print("3. More stable gradient estimates")
    print("4. Requires LARS optimizer for convergence")
```

## SimCLR v2 Improvements

SimCLR v2 introduces several improvements:

```python
class SimCLRv2(nn.Module):
    """
    SimCLR v2 improvements over v1:
    
    1. Deeper projection head (3 layers instead of 2)
    2. MoCo-style memory mechanism (optional)
    3. Knowledge distillation from larger models
    4. Selective kernels (SK) in ResNet
    """
    def __init__(
        self,
        base_encoder='resnet50',
        projection_dim=128,
        hidden_dim=2048,
        temperature=0.1,  # Lower temperature than v1
        use_sk=True  # Selective kernels
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Build encoder (potentially with SK)
        if base_encoder == 'resnet50':
            # Standard ResNet for simplicity
            # SK-ResNet would use selective kernel attention
            self.encoder = models.resnet50(weights=None)
            encoder_dim = 2048
            self.encoder.fc = nn.Identity()
        
        self.encoder_dim = encoder_dim
        
        # Deeper projection head (3 layers)
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, x1, x2=None):
        h1 = self.encoder(x1)
        
        if x2 is None:
            return h1
        
        h2 = self.encoder(x2)
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        
        return z1, z2
```

## Usage Example

```python
if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SimCLR(
        base_encoder='resnet50',
        projection_dim=128,
        temperature=0.5
    )
    
    print(f"\nSimCLR Model:")
    print(f"  Encoder dim: {model.encoder_dim}")
    print(f"  Projection dim: 128")
    print(f"  Temperature: 0.5")
    
    # Test forward pass
    x1 = torch.randn(32, 3, 224, 224)
    x2 = torch.randn(32, 3, 224, 224)
    
    model.eval()
    z1, z2 = model(x1, x2)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {x1.shape}")
    print(f"  Output shape: {z1.shape}")
    
    # Test loss
    criterion = NTXentLoss(temperature=0.5)
    loss = criterion(z1, z2)
    print(f"  Loss: {loss.item():.4f}")
    
    # # For actual training, use:
    # train_dataset = CIFAR10(root='./data', train=True, download=True)
    # model, losses = train_simclr(
    #     model=model,
    #     train_dataset=train_dataset,
    #     epochs=100,
    #     batch_size=256,
    #     device=device
    # )
```

## Summary

SimCLR demonstrates that simple design choices can achieve strong results:

| Component | Choice | Impact |
|-----------|--------|--------|
| Data Augmentation | Crop + Color + Blur + Grayscale | Critical |
| Projection Head | 2-layer MLP | Important |
| Temperature | 0.5 (v1), 0.1 (v2) | Moderate |
| Batch Size | ≥4096 | Important |
| Training Duration | ≥1000 epochs | Important |

The simplicity of SimCLR has made it a baseline for many subsequent methods, though its requirement for large batch sizes has led to alternatives like MoCo.

## References

1. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.
2. Chen, T., Kornblith, S., Swersky, K., Norouzi, M., & Hinton, G. (2020). Big Self-Supervised Models are Strong Semi-Supervised Learners. NeurIPS.
3. Goyal, P., et al. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv.
