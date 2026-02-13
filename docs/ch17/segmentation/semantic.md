# Semantic Segmentation

## Learning Objectives

By the end of this section, you will be able to:

- Understand the complete semantic segmentation pipeline from data to inference
- Implement effective data augmentation strategies with synchronized transforms
- Apply attention mechanisms (channel, spatial, self-attention) to enhance segmentation
- Use test-time augmentation and mixed precision training for production deployment
- Select appropriate training strategies for different segmentation scenarios

## The Semantic Segmentation Task

Semantic segmentation assigns a class label to every pixel in an image. Given an input $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$, the model produces:

$$\mathbf{Y} \in \{0, 1, \ldots, K-1\}^{H \times W}$$

where each pixel $(i, j)$ receives a label $y_{i,j}$ from $K$ predefined categories. In practice, the network outputs a probability distribution $\hat{\mathbf{Y}} \in [0, 1]^{H \times W \times K}$ and the final prediction is $y_{i,j} = \arg\max_k \hat{y}_{i,j,k}$.

Unlike instance segmentation (which distinguishes individual objects) or panoptic segmentation (which unifies both), semantic segmentation treats all pixels of the same class identically—all cars map to the "car" label regardless of how many individual vehicles appear.

### Core Architecture Paradigm

Nearly all modern semantic segmentation networks follow the **encoder-decoder** design:

1. **Encoder**: A backbone network (typically pretrained on ImageNet) extracts hierarchical features at progressively lower spatial resolutions
2. **Decoder**: Recovers spatial resolution through upsampling and feature fusion
3. **Skip connections**: Bridge encoder and decoder to combine semantic and spatial information

The specific architectures—FCN, U-Net, and DeepLab—are covered in dedicated sections. This page focuses on the training methodology, attention mechanisms, and inference strategies that apply broadly across all architectures.

## Attention Mechanisms for Segmentation

Attention mechanisms allow networks to focus on relevant features while suppressing noise. In segmentation, they address three challenges: selecting informative feature channels, highlighting relevant spatial regions, and capturing long-range pixel dependencies.

### Channel Attention: Squeeze-and-Excitation

Channel attention learns to emphasize informative feature channels by computing per-channel importance weights:

```python
import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel attention.
    
    1. Squeeze: Global average pooling captures channel-wise statistics
    2. Excitation: FC layers learn channel interdependencies
    3. Scale: Reweight features by learned channel weights
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

### Spatial Attention

Spatial attention identifies *where* to focus by computing location-wise importance:

```python
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Uses channel-wise max and average pooling to compute spatial
    attention weights, highlighting important spatial locations.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention
```

### CBAM: Combined Channel and Spatial Attention

The Convolutional Block Attention Module applies channel attention followed by spatial attention sequentially:

```python
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Sequentially applies channel and spatial attention for
    comprehensive feature refinement.
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial attention
        padding = kernel_size // 2
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_ch = self.fc(self.avg_pool(x))
        max_ch = self.fc(self.max_pool(x))
        channel_att = torch.sigmoid(avg_ch + max_ch)
        x = x * channel_att
        
        # Spatial attention
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_sp, max_sp], dim=1)))
        x = x * spatial_att
        
        return x
```

### Attention Gates for Skip Connections

Attention gates filter encoder features before they reach the decoder, learning to suppress irrelevant spatial regions. This is particularly effective in U-Net architectures:

```python
class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections.
    
    Filters encoder features based on decoder context, focusing
    on relevant regions and suppressing irrelevant ones.
    
    Args:
        gate_channels: Channels in gating signal (from decoder)
        skip_channels: Channels in skip connection (from encoder)
        inter_channels: Intermediate channels for attention computation
    """
    def __init__(self, gate_channels: int, skip_channels: int, 
                 inter_channels: int = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = skip_channels // 2
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[2:] != skip.shape[2:]:
            gate = nn.functional.interpolate(
                gate, size=skip.shape[2:], mode='bilinear', align_corners=True
            )
        g = self.W_g(gate)
        x = self.W_x(skip)
        attention = self.psi(self.relu(g + x))
        return skip * attention
```

### Self-Attention for Long-Range Dependencies

Standard convolutions have limited receptive fields. Self-attention captures relationships between all spatial positions:

```python
class SelfAttention2D(nn.Module):
    """
    Self-attention for capturing long-range spatial dependencies.
    
    Computes attention between all spatial positions, relating
    distant regions of the feature map.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        inter_channels = channels // reduction
        
        self.query = nn.Conv2d(channels, inter_channels, 1)
        self.key = nn.Conv2d(channels, inter_channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        value = self.value(x).view(B, -1, H * W)
        
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(B, C, H, W)
        
        return self.gamma * out + x
```

| Attention Type | Purpose | Typical IoU Gain |
|---------------|---------|-----------------|
| Channel (SE) | Feature selection | +0.5–1.0% |
| Spatial | Region focus | +0.5–1.5% |
| CBAM | Both | +1.0–2.0% |
| Attention Gate | Skip filtering | +1.0–3.0% |
| Self-Attention | Long-range context | +1.0–2.0% |

## Data Augmentation for Segmentation

### Synchronized Transforms

The critical constraint in segmentation augmentation: **identical geometric transforms** must be applied to both image and mask. Color/intensity transforms apply only to the image.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation(image_size: int = 512):
    """Comprehensive training augmentation with Albumentations."""
    return A.Compose([
        # Geometric transforms (applied to both image and mask)
        A.RandomResizedCrop(image_size, image_size, scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        
        # Elastic deformation (effective for medical images)
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1),
            A.GridDistortion(p=1),
        ], p=0.3),
        
        # Color transforms (image only—Albumentations handles this automatically)
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.HueSaturationValue(p=1),
        ], p=0.5),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_validation_augmentation(image_size: int = 512):
    """Validation: resize and normalize only."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

## Test-Time Augmentation (TTA)

TTA averages predictions from multiple augmented views, typically improving mIoU by 1–3%:

```python
import torch.nn.functional as F

def test_time_augmentation(model, image, scales=[0.75, 1.0, 1.25], use_flip=True):
    """
    Multi-scale TTA with optional horizontal flipping.
    
    Args:
        model: Trained segmentation model
        image: Input tensor (1, C, H, W)
        scales: Resize scales to evaluate
        use_flip: Whether to include horizontal flip
    
    Returns:
        Averaged probability map (1, K, H, W)
    """
    model.eval()
    original_size = image.shape[2:]
    predictions = []
    
    with torch.no_grad():
        for scale in scales:
            size = (int(original_size[0] * scale), int(original_size[1] * scale))
            scaled = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
            
            pred = F.softmax(model(scaled), dim=1)
            pred = F.interpolate(pred, size=original_size, mode='bilinear', align_corners=False)
            predictions.append(pred)
            
            if use_flip:
                flipped = torch.flip(scaled, dims=[3])
                pred_flip = F.softmax(model(flipped), dim=1)
                pred_flip = torch.flip(pred_flip, dims=[3])
                pred_flip = F.interpolate(pred_flip, size=original_size, 
                                          mode='bilinear', align_corners=False)
                predictions.append(pred_flip)
    
    return torch.stack(predictions).mean(dim=0)
```

## Mixed Precision Training

FP16 training reduces memory usage by ~50% and accelerates computation on modern GPUs:

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, train_loader, criterion, optimizer, device):
    """Training loop with automatic mixed precision."""
    scaler = GradScaler()
    model.train()
    
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## Learning Rate Scheduling

### Polynomial Decay (Standard for DeepLab)

$$\text{lr} = \text{lr}_{\text{base}} \cdot \left(1 - \frac{\text{iter}}{\text{max\_iter}}\right)^{0.9}$$

```python
class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial learning rate decay, standard for DeepLab training."""
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
        self.max_iters = max_iters
        self.power = power
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [
            base_lr * (1 - self.last_epoch / self.max_iters) ** self.power
            for base_lr in self.base_lrs
        ]
```

### Warmup + Cosine Decay

```python
import math

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (1 - min_lr) * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## Complete Training Pipeline

```python
def train_segmentation_model(model, train_loader, val_loader, num_epochs=50,
                              lr=1e-4, device='cuda', use_amp=True):
    """Complete segmentation training pipeline with best practices."""
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = CombinedSegmentationLoss(ce_weight=0.5, dice_weight=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )
    scaler = GradScaler() if use_amp else None
    
    best_iou = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        val_iou = evaluate_model(model, val_loader, device)
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, IoU={val_iou:.4f}")
    
    return best_iou
```

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| OOM errors | Reduce batch size, use gradient checkpointing, mixed precision |
| Poor boundaries | Add boundary loss, use attention gates |
| Class imbalance | Use Focal or Dice loss, weighted sampling |
| Slow convergence | Pretrained encoder, learning rate warmup |
| Overfitting | Stronger augmentation, dropout, early stopping |

## Summary

Effective semantic segmentation combines architectural design with training methodology:

1. **Attention mechanisms** (SE, CBAM, attention gates, self-attention) enhance feature quality with 1–3% IoU gains
2. **Synchronized augmentation** is mandatory—geometric transforms must be identical for image and mask
3. **TTA** provides 1–3% inference-time improvement at the cost of $N\times$ compute
4. **Mixed precision** cuts memory in half with negligible accuracy impact
5. **Loss function selection** (covered in the dedicated Loss Functions page) is critical for handling class imbalance and boundary precision

## References

1. Hu, J., et al. (2018). Squeeze-and-Excitation Networks. CVPR.
2. Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. ECCV.
3. Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. MIDL.
4. Wang, X., et al. (2018). Non-local Neural Networks. CVPR.
