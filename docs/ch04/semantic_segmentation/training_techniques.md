# Training Techniques and Best Practices

## Learning Objectives

By the end of this section, you will be able to:

- Implement effective data augmentation strategies for segmentation
- Apply multi-scale training and inference techniques
- Use test-time augmentation (TTA) to boost performance
- Implement mixed precision training for efficiency
- Apply proper learning rate scheduling

## Data Augmentation for Segmentation

### Key Principle: Synchronized Transforms

Segmentation augmentation must apply **identical geometric transforms** to both image and mask.

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
        
        # Elastic deformation
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1),
            A.GridDistortion(p=1),
        ], p=0.3),
        
        # Color transforms (image only)
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.HueSaturationValue(p=1),
        ], p=0.5),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

## Test-Time Augmentation (TTA)

TTA averages predictions from multiple augmented versions of the input:

```python
import torch
import torch.nn.functional as F

def test_time_augmentation(model, image, scales=[0.75, 1.0, 1.25], use_flip=True):
    """
    TTA with multi-scale and flip augmentation.
    Typically improves IoU by 1-3%.
    """
    model.eval()
    original_size = image.shape[2:]
    predictions = []
    
    with torch.no_grad():
        for scale in scales:
            # Resize
            size = (int(original_size[0] * scale), int(original_size[1] * scale))
            scaled = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
            
            # Original
            pred = F.softmax(model(scaled), dim=1)
            pred = F.interpolate(pred, size=original_size, mode='bilinear', align_corners=False)
            predictions.append(pred)
            
            if use_flip:
                # Horizontal flip
                flipped = torch.flip(scaled, dims=[3])
                pred_flip = F.softmax(model(flipped), dim=1)
                pred_flip = torch.flip(pred_flip, dims=[3])
                pred_flip = F.interpolate(pred_flip, size=original_size, mode='bilinear', align_corners=False)
                predictions.append(pred_flip)
    
    return torch.stack(predictions).mean(dim=0)
```

## Mixed Precision Training

Mixed precision (FP16) training accelerates training and reduces memory:

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
    
    return loss.item()
```

## Learning Rate Scheduling

### Polynomial Decay (Standard for DeepLab)

```python
class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """lr = base_lr * (1 - iter/max_iter)^power"""
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
    """Complete training pipeline with best practices."""
    model = model.to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Combined loss
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )
    
    # Mixed precision
    scaler = GradScaler() if use_amp else None
    
    best_iou = 0.0
    
    for epoch in range(num_epochs):
        # Training
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
        
        # Validation
        val_iou = evaluate_model(model, val_loader, device)
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, IoU={val_iou:.4f}")
    
    return best_iou
```

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| OOM errors | Reduce batch size, use gradient checkpointing |
| Poor boundaries | Add boundary loss, use attention |
| Class imbalance | Use Focal or Dice loss, weighted sampling |
| Slow convergence | Use pretrained encoder, warmup LR |
| Overfitting | More augmentation, dropout, early stopping |

## Summary

Effective segmentation training combines:

1. **Strong augmentation** with geometric + intensity transforms
2. **Multi-scale training** for scale invariance
3. **Mixed precision** for efficiency
4. **Appropriate loss functions** for the task
5. **TTA at inference** for accuracy boost
