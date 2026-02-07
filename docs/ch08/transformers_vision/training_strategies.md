# Vision Transformer Training Strategies

## Introduction

Training Vision Transformers effectively requires different strategies than CNNs due to their lack of inductive biases and higher data requirements. This guide covers the essential techniques for successful ViT training.

## Key Differences from CNN Training

| Aspect | CNN | ViT |
|--------|-----|-----|
| Learning rate | 0.1 (SGD) | 1e-4 to 1e-3 (AdamW) |
| Warmup | Optional | Essential |
| Data augmentation | Moderate | Heavy |
| Regularization | Dropout, weight decay | + Stochastic depth, mixup |
| Epochs | 90-100 | 300+ |
| Batch size | 256 | 1024+ |

## Optimizer Configuration

### AdamW is Standard

```python
import torch
import torch.nn as nn

def create_optimizer(model: nn.Module, 
                    lr: float = 1e-3,
                    weight_decay: float = 0.05) -> torch.optim.AdamW:
    """
    Create AdamW optimizer with proper weight decay handling.
    
    Key points:
    - Don't apply weight decay to bias terms and LayerNorm
    - Use decoupled weight decay (AdamW, not Adam + L2)
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # No decay for bias, LayerNorm, and embeddings
        if 'bias' in name or 'norm' in name or 'embed' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
```

### Layer-wise Learning Rate Decay

For fine-tuning, use lower learning rates for earlier layers:

```python
def create_layerwise_optimizer(model: nn.Module,
                               base_lr: float = 1e-3,
                               weight_decay: float = 0.05,
                               layer_decay: float = 0.75) -> torch.optim.AdamW:
    """
    Create optimizer with layer-wise learning rate decay.
    
    Earlier layers get smaller learning rates than later layers.
    """
    param_groups = []
    num_layers = len(model.blocks)
    
    # Embedding layers
    param_groups.append({
        'params': list(model.patch_embed.parameters()) + 
                  [model.cls_token, model.pos_embed],
        'lr': base_lr * (layer_decay ** num_layers),
        'weight_decay': 0.0  # No decay for embeddings
    })
    
    # Transformer blocks
    for layer_idx, block in enumerate(model.blocks):
        lr_scale = layer_decay ** (num_layers - layer_idx - 1)
        
        for name, param in block.named_parameters():
            wd = 0.0 if 'bias' in name or 'norm' in name else weight_decay
            param_groups.append({
                'params': [param],
                'lr': base_lr * lr_scale,
                'weight_decay': wd
            })
            
    # Classification head
    param_groups.append({
        'params': list(model.head.parameters()),
        'lr': base_lr,
        'weight_decay': weight_decay
    })
    
    return torch.optim.AdamW(param_groups)
```

## Learning Rate Schedule

### Warmup + Cosine Decay

The standard schedule for ViT training:

```python
import math

class WarmupCosineScheduler:
    """
    Learning rate schedule with linear warmup and cosine decay.
    
    lr = base_lr * warmup_factor during warmup
    lr = base_lr * 0.5 * (1 + cos(pi * progress)) after warmup
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6, warmup_start_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch: int):
        """Update learning rate for given epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.min_lr + 0.5 * (base_lr - self.min_lr) * \
                                    (1 + math.cos(math.pi * progress))
                                    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
```

## Data Augmentation

### Standard ViT Augmentation

```python
from torchvision import transforms
from timm.data import create_transform
from timm.data.auto_augment import rand_augment_transform

def create_train_transform(img_size: int = 224,
                          color_jitter: float = 0.4,
                          auto_augment: str = 'rand-m9-mstd0.5-inc1',
                          reprob: float = 0.25,
                          remode: str = 'pixel'):
    """
    Create training augmentation pipeline for ViT.
    """
    # Primary transform
    primary_tfl = [
        transforms.RandomResizedCrop(
            img_size, 
            scale=(0.08, 1.0),
            ratio=(0.75, 1.333),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    
    # Color augmentation
    color_tfl = [
        transforms.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=color_jitter * 0.25
        )
    ]
    
    # AutoAugment / RandAugment
    if auto_augment:
        aa_transform = rand_augment_transform(
            auto_augment,
            {'translate_const': int(img_size * 0.45)}
        )
        primary_tfl.append(aa_transform)
    else:
        primary_tfl.extend(color_tfl)
    
    # Final processing
    final_tfl = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    # Random erasing
    if reprob > 0:
        from timm.data.random_erasing import RandomErasing
        final_tfl.append(RandomErasing(
            probability=reprob,
            mode=remode,
            max_count=1
        ))
    
    return transforms.Compose(primary_tfl + final_tfl)


def create_val_transform(img_size: int = 224):
    """Create validation transform."""
    return transforms.Compose([
        transforms.Resize(
            int(img_size * 1.143),  # 256 for img_size=224
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
```

### Mixup and CutMix

```python
import torch
import numpy as np

class MixupCutmix:
    """
    Mixup and CutMix augmentation for batch-level mixing.
    
    Crucial for ViT training - provides regularization and 
    prevents overfitting.
    """
    def __init__(self, 
                 mixup_alpha: float = 0.8,
                 cutmix_alpha: float = 1.0,
                 prob: float = 1.0,
                 switch_prob: float = 0.5,
                 num_classes: int = 1000,
                 label_smoothing: float = 0.1):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
    def __call__(self, images: torch.Tensor, 
                 targets: torch.Tensor) -> tuple:
        """
        Apply mixup or cutmix to batch.
        
        Returns:
            Mixed images and soft labels
        """
        if np.random.rand() > self.prob:
            return images, self._smooth_labels(targets)
            
        if np.random.rand() < self.switch_prob:
            return self._mixup(images, targets)
        else:
            return self._cutmix(images, targets)
            
    def _mixup(self, images: torch.Tensor, 
               targets: torch.Tensor) -> tuple:
        """Apply mixup: linear interpolation of images and labels."""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Soft labels
        targets_a = self._smooth_labels(targets)
        targets_b = self._smooth_labels(targets[index])
        mixed_targets = lam * targets_a + (1 - lam) * targets_b
        
        return mixed_images, mixed_targets
        
    def _cutmix(self, images: torch.Tensor,
                targets: torch.Tensor) -> tuple:
        """Apply cutmix: patch replacement from another image."""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        _, _, H, W = images.shape
        
        # Sample bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # Soft labels
        targets_a = self._smooth_labels(targets)
        targets_b = self._smooth_labels(targets[index])
        mixed_targets = lam * targets_a + (1 - lam) * targets_b
        
        return images, mixed_targets
        
    def _smooth_labels(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert to one-hot with label smoothing."""
        one_hot = torch.zeros(targets.size(0), self.num_classes, 
                             device=targets.device)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        smooth = self.label_smoothing / self.num_classes
        one_hot = one_hot * (1 - self.label_smoothing) + smooth
        
        return one_hot
```

## Regularization Techniques

### Stochastic Depth (DropPath)

```python
class DropPath(nn.Module):
    """
    Stochastic Depth: randomly drop entire residual branches.
    
    Essential for training deep ViTs - enables training of 
    models with 24+ layers.
    """
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        
        output = x.div(keep_prob) * random_tensor
        return output


class TransformerBlockWithDropPath(nn.Module):
    """Transformer block with stochastic depth."""
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.,
                 drop_path: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

### Progressive Drop Path Rates

```python
def get_drop_path_rates(depth: int, max_drop_path: float = 0.1) -> list:
    """
    Get linearly increasing drop path rates for each block.
    
    Earlier layers have lower drop probability.
    """
    return [x.item() for x in torch.linspace(0, max_drop_path, depth)]
```

## Complete Training Loop

```python
class ViTTrainer:
    """Complete training pipeline for Vision Transformer."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda',
                 n_classes: int = 1000):
        self.model = model.to(device)
        self.device = device
        self.n_classes = n_classes
        
    def train(self, 
              train_loader,
              val_loader,
              epochs: int = 300,
              lr: float = 1e-3,
              weight_decay: float = 0.05,
              warmup_epochs: int = 5,
              label_smoothing: float = 0.1,
              mixup_alpha: float = 0.8,
              cutmix_alpha: float = 1.0,
              drop_path_rate: float = 0.1,
              clip_grad: float = 1.0):
        """
        Full training loop with all ViT best practices.
        """
        # Setup
        optimizer = create_optimizer(self.model, lr, weight_decay)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        mixup = MixupCutmix(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            num_classes=self.n_classes,
            label_smoothing=label_smoothing
        )
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_acc = 0.0
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            
            for images, targets in train_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Apply mixup/cutmix
                images, mixed_targets = mixup(images, targets)
                
                # Forward
                optimizer.zero_grad()
                outputs = self.model(images)
                
                # Soft cross-entropy for mixed labels
                loss = -(mixed_targets * outputs.log_softmax(dim=-1)).sum(dim=-1).mean()
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), clip_grad
                    )
                
                optimizer.step()
                train_loss += loss.item()
                
            # Update learning rate
            scheduler.step(epoch)
            
            # Validate
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            # Record history
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            
        return history
    
    @torch.no_grad()
    def _validate(self, val_loader, criterion):
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in val_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        return total_loss / len(val_loader), correct / total
```

## Best Practices Summary

1. **Optimizer**: AdamW with decoupled weight decay
2. **Learning Rate**: 1e-3 with warmup and cosine decay
3. **Augmentation**: RandAugment + Mixup + CutMix
4. **Regularization**: Label smoothing + Stochastic depth
5. **Epochs**: 300+ for training from scratch
6. **Batch Size**: As large as possible (1024+)
7. **Gradient Clipping**: Clip at 1.0 for stability

## References

1. Touvron, H., et al. "Training data-efficient image transformers." ICML 2021.
2. Wightman, R., et al. "ResNet strikes back." NeurIPS Workshop 2021.
3. Steiner, A., et al. "How to train your ViT?" 2021.
