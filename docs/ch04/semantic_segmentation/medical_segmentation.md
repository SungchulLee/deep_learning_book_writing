# Medical Image Segmentation

## Learning Objectives

By the end of this section, you will be able to:

- Understand unique challenges in medical imaging segmentation
- Implement specialized loss functions for clinical applications
- Apply proper evaluation metrics (Dice, Sensitivity, Specificity)
- Handle extreme class imbalance in lesion segmentation
- Follow best practices for medical AI validation

## Challenges in Medical Imaging

Medical image segmentation presents unique challenges:

| Challenge | Description | Solution |
|-----------|-------------|----------|
| Limited data | 10s to 100s of images typical | Strong augmentation, transfer learning |
| Class imbalance | Lesions are tiny fraction of image | Dice loss, Focal loss |
| Annotation cost | Expert radiologists required | Semi-supervised learning |
| High stakes | Errors have clinical impact | Uncertainty quantification |
| 3D volumes | CT/MRI are volumetric | 3D architectures, slice-by-slice |

## Domain-Specific Loss Functions

### Dice Loss for Medical Imaging

Dice loss is the de facto standard in medical segmentation:

```python
import torch
import torch.nn as nn

class MedicalDiceLoss(nn.Module):
    """
    Dice Loss optimized for medical imaging.
    
    Handles extreme class imbalance by focusing on overlap
    rather than per-pixel accuracy.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        return 1 - dice
```

### Combined Loss for Clinical Applications

```python
class MedicalCombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss commonly used in medical imaging.
    
    BCE provides stable gradients, Dice handles imbalance.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = MedicalDiceLoss()
    
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
```

## Clinical Evaluation Metrics

### Comprehensive Metric Suite

```python
def calculate_medical_metrics(pred: torch.Tensor, target: torch.Tensor,
                               threshold: float = 0.5) -> dict:
    """
    Calculate clinically relevant segmentation metrics.
    
    Returns:
        Dictionary with Dice, Sensitivity, Specificity, Precision
    """
    with torch.no_grad():
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        
        # Flatten for calculation
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        # Components
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()
        
        eps = 1e-6
        
        # Metrics
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        sensitivity = (tp + eps) / (tp + fn + eps)  # Recall / True Positive Rate
        specificity = (tn + eps) / (tn + fp + eps)  # True Negative Rate
        precision = (tp + eps) / (tp + fp + eps)
        
        return {
            'dice': dice.item(),
            'sensitivity': sensitivity.item(),  # Critical: detect lesions
            'specificity': specificity.item(),  # Avoid false alarms
            'precision': precision.item()
        }
```

### Why Sensitivity Matters

In medical imaging, **sensitivity (recall)** is often more critical than specificity:

- **High sensitivity**: Few missed lesions (low false negatives)
- **Missing a tumor** is worse than a false alarm
- Tune threshold to prioritize sensitivity over specificity

```python
def find_optimal_threshold(model, val_loader, target_sensitivity=0.95):
    """Find threshold that achieves target sensitivity."""
    model.eval()
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            probs = torch.sigmoid(model(images))
            all_probs.append(probs.cpu())
            all_targets.append(masks.cpu())
    
    all_probs = torch.cat(all_probs).view(-1)
    all_targets = torch.cat(all_targets).view(-1)
    
    # Binary search for threshold
    for threshold in torch.linspace(0.01, 0.99, 100):
        pred = (all_probs > threshold).float()
        tp = (pred * all_targets).sum()
        fn = ((1 - pred) * all_targets).sum()
        sensitivity = tp / (tp + fn + 1e-6)
        
        if sensitivity >= target_sensitivity:
            return threshold.item()
    
    return 0.5  # Default
```

## Data Validation Best Practices

### Patient-Level Splits

**Critical**: Never split by image, always by patient to prevent data leakage.

```python
from sklearn.model_selection import GroupKFold

def create_patient_level_splits(patient_ids, n_splits=5):
    """
    Create train/val/test splits at patient level.
    
    Prevents data leakage where same patient appears in
    both training and validation sets.
    """
    group_kfold = GroupKFold(n_splits=n_splits)
    
    splits = []
    dummy_X = range(len(patient_ids))
    
    for train_idx, val_idx in group_kfold.split(dummy_X, groups=patient_ids):
        splits.append({
            'train': train_idx,
            'val': val_idx
        })
    
    return splits
```

### Cross-Validation for Small Datasets

```python
def medical_cross_validation(model_class, dataset, patient_ids, n_folds=5):
    """
    K-fold cross-validation with patient-level splits.
    """
    splits = create_patient_level_splits(patient_ids, n_folds)
    fold_results = []
    
    for fold, split in enumerate(splits):
        print(f"Training fold {fold + 1}/{n_folds}")
        
        # Create data subsets
        train_subset = torch.utils.data.Subset(dataset, split['train'])
        val_subset = torch.utils.data.Subset(dataset, split['val'])
        
        # Train model
        model = model_class()
        metrics = train_and_evaluate(model, train_subset, val_subset)
        fold_results.append(metrics)
    
    # Aggregate results
    mean_dice = np.mean([r['dice'] for r in fold_results])
    std_dice = np.std([r['dice'] for r in fold_results])
    
    print(f"Cross-validation Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    return fold_results
```

## U-Net for Medical Imaging

U-Net was specifically designed for biomedical segmentation:

```python
class MedicalUNet(nn.Module):
    """
    U-Net optimized for medical imaging applications.
    
    Features:
    - Deep supervision for better gradients
    - Dropout for regularization (small datasets)
    - Optional attention gates
    """
    def __init__(self, in_channels=1, num_classes=1, use_attention=True):
        super().__init__()
        
        features = [64, 128, 256, 512, 1024]
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_ch = in_channels
        for f in features[:-1]:
            self.encoders.append(self._double_conv(in_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = f
        
        # Bottleneck
        self.bottleneck = self._double_conv(features[-2], features[-1])
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if use_attention else None
        
        for i in range(len(features) - 1):
            self.upconvs.append(
                nn.ConvTranspose2d(features[-1-i], features[-2-i], 2, stride=2)
            )
            self.decoders.append(
                self._double_conv(features[-1-i], features[-2-i])
            )
            if use_attention:
                self.attention_gates.append(
                    AttentionGate(features[-2-i], features[-2-i])
                )
        
        self.final = nn.Conv2d(features[0], num_classes, 1)
        self.dropout = nn.Dropout2d(0.5)
    
    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder path
        enc_features = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            enc_features.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.dropout(x)
        
        # Decoder path
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = enc_features[-(i+1)]
            
            # Apply attention if available
            if self.attention_gates:
                skip = self.attention_gates[i](x, skip)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        return self.final(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""
    def __init__(self, gate_channels, skip_channels):
        super().__init__()
        inter_channels = skip_channels // 2
        
        self.W_g = nn.Conv2d(gate_channels, inter_channels, 1, bias=False)
        self.W_x = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate, skip):
        if gate.shape[2:] != skip.shape[2:]:
            gate = nn.functional.interpolate(gate, size=skip.shape[2:], mode='bilinear')
        
        g = self.W_g(gate)
        x = self.W_x(skip)
        attention = self.psi(self.relu(g + x))
        return skip * attention
```

## Summary

Medical image segmentation requires specialized approaches:

1. **Dice loss** handles extreme class imbalance
2. **Patient-level splits** prevent data leakage
3. **Sensitivity** is often more important than specificity
4. **Attention mechanisms** improve small lesion detection
5. **Cross-validation** essential for small datasets

## Common Medical Imaging Datasets

| Dataset | Task | Modality | Classes |
|---------|------|----------|---------|
| ISIC | Skin lesion | Dermoscopy | 2 |
| BraTS | Brain tumor | MRI | 4 |
| DRIVE | Retinal vessels | Fundus | 2 |
| LiTS | Liver tumor | CT | 3 |
| ACDC | Cardiac | MRI | 4 |

## References

1. Ronneberger, O., et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
2. Isensee, F., et al. (2021). nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation.
3. Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas.
