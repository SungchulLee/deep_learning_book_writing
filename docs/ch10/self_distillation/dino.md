# DINO: Self-Distillation with No Labels

## Learning Objectives

By the end of this section, you will be able to:

- Understand the self-distillation framework of DINO
- Implement DINO with Vision Transformers in PyTorch
- Explain how DINO produces semantically meaningful attention maps
- Apply multi-crop training strategy for improved representations
- Compare DINO with other self-supervised methods

## Introduction

DINO (Self-**Di**stillation with **No** Labels) introduces a self-supervised learning approach based on **knowledge distillation without labels**. A student network learns to match the output distribution of a teacher network, which is updated as an exponential moving average of the student.

### Key Discoveries

1. **Emergent properties**: ViT trained with DINO produces attention maps that segment objects without supervision
2. **Self-distillation works**: Teacher-student framework without labels achieves strong results
3. **k-NN classification**: DINO features work well with simple k-NN, no fine-tuning needed

## DINO Architecture

```
                 ┌──────────────────────────────────────┐
                 │           Multi-Crop Views           │
                 │  Global (224×224) + Local (96×96)    │
                 └──────────────────┬───────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
    ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
    │   Global 1  │          │   Global 2  │          │  Local 1-6  │
    └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
           │                        │                        │
           ▼                        ▼                        ▼
    ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
    │   Student   │          │   Teacher   │          │   Student   │
    │   Network   │          │   Network   │          │   Network   │
    │     θs      │          │     θt      │          │     θs      │
    └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
           │                        │                        │
           ▼                        ▼                        ▼
    ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
    │  Projection │          │  Projection │          │  Projection │
    │    Head     │          │    Head     │          │    Head     │
    └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
           │                        │                        │
           ▼                        ▼                        ▼
        ps(x)                    pt(x)                    ps(x')
           │                        │                        │
           └────────────────────────┴────────────────────────┘
                                    │
                                    ▼
                          Cross-Entropy Loss
                    H(pt(global), ps(all views))
```

## Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import copy
from PIL import ImageFilter, ImageOps
import random
import math


# =============================================================================
# Data Augmentation: Multi-Crop Strategy
# =============================================================================

class GaussianBlur:
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class Solarization:
    def __init__(self, threshold=128):
        self.threshold = threshold
    
    def __call__(self, x):
        return ImageOps.solarize(x, self.threshold)


class DINOAugmentation:
    """
    Multi-crop augmentation for DINO.
    
    Creates multiple views of each image:
    - 2 global views (224×224): Larger crops covering 40-100% of image
    - N local views (96×96): Smaller crops covering 5-40% of image
    
    The teacher only sees global views, while the student sees all views.
    This asymmetry helps the model learn local-to-global correspondences.
    """
    def __init__(
        self,
        global_crops_size=224,
        local_crops_size=96,
        n_local_crops=8,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4)
    ):
        self.n_local_crops = n_local_crops
        
        # Common transforms
        flip_and_color = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Global crop 1: with Gaussian blur
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale),
            flip_and_color,
            GaussianBlur(sigma=(0.1, 2.0)),
            normalize,
        ])
        
        # Global crop 2: with solarization
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale),
            flip_and_color,
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([Solarization()], p=0.2),
            normalize,
        ])
        
        # Local crops: smaller, different augmentation
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale),
            flip_and_color,
            transforms.RandomApply([GaussianBlur()], p=0.5),
            normalize,
        ])
    
    def __call__(self, image):
        """
        Returns:
            crops: List of tensors [global1, global2, local1, ..., localN]
        """
        crops = []
        
        # Global crops (used by both teacher and student)
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        
        # Local crops (only used by student)
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(image))
        
        return crops


# =============================================================================
# DINO Head
# =============================================================================

class DINOHead(nn.Module):
    """
    DINO projection head.
    
    Projects backbone features to a lower-dimensional space for distillation.
    Uses GELU activation and L2 normalization at the output.
    
    Architecture:
        Linear -> GELU -> Linear -> GELU -> Linear -> L2-Norm
    
    Args:
        in_dim: Input dimension (backbone output)
        out_dim: Output dimension (typically 65536 for ImageNet)
        hidden_dim: Hidden layer dimension
        bottleneck_dim: Bottleneck dimension before output
        use_bn: Whether to use batch normalization
    """
    def __init__(
        self,
        in_dim,
        out_dim=65536,
        hidden_dim=2048,
        bottleneck_dim=256,
        use_bn=False
    ):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        # Second layer
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        # Bottleneck layer
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Last layer (no bias, weight normalized)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False
    
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


# =============================================================================
# DINO Model
# =============================================================================

class DINO(nn.Module):
    """
    DINO: Self-Distillation with No Labels
    
    Key components:
    1. Student network: Trained with gradients
    2. Teacher network: EMA of student (no gradients)
    3. Centering: Prevents collapse by centering teacher outputs
    4. Sharpening: Temperature controls output distribution sharpness
    
    The loss is cross-entropy between sharpened teacher output and
    student output, summed over all view pairs where teacher sees
    global views and student sees all views.
    
    Args:
        backbone: Backbone architecture ('resnet50' or 'vit_small')
        out_dim: Output dimension of projection head
        student_temp: Temperature for student (higher = smoother)
        teacher_temp: Temperature for teacher (lower = sharper)
        center_momentum: Momentum for center vector update
        ema_momentum: Momentum for teacher EMA update
    """
    def __init__(
        self,
        backbone='vit_small',
        out_dim=65536,
        student_temp=0.1,
        teacher_temp=0.04,
        center_momentum=0.9,
        ema_momentum=0.996
    ):
        super().__init__()
        
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.ema_momentum = ema_momentum
        self.out_dim = out_dim
        
        # Build backbone
        self.student_backbone, embed_dim = self._build_backbone(backbone)
        self.teacher_backbone, _ = self._build_backbone(backbone)
        
        # Build projection heads
        self.student_head = DINOHead(embed_dim, out_dim)
        self.teacher_head = DINOHead(embed_dim, out_dim)
        
        # Initialize teacher from student
        self._init_teacher()
        
        # Center vector for preventing collapse
        self.register_buffer("center", torch.zeros(1, out_dim))
    
    def _build_backbone(self, backbone):
        """Build backbone network."""
        if backbone == 'resnet50':
            model = models.resnet50(weights=None)
            embed_dim = 2048
            model.fc = nn.Identity()
        elif backbone == 'vit_small':
            # Simplified ViT-Small (would use timm in practice)
            from torchvision.models import vit_b_16
            model = vit_b_16(weights=None)
            embed_dim = 768
            model.heads = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        return model, embed_dim
    
    def _init_teacher(self):
        """Initialize teacher as copy of student."""
        for param_s, param_t in zip(
            self.student_backbone.parameters(),
            self.teacher_backbone.parameters()
        ):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        
        for param_s, param_t in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
    
    @torch.no_grad()
    def update_teacher(self):
        """EMA update of teacher network."""
        for param_s, param_t in zip(
            self.student_backbone.parameters(),
            self.teacher_backbone.parameters()
        ):
            param_t.data = (
                self.ema_momentum * param_t.data +
                (1 - self.ema_momentum) * param_s.data
            )
        
        for param_s, param_t in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_t.data = (
                self.ema_momentum * param_t.data +
                (1 - self.ema_momentum) * param_s.data
            )
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output centering.
        
        Centering prevents collapse: if all outputs are identical,
        centering makes them all zero, giving uniform softmax = bad loss.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = (
            self.center_momentum * self.center +
            (1 - self.center_momentum) * batch_center
        )
    
    def forward(self, crops):
        """
        Forward pass for DINO.
        
        Args:
            crops: List of image crops [global1, global2, local1, ..., localN]
                   First 2 are global crops, rest are local crops
        
        Returns:
            loss: DINO loss
        """
        n_global = 2
        n_crops = len(crops)
        
        # Concatenate all crops for efficient processing
        # Global crops: (2 * B, C, 224, 224)
        # Local crops: ((n_crops - 2) * B, C, 96, 96)
        global_crops = torch.cat(crops[:n_global], dim=0)
        
        # Teacher only processes global crops
        with torch.no_grad():
            teacher_feat = self.teacher_backbone(global_crops)
            teacher_out = self.teacher_head(teacher_feat)
            
            # Center and sharpen teacher output
            teacher_out = teacher_out - self.center
            teacher_out = F.softmax(teacher_out / self.teacher_temp, dim=-1)
            
            # Split back to individual crops
            teacher_out = teacher_out.chunk(n_global)
        
        # Student processes all crops
        student_out = []
        for crop in crops:
            feat = self.student_backbone(crop)
            out = self.student_head(feat)
            out = F.log_softmax(out / self.student_temp, dim=-1)
            student_out.append(out)
        
        # Compute loss
        # Teacher sees global views, student sees all views
        # Loss = sum over all (teacher_global, student_any) pairs where they differ
        total_loss = 0
        n_loss_terms = 0
        
        for t_idx in range(n_global):  # Teacher views (global only)
            for s_idx in range(n_crops):  # Student views (all)
                if t_idx == s_idx:
                    continue  # Skip same view
                
                # Cross-entropy: H(teacher, student) = -sum(teacher * log(student))
                loss = -torch.sum(teacher_out[t_idx] * student_out[s_idx], dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        
        # Update center with all teacher outputs
        self.update_center(torch.cat([t for t in teacher_out], dim=0))
        
        return total_loss
    
    def get_attention_maps(self, img, patch_size=16):
        """
        Get attention maps from the last layer of ViT teacher.
        
        DINO produces semantically meaningful attention maps that
        can segment objects without any supervision!
        """
        # This would work with actual ViT implementation
        # Here we show the concept
        self.teacher_backbone.eval()
        
        with torch.no_grad():
            # Would extract attention weights from ViT
            # attention_maps = self.teacher_backbone.get_attention_maps(img)
            pass
        
        return None  # Placeholder


# =============================================================================
# Training
# =============================================================================

class DINOLoss(nn.Module):
    """
    Standalone DINO loss for flexibility.
    
    Can be used with custom training loops.
    """
    def __init__(
        self,
        out_dim,
        n_crops,
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=0,
        student_temp=0.1,
        center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.n_crops = n_crops
        
        self.register_buffer("center", torch.zeros(1, out_dim))
    
    def forward(self, student_output, teacher_output, epoch=0):
        """
        Args:
            student_output: List of student outputs for all crops
            teacher_output: List of teacher outputs for global crops
        """
        # Apply temperature and softmax
        student_out = [F.log_softmax(s / self.student_temp, dim=-1) 
                       for s in student_output]
        
        teacher_out = [(t - self.center) for t in teacher_output]
        teacher_out = [F.softmax(t / self.teacher_temp, dim=-1).detach()
                       for t in teacher_out]
        
        # Cross-entropy loss
        total_loss = 0
        n_loss_terms = 0
        n_global = len(teacher_output)
        
        for t_idx in range(n_global):
            for s_idx in range(len(student_output)):
                if t_idx == s_idx:
                    continue
                loss = -torch.sum(teacher_out[t_idx] * student_out[s_idx], dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        
        # Update center
        self.update_center(teacher_output)
        
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.cat(teacher_output, dim=0).mean(dim=0, keepdim=True)
        self.center = (
            self.center_momentum * self.center +
            (1 - self.center_momentum) * batch_center
        )


def train_dino_epoch(model, dataloader, optimizer, device, epoch):
    """Train DINO for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, crops in enumerate(dataloader):
        # Move all crops to device
        crops = [c.to(device) for c in crops]
        
        # Forward pass
        loss = model(crops)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        
        optimizer.step()
        
        # Update teacher
        model.update_teacher()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    return total_loss / num_batches


# =============================================================================
# Emergent Properties: Attention Visualization
# =============================================================================

def visualize_attention(model, img, patch_size=16):
    """
    Visualize DINO attention maps.
    
    DINO ViT produces attention maps where:
    - [CLS] token attends to semantically meaningful regions
    - Objects are naturally segmented without supervision
    
    This is an emergent property not seen with supervised training!
    """
    # This requires actual ViT with attention extraction
    # The attention to [CLS] token often highlights objects
    
    print("DINO Attention Analysis:")
    print("  - [CLS] attention focuses on objects")
    print("  - Different heads capture different parts")
    print("  - Can be used for unsupervised segmentation")


# =============================================================================
# k-NN Evaluation
# =============================================================================

def knn_evaluation(model, train_loader, test_loader, device, k=20, temp=0.07):
    """
    k-NN evaluation for DINO.
    
    DINO features work remarkably well with simple k-NN classification,
    demonstrating the quality of learned representations.
    """
    model.eval()
    
    # Extract features
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    
    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            feat = model.teacher_backbone(imgs)
            train_features.append(feat)
            train_labels.append(labels)
        
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            feat = model.teacher_backbone(imgs)
            test_features.append(feat)
            test_labels.append(labels)
    
    train_features = torch.cat(train_features)
    train_labels = torch.cat(train_labels).to(device)
    test_features = torch.cat(test_features)
    test_labels = torch.cat(test_labels).to(device)
    
    # Normalize
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)
    
    # k-NN
    sim = torch.mm(test_features, train_features.T)
    topk_sim, topk_idx = sim.topk(k, dim=1)
    
    # Weighted voting
    topk_labels = train_labels[topk_idx]
    weights = F.softmax(topk_sim / temp, dim=1)
    
    predictions = torch.zeros(test_features.size(0), train_labels.max() + 1, device=device)
    predictions.scatter_add_(1, topk_labels, weights)
    pred_labels = predictions.argmax(dim=1)
    
    accuracy = (pred_labels == test_labels).float().mean().item() * 100
    
    return accuracy


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = DINO(
        backbone='resnet50',  # Use 'vit_small' in practice
        out_dim=65536,
        student_temp=0.1,
        teacher_temp=0.04
    )
    
    print("\nDINO Model:")
    print(f"  Output dimension: {model.out_dim}")
    print(f"  Student temperature: {model.student_temp}")
    print(f"  Teacher temperature: {model.teacher_temp}")
    
    # Test with multi-crop
    batch_size = 4
    crops = [
        torch.randn(batch_size, 3, 224, 224),  # Global 1
        torch.randn(batch_size, 3, 224, 224),  # Global 2
        torch.randn(batch_size, 3, 96, 96),    # Local 1
        torch.randn(batch_size, 3, 96, 96),    # Local 2
    ]
    
    model.eval()
    loss = model(crops)
    print(f"\nForward pass:")
    print(f"  Loss: {loss.item():.4f}")
```

## Key Design Choices

### 1. Multi-Crop Strategy

| View Type | Size | Scale | Purpose |
|-----------|------|-------|---------|
| Global 1 | 224×224 | 0.4-1.0 | Teacher + Student |
| Global 2 | 224×224 | 0.4-1.0 | Teacher + Student |
| Local 1-8 | 96×96 | 0.05-0.4 | Student only |

Local crops encourage learning local-to-global correspondences.

### 2. Centering and Sharpening

- **Centering**: Subtracts running mean from teacher output
- **Sharpening**: Low teacher temperature (0.04) creates peaked distribution

Both prevent collapse by ensuring non-trivial outputs.

### 3. Temperature Asymmetry

- Student: τ_s = 0.1 (smoother)
- Teacher: τ_t = 0.04 (sharper)

This asymmetry is crucial for stable training.

## Emergent Properties

DINO-trained ViTs exhibit remarkable properties:

1. **Object segmentation**: [CLS] attention highlights objects
2. **k-NN works well**: Simple k-NN achieves strong accuracy
3. **Scene understanding**: Different heads capture different semantics

## Summary

| Aspect | DINO |
|--------|------|
| Framework | Self-distillation |
| Negatives | Not required |
| Multi-crop | Yes (global + local) |
| Collapse prevention | Centering + sharpening |
| Best backbone | Vision Transformer |
| Emergent property | Semantic attention maps |

## References

1. Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. ICCV.
2. Oquab, M., et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. TMLR.
