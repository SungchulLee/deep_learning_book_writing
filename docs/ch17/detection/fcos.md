# FCOS: Fully Convolutional One-Stage Detection

FCOS predicts boxes by regressing distances from each location to box edges:

```
┌─────────────────────────────────────────────────────────────────┐
│                      FCOS Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each pixel location (x, y):                                │
│                                                                  │
│       l ←───┬───→ r                                             │
│             │                                                   │
│       t     ●     (x, y)  feature map location                 │
│       ↑     │                                                   │
│       │     ↓                                                   │
│             b                                                   │
│                                                                  │
│  Predict:                                                        │
│    • (l, t, r, b): Distances to box edges                       │
│    • class score: What object (if any)?                         │
│    • centerness: How close to object center?                    │
│                                                                  │
│  Box = (x - l, y - t, x + r, y + b)                             │
└─────────────────────────────────────────────────────────────────┘
```

### FCOS Implementation

```python
class FCOSHead(nn.Module):
    """
    FCOS detection head with centerness prediction.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_convs: int = 4,
        prior_prob: float = 0.01
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Classification branch
        cls_tower = []
        for _ in range(num_convs):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)
        
        # Regression branch  
        reg_tower = []
        for _ in range(num_convs):
            reg_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_tower)
        
        # Output heads
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        
        # Learnable scale for each FPN level
        self.scales = nn.ModuleList([nn.Conv2d(1, 1, 1) for _ in range(5)])
        
        # Initialize bias for focal loss
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)
    
    def forward(self, features: list) -> tuple:
        """
        Args:
            features: List of feature maps from FPN
            
        Returns:
            cls_scores: List of (B, num_classes, H, W) per level
            bbox_preds: List of (B, 4, H, W) per level
            centernesses: List of (B, 1, H, W) per level
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []
        
        for i, feature in enumerate(features):
            cls_feat = self.cls_tower(feature)
            reg_feat = self.reg_tower(feature)
            
            # Classification
            cls_score = self.cls_logits(cls_feat)
            cls_scores.append(cls_score)
            
            # Box regression with exp for positive values
            bbox_pred = self.bbox_pred(reg_feat)
            bbox_pred = F.relu(bbox_pred) * self.scales[i](torch.ones(1, 1, 1, 1, device=feature.device))
            bbox_preds.append(bbox_pred)
            
            # Centerness (predicted from regression features)
            centerness = self.centerness(reg_feat)
            centernesses.append(centerness)
        
        return cls_scores, bbox_preds, centernesses


def compute_centerness(
    left: torch.Tensor,
    top: torch.Tensor,
    right: torch.Tensor,
    bottom: torch.Tensor
) -> torch.Tensor:
    """
    Compute centerness targets.
    
    Centerness measures how close a location is to the object center.
    Ranges from 0 (corner) to 1 (center).
    
    centerness = sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
    """
    lr_min = torch.min(left, right)
    lr_max = torch.max(left, right)
    tb_min = torch.min(top, bottom)
    tb_max = torch.max(top, bottom)
    
    centerness = torch.sqrt(
        (lr_min / (lr_max + 1e-6)) * (tb_min / (tb_max + 1e-6))
    )
    
    return centerness
```

### FCOS Training

**Positive Sample Assignment**:
- A location is positive if it's inside a ground truth box
- Multi-scale assignment: Different FPN levels handle different object sizes

**Loss Function**:
$$L = L_{cls} + \lambda_1 L_{reg} + \lambda_2 L_{centerness}$$

```python
class FCOSLoss(nn.Module):
    """FCOS loss function."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(
        self,
        cls_scores: list,
        bbox_preds: list,
        centernesses: list,
        targets: dict
    ) -> dict:
        """Compute FCOS losses."""
        
        # Flatten predictions
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for s in cls_scores
        ])
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(-1, 4)
            for b in bbox_preds
        ])
        all_centernesses = torch.cat([
            c.permute(0, 2, 3, 1).reshape(-1)
            for c in centernesses
        ])
        
        # Get targets
        labels = targets['labels']
        bbox_targets = targets['bbox_targets']
        centerness_targets = targets['centerness_targets']
        
        # Positive mask
        pos_mask = labels > 0
        num_pos = pos_mask.sum().float()
        
        # Classification loss (Focal Loss)
        cls_loss = sigmoid_focal_loss(
            all_cls_scores,
            labels,
            reduction='sum'
        ) / num_pos
        
        if pos_mask.sum() > 0:
            # Regression loss (IoU Loss)
            reg_loss = iou_loss(
                all_bbox_preds[pos_mask],
                bbox_targets[pos_mask],
                reduction='sum'
            ) / num_pos
            
            # Centerness loss (BCE)
            centerness_loss = F.binary_cross_entropy_with_logits(
                all_centernesses[pos_mask],
                centerness_targets[pos_mask],
                reduction='sum'
            ) / num_pos
        else:
            reg_loss = all_bbox_preds.sum() * 0
            centerness_loss = all_centernesses.sum() * 0
        
        return {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'centerness_loss': centerness_loss,
            'total': cls_loss + reg_loss + centerness_loss
        }
```


## Comparison: Anchor-Based vs Anchor-Free

| Aspect | Anchor-Based | Anchor-Free |
|--------|-------------|-------------|
| Hyperparameters | Anchor sizes, ratios, IoU thresholds | Fewer (no anchor design) |
| Flexibility | Fixed aspect ratios | Arbitrary shapes |
| Complexity | Anchor matching logic | Simpler training |
| Speed | NMS required | Some methods NMS-free |
| Accuracy | Mature, well-tuned | Competitive |

### When to Use Each

- **Anchor-based**: Well-understood problem with known object sizes and ratios
- **Anchor-free**: Novel domains, unusual aspect ratios, rapid prototyping

## References

1. Tian, Z., et al. (2019). FCOS: Fully Convolutional One-Stage Object Detection. ICCV.
2. Lin, T.-Y., et al. (2017). Feature Pyramid Networks for Object Detection. CVPR.
