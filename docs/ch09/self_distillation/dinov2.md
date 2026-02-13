# DINOv2: Learning Robust Visual Features without Supervision

DINOv2 scales the DINO self-distillation approach to produce universal visual features that work across a wide range of tasks without fine-tuning.

## Key Improvements over DINO

| Aspect | DINO | DINOv2 |
|--------|------|--------|
| Data | ImageNet-1K | LVD-142M (curated) |
| Architecture | ViT-S/B | ViT-S/B/L/g |
| Training objective | Self-distillation | Self-distillation + iBOT |
| Regularisation | Centering + sharpening | + KoLeo + Sinkhorn-Knopp |
| Features | Good linear probing | Excellent across all protocols |

## Combined Objective

DINOv2 combines image-level self-distillation (DINO) with patch-level masked modeling (iBOT):

$$\mathcal{L} = \mathcal{L}_{\text{DINO}}^{\text{[CLS]}} + \mathcal{L}_{\text{iBOT}}^{\text{patches}}$$

The DINO loss operates on [CLS] tokens for global representation learning, while the iBOT loss operates on masked patch tokens for dense feature learning.

## Data Curation Pipeline

DINOv2 introduces an automated data curation pipeline:

1. Start with a large uncurated pool of web images
2. Use pretrained features to deduplicate
3. Self-supervised retrieval to find images similar to curated datasets
4. Result: LVD-142M—a large, diverse, and high-quality dataset

## Usage

```python
import torch

# Load pretrained DINOv2 model
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitl14.eval()

with torch.no_grad():
    # Image-level features (CLS token)
    features = dinov2_vitl14(images)  # (B, 1024) for ViT-L
    
    # Patch-level features (for dense prediction)
    patch_features = dinov2_vitl14.forward_features(images)
```

## Feature Quality

DINOv2 features work remarkably well as frozen features across diverse tasks:

| Task | Method | Performance |
|------|--------|------------|
| ImageNet classification | Linear probe | 86.3% (ViT-g) |
| ADE20K segmentation | Linear probe | 49.0 mIoU |
| Depth estimation | Linear probe | Competitive with task-specific models |
| Instance retrieval | k-NN | State-of-the-art |

## Practical Applications

DINOv2 is particularly valuable for:

1. **Feature backbone**: Frozen feature extractor for any visual task
2. **Few-shot learning**: Strong features enable learning from very few examples
3. **Dense prediction**: Patch tokens provide strong spatial features
4. **Retrieval**: Excellent for image similarity and retrieval tasks

## References

1. Oquab, M., et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." *TMLR*.
2. Darcet, T., et al. (2024). "Vision Transformers Need Registers." *ICLR*.
