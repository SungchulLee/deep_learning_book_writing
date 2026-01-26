# Panoptic Segmentation

## Learning Objectives

By the end of this section, you will be able to:

- Understand panoptic segmentation as a unified task
- Distinguish "stuff" (amorphous regions) from "things" (countable objects)
- Explain the Panoptic Quality (PQ) metric
- Implement panoptic segmentation with modern architectures

## What is Panoptic Segmentation?

Panoptic segmentation unifies semantic and instance segmentation:

```
Semantic Segmentation:  Assigns class to every pixel (no instance distinction)
Instance Segmentation:  Detects and masks individual objects (ignores stuff)
Panoptic Segmentation:  Both - every pixel gets (class, instance_id)
```

### Stuff vs Things

| Category | Description | Examples | Treatment |
|----------|-------------|----------|-----------|
| **Stuff** | Amorphous regions, uncountable | Sky, road, grass, water | Semantic only |
| **Things** | Countable objects | Person, car, dog | Instance-aware |

```
Panoptic Output:
┌─────────────────────────────────────────┐
│   sky (stuff)                           │
│ ┌───────┐ ┌───────┐                    │
│ │person1│ │person2│  ← things (instances)│
│ └───────┘ └───────┘                    │
│                                         │
│   grass (stuff)                         │
└─────────────────────────────────────────┘
```

## Panoptic Quality (PQ) Metric

PQ evaluates both segmentation and recognition quality:

$$PQ = \frac{\sum_{(p,g) \in TP} IoU(p,g)}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}$$

This decomposes into:

$$PQ = \underbrace{SQ}_{\text{Segmentation Quality}} \times \underbrace{RQ}_{\text{Recognition Quality}}$$

where:
- $SQ = \frac{\sum_{(p,g) \in TP} IoU(p,g)}{|TP|}$ (average IoU of matched segments)
- $RQ = \frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}$ (F1 score)

```python
def compute_panoptic_quality(pred_segments, gt_segments, 
                              pred_labels, gt_labels, iou_threshold=0.5):
    """
    Compute Panoptic Quality metric.
    
    Args:
        pred_segments: Dict mapping instance_id to binary mask
        gt_segments: Dict mapping instance_id to binary mask
        pred_labels: Dict mapping instance_id to class label
        gt_labels: Dict mapping instance_id to class label
        iou_threshold: IoU threshold for matching (default 0.5)
    
    Returns:
        Dictionary with PQ, SQ, RQ metrics
    """
    matched_pred = set()
    matched_gt = set()
    iou_sum = 0.0
    
    # Match predictions to ground truth
    for gt_id, gt_mask in gt_segments.items():
        best_iou = 0.0
        best_pred_id = None
        
        for pred_id, pred_mask in pred_segments.items():
            if pred_id in matched_pred:
                continue
            if pred_labels.get(pred_id) != gt_labels.get(gt_id):
                continue
            
            # Compute IoU
            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            iou = intersection / (union + 1e-6)
            
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_pred_id = pred_id
        
        if best_pred_id is not None:
            matched_pred.add(best_pred_id)
            matched_gt.add(gt_id)
            iou_sum += best_iou
    
    # Compute metrics
    tp = len(matched_gt)
    fp = len(pred_segments) - len(matched_pred)
    fn = len(gt_segments) - len(matched_gt)
    
    sq = iou_sum / tp if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq
    
    return {'PQ': pq, 'SQ': sq, 'RQ': rq, 'TP': tp, 'FP': fp, 'FN': fn}
```

## Architecture Approaches

### Bottom-Up (Merging)

1. Run semantic segmentation
2. Run instance segmentation
3. Merge results (heuristic post-processing)

### Top-Down (Unified)

Modern end-to-end approaches:

#### Panoptic FPN

Extends Mask R-CNN with semantic segmentation branch:

```python
import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn

class PanopticFPN(nn.Module):
    """
    Simplified Panoptic FPN architecture.
    
    Combines instance segmentation (Mask R-CNN style) with
    semantic segmentation for stuff classes.
    """
    def __init__(self, num_things, num_stuff):
        super().__init__()
        
        # Shared backbone with FPN
        self.backbone = ...  # ResNet + FPN
        
        # Instance head (for things)
        self.instance_head = ...  # RPN + Box/Mask heads
        
        # Semantic head (for stuff)
        self.semantic_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_stuff, 1)
        )
    
    def forward(self, images):
        features = self.backbone(images)
        
        # Instance predictions
        instances = self.instance_head(features)
        
        # Semantic predictions (for stuff)
        semantic = self.semantic_head(features['p2'])  # Highest resolution
        
        return {
            'instances': instances,
            'semantic': semantic
        }
```

#### DETR for Panoptic

DETR (DEtection TRansformer) naturally extends to panoptic:

```python
# Using HuggingFace Transformers
from transformers import DetrForSegmentation, DetrImageProcessor

def run_panoptic_detr(image):
    """Run panoptic segmentation with DETR."""
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process to get panoptic segmentation
    result = processor.post_process_panoptic_segmentation(
        outputs, 
        target_sizes=[(image.height, image.width)]
    )[0]
    
    return result  # Contains 'segmentation' map and 'segments_info'
```

#### MaskFormer / Mask2Former

State-of-the-art unified architecture:

- Predicts set of mask embeddings via transformer
- Each embedding generates a binary mask
- Works for semantic, instance, and panoptic

```python
from transformers import Mask2FormerForUniversalSegmentation

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-base-coco-panoptic"
)
```

## Merging Instance and Semantic Predictions

When using separate models, merge predictions heuristically:

```python
def merge_to_panoptic(semantic_pred, instance_pred, stuff_classes, thing_classes):
    """
    Merge semantic and instance predictions into panoptic format.
    
    Args:
        semantic_pred: (H, W) class predictions
        instance_pred: Dict with 'masks', 'labels', 'scores'
        stuff_classes: List of stuff class indices
        thing_classes: List of thing class indices
    
    Returns:
        panoptic_seg: (H, W) with unique IDs
        segments_info: List of dicts with segment metadata
    """
    H, W = semantic_pred.shape
    panoptic_seg = torch.zeros((H, W), dtype=torch.int64)
    segments_info = []
    current_id = 1
    
    # Add instance predictions (things) first - they override stuff
    for mask, label, score in zip(
        instance_pred['masks'],
        instance_pred['labels'],
        instance_pred['scores']
    ):
        if score < 0.5:
            continue
        
        mask_binary = mask.squeeze() > 0.5
        panoptic_seg[mask_binary] = current_id
        
        segments_info.append({
            'id': current_id,
            'category_id': label.item(),
            'isthing': True,
            'score': score.item()
        })
        current_id += 1
    
    # Add stuff regions (where no instance was placed)
    for stuff_class in stuff_classes:
        stuff_mask = (semantic_pred == stuff_class) & (panoptic_seg == 0)
        
        if stuff_mask.sum() > 0:
            panoptic_seg[stuff_mask] = current_id
            segments_info.append({
                'id': current_id,
                'category_id': stuff_class,
                'isthing': False
            })
            current_id += 1
    
    return panoptic_seg, segments_info
```

## Applications

Panoptic segmentation enables richer scene understanding:

| Application | Why Panoptic? |
|-------------|---------------|
| Autonomous driving | Need to track cars (instance) AND know road layout (stuff) |
| Robotics | Navigate around objects AND understand surfaces |
| Image editing | Select individual objects AND region-based editing |
| Scene understanding | Complete pixel-level annotation |

## Summary

Panoptic segmentation provides complete scene understanding:

1. **Unifies** semantic (stuff) and instance (things) segmentation
2. **PQ metric** evaluates both segmentation and recognition quality
3. **Modern approaches** (DETR, Mask2Former) are end-to-end trainable
4. **Bottom-up merging** works but unified approaches are superior

## References

1. Kirillov, A., et al. (2019). Panoptic Segmentation. CVPR.
2. Kirillov, A., et al. (2019). Panoptic Feature Pyramid Networks. CVPR.
3. Carion, N., et al. (2020). End-to-End Object Detection with Transformers (DETR). ECCV.
4. Cheng, B., et al. (2022). Masked-attention Mask Transformer for Universal Image Segmentation. CVPR.
