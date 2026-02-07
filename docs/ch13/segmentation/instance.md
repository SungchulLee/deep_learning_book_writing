# Instance Segmentation Overview

## Learning Objectives

By the end of this section, you will be able to:

- Distinguish instance segmentation from semantic segmentation
- Understand the Mask R-CNN architecture
- Explain two-stage vs one-stage instance segmentation approaches
- Implement basic instance segmentation inference with pre-trained models

## Semantic vs Instance Segmentation

| Aspect | Semantic Segmentation | Instance Segmentation |
|--------|----------------------|----------------------|
| Output | Class per pixel | Class + instance ID per pixel |
| Separates | Background from classes | Individual objects |
| Example | All cars → "car" class | Car 1, Car 2, Car 3... |
| Use case | Scene understanding | Object counting, tracking |

```
Semantic Segmentation:        Instance Segmentation:
┌─────────────────────┐       ┌─────────────────────┐
│  ████  ████  ████   │       │  ████  ████  ████   │
│  car   car   car    │       │  car1  car2  car3   │
│  (all same color)   │       │  (different colors) │
└─────────────────────┘       └─────────────────────┘
```

## Mask R-CNN: The Standard Approach

Mask R-CNN extends Faster R-CNN by adding a mask prediction branch:

```
Input Image
     │
     ▼
┌────────────────────────┐
│  Backbone (ResNet/FPN) │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ Region Proposal Network│
│      (RPN)             │
└──────────┬─────────────┘
           │
           ▼ (Regions of Interest)
┌────────────────────────┐
│   RoI Align            │
└──────────┬─────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌─────────┐  ┌─────────┐
│Box Head │  │Mask Head│
│(class + │  │(binary  │
│ bbox)   │  │ mask)   │
└─────────┘  └─────────┘
```

## Using Pre-trained Mask R-CNN

```python
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

def load_maskrcnn():
    """Load pre-trained Mask R-CNN model."""
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def run_instance_segmentation(model, image: torch.Tensor, threshold: float = 0.5):
    """
    Run instance segmentation on an image.
    
    Args:
        model: Pre-trained Mask R-CNN
        image: Input tensor (3, H, W), values in [0, 1]
        threshold: Confidence threshold for detections
    
    Returns:
        Dictionary with boxes, labels, scores, and masks
    """
    with torch.no_grad():
        predictions = model([image])[0]
    
    # Filter by confidence
    keep = predictions['scores'] > threshold
    
    return {
        'boxes': predictions['boxes'][keep],       # (N, 4) in xyxy format
        'labels': predictions['labels'][keep],     # (N,) class indices
        'scores': predictions['scores'][keep],     # (N,) confidence scores
        'masks': predictions['masks'][keep] > 0.5  # (N, 1, H, W) binary masks
    }

def visualize_instance_segmentation(image, predictions, class_names):
    """Visualize instance segmentation results."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert image for display
    img = image.permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # Generate random colors for each instance
    colors = plt.cm.rainbow(np.linspace(0, 1, len(predictions['boxes'])))
    
    for box, label, score, mask, color in zip(
        predictions['boxes'],
        predictions['labels'],
        predictions['scores'],
        predictions['masks'],
        colors
    ):
        # Draw mask
        mask_np = mask.squeeze().numpy()
        colored_mask = np.zeros((*mask_np.shape, 4))
        colored_mask[mask_np] = [*color[:3], 0.5]
        ax.imshow(colored_mask)
        
        # Draw box
        x1, y1, x2, y2 = box.numpy()
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                            fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # Add label
        class_name = class_names[label]
        ax.text(x1, y1 - 5, f'{class_name}: {score:.2f}',
               color='white', fontsize=10,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    return fig
```

## One-Stage Instance Segmentation

Recent approaches achieve instance segmentation without region proposals:

### YOLACT (You Only Look At CoefficienTs)

- Predicts prototype masks and coefficients per instance
- Combines prototypes linearly to generate instance masks
- Much faster than Mask R-CNN

### SOLOv2

- Divides image into grid cells
- Each cell predicts mask for instance at that location
- No NMS required

## Comparison of Approaches

| Method | Type | Speed | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| Mask R-CNN | Two-stage | Slow | High | Accuracy-critical |
| YOLACT | One-stage | Fast | Medium | Real-time |
| SOLOv2 | One-stage | Medium | High | Balanced |

## COCO Instance Categories

Pre-trained models typically support 80 COCO categories:

```python
COCO_INSTANCE_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
```

## Summary

Instance segmentation extends semantic segmentation to separate individual objects:

1. **Mask R-CNN** remains the standard two-stage approach
2. **One-stage methods** (YOLACT, SOLOv2) offer speed advantages
3. **RoI Align** is key for accurate mask predictions
4. Pre-trained models work well for COCO categories

## References

1. He, K., et al. (2017). Mask R-CNN. ICCV.
2. Bolya, D., et al. (2019). YOLACT: Real-time Instance Segmentation. ICCV.
3. Wang, X., et al. (2020). SOLOv2: Dynamic and Fast Instance Segmentation. NeurIPS.
