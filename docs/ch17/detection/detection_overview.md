# Detection Overview

## Learning Objectives

By the end of this section, you will be able to:

- Understand the object detection task and its formal problem definition
- Distinguish one-stage from two-stage detection paradigms
- Identify the key challenges in object detection (scale variation, class imbalance, occlusion)
- Recognize the major application domains for detection systems
- Understand confidence scoring and thresholding

## What is Object Detection?

Object detection combines localization (where) with classification (what), producing a set of bounding boxes, class labels, and confidence scores for each object in an image.

### Task Comparison

| Task | Output | Granularity |
|------|--------|-------------|
| Classification | Single label per image | Image-level |
| Object Detection | Bounding boxes + labels | Object-level |
| Semantic Segmentation | Class per pixel | Pixel-level |
| Instance Segmentation | Masks + labels per object | Pixel + instance |

### Formal Problem Definition

Given an input image $\mathbf{X} \in \mathbb{R}^{H \times W \times 3}$, object detection produces a variable-length set of detections:

$$\mathcal{D} = \{(b_i, c_i, s_i)\}_{i=1}^{N}$$

where each detection consists of:

- $b_i = (x_1, y_1, x_2, y_2)$: bounding box coordinates
- $c_i \in \{1, \ldots, K\}$: class label from $K$ categories
- $s_i \in [0, 1]$: confidence score

## The Detection Pipeline

All modern detectors share a common pipeline structure:

```
Input Image
     │
     ▼
┌───────────────┐
│   Backbone    │  Feature extraction (ResNet, VGG, CSPDarknet)
└──────┬────────┘
       │
       ▼
┌───────────────┐
│     Neck      │  Multi-scale feature fusion (FPN, PAN)
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  Detection    │  Classification + box regression
│    Head       │  (per anchor/point/query)
└──────┬────────┘
       │
       ▼
┌───────────────┐
│    Post-      │  NMS, score thresholding
│  Processing   │
└──────┬────────┘
       │
       ▼
  Final Detections
```

## One-Stage vs Two-Stage Detectors

### Two-Stage Detectors

1. **Stage 1**: Generate region proposals (potential object locations)
2. **Stage 2**: Classify and refine each proposal

Examples: R-CNN, Fast R-CNN, Faster R-CNN

**Advantages**: Higher accuracy, better for small objects
**Disadvantages**: Slower, more complex architecture

### One-Stage Detectors

Predict bounding boxes and classes directly from feature maps in a single pass.

Examples: YOLO, SSD, RetinaNet, FCOS

**Advantages**: Faster, simpler architecture, real-time capable
**Disadvantages**: Historically lower accuracy (gap now largely closed)

### Comparison

| Aspect | Two-Stage | One-Stage |
|--------|-----------|-----------|
| Speed | 5–15 FPS | 30–150+ FPS |
| Accuracy | Higher (historically) | Competitive (modern) |
| Small objects | Better | Improving |
| Architecture | Complex | Simpler |
| Use case | Accuracy-critical | Real-time |

## Key Challenges in Object Detection

### Scale Variation

Objects of the same class span orders of magnitude in pixel size (a nearby car vs. a distant car). Solutions include multi-scale feature maps (FPN), image pyramids, and anchor boxes at multiple scales.

### Aspect Ratio Variation

Objects vary dramatically in shape (a standing person vs. a bus). Anchor-based methods address this with multiple aspect ratio priors; anchor-free methods predict arbitrary shapes directly.

### Occlusion

Partially hidden objects must still be detected. Two-stage detectors with larger receptive fields tend to handle occlusion better than methods relying on local features.

### Class Imbalance

The vast majority of candidate locations in an image are background. This foreground/background imbalance (often 1:1000 or worse) makes naive training collapse to predicting "background" everywhere. Solutions include hard negative mining (SSD), focal loss (RetinaNet), and sampling strategies (Faster R-CNN).

### Dense Scenes

When objects overlap heavily, non-maximum suppression (NMS) may incorrectly remove valid detections. Soft-NMS and end-to-end approaches (DETR) address this.

## Confidence Scores and Thresholding

Detection confidence represents the model's estimate of $P(\text{object}) \times P(\text{correct class})$:

```python
import torch

def filter_detections(boxes, scores, labels, score_threshold=0.5):
    """
    Filter detections by confidence threshold.
    
    Args:
        boxes: Bounding boxes (N, 4)
        scores: Confidence scores (N,)
        labels: Class labels (N,)
        score_threshold: Minimum confidence to keep
    
    Returns:
        Filtered boxes, scores, and labels
    """
    keep = scores > score_threshold
    return boxes[keep], scores[keep], labels[keep]
```

The threshold trades precision against recall: lower thresholds catch more objects but introduce more false positives.

## Applications

| Domain | Key Requirements | Typical Models |
|--------|-----------------|----------------|
| Autonomous Driving | Real-time, high recall | YOLOv8, CenterNet |
| Medical Imaging | High sensitivity | Faster R-CNN, RetinaNet |
| Retail/Inventory | Product variety | YOLO, SSD |
| Security/Surveillance | Real-time, multi-class | YOLO, SSD |
| Agriculture | Aerial imagery | Faster R-CNN |
| Document Analysis | Small, dense objects | FCOS, DETR |

## Summary

Object detection combines localization and classification into a unified task. The field has evolved from computationally expensive two-stage approaches (R-CNN family) to efficient one-stage detectors (YOLO, SSD) and modern anchor-free and transformer-based designs (FCOS, DETR). Understanding the fundamental pipeline—backbone, neck, head, post-processing—provides the framework for comprehending all specific architectures covered in subsequent sections.

## References

1. Zou, Z., et al. (2023). Object Detection in 20 Years: A Survey. Proceedings of the IEEE.
2. Liu, L., et al. (2020). Deep Learning for Generic Object Detection: A Survey. IJCV.
