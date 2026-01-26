# Detection Fundamentals

## Learning Objectives

By the end of this section, you will be able to:

- Define object detection and distinguish it from related computer vision tasks
- Understand the core components of a detection system: localization, classification, and multi-object handling
- Describe the general pipeline from input image to final predictions
- Explain the role of backbone networks, detection heads, and post-processing
- Identify key challenges in object detection and their practical implications

## What is Object Detection?

Object detection is a fundamental computer vision task that combines two related problems:

1. **Classification**: Determining *what* objects are present in an image
2. **Localization**: Determining *where* those objects are located

Unlike image classification, which assigns a single label to an entire image, object detection identifies multiple objects and provides precise spatial information about each one through **bounding boxes**—rectangular regions that enclose detected objects.

### Task Comparison

Understanding how object detection relates to other vision tasks clarifies its unique requirements:

| Task | Output | Example |
|------|--------|---------|
| **Image Classification** | Single class label | "This image contains a dog" |
| **Object Detection** | Class labels + bounding boxes | "Dog at (100, 150, 300, 400)" |
| **Semantic Segmentation** | Pixel-wise class labels | Every pixel labeled as dog, background, etc. |
| **Instance Segmentation** | Per-object pixel masks | Separate masks for each dog instance |

Object detection occupies a middle ground: it provides more spatial information than classification but is computationally lighter than pixel-level segmentation.

### Formal Problem Definition

Given an input image $I \in \mathbb{R}^{H \times W \times C}$, object detection produces a set of predictions:

$$\{(b_i, c_i, s_i)\}_{i=1}^{N}$$

where:
- $b_i = (x, y, w, h)$ or $(x_{min}, y_{min}, x_{max}, y_{max})$ defines the bounding box
- $c_i \in \{1, 2, \ldots, K\}$ is the class label from $K$ possible classes
- $s_i \in [0, 1]$ is the confidence score
- $N$ is the number of detected objects (variable per image)

## Bounding Box Representations

Bounding boxes can be represented in several formats, each with advantages for different purposes:

### Format 1: Corner Coordinates (xyxy)

$$\text{box} = (x_{min}, y_{min}, x_{max}, y_{max})$$

- $(x_{min}, y_{min})$: Top-left corner
- $(x_{max}, y_{max})$: Bottom-right corner
- **Advantage**: Direct representation, easy IoU computation
- **Used by**: COCO dataset, many evaluation scripts

### Format 2: Position and Size (xywh)

$$\text{box} = (x, y, w, h)$$

- $(x, y)$: Top-left corner position
- $(w, h)$: Width and height
- **Advantage**: Intuitive for humans
- **Used by**: PASCAL VOC dataset

### Format 3: Center and Size (cxcywh)

$$\text{box} = (c_x, c_y, w, h)$$

- $(c_x, c_y)$: Center coordinates
- $(w, h)$: Width and height
- **Advantage**: Natural for neural network predictions, scale-invariant representations
- **Used by**: YOLO family, most modern detectors

### Normalized Coordinates

Many detectors use normalized coordinates where all values are in $[0, 1]$:

$$c_x^{norm} = \frac{c_x}{W_{img}}, \quad c_y^{norm} = \frac{c_y}{H_{img}}$$
$$w^{norm} = \frac{w}{W_{img}}, \quad h^{norm} = \frac{h}{H_{img}}$$

This makes the representation **resolution-independent** and simplifies training across different image sizes.

### PyTorch Implementation

```python
import torch

def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x_min, y_min, x_max, y_max) to (cx, cy, w, h).
    
    Args:
        boxes: Tensor of shape (N, 4) in xyxy format
        
    Returns:
        Tensor of shape (N, 4) in cxcywh format
    """
    x_min, y_min, x_max, y_max = boxes.unbind(-1)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (cx, cy, w, h) to (x_min, y_min, x_max, y_max).
    
    Args:
        boxes: Tensor of shape (N, 4) in cxcywh format
        
    Returns:
        Tensor of shape (N, 4) in xyxy format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def normalize_boxes(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """
    Normalize box coordinates to [0, 1] range.
    
    Args:
        boxes: Tensor of shape (N, 4) in cxcywh format (absolute coordinates)
        img_w: Image width
        img_h: Image height
        
    Returns:
        Tensor of shape (N, 4) with normalized coordinates
    """
    scale = torch.tensor([img_w, img_h, img_w, img_h], 
                         dtype=boxes.dtype, device=boxes.device)
    return boxes / scale
```

## The Detection Pipeline

A modern object detection system follows a general pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                               │
│                      (H × W × 3)                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                                 │
│  • Resize to fixed dimensions (e.g., 640×640)                   │
│  • Normalize pixel values                                        │
│  • Convert to tensor format                                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BACKBONE NETWORK                               │
│  • Feature extraction (ResNet, CSPDarknet, EfficientNet)        │
│  • Produces multi-scale feature maps                            │
│  • Output: Feature pyramid F = {F₁, F₂, F₃, ...}               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       NECK (Optional)                            │
│  • Feature Pyramid Network (FPN)                                 │
│  • Path Aggregation Network (PAN)                                │
│  • Combines multi-scale features                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION HEAD                                │
│  • Predicts bounding boxes, class scores, objectness            │
│  • Per location: (Δx, Δy, Δw, Δh, obj, c₁, c₂, ..., cₖ)        │
│  • Thousands of raw predictions                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   POST-PROCESSING                                │
│  • Decode box coordinates                                        │
│  • Confidence thresholding (remove low-confidence detections)   │
│  • Non-Maximum Suppression (remove duplicate detections)        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FINAL OUTPUT                                 │
│  • List of (box, class, confidence) tuples                      │
│  • Typically 0-100 detections per image                         │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

**Backbone Network**: Extracts hierarchical features from the input image. Early layers capture low-level features (edges, textures), while deeper layers capture high-level semantics (object parts, whole objects). Common backbones include ResNet, VGG, CSPDarknet, and EfficientNet.

**Neck**: Combines features from different backbone levels to handle objects at various scales. The Feature Pyramid Network (FPN) creates a top-down pathway with lateral connections, while PANet adds bottom-up paths for better localization.

**Detection Head**: Makes predictions at each spatial location. For anchor-based detectors, it predicts offsets from predefined anchor boxes. For anchor-free detectors, it directly predicts box coordinates.

**Post-Processing**: Filters and refines raw predictions. Confidence thresholding removes unlikely detections. Non-Maximum Suppression (NMS) eliminates redundant overlapping boxes for the same object.

## One-Stage vs Two-Stage Detectors

Object detection architectures fall into two main categories:

### Two-Stage Detectors

Two-stage detectors separate detection into region proposal and classification:

**Stage 1 - Region Proposal Network (RPN)**:
- Generates class-agnostic region proposals
- High recall, many proposals (~2000)
- Proposals are likely to contain objects

**Stage 2 - Classification and Refinement**:
- Classifies each proposal
- Refines bounding box coordinates
- Produces final detections

**Examples**: R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN

**Characteristics**:
- Higher accuracy, especially for small objects
- Slower inference due to two-stage processing
- Better localization precision

### One-Stage Detectors

One-stage detectors predict boxes and classes directly in a single pass:

- Dense predictions over feature maps
- Predefined anchor boxes or anchor-free predictions
- Direct box regression and classification

**Examples**: YOLO family, SSD, RetinaNet, FCOS

**Characteristics**:
- Faster inference, suitable for real-time applications
- Simpler architecture
- May have lower accuracy on challenging cases (improved in modern versions)

### Comparison

| Aspect | Two-Stage | One-Stage |
|--------|-----------|-----------|
| **Speed** | Slower (~5-15 FPS) | Faster (~30-100+ FPS) |
| **Accuracy** | Generally higher | Competitive with modern methods |
| **Small Objects** | Better detection | Improved with FPN |
| **Architecture** | More complex | Simpler |
| **Use Cases** | Medical imaging, autonomous driving | Real-time video, mobile |

## Key Challenges in Object Detection

### 1. Scale Variation

Objects in images vary dramatically in size:
- A person close to the camera vs. far away
- A car vs. a traffic sign in the same scene

**Solutions**:
- Multi-scale feature pyramids (FPN)
- Image pyramid inference (test-time augmentation)
- Anchor boxes at multiple scales

### 2. Aspect Ratio Variation

Objects have different shapes:
- Tall, narrow objects (people standing)
- Wide, short objects (buses, benches)
- Square objects (faces, balls)

**Solutions**:
- Multiple anchor aspect ratios
- Anchor-free methods that don't constrain shape
- Deformable convolutions

### 3. Occlusion

Objects may be partially hidden:
- One object behind another
- Objects cut off by image boundaries

**Solutions**:
- Part-based models
- Context information
- Robust feature representations

### 4. Class Imbalance

Training data is heavily imbalanced:
- Vast majority of locations contain background
- Foreground objects are sparse
- Some classes are rare

**Solutions**:
- Focal Loss (RetinaNet)
- Hard negative mining
- Class-balanced sampling

### 5. Dense Scenes

Many objects in close proximity:
- Crowds of people
- Parking lots full of cars

**Solutions**:
- Soft-NMS instead of hard NMS
- Set-based predictions (DETR)
- Better anchor assignment strategies

### 6. Real-Time Requirements

Many applications need fast inference:
- Autonomous vehicles
- Video surveillance
- Mobile applications

**Solutions**:
- Efficient architectures (MobileNet backbones)
- Model pruning and quantization
- Hardware acceleration (TensorRT, ONNX)

## Confidence Scores

Detection confidence combines two factors:

$$\text{Confidence} = P(\text{object}) \times \text{IoU}(\text{pred}, \text{truth})$$

Or in practice:

$$\text{Confidence} = \text{Objectness} \times P(\text{class} | \text{object})$$

- **Objectness**: Probability that the box contains any object
- **Class Probability**: Probability of specific class given an object exists
- **Final Score**: Product indicates overall detection reliability

### Confidence Thresholding

```python
def filter_by_confidence(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter detections by confidence threshold.
    
    Args:
        boxes: (N, 4) bounding boxes
        scores: (N,) confidence scores
        labels: (N,) class labels
        threshold: Minimum confidence to keep
        
    Returns:
        Filtered boxes, scores, and labels
    """
    mask = scores >= threshold
    return boxes[mask], scores[mask], labels[mask]
```

**Threshold Trade-offs**:
- **High threshold**: Fewer false positives, may miss objects (low recall)
- **Low threshold**: Catches more objects, more false positives (low precision)

## Applications of Object Detection

Object detection powers numerous real-world applications:

### Autonomous Driving
- Detect vehicles, pedestrians, cyclists, traffic signs
- Critical for navigation and safety decisions
- Requires real-time performance and high reliability

### Retail and Inventory
- Product detection on shelves
- Automated checkout systems
- Stock level monitoring

### Medical Imaging
- Tumor detection in CT/MRI scans
- Cell counting in microscopy
- Organ localization

### Security and Surveillance
- Person and vehicle tracking
- Intrusion detection
- Anomaly identification

### Agriculture
- Crop health monitoring
- Pest detection
- Yield estimation

### Manufacturing
- Defect detection in quality control
- Assembly verification
- Safety monitoring

## Summary

Object detection is a cornerstone computer vision task that identifies and localizes multiple objects within images. Key concepts include:

1. **Bounding boxes** encode object locations in various formats (xyxy, xywh, cxcywh)
2. **Detection pipelines** consist of backbone, neck, head, and post-processing stages
3. **Two-stage detectors** prioritize accuracy through region proposals
4. **One-stage detectors** prioritize speed through direct predictions
5. **Major challenges** include scale variation, occlusion, class imbalance, and real-time requirements
6. **Confidence scores** combine objectness and class probabilities

Understanding these fundamentals provides the foundation for studying specific architectures like Faster R-CNN, YOLO, and SSD in subsequent sections.

## References

1. Girshick, R. (2015). Fast R-CNN. *ICCV*.
2. Ren, S., et al. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS*.
3. Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR*.
4. Liu, W., et al. (2016). SSD: Single Shot MultiBox Detector. *ECCV*.
5. Lin, T.Y., et al. (2017). Feature Pyramid Networks for Object Detection. *CVPR*.
