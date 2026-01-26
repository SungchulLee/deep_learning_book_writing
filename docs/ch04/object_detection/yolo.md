# YOLO: You Only Look Once

## Learning Objectives

By the end of this section, you will be able to:

- Explain the YOLO philosophy and how it differs from two-stage detectors
- Understand grid-based detection and anchor box predictions
- Trace the evolution from YOLOv1 through YOLOv8
- Implement YOLO-style detection heads and loss functions
- Use pre-trained YOLO models for inference and fine-tuning
- Optimize YOLO models for real-time applications

## The YOLO Philosophy

YOLO (You Only Look Once) revolutionized object detection by framing it as a single regression problem, directly predicting bounding boxes and class probabilities from full images in one evaluation.

### Key Insight

Instead of proposing regions and classifying them separately, YOLO divides the image into a grid and predicts all boxes and classes simultaneously:

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOLO Detection Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Image (448×448)                                          │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────┐                    │
│  │        Single CNN Forward Pass           │                    │
│  │    (Feature extraction + prediction)     │                    │
│  └────────────────────┬────────────────────┘                    │
│                       │                                          │
│                       ▼                                          │
│  ┌─────────────────────────────────────────┐                    │
│  │           S × S Grid Output              │                    │
│  │    Each cell: B boxes + C classes       │                    │
│  │    Shape: (S, S, B×5 + C)               │                    │
│  └────────────────────┬────────────────────┘                    │
│                       │                                          │
│                       ▼                                          │
│              NMS + Final Detections                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Advantages Over Two-Stage Detectors

| Aspect | YOLO | Two-Stage (Faster R-CNN) |
|--------|------|--------------------------|
| **Speed** | 45-155+ FPS | 5-15 FPS |
| **Global Context** | Sees full image | Sees only proposals |
| **Architecture** | Simpler, unified | Complex, multi-component |
| **Background Errors** | Fewer false positives | More background confusion |
| **Small Objects** | More challenging | Better with FPN |

## Grid-Based Detection

YOLO divides the input image into an S×S grid. Each grid cell is responsible for detecting objects whose center falls within that cell.

### Cell Predictions

For each grid cell, YOLO predicts:
- **B bounding boxes**: Each with (x, y, w, h, confidence)
- **C class probabilities**: P(class_i | object)

```
Grid Cell Output:
┌─────────────────────────────────────────────────────────────┐
│  Box 1: [x₁, y₁, w₁, h₁, conf₁]                            │
│  Box 2: [x₂, y₂, w₂, h₂, conf₂]                            │
│  ...                                                        │
│  Box B: [xB, yB, wB, hB, confB]                             │
│  Classes: [P(c₁|obj), P(c₂|obj), ..., P(cC|obj)]          │
└─────────────────────────────────────────────────────────────┘

Total predictions per cell: B × 5 + C
Total output tensor: S × S × (B × 5 + C)
```

### Coordinate Encoding

YOLO uses normalized coordinates relative to the grid cell:

- **(x, y)**: Offset from grid cell corner, normalized to [0, 1]
- **(w, h)**: Relative to image size, normalized to [0, 1]
- **confidence**: P(object) × IoU(pred, truth)

```python
import torch


def decode_yolo_boxes(
    predictions: torch.Tensor,
    grid_size: int,
    num_boxes: int,
    image_size: int
) -> torch.Tensor:
    """
    Decode YOLO predictions to absolute box coordinates.
    
    Args:
        predictions: (batch, S, S, B*5+C) raw predictions
        grid_size: S (grid dimension)
        num_boxes: B (boxes per cell)
        image_size: Input image dimension
        
    Returns:
        boxes: (batch, S*S*B, 4) in xyxy format
    """
    batch_size = predictions.shape[0]
    cell_size = image_size / grid_size
    
    # Create grid offsets
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size),
        torch.arange(grid_size),
        indexing='ij'
    )
    grid_x = grid_x.to(predictions.device).float()
    grid_y = grid_y.to(predictions.device).float()
    
    boxes = []
    for b in range(num_boxes):
        start_idx = b * 5
        
        # Extract predictions
        x = predictions[..., start_idx + 0]      # Relative x in cell
        y = predictions[..., start_idx + 1]      # Relative y in cell
        w = predictions[..., start_idx + 2]      # Width relative to image
        h = predictions[..., start_idx + 3]      # Height relative to image
        
        # Convert to absolute coordinates
        x_abs = (grid_x + x) * cell_size
        y_abs = (grid_y + y) * cell_size
        w_abs = w * image_size
        h_abs = h * image_size
        
        # Convert to xyxy format
        x1 = x_abs - w_abs / 2
        y1 = y_abs - h_abs / 2
        x2 = x_abs + w_abs / 2
        y2 = y_abs + h_abs / 2
        
        box = torch.stack([x1, y1, x2, y2], dim=-1)
        boxes.append(box.reshape(batch_size, -1, 4))
    
    return torch.cat(boxes, dim=1)
```

## YOLO Evolution

### YOLOv1 (2015)

The original YOLO introduced the single-shot detection paradigm:

- 7×7 grid, 2 boxes per cell, 20 classes (PASCAL VOC)
- Output: 7×7×30 tensor
- 24 convolutional layers + 2 fully connected layers
- 45 FPS on GPU

**Limitations**:
- Struggles with small objects and objects in groups
- Limited to 2 boxes per cell
- Spatial constraints on predictions

### YOLOv2/YOLO9000 (2016)

Key improvements:
- **Batch normalization** on all conv layers
- **High-resolution classifier**: Fine-tune at 448×448
- **Anchor boxes**: Learn box priors from data using k-means
- **Passthrough layer**: Fine-grained features from earlier layers
- **Multi-scale training**: Train on multiple resolutions

### YOLOv3 (2018)

Major architectural changes:
- **Darknet-53 backbone**: 53-layer residual network
- **Multi-scale predictions**: Detect at 3 different scales
- **Independent logistic classifiers**: Better for multi-label

```python
import torch.nn as nn


class DarknetBlock(nn.Module):
    """
    Darknet residual block.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class YOLOv3Head(nn.Module):
    """
    YOLOv3 detection head for one scale.
    """
    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 3,
        num_classes: int = 80
    ):
        super().__init__()
        
        # Each anchor predicts: 4 coords + 1 objectness + num_classes
        out_channels = num_anchors * (5 + num_classes)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels * 2, out_channels, 1)
        )
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Returns:
            (batch, num_anchors, H, W, 5 + num_classes)
        """
        out = self.conv(x)
        batch, _, H, W = out.shape
        
        out = out.view(batch, self.num_anchors, 5 + self.num_classes, H, W)
        out = out.permute(0, 1, 3, 4, 2)
        
        return out
```

### YOLOv4 (2020)

Incorporated state-of-the-art techniques:
- **CSPDarknet53 backbone**: Cross Stage Partial connections
- **SPP (Spatial Pyramid Pooling)**: Multi-scale feature aggregation
- **PANet neck**: Path Aggregation Network
- **Advanced augmentation**: Mosaic, CutMix, Self-Adversarial Training

### YOLOv5 (2020)

PyTorch reimplementation with focus on usability:
- Clean PyTorch codebase
- Easy training and deployment
- Multiple model sizes (n, s, m, l, x)
- Built-in data augmentation

### YOLOv6 (2022)

Industrial-focused improvements:
- EfficientRep backbone
- Rep-PAN neck
- Optimized for deployment

### YOLOv7 (2022)

Training innovations:
- Extended efficient layer aggregation (E-ELAN)
- Planned re-parameterized convolution
- Compound scaling for different sizes

### YOLOv8 (2023)

Latest generation with anchor-free detection:

```python
# Using ultralytics YOLOv8
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # nano model

# Inference
results = model('image.jpg')

# Training
model.train(data='coco.yaml', epochs=100, imgsz=640)

# Export
model.export(format='onnx')
```

## YOLOv8 Architecture

### Backbone: CSPDarknet

The backbone uses Cross Stage Partial (CSP) connections:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CSPDarknet Backbone                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input (640×640)                                                 │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────┐                                            │
│  │   Stem (Focus)   │ → 80×80                                   │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  CSP Block 1    │ → 80×80                                    │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  CSP Block 2    │ → 40×40  (P3)                              │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  CSP Block 3    │ → 20×20  (P4)                              │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  CSP Block 4    │ → 10×10  (P5)                              │
│  │     + SPPF      │                                            │
│  └─────────────────┘                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Neck: FPN + PAN

Feature Pyramid Network with Path Aggregation:

```python
class PANNeck(nn.Module):
    """
    Path Aggregation Network neck for multi-scale feature fusion.
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        # Top-down path (FPN)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1)
            for ch in in_channels_list
        ])
        
        # Bottom-up path (PAN)
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            for _ in range(len(in_channels_list) - 1)
        ])
        
        # Fusion convs
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
            for _ in range(len(in_channels_list) - 1)
        ])
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps [P3, P4, P5]
            
        Returns:
            List of fused features at each scale
        """
        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = nn.functional.interpolate(
                laterals[i], scale_factor=2, mode='nearest'
            )
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Bottom-up pathway
        outputs = [laterals[0]]
        for i in range(len(laterals) - 1):
            downsampled = self.downsample_convs[i](outputs[-1])
            fused = torch.cat([downsampled, laterals[i+1]], dim=1)
            outputs.append(self.fusion_convs[i](fused))
        
        return outputs
```

### Anchor-Free Detection Head

YOLOv8 uses an anchor-free approach with decoupled heads:

```python
class YOLOv8Head(nn.Module):
    """
    Anchor-free detection head with decoupled classification and regression.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 80,
        reg_max: int = 16
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        
        # Regression branch
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        # Distribution Focal Loss: predict 4 × reg_max values
        self.reg_pred = nn.Conv2d(in_channels, 4 * reg_max, 1)
    
    def forward(self, x):
        """
        Returns:
            cls_out: (batch, num_classes, H, W)
            reg_out: (batch, 4*reg_max, H, W)
        """
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        
        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)
        
        return cls_out, reg_out
```

## YOLO Loss Function

YOLO uses a multi-part loss function:

### YOLOv1-v3 Loss

$$L = \lambda_{coord} L_{coord} + L_{conf} + L_{cls}$$

**Coordinate Loss** (only for cells with objects):
$$L_{coord} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right]$$
$$+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]$$

**Confidence Loss**:
$$L_{conf} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 + \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2$$

**Classification Loss** (only for cells with objects):
$$L_{cls} = \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2$$

```python
import torch
import torch.nn.functional as F


def yolo_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    num_boxes: int = 2,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5
) -> torch.Tensor:
    """
    YOLOv1-style loss function.
    
    Args:
        predictions: (batch, S, S, B*5 + C)
        targets: (batch, S, S, 5 + C) with [x, y, w, h, obj, classes...]
        num_classes: Number of classes C
        num_boxes: Number of boxes per cell B
        
    Returns:
        Total loss
    """
    batch_size, S, _, _ = predictions.shape
    
    # Parse predictions
    pred_boxes = []
    pred_confs = []
    for b in range(num_boxes):
        start = b * 5
        pred_boxes.append(predictions[..., start:start+4])
        pred_confs.append(predictions[..., start+4:start+5])
    
    pred_classes = predictions[..., num_boxes*5:]
    
    # Parse targets
    target_box = targets[..., :4]
    target_obj = targets[..., 4:5]
    target_classes = targets[..., 5:]
    
    # Object mask
    obj_mask = target_obj.squeeze(-1) == 1  # (batch, S, S)
    noobj_mask = target_obj.squeeze(-1) == 0
    
    # Find responsible predictor (highest IoU with target)
    ious = []
    for pred_box in pred_boxes:
        iou = compute_iou(pred_box, target_box)  # (batch, S, S)
        ious.append(iou)
    
    ious = torch.stack(ious, dim=-1)  # (batch, S, S, B)
    best_box = ious.argmax(dim=-1)  # (batch, S, S)
    
    # Coordinate loss (responsible predictor only)
    coord_loss = 0
    for b in range(num_boxes):
        responsible = (best_box == b) & obj_mask
        if responsible.sum() > 0:
            pred = pred_boxes[b][responsible]
            target = target_box[responsible]
            
            # xy loss
            coord_loss += F.mse_loss(pred[:, :2], target[:, :2], reduction='sum')
            
            # wh loss (sqrt for scale invariance)
            coord_loss += F.mse_loss(
                torch.sqrt(pred[:, 2:4].abs() + 1e-6),
                torch.sqrt(target[:, 2:4].abs() + 1e-6),
                reduction='sum'
            )
    
    coord_loss *= lambda_coord
    
    # Confidence loss
    conf_loss = 0
    for b in range(num_boxes):
        responsible = (best_box == b) & obj_mask
        
        # Object confidence
        if responsible.sum() > 0:
            pred_conf = pred_confs[b][responsible]
            target_iou = ious[..., b][responsible]
            conf_loss += F.mse_loss(pred_conf.squeeze(-1), target_iou, reduction='sum')
        
        # No-object confidence
        not_responsible = ~responsible & noobj_mask
        if not_responsible.sum() > 0:
            pred_conf = pred_confs[b][not_responsible]
            conf_loss += lambda_noobj * F.mse_loss(
                pred_conf.squeeze(-1),
                torch.zeros_like(pred_conf.squeeze(-1)),
                reduction='sum'
            )
    
    # Classification loss
    if obj_mask.sum() > 0:
        cls_loss = F.mse_loss(
            pred_classes[obj_mask],
            target_classes[obj_mask],
            reduction='sum'
        )
    else:
        cls_loss = 0
    
    # Total loss normalized by batch size
    total_loss = (coord_loss + conf_loss + cls_loss) / batch_size
    
    return total_loss
```

### Modern YOLO Losses

YOLOv5+ use more sophisticated losses:

- **CIoU Loss** for bounding box regression
- **Binary Cross-Entropy** for objectness and classification
- **Focal Loss** to handle class imbalance

```python
def modern_yolo_loss(
    pred_boxes: torch.Tensor,
    pred_obj: torch.Tensor,
    pred_cls: torch.Tensor,
    target_boxes: torch.Tensor,
    target_obj: torch.Tensor,
    target_cls: torch.Tensor,
    box_weight: float = 7.5,
    obj_weight: float = 1.0,
    cls_weight: float = 0.5
) -> dict:
    """
    Modern YOLO loss with CIoU and BCE.
    """
    # Box loss (CIoU)
    ciou = compute_ciou(pred_boxes, target_boxes)
    box_loss = (1 - ciou).mean()
    
    # Objectness loss (BCE with logits)
    obj_loss = F.binary_cross_entropy_with_logits(
        pred_obj, target_obj, reduction='mean'
    )
    
    # Classification loss (BCE with logits)
    cls_loss = F.binary_cross_entropy_with_logits(
        pred_cls, target_cls, reduction='mean'
    )
    
    total_loss = (
        box_weight * box_loss +
        obj_weight * obj_loss +
        cls_weight * cls_loss
    )
    
    return {
        'loss': total_loss,
        'box_loss': box_loss,
        'obj_loss': obj_loss,
        'cls_loss': cls_loss
    }
```

## Using YOLO in Practice

### Ultralytics YOLOv8

```python
from ultralytics import YOLO
import torch

# Load models (downloads automatically)
model_nano = YOLO('yolov8n.pt')    # Fastest
model_small = YOLO('yolov8s.pt')
model_medium = YOLO('yolov8m.pt')
model_large = YOLO('yolov8l.pt')
model_xlarge = YOLO('yolov8x.pt')  # Most accurate

# Inference
results = model_nano('image.jpg')

# Process results
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    
    for box, score, cls in zip(boxes, scores, classes):
        print(f"Class {int(cls)}: {score:.2f} at {box}")

# Batch inference
results = model_nano(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# Inference with options
results = model_nano(
    'image.jpg',
    conf=0.25,        # Confidence threshold
    iou=0.45,         # NMS IoU threshold
    max_det=300,      # Max detections
    classes=[0, 2],   # Filter classes (person, car)
    device='cuda:0'
)
```

### Training Custom Model

```python
from ultralytics import YOLO

# Start from pre-trained model
model = YOLO('yolov8n.pt')

# Train on custom dataset
results = model.train(
    data='custom_data.yaml',  # Dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    workers=8,
    device='cuda',
    patience=50,         # Early stopping
    save=True,
    project='runs/detect',
    name='custom_yolo'
)

# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# Export
model.export(format='onnx', dynamic=True)
model.export(format='torchscript')
model.export(format='tensorrt', half=True)
```

### Dataset Configuration (custom_data.yaml)

```yaml
# custom_data.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # Optional

nc: 3  # Number of classes
names: ['class1', 'class2', 'class3']
```

## Model Comparison

### YOLOv8 Variants

| Model | Params | FLOPs | mAP@50 | mAP@50:95 | Speed (T4) |
|-------|--------|-------|--------|-----------|------------|
| YOLOv8n | 3.2M | 8.7G | 52.6% | 37.3% | 0.99ms |
| YOLOv8s | 11.2M | 28.6G | 61.8% | 44.9% | 1.20ms |
| YOLOv8m | 25.9M | 78.9G | 67.2% | 50.2% | 1.83ms |
| YOLOv8l | 43.7M | 165.2G | 69.8% | 52.9% | 2.39ms |
| YOLOv8x | 68.2M | 257.8G | 71.0% | 53.9% | 3.53ms |

### Choosing Model Size

```
Use Case                    Recommended Model
─────────────────────────────────────────────
Real-time video (>30 FPS)   YOLOv8n, YOLOv8s
Mobile/Edge deployment      YOLOv8n
General applications        YOLOv8m
High accuracy required      YOLOv8l, YOLOv8x
Research/Benchmarking       YOLOv8x
```

## Summary

YOLO revolutionized object detection with its single-shot approach:

1. **Single Network**: One forward pass for all detections
2. **Grid-Based**: Image divided into cells, each predicting boxes
3. **End-to-End**: Direct regression from pixels to boxes
4. **Real-Time**: 30-155+ FPS depending on model size
5. **Evolving**: YOLOv8 uses anchor-free detection with modern training

Key implementation details:
- Multi-scale prediction at different feature levels
- Anchor boxes (v2-v7) or anchor-free (v8) predictions
- CIoU loss for accurate box regression
- Strong data augmentation (Mosaic, MixUp)

YOLO models provide the best speed-accuracy tradeoff for real-time applications.

## References

1. Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR*.
2. Redmon, J., & Farhadi, A. (2017). YOLO9000: Better, Faster, Stronger. *CVPR*.
3. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. *arXiv*.
4. Bochkovskiy, A., et al. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. *arXiv*.
5. Jocher, G. (2020-2023). Ultralytics YOLOv5/YOLOv8. *GitHub*.
