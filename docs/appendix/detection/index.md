# A4 Detection Models

## Overview

This appendix provides complete PyTorch implementations of object detection architectures. Detection models localize and classify objects within images by predicting bounding boxes and class labels. The evolution from two-stage (Faster R-CNN) to one-stage (YOLO, SSD) to transformer-based (DETR, DINO) detectors illustrates a progression toward simpler pipelines with stronger performance.

## Architectures

### Two-Stage Detectors

| Model | Year | Key Innovation |
|-------|------|----------------|
| [Faster R-CNN](faster_rcnn.py) | 2015 | Region proposal network (RPN) + Fast R-CNN in a unified architecture |

### One-Stage Detectors

| Model | Year | Key Innovation |
|-------|------|----------------|
| [YOLO v3](yolo_v3.py) | 2018 | Multi-scale detection with Darknet-53 backbone, real-time inference |
| [SSD](ssd.py) | 2016 | Multi-scale feature maps with default anchor boxes |
| [RetinaNet](retinanet.py) | 2017 | Focal loss to address class imbalance in dense detection |

### Transformer-Based Detectors

| Model | Year | Key Innovation |
|-------|------|----------------|
| [DETR](detr.py) | 2020 | End-to-end detection with transformers, bipartite matching loss |
| [DINO](dino.py) | 2022 | Denoising anchor boxes, contrastive denoising training |

## Key Concepts

### Detection Pipeline Components

1. **Backbone**: Feature extraction network (ResNet, FPN, Swin Transformer)
2. **Neck / FPN**: Multi-scale feature fusion for detecting objects at different sizes
3. **Head**: Predicts bounding box coordinates and class probabilities
4. **Post-processing**: Non-maximum suppression (NMS) or learned set prediction

### Anchor-Based vs. Anchor-Free

- **Anchor-based**: Pre-defined reference boxes at each spatial location (Faster R-CNN, SSD, RetinaNet)
- **Anchor-free**: Predict offsets from points or learn object queries directly (DETR, DINO)

### Loss Functions

- **Classification**: Cross-entropy, focal loss (addresses foreground–background imbalance)
- **Localization**: Smooth L1, GIoU, DIoU losses for bounding box regression
- **Set prediction**: Hungarian matching + combined classification and box loss (DETR)

### Evaluation Metrics

- **AP (Average Precision)**: Area under the precision–recall curve at various IoU thresholds
- **AP$_{50}$, AP$_{75}$**: AP at IoU = 0.50 and 0.75
- **mAP**: Mean AP across all object categories
- **FPS**: Inference speed, critical for real-time applications

## Quantitative Finance Applications

- **Document object detection**: Locate tables, charts, signatures, and stamps in financial filings
- **Satellite object counting**: Detect vehicles in parking lots, ships in ports, or containers at terminals for economic activity estimation
- **Trading floor monitoring**: Real-time activity detection from surveillance feeds
- **Chart element detection**: Identify data points, trend lines, and annotations in financial charts for automated digitization

## Prerequisites

- [A1: Classic CNNs](../cnn/index.md) — backbone architectures and feature pyramid networks
- [A2: Vision Transformers](../vit/index.md) — transformer encoders for DETR and DINO
- [A10: Utility Modules — Loss Functions](../utils/losses.py) — focal loss, IoU-based losses
- [Ch5: Convolutional Neural Networks](../../ch05/index.md) — anchor generation, RoI pooling
