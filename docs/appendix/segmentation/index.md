# A3 Segmentation Models

## Overview

This appendix provides complete PyTorch implementations of semantic and instance segmentation architectures. Segmentation models assign a class label to every pixel in an image, enabling fine-grained spatial understanding. In quantitative finance, these models are applied to satellite imagery analysis, document layout parsing, and extracting structured information from visual financial data.

## Architectures

| Model | Year | Key Innovation | Segmentation Type |
|-------|------|----------------|-------------------|
| [FCN](fcn.py) | 2015 | Fully convolutional networks, no fully-connected layers | Semantic |
| [U-Net](u_net.py) | 2015 | Symmetric encoder–decoder with skip connections | Semantic |
| [DeepLab v3+](deeplabv3.py) | 2018 | Atrous spatial pyramid pooling (ASPP), encoder–decoder | Semantic |
| [PSPNet](pspnet.py) | 2017 | Pyramid pooling module for multi-scale global context | Semantic |
| [Mask R-CNN](mask_rcnn.py) | 2017 | Instance-level masks via RoIAlign | Instance |
| [SegFormer](segformer.py) | 2021 | Hierarchical transformer encoder with lightweight MLP decoder | Semantic |

## Key Concepts

### Semantic vs. Instance Segmentation

- **Semantic segmentation**: Assigns a class label to each pixel; all instances of the same class share one label
- **Instance segmentation**: Distinguishes individual object instances within the same class
- **Panoptic segmentation**: Unifies both — every pixel gets a class label and an instance ID

### Core Design Patterns

1. **Encoder–decoder**: Downsample to capture context, upsample to recover spatial resolution (U-Net, DeepLab v3+)
2. **Skip connections**: Fuse high-resolution encoder features with decoded features to preserve fine details (U-Net, FCN)
3. **Multi-scale context**: Capture features at multiple receptive field sizes via dilated/atrous convolutions (DeepLab), pyramid pooling (PSPNet), or hierarchical transformers (SegFormer)
4. **Region-based**: Detect objects first, then segment within each region of interest (Mask R-CNN)

### Evaluation Metrics

- **Mean Intersection over Union (mIoU)**: Standard metric for semantic segmentation
- **Pixel accuracy**: Fraction of correctly classified pixels
- **AP$^{\text{mask}}$**: Average precision computed on instance masks (Mask R-CNN)

## Quantitative Finance Applications

- **Satellite imagery**: Segment agricultural land, urban development, or infrastructure for commodity and real estate analysis
- **Document layout parsing**: Identify tables, figures, headers, and text blocks in financial reports and SEC filings
- **Chart decomposition**: Segment chart regions (axes, legends, plot areas) for automated data extraction
- **Geospatial risk assessment**: Flood zone mapping, deforestation monitoring for ESG scoring

## Prerequisites

- [A1: Classic CNNs](../cnn/index.md) — backbone architectures (ResNet, EfficientNet)
- [A2: Vision Transformers](../vit/index.md) — transformer-based encoders (SegFormer)
- [A4: Detection Models](../detection/index.md) — region proposal networks (Mask R-CNN)
- [Ch5: Convolutional Neural Networks](../../ch05/index.md) — transposed convolutions, dilated convolutions
