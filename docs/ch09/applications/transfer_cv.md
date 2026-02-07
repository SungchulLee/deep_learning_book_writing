# Transfer Learning for Computer Vision

Transfer learning is the default approach in computer vision. Pretrained models on ImageNet (or larger datasets) provide powerful feature extractors that adapt to diverse visual tasks with minimal labeled data.

## Standard Transfer Pipeline

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights


def create_transfer_model(num_classes, backbone='resnet50', strategy='feature_extraction'):
    """Create a transfer learning model for image classification."""
    
    if backbone == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        feature_dim = model.fc.in_features
        classifier_attr = 'fc'
    elif backbone == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        feature_dim = model.classifier[1].in_features
        classifier_attr = 'classifier'
    
    if strategy == 'feature_extraction':
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classifier
    if classifier_attr == 'fc':
        model.fc = nn.Linear(feature_dim, num_classes)
    else:
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )
    
    return model
```

## Architecture Selection Guide

| Model | Params | ImageNet Top-1 | Inference | Best for |
|-------|--------|---------------|-----------|----------|
| ResNet-18 | 11M | 69.8% | Fast | Prototyping |
| ResNet-50 | 25M | 80.9% | Medium | Balanced |
| EfficientNet-B0 | 5M | 77.1% | Fast | Mobile/edge |
| EfficientNet-B4 | 19M | 82.9% | Medium | High accuracy |
| ViT-B/16 | 86M | 81.1% | Slow | Large datasets |
| ConvNeXt-T | 29M | 82.1% | Medium | Modern CNNs |

## Task-Specific Adaptations

### Object Detection

Replace the classification head with detection-specific components:

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


def create_detection_model(num_classes):
    """Transfer pretrained Faster R-CNN to custom detection task."""
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    
    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
    
    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas
```

### Semantic Segmentation

```python
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


def create_segmentation_model(num_classes):
    """Transfer pretrained DeepLabV3 to custom segmentation task."""
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model
```

### Medical Imaging

Medical images often differ substantially from ImageNet. Key adaptations:

```python
def create_medical_transfer_model(num_classes, in_channels=1):
    """Transfer model adapted for medical imaging (e.g., X-ray, CT)."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Adapt first conv for single-channel input
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    
    # Initialise from pretrained weights (average across RGB channels)
    with torch.no_grad():
        model.conv1.weight[:, 0] = old_conv.weight.mean(dim=1)
    
    # Use discriminative LR (medical domain differs from ImageNet)
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Heavy dropout for small medical datasets
        nn.Linear(model.fc.in_features, num_classes)
    )
    
    return model
```

## Data Augmentation for Transfer

Augmentation strategy depends on domain similarity:

```python
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Similar domain (natural images)
similar_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Different domain (satellite, medical) - heavier augmentation
different_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

## Comparing Pretrained Sources

Beyond ImageNet, several pretraining sources are available:

| Pretrained source | Data size | Best for |
|-------------------|-----------|----------|
| ImageNet-1K | 1.3M | General vision |
| ImageNet-21K | 14M | Better features, more classes |
| CLIP (image-text) | 400M | Zero-shot, open-vocabulary |
| DINO/DINOv2 (self-supervised) | Variable | Dense prediction, segmentation |
| Domain-specific (CheXpert, etc.) | Variable | Medical, satellite, etc. |

## Summary

| Task | Recommended backbone | Strategy |
|------|---------------------|----------|
| Classification (small data) | EfficientNet-B0 | Feature extraction |
| Classification (large data) | ResNet-50 or ConvNeXt | Full fine-tuning |
| Detection | ResNet-50 FPN | Fine-tune with lower backbone LR |
| Segmentation | ResNet-50 + DeepLabV3 | Fine-tune decoder, freeze encoder |
| Medical imaging | ResNet-50 adapted | Discriminative LR, heavy dropout |

## References

1. Kornblith, S., et al. (2019). "Do Better ImageNet Models Transfer Better?" *CVPR*.
2. He, K., et al. (2019). "Rethinking ImageNet Pre-training." *ICCV*.
3. Raghu, M., et al. (2019). "Transfusion: Understanding Transfer Learning for Medical Imaging." *NeurIPS*.
