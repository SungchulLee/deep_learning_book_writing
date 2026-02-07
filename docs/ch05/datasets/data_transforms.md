# Data Transforms

## Overview

Transforms are callable objects that preprocess data samples before they enter the model. PyTorch's `torchvision.transforms` module provides a rich library of image transformations, and the `Compose` class chains them into a pipeline.

## The Transform Pipeline

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

`Compose` applies transforms sequentially. The ordering matters: geometric transforms should precede `ToTensor()`, which should precede `Normalize`.

## Common Transforms

### Geometric Transforms

```python
transforms.Resize(256)              # Resize shortest edge to 256
transforms.Resize((224, 224))       # Resize to exact dimensions
transforms.CenterCrop(224)          # Crop center 224×224
transforms.RandomCrop(224)          # Crop random 224×224
transforms.RandomResizedCrop(224)   # Random crop + resize (scale 0.08–1.0)
transforms.RandomHorizontalFlip()   # Flip with p=0.5
transforms.RandomVerticalFlip()     # Flip with p=0.5
transforms.RandomRotation(30)       # Rotate ±30 degrees
```

### Color/Intensity Transforms

```python
transforms.ColorJitter(
    brightness=0.2, contrast=0.2,
    saturation=0.2, hue=0.1
)
transforms.Grayscale(num_output_channels=1)
transforms.RandomGrayscale(p=0.1)
```

### Conversion Transforms

```python
transforms.ToTensor()       # PIL Image/ndarray → FloatTensor, scales [0,255] → [0,1]
transforms.ToPILImage()     # Tensor/ndarray → PIL Image
transforms.ConvertImageDtype(torch.float32)
```

### Normalization

```python
# Per-channel: output = (input - mean) / std
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],   # ImageNet statistics
    std=[0.229, 0.224, 0.225]
)
```

Normalization is applied **after** `ToTensor()`. The ImageNet statistics above are standard for models pretrained on ImageNet. For custom datasets, compute statistics from the training set:

```python
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0
    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_size
    return mean / n_samples, std / n_samples
```

## Training vs. Evaluation Transforms

A critical practice is to use **different** transforms for training and evaluation:

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)
```

Training transforms include stochastic augmentation; validation transforms are deterministic to ensure reproducible evaluation.

## v2 Transforms API

PyTorch's newer `transforms.v2` API provides joint transforms that operate consistently on images, bounding boxes, segmentation masks, and other annotation types:

```python
from torchvision.transforms import v2

transform = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])
```

## Key Takeaways

- Transforms form a pipeline that preprocesses data before model ingestion.
- Ordering matters: geometric → `ToTensor()` → `Normalize`.
- Use stochastic transforms for training and deterministic transforms for evaluation.
- Compute normalization statistics from the training set, not the full dataset.
- The v2 API handles joint transforms for images and their annotations.
