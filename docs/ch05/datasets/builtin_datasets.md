# Built-in Datasets

## Overview

PyTorch provides a rich collection of ready-to-use datasets through `torchvision.datasets`, `torchaudio.datasets`, and `torchtext.datasets`. These built-in datasets handle downloading, caching, and basic preprocessing, making them invaluable for prototyping, benchmarking, and educational purposes.

## Torchvision Datasets

The most commonly used datasets reside in `torchvision.datasets`:

```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# MNIST: 60k train + 10k test, 28×28 grayscale digits (0–9)
mnist = datasets.MNIST(root='./data', train=True, download=True,
                       transform=transforms.ToTensor())

# FashionMNIST: same dimensions, 10 clothing categories
fashion = datasets.FashionMNIST(root='./data', train=True, download=True,
                                transform=transforms.ToTensor())

# CIFAR-10: 50k train + 10k test, 32×32 RGB, 10 classes
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True,
                           transform=transforms.ToTensor())

# CIFAR-100: same images, 100 fine-grained classes
cifar100 = datasets.CIFAR100(root='./data', train=True, download=True,
                             transform=transforms.ToTensor())

# ImageNet: 1.2M train + 50k val, 1000 classes (requires manual download)
imagenet = datasets.ImageNet(root='./data/imagenet', split='train',
                             transform=transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()
                             ]))
```

## Common Interface

All built-in datasets follow a consistent interface:

```python
dataset = datasets.MNIST(
    root='./data',       # Where to store/find the data
    train=True,          # Training set (True) or test set (False)
    download=True,       # Download if not present
    transform=None,      # Transform applied to samples
    target_transform=None  # Transform applied to targets
)

# Standard Dataset interface
print(len(dataset))          # Number of samples
img, label = dataset[0]      # Access by index
print(img.shape, label)      # torch.Size([1, 28, 28]), 5
```

## Dataset Properties

| Dataset | Train Size | Test Size | Image Size | Classes | Task |
|---------|-----------|----------|------------|---------|------|
| MNIST | 60,000 | 10,000 | 28×28×1 | 10 | Digit recognition |
| FashionMNIST | 60,000 | 10,000 | 28×28×1 | 10 | Clothing classification |
| CIFAR-10 | 50,000 | 10,000 | 32×32×3 | 10 | Object recognition |
| CIFAR-100 | 50,000 | 10,000 | 32×32×3 | 100 | Fine-grained recognition |
| ImageNet | 1,281,167 | 50,000 | Variable | 1,000 | Object recognition |

## Folder-Based Datasets

For custom image collections organized into class folders, `ImageFolder` provides automatic label assignment:

```python
# Directory structure:
# data/hymenoptera/
#   train/
#     ants/
#       img001.jpg
#     bees/
#       img001.jpg
#   val/
#     ants/
#     bees/

from torchvision.datasets import ImageFolder

dataset = ImageFolder(
    root='data/hymenoptera/train',
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)

print(dataset.classes)        # ['ants', 'bees']
print(dataset.class_to_idx)   # {'ants': 0, 'bees': 1}
```

## Using Built-in Datasets for Prototyping

Built-in datasets serve as controlled environments for testing training pipeline components before applying them to domain-specific data:

```python
# Quick model sanity check with MNIST
from torch.utils.data import DataLoader

train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data', train=False, download=True,
                              transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
```

## Key Takeaways

- Built-in datasets handle downloading, caching, and basic loading, enabling rapid prototyping.
- All datasets share a consistent interface: `root`, `train`, `download`, `transform`, and `target_transform`.
- `ImageFolder` maps directory structure to class labels automatically.
- These datasets are best used for validating pipeline components before transitioning to domain-specific data.
