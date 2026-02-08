#!/usr/bin/env python3
"""
Data Augmentation - Common vision augmentations (tensor-based)
Includes:
  - Random horizontal flip
  - Random crop (naive)
  - Color jitter (simple brightness/contrast)
  - Mixup (classification)

File: appendix/utils/augmentation.py
Note: Educational implementations (not as feature-complete as torchvision.transforms).
"""

import torch


def random_horizontal_flip(x, p=0.5):
    """
    Randomly flip images horizontally.

    x: (B, C, H, W)
    """
    if torch.rand(1).item() < p:
        return torch.flip(x, dims=[3])  # flip width dimension
    return x


def random_crop(x, crop_h, crop_w):
    """
    Naive random crop.

    x: (B, C, H, W)
    """
    B, C, H, W = x.shape
    if crop_h > H or crop_w > W:
        raise ValueError("Crop size must be <= image size")

    top = torch.randint(0, H - crop_h + 1, (1,)).item()
    left = torch.randint(0, W - crop_w + 1, (1,)).item()
    return x[:, :, top:top + crop_h, left:left + crop_w]


def mixup(x, y, alpha=0.2):
    """
    Mixup augmentation for classification.

    x: (B, C, H, W)
    y: (B,) class indices OR (B, num_classes) one-hot

    Returns:
      x_mix, y_a, y_b, lam
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B = x.size(0)
    perm = torch.randperm(B)

    x_mix = lam * x + (1 - lam) * x[perm]
    y_a = y
    y_b = y[perm]
    return x_mix, y_a, y_b, lam
