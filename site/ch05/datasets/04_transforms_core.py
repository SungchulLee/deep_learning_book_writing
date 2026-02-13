#!/usr/bin/env python3
# ========================================================
# 12_dataloader_5_transforms_core.py
# ========================================================
"""
Core transforms (no torchvision): input, target, and compose.
Shows transform=..., target_transform=... on a map-style Dataset.
"""

from typing import Callable, Optional, Tuple
import torch
from torch.utils.data import Dataset

# ----- simple transform objects -----
class Compose:
    def __init__(self, fns): self.fns = list(fns)
    def __call__(self, x):
        for f in self.fns: x = f(x)
        return x

class ToTensor:
    def __call__(self, x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x)

class Standardize:
    def __init__(self, mean, std, eps=1e-8): self.mean, self.std, self.eps = float(mean), float(std), eps
    def __call__(self, t: torch.Tensor):
        return (t - self.mean) / (self.std + self.eps)

class MinMaxScale:
    def __init__(self, min_v, max_v, out_min=0.0, out_max=1.0, eps=1e-8):
        self.min_v, self.max_v, self.out_min, self.out_max, self.eps = min_v, max_v, out_min, out_max, eps
    def __call__(self, t: torch.Tensor):
        scaled = (t - self.min_v) / (self.max_v - self.min_v + self.eps)
        return scaled * (self.out_max - self.out_min) + self.out_min

class AddGaussianNoise:
    def __init__(self, std=0.05, seed=0):
        self.std = std
        self.g = torch.Generator().manual_seed(seed)
    def __call__(self, t: torch.Tensor):
        return t + torch.normal(0, self.std, size=t.shape, generator=self.g)

class OneHot:
    def __init__(self, num_classes:int):
        self.num_classes = num_classes
    def __call__(self, y: torch.Tensor):
        oh = torch.zeros(self.num_classes, dtype=torch.float32)
        oh[y.long().item()] = 1.0
        return oh

# ----- dataset using transforms -----
class ToyRegression(Dataset):
    """ y = 3x + 1 + ε; demonstrates transform and target_transform. """
    def __init__(self, n=12, noise=0.3, seed=0,
                 transform: Optional[Callable]=None,
                 target_transform: Optional[Callable]=None):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.empty(n, 1).uniform_(-2, 2, generator=g)
        self.y = 3*self.x + 1 + noise*torch.randn_like(self.x, generator=g)
        self.transform, self.target_transform = transform, target_transform
        self.xm, self.xs = self.x.mean(), self.x.std(unbiased=False)
        self.ym, self.ys = self.y.mean(), self.y.std(unbiased=False)

    def __len__(self): return len(self.x)
    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        xi, yi = self.x[i], self.y[i]
        if self.transform: xi = self.transform(xi)
        if self.target_transform: yi = self.target_transform(yi)
        return xi, yi

def main():
    base = ToyRegression(n=8, seed=1)
    x_tf = Compose([Standardize(base.xm, base.xs), AddGaussianNoise(0.02, seed=123)])
    y_tf = Standardize(base.ym, base.ys)
    ds = ToyRegression(n=8, seed=1, transform=x_tf, target_transform=y_tf)
    for i in range(len(ds)):
        x, y = ds[i]
        print(f"{i}: x={x.flatten().tolist()} y={y.flatten().tolist()}")

if __name__ == "__main__":
    main()

