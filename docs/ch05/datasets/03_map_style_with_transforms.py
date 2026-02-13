#!/usr/bin/env python3
# ========================================================
# 12_dataset_03_map_style_with_transforms.py
# ========================================================
"""
Map-style Dataset with per-sample transforms.

Idea:
- Dataset = mapping {0, ..., N-1} → sample (or (input, target))
- Transforms are callables applied inside __getitem__.
- Plot: left = raw (no transforms), right = transformed.
"""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple


# --------------------------------------------------------
# Compose: run multiple transforms in sequence (x → f2(f1(x)))
# --------------------------------------------------------
class Compose:
    def __init__(self, fns):
        self.fns = list(fns)
    def __call__(self, x):
        for f in self.fns:
            x = f(x)
        return x


# --------------------------------------------------------
# AddNoise: x → x + ε,  ε ~ N(0, std^2)
# (uses its own torch.Generator for reproducibility)
# --------------------------------------------------------
class AddNoise:
    def __init__(self, std=0.05, seed=0):
        self.std = std
        self.g = torch.Generator().manual_seed(seed)
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t + torch.normal(0, self.std, t.shape, generator=self.g)


# --------------------------------------------------------
# Standardize: x → (x - mean) / (std + eps)
# --------------------------------------------------------
class Standardize:
    def __init__(self, mean: float, std: float, eps: float = 1e-8):
        self.mean, self.std, self.eps = mean, std, eps
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.mean) / (self.std + self.eps)


# --------------------------------------------------------
# ToyRegression: y = 3x + 1 + ε, ε ~ N(0, noise_std^2)
# - x sampled once at init; random-access via index
# - Note: .uniform_ is an in-place Tensor method (allocate → fill).
#   Alt: torch.rand(n,1)*4 - 2  (maps [0,1) → [-2,2]).
# --------------------------------------------------------
class ToyRegression(Dataset):
    """y = 3x + 1 + ε (generated once; random-access by index)."""
    def __init__(self, n=64, noise_std=0.3, seed=0,
                 transform: Optional[Callable]=None,
                 target_transform: Optional[Callable]=None):
        g = torch.Generator().manual_seed(seed)

        # x ~ U[-2, 2]
        # ------------------------------------------------------------ 
        # Why torch.empty(...).uniform_() instead of torch.Tensor.uniform_? 
        #   - torch.empty(n, 1) creates an uninitialized tensor of shape (n,1). 
        #   - .uniform_(-2, 2, generator=g) fills it in-place with uniform values. 
        #   - This works because uniform_ is an INSTANCE method of Tensor. 
        # torch.Tensor is just a class constructor, not a function. 
        # There is no torch.Tensor.uniform_(size=...) convenience form. 
        # So you cannot do torch.Tensor.uniform_(-2,2, size=(n,1), generator=g). 
        # That would error, since uniform_ expects an existing Tensor (self).  
        # Alternatives: 
        #   1) torch.empty((n,1), generator=g).uniform_(-2, 2) 
        #   2) torch.rand(n,1, generator=g) * 4 - 2 (scales [0,1) → [-2,2])  
        # Key idea: PyTorch random sampling is usually "allocate empty → fill in-place". 
        # ------------------------------------------------------------
        self.x = torch.empty(n, 1).uniform_(-2, 2, generator=g)

        # y = 3x + 1 + noise
        self.y = 3 * self.x + 1 + noise_std * torch.randn(self.x.shape, generator=g)

        # Optional transforms applied in __getitem__
        self.transform = transform
        self.target_transform = target_transform

        # Stats (from the raw base data) for normalization
        self.xm, self.xs = self.x.mean(), self.x.std(unbiased=False)
        self.ym, self.ys = self.y.mean(), self.y.std(unbiased=False)

    def __len__(self):  # dataset size
        return self.x.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        xi, yi = self.x[idx], self.y[idx]
        if self.transform:        # input transform
            xi = self.transform(xi)
        if self.target_transform: # target transform
            yi = self.target_transform(yi)
        return xi, yi


# --------------------------------------------------------
# Demo with visualization
# --------------------------------------------------------
def main():
    # Base dataset (raw; no transforms)
    base = ToyRegression(n=64, seed=1)

    # Build transforms using base stats
    x_tf = Compose([
        Standardize(base.xm.item(), base.xs.item()),
        AddNoise(0.2, seed=123),
    ])
    y_tf = Standardize(base.ym.item(), base.ys.item())

    # Dataset that applies transforms on-the-fly in __getitem__
    ds = ToyRegression(n=64, seed=1, transform=x_tf, target_transform=y_tf)

    # ---------------- Plotting ----------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

    # Original (no transforms)
    ax0.scatter(
        base.x.detach().cpu().numpy(),
        base.y.detach().cpu().numpy(),
        c="blue", alpha=0.7, label="original"
    )
    ax0.set_title("Original Data")
    ax0.set_xlabel("x"); ax0.set_ylabel("y")
    ax0.legend()

    # Transformed (sample-by-sample via __getitem__)
    # Build a Python list of length N by indexing the dataset:
    #   - Each ds[i] calls __getitem__(i), which applies transform/target_transform
    #     on the fly and returns a pair (xi, yi) of tensors.
    #   - The list thus looks like [(x0,y0), (x1,y1), ..., (x_{N-1}, y_{N-1})].
    #   pairs       : [(x0, y0), (x1, y1), ..., (x_{N-1}, y_{N-1})]
    #   *pairs      : (x0,y0), (x1,y1), ..., (x_{N-1},y_{N-1})
    #   zip(*pairs) : zip((x0,y0), (x1,y1), ..., (x_{N-1},y_{N-1}))
    #   xs          : (x0, x1, ..., x_{N-1})
    #   ys          : (y0, y1, ..., y_{N-1})
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    ax1.scatter(
        # xs : list of 1d tensors of shape (1,)
        # torch.stack(xs).shape : (N,1)
        torch.stack(xs).detach().cpu().numpy().reshape((-1,)),
        torch.stack(ys).detach().cpu().numpy().reshape((-1,)),
        c="red", alpha=0.7, label="transformed"
    )
    ax1.set_title("Transformed Data")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()