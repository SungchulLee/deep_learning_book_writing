#!/usr/bin/env python3
# ========================================================
# 12_dataloader_6_transforms_joint_transforms_spatial.py
# ========================================================
"""
Joint transforms: apply the SAME random params to (x, y) together.
Example: 2D crop/flip applied to an "image" and its mask/label map.
"""

from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
import random

# ----- joint transforms -----
class RandomHorizontalFlip2D:
    def __init__(self, p=0.5, seed=None): self.p, self.seed = p, seed
    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if self.seed is not None: random.seed(self.seed)
        if random.random() < self.p:
            # tensors [C,H,W] or [H,W]; flip width dim
            if x.ndim == 3: x = x.flip(dims=[2])
            else: x = x.flip(dims=[1])
            y = y.flip(dims=[1])
        return x, y

class RandomCrop2D:
    def __init__(self, size: Tuple[int,int], seed=None):
        self.th, self.tw = size; self.seed = seed
    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if self.seed is not None: random.seed(self.seed)
        C,H,W = (x.shape if x.ndim==3 else (1,)+tuple(x.shape))
        if H < self.th or W < self.tw:  # pad if needed (simple zero-pad)
            pad_h = max(0, self.th - H); pad_w = max(0, self.tw - W)
            pad = (0, pad_w, 0, pad_h)  # (left,right,top,bottom)
            xp = torch.nn.functional.pad(x, pad=(pad[0], pad[1], pad[2], pad[3]))
            yp = torch.nn.functional.pad(y, pad=(pad[0], pad[1], pad[2], pad[3]))
            x, y = xp, yp
            if x.ndim==2: H, W = x.shape
            else: _, H, W = x.shape
        i = random.randint(0, H - self.th)
        j = random.randint(0, W - self.tw)
        if x.ndim==3: x = x[:, i:i+self.th, j:j+self.tw]
        else: x = x[i:i+self.th, j:j+self.tw]
        y = y[i:i+self.th, j:j+self.tw]
        return x, y

# ----- dataset producing image & mask -----
class ToySeg(Dataset):
    """Creates a simple 1-channel 'image' and a square 'mask'."""
    def __init__(self, n=5, H=32, W=32, seed=0, joint_transform: Optional[callable]=None):
        g = torch.Generator().manual_seed(seed)
        self.imgs, self.masks = [], []
        for _ in range(n):
            img = torch.randn(1, H, W, generator=g)               # [C,H,W]
            mask = torch.zeros(H, W, dtype=torch.long)
            # draw a random square
            top = int(torch.randint(4, H-8, (1,), generator=g).item())
            left = int(torch.randint(4, W-8, (1,), generator=g).item())
            size = int(torch.randint(6, 10, (1,), generator=g).item())
            mask[top:top+size, left:left+size] = 1
            self.imgs.append(img); self.masks.append(mask)
        self.joint_transform = joint_transform

    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        x, y = self.imgs[i].clone(), self.masks[i].clone()
        if self.joint_transform: x, y = self.joint_transform(x, y)
        return x, y

def main():
    jt = lambda x,y: RandomCrop2D((24,24))( *RandomHorizontalFlip2D(p=0.9)(x,y) )
    ds = ToySeg(n=3, joint_transform=jt)
    for i in range(len(ds)):
        x, y = ds[i]
        print(f"{i}: x={tuple(x.shape)} y={tuple(y.shape)}, mask_sum={int(y.sum().item())}")

if __name__ == "__main__":
    main()

