#!/usr/bin/env python3
"""
SegFormer - Simple and Efficient Design for Semantic Segmentation
Paper: "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" (2021)
Authors: Enze Xie et al.
Key: Transformer-based encoder with lightweight MLP decoder; no positional
encoding and no convolutions in the decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP used in SegFormer decoder"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


class SegFormer(nn.Module):
    """
    Simplified SegFormer-style model (educational version)

    - Transformer-like encoder is mocked by convolution layers
    - Lightweight MLP decoder
    - Suitable for appendix / conceptual understanding
    """

    def __init__(self, num_classes=21):
        super().__init__()

        # Encoder (simplified, CNN-based for clarity)
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Decoder (MLP head)
        self.mlp1 = MLP(64, 256)
        self.mlp2 = MLP(128, 256)
        self.mlp3 = MLP(256, 256)
        self.mlp4 = MLP(512, 256)

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        f1 = F.relu(self.enc1(x))              # (B, 64, H, W)
        f2 = F.relu(self.enc2(f1))             # (B, 128, H, W)
        f3 = F.relu(self.enc3(f2))             # (B, 256, H, W)
        f4 = F.relu(self.enc4(f3))             # (B, 512, H, W)

        # Flatten spatial dimensions
        def mlp_process(f, mlp):
            B, C, H, W = f.shape
            f = f.flatten(2).transpose(1, 2)   # (B, HW, C)
            f = mlp(f)
            f = f.transpose(1, 2).reshape(B, -1, H, W)
            return f

        f1 = mlp_process(f1, self.mlp1)
        f2 = mlp_process(f2, self.mlp2)
        f3 = mlp_process(f3, self.mlp3)
        f4 = mlp_process(f4, self.mlp4)

        # Fuse features
        fused = f1 + f2 + f3 + f4

        # Segmentation head
        out = self.classifier(fused)
        return out


if __name__ == "__main__":
    model = SegFormer(num_classes=19)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)  # (1, 19, 224, 224)
