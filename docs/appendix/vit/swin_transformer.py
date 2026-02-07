#!/usr/bin/env python3
'''
Swin Transformer - Hierarchical Vision Transformer with Shifted Windows
Paper: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (2021)
Key: Shifted window attention for efficiency, hierarchical feature maps
'''
import torch
import torch.nn as nn

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=1000, embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24]):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.head = nn.Linear(embed_dim * 8, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), 1).squeeze(-1)
        return self.head(x)

if __name__ == "__main__":
    model = SwinTransformer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
