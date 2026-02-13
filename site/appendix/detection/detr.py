#!/usr/bin/env python3
"""
DETR - End-to-End Object Detection with Transformers
Paper: "End-to-End Object Detection with Transformers" (2020)
Authors: Nicolas Carion et al.
Key: CNN backbone + Transformer encoder/decoder + fixed set of object queries.
     Predicts a set of boxes/classes directly (no anchors, no NMS in original formulation).

File: appendix/detection/detr.py
Note: Educational/simplified implementation focusing on the model forward pass.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    """
    Simple 2D sine-cosine positional encoding for feature maps.
    Real DETR uses learned or sine-cos embeddings; this is a compact sine-cos version.
    """
    def __init__(self, d_model: int = 256, temperature: int = 10000):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sine-cos encoding.")
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W)
        B, C, H, W = feat.shape
        device = feat.device

        y = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)  # (H, W)
        x = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)  # (H, W)

        # Normalize to [0, 2pi] range
        y = y / (H - 1 + 1e-6) * 2 * math.pi
        x = x / (W - 1 + 1e-6) * 2 * math.pi

        # Compute frequencies
        dim_t = torch.arange(self.d_model // 4, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.d_model // 2))

        # (H, W, D/4)
        pos_x = x[..., None] / dim_t
        pos_y = y[..., None] / dim_t

        # sine-cos pairs
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)  # (H, W, D/2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)  # (H, W, D/2)

        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H, W, D)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, D, H, W)
        return pos


class TinyBackbone(nn.Module):
    """
    Very small CNN backbone producing a feature map.
    In real DETR: ResNet + 1x1 projection to d_model.
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(128, d_model, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # (B, d_model, H/4, W/4)


class DETR(nn.Module):
    """
    Simplified DETR:
    - backbone -> feature map
    - add pos encoding
    - transformer encoder/decoder
    - object queries -> decoder outputs
    - heads: class logits + box coordinates

    Output:
      - pred_logits: (B, num_queries, num_classes+1)  (+1 for "no object")
      - pred_boxes : (B, num_queries, 4) in normalized cx,cy,w,h (sigmoid)
    """
    def __init__(self, num_classes=20, num_queries=100, d_model=256, nhead=8, num_enc=6, num_dec=6):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        self.backbone = TinyBackbone(d_model=d_model)
        self.pos_enc = PositionalEncoding2D(d_model=d_model)

        # PyTorch Transformer uses sequence-first: (S, B, E)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=False,
        )

        # Learnable object queries (num_queries, d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Prediction heads
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1: "no-object"
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

    def forward(self, x):
        """
        x: (B, 3, H, W)

        Steps:
        1) backbone -> feat (B, C, H', W')
        2) flatten spatial -> src sequence (S=H'*W', B, C)
        3) create query sequence (T=num_queries, B, C)
        4) transformer(src, tgt) -> hs (T, B, C)
        5) heads -> logits/boxes
        """
        feat = self.backbone(x)               # (B, C, H', W')
        pos = self.pos_enc(feat)              # (B, C, H', W')

        B, C, H, W = feat.shape
        src = feat.flatten(2).permute(2, 0, 1)  # (S=H*W, B, C)
        pos = pos.flatten(2).permute(2, 0, 1)   # (S, B, C)

        # Add position to src (common DETR trick)
        src = src + pos

        # Object queries as initial target tokens (T, B, C)
        query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        # Transformer: encoder processes src, decoder processes query attending to src
        hs = self.transformer(src=src, tgt=query)  # (T, B, C)

        # Convert to batch-first: (B, T, C)
        hs = hs.permute(1, 0, 2)

        pred_logits = self.class_head(hs)           # (B, T, num_classes+1)
        pred_boxes = torch.sigmoid(self.box_head(hs))  # (B, T, 4) normalized

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


if __name__ == "__main__":
    model = DETR(num_classes=20, num_queries=100)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("pred_logits:", y["pred_logits"].shape)  # (2, 100, 21)
    print("pred_boxes :", y["pred_boxes"].shape)   # (2, 100, 4)
