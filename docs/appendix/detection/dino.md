# DINO

DINO was introduced in the 2022 paper "DINO: DETR with Improved Denoising Anchor Boxes for End-to-End Object Detection."

This implementation provides a concise, educational reference for DINO. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
DINO - DETR with Improved Denoising Anchor Boxes
Paper: "DINO: DETR with Improved Denoising Anchor Boxes for End-to-End Object Detection" (2022)
Authors: IDEA Research (Feng Li et al.)
Key ideas (high-level):
  1) Denoising training (DN): add noisy GT queries to stabilize training.
  2) Better query initialization (often anchor-like / reference points).
  3) Multi-scale features (often via deformable attention variants in practice).

File: appendix/detection/dino.py
Note: Educational/simplified implementation:
  - Shows "reference points" concept and a DN-style query augmentation hook.
  - Uses nn.Transformer (not deformable attention).
  - Skips full matching/loss details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================


class TinyBackbone(nn.Module):
    """Small CNN backbone producing a feature map (single-scale for simplicity)."""
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


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Common DETR-family trick for box refinement in logit space."""
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class DINO(nn.Module):
    """
    Simplified DINO-like detector:
    - object queries + "reference points" (normalized boxes) used for iterative refinement idea
    - optional denoising queries (DN) in training mode

    Output:
      - pred_logits: (B, num_queries, num_classes+1)
      - pred_boxes : (B, num_queries, 4) normalized
    """
    def __init__(
        self,
        num_classes=20,
        num_queries=300,
        d_model=256,
        nhead=8,
        num_enc=6,
        num_dec=6,
        dn_num_queries=50,     # number of denoising queries (educational default)
        dn_noise_scale=0.4,    # how much noise to add to GT boxes (educational)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        self.backbone = TinyBackbone(d_model=d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=False,
        )

        # Learnable queries (content embeddings)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Reference points (anchor-like initialization in normalized coordinates)
        # In real DINO, these can be predicted/updated per decoder layer.
        self.refpoint_embed = nn.Embedding(num_queries, 4)

        # Heads
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

        # DN config (used only if gt provided)
        self.dn_num_queries = dn_num_queries
        self.dn_noise_scale = dn_noise_scale

    def make_denoising_queries(self, gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        """
        Create denoising (DN) queries from ground-truth:
          - Add noise to GT boxes
          - Use them as extra queries during training to stabilize learning

        gt_boxes: (B, M, 4) normalized (cx,cy,w,h)
        gt_labels: (B, M) class ids in [0, num_classes-1]

        Returns:
          dn_query_embed: (Tdn, B, C)
          dn_refpoints : (Tdn, B, 4)
        """
        B, M, _ = gt_boxes.shape
        device = gt_boxes.device

        # Choose up to dn_num_queries GT boxes per batch (simple truncation)
        M_use = min(M, self.dn_num_queries)
        boxes = gt_boxes[:, :M_use, :]  # (B, M_use, 4)

        # Add noise in normalized space (educational)
        noise = (torch.rand_like(boxes) - 0.5) * self.dn_noise_scale
        noisy_boxes = (boxes + noise).clamp(0.0, 1.0)

        # Create DN content embeddings:
        # Real methods embed labels + mask; here we just use a learned "dn token" repeated.
        dn_token = torch.zeros(B, M_use, self.d_model, device=device)
        dn_token = dn_token.permute(1, 0, 2).contiguous()  # (Tdn=M_use, B, C)

        dn_ref = noisy_boxes.permute(1, 0, 2).contiguous()  # (Tdn, B, 4)
        return dn_token, dn_ref

    def forward(self, x, gt_boxes=None, gt_labels=None):
        """
        x: (B, 3, H, W)
        gt_boxes/gt_labels: optional, used to add DN queries in training-like mode
        """
        feat = self.backbone(x)
        B, C, H, W = feat.shape

        # Encoder input: flatten spatial feature map
        src = feat.flatten(2).permute(2, 0, 1)  # (S=H*W, B, C)

        # Standard learned queries
        q_content = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (T, B, C)
        q_ref = torch.sigmoid(self.refpoint_embed.weight).unsqueeze(1).repeat(1, B, 1)  # (T, B, 4)

        # Optionally add denoising queries (DN) in front of normal queries
        if (gt_boxes is not None) and (gt_labels is not None):
            dn_content, dn_ref = self.make_denoising_queries(gt_boxes, gt_labels)
            q_content = torch.cat([dn_content, q_content], dim=0)  # (Tdn+T, B, C)
            q_ref = torch.cat([dn_ref, q_ref], dim=0)              # (Tdn+T, B, 4)

        # Decoder output tokens
        hs = self.transformer(src=src, tgt=q_content)  # (Ttotal, B, C)
        hs = hs.permute(1, 0, 2)                       # (B, Ttotal, C)

        # Predict class logits
        pred_logits = self.class_head(hs)  # (B, Ttotal, num_classes+1)

        # Predict box deltas, then "refine" around reference points (common DETR-family idea)
        # In practice, refinement happens layer-by-layer. Here: one-shot refinement.
        delta = self.box_head(hs)                 # (B, Ttotal, 4)
        ref = q_ref.permute(1, 0, 2).contiguous() # (B, Ttotal, 4)

        # refinement in logit space: inv_sigmoid(ref) + delta -> sigmoid
        pred_boxes = torch.sigmoid(inverse_sigmoid(ref) + delta)  # (B, Ttotal, 4)

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


if __name__ == "__main__":
    model = DINO(num_classes=20, num_queries=300)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("no-DN pred_logits:", y["pred_logits"].shape)
    print("no-DN pred_boxes :", y["pred_boxes"].shape)

    # Example DN mode with fake GT boxes/labels
    gt_boxes = torch.rand(2, 10, 4)     # (B=2, M=10, 4) normalized
    gt_labels = torch.randint(0, 20, (2, 10))
    y_dn = model(x, gt_boxes=gt_boxes, gt_labels=gt_labels)
    print("DN pred_logits:", y_dn["pred_logits"].shape)
    print("DN pred_boxes :", y_dn["pred_boxes"].shape)```

## Discussion

The implementation defines 2 classes (`TinyBackbone`, `DINO`) that work together to form the complete object detection architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Trace the tensor shapes through the `TinyBackbone` forward pass. For a batch of 4 input samples with the default parameters, write down the shape after each major operation (convolution, pooling, linear layer).

??? success "Solution to Exercise 1"
    Start from the input shape and apply each layer sequentially. For each `Conv2d(in_c, out_c, k)`, the spatial dimensions change as $H_{\text{out}} = H_{\text{in}} - k + 1$ (without padding) or remain the same (with `padding=k//2`). Pooling with kernel 2 halves spatial dimensions. Linear layers transform the last dimension. Track the batch dimension throughout: it remains unchanged. Write each intermediate shape as $(B, C, H, W)$ for convolutional layers and $(B, F)$ after flattening.

---

**Exercise 2.**
Modify the architecture to accept RGB images of size $64 \times 64$ (input shape: $3 \times 64 \times 64$). Update all layer dimensions accordingly and verify the model runs without errors.

??? success "Solution to Exercise 2"
    Change the first convolutional layer's `in_channels` from the current value to 3. Recalculate the spatial dimensions after each convolution and pooling layer using the formulas $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$. Update the first linear layer's `in_features` to match the flattened output of the last convolutional/pooling layer. Verify with: `model = TinyBackbone(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**Exercise 3.**
Explain the computational complexity of self-attention as a function of sequence length $n$ and model dimension $d$. Why does this motivate architectures like Longformer or Linformer for long sequences?

??? success "Solution to Exercise 3"
    Standard self-attention computes an $n \times n$ attention matrix, giving $O(n^2 d)$ time complexity and $O(n^2)$ memory for the attention weights. For long sequences (e.g., $n = 4096$), this becomes prohibitive. Longformer uses a combination of local sliding-window attention ($O(n \cdot w \cdot d)$ where $w$ is window size) and sparse global attention for selected tokens. Linformer projects keys and values to a lower dimension $k \ll n$, reducing complexity to $O(n \cdot k \cdot d)$. Both trade some expressiveness for practical efficiency on long inputs.

---

**Exercise 4.**
Extend `TinyBackbone` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = TinyBackbone(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
