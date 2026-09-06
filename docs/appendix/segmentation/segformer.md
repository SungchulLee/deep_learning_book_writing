# SegFormer

SegFormer was introduced in the 2021 paper "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." Transformer-based encoder with lightweight MLP decoder; no positional.

This implementation provides a concise, educational reference for SegFormer. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
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

# ========================================================================
# Main
# ========================================================================


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
    print(y.shape)  # (1, 19, 224, 224)```

## Discussion

The implementation defines 2 classes (`MLP`, `SegFormer`) that work together to form the complete image segmentation architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Trace the tensor shapes through the `MLP` forward pass. For a batch of 4 input samples with the default parameters, write down the shape after each major operation (convolution, pooling, linear layer).

??? success "Solution to Exercise 1"
    Start from the input shape and apply each layer sequentially. For each `Conv2d(in_c, out_c, k)`, the spatial dimensions change as $H_{\text{out}} = H_{\text{in}} - k + 1$ (without padding) or remain the same (with `padding=k//2`). Pooling with kernel 2 halves spatial dimensions. Linear layers transform the last dimension. Track the batch dimension throughout: it remains unchanged. Write each intermediate shape as $(B, C, H, W)$ for convolutional layers and $(B, F)$ after flattening.

---

**Exercise 2.**
Modify the architecture to accept RGB images of size $64 \times 64$ (input shape: $3 \times 64 \times 64$). Update all layer dimensions accordingly and verify the model runs without errors.

??? success "Solution to Exercise 2"
    Change the first convolutional layer's `in_channels` from the current value to 3. Recalculate the spatial dimensions after each convolution and pooling layer using the formulas $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$. Update the first linear layer's `in_features` to match the flattened output of the last convolutional/pooling layer. Verify with: `model = MLP(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**Exercise 3.**
Compare the number of parameters and FLOPs between a standard convolution and a depthwise separable convolution for the same input/output dimensions. When is the computational saving most significant?

??? success "Solution to Exercise 3"
    A standard `Conv2d(C_in, C_out, k)` has $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$ parameters. A depthwise separable convolution splits this into: (1) depthwise: $C_{{\text{{in}}}} \times k^2$ parameters (one filter per input channel), and (2) pointwise: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$ parameters (1x1 conv). The ratio of parameters is approximately $1/C_{{\text{{out}}}} + 1/k^2$. For $k=3$ and $C_{{\text{{out}}}}=256$, this is about $8{-}9\times$ fewer parameters. The savings are most significant when both $C_{{\text{{out}}}}$ and $k$ are large.

---

**Exercise 4.**
Extend `MLP` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = MLP(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
