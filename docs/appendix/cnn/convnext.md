# ConvNeXt

ConvNeXt, presented in the 2022 paper "A ConvNet for the 2020s," systematically modernizes the classic ResNet architecture by incorporating design principles borrowed from Vision Transformers. The result is a pure convolutional architecture that matches or exceeds the performance of Swin Transformer across multiple vision benchmarks, demonstrating that the inductive biases of convolutions remain highly competitive when paired with modern training recipes and architectural choices.

## Code

```python
#!/usr/bin/env python3
'''
ConvNeXt - A ConvNet for the 2020s
Paper: "A ConvNet for the 2020s" (2022)
Key: Modernized ResNet with design choices from Vision Transformers
'''
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + x

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 4, stride=4),
            nn.LayerNorm([96, 56, 56])
        )
        self.stages = nn.Sequential(*[ConvNeXtBlock(96) for _ in range(3)])
        self.norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = x.mean([2, 3])
        x = self.norm(x)
        return self.head(x)

if __name__ == "__main__":
    model = ConvNeXt()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Discussion

The ConvNeXt block draws several design choices from transformers. The $7 \times 7$ depthwise convolution mirrors the large receptive field of self-attention, while using `groups=dim` keeps computation efficient. LayerNorm replaces BatchNorm, which aligns with transformer practice and improves performance especially in transfer learning scenarios where batch statistics may be unreliable. The inverted bottleneck design (expanding to $4 \times$ the dimension, then projecting back) directly parallels the feedforward network in a transformer block.

Layer scale, controlled by the learnable parameter `gamma` initialized to a very small value ($10^{-6}$), is another critical ingredient. It allows the network to start training with near-identity residual connections, which stabilizes optimization for deeper models. The use of GELU activation instead of ReLU and the patchify stem (a single $4 \times 4$ stride-4 convolution) rather than the traditional multi-layer aggressive downsampling stem are additional modernizations that collectively close the gap with transformer architectures.

The broader lesson of ConvNeXt is that many perceived advantages of vision transformers stem not from self-attention itself, but from accompanying design and training improvements. When these improvements are transferred back to convolutional architectures, pure ConvNets remain highly competitive, while retaining the efficiency benefits of translation equivariance and local connectivity.

## Exercises

**Exercise 1.**
Calculate the number of parameters in a single `ConvNeXtBlock` with `dim=96`.

??? success "Solution to Exercise 1"
    The depthwise convolution has $96 \times 7 \times 7 + 96 = 4800$ parameters (weights + bias). LayerNorm has $96 + 96 = 192$ parameters (weight and bias). The first pointwise linear layer: $96 \times 384 + 384 = 37,248$. The second pointwise linear layer: $384 \times 96 + 96 = 36,960$. Layer scale gamma: $96$. Total: $4800 + 192 + 37,248 + 36,960 + 96 = 79,296$ parameters.

---

**Exercise 2.**
Explain the purpose of the `permute` operations in the forward method of `ConvNeXtBlock`. Why not use `nn.LayerNorm` directly on the NCHW tensor?

??? success "Solution to Exercise 2"
    PyTorch's `nn.LayerNorm` normalizes over the last dimension(s) of the input. For a tensor in NCHW format, applying LayerNorm would normalize over W (width) only, or require specifying `[C, H, W]` which would couple the normalization to a specific spatial resolution. By permuting to NHWC format, LayerNorm naturally normalizes over the channel dimension C at each spatial location, which is the desired behavior (analogous to how LayerNorm works in transformers over the feature dimension). The linear layers also operate on the last dimension, so the NHWC layout lets them process channels directly. The final permute restores NCHW for compatibility with the depthwise convolution.

---

**Exercise 3.**
Modify the `ConvNeXt` model to support a multi-stage architecture with downsampling between stages. Add a second stage with `dim=192` and a downsampling layer between the two stages.

??? success "Solution to Exercise 3"
    ```python
    class ConvNeXtMultiStage(nn.Module):
        def __init__(self, num_classes=1000):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 96, 4, stride=4),
                nn.LayerNorm([96, 56, 56])
            )
            self.stage1 = nn.Sequential(*[ConvNeXtBlock(96) for _ in range(3)])
            self.downsample = nn.Sequential(
                nn.LayerNorm([96, 56, 56]),
                nn.Conv2d(96, 192, 2, stride=2),
            )
            self.stage2 = nn.Sequential(*[ConvNeXtBlock(192) for _ in range(3)])
            self.norm = nn.LayerNorm(192)
            self.head = nn.Linear(192, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.downsample(x)
            x = self.stage2(x)
            x = x.mean([2, 3])
            x = self.norm(x)
            return self.head(x)
    ```
    The downsampling layer uses LayerNorm followed by a $2 \times 2$ stride-2 convolution, which halves the spatial resolution while doubling the channel count from 96 to 192.
