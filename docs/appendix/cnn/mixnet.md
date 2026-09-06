# MixNet

MixNet introduced the idea of using multiple kernel sizes within a single depthwise convolution layer, allowing the network to capture patterns at different spatial scales simultaneously. Proposed in the 2019 paper "MixConv: Mixed Depthwise Convolutional Kernels," MixNet builds on the efficiency of depthwise separable convolutions used in MobileNets while enriching the feature extraction by mixing kernels of sizes 3, 5, and 7.

## Code

```python
#!/usr/bin/env python3
'''
MixNet - Mixed Depthwise Convolutional Kernels
Paper: "MixConv: Mixed Depthwise Convolutional Kernels" (2019)
Key: Multiple kernel sizes in a single depthwise convolution layer
'''
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================

class MixConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.groups = len(kernel_sizes)
        assert out_channels % self.groups == 0
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels // self.groups, out_channels // self.groups, 
                     k, padding=k//2, groups=in_channels // self.groups)
            for k in kernel_sizes
        ])
    
    def forward(self, x):
        chunks = torch.chunk(x, self.groups, dim=1)
        outs = [conv(chunk) for conv, chunk in zip(self.convs, chunks)]
        return torch.cat(outs, dim=1)

class MixNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.mixconv = MixConv2d(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = torch.nn.functional.relu6(self.bn1(self.conv1(x)))
        out = torch.nn.functional.relu6(self.bn2(self.mixconv(out)))
        out = self.bn3(self.conv2(out))
        if x.shape == out.shape:
            out = out + x
        return out

class MixNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.blocks = nn.Sequential(
            MixNetBlock(16, 24),
            MixNetBlock(24, 24),
        )
        self.conv_head = nn.Conv2d(24, 1536, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1536)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1536, num_classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu6(self.bn1(self.stem(x)))
        x = self.blocks(x)
        x = torch.nn.functional.relu6(self.bn2(self.conv_head(x)))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = MixNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")```

## Discussion

The implementation defines 3 classes (`MixConv2d`, `MixNetBlock`, `MixNet`) that work together to form the complete convolutional neural network architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Trace the tensor shapes through the `MixConv2d` forward pass. For a batch of 4 input samples with the default parameters, write down the shape after each major operation (convolution, pooling, linear layer).

??? success "Solution to Exercise 1"
    Start from the input shape and apply each layer sequentially. For each `Conv2d(in_c, out_c, k)`, the spatial dimensions change as $H_{\text{out}} = H_{\text{in}} - k + 1$ (without padding) or remain the same (with `padding=k//2`). Pooling with kernel 2 halves spatial dimensions. Linear layers transform the last dimension. Track the batch dimension throughout: it remains unchanged. Write each intermediate shape as $(B, C, H, W)$ for convolutional layers and $(B, F)$ after flattening.

---

**Exercise 2.**
Modify the architecture to accept RGB images of size $64 \times 64$ (input shape: $3 \times 64 \times 64$). Update all layer dimensions accordingly and verify the model runs without errors.

??? success "Solution to Exercise 2"
    Change the first convolutional layer's `in_channels` from the current value to 3. Recalculate the spatial dimensions after each convolution and pooling layer using the formulas $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$. Update the first linear layer's `in_features` to match the flattened output of the last convolutional/pooling layer. Verify with: `model = MixConv2d(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**Exercise 3.**
Compare the number of parameters and FLOPs between a standard convolution and a depthwise separable convolution for the same input/output dimensions. When is the computational saving most significant?

??? success "Solution to Exercise 3"
    A standard `Conv2d(C_in, C_out, k)` has $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$ parameters. A depthwise separable convolution splits this into: (1) depthwise: $C_{{\text{{in}}}} \times k^2$ parameters (one filter per input channel), and (2) pointwise: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$ parameters (1x1 conv). The ratio of parameters is approximately $1/C_{{\text{{out}}}} + 1/k^2$. For $k=3$ and $C_{{\text{{out}}}}=256$, this is about $8{-}9\times$ fewer parameters. The savings are most significant when both $C_{{\text{{out}}}}$ and $k$ are large.

---

**Exercise 4.**
Extend `MixConv2d` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = MixConv2d(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
