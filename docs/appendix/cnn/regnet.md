# RegNet

RegNet, introduced by Facebook AI Research in 2020, takes a design-space approach to neural architecture development. Rather than searching for a single optimal network, RegNet defines a structured design space parameterized by a small number of variables (depth, initial width, slope, and group width). Networks sampled from this space follow a simple linear parameterization of widths across stages, yielding models that are fast, memory-efficient, and highly competitive.

## Code

```python
#!/usr/bin/env python3
'''
RegNet - Designing Network Design Spaces
Paper: "Designing Network Design Spaces" (2020)
Key: Quantized linear parametrization of network width/depth
'''
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================

class RegNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)

class RegNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.layer1 = self._make_layer(32, 64, 2)
        self.head = nn.Linear(64, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1
            layers.append(RegNetBlock(in_channels if i == 0 else out_channels, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

if __name__ == "__main__":
    model = RegNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")```

## Discussion

The implementation defines 2 classes (`RegNetBlock`, `RegNet`) that work together to form the complete convolutional neural network architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Trace the tensor shapes through the `RegNetBlock` forward pass. For a batch of 4 input samples with the default parameters, write down the shape after each major operation (convolution, pooling, linear layer).

??? success "Solution to Exercise 1"
    Start from the input shape and apply each layer sequentially. For each `Conv2d(in_c, out_c, k)`, the spatial dimensions change as $H_{\text{out}} = H_{\text{in}} - k + 1$ (without padding) or remain the same (with `padding=k//2`). Pooling with kernel 2 halves spatial dimensions. Linear layers transform the last dimension. Track the batch dimension throughout: it remains unchanged. Write each intermediate shape as $(B, C, H, W)$ for convolutional layers and $(B, F)$ after flattening.

---

**Exercise 2.**
Modify the architecture to accept RGB images of size $64 \times 64$ (input shape: $3 \times 64 \times 64$). Update all layer dimensions accordingly and verify the model runs without errors.

??? success "Solution to Exercise 2"
    Change the first convolutional layer's `in_channels` from the current value to 3. Recalculate the spatial dimensions after each convolution and pooling layer using the formulas $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$. Update the first linear layer's `in_features` to match the flattened output of the last convolutional/pooling layer. Verify with: `model = RegNetBlock(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**Exercise 3.**
Compare the number of parameters and FLOPs between a standard convolution and a depthwise separable convolution for the same input/output dimensions. When is the computational saving most significant?

??? success "Solution to Exercise 3"
    A standard `Conv2d(C_in, C_out, k)` has $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$ parameters. A depthwise separable convolution splits this into: (1) depthwise: $C_{{\text{{in}}}} \times k^2$ parameters (one filter per input channel), and (2) pointwise: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$ parameters (1x1 conv). The ratio of parameters is approximately $1/C_{{\text{{out}}}} + 1/k^2$. For $k=3$ and $C_{{\text{{out}}}}=256$, this is about $8{-}9\times$ fewer parameters. The savings are most significant when both $C_{{\text{{out}}}}$ and $k$ are large.

---

**Exercise 4.**
Extend `RegNetBlock` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = RegNetBlock(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
