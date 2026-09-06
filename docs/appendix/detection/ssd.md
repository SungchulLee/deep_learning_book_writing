# SSD

SSD was introduced in the 2016 paper "SSD: Single Shot MultiBox Detector." Multi-scale feature maps for detection, single forward pass.

This implementation provides a concise, educational reference for SSD. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
'''
SSD - Single Shot MultiBox Detector
Paper: "SSD: Single Shot MultiBox Detector" (2016)
Key: Multi-scale feature maps for detection, single forward pass
'''
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================

class SSD(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # Base network (VGG-like)
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Extra layers for multi-scale detection
        self.extras = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Detection heads
        self.loc = nn.Conv2d(256, 4 * 4, 3, padding=1)  # 4 default boxes * 4 coords
        self.conf = nn.Conv2d(256, 4 * num_classes, 3, padding=1)  # 4 boxes * num_classes
    
    def forward(self, x):
        x = self.base(x)
        x = self.extras(x)
        
        loc = self.loc(x)
        conf = self.conf(x)
        
        return {'loc': loc, 'conf': conf}

if __name__ == "__main__":
    model = SSD()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")```

## Discussion

The `SSD` class encapsulates the model architecture using PyTorch's `nn.Module` interface. The `forward` method defines the computational graph, allowing PyTorch's autograd system to handle gradient computation automatically during training. This modular design makes it straightforward to modify individual components or integrate the model into larger pipelines.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Trace the tensor shapes through the `SSD` forward pass. For a batch of 4 input samples with the default parameters, write down the shape after each major operation (convolution, pooling, linear layer).

??? success "Solution to Exercise 1"
    Start from the input shape and apply each layer sequentially. For each `Conv2d(in_c, out_c, k)`, the spatial dimensions change as $H_{\text{out}} = H_{\text{in}} - k + 1$ (without padding) or remain the same (with `padding=k//2`). Pooling with kernel 2 halves spatial dimensions. Linear layers transform the last dimension. Track the batch dimension throughout: it remains unchanged. Write each intermediate shape as $(B, C, H, W)$ for convolutional layers and $(B, F)$ after flattening.

---

**Exercise 2.**
Modify the architecture to accept RGB images of size $64 \times 64$ (input shape: $3 \times 64 \times 64$). Update all layer dimensions accordingly and verify the model runs without errors.

??? success "Solution to Exercise 2"
    Change the first convolutional layer's `in_channels` from the current value to 3. Recalculate the spatial dimensions after each convolution and pooling layer using the formulas $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$. Update the first linear layer's `in_features` to match the flattened output of the last convolutional/pooling layer. Verify with: `model = SSD(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**Exercise 3.**
Compare the number of parameters and FLOPs between a standard convolution and a depthwise separable convolution for the same input/output dimensions. When is the computational saving most significant?

??? success "Solution to Exercise 3"
    A standard `Conv2d(C_in, C_out, k)` has $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$ parameters. A depthwise separable convolution splits this into: (1) depthwise: $C_{{\text{{in}}}} \times k^2$ parameters (one filter per input channel), and (2) pointwise: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$ parameters (1x1 conv). The ratio of parameters is approximately $1/C_{{\text{{out}}}} + 1/k^2$. For $k=3$ and $C_{{\text{{out}}}}=256$, this is about $8{-}9\times$ fewer parameters. The savings are most significant when both $C_{{\text{{out}}}}$ and $k$ are large.

---

**Exercise 4.**
Extend `SSD` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = SSD(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
