# ResNeSt

ResNeSt (Split-Attention Networks), proposed in 2020, extends ResNet by introducing split-attention blocks that apply channel-wise attention across multiple feature-map groups within each residual block. Inspired by SE-Net and SK-Net, ResNeSt captures cross-channel interactions at a finer granularity, improving representation learning for tasks like image classification, object detection, and semantic segmentation without significantly increasing computational cost.

## Code

```python
#!/usr/bin/env python3
'''
ResNeSt - Split-Attention Networks
Paper: "ResNeSt: Split-Attention Networks" (2020)
Key: Split-attention blocks with channel-wise attention across feature groups
'''
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================

class SplitAttention(nn.Module):
    def __init__(self, channels, radix=2, groups=1):
        super().__init__()
        self.radix = radix
        self.groups = groups
        inter_channels = max(channels * radix // 4, 32)
        
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=groups)
        self.rsoftmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        batch = x.size(0)
        x = x.view(batch, self.radix, -1, *x.shape[2:])
        gap = x.sum(dim=1)
        gap = nn.functional.adaptive_avg_pool2d(gap, 1)
        
        atten = self.fc1(gap)
        atten = self.bn1(atten)
        atten = self.relu(atten)
        atten = self.fc2(atten)
        atten = atten.view(batch, self.radix, -1)
        atten = self.rsoftmax(atten)
        atten = atten.view(batch, self.radix, -1, 1, 1)
        
        out = (x * atten).sum(dim=1)
        return out

class ResNeStBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, radix=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * radix, 3, stride, 1, groups=radix, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * radix)
        self.split_attention = SplitAttention(out_channels, radix)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.split_attention(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNeSt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            stride = 2 if i == 0 and in_channels != 64 else 1
            layers.append(ResNeStBlock(in_channels if i == 0 else out_channels * 4, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = ResNeSt()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")```

## Discussion

The implementation defines 3 classes (`SplitAttention`, `ResNeStBlock`, `ResNeSt`) that work together to form the complete convolutional neural network architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Trace the tensor shapes through the `SplitAttention` forward pass. For a batch of 4 input samples with the default parameters, write down the shape after each major operation (convolution, pooling, linear layer).

??? success "Solution to Exercise 1"
    Start from the input shape and apply each layer sequentially. For each `Conv2d(in_c, out_c, k)`, the spatial dimensions change as $H_{\text{out}} = H_{\text{in}} - k + 1$ (without padding) or remain the same (with `padding=k//2`). Pooling with kernel 2 halves spatial dimensions. Linear layers transform the last dimension. Track the batch dimension throughout: it remains unchanged. Write each intermediate shape as $(B, C, H, W)$ for convolutional layers and $(B, F)$ after flattening.

---

**Exercise 2.**
Modify the architecture to accept RGB images of size $64 \times 64$ (input shape: $3 \times 64 \times 64$). Update all layer dimensions accordingly and verify the model runs without errors.

??? success "Solution to Exercise 2"
    Change the first convolutional layer's `in_channels` from the current value to 3. Recalculate the spatial dimensions after each convolution and pooling layer using the formulas $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$. Update the first linear layer's `in_features` to match the flattened output of the last convolutional/pooling layer. Verify with: `model = SplitAttention(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**Exercise 3.**
Explain the computational complexity of self-attention as a function of sequence length $n$ and model dimension $d$. Why does this motivate architectures like Longformer or Linformer for long sequences?

??? success "Solution to Exercise 3"
    Standard self-attention computes an $n \times n$ attention matrix, giving $O(n^2 d)$ time complexity and $O(n^2)$ memory for the attention weights. For long sequences (e.g., $n = 4096$), this becomes prohibitive. Longformer uses a combination of local sliding-window attention ($O(n \cdot w \cdot d)$ where $w$ is window size) and sparse global attention for selected tokens. Linformer projects keys and values to a lower dimension $k \ll n$, reducing complexity to $O(n \cdot k \cdot d)$. Both trade some expressiveness for practical efficiency on long inputs.

---

**Exercise 4.**
Extend `SplitAttention` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = SplitAttention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
