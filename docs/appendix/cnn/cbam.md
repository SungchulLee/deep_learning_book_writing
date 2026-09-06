# CBAM

The Convolutional Block Attention Module (CBAM), introduced in the 2018 paper of the same name, applies sequential channel and spatial attention to refine feature maps. By learning "what" to attend to (channel attention) and "where" to attend (spatial attention), CBAM provides a lightweight yet effective mechanism that can be plugged into any CNN backbone with minimal overhead.

## Code

```python
#!/usr/bin/env python3
'''
CBAM - Convolutional Block Attention Module
Paper: "CBAM: Convolutional Block Attention Module" (2018)
Key: Sequential channel and spatial attention mechanisms
'''
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class CBAM_ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.cbam = CBAMBlock(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.cbam(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = CBAM_ResNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Discussion

CBAM consists of two sequential sub-modules. The channel attention module exploits inter-channel relationships by aggregating spatial information using both average pooling and max pooling, passing each through a shared multi-layer perceptron, and combining the results. This dual-pooling strategy captures both the average response (what features are generally present) and the most salient response (what features are strongest), producing a richer channel descriptor than using either alone.

The spatial attention module complements channel attention by identifying informative regions within each feature map. It compresses channel information via average and max pooling along the channel axis, concatenates the two descriptors, and applies a convolution to produce a spatial attention map. The sequential application of channel-then-spatial attention is key: channel attention first selects "what" features matter, then spatial attention determines "where" those features are most relevant.

The design is deliberately lightweight. The reduction ratio in channel attention (defaulting to 16) keeps the bottleneck MLP small, and the spatial attention uses a single convolution. This makes CBAM easy to integrate into existing architectures like ResNet, with typically less than 1% parameter overhead while providing consistent accuracy improvements across classification, detection, and segmentation tasks.

## Exercises

**Exercise 1.**
Given an input feature map of shape $(B, 64, H, W)$ with reduction ratio 16, compute the number of parameters in the `ChannelAttention` module.

??? success "Solution to Exercise 1"
    The shared MLP consists of two $1 \times 1$ convolutions (without bias): the first is $64 \to 64/16 = 4$ with $64 \times 4 = 256$ parameters, and the second is $4 \to 64$ with $4 \times 64 = 256$ parameters. The total is $256 + 256 = 512$ parameters. Note that the pooling layers have no learnable parameters, and the sigmoid is parameter-free.

---

**Exercise 2.**
Why does CBAM apply channel attention before spatial attention rather than the reverse order or in parallel? Discuss the intuitive reasoning.

??? success "Solution to Exercise 2"
    Channel attention first determines which feature channels are important, effectively answering "what" to focus on. Once the features are re-weighted by channel importance, spatial attention can more effectively determine "where" those important features are located. If spatial attention were applied first, it would operate on all channels equally without knowing which channels carry the most relevant information. The sequential ordering creates an information cascade: channel selection refines the features, giving spatial attention a cleaner signal to localize. The original paper empirically confirms that channel-first ordering consistently outperforms spatial-first and parallel configurations.

---

**Exercise 3.**
Extend the `SpatialAttention` module to accept a configurable number of pooling operations (e.g., adding $L^2$-norm pooling in addition to average and max pooling). Write the modified class.

??? success "Solution to Exercise 3"
    ```python
    class SpatialAttentionExtended(nn.Module):
        def __init__(self, kernel_size=7, use_l2=True):
            super().__init__()
            in_channels = 3 if use_l2 else 2
            self.use_l2 = use_l2
            self.conv = nn.Conv2d(in_channels, 1, kernel_size,
                                  padding=kernel_size // 2, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            descriptors = [avg_out, max_out]
            if self.use_l2:
                l2_out = torch.norm(x, p=2, dim=1, keepdim=True)
                descriptors.append(l2_out)
            combined = torch.cat(descriptors, dim=1)
            return self.sigmoid(self.conv(combined))
    ```
    The $L^2$-norm pooling provides a measure of the overall energy at each spatial location, complementing the average (mean signal) and max (peak signal) descriptors.
