# EfficientNet

EfficientNet, introduced in the 2019 paper "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," proposes a principled method for scaling CNNs along three dimensions simultaneously: depth, width, and resolution. By using a compound scaling coefficient, the architecture achieves state-of-the-art accuracy with significantly fewer parameters and FLOPs than previous models. The base architecture, EfficientNet-B0, was discovered through neural architecture search.

## 코드

```python
#!/usr/bin/env python3
'''
EfficientNet - Rethinking Model Scaling
Paper: "EfficientNet: Rethinking Model Scaling for CNNs" (2019)
Key: Compound scaling (depth, width, resolution), highly efficient
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            MBConv(32, 16, expand_ratio=1),
            MBConv(16, 24, stride=2),
            MBConv(24, 40, stride=2),
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(40, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

if __name__ == "__main__":
    model = EfficientNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

The MBConv (Mobile Inverted Bottleneck Convolution) block is the fundamental building unit of EfficientNet. It follows an inverted residual structure: first expanding the channel dimension by a factor (typically 6x), applying a depthwise separable convolution at the expanded dimension, then projecting back to a smaller output dimension. The residual connection bypasses the entire block when input and output dimensions match, facilitating gradient flow. The SiLU (Swish) activation replaces ReLU, providing smoother gradients.

The compound scaling method is EfficientNet's key theoretical contribution. Rather than scaling only depth (more layers), width (more channels), or resolution (larger images) independently, it scales all three dimensions together using a compound coefficient $\phi$: depth $d = \alpha^\phi$, width $w = \beta^\phi$, resolution $r = \gamma^\phi$, subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$. This constraint ensures that FLOPs roughly double for each unit increase in $\phi$, and the relationship between the three scaling dimensions is maintained across model sizes from B0 to B7.

The practical impact of compound scaling is significant. EfficientNet-B7 achieves 84.3% top-1 accuracy on ImageNet with 66M parameters, compared to GPipe's 84.3% with 557M parameters. This demonstrates that balanced scaling across all dimensions is far more efficient than scaling any single dimension to extremes.

## 익힘 문제

**익힘 1.**
Given compound scaling parameters $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$ and $\phi = 2$, compute the scaled depth, width, and resolution multipliers.

??? success "익힘 1 풀이"
    Depth multiplier: $d = \alpha^\phi = 1.2^2 = 1.44$. Width multiplier: $w = \beta^\phi = 1.1^2 = 1.21$. Resolution multiplier: $r = \gamma^\phi = 1.15^2 = 1.3225$. Verification: $\alpha \cdot \beta^2 \cdot \gamma^2 = 1.2 \times 1.21 \times 1.3225 \approx 1.919 \approx 2$. So with $\phi = 2$, the model uses roughly $1.44 \times$ more layers, $1.21 \times$ wider channels, and $1.32 \times$ larger input resolution compared to the base model.

---

**익힘 2.**
Why does the MBConv block use a $1 \times 1$ convolution for the final projection instead of applying another depthwise convolution? Discuss the role of the linear bottleneck.

??? success "익힘 2 풀이"
    The $1 \times 1$ projection serves as a "linear bottleneck" that compresses the expanded representation back to a lower-dimensional space without applying a nonlinear activation. This is deliberate: the expanded space (6x channels) captures rich feature interactions through the depthwise convolution, and the projection linearly combines these features. Adding a nonlinearity here would destroy information in the low-dimensional bottleneck space, as shown by the MobileNetV2 paper. A depthwise convolution at the bottleneck dimension would only allow channel-independent spatial filtering at the already compressed dimension, missing the inter-channel mixing that the $1 \times 1$ convolution provides.

---

**익힘 3.**
Add a Squeeze-and-Excitation (SE) module to the `MBConv` block between the depthwise convolution and the final $1 \times 1$ projection.

??? success "익힘 3 풀이"
    ```python
    class MBConvSE(nn.Module):
        def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1, se_ratio=0.25):
            super().__init__()
            hidden_dim = in_channels * expand_ratio
            self.use_res = stride == 1 and in_channels == out_channels

            layers = []
            if expand_ratio != 1:
                layers.extend([
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True)
                ])
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True),
            ])
            self.pre_se = nn.Sequential(*layers)

            se_ch = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_ch, 1), nn.SiLU(inplace=True),
                nn.Conv2d(se_ch, hidden_dim, 1), nn.Sigmoid(),
            )
            self.project = nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            out = self.pre_se(x)
            out = out * self.se(out)
            out = self.project(out)
            if self.use_res:
                out = out + x
            return out
    ```
    The SE module is placed after the depthwise convolution and before the projection, operating at the expanded channel dimension.
