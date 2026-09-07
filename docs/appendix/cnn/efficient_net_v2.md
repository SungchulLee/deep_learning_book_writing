# EfficientNet V2

EfficientNetV2, presented in the 2021 paper "EfficientNetV2: Smaller Models and Faster Training," improves upon the original EfficientNet with Fused-MBConv layers and a progressive learning strategy. These changes significantly reduce training time while maintaining or improving accuracy. The architecture was discovered through a combination of neural architecture search and manual refinement, optimizing for both training speed and parameter efficiency.

## 코드

```python
#!/usr/bin/env python3
'''
EfficientNetV2 - Improved Efficiency and Speed
Paper: "EfficientNetV2: Smaller Models and Faster Training" (2021)
Key: Fused-MBConv layers, progressive learning, improved training speed
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.conv(x)

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 24, 3, 2, 1, bias=False)
        self.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

if __name__ == "__main__":
    model = EfficientNetV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

The central architectural innovation in EfficientNetV2 is the Fused-MBConv block, which replaces the depthwise separable convolution in early stages with a regular $3 \times 3$ convolution. While depthwise separable convolutions are parameter-efficient, they underutilize modern accelerator hardware (GPUs/TPUs) due to low arithmetic intensity. The fused variant uses a standard $3 \times 3$ convolution to expand channels, followed by a $1 \times 1$ projection, achieving better hardware utilization at the cost of slightly more parameters. EfficientNetV2 uses Fused-MBConv in early stages (where feature maps are large) and standard MBConv in later stages.

The progressive learning strategy is another key contribution. During training, the image resolution and regularization strength (dropout, data augmentation) are gradually increased. Early epochs use smaller images with weaker augmentation, allowing the model to learn coarse features quickly, while later epochs use full-resolution images with strong regularization. This adaptive approach can reduce training time by up to 11x compared to fixed-resolution training.

EfficientNetV2 also uses SiLU (Swish) activation throughout, which has been shown to outperform ReLU in many settings. The SiLU function $f(x) = x \cdot \sigma(x)$ is smooth and non-monotonic, providing better gradient flow during training.

## 익힘 문제

**익힘 1.**
Compare the computational cost (in FLOPs) of a standard MBConv block versus a FusedMBConv block, given input shape $(1, 32, 56, 56)$ with expand ratio 4 and output channels 32.

??? success "익힘 1 풀이"
    For MBConv: (1) $1 \times 1$ expansion: $32 \times 128 \times 56 \times 56 \approx 12.8M$ FLOPs. (2) $3 \times 3$ depthwise: $128 \times 9 \times 56 \times 56 \approx 3.6M$ FLOPs. (3) $1 \times 1$ projection: $128 \times 32 \times 56 \times 56 \approx 12.8M$ FLOPs. Total: $\approx 29.2M$ FLOPs. For FusedMBConv: (1) $3 \times 3$ regular conv: $32 \times 128 \times 9 \times 56 \times 56 \approx 115.6M$ FLOPs. (2) $1 \times 1$ projection: $128 \times 32 \times 56 \times 56 \approx 12.8M$ FLOPs. Total: $\approx 128.4M$ FLOPs. FusedMBConv uses about 4.4x more FLOPs but achieves higher hardware utilization, often resulting in faster wall-clock time on GPUs.

---

**익힘 2.**
Explain why progressive learning (increasing resolution during training) is more beneficial for EfficientNet-style models than for fixed-resolution models like ResNet.

??? success "익힘 2 풀이"
    EfficientNet models use compound scaling that jointly adjusts depth, width, and resolution. Their architecture is specifically designed to work across different resolutions. Progressive learning exploits this by starting at low resolution (fewer pixels to process, faster iterations) and gradually increasing to full resolution. The features learned at lower resolution transfer naturally because the network structure is resolution-agnostic. ResNets have fixed architectural assumptions and their performance is more sensitive to resolution changes. Additionally, the regularization scheduling in progressive learning (weaker augmentation at low resolution, stronger at high resolution) prevents the model from overfitting to augmented small images, a problem less relevant for ResNets that train at fixed resolution.

---

**익힘 3.**
Implement a complete `FusedMBConv` block with squeeze-and-excitation (SE) attention and a skip connection.

??? success "익힘 3 풀이"
    ```python
    class FusedMBConvSE(nn.Module):
        def __init__(self, in_ch, out_ch, expand_ratio=4, se_ratio=0.25):
            super().__init__()
            hidden = in_ch * expand_ratio
            self.use_skip = (in_ch == out_ch)
            self.expand = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 3, 1, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            )
            se_ch = max(1, int(in_ch * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden, se_ch, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_ch, hidden, 1),
                nn.Sigmoid(),
            )
            self.project = nn.Sequential(
                nn.Conv2d(hidden, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        def forward(self, x):
            out = self.expand(x)
            out = out * self.se(out)
            out = self.project(out)
            if self.use_skip:
                out = out + x
            return out
    ```
