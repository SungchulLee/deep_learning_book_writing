# HRNet

HRNet (High-Resolution Network), introduced in the 2019 paper "Deep High-Resolution Representation Learning for Visual Recognition," maintains high-resolution representations throughout the entire network. Unlike conventional architectures that progressively downsample feature maps and then recover resolution through upsampling, HRNet connects multi-resolution streams in parallel and repeatedly exchanges information between them. This design produces richer and more spatially precise features, making it particularly effective for tasks like pose estimation and semantic segmentation.

## 코드

```python
#!/usr/bin/env python3
'''
HRNet - High-Resolution Network
Paper: "Deep High-Resolution Representation Learning" (2019)
Key: Maintains high-resolution representations through the network
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class HRModule(nn.Module):
    def __init__(self, num_branches, channels):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([
            nn.Sequential(*[BasicBlock(channels[i], channels[i]) for _ in range(4)])
            for i in range(num_branches)
        ])
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(nn.Identity())
                elif j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm2d(channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(channels[j], channels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(channels[i])
                            ))
                        else:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(channels[j], channels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(channels[j]),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            self.fuse_layers.append(fuse_layer)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

class HRNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.stage = HRModule(2, [32, 64])
        self.incre_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 128, 3, 1, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        ])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = [x, torch.nn.functional.avg_pool2d(x, 2)]
        x = self.stage(x)
        x = [incre(xi) for incre, xi in zip(self.incre_modules, [x[0]])]
        x = self.avgpool(x[0]).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = HRNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

The defining characteristic of HRNet is its parallel multi-resolution architecture. Instead of the typical encoder-decoder or feature pyramid approach where resolution is first reduced and then recovered, HRNet maintains a high-resolution stream throughout and progressively adds lower-resolution parallel streams. The multi-scale fusion modules repeatedly exchange information between all resolution levels, allowing the high-resolution stream to benefit from the semantic richness of low-resolution features while preserving fine spatial detail.

The fusion mechanism is carefully designed. When fusing from low to high resolution, a $1 \times 1$ convolution adjusts channels followed by bilinear or nearest-neighbor upsampling. When fusing from high to low resolution, strided $3 \times 3$ convolutions progressively reduce spatial dimensions. This asymmetry reflects the different information needs: upsampling requires channel adjustment, while downsampling requires spatial aggregation.

HRNet's design is particularly powerful for dense prediction tasks. For human pose estimation, the spatially precise high-resolution representations directly provide accurate keypoint heatmaps. For semantic segmentation, the rich multi-scale features enable fine-grained boundary delineation. The architecture consistently outperforms methods that rely on post-hoc resolution recovery.

## 익힘 문제

**익힘 1.**
In an `HRModule` with 3 branches at resolutions $H \times W$, $H/2 \times W/2$, and $H/4 \times W/4$, how many fusion paths exist? List the upsampling and downsampling operations needed.

??? success "익힘 1 풀이"
    With 3 branches, there are $3 \times 3 = 9$ fusion paths (each branch receives from all 3 branches including itself). Identity paths: 3 (branch $i$ to itself). Upsampling paths: 3 (branch 1 to 0: $2\times$ up; branch 2 to 0: $4\times$ up; branch 2 to 1: $2\times$ up). Downsampling paths: 3 (branch 0 to 1: one stride-2 conv; branch 0 to 2: two stride-2 convs; branch 1 to 2: one stride-2 conv). Each upsampling path uses a $1 \times 1$ convolution for channel adjustment followed by upsampling. Each downsampling path uses a sequence of stride-2 $3 \times 3$ convolutions.

---

**익힘 2.**
Compare HRNet's approach to maintaining spatial resolution with U-Net's encoder-decoder approach with skip connections. What are the advantages and disadvantages of each?

??? success "익힘 2 풀이"
    HRNet maintains high-resolution features throughout, so spatial information is never completely lost. Multi-scale fusion happens repeatedly, allowing gradual refinement. However, this is computationally expensive since the high-resolution stream processes features at full resolution for the entire network depth. U-Net progressively downsamples, reducing computation in deeper layers, and uses skip connections to recover spatial detail during upsampling. This is more computationally efficient but relies on skip connections to bridge the information gap across the bottleneck. U-Net's skip connections are simple concatenations at matched resolutions, while HRNet's fusion is more thorough, combining all resolution levels at every stage. HRNet generally produces more accurate spatial predictions but at higher computational cost.

---

**익힘 3.**
Design a lightweight version of `HRModule` that uses depthwise separable convolutions in the branches and $1 \times 1$ convolutions for all fusion operations (replacing strided $3 \times 3$ convolutions with pooling + $1 \times 1$ convolution).

??? success "익힘 3 풀이"
    ```python
    class LightBasicBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.pw = nn.Conv2d(channels, channels, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x):
            out = self.relu(self.bn1(self.dw(x)))
            out = self.bn2(self.pw(out))
            return self.relu(out + x)

    class LightHRModule(nn.Module):
        def __init__(self, num_branches, channels):
            super().__init__()
            self.num_branches = num_branches
            self.branches = nn.ModuleList([
                nn.Sequential(*[LightBasicBlock(channels[i]) for _ in range(2)])
                for i in range(num_branches)
            ])
            # All fusions use pool + 1x1 conv (down) or 1x1 conv + upsample (up)
            # Implementation follows the same fuse_layers pattern but replaces
            # strided 3x3 convolutions with avg_pool2d + 1x1 convolution.
    ```
