# ResNet

ResNet (Deep Residual Learning) was introduced in the 2015 paper "Deep Residual Learning for Image Recognition" by He et al. The core idea is the residual connection, which allows the network to learn identity mappings by default, solving the vanishing gradient problem and enabling training of networks with hundreds of layers. ResNet-50, implemented here, uses bottleneck blocks with $1 \times 1$, $3 \times 3$, and $1 \times 1$ convolutions.

## 코드

```python
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        layers = [Bottleneck(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels * 4, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


if __name__ == "__main__":
    model = ResNet50()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

The residual connection is the defining feature of ResNet. Each bottleneck block computes a residual function $F(\mathbf{x})$ and adds it to the input: $\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$. If the optimal transformation is close to the identity, the network only needs to learn a small residual, which is easier than learning the full mapping from scratch. This insight enabled training networks with 50, 101, and even 152 layers, far deeper than anything before.

The bottleneck design uses a $1 \times 1$ convolution to reduce the channel dimension, a $3 \times 3$ convolution for spatial processing, and another $1 \times 1$ convolution to restore the channel dimension (expanded by a factor of 4). This keeps the computational cost manageable while maintaining a large number of output channels. When spatial dimensions change between stages, a learned $1 \times 1$ projection with matching stride aligns the shortcut path.

ResNet이 깊은 배움에 미친 영향은 아무리 말해도 지나치지 않다. 건너뛰는 이음이라는 원리는 DenseNet에서 변환기까지 요즘 얼개 거의 모두에 나타난다. ResNet-50은 갈래 매기기, 알아내기, 나누기를 아우르는 여러 일에서 옮겨 배우기의 표준 등뼈로 남아 있다.

## 연습문제

**연습문제 1.**
Calculate the total number of multiply-add operations (FLOPs) for a single Bottleneck block with `in_channels=256`, `out_channels=64`, and spatial dimensions $56 \times 56$.

??? success "연습문제 1 풀이"
    For each convolution, FLOPs $\approx 2 \times C_{\text{in}} \times C_{\text{out}} \times K^2 \times H \times W$:

    - Conv1 ($1 \times 1$, $256 \to 64$): $2 \times 256 \times 64 \times 1 \times 56 \times 56 = 102{,}760{,}448$
    - Conv2 ($3 \times 3$, $64 \to 64$): $2 \times 64 \times 64 \times 9 \times 56 \times 56 = 231{,}211{,}008$
    - Conv3 ($1 \times 1$, $64 \to 256$): $2 \times 64 \times 256 \times 1 \times 56 \times 56 = 102{,}760{,}448$

    Total: approximately $436{,}731{,}904$ FLOPs ($\approx 437$ MFLOPs) per block.

---

**연습문제 2.**
`downsample` 길이 왜 단계마다 첫 덩이에만 필요한지 설명하여라. 그것을 없애면 어떻게 되는가?

??? success "연습문제 2 풀이"
    The downsample path (a $1 \times 1$ convolution with stride 2) is needed when the spatial dimensions or channel count change between the input and output of a block. This only happens at the first block of each stage, where the stride changes from 1 to 2 and the output channels increase. Subsequent blocks within the same stage have matching input and output dimensions, so the identity shortcut works directly.

    If the downsample path is removed at stage boundaries, the addition $\text{out} + \text{identity}$ would fail because the tensors have different shapes (different spatial resolution and channel count), resulting in a runtime error.

---

**연습문제 3.**
Implement a ResNet-18 variant using basic blocks (two $3 \times 3$ convolutions without bottleneck) with layer configuration $[2, 2, 2, 2]$. Compare the parameter count to ResNet-50.

??? success "연습문제 3 풀이"
    ```python
    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample:
                identity = self.downsample(x)
            return self.relu(out + identity)
    ```

    기본 덩이가 $[2, 2, 2, 2]$인 ResNet-18은 매개변수가 약 1170만 개이고, 병목 덩이를 쓰는 ResNet-50은 약 2560만 개로 대략 2.2배 많다.
