# DenseNet

Huang 외의 2017년 논문에서 나온 DenseNet(촘촘히 이은 누비기 그물)은 층마다 뒤따르는 모든 층에 앞먹임 꼴로 잇는다. 들임을 내놓음에 더하는 ResNet과 달리 DenseNet은 앞선 모든 층의 특징 지도를 이어 붙인다. 이 촘촘한 이음 무늬는 특징을 다시 쓰게 북돋우고 기울기 흐름을 튼튼히 하며 필요한 매개변수 수를 크게 줄인다.

## 코드

```python
import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)
    
    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, dim=1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        return self.pool(self.conv(torch.relu(self.bn(x))))


class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        num_channels = 64
        self.dense1 = DenseBlock(6, num_channels, growth_rate)
        num_channels += 6 * growth_rate
        self.trans1 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense2 = DenseBlock(12, num_channels, growth_rate)
        num_channels += 12 * growth_rate
        self.trans2 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense3 = DenseBlock(24, num_channels, growth_rate)
        num_channels += 24 * growth_rate
        self.trans3 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.dense4 = DenseBlock(16, num_channels, growth_rate)
        num_channels += 16 * growth_rate
        
        self.bn_final = nn.BatchNorm2d(num_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, num_classes)
    
    def forward(self, x):
        x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = torch.relu(self.bn_final(x))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


if __name__ == "__main__":
    model = DenseNet121()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

The dense connectivity pattern means that layer $\ell$ receives the feature maps of all preceding layers $x_0, x_1, \ldots, x_{\ell-1}$ as input: $x_\ell = H_\ell([x_0, x_1, \ldots, x_{\ell-1}])$, where $[\cdot]$ denotes concatenation. Each layer adds $k$ new feature maps (the growth rate), so after $L$ layers the block has $k_0 + L \times k$ channels. This is in contrast to ResNet, where each block replaces the feature representation.

Transition layers between dense blocks reduce dimensionality. They apply batch normalization, a $1 \times 1$ convolution to halve the channel count (compression factor $\theta = 0.5$), and $2 \times 2$ average pooling to halve the spatial resolution. Without these, the channel count would grow without bound.

DenseNet은 층마다 손실 함수의 기울기와 앞선 모든 층의 특징에 곧바로 닿을 수 있기 때문에, ResNet보다 훨씬 적은 매개변수로 견줄 만한 정확도를 낸다. 이 넌지시 이루어지는 깊은 이끎과 특징 다시 쓰기 덕분에 층마다 좁아도(자람 비율이 작아도) 되어 모델이 아담하게 유지된다. DenseNet-121은 매개변수가 약 800만 개로 ResNet-50의 2560만 개와 견주어 적다.

## 연습문제

**연습문제 1.**
층이 6개, 자람 비율이 $k = 32$, 처음 채널이 64인 촘촘 덩이에서 내놓는 채널 수와 촘촘 층의 매개변수 전체 개수를 셈하여라(치우침은 헤아리지 않는다).

??? success "연습문제 1 풀이"
    Output channels: $64 + 6 \times 32 = 256$.

    For each DenseLayer $i$ (0-indexed), input channels = $64 + i \times 32$:

    - Layer 0: bottleneck $64 \to 128$ ($1 \times 1$: $64 \times 128 = 8{,}192$), then $128 \to 32$ ($3 \times 3$: $128 \times 32 \times 9 = 36{,}864$). Total: $45{,}056$.
    - Layer 1: $96 \times 128 + 128 \times 32 \times 9 = 12{,}288 + 36{,}864 = 49{,}152$.
    - 2~5층: 들임 채널이 늘어나며 마찬가지이다.

    Total parameters: $\sum_{i=0}^{5} [(64 + 32i) \times 128 + 128 \times 32 \times 9] = \sum_{i=0}^{5} [(64 + 32i) \times 128 + 36{,}864]$.

---

**연습문제 2.**
Explain the purpose of the bottleneck ($1 \times 1$ convolution) in each DenseLayer. What would happen to memory usage without it?

??? success "연습문제 2 풀이"
    The bottleneck $1 \times 1$ convolution reduces the number of input channels to $4k$ (where $k$ is the growth rate) before the expensive $3 \times 3$ convolution. Without it, the $3 \times 3$ convolution would operate on an ever-growing number of input channels (since dense connectivity concatenates all prior features).

    For the last layer in a 24-layer block with $k=32$ and initial channels 256, the input would have $256 + 23 \times 32 = 992$ channels. A $3 \times 3$ conv from 992 to 32 channels would require $992 \times 32 \times 9 = 285{,}696$ parameters. With the bottleneck, it first reduces to 128 channels ($992 \times 128 = 126{,}976$) then applies $3 \times 3$ ($128 \times 32 \times 9 = 36{,}864$) for a total of $163{,}840$ -- a 43% parameter reduction.

---

**연습문제 3.**
Implement a DenseNet variant for CIFAR-10 with growth rate $k = 12$, block configuration $[6, 12, 24, 16]$, and compression $\theta = 0.5$. Report the parameter count.

??? success "연습문제 3 풀이"
    ```python
    model = DenseNet121(growth_rate=12, num_classes=10)
    # 32x32 들임에 맞게 첫 누비기를 고친다:
    # 7x7 성큼 2 누비기를 3x3 성큼 1로 갈음
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    params = sum(p.numel() for p in model.parameters())
    print(f"CIFAR DenseNet: {params:,} parameters")
    ```

    $k=12$이고 덩이가 $[6, 12, 24, 16]$인 DenseNet은 매개변수가 약 80만 개여서 CIFAR-10 같은 작은 자료 뭉치에 잘 맞는다.
