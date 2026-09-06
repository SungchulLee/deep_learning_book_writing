# 앞선 DenseNet

이 앞선 DenseNet 짜기는 촘촘한 이음을 더 자세히 보여 주며, 정할 수 있는 자람 비율, 눌러 담기 계수, 떨구기를 곁들인 병목 층, 여러 DenseNet 변종(DenseNet-121, DenseNet-169) 받침을 담고 있다.

## 코드

```python
import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """BN -> ReLU -> Conv(1x1) -> BN -> ReLU -> Conv(3x3), 떨구기는 있어도 되고 없어도 된다."""
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        new_features = self.conv1(self.relu1(self.bn1(x)))
        new_features = self.conv2(self.relu2(self.bn2(new_features)))
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(
                in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
                 compression=0.5, num_classes=10, drop_rate=0.0):
        super().__init__()
        num_init_features = 2 * growth_rate
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate,
                             bn_size=4, drop_rate=drop_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, int(num_features * compression))
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)
        self.features.add_module('bn_final', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))


def densenet121(num_classes=10):
    return DenseNet(32, (6, 12, 24, 16), 0.5, num_classes)


if __name__ == "__main__":
    pass
```

## 논의

촘촘한 이음은 앎이 가장 잘 흐르게 한다. 곧 층마다 손실의 기울기와 앞선 모든 층의 특징 지도에 곧바로 닿는다. 이는 넌지시 이루어지는 깊은 이끎을 만들고 특징을 다시 쓰게 북돋운다. 자람 비율 $k$은 층마다 새 앎을 얼마나 보태는지를 다스린다. 그물이 앞선 층에 쌓인 특징을 끌어 쓸 수 있으므로 $k = 12$처럼 작은 값도 잘 된다.

The compression factor $\theta$ in transition layers controls how aggressively channel counts are reduced. Setting $\theta = 0.5$ halves the channels at each transition, preventing the total channel count from growing too large. The bottleneck design ($1 \times 1$ convolution producing $4k$ feature maps before the $3 \times 3$ convolution) further controls computational cost.

## 연습문제

**연습문제 1.**
층이 $L$개이고 자람 비율이 $k$인 촘촘 덩이에서 들임부터 내놓음까지의 이음(곧은 길) 전체 개수 식을 이끌어 내어라.

??? success "연습문제 1 풀이"
    In a DenseBlock with $L$ layers, layer $\ell$ receives input from all $\ell$ preceding layers plus the original input. The total number of connections is $\frac{L(L+1)}{2}$. For $L = 6$: $\frac{6 \times 7}{2} = 21$ connections. This quadratic growth in connections is what gives DenseNet its name and its strong gradient flow properties.

---

**연습문제 2.**
DenseNet-121과 ResNet-50을 익힐 때의 기억 공간 씀씀이를 견주고, 매개변수가 더 적은데도 DenseNet이 기억 공간을 더 쓸 수 있는 까닭을 설명하여라.

??? success "연습문제 2 풀이"
    DenseNet은 앞선 층의 특징 지도를 모두 이어 붙이므로 덩이 안에서 가운데 깨어남이 깊이에 선형으로 늘어난다. 층이 $L$개이고 자람 비율이 $k$인 덩이에서 마지막 층의 들임은 채널이 $k_0 + (L-1)k$개다. 이어 붙인 이 특징 지도를 뒤먹임을 위해 모두 기억 공간에 담아 두어야 한다. 반면 ResNet은 층마다 지금의 특징 지도(와 건너뛰는 이음의 들임)만 지닌다. 그래서 DenseNet은 무게 매개변수가 더 적어도 깨어남 기억 공간을 더 쓴다.

---

**연습문제 3.**
Implement an ablation study varying the growth rate $k \in \{8, 12, 24, 32\}$ and report the parameter counts for each configuration.

??? success "연습문제 3 풀이"
    ```python
    for k in [8, 12, 24, 32]:
        model = DenseNet(growth_rate=k, block_config=(6, 12, 24, 16),
                        compression=0.5, num_classes=10)
        params = sum(p.numel() for p in model.parameters())
        print(f"k={k}: {params:,} parameters")
    ```

    기대되는 결과: $k=8$은 약 30만, $k=12$은 약 80만, $k=24$은 약 280만, $k=32$은 약 490만. 매개변수 개수는 $k$의 대략 제곱으로 늘어난다.
