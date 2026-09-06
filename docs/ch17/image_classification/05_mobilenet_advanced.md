# 앞선 MobileNet

MobileNet은 깊이별로 갈라지는 누비기로 손전화와 내장 기기의 보기 쓰임새에 맞는 가벼운 깊은 신경망을 세운다. 2017년 Howard 외가 내놓았고, MobileNetV2는 뒤집은 잔차와 선형 병목을 들여와 처음 꾸밈보다 정확도와 효율을 모두 낫게 했다.

## 코드

```python
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """깊이별로 갈라지는 누비기 = 깊이별 + 점별."""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1,
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class InvertedResidual(nn.Module):
    """뒤집은 잔차 덩이(MobileNetV2)."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        config = [
            [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
            [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        in_channels = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        blocks = []
        for expand, channels, num_blocks, stride in config:
            out_channels = int(channels * width_mult)
            for i in range(num_blocks):
                blocks.append(InvertedResidual(
                    in_channels, out_channels,
                    stride if i == 0 else 1, expand
                ))
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        final_channels = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(final_channels, num_classes)
        )
    
    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


if __name__ == "__main__":
    pass
```

## 논의

The inverted residual block reverses the traditional bottleneck pattern. Standard residual blocks go wide-narrow-wide, while inverted residuals go narrow-wide-narrow. The expansion phase uses a $1 \times 1$ convolution to increase the channel count by a factor (typically 6), the depthwise convolution processes spatial information cheaply, and the projection phase compresses back. Critically, the projection uses no activation function (linear bottleneck), preserving information in the low-dimensional output.

Depthwise separable convolutions factorize a standard convolution into a depthwise convolution (one filter per input channel) and a pointwise $1 \times 1$ convolution (combining channels). For a $3 \times 3$ kernel, this reduces parameters by approximately $8{-}9\times$. The width multiplier further controls model size by uniformly scaling channel counts across all layers.

## 연습문제

**연습문제 1.**
Calculate the parameter reduction when using depthwise separable convolutions instead of standard convolutions for a $3 \times 3$ layer with 256 input and 256 output channels.

??? success "연습문제 1 풀이"

    - Standard: $256 \times 256 \times 9 = 589{,}824$ parameters
    - Depthwise separable: $256 \times 9 + 256 \times 256 = 2{,}304 + 65{,}536 = 67{,}840$ parameters
    - Reduction: $589{,}824 / 67{,}840 \approx 8.7\times$

---

**연습문제 2.**
뒤집은 잔차에서 선형 병목(내리쬐기 뒤에 ReLU를 안 두는 것)이 왜 중요한지 설명하여라.

??? success "연습문제 2 풀이"
    ReLU는 음수를 0으로 뭉개므로 낮은 차원의 공간에서 앎을 잃게 한다. 내리쬐기가 채널을 좁은 병목으로 줄이므로 여기에 ReLU를 쓰면 되찾을 수 없는 앎이 사라진다. 선형 깨어남은 낮은 차원 다양체의 나타내는 힘을 온전히 지키며, 지은이들은 이것이 더 나은 정확도로 이어짐을 보였다.

---

**연습문제 3.**
너비 곱셈수를 0.5, 0.75, 1.0으로 두고 MobileNetV2를 익힌 뒤 매개변수 개수와 정확도 맞바꿈을 견주어라.

??? success "연습문제 3 풀이"
    ```python
    for mult in [0.5, 0.75, 1.0]:
        model = MobileNetV2(num_classes=10, width_mult=mult)
        params = sum(p.numel() for p in model.parameters())
        print(f"width_mult={mult}: {params:,} parameters")
    ```

    어림한 개수: 0.5는 약 190만, 0.75는 약 260만, 1.0은 약 350만. 너비 곱셈수가 작으면 정확도를 내주고 빠르기와 크기를 얻어, 자원이 빠듯한 곳에 펼치기 알맞다.
