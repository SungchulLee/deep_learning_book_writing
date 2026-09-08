# 앞선 MobileNet

MobileNet은 깊이별로 갈라지는 누비기로 손전화와 내장 기기의 보기 쓰임새에 맞는 가벼운 깊은 신경망을 세운다. 2017년 Howard 외가 내놓았고, MobileNetV2는 뒤집은 잔차와 선형 병목을 들여와 처음 꾸밈보다 정확도와 효율을 모두 낫게 했다.

## 1. 코드

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

## 2. 논의

뒤집힌 잔차 블록은 여느 병목 무늬를 뒤집는다. 여느 잔차 블록은 넓게-좁게-넓게 가지만 뒤집힌 잔차는 좁게-넓게-좁게 간다. 넓히는 단계는 $1 \times 1$ 합성곱으로 채널 수를 몇 배(흔히 6배)로 늘리고, 깊이별 합성곱이 공간 정보를 값싸게 다루며, 투영 단계가 다시 압축한다. 종요롭게도 투영에는 활성 함수를 쓰지 않는데(선형 병목), 그래야 낮은 차원의 날임에서 정보가 지켜진다.

깊이별 분리 합성곱은 여느 합성곱을 깊이별 합성곱(들임 채널마다 거르개 하나)과 점별 $1 \times 1$ 합성곱(채널 섞기)으로 나눈다. $3 \times 3$ 커널이면 매개변수가 약 $8{-}9\times$ 줄어든다. 너비 곱값은 모든 켜의 채널 수를 고르게 키워 모델 크기를 더 다스린다.

## 연습문제

**연습문제 1.**
들임 채널 256개, 날임 채널 256개인 $3 \times 3$ 켜에서 여느 합성곱 대신 깊이별 분리 합성곱을 쓸 때 매개변수가 얼마나 줄어드는지 셈하여라.

??? success "연습문제 1 풀이"

    - 여느 합성곱: 매개변수 $256 \times 256 \times 9 = 589{,}824$개
    - 깊이별 분리: $256 \times 9 + 256 \times 256 = 2{,}304 + 65{,}536 = 67{,}840$개
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

## 정리하며

**다룬 것** — 앞선 MobileNet

뒤집힌 잔차 블록은 여느 병목 무늬를 뒤집는다.

고갱이 갈래는 `DepthwiseSeparableConv`, `InvertedResidual`, `MobileNetV2`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
