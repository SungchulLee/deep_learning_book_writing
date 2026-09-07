# EfficientNet

EfficientNet은 겹친 잣수 맞추기와 MBConv 덩이로 매개변수를 아주 적게 쓰면서 높은 정확도를 낸다. 여기 짠 것은 EfficientNet-B0 얼개를 간추린 판으로, 핵심 벽돌인 줄기, MBConv 단계, 갈래 매기기 머리를 보여 준다.

## 코드

```python
import torch
import torch.nn as nn


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
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                     groups=hidden_dim, bias=False),
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
        return self.head(self.blocks(self.stem(x)))


if __name__ == "__main__":
    model = EfficientNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

MBConv(모바일 뒤집힌 병목 합성곱) 블록이 EfficientNet의 고갱이 벽돌이다. 뒤집힌 병목 구조를 쓰는데, 채널을 먼저 넓히고 깊이별 합성곱으로 다룬 뒤 더 작은 차원으로 되비춘다. $f(x) = x \cdot \sigma(x)$으로 매긴 SiLU(Swish) 활성 함수는 ReLU보다 매끄러운 기울기를 준다.

이 간추린 짜기는 EfficientNet의 뼈대를 보여 준다. 곧 처음 특징을 뽑는 누비기 줄기, 채널이 늘고 자리 해상도가 줄어드는 MBConv 덩이의 차례, 그리고 전체 평균 모으기 앞에서 1280채널로 부풀리는 갈래 매기기 머리이다.

## 연습문제

**연습문제 1.**
SiLU/Swish 활성 $f(x) = x \cdot \sigma(x)$을 ReLU와 견주어라. $x \in [-5, 5]$에서 두 함수와 그 도함수를 그려라.

??? success "연습문제 1 풀이"
    SiLU은 매끄럽고 단조롭지 않으며 $x \approx -1.28$ 언저리에 작은 음수 구간이 있다. ReLU과 달리 음수 들임에서도 기울기가 0이 아니어서 죽은 뉴런을 막는 데 도움이 된다. 도함수는 $f'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x)(1 + x(1 - \sigma(x)))$이다. $x = 0$에서 SiLU은 0을 내놓고 기울기가 0.5이지만 ReLU은 0을 내놓고 기울기가 정해지지 않는다.

---

**연습문제 2.**
첫 MBConv 덩이의 부풀림 비는 1이고 뒤따르는 덩이는 6인 까닭을 설명하여라.

??? success "연습문제 2 풀이"
    첫 덩이는 채널이 적은 곳(줄기에서 온 32개)에서 돌아가므로 부풀릴 까닭이 없고 셈만 버리게 된다. 뒤따르는 덩이는 병목 꾸밈 탓에 들임 채널이 좁으므로, 깊이별 누비기가 뜻있는 자리 특징을 배울 만한 힘을 주려면 부풀려야 한다.

---

**연습문제 3.**
MBConv 덩이에 쥐어짜기-북돋우기를 더하고 늘어난 매개변수를 재어라.

??? success "연습문제 3 풀이"
    ```python
    class SE(nn.Module):
        def __init__(self, channels, reduction=4):
            super().__init__()
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction, 1),
                nn.SiLU(),
                nn.Conv2d(channels // reduction, channels, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return x * self.se(x)
    ```

    숨은 차원이 192이면(채널 32개에 넓힘 비율 6) SE은 매개변수 $192 \times 48 + 48 \times 192 = 18{,}432$개를 더하니 블록마다 대략 1~2% 늘어난다.
