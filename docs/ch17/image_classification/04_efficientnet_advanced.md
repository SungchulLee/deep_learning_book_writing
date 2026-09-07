# 앞선 EfficientNet

EfficientNet은 겹친 잣수 맞추기를 들여왔는데, 붙박이 잣수 계수 한 벌로 그물의 깊이, 너비, 해상도를 고르게 키운다. 2019년 Tan과 Le가 내놓은 이 방식은 신경 얼개 찾기로 가장 좋은 바탕(EfficientNet-B0)을 찾은 뒤 그것을 짜임새 있게 키운다. 이로써 사람이 손수 꾸민 얼개보다 나은 정확도와 효율을 얻는다.

## 코드

```python
import torch
import torch.nn as nn
import math


class SwishActivation(nn.Module):
    """스위시 깨어남: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """쥐어짜기-북돋우기 덩이."""
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """손전화용 뒤집은 병목 누비기(MBConv) 덩이."""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                     kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(hidden_dim, se_channels)
        
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x


class EfficientNet(nn.Module):
    """겹친 잣수 맞추기를 쓴 EfficientNet 얼개."""
    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=10,
                 dropout=0.2):
        super().__init__()
        
        base_config = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        
        out_channels = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        in_channels = out_channels
        blocks = []
        for expand, channels, num_layers, stride, kernel in base_config:
            out_channels = self._round_filters(channels, width_mult)
            num_layers = self._round_repeats(num_layers, depth_mult)
            for i in range(num_layers):
                blocks.append(MBConvBlock(
                    in_channels, out_channels, kernel,
                    stride if i == 0 else 1, expand
                ))
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        
        final_channels = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(final_channels, num_classes)
        )
    
    def _round_filters(self, filters, width_mult):
        filters *= width_mult
        new_filters = int(filters + 4) // 8 * 8
        new_filters = max(8, new_filters)
        if new_filters < 0.9 * filters:
            new_filters += 8
        return int(new_filters)
    
    def _round_repeats(self, repeats, depth_mult):
        return int(math.ceil(depth_mult * repeats))
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def efficientnet_b0(num_classes=10):
    return EfficientNet(width_mult=1.0, depth_mult=1.0, num_classes=num_classes)


if __name__ == "__main__":
    pass
```

## 논의

복합 스케일링이 EfficientNet의 고갱이 이바지다. 여느 방법은 그물을 한 차원으로만 키우지만(더 깊게, 더 넓게, 또는 더 높은 해상도로) EfficientNet은 계수 $\alpha$, $\beta$, $\gamma$에 공통 지수 $\phi$을 올려 셋을 한꺼번에 키운다. 깊이 $= \alpha^\phi$, 너비 $= \beta^\phi$, 해상도 $= \gamma^\phi$이다. 격자 탐색으로 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$이라는 제약 아래 가장 좋은 $\alpha$, $\beta$, $\gamma$을 찾고, 그다음 $\phi$이 모델 전체 크기를 다스린다.

MBConv 덩이는 깊이별로 갈라지는 누비기와 쥐어짜기-북돋우기 눈길을 아우른다. 부풀리기 단계가 채널을 늘리고, 깊이별 누비기가 매개변수를 거의 안 쓰고 자리 앎을 다루며, SE 단원이 채널의 중요함을 다시 매기고, 내리쬐기 단계가 내놓는 차원으로 다시 눌러 담는다. 들임과 내놓음의 꼴이 맞을 때 건너뛰는 이음을 쓴다.

EfficientNet은 원칙 있는 잣수 맞추기가 임시변통 꾸밈보다 정확도와 효율의 맞바꿈에서 더 낫다는 것을 보여 준다. EfficientNet-B0은 매개변수를 약 5분의 1만 쓰고도 ResNet-50의 정확도에 맞먹으며, B7까지 키우면 ImageNet에서 가장 앞선 성능을 낸다.

## 연습문제

**연습문제 1.**
$\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$, $\phi = 2$인 복합 스케일링 규칙이 주어졌을 때 EfficientNet-B2 꼴 모델의 깊이 곱값, 너비 곱값, 들임 해상도 곱값을 셈하여라.

??? success "연습문제 1 풀이"

    - 깊이 곱값: $\alpha^\phi = 1.2^2 = 1.44$
    - 너비 곱값: $\beta^\phi = 1.1^2 = 1.21$
    - 해상도 곱값: $\gamma^\phi = 1.15^2 = 1.3225$

    밑 해상도가 224이면 키운 해상도는 $224 \times 1.3225 \approx 296$이고 8의 배수로 반올림하면 $296$이다.

---

**연습문제 2.**
들임 채널 128개, 날임 채널 128개, $3 \times 3$ 커널인 켜에서 깊이별 분리 합성곱과 여느 합성곱의 매개변수 줄임을 견주어라.

??? success "연습문제 2 풀이"

    - 여느 합성곱: 매개변수 $128 \times 128 \times 3 \times 3 = 147{,}456$개
    - 깊이별 분리: 깊이별($128 \times 3 \times 3 = 1{,}152$) + 점별($128 \times 128 \times 1 = 16{,}384$) = 매개변수 $17{,}536$개
    - 줄임 배수: $147{,}456 / 17{,}536 \approx 8.4\times$

    이는 대략 $K^2 + 1/C_{\text{out}}$만큼 아끼는 것이며 $K=3$이면 이론상 약 $9\times$ 줄어든다.

---

**연습문제 3.**
알맞은 너비 곱셈수와 깊이 곱셈수를 정해 EfficientNet-B3부터 B7까지 만드는 함수를 짜고, 변종마다 매개변수 개수를 찍어라.

??? success "연습문제 3 풀이"
    ```python
    configs = {
        'B3': (1.2, 1.4),
        'B4': (1.4, 1.8),
        'B5': (1.6, 2.2),
        'B6': (1.8, 2.6),
        'B7': (2.0, 3.1),
    }

    for name, (width, depth) in configs.items():
        model = EfficientNet(width_mult=width, depth_mult=depth, num_classes=1000)
        params = sum(p.numel() for p in model.parameters())
        print(f"EfficientNet-{name}: {params:,} parameters")
    ```

    어림한 매개변수 개수: B3 약 1200만, B4 약 1900만, B5 약 3000만, B6 약 4300만, B7 약 6600만.
