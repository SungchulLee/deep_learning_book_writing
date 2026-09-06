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

Compound scaling is EfficientNet's key contribution. Traditional approaches scale networks along a single dimension (deeper, wider, or higher resolution), but EfficientNet scales all three simultaneously using coefficients $\alpha$, $\beta$, and $\gamma$ raised to a shared exponent $\phi$: depth $= \alpha^\phi$, width $= \beta^\phi$, resolution $= \gamma^\phi$. A grid search finds optimal $\alpha$, $\beta$, $\gamma$ under a constraint $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$, and then $\phi$ controls overall model size.

MBConv 덩이는 깊이별로 갈라지는 누비기와 쥐어짜기-북돋우기 눈길을 아우른다. 부풀리기 단계가 채널을 늘리고, 깊이별 누비기가 매개변수를 거의 안 쓰고 자리 앎을 다루며, SE 단원이 채널의 중요함을 다시 매기고, 내리쬐기 단계가 내놓는 차원으로 다시 눌러 담는다. 들임과 내놓음의 꼴이 맞을 때 건너뛰는 이음을 쓴다.

EfficientNet은 원칙 있는 잣수 맞추기가 임시변통 꾸밈보다 정확도와 효율의 맞바꿈에서 더 낫다는 것을 보여 준다. EfficientNet-B0은 매개변수를 약 5분의 1만 쓰고도 ResNet-50의 정확도에 맞먹으며, B7까지 키우면 ImageNet에서 가장 앞선 성능을 낸다.

## 연습문제

**연습문제 1.**
Given the compound scaling rule with $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$, and $\phi = 2$, compute the depth multiplier, width multiplier, and input resolution multiplier for an EfficientNet-B2-like model.

??? success "연습문제 1 풀이"

    - Depth multiplier: $\alpha^\phi = 1.2^2 = 1.44$
    - Width multiplier: $\beta^\phi = 1.1^2 = 1.21$
    - Resolution multiplier: $\gamma^\phi = 1.15^2 = 1.3225$

    If the base resolution is 224, the scaled resolution would be $224 \times 1.3225 \approx 296$, rounded to the nearest multiple of 8: $296$.

---

**연습문제 2.**
Compare the parameter reduction of depthwise separable convolutions versus standard convolutions for a layer with 128 input channels, 128 output channels, and a $3 \times 3$ kernel.

??? success "연습문제 2 풀이"

    - Standard convolution: $128 \times 128 \times 3 \times 3 = 147{,}456$ parameters
    - Depthwise separable: depthwise ($128 \times 3 \times 3 = 1{,}152$) + pointwise ($128 \times 128 \times 1 = 16{,}384$) = $17{,}536$ parameters
    - Reduction factor: $147{,}456 / 17{,}536 \approx 8.4\times$

    This is approximately $K^2 + 1/C_{\text{out}}$ savings, which for $K=3$ gives about $9\times$ theoretical reduction.

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
