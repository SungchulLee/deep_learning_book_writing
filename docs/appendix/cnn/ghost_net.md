# Ghost Net

GhostNet은 2020년 글 "GhostNet: 값싼 셈으로 더 많은 결 얻기"에서 나왔으며, 익힌 CNN의 결 그림 가운데 많은 것이 서로 닮았고 몇 안 되는 본디 결 그림을 값싸게 곧게 바꾸어 만들 수 있음을 짚는다. 그림자 묶음은 이 군더더기를 써서, 먼저 여느 엮음으로 결 그림 몇 개를 만들고 다음으로 단순한 깊이별 선형 셈으로 덧붙은 "그림자" 결을 내어, 매개변수와 뜨는 셈 횟수를 크게 줄이면서도 비슷한 맞음을 이룬다.

## 코드

```python
#!/usr/bin/env python3
'''
GhostNet - 값싼 셈으로 특징 더 뽑기
논문: "GhostNet: More Features from Cheap Operations" (2020)
고갱이: 그림자 묶음이 더 적은 매개변수로 더 많은 특징을 만들어 낸다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3):
        super().__init__()
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, 1, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.ghost1 = GhostModule(in_channels, out_channels)
        
        if stride > 1:
            self.conv_dw = nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=out_channels, bias=False)
            self.bn_dw = nn.BatchNorm2d(out_channels)
        self.stride = stride
        
        self.ghost2 = GhostModule(out_channels, out_channels, ratio=1)
        
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.bn_dw(self.conv_dw(x))
        x = self.ghost2(x)
        return x + self.shortcut(residual)

class GhostNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)
        
        self.blocks = nn.Sequential(
            GhostBottleneck(16, 16),
            GhostBottleneck(16, 24, 2),
            GhostBottleneck(24, 24),
        )
        
        self.conv_head = nn.Conv2d(24, 960, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.act2 = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(960, num_classes)
    
    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x = self.blocks(x)
        x = self.act2(self.bn2(self.conv_head(x)))
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

if __name__ == "__main__":
    model = GhostNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

그림자 묶음은 겪어 본 살핌에 바탕을 둔다. 익힌 CNN의 결 그림을 눈으로 보면 많은 날임 갈래가 서로 몹시 얽혀 있고, 매끄럽게 하기, 가장자리 알아내기, 자리를 조금 옮기기 같은 작은 선형 바꿈만큼만 다르다. 값비싼 엮음으로 모든 갈래를 따로 셈하는 대신, 그림자 묶음은 "본디" 결 그림 한 벌을 만들고 값싼 깊이별 엮음으로 덧붙은 갈래를 이끌어 낸다.

견줌이 2이면 으뜸 엮음이 날임 갈래의 반을 내고 값싼 셈이 나머지 반을 만든다. 아끼는 셈은 대략 견줌에 비례한다. 견줌이 $s$인 그림자 묶음은 여느 엮음의 대략 $1/s$만큼만 셈한다. 값싼 셈의 깊이별 엮음은 갈래마다 따로 움직이므로 매개변수가 아주 적다.

그림자 목은 MobileNetV2의 뒤집힌 나머지와 비슷하게 꾸미되 여느 엮음을 그림자 묶음으로 갈음한다. 첫 그림자 묶음이 드러냄을 넓히고, 골라 쓰는 깊이별 엮음이 자리를 성기게 하며, 둘째 그림자 묶음이 날임 차수로 되비춘다. 이 꾸밈은 맞음과 잘 듦 사이를 좋게 맞바꾸어 GhostNet을 손전화나 가장자리 기기에 올리기에 알맞게 한다.

## 익힘 문제

**익힘 1.**
`in_channels=64`, `out_channels=128`, `ratio=2`, `dw_size=3`일 때 `GhostModule`이 여느 엮음에 견주어 매개변수를 얼마나 줄이는지 셈하여라.

??? success "익힘 1 풀이"
    여느 엮음($1 \times 1$): $64 \times 128 = 8192$개. `GhostModule`에서는 으뜸 엮음($1 \times 1$)이 $64 \times 64 = 4096$개, 값싼 셈(깊이별 $3 \times 3$)이 $64 \times 9 = 576$개다. 그림자 모두: $4096 + 576 = 4672$개. 줄임 견줌: $8192 / 4672 \approx 1.75\times$. 묶음 잣대 잡기와 치우침 없는 설정까지 넣으면 그림자 묶음은 여느 엮음의 매개변수 가운데 대략 57%만 쓴다.

---

**익힘 2.**
값싼 셈을 갈래마다의 단순한 선형 바꿈(잣대 곱하기 따위)이 아니라 깊이별 엮음으로 짜는 까닭은 무엇인가?

??? success "익힘 2 풀이"
    갈래마다 잣대만 곱하면 결 그림의 크기만 바뀔 뿐 자리 사이는 하나도 담기지 않는다. 깊이별 엮음은 본디 결 그림마다 작은 자리 거르개를 걸어, 가장자리를 돋운 것이나 매끄럽게 한 것처럼 뜻있는 갈래를 만들 수 있다. 이는 익힌 그물에서 살핀 군더더기 무늬, 곧 닮은 결 그림이 단순한 잣대가 아니라 자리 바꿈만큼 다른 것을 흉내 낸다. $3 \times 3$ 깊이별 엮음은 갈래마다 매개변수가 9개뿐이라 셈이 값싸면서도 쓸모 있는 그림자 결을 낼 만큼 넉넉하다.

---

**익힘 3.**
`GhostModule`을 값싼 셈의 도막 수를 마음대로 둘 수 있게 고쳐(그림자 결에서 또 그림자 결을 만드는 따위) 여러 켜의 그림자 층을 이루어라.

??? success "익힘 3 풀이"
    ```python
    class MultiLevelGhostModule(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=1, levels=3, dw_size=3):
            super().__init__()
            channels_per_level = out_channels // levels
            remainder = out_channels - channels_per_level * levels
            self.primary = nn.Sequential(
                nn.Conv2d(in_channels, channels_per_level + remainder, kernel_size, 1, kernel_size // 2, bias=False),
                nn.BatchNorm2d(channels_per_level + remainder),
                nn.ReLU(inplace=True),
            )
            self.ghosts = nn.ModuleList()
            for i in range(levels - 1):
                in_ch = channels_per_level + remainder if i == 0 else channels_per_level
                self.ghosts.append(nn.Sequential(
                    nn.Conv2d(in_ch, channels_per_level, dw_size, 1, dw_size // 2, groups=min(in_ch, channels_per_level), bias=False),
                    nn.BatchNorm2d(channels_per_level),
                    nn.ReLU(inplace=True),
                ))

        def forward(self, x):
            parts = [self.primary(x)]
            for ghost in self.ghosts:
                parts.append(ghost(parts[-1]))
            return torch.cat(parts, dim=1)
    ```
