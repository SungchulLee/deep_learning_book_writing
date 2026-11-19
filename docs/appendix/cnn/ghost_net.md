# Ghost Net

GhostNet은 2020년 논문 "GhostNet: More Features from Cheap Operations"에서 나왔으며, 학습된 CNN의 피처 맵 가운데 많은 것이 서로 닮았고 소수의 고유한 피처 맵에 값싼 선형 변환을 적용해 만들 수 있다는 점에 주목한다. 고스트 모듈(Ghost module)은 이 중복성을 이용해, 먼저 일반적인 컨볼루션으로 피처 맵 몇 개를 만들고 다음으로 단순한 깊이별 선형 연산으로 추가 "고스트" 피처를 만들어, 파라메터와 FLOPs(부동소수점 연산 횟수)를 크게 줄이면서도 비슷한 정확도를 이룬다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
GhostNet - 값싼 연산으로 피처 더 얻기
논문: "GhostNet: More Features from Cheap Operations" (2020)
핵심: 고스트 모듈이 더 적은 파라메터로 더 많은 피처를 만들어 낸다
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

**출력:**

```
Parameters: 989,680
```

## 2. 논의

고스트 모듈은 경험적 관찰에 기반을 둔다. 학습된 CNN의 피처 맵을 시각화해 보면 많은 출력 채널이 서로 상관관계가 높고, 평활화, 에지 검출, 약간의 공간 이동 같은 작은 선형 변환만큼만 다르다. 값비싼 컨볼루션으로 모든 채널을 따로 계산하는 대신, 고스트 모듈은 "고유한" 피처 맵 한 벌을 만들고 값싼 깊이별 컨볼루션으로 추가 채널을 만들어 낸다.

비율이 2이면 주 컨볼루션(primary convolution)이 출력 채널의 반을 내고 값싼 연산이 나머지 반을 만든다. 절약되는 계산량은 대략 비율에 비례한다. 비율이 $s$인 고스트 모듈은 일반 컨볼루션의 대략 $1/s$만큼만 계산한다. 값싼 연산의 깊이별 컨볼루션은 채널마다 따로 동작하므로 파라메터가 아주 적다.

고스트 병목(Ghost bottleneck)은 MobileNetV2의 역 잔차와 비슷하게 설계하되 일반 컨볼루션을 고스트 모듈로 대체한다. 첫 고스트 모듈이 표현을 확장하고, 선택적인 깊이별 컨볼루션이 공간 다운샘플링을 하며, 둘째 고스트 모듈이 출력 차원으로 사영한다. 이 설계는 정확도와 효율 사이에서 좋은 트레이드오프를 이루어 GhostNet을 모바일이나 에지 기기에 배포하기에 알맞게 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`in_channels=64`, `out_channels=128`, `ratio=2`, `dw_size=3`일 때 `GhostModule`이 일반 컨볼루션에 비해 파라메터를 얼마나 줄이는지 계산하여라.

</div>

??? success "풀이"
    일반 컨볼루션($1 \times 1$): $64 \times 128 = 8192$개. `GhostModule`에서는 주 컨볼루션($1 \times 1$)이 $64 \times 64 = 4096$개, 값싼 연산(깊이별 $3 \times 3$)이 $64 \times 9 = 576$개다. 고스트 모듈 전체: $4096 + 576 = 4672$개. 축소 비율: $8192 / 4672 \approx 1.75\times$. 배치 정규화와 바이어스 없는 설정까지 넣으면 고스트 모듈은 일반 컨볼루션의 파라메터 가운데 대략 57%만 쓴다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
값싼 연산을 채널마다의 단순한 선형 변환(스칼라 곱 등)이 아니라 깊이별 컨볼루션으로 구현하는 까닭은 무엇인가?

</div>

??? success "풀이"
    채널마다 스칼라만 곱하면 피처 맵의 크기만 바뀔 뿐 공간적 관계는 전혀 담기지 않는다. 깊이별 컨볼루션은 고유한 피처 맵마다 작은 공간 필터를 적용해, 에지를 강조한 것이나 평활화한 것처럼 의미 있는 변형을 만들 수 있다. 이는 학습된 네트워크에서 관찰되는 중복 패턴, 곧 닮은 피처 맵들이 단순한 스케일링이 아니라 공간 변환만큼 다르다는 점을 모방한다. $3 \times 3$ 깊이별 컨볼루션은 채널마다 파라메터가 9개뿐이라 계산이 값싸면서도 쓸모 있는 고스트 피처를 낼 만큼 표현력이 충분하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`GhostModule`을 값싼 연산의 단계 수를 설정할 수 있게 고쳐(고스트 피처에서 또 고스트 피처를 만드는 식이다) 다단계 고스트 계층을 구성하여라.

</div>

??? success "풀이"
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

## 정리하며

**다룬 것** — Ghost Net

고스트 모듈은 경험적 관찰에 기반을 둔다.

핵심 클래스는 `GhostModule`, `GhostBottleneck`, `GhostNet`, `MultiLevelGhostModule`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
