# CBAM

컨볼루션 블록 어텐션 모듈(Convolutional Block Attention Module, CBAM)은 2018년 같은 이름의 논문에서 나왔으며, 채널 어텐션과 공간 어텐션을 차례로 적용해 피처 맵을 정제한다. "무엇"에 주목할지(채널 어텐션)와 "어디"에 주목할지(공간 어텐션)를 학습하여, 어떤 CNN 백본에도 추가 비용이 거의 없이 끼워 넣을 수 있는 가볍고 효과적인 모듈이다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
CBAM - 컨볼루션 블록 어텐션 모듈
논문: "CBAM: Convolutional Block Attention Module" (2018)
핵심: 채널 어텐션과 공간 어텐션을 차례로 적용하는 구조
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class CBAM_ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.cbam = CBAMBlock(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.cbam(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = CBAM_ResNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 75,146
```

## 2. 논의

CBAM은 잇따르는 두 하위 모듈로 이루어진다. 채널 어텐션 모듈은 평균 풀링과 최대 풀링으로 공간 정보를 집계하여 채널 사이의 관계를 이용한다. 두 결과를 공유 다층 퍼셉트론(MLP)에 넣고 그 출력을 합친다. 이렇게 두 가지로 풀링하면 평균적인 반응(어떤 피처가 전반적으로 나타나는지)과 가장 두드러진 반응(어떤 피처가 가장 강한지)을 함께 담아, 하나만 쓸 때보다 풍부한 채널 기술자(descriptor)를 얻는다.

공간 어텐션 모듈은 피처 맵 안에서 정보량이 많은 위치를 찾아 채널 어텐션을 보완한다. 채널 축을 따라 평균 풀링과 최대 풀링으로 채널 정보를 압축하고, 두 기술자를 이어 붙인 뒤 컨볼루션을 적용해 공간 어텐션 맵을 만든다. 채널 어텐션 다음에 공간 어텐션을 적용하는 순서가 핵심이다. 채널 어텐션이 먼저 "무엇"이 중요한지 고르고, 그다음 공간 어텐션이 그 피처가 "어디"에서 가장 중요한지 정한다.

설계는 일부러 가볍게 했다. 채널 어텐션의 축소 비율(기본값 16)이 병목 MLP를 작게 하고, 공간 어텐션은 컨볼루션 하나만 쓴다. 그래서 ResNet 같은 기존 구조에 넣기 쉽고, 파라메터는 흔히 1%도 안 늘면서 분류, 탐지, 세그멘테이션 과제에서 일관되게 정확도를 올린다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
형태가 $(B, 64, H, W)$인 입력 피처 맵과 축소 비율 16이 주어졌을 때 `ChannelAttention` 모듈의 파라메터 수를 계산하여라.

</div>

??? success "풀이"
    공유 MLP는 바이어스 없는 $1 \times 1$ 컨볼루션 둘로 이루어진다. 첫째는 $64 \to 64/16 = 4$로 파라메터가 $64 \times 4 = 256$개, 둘째는 $4 \to 64$로 $4 \times 64 = 256$개다. 모두 $256 + 256 = 512$개다. 풀링 레이어에는 학습할 파라메터가 없고 시그모이드에도 파라메터가 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
CBAM이 공간 어텐션보다 채널 어텐션을 먼저 적용하는 까닭은 무엇인가? 순서를 뒤집거나 병렬로 두지 않는 이유를 직관적으로 설명하여라.

</div>

??? success "풀이"
    채널 어텐션이 먼저 어느 피처 채널이 중요한지 정하여 "무엇"에 주목할지 답한다. 채널의 중요도로 피처에 웨이트를 다시 준 뒤라야 공간 어텐션이 그 피처가 "어디"에 있는지를 더 잘 찾는다. 공간 어텐션을 먼저 적용하면 어느 채널이 가장 중요한 정보를 지녔는지 모른 채 모든 채널을 똑같이 다루게 된다. 차례로 적용하면 정보가 단계적으로 이어진다. 채널을 골라 피처를 정제해 주므로 공간 어텐션이 더 깨끗한 신호로 위치를 찾는다. 원 논문은 채널을 먼저 적용하는 순서가 공간을 먼저 적용하거나 병렬로 적용하는 것보다 일관되게 낫다는 것을 실험으로 보였다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`SpatialAttention` 모듈을 풀링 연산의 수를 설정할 수 있게 확장하여라(평균 풀링과 최대 풀링에 더해 $L^2$ 노름 풀링을 넣는 식이다). 고친 클래스를 작성하여라.

</div>

??? success "풀이"
    ```python
    class SpatialAttentionExtended(nn.Module):
        def __init__(self, kernel_size=7, use_l2=True):
            super().__init__()
            in_channels = 3 if use_l2 else 2
            self.use_l2 = use_l2
            self.conv = nn.Conv2d(in_channels, 1, kernel_size,
                                  padding=kernel_size // 2, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            descriptors = [avg_out, max_out]
            if self.use_l2:
                l2_out = torch.norm(x, p=2, dim=1, keepdim=True)
                descriptors.append(l2_out)
            combined = torch.cat(descriptors, dim=1)
            return self.sigmoid(self.conv(combined))
    ```
    $L^2$ 노름 풀링은 위치마다의 전체 에너지를 재어, 평균(평균 신호)과 최댓값(피크 신호) 기술자를 보완한다.

## 정리하며

**다룬 것** — CBAM

CBAM은 잇따르는 두 하위 모듈로 이루어진다.

핵심 클래스는 `ChannelAttention`, `SpatialAttention`, `CBAMBlock`, `CBAM_ResNet`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
