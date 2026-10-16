# ResNeXt

ResNeXt은 2017년 글 "깊은 뉴럴 넷트웍을 위한 모은 나머지 바꿈"에서 나왔으며, 여느 나머지 덩이를 모은 바꿈으로 갈음해 ResNet을 넓혔다. 덩이마다 계산을 여러 나란한 길(클래스 수)로 가르고, 길마다 같은 바꿈을 걸되 배운 짐은 서로 다르다. 이 여러 가지 꾸밈은 단순하고 고른 짜임새를 지키면서 네트워크가 드러내는 힘을 키운다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
ResNeXt - 모아 엮은 나머지 바꿈
논문: "Aggregated Residual Transformations for Deep Neural Networks" (2017)
핵심: 깊이와 너비를 넘어선 종요로운 차원인 클래스 수(길의 수)
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality=32, bottleneck_width=4, stride=1):
        super().__init__()
        D = int(out_channels * (bottleneck_width / 64))
        group_width = cardinality * D
        
        self.conv1 = nn.Conv2d(in_channels, group_width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, 3, stride, 1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class ResNeXt50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            stride = 2 if i == 0 and in_channels != 64 else 1
            layers.append(ResNeXtBlock(in_channels if i == 0 else out_channels * 4, out_channels, stride=stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = ResNeXt50()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 472,360
```

## 2. 논의

이 구현은 클래스 2개(`ResNeXtBlock`, `ResNeXt50`)를 매기고, 이들이 어울려 온전한 컨볼루션 뉴럴 넷트웍 구조를 이룬다. 클래스마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 계산 그림을 매긴다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 증강 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`ResNeXtBlock`의 앞으로 스텝을 따라 텐서의 꼴을 좇아라. 기본 파라메터로 입력 보기 4개를 묶어 넣었을 때, 큰 계산(컨볼루션, 모으기, 선형 켜)마다 꼴이 어떻게 되는지 적어라.

</div>

??? success "연습문제 1 풀이"
    입력의 꼴에서 비롯해 켜를 차례로 건다. `Conv2d(in_c, out_c, k)`마다 자리 차수는 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(덧대기 없이) `padding=k//2`이면 그대로다. 알갱이 2로 모으면 자리 차수가 반이 된다. 선형 켜는 마지막 차수를 바꾼다. 배치 차수는 내내 그대로임을 좇아라. 컨볼루션 켜에서는 $(B, C, H, W)$, 편 뒤에는 $(B, F)$으로 가운데 꼴을 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 그림(입력 꼴: $3 \times 64 \times 64$)을 받도록 고쳐라. 켜의 차수를 모두 그에 맞게 손보고 모델이 어긋남 없이 도는지 따져라.

</div>

??? success "연습문제 2 풀이"
    첫 컨볼루션 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 컨볼루션과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 계산한다. 첫 선형 켜의 `in_features`을 마지막 컨볼루션/모으기 켜의 편 출력에 맞춘다. `model = ResNeXtBlock(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력과 출력의 차수가 같을 때 여느 컨볼루션과 깊이별로 가른 컨볼루션의 파라메터 수와 뜨는 계산 횟수를 비교하여라. 계산을 가장 크게 아끼는 때는 언제인가?

</div>

??? success "연습문제 3 풀이"
    여느 `Conv2d(C_in, C_out, k)`의 파라메터는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별로 가른 컨볼루션은 이를 (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 클래스마다 거르개 하나)와 (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 컨볼루션)로 나눈다. 파라메터의 비교는 어림잡아 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 파라메터가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$이 모두 클 때 가장 크게 아낀다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`ResNeXtBlock`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라메터를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = ResNeXtBlock(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — ResNeXt

이 구현은 클래스 2개(`ResNeXtBlock`, `ResNeXt50`)를 매기고, 이들이 어울려 온전한 컨볼루션 뉴럴 넷트웍 구조를 이룬다.

핵심 클래스는 `ResNeXtBlock`, `ResNeXt50`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
