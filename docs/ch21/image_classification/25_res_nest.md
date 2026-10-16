# ResNeSt

ResNeSt는 여러 분기의 피처를 채널별 어텐션으로 결합하는 split-attention 잔차 네트워크다. radix는 결합할 분기의 수이며, 미니배치 크기나 그룹 컨볼루션의 그룹 수와 다르다. 아래 코드는 기본적인 가중 결합과 잔차를 보여 주는 한 단계의 축약형이다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
ResNeSt — 쪼갠 어텐션 넷트웍
논문: "ResNeSt: Split-Attention Networks" (2020)
핵심: 피처 배치에 걸쳐 채널마다 어텐션을 주는 쪼갠 어텐션 덩이
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class SplitAttention(nn.Module):
    def __init__(self, channels, radix=2, groups=1):
        super().__init__()
        self.radix = radix
        self.groups = groups
        inter_channels = max(channels * radix // 4, 32)
        
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=groups)
        self.rsoftmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        batch = x.size(0)
        x = x.view(batch, self.radix, -1, *x.shape[2:])
        gap = x.sum(dim=1)
        gap = nn.functional.adaptive_avg_pool2d(gap, 1)
        
        atten = self.fc1(gap)
        atten = self.bn1(atten)
        atten = self.relu(atten)
        atten = self.fc2(atten)
        atten = atten.view(batch, self.radix, -1)
        atten = self.rsoftmax(atten)
        atten = atten.view(batch, self.radix, -1, 1, 1)
        
        out = (x * atten).sum(dim=1)
        return out

class ResNeStBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, radix=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * radix, 3, stride, 1, groups=radix, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * radix)
        self.split_attention = SplitAttention(out_channels, radix)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.split_attention(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNeSt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            stride = 2 if i == 0 and in_channels != 64 else 1
            layers.append(ResNeStBlock(in_channels if i == 0 else out_channels * 4, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = ResNeSt()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 424,936
```

## 2. 논의

### 분기별 피처와 가중합

`SplitAttention(C,R)`의 입력은 N×(RC)×H×W다. 이를 N×R×C×H×W로 나누고 분기를 더한 N×C×H×W에서 전역 평균을 구한다. 축소·확장 1×1 변환으로 RC개의 로짓을 만든 뒤 N×R×C로 바꾸어 radix 축에 softmax를 적용한다.

각 표본·채널에서 R개 웨이트의 합은 1이다. 최종 Y_nc=∑ᵣa_nrc X_nrc는 N×C×H×W다. SE의 독립 sigmoid 게이트와 달리 분기 사이에서 정규화한 가중합이며, 공간 위치별로 다른 웨이트나 위치 쌍의 셀프 어텐션을 계산하지 않는다.

### 형태와 정규화 조건

C는 분기당 채널 수다. C채널만 입력하고 R로 나누면 풀링 뒤 C/R채널이 되어 fc1의 C입력과 맞지 않는다. RC채널과 view 가능한 메모리 형태를 확인해야 한다. `groups`는 두 1×1의 그룹 수이며 C·중간 폭·RC가 그 그룹 수에 맞아야 한다. radix·groups를 같은 의미로 쓰면 안 된다.

전역 평균 뒤 BN은 N×inter_channels×1×1을 받으므로 학습 모드의 N=1에서는 채널당 통계 표본이 하나여서 문제가 된다. 원 영상의 공간 크기가 커도 이 위치에서는 이미 1×1로 축소되어 있다. R=1이면 코드의 softmax 웨이트는 항상 1이다. 이 특수 경우에 sigmoid 게이트를 사용하는 다른 구현과 동일한 기능이라고 가정하지 않는다.

### 잔차 블록과 전체 모델

주 분기는 1×1 뒤 R배 채널을 만드는 그룹 3×3, split-attention, 최종 4배 폭의 1×1을 적용한다. 필요한 경우 지름길도 1×1로 형태를 맞춘다. 본체는 64 기준 폭의 블록 두 개와 분류기만 사용하며 전체 다단계 ResNeSt-50 등의 모델은 아니다.

64×64 RGB 입력은 stem 뒤 32×32, 풀링 뒤 16×16이다. 첫 단계의 최종 폭은 256이고 전역 풀링 후 fc의 256입력과 맞는다. 작은 배치에서의 위 BN 조건과 실제 코드의 순전파 범위를 별도로 확인한다. 기존 실행부는 파라미터 수만 출력한다.

연습의 `YourBlock` 목록은 미완성 반복 골격이다. 블록을 늘리려면 첫 블록과 후속 블록의 확장 폭, stride·투영을 추적해야 한다. 본체 이름이나 연결 수만으로 [원 논문](https://arxiv.org/abs/2004.08955)의 정확도·탐지·분할 성능을 재현했다고 판단하지 않는다. 코드·출력은 실행하지 않고 유지했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
C=64, R=2, 입력 (4,128,16,16)의 split-attention 형태를 추적하라.

</div>

??? success "연습문제 1 풀이"
    분리 뒤 (4,2,64,16,16), 분기 합 뒤 (4,64,16,16), 전역 풀링 뒤 (4,64,1,1)이다. 중간 폭은 max(64×2/4,32)=32이고 fc2는 128채널을 만든다. (4,2,64)의 radix softmax 뒤 웨이트를 곱해 분기를 합하면 (4,64,16,16)이 된다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
학습 모드에서 배치 1의 큰 이미지가 전역 풀링 뒤 BN 문제를 해결하는가?

</div>

??? success "연습문제 2 풀이"
    해결하지 않는다. split-attention의 전역 풀링 뒤 형태는 (1,inter_channels,1,1)이므로 채널별 통계값이 하나다. 앞의 큰 공간은 이 BN에 남아 있지 않다. 배치·정규화 설계와 평가 모드의 통계 사용을 구분해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
radix softmax와 SE의 sigmoid, 위치 쌍 셀프 어텐션을 비교하라.

</div>

??? success "연습문제 3 풀이"
    radix softmax는 표본·채널마다 분기 웨이트 합을 1로 만들고 피처를 가중합한다. SE는 채널별 독립 게이트를 곱하며 합이 1일 필요가 없다. 둘 다 여기서는 공간 위치 쌍의 n×n 어텐션을 계산하지 않는다. R=1에서 이 코드의 softmax는 항상 1이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 반복 골격으로 ResNeSt 단계를 만들 때 필요한 입력 폭을 설명하라.

</div>

??? success "연습문제 4 풀이"

    첫 블록은 이전 단계 폭을 받고 4×기준 폭을 출력한다. 뒤 블록은 그 확장 폭을 입력받아야 하며 필요한 stride·투영과 R배 split-attention 입력을 맞춘다. YourBlock·dim 등이 정의되지 않은 코드이므로 ModuleList만 붙여 완성된 단계나 동적인 SplitAttention이 되지는 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

Split-attention은 분기당 채널과 radix를 구분해 가중합한다. softmax 축, RC입력, 전역 풀링 뒤 BN의 유효 표본 수와 확장 출력 폭을 실제 코드 기준으로 확인한다. 한 단계 예제는 전체 ResNeSt와 다르다.
