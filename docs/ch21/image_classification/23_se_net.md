# SENet

Squeeze-and-Excitation(SE)는 공간 피처를 채널별 통계로 요약하고 입력에 따라 채널 게이트를 계산하는 모듈이다. Hu 외의 CVPR 2018 논문은 이 모듈을 여러 CNN에 결합해 평가했다. 아래 구현은 SE와 잔차 블록의 관계를 보여 주는 축약 분류기이며 모든 구조에서 성능 향상을 보장하는 것은 아니다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
SENet — 쥐어짜기-북돋우기 넷트웍
논문: "Squeeze-and-Excitation Networks" (2018)
ILSVRC 2017에서 우승
핵심: 채널 어텐션 구조로 채널마다의 피처를 다시 매긴다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)

class SEResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            stride = 2 if i == 0 and in_channels != 64 else 1
            layers.append(SEResNetBlock(in_channels if i == 0 else out_channels, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = SEResNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 223,528
```

## 2. 논의

### 공간 요약과 채널 게이트

입력 X∈Rᴺˣᶜˣᴴˣᵂ를 전역 평균 풀링해 z∈Rᴺˣᶜ를 만든다. 축소 폭 m=⌊C/r⌋에 대해 g=sigmoid(W₂ ReLU(W₁z))를 계산하고 Y_nchw=X_nchw g_nc로 출력한다. 게이트는 같은 채널의 모든 공간 위치에 공유된다.

각 게이트는 0과 1 사이지만 채널 합이 1인 확률은 아니다. 현재 입력을 기준으로 채널별 기여를 조절하는 학습된 변환이지 채널의 객관적 중요도나 인과적 설명을 보장하지 않는다. 공간을 평균하므로 이 게이트 자체는 위치별 마스크도 아니다.

### 파라메터와 계산량

코드의 두 Linear는 편향이 없어 웨이트 수는 Cm+mC=2Cm이다. C=64, r=16이면 m=4, 512개다. 다만 풀링과 전체 피처에 대한 곱셈 비용은 NHWC에 따라 달라져 웨이트 수만으로 실행비용 전체를 평가할 수 없다. r은 양수이고 의미 있는 병목에는 m≥1이 필요하다. 원시 RGB 3채널에 r=16을 그대로 쓰면 축소 폭이 0이 되어 유용한 채널 표현을 만들지 못한다.

### 잔차와 본체의 범위

`SEResNetBlock`은 두 3×3 주 분기의 BN 뒤 SE를 적용하고 지름길과 더한 후 ReLU를 취한다. 형태가 바뀌면 지름길은 1×1과 BN으로 투영한다. SE가 지름길까지 한꺼번에 게이팅하는 배치와 다르며 삽입 위치도 모델 정의의 일부다.

`SEResNet`은 64채널 블록 두 개의 한 단계만 사용한다. 완성된 SE-ResNet-50 등 공개 대형 모델이 아니지만 최종 64채널과 fc 입력은 맞는 축약형이다. 64×64 RGB 입력은 stem 뒤 32×32, 풀링 뒤 16×16이 되고 해당 단계에서 크기를 유지한 뒤 적응형 풀링된다. 공간 크기를 바꿀 때 fc의 입력을 H×W배 할 필요는 없다.

### 평가와 확장

SE를 추가하는 것과 전체 모델의 깊이를 바꾸는 것은 다른 실험이다. 깊이를 바꾸려면 잔차 블록을 반복하고 첫 블록의 폭·stride·투영을 맞춘다. 연습의 `YourBlock` 코드는 정의되지 않은 표시자를 사용하므로 완성된 SEBlock 확장 구현이 아니다. SEBlock을 단순히 반복하면 게이트가 계속 곱해지는 다른 구조가 된다.

SE의 배경과 실험 결과는 [원 논문](https://arxiv.org/abs/1709.01507)을 참고한다. 코드·출력은 실행하지 않고 유지했으며 파라메터 개수 출력은 정확도 검증이 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 (4,64,16,16), reduction=16인 SEBlock의 중간 형태를 추적하라.

</div>

??? success "연습문제 1 풀이"
    전역 평균 풀링 뒤 (4,64,1,1), view 뒤 (4,64), 첫 Linear 뒤 (4,4), 둘째 Linear와 sigmoid 뒤 (4,64)다. 다시 (4,64,1,1)로 바꿔 입력에 브로드캐스트 곱하므로 최종은 (4,64,16,16)이다. SE는 공간 해상도를 유지한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 입력에 SEBlock을 직접 적용하는 것과 SEResNet에 입력하는 것을 구분하라.

</div>

??? success "연습문제 2 풀이"
    SEResNet은 이미 RGB 입력을 받고 stem이 64채널을 만든다. stem·풀링 뒤 16×16이며 최종 적응형 풀링 후 64차원을 fc에 전달한다. SEBlock 단독은 채널 게이팅 모듈이므로 입력 채널과 맞는 C를 지정해야 하고 C=3에 기본 reduction=16을 쓰면 축소 폭이 0이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
SE의 파라메터·공간 마스크·셀프 어텐션 차이를 설명하라.

</div>

??? success "연습문제 3 풀이"
    편향 없는 SE는 2C⌊C/r⌋개의 웨이트를 갖고 공간 평균을 이용해 채널별 게이트 하나를 만든다. 위치 쌍의 n×n 어텐션을 계산하지 않고 같은 채널의 모든 위치에 같은 게이트를 곱한다. 공간 위치별 마스크나 인과적 중요도의 직접 측정도 아니다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 골격으로 SE 잔차 모델을 확장할 때 결정할 항목을 설명하라.

</div>

??? success "연습문제 4 풀이"

    YourBlock과 dim 등이 정의되지 않은 일반 골격이다. 반복할 단위가 SEBlock인지 SEResNetBlock인지, 단계별 폭·stride·투영과 forward 순회를 정해야 한다. SEBlock만 반복하면 별도 피처 변환 없이 게이트를 여러 번 곱하는 구조다. ModuleList 등록만으로 실행 순서나 형태가 자동 해결되지는 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

SE는 전역 평균 피처에서 채널 게이트를 계산한다. 게이트와 확률·설명, 채널 변환과 공간 마스크를 구분하며, 삽입 위치·축소 폭·실제 실행비용을 확인해야 한다.
