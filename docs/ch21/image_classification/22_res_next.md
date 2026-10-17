# ResNeXt

ResNeXt는 같은 형태의 여러 변환을 집계하는 잔차 구조다. 변환의 수인 cardinality를 깊이·폭과 별도의 설계 축으로 다루며, 그룹 컨볼루션으로 구현할 수 있다. 같은 형태의 경로가 웨이트를 공유하는 것은 아니다. 아래 본체는 한 단계만 구현해 전체 ResNeXt-50과 다르다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
ResNeXt — 모아 붙인 잔차 변환
논문: "Aggregated Residual Transformations for Deep Neural Networks" (2017)
핵심: 깊이·너비를 넘어 잣수(길의 수)가 중요한 차원이다
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

### Cardinality와 그룹 폭

코드는 D=int(out_channels·bottleneck_width/64), 내부 폭 B=cardinality·D를 사용한다. 기본 out_channels=64, bottleneck_width=4, cardinality=32이면 D=4, B=128이다. 1×1로 내부 폭을 만든 뒤 groups=32인 3×3을 적용하고 1×1로 256채널을 출력한다.

그룹 3×3의 웨이트는 9B²/cardinality=9·cardinality·D²개다. 양쪽 1×1은 채널을 혼합하므로 전체 블록의 모든 연산이 서로 완전히 독립인 것은 아니다. Cardinality를 바꾸면서 D를 고정하면 내부 폭과 비용도 커진다. 공정한 비교에는 전체 비용을 맞추는 폭 조정이 필요하다.

### 잔차 형태와 본체의 범위

입력과 출력 채널 또는 stride가 다르면 지름길에 1×1 투영과 BN을 사용한다. 주 분기와 투영의 공간 크기가 같아야 덧셈할 수 있다. out_channels는 블록의 최종 폭이 아니라 4배 확장 전 기준 폭이다.

`ResNeXt50`은 stem 뒤 [3]개 블록의 첫 단계만 사용한다. 일반적인 [3,4,6,3] 전체 모델의 뒤 세 단계가 없으므로 이름의 50레이어 구조를 재현하지 않는다. 대신 최종 256채널과 분류기의 입력은 서로 맞는 축약 분류기다. 기존 출력은 파라미터 수이며 정확도나 완전한 구조의 확인은 아니다.

### 입력 크기와 블록 수 변경

입력 RGB 채널은 이미 3이고 적응형 풀링을 사용하므로 64×64 이미지를 받는다고 분류기 폭을 공간 크기만큼 늘릴 필요는 없다. 이 예에서는 64→32(stem)→16(풀링)이고 첫 단계는 16×16을 유지한다. 더 작은 이미지용 stem 변경은 필수 차원 수정과 별도의 설계 선택이다.

블록 수를 바꿀 때 첫 블록은 이전 단계의 폭을 받아 투영할 수 있고 뒤 블록은 확장된 출력 폭을 받아야 한다. `ModuleList`는 파라메터 등록을 하지만 자동으로 forward를 실행하지 않는다. 연습의 `YourBlock` 골격은 정의되지 않은 표시자이므로 완성된 ResNeXt 확장 코드가 아니다. [원 논문](https://arxiv.org/abs/1611.05431)의 다중 변환 설계와 이 예제를 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
B=4, 입력 64×56×56, out_channels=64인 기본 블록의 형태를 추적하라.

</div>

??? success "풀이"
    내부 D=4, cardinality=32이므로 폭 128이다. stride=1에서 conv1 뒤 (4,128,56,56), 그룹 conv2 뒤 같은 형태, conv3 뒤 (4,256,56,56)다. 지름길도 64→256으로 투영해 같은 형태를 만들고 더한다. 기준 out_channels와 최종 4배 폭을 구분한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 입력에서 이 축약 본체의 공간 크기와 분류기 입력을 계산하라.

</div>

??? success "풀이"
    7×7 stride2 padding3 뒤 32×32, 최대 풀링 뒤 16×16이다. 첫 단계는 stride1이라 공간을 유지하며 256채널을 만든다. 적응형 평균 풀링 후 N×256이므로 기존 fc(256,num_classes)에 맞는다. 입력 채널은 이미 3이며 블록만 RGB에 직접 연결하는 것과 본체에 넣는 것을 혼동하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
cardinality=A, 그룹당 폭 D인 3×3의 웨이트 수를 구하고 비용 비교 조건을 쓰라.

</div>

??? success "풀이"
    내부 폭 B=AD이므로 그룹 컨볼루션은 9B²/A=9AD²개다. 같은 B의 표준 컨볼루션은 9B²개다. 그러나 A를 늘릴 때 D를 고정하면 B도 커지고 양쪽 1×1 비용도 증가한다. 전체 블록 비용을 맞춰 비교해야 하며 depthwise separable의 비용 공식과 다른 구조다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 골격을 완성하려면 필요한 채널 추적과 실행을 설명하라.

</div>

??? success "풀이"

    YourBlock·dim·num_layers가 정의되지 않은 골격이다. 첫 블록은 입력 폭에서 4×out_channels로 바꾸고 다음 블록은 그 확장 폭을 입력받아야 한다. 필요한 stride·투영을 구성하고 ModuleList를 forward에서 순회해야 한다. 반복 개수를 바꾸는 것과 실제 ResNeXt-50의 단계별 블록 구성을 재현하는 것은 구분한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

ResNeXt의 cardinality는 그룹당 폭과 함께 내부 채널·비용을 결정한다. 잔차 출력의 4배 확장과 첫 블록의 투영을 추적하고, 한 단계 본체를 전체 ResNeXt-50의 결과로 해석하지 않는다.
