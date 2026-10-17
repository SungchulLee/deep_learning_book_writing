# ResNet

ResNet은 잔차 함수를 입력에 더하는 연결로 깊은 네트워크의 최적화를 돕는다. He 외의 논문은 2015년에 사전 공개되고 CVPR 2016에 발표되었다. 아래 ResNet-50은 1×1–3×3–1×1 병목 블록과 [3,4,6,3]의 블록 구성을 사용한다. 잔차 연결이 모든 경사 소실이나 학습 문제를 완전히 없애는 것은 아니다.

## 1. 코드

```python
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        layers = [Bottleneck(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels * 4, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


if __name__ == "__main__":
    model = ResNet50()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 25,557,032
```

## 2. 논의

### 잔차 경로와 활성화

코드의 블록은 y=ReLU(F(x)+s(x))다. 형태가 같으면 s(x)=x이고, 채널 수나 해상도가 달라지면 학습 가능한 투영을 사용한다. F를 0에 가깝게 두면 더하기 전에는 지름길이 남지만, 덧셈 뒤 ReLU가 있으므로 모든 실수 입력에서 정확한 항등 함수가 되는 것은 아니다. 같은 형태의 항등 지름길에서 덧셈 전 미분은 I+J_F이므로 직접적인 경로를 제공한다. 이후 활성화와 여러 레이어의 곱까지 고려하면 경사가 항상 잘 보존된다는 보장은 없다.

### 병목과 투영

병목의 `out_channels`는 내부 폭이고 최종 폭은 그 4배다. 1×1 컨볼루션으로 채널을 줄인 뒤 3×3 공간 연산을 수행하고 다시 채널을 확장한다. 비용 절감은 비교할 입출력 폭과 구조를 고정했을 때 계산해야 한다.

첫 단계도 입력 64채널에서 출력 256채널로 바뀌므로 stride=1인 투영이 필요하다. 뒤 단계의 첫 블록은 stride=2와 채널 확장을 함께 처리한다. 같은 단계의 나머지 블록은 형태가 같아 항등 지름길을 쓴다. `downsample`은 이름과 달리 공간 축소 없이 채널만 맞추기도 한다.

### 구현의 범위

아래 예제는 병목의 3×3 컨볼루션에 stride를 둔다. 출력은 분류 로짓이며 확률이 필요하면 별도로 정규화한다. 적응형 평균 풀링은 마지막 공간 크기를 1×1로 만들지만 매우 작은 입력이나 학습 모드의 작은 배치에서는 배치 정규화의 유효 표본 수를 고려해야 한다.

연습문제의 코드는 BasicBlock만 정의하며 완성된 ResNet-18을 구성하지 않는다. 블록 수뿐 아니라 초기 stem, 단계별 채널 추적·투영, 최종 분류기까지 맞춰야 한다. 코드와 기존 출력은 실행하지 않고 유지했다. [원 논문](https://arxiv.org/abs/1512.03385)은 일반 깊은 네트워크의 학습 오차 증가 문제와 잔차 파라메터화를 비교한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
stride=1, 투영 없는 256→64→64→256 병목의 56×56 컨볼루션 MAC 수와 FLOP 수를 계산하라.

</div>

??? success "풀이"
    바이어스·배치 정규화·ReLU·덧셈을 제외하면 각 컨볼루션 MAC은 HWC_inC_outK²다. 세 레이어는 51,380,224, 115,605,504, 51,380,224 MAC으로 총 218,365,952 MAC이다. 곱셈과 덧셈을 각각 1 FLOP으로 세면 총 436,731,904 FLOP이다. MAC 1회와 FLOP 1회의 관례를 먼저 명시한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
첫 번째 잔차 단계에서 stride=1인 투영이 필요한 이유를 설명하라.

</div>

??? success "풀이"
    입력은 64채널이고 병목 출력은 64×4=256채널이므로 해상도가 같아도 덧셈을 위해 채널을 맞춰야 한다. 뒤 단계에서는 채널과 공간 크기가 함께 바뀐다. 필요한 투영을 제거하면 out과 identity의 형태가 달라 덧셈이 실패한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 BasicBlock을 이용해 ResNet-18을 구성하려면 추가로 무엇이 필요한가? 표준 분류기의 대략적인 규모도 비교하라.

</div>

??? success "풀이"

    아래 정의는 블록 하나의 연산만 제공한다. [2,2,2,2] 블록 수, expansion=1에 맞춘 단계별 채널 추적과 투영, stem·풀링·분류기를 구성해야 한다. 표준 1,000클래스 ResNet-18은 약 1,170만, ResNet-50은 약 2,560만 파라메터지만 클래스 수와 구성에 따라 달라진다. Bottleneck의 4배 채널 규칙을 그대로 복사하면 안 된다.

    ```python
    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample:
                identity = self.downsample(x)
            return self.relu(out + identity)
    ```


## 정리하며

ResNet은 잔차 경로와 지름길을 더해 깊은 모델의 최적화를 돕는다. 실제 블록의 덧셈 뒤 활성화, 채널·해상도 투영과 병목 폭을 구분한다. 연산량은 MAC/FLOP의 정의와 포함하는 연산을 명시해 비교한다.
