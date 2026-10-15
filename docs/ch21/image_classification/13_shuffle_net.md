# ShuffleNet

ShuffleNet은 그룹 합성곱과 채널 셔플을 결합해 적은 연산량의 CNN을 구성한다. 여기서 group은 채널 그룹이며 데이터의 미니배치를 뜻하지 않는다. 아래 코드는 V1 계열 연산의 일부와 미완성 본체를 담고 있어, 클래스 이름 `ShuffleNetV2`만으로 V2 전체 구현이라고 해석하면 안 된다.

## 1. 코드

```python
import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batch_size, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super().__init__()
        self.stride = stride
        mid_channels = out_channels // 4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1,
                     groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.AvgPool2d(3, 2, 1)
    
    def forward(self, x):
        out = self.bottleneck(x)
        out = channel_shuffle(out, 2)
        out = self.depthwise(out)
        out = self.expand(out)
        if self.stride == 2:
            out = torch.cat([out, self.shortcut(x)], 1)
        else:
            out += x
        return nn.functional.relu(out)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    model = ShuffleNetV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,025,696
```

## 2. 논의

### 그룹 합성곱과 채널 셔플

그룹 수 g가 입력·출력 채널 수를 나누면 표준 합성곱의 K²C_inC_out 가중치를 g분의 1로 줄일 수 있다. 각 출력 그룹은 대응 입력 그룹만 참조하므로 같은 분할을 반복하면 그룹 사이의 정보 교환이 제한된다.

채널 셔플은 N×C×H×W를 N×g×(C/g)×H×W로 바꾸고 두 채널 관련 축을 교환해 다시 N×C×H×W로 편다. 배치 축을 섞는 연산이 아니다. 학습 가중치가 없는 순열이지만 `contiguous`에 따른 메모리 복사·접근 비용이 생길 수 있다.

셔플 뒤 다음 그룹이 여러 이전 그룹의 채널을 받게 할 수 있지만, 모든 설정에서 각 그룹이 이전의 모든 그룹을 한 번에 포함하는 것은 아니다. C와 g, 다음 분할의 폭에 따라 실제 채널 구성을 확인해야 한다.

### 단위 블록의 구현 조건

`ShuffleUnit`의 두 1×1은 인수 `groups`를 쓰지만 `channel_shuffle(out,2)`는 항상 그룹 수 2를 쓴다. 따라서 그룹 합성곱의 설정과 섞기 설정이 일반적으로 일치하지 않는다. 중간 폭 `out_channels//4`와 입출력 폭의 나눗셈 조건도 충족해야 한다.

stride=1이면 주 분기의 `out_channels`와 입력 채널 수가 같아야 덧셈이 가능하다. stride=2이면 평균 풀링한 입력과 주 분기를 연결하므로 실제 출력 폭은 **out_channels+in_channels**다. `out_channels`를 전체 블록의 최종 폭이라고 가정하면 다음 단계가 맞지 않는다. 원하는 최종 폭을 기준으로 두 분기의 폭을 따로 설계해야 한다.

### 본체와 버전 구분

코드의 `ShuffleNetV2`는 24채널 stem과 풀링 뒤 1024입력 선형층을 바로 연결한다. 중간 셔플 단계가 없고 전역 풀링한 특징은 24차원이므로 순전파에서 차원 불일치가 발생한다. 매개변수 개수만 출력하는 기존 실행부는 이 문제를 검사하지 않는다.

V1의 그룹 1×1과 셔플 설계는 [원 ShuffleNet 논문](https://arxiv.org/abs/1707.01083)을, 채널 분할·분기 및 실제 비용을 고려한 별도 V2 설계는 [ShuffleNet V2 논문](https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html)을 기준으로 구분한다. 이론적 MAC 절감이 같은 비율의 장치 속도 개선을 뜻하지는 않는다. 코드·출력은 그대로 유지했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
NCHW 입력에서 C=6, g=2인 채널 셔플 순서를 설명하라.

</div>

??? success "연습문제 1 풀이"
    채널 [a0,a1,a2,b0,b1,b2]를 2×3으로 묶고 3×2로 전치한 뒤 펴면 [a0,b0,a1,b1,a2,b2]다. 각 표본과 공간 위치에 같은 순열을 적용하며 배치의 서로 다른 표본을 섞지 않는다. g>0이고 C가 g로 나누어져야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 코드의 stride=2 ShuffleUnit이 내는 채널 수와 최종 폭 설계의 관계를 설명하라.

</div>

??? success "연습문제 2 풀이"
    주 분기는 out_channels, 풀링 지름길은 in_channels를 유지하므로 연결 결과는 둘의 합이다. out_channels가 최종 목표 폭이라고 생각하고 블록을 쌓으면 다음 층이 맞지 않는다. stride=1의 덧셈에는 같은 채널 수·공간 크기가 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
그룹 합성곱의 매개변수 비와 실제 실행비용의 차이를 설명하라.

</div>

??? success "연습문제 3 풀이"
    가중치는 K²C_inC_out/g개로 표준의 1/g이며 같은 출력 공간에서 MAC도 같은 비로 줄어든다. 편향·정규화·셔플·메모리 접근·커널 호출은 별도 비용이다. g가 양쪽 채널 수를 나누어야 하며 작은 연산이 장치에서 비효율적이면 실제 속도 비율은 달라진다.

## 정리하며

그룹 합성곱은 채널을 나누고 셔플은 채널 순서를 바꾼다. 그룹과 배치, 덧셈과 연결의 채널 규칙을 구분한다. 이 파일의 고정 셔플 설정과 미완성 본체는 원 V1·V2 전체 구조의 동작을 보장하지 않는다.
