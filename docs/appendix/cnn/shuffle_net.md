# ShuffleNet

ShuffleNet은 2018년 논문 "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices"에서 나왔으며, 핵심 연산 둘로 높은 효율을 얻는다. 계산을 줄이는 점별 그룹 컨볼루션(pointwise group convolution)과, 그룹 사이에 정보가 흐르게 하는 채널 셔플(channel shuffle)이다. 이 설계는 경쟁력 있는 정확도를 유지하면서 모바일 기기에서 실시간 추론을 할 수 있게 한다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
ShuffleNet - 채널 셔플을 쓰는 효율적인 컨볼루션 뉴럴 네트워크
논문: "ShuffleNet: An Extremely Efficient CNN for Mobile Devices" (2017)
핵심: 채널 셔플과 효율을 위한 그룹 컨볼루션
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

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
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
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

이 구현은 클래스 2개(`ShuffleUnit`, `ShuffleNetV2`)를 정의하며, 이들이 함께 하나의 컨볼루션 뉴럴 네트워크 구조를 이룬다. 클래스마다 역할이 나뉘어 있어 코드가 모듈화되고 확장하기 쉽다. `forward` 메서드는 PyTorch가 자동 미분에 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`ShuffleUnit`의 순전파를 따라 텐서의 형태를 추적하여라. 기본 파라메터로 샘플 4개짜리 배치를 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(패딩이 없을 때) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 끝까지 그대로임을 확인하여라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화한 뒤에는 $(B, F)$로 중간 형태를 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하여라.

</div>

??? success "풀이"
    첫 컨볼루션 레이어의 `in_channels`를 현재 값에서 3으로 바꾼다. 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 공간 차원을 다시 계산한다. 첫 선형 레이어의 `in_features`를 마지막 컨볼루션/풀링 레이어의 평탄화된 출력에 맞춘다. `model = ShuffleUnit(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력과 출력의 차원이 같을 때 표준 컨볼루션과 깊이별 분리 컨볼루션(depthwise separable convolution)의 파라메터 수와 FLOPs(부동소수점 연산 횟수)를 비교하여라. 계산을 가장 크게 절약하는 때는 언제인가?

</div>

??? success "풀이"
    표준 `Conv2d(C_in, C_out, k)`의 파라메터는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별 분리 컨볼루션은 이를 (1) 깊이별(depthwise): $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나)와 (2) 점별(pointwise): $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 컨볼루션)로 나눈다. 파라메터 수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 파라메터가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크게 절약한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`ShuffleUnit`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모두 모듈에 등록하여 옵티마이저가 갱신할 수 있다. `for n in [2, 4, 8]: model = ShuffleUnit(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — ShuffleNet

이 구현은 클래스 2개(`ShuffleUnit`, `ShuffleNetV2`)를 정의하며, 이들이 함께 하나의 컨볼루션 뉴럴 네트워크 구조를 이룬다.

핵심 클래스는 `ShuffleUnit`, `ShuffleNetV2`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
