# RegNet

RegNet은 2020년 페이스북 AI 연구소가 발표했으며, 뉴럴 네트워크 구조를 만드는 데 설계 공간(design space)이라는 접근을 쓴다. 가장 좋은 네트워크 하나를 찾는 대신, RegNet은 소수의 변수(깊이, 초기 너비, 기울기, 그룹 너비)로 파라메터화되는 체계적인 설계 공간을 정의한다. 이 공간에서 뽑은 네트워크는 스테이지를 지나며 너비가 단순한 선형 함수로 정해지고, 빠르고 메모리 효율적이며 성능이 높은 모델이 나온다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
RegNet - 네트워크 설계 공간 설계하기
논문: "Designing Network Design Spaces" (2020)
핵심: 네트워크의 너비와 깊이를 양자화된 선형 함수로 파라메터화하기
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class RegNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)

class RegNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.layer1 = self._make_layer(32, 64, 2)
        self.head = nn.Linear(64, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1
            layers.append(RegNetBlock(in_channels if i == 0 else out_channels, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

if __name__ == "__main__":
    model = RegNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 156,872
```

## 2. 논의

이 구현은 클래스 2개(`RegNetBlock`, `RegNet`)를 정의하며, 이들이 함께 하나의 컨볼루션 뉴럴 네트워크 구조를 이룬다. 클래스마다 역할이 나뉘어 있어 코드가 모듈화되고 확장하기 쉽다. `forward` 메서드는 PyTorch가 자동 미분에 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`RegNetBlock`의 순전파를 따라 텐서의 형태를 추적하여라. 기본 파라메터로 샘플 4개짜리 배치를 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(패딩이 없을 때) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 끝까지 그대로임을 확인하여라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화한 뒤에는 $(B, F)$로 중간 형태를 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하여라.

</div>

??? success "풀이"
    첫 컨볼루션 레이어의 `in_channels`를 현재 값에서 3으로 바꾼다. 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 공간 차원을 다시 계산한다. 첫 선형 레이어의 `in_features`를 마지막 컨볼루션/풀링 레이어의 평탄화된 출력에 맞춘다. `model = RegNetBlock(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력과 출력의 차원이 같을 때 표준 컨볼루션과 깊이별 분리 컨볼루션(depthwise separable convolution)의 파라메터 수와 FLOPs(부동소수점 연산 횟수)를 비교하여라. 계산을 가장 크게 절약하는 때는 언제인가?

</div>

??? success "풀이"
    표준 `Conv2d(C_in, C_out, k)`의 파라메터는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별 분리 컨볼루션은 이를 (1) 깊이별(depthwise): $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나)와 (2) 점별(pointwise): $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 컨볼루션)로 나눈다. 파라메터 수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 파라메터가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$이 모두 클 때 가장 크게 절약한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`RegNetBlock`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모두 모듈에 등록하여 옵티마이저가 갱신할 수 있다. `for n in [2, 4, 8]: model = RegNetBlock(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — RegNet

이 구현은 클래스 2개(`RegNetBlock`, `RegNet`)를 정의하며, 이들이 함께 하나의 컨볼루션 뉴럴 네트워크 구조를 이룬다.

핵심 클래스는 `RegNetBlock`, `RegNet`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
