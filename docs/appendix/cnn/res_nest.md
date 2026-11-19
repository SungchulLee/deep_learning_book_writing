# ResNeSt

ResNeSt(Split-Attention Network)는 2020년에 나왔으며, 잔차 블록마다 여러 피처 맵 그룹에 채널별 어텐션을 적용하는 분할 어텐션(split-attention) 블록을 넣어 ResNet을 확장했다. SE-Net과 SK-Net에서 착안하여, ResNeSt는 채널 간 상호작용을 더 세밀하게 담아, 계산 비용을 크게 늘리지 않고도 이미지 분류, 객체 탐지, 시맨틱 세그멘테이션 같은 과제의 표현 학습을 개선한다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
ResNeSt - 분할 어텐션 네트워크
논문: "ResNeSt: Split-Attention Networks" (2020)
핵심: 피처 그룹에 걸쳐 채널별 어텐션을 적용하는 분할 어텐션 블록
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

이 구현은 클래스 3개(`SplitAttention`, `ResNeStBlock`, `ResNeSt`)를 정의하며, 이들이 함께 하나의 컨볼루션 뉴럴 네트워크 구조를 이룬다. 클래스마다 역할이 나뉘어 있어 코드가 모듈화되고 확장하기 쉽다. `forward` 메서드는 PyTorch가 자동 미분에 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`SplitAttention`의 순전파를 따라 텐서의 형태를 추적하여라. 기본 파라메터로 샘플 4개짜리 배치를 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(패딩이 없을 때) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 끝까지 그대로임을 확인하여라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화한 뒤에는 $(B, F)$로 중간 형태를 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하여라.

</div>

??? success "풀이"
    첫 컨볼루션 레이어의 `in_channels`를 현재 값에서 3으로 바꾼다. 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 공간 차원을 다시 계산한다. 첫 선형 레이어의 `in_features`를 마지막 컨볼루션/풀링 레이어의 평탄화된 출력에 맞춘다. `model = SplitAttention(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
셀프 어텐션의 계산 복잡도를 시퀀스 길이 $n$과 모델 차원 $d$의 함수로 설명하여라. 이것이 긴 시퀀스를 위한 Longformer나 Linformer 같은 구조의 동기가 되는 까닭은 무엇인가?

</div>

??? success "풀이"
    표준 셀프 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도는 $O(n^2 d)$, 어텐션 웨이트의 메모리는 $O(n^2)$이다. 시퀀스가 길면($n = 4096$ 등) 감당할 수 없다. Longformer는 국소 슬라이딩 윈도 어텐션($O(n \cdot w \cdot d)$, $w$는 윈도 크기)과 선택한 토큰에 대한 희소한 전역 어텐션을 함께 쓴다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 얼마쯤 내주고 긴 입력에서 효율을 얻는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`SplitAttention`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모두 모듈에 등록하여 옵티마이저가 갱신할 수 있다. `for n in [2, 4, 8]: model = SplitAttention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — ResNeSt

이 구현은 클래스 3개(`SplitAttention`, `ResNeStBlock`, `ResNeSt`)를 정의하며, 이들이 함께 하나의 컨볼루션 뉴럴 네트워크 구조를 이룬다.

핵심 클래스는 `SplitAttention`, `ResNeStBlock`, `ResNeSt`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
