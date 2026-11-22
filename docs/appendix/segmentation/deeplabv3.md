# DeepLabV3

DeepLabV3는 2017년 논문 "Rethinking Atrous Convolution for Semantic Image Segmentation"에서 나왔다. 아트러스 공간 피라미드 풀링(ASPP)과 여러 팽창률(dilation rate)을 쓴다. 아트러스 컨볼루션(팽창 컨볼루션)은 필터 원소 사이를 팽창률만큼 띄워, 파라메터 수를 늘리지 않고 수용 영역을 넓히는 컨볼루션이다. ASPP는 팽창률이 서로 다른 아트러스 컨볼루션과 전역 평균 풀링을 나란히 적용해 여러 스케일의 문맥을 한꺼번에 모은다.

여기 구현은 DeepLabV3를 짧고 트레인하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
DeepLabV3 - 시맨틱 분할을 위한 아트러스(팽창) 컨볼루션 다시 보기
논문: "Rethinking Atrous Convolution for Semantic Image Segmentation" (2017)
핵심: 아트러스 공간 피라미드 풀링(ASPP), 여러 팽창률
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = torch.nn.functional.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.conv_out(x)

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.aspp = ASPP(64, 256)
        self.classifier = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x):
        size = x.shape[2:]
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.classifier(x)
        return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

if __name__ == "__main__":
    model = DeepLabV3()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 820,821
```

## 2. 논의

이 구현은 클래스 2개(`ASPP`, `DeepLabV3`)를 정의하고, 이들이 함께 이미지 분할 구조를 이룬다. 다만 백본은 ResNet이 아니라 컨볼루션 하나와 최대 풀링으로 줄였고, `ASPP`는 팽창률 6, 12, 18인 $3 \times 3$ 브랜치 셋과 $1 \times 1$ 브랜치, 전역 평균 풀링 브랜치를 이어 붙인 원래 구성을 그대로 따른다. 분류 레이어가 낸 1/4 해상도의 점수 맵은 쌍선형 보간으로 입력 크기까지 키운다. 클래스마다 고유한 역할을 맡아 코드를 모듈화하고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 트레인에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 트레인, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`ASPP`의 순전파를 따라 텐서의 형태를 추적하라. 기본 파라메터로 입력 표본 4개의 배치를 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$로 바뀌고(패딩 없이) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임을 확인하라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화(flatten) 뒤에는 $(B, F)$로 중간 형태를 적어라. `ASPP`의 아트러스 브랜치는 패딩을 팽창률과 같게 두어 공간 크기를 유지한다. `DeepLabV3` 안에서처럼 `ASPP(64, 256)`에 $(4, 64, H, W)$를 넣으면 다섯 브랜치가 저마다 $(4, 256, H, W)$를 내고(전역 평균 풀링 브랜치는 $(4, 256, 1, 1)$을 보간해 키운다), 이어 붙이면 $(4, 1280, H, W)$, `conv_out` 뒤에 $(4, 256, H, W)$다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하라.

</div>

??? success "풀이"
    `DeepLabV3`의 첫 컨볼루션 레이어는 이미 입력 채널이 3이다. 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$로 공간 차원을 다시 계산하면 $64 \to 32 \to 16$이고, `ASPP`는 $16 \times 16$을 유지하며, 마지막 보간이 출력을 입력 크기 $64 \times 64$로 되돌린다. 선형 레이어가 없으므로 고칠 차원은 없다. 다만 $16 \times 16$ 맵에서 팽창률 18인 $3 \times 3$ 필터는 바깥 원소가 모두 패딩에 떨어져 사실상 $1 \times 1$ 컨볼루션처럼 동작한다. 작은 입력에서는 팽창률을 줄이는 편이 낫다. `model = ASPP(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`의 `ASPP` 자리에는 `DeepLabV3()`를 넣어 확인한다. 배치 하나로 트레인 모드에서 돌리면 전역 평균 풀링 브랜치의 배치 정규화가 값 하나로 통계를 낼 수 없어 오류가 나므로, `model.eval()`을 먼저 부르거나 배치를 둘 이상으로 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력과 출력의 차원이 같을 때 표준 컨볼루션과 깊이별 분리 컨볼루션(depthwise separable convolution)의 파라메터 수와 부동소수점 연산 횟수(FLOPs)를 비교하여라. 계산이 가장 크게 절약되는 때는 언제인가?

</div>

??? success "풀이"
    표준 `Conv2d(C_in, C_out, k)`의 파라메터는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별 분리 컨볼루션은 이를 (1) 깊이별(depthwise): $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나)와 (2) 점별(pointwise): $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 컨볼루션)로 나눈다. 파라메터 수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 파라메터가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크게 절약된다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`ASPP`를 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모두 모듈에 등록하여 옵티마이저가 갱신할 수 있다. `for n in [2, 4, 8]: model = ASPP(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — DeepLabV3

이 구현은 클래스 2개(`ASPP`, `DeepLabV3`)를 정의하고, 이들이 함께 단순화한 이미지 분할 구조를 이룬다.

핵심 클래스는 `ASPP`, `DeepLabV3`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
