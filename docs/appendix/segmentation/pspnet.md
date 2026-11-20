# PSPNet

PSPNet은 2017년 논문 "Pyramid Scene Parsing Network"에서 나왔다. 여러 스케일의 문맥을 모으는 피라미드 풀링 모듈을 쓴다. 피처 맵을 $1 \times 1$, $2 \times 2$, $3 \times 3$, $6 \times 6$ 격자로 평균 풀링해 각각 요약한 뒤, 원래 크기로 업샘플링해 원래 피처 맵에 이어 붙이는 모듈이다.

여기 구현은 PSPNet을 짧고 학습하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
PSPNet - 피라미드 장면 파싱 네트워크
논문: "Pyramid Scene Parsing Network" (2017)
핵심: 여러 스케일의 문맥을 모으는 피라미드 풀링 모듈
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])
    
    def forward(self, x):
        h, w = x.shape[2:]
        out = [x]
        for stage in self.stages:
            out.append(torch.nn.functional.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True))
        return torch.cat(out, dim=1)

class PSPNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.ppm = PyramidPooling(64)
        
        self.final = nn.Sequential(
            nn.Conv2d(64 + 64, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, 1)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        x = self.conv1(x)
        x = self.ppm(x)
        x = self.final(x)
        return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

if __name__ == "__main__":
    model = PSPNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 607,701
```

## 2. 논의

이 구현은 클래스 2개(`PyramidPooling`, `PSPNet`)를 정의하고, 이들이 함께 이미지 분할 구조를 이룬다. 백본은 ResNet이 아니라 스트라이드 2인 컨볼루션 하나로 줄였다. 피라미드 풀링의 네 브랜치는 저마다 채널을 $64 / 4 = 16$으로 줄이므로, 원래 64채널과 이어 붙이면 128채널이 되고 `final`의 첫 컨볼루션이 이를 받는다. 클래스마다 고유한 역할을 맡아 코드를 모듈화하고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`PyramidPooling`의 순전파를 따라 텐서의 형태를 추적하라. 기본 파라메터로 입력 표본 4개의 배치를 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$로 바뀌고(패딩 없이) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임을 확인하라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화(flatten) 뒤에는 $(B, F)$로 중간 형태를 적어라. `PSPNet` 안에서처럼 `PyramidPooling(64)`에 $(4, 64, H, W)$를 넣으면 네 브랜치가 $(4, 16, 1, 1)$, $(4, 16, 2, 2)$, $(4, 16, 3, 3)$, $(4, 16, 6, 6)$을 내고, 저마다 $(4, 16, H, W)$로 보간된다. 입력과 이어 붙이면 $(4, 128, H, W)$다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하라.

</div>

??? success "풀이"
    `PSPNet`의 첫 컨볼루션 레이어는 이미 입력 채널이 3이다. $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$로 공간 차원을 다시 계산하면 $64 \to 32$이고, 피라미드 풀링은 적응형 풀링이라 어떤 크기에서도 $1, 2, 3, 6$ 격자를 만들며, 마지막 보간이 출력을 $64 \times 64$로 되돌린다. 선형 레이어가 없으므로 고칠 차원은 없다. `model = PyramidPooling(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`의 `PyramidPooling` 자리에는 `PSPNet()`을 넣어 확인한다. 배치 하나로 학습 모드에서 돌리면 $1 \times 1$ 브랜치의 배치 정규화가 값 하나로 통계를 낼 수 없어 오류가 나므로, `model.eval()`을 먼저 부르거나 배치를 둘 이상으로 한다.

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
`PyramidPooling`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모두 모듈에 등록하여 옵티마이저가 갱신할 수 있다. `for n in [2, 4, 8]: model = PyramidPooling(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — PSPNet

이 구현은 클래스 2개(`PyramidPooling`, `PSPNet`)를 정의하고, 이들이 함께 단순화한 이미지 분할 구조를 이룬다.

핵심 클래스는 `PyramidPooling`, `PSPNet`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
