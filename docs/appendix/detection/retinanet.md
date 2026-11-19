# RetinaNet

RetinaNet은 2017년 논문 "Focal Loss for Dense Object Detection"에서 나왔다. 클래스 불균형을 다루는 초점 로스(focal loss)와 피처 피라미드 네트워크(FPN)를 쓴다.

여기 구현은 RetinaNet을 짧게 줄인 학습용 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
RetinaNet - 밀집 객체 탐지를 위한 초점 로스
논문: "Focal Loss for Dense Object Detection" (2017)
핵심: 클래스 불균형을 다루는 초점 로스와 피처 피라미드 네트워크(FPN)
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        # 상향식(bottom-up) 경로
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # 측면 연결(lateral connection)
        self.lateral3 = nn.Conv2d(64, 256, 1)
        
        # 하향식(top-down) 경로
        self.smooth = nn.Conv2d(256, 256, 3, 1, 1)
    
    def forward(self, x):
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)
        
        # 측면 연결(lateral connection)
        p3 = self.lateral3(c1)
        
        # 스무딩
        p3 = self.smooth(p3)
        
        return [p3]

class RetinaNet(nn.Module):
    def __init__(self, num_classes=80, num_anchors=9):
        super().__init__()
        self.fpn = FPN()
        
        # 분류 서브네트워크
        self.cls_subnet = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * num_classes, 3, 1, 1)
        )
        
        # 상자 회귀 서브네트워크
        self.box_subnet = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors * 4, 3, 1, 1)
        )
    
    def forward(self, x):
        features = self.fpn(x)
        
        cls_outputs = []
        box_outputs = []
        
        for feat in features:
            cls_outputs.append(self.cls_subnet(feat))
            box_outputs.append(self.box_subnet(feat))
        
        return {'classifications': cls_outputs, 'regressions': box_outputs}

if __name__ == "__main__":
    model = RetinaNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 4,719,156
```

## 2. 논의

이 구현은 클래스 2개(`FPN`, `RetinaNet`)를 정의하고, 이들이 함께 완전한 객체 탐지 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈로 나누고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분이 쓰는 계산 그래프를 정의한다.

로스 계산은 모델의 출력을 최적화의 목표와 잇는다. 알맞은 로스 함수를 고르는 일은 몹시 중요하다. 모델이 무엇을 최적화하도록 학습할지를 정하고, 학습된 표현과 결정 경계를 직접 좌우하기 때문이다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`FPN`의 순전파를 따라 텐서의 형태를 추적하라. 기본 파라메터로 입력 샘플 4개를 한 배치로 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(패딩 없이) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임을 확인하라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화한 뒤에는 $(B, F)$으로 중간 형태를 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)을 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하라.

</div>

??? success "풀이"
    첫 컨볼루션 레이어의 `in_channels`을 현재 값에서 3으로 바꾼다. 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 공간 차원을 다시 계산한다. 첫 선형 레이어의 `in_features`을 마지막 컨볼루션/풀링 레이어의 평탄화된 출력에 맞춘다. `model = FPN(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력과 출력의 차원이 같을 때 일반 컨볼루션과 깊이별 분리 컨볼루션(depthwise separable convolution)의 파라메터 수와 부동소수점 연산 횟수(FLOPs)를 비교하여라. 계산이 가장 크게 절약되는 때는 언제인가?

</div>

??? success "풀이"
    일반 `Conv2d(C_in, C_out, k)`의 파라메터는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별 분리 컨볼루션은 이를 (1) 깊이별(depthwise): $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나)와 (2) 점별(pointwise): $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 컨볼루션)로 나눈다. 파라메터 수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 파라메터가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$이 모두 클 때 가장 크게 절약된다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`FPN`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 파라메터를 추가하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 구성하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 최적화 대상으로 등록한다. `for n in [2, 4, 8]: model = FPN(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — RetinaNet

이 구현은 클래스 2개(`FPN`, `RetinaNet`)를 정의하고, 이들이 함께 완전한 객체 탐지 구조를 이룬다.

핵심 클래스는 `FPN`, `RetinaNet`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
