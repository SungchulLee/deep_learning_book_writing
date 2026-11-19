# SegFormer

SegFormer은 2021년 논문 "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"에서 나왔다. 트랜스포머 기반 인코더에 가벼운 MLP 디코더를 붙였고 위치 인코딩은 없다.

여기 구현은 SegFormer을 짧고 학습하기 좋게 보인 예제다. 코드는 핵심 구조와 순전파(forward pass)에 초점을 두어, 핵심 구성 요소를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
SegFormer - 시맨틱 분할을 위한 단순하고 효율적인 설계
논문: "SegFormer: 트랜스포머를 쓴 시맨틱 분할의 단순하고 효율적인 설계" (2021)
지은이: 언쩌 셰 외
핵심: 트랜스포머 기반 인코더에 가벼운 MLP 디코더를 붙였고,
디코더에는 위치 인코딩도 컨볼루션도 없다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class MLP(nn.Module):
    """SegFormer 디코더에서 쓰는 단순한 MLP"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


class SegFormer(nn.Module):
    """
    단순화한 SegFormer 스타일 모델(교육용 버전)

    - 트랜스포머 방식 인코더를 컨볼루션 레이어로 모사한다
    - 가벼운 MLP 디코더
    - 부록이나 개념 이해에 알맞다
    """

    def __init__(self, num_classes=21):
        super().__init__()

        # 인코더(이해하기 쉽도록 CNN 기반으로 단순화함)
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # 디코더(MLP 헤드)
        self.mlp1 = MLP(64, 256)
        self.mlp2 = MLP(128, 256)
        self.mlp3 = MLP(256, 256)
        self.mlp4 = MLP(512, 256)

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # 인코더
        f1 = F.relu(self.enc1(x))              # (B, 64, H, W)
        f2 = F.relu(self.enc2(f1))             # (B, 128, H, W)
        f3 = F.relu(self.enc3(f2))             # (B, 256, H, W)
        f4 = F.relu(self.enc4(f3))             # (B, 512, H, W)

        # 공간 차원을 평탄화한다
        def mlp_process(f, mlp):
            B, C, H, W = f.shape
            f = f.flatten(2).transpose(1, 2)   # (B, HW, C)
            f = mlp(f)
            f = f.transpose(1, 2).reshape(B, -1, H, W)
            return f

        f1 = mlp_process(f1, self.mlp1)
        f2 = mlp_process(f2, self.mlp2)
        f3 = mlp_process(f3, self.mlp3)
        f4 = mlp_process(f4, self.mlp4)

        # 피처를 융합한다
        fused = f1 + f2 + f3 + f4

        # 분할 헤드
        out = self.classifier(fused)
        return out


if __name__ == "__main__":
    model = SegFormer(num_classes=19)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)  # (1, 19, 224, 224)
```

**출력:**

```
torch.Size([1, 19, 224, 224])
```

## 2. 논의

이 구현은 클래스 2개(`MLP`, `SegFormer`)를 정의하고, 이들이 함께 완전한 이미지 분할 구조를 이룬다. 클래스마다 고유한 역할을 맡아 코드를 모듈화하고 확장하기 쉽게 한다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`MLP`의 순전파를 따라 텐서의 형태를 추적하라. 기본 파라메터로 입력 표본 4개의 배치를 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(패딩 없이) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임을 확인하라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화(flatten) 뒤에는 $(B, F)$으로 중간 형태를 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하라.

</div>

??? success "풀이"
    첫 컨볼루션 레이어의 `in_channels`을 현재 값에서 3으로 바꾼다. 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 공간 차원을 다시 계산한다. 첫 선형 레이어의 `in_features`을 마지막 컨볼루션/풀링 레이어의 평탄화된 출력에 맞춘다. `model = MLP(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력과 출력의 차원이 같을 때 표준 컨볼루션과 깊이별 분리 컨볼루션(depthwise separable convolution)의 파라메터 수와 부동소수점 연산 횟수(FLOPs)를 비교하여라. 계산이 가장 크게 절약되는 때는 언제인가?

</div>

??? success "풀이"
    표준 `Conv2d(C_in, C_out, k)`의 파라메터는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별 분리 컨볼루션은 이를 (1) 깊이별(depthwise): $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나)와 (2) 점별(pointwise): $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 컨볼루션)로 나눈다. 파라메터 수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 파라메터가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$이 모두 클 때 가장 크게 절약된다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`MLP`을 레이어나 블록의 수를 설정할 수 있게 확장하라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 만들어라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`을 써야 PyTorch가 모든 파라메터를 옵티마이저에 등록한다. `for n in [2, 4, 8]: model = MLP(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — SegFormer

이 구현은 클래스 2개(`MLP`, `SegFormer`)를 정의하고, 이들이 함께 완전한 이미지 분할 구조를 이룬다.

핵심 클래스는 `MLP`, `SegFormer`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
