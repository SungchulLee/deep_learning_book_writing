# SegFormer

SegFormer은 2021년 글 "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"에서 나왔다. 트랜스포머 기반 인코더에 가벼운 MLP 디코더를 붙였고 위치 인코딩은 없다.

여기 짜보기는 SegFormer을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 걸음에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
SegFormer - 뜻 나누기를 위한 단순하고 잘 드는 꾸밈
글: "SegFormer: 변환기로 뜻 나누기를 하는 단순하고 잘 드는 꾸밈" (2021)
지은이: 언쩌 셰 외
핵심: 변환기 바탕 인코더에 가벼운 MLP 디코더를 붙였고,
디코더에는 자리 담기도 엮음도 없다.
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
    단순하게 만든 SegFormer 결의 모형(배우기 위한 갈래)

    - 변환기 꼴 인코더를 엮음 켜로 흉내 낸다
    - 가벼운 MLP 디코더
    - 부록이나 깨침을 잡는 데 알맞다
    """

    def __init__(self, num_classes=21):
        super().__init__()

        # 인코더(알아보기 쉽도록 CNN 바탕으로 단순하게 만듦)
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # 디코더(MLP 머리)
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

        # 자리 차수를 펼친다
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

        # 결을 녹여 아우른다
        fused = f1 + f2 + f3 + f4

        # 나누기 머리
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

이 짜보기는 갈래 2개(`MLP`, `SegFormer`)를 매기고, 이들이 어울려 전체 이미지 나누기 구조를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 계산 그림을 매긴다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 증강 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`MLP`의 앞으로 걸음을 따라 텐서의 꼴을 좇아라. 기본 매개변수로 입력 보기 4개를 묶어 넣었을 때, 큰 계산(엮음, 모으기, 선형 켜)마다 꼴이 어떻게 되는지 적어라.

</div>

??? success "연습문제 1 풀이"
    입력의 꼴에서 비롯해 켜를 차례로 건다. `Conv2d(in_c, out_c, k)`마다 자리 차수는 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(덧대기 없이) `padding=k//2`이면 그대로다. 알갱이 2로 모으면 자리 차수가 반이 된다. 선형 켜는 마지막 차수를 바꾼다. 배치 차수는 내내 그대로임을 좇아라. 엮음 켜에서는 $(B, C, H, W)$, 편 뒤에는 $(B, F)$으로 가운데 꼴을 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 그림(입력 꼴: $3 \times 64 \times 64$)을 받도록 고쳐라. 켜의 차수를 모두 그에 맞게 손보고 모델이 어긋남 없이 도는지 따져라.

</div>

??? success "연습문제 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 계산한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 출력에 맞춘다. `model = MLP(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력과 출력의 차수가 같을 때 여느 엮음과 깊이별로 가른 엮음의 매개변수 수와 뜨는 계산 횟수를 비교하여라. 계산을 가장 크게 아끼는 때는 언제인가?

</div>

??? success "연습문제 3 풀이"
    여느 `Conv2d(C_in, C_out, k)`의 매개변수는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별로 가른 엮음은 이를 (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 갈래마다 거르개 하나)와 (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 엮음)로 나눈다. 매개변수의 비교는 어림잡아 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$이 모두 클 때 가장 크게 아낀다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`MLP`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = MLP(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — SegFormer

이 짜보기는 갈래 2개(`MLP`, `SegFormer`)를 매기고, 이들이 어울려 전체 이미지 나누기 구조를 이룬다.

핵심 갈래는 `MLP`, `SegFormer`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
