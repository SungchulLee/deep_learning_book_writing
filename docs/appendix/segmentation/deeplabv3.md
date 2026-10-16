# DeepLabV3

DeepLabV3은 2017년 글 "Rethinking Atrous Convolution for Semantic Image Segmentation"에서 나왔다. 구멍 난 자리 두겁 모으기(ASPP)와 여러 벌림 비율을 쓴다.

여기 구현은 DeepLabV3을 짧고 학습하기 좋게 보인 본이다. 코드는 핵심 구조와 앞으로 스텝에 마음을 두어, 핵심 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
DeepLabV3 - 뜻 나누기를 위한 성긴 컨볼루션 다시 보기
논문: "Rethinking Atrous Convolution for Semantic Image Segmentation" (2017)
핵심: 성긴 자리 피라미드 모으기(ASPP), 여러 벌림 비율
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

이 구현은 클래스 2개(`ASPP`, `DeepLabV3`)를 매기고, 이들이 어울려 전체 이미지 나누기 구조를 이룬다. 클래스마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 계산 그림을 매긴다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 증강 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`ASPP`의 앞으로 스텝을 따라 텐서의 꼴을 좇아라. 기본 파라메터로 입력 보기 4개를 묶어 넣었을 때, 큰 계산(컨볼루션, 모으기, 선형 켜)마다 꼴이 어떻게 되는지 적어라.

</div>

??? success "연습문제 1 풀이"
    입력의 꼴에서 비롯해 켜를 차례로 건다. `Conv2d(in_c, out_c, k)`마다 자리 차수는 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(덧대기 없이) `padding=k//2`이면 그대로다. 알갱이 2로 모으면 자리 차수가 반이 된다. 선형 켜는 마지막 차수를 바꾼다. 배치 차수는 내내 그대로임을 좇아라. 컨볼루션 켜에서는 $(B, C, H, W)$, 편 뒤에는 $(B, F)$으로 가운데 꼴을 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 그림(입력 꼴: $3 \times 64 \times 64$)을 받도록 고쳐라. 켜의 차수를 모두 그에 맞게 손보고 모델이 어긋남 없이 도는지 따져라.

</div>

??? success "연습문제 2 풀이"
    첫 컨볼루션 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 컨볼루션과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 계산한다. 첫 선형 켜의 `in_features`을 마지막 컨볼루션/모으기 켜의 편 출력에 맞춘다. `model = ASPP(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
입력과 출력의 차수가 같을 때 여느 컨볼루션과 깊이별로 가른 컨볼루션의 파라메터 수와 뜨는 계산 횟수를 비교하여라. 계산을 가장 크게 아끼는 때는 언제인가?

</div>

??? success "연습문제 3 풀이"
    여느 `Conv2d(C_in, C_out, k)`의 파라메터는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별로 가른 컨볼루션은 이를 (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 클래스마다 거르개 하나)와 (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 컨볼루션)로 나눈다. 파라메터의 비교는 어림잡아 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 파라메터가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$이 모두 클 때 가장 크게 아낀다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`ASPP`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라메터를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = ASPP(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — DeepLabV3

이 구현은 클래스 2개(`ASPP`, `DeepLabV3`)를 매기고, 이들이 어울려 전체 이미지 나누기 구조를 이룬다.

핵심 클래스는 `ASPP`, `DeepLabV3`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
