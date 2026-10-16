# MixNet

MixNet은 깊이별 컨볼루션 켜 하나 안에서 여러 알갱이 크기를 쓰는 깨침을 내놓아, 네트워크가 서로 다른 자리 잣대의 무늬를 한꺼번에 담게 했다. 2019년 글 "MixConv: 섞은 깊이별 컨볼루션 알갱이"에서 나왔으며, MobileNet이 쓰는 깊이별로 가른 컨볼루션의 잘 듦에 기대면서 크기 3, 5, 7의 알갱이를 섞어 결 샘플링을 넉넉하게 한다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
MixNet - 섞은 깊이 방향 컨볼루션 커널
논문: "MixConv: Mixed Depthwise Convolutional Kernels" (2019)
핵심: 깊이 방향 컨볼루션 레이어 하나에 여러 커널 크기를 담는다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class MixConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.groups = len(kernel_sizes)
        assert out_channels % self.groups == 0
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels // self.groups, out_channels // self.groups, 
                     k, padding=k//2, groups=in_channels // self.groups)
            for k in kernel_sizes
        ])
    
    def forward(self, x):
        chunks = torch.chunk(x, self.groups, dim=1)
        outs = [conv(chunk) for conv, chunk in zip(self.convs, chunks)]
        return torch.cat(outs, dim=1)

class MixNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.mixconv = MixConv2d(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = torch.nn.functional.relu6(self.bn1(self.conv1(x)))
        out = torch.nn.functional.relu6(self.bn2(self.mixconv(out)))
        out = self.bn3(self.conv2(out))
        if x.shape == out.shape:
            out = out + x
        return out

class MixNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.blocks = nn.Sequential(
            MixNetBlock(16, 24),
            MixNetBlock(24, 24),
        )
        self.conv_head = nn.Conv2d(24, 1536, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1536)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1536, num_classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu6(self.bn1(self.stem(x)))
        x = self.blocks(x)
        x = torch.nn.functional.relu6(self.bn2(self.conv_head(x)))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = MixNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,596,088
```

## 2. 논의

이 구현은 클래스 3개(`MixConv2d`, `MixNetBlock`, `MixNet`)를 매기고, 이들이 어울려 온전한 컨볼루션 뉴럴 넷트웍 구조를 이룬다. 클래스마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 계산 그림을 매긴다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 증강 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`MixConv2d`의 앞으로 걸음을 따라 텐서의 꼴을 좇아라. 기본 파라메터로 입력 보기 4개를 묶어 넣었을 때, 큰 계산(컨볼루션, 모으기, 선형 켜)마다 꼴이 어떻게 되는지 적어라.

</div>

??? success "연습문제 1 풀이"
    입력의 꼴에서 비롯해 켜를 차례로 건다. `Conv2d(in_c, out_c, k)`마다 자리 차수는 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(덧대기 없이) `padding=k//2`이면 그대로다. 알갱이 2로 모으면 자리 차수가 반이 된다. 선형 켜는 마지막 차수를 바꾼다. 배치 차수는 내내 그대로임을 좇아라. 컨볼루션 켜에서는 $(B, C, H, W)$, 편 뒤에는 $(B, F)$으로 가운데 꼴을 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 그림(입력 꼴: $3 \times 64 \times 64$)을 받도록 고쳐라. 켜의 차수를 모두 그에 맞게 손보고 모델이 어긋남 없이 도는지 따져라.

</div>

??? success "연습문제 2 풀이"
    첫 컨볼루션 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 컨볼루션과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 계산한다. 첫 선형 켜의 `in_features`을 마지막 컨볼루션/모으기 켜의 편 출력에 맞춘다. `model = MixConv2d(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

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
`MixConv2d`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라메터를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 구조를 짜라. 켜 2개, 4개, 8개로 시험하여라.

</div>

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 파라메터를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = MixConv2d(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — MixNet

이 구현은 클래스 3개(`MixConv2d`, `MixNetBlock`, `MixNet`)를 매기고, 이들이 어울려 온전한 컨볼루션 뉴럴 넷트웍 구조를 이룬다.

핵심 클래스는 `MixConv2d`, `MixNetBlock`, `MixNet`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
