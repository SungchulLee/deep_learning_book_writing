# MixNet

MixNet은 깊이별 컨볼루션(depthwise convolution) 레이어 하나 안에서 여러 커널 크기를 쓰는 아이디어를 제시하여, 네트워크가 서로 다른 공간 스케일의 패턴을 한꺼번에 담게 했다. 2019년 논문 "MixConv: Mixed Depthwise Convolutional Kernels"에서 나왔으며, MobileNet이 쓰는 깊이별 분리 컨볼루션의 효율을 바탕으로 크기 3, 5, 7의 커널을 섞어 피처 추출을 풍부하게 한다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
MixNet - 혼합 깊이별 컨볼루션 커널
논문: "MixConv: Mixed Depthwise Convolutional Kernels" (2019)
핵심: 깊이별 컨볼루션 레이어 하나에 여러 커널 크기를 담는다
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

이 구현은 클래스 3개(`MixConv2d`, `MixNetBlock`, `MixNet`)를 정의하며, 이들이 함께 하나의 컨볼루션 뉴럴 네트워크 구조를 이룬다. 클래스마다 역할이 나뉘어 있어 코드가 모듈화되고 확장하기 쉽다. `forward` 메서드는 PyTorch가 자동 미분에 쓰는 계산 그래프를 정의한다.

이 코드는 구조를 이해하기 위한 교육용 구현이다. 실제 학습에서는 데이터와 하드웨어에 맞춰 혼합 정밀도, 분산 학습, 데이터 어그멘테이션 등을 검토할 수 있다. 다만 이런 기법의 적용 조건과 효과는 모델별로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`MixConv2d`의 순전파를 따라 텐서의 형태를 추적하여라. 기본 파라메터로 샘플 4개짜리 배치를 넣었을 때, 주요 연산(컨볼루션, 풀링, 선형 레이어)마다 형태가 어떻게 되는지 적어라.

</div>

??? success "풀이"
    입력의 형태에서 출발해 레이어를 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(패딩이 없을 때) `padding=k//2`이면 그대로다. 커널 크기 2로 풀링하면 공간 차원이 반이 된다. 선형 레이어는 마지막 차원을 바꾼다. 배치 차원은 끝까지 그대로임을 확인하여라. 컨볼루션 레이어에서는 $(B, C, H, W)$, 평탄화한 뒤에는 $(B, F)$로 중간 형태를 적어라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
구조를 크기 $64 \times 64$의 RGB 이미지(입력 형태: $3 \times 64 \times 64$)를 받도록 고쳐라. 레이어의 차원을 모두 그에 맞게 조정하고 모델이 오류 없이 실행되는지 확인하여라.

</div>

??? success "풀이"
    첫 컨볼루션 레이어의 `in_channels`를 현재 값에서 3으로 바꾼다. 컨볼루션과 풀링 레이어마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 공간 차원을 다시 계산한다. 첫 선형 레이어의 `in_features`를 마지막 컨볼루션/풀링 레이어의 평탄화된 출력에 맞춘다. `model = MixConv2d(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 확인한다.

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
`MixConv2d`을 레이어나 블록의 수를 설정할 수 있게 확장하여라. `__init__`에 `num_layers` 인자를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하여라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    하드코딩된 레이어를 다음으로 대체한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 반복한다. 일반 파이썬 리스트가 아니라 `nn.ModuleList`를 써야 PyTorch가 그 안의 파라메터를 모두 모듈에 등록하여 옵티마이저가 갱신할 수 있다. `for n in [2, 4, 8]: model = MixConv2d(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — MixNet

이 구현은 클래스 3개(`MixConv2d`, `MixNetBlock`, `MixNet`)를 정의하며, 이들이 함께 하나의 컨볼루션 뉴럴 네트워크 구조를 이룬다.

핵심 클래스는 `MixConv2d`, `MixNetBlock`, `MixNet`이며 앞의 연습문제 4개로 스스로 확인해 볼 수 있다.
