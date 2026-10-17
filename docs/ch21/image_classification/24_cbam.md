# CBAM

CBAM은 채널 어텐션과 공간 어텐션을 순서대로 적용하는 CNN 모듈이다. Woo 외의 ECCV 2018 논문은 피처의 전역 통계에서 채널 게이트를 구하고, 이어서 채널을 요약한 피처에서 공간 게이트를 구한다. 아래 코드는 각 게이트의 형태와 브로드캐스트 연산을 살피는 예제다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
CBAM — 컨볼루션 덩이 어텐션 단원
논문: "CBAM: Convolutional Block Attention Module" (2018)
핵심: 채널 어텐션과 자리 어텐션을 차례로 쓰는 구조
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class CBAM_ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.cbam = CBAMBlock(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.cbam(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = CBAM_ResNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 75,146
```

## 2. 논의

### 채널 어텐션

입력 X의 공간 평균과 공간 최댓값을 각각 N×C×1×1로 구한다. 두 요약을 **같은** 축소·확장 MLP에 통과시킨 뒤 합하고 sigmoid를 취해 채널 게이트 M_c(X)를 얻는다. 코드의 `ChannelAttention.forward`는 변환된 피처가 아니라 게이트를 반환하며 `CBAMBlock`이 X′=X⊙M_c(X)를 계산한다.

두 풀링 경로가 MLP 웨이트를 공유하므로 파라메터 수를 두 배로 세지 않는다. 축소 폭 m=⌊C/r⌋에서 편향 없는 웨이트는 2Cm개다. 양의 r과 유효한 m이 필요하며, 작은 채널에 기본 r을 그대로 사용하는 것은 적절하지 않을 수 있다.

### 공간 어텐션

X′를 채널 축으로 평균·최대 요약해 각각 N×1×H×W를 만든다. 연결한 N×2×H×W에 k×k 컨볼루션과 sigmoid를 적용해 N×1×H×W의 M_s(X′)를 얻는다. 최종 X″=X′⊙M_s(X′)다. 모든 채널이 같은 공간 게이트를 공유한다.

채널 게이트 뒤의 피처에서 공간 게이트를 계산하므로 순서를 바꾸면 일반적으로 다른 연산이다. sigmoid 게이트는 합이 1인 어텐션 확률이 아니며, n×n 위치 쌍을 비교하는 셀프 어텐션과도 다르다. 게이트를 시각화할 수 있지만 객관적인 설명·인과적 중요도를 자동으로 제공하는 것은 아니다.

### 비용과 형태 조건

공간 컨볼루션은 바이어스 없이 2k²개의 웨이트를 갖는다. C=64, r=16, k=7이면 채널 MLP는 512개, 공간 컨볼루션은 98개로 합계 610개다. 하지만 전체 피처의 두 번의 풀링·곱셈과 공간 컨볼루션 비용도 있어 모듈 비용이 웨이트 수 하나로 정해지지는 않는다. 홀수 k에서 padding=k//2가 공간 크기를 유지하며 짝수 k는 같은 형태를 보장하지 않는다.

### 본체의 범위와 확장

`CBAM_ResNet`은 64채널 stem과 풀링 뒤 CBAM 하나를 적용한 분류기다. 이름과 달리 잔차 블록이나 다단계 ResNet backbone은 없다. 입력 64×64에서는 stem 후 32×32, 풀링 후 16×16, CBAM 후 같은 형태이고 전역 풀링 뒤 64차원이다. 현재 fc(64,num_classes)와 맞으며 입력 크기 변경만으로 fc를 공간 크기만큼 바꿀 필요는 없다.

연습의 ModuleList 골격은 `YourBlock` 등이 정의되지 않아 그대로 실행할 수 없다. 추가 블록과 게이트의 삽입 위치, 공유 여부·채널 수를 정하고 forward에서 실제로 호출해야 한다. 채널 게이트만 반복하는 것과 CNN backbone을 깊게 하는 것은 다른 모델이다. 성능은 [원 논문](https://arxiv.org/abs/1807.06521)의 평가와 이 축약 예제를 구분해 해석한다. 코드·출력은 실행하지 않고 보존했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
(4,64,16,16) 입력에서 채널·공간 게이트의 형태를 추적하라.

</div>

??? success "풀이"
    공간 평균·최댓값은 각각 (4,64,1,1)이고 공유 MLP는 64→4→64를 적용한다. 합과 sigmoid 뒤 채널 게이트는 (4,64,1,1)이다. 이를 곱한 피처의 채널 평균·최댓값은 각각 (4,1,16,16)이고 연결 뒤 (4,2,16,16), 7×7 컨볼루션 뒤 공간 게이트는 (4,1,16,16)이다. 최종 피처는 원래 형태를 유지한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
CBAM_ResNet에 64×64 RGB를 입력할 때 fc의 입력 폭을 바꿔야 하는가?

</div>

??? success "풀이"
    아니다. 입력 채널은 이미 3이며 stem과 풀링이 64×16×16 피처를 만든다. CBAM은 그 형태를 유지하고 적응형 풀링은 64차원으로 요약하므로 fc 입력도 64다. ChannelAttention 단독은 게이트를 반환하는 모듈이며 완전한 분류기가 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
C=64, r=16, k=7에서 모듈의 웨이트 수와 셀프 어텐션과의 차이를 설명하라.

</div>

??? success "풀이"
    공유 채널 MLP는 2×64×4=512개이며 두 풀링 경로 때문에 두 배가 되지 않는다. 공간 컨볼루션은 2×7²=98개로 총 610개다. CBAM은 요약 통계에서 채널·위치 게이트를 만들고 n×n의 위치 쌍 어텐션을 계산하지 않는다. 활성 연산·메모리 비용은 별도로 포함한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 일반 ModuleList 골격을 CBAM 모델에 적용할 때 필요한 설계를 설명하라.

</div>

??? success "풀이"

    정의되지 않은 YourBlock·dim을 구체적인 backbone 블록과 폭으로 바꾸고, CBAM의 삽입 위치·채널·forward 호출 순서를 정해야 한다. ModuleList는 등록 컨테이너이며 실행을 자동 수행하지 않는다. 이 골격은 ChannelAttention의 num_layers 인수를 구현하지도 전체 ResNet을 제공하지도 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

CBAM은 공유 MLP의 채널 게이트를 적용한 뒤 공간 게이트를 계산한다. 게이트 반환값과 피처, 두 종류의 풀링 축·브로드캐스트를 구분한다. 작은 웨이트 수나 어텐션이라는 이름만으로 성능·설명 가능성을 보장하지 않는다.
