# MixNet

MixConv는 채널을 여러 그룹으로 나누고 서로 다른 크기의 depthwise 커널을 적용하는 연산이다. Tan과 Le의 2019년 논문은 이를 사용한 MixNet을 제안했다. 같은 입력 전체에 모든 커널을 적용하는 방식과 달리, 각 채널 부분집합이 서로 다른 공간 범위를 처리한다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
MixNet — 섞은 깊이별 컨볼루션 커널
논문: "MixConv: Mixed Depthwise Convolutional Kernels" (2019)
핵심: 깊이별 컨볼루션 레이어 하나에 커널 크기를 여럿 쓴다
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

### 채널 분할과 혼합 커널

같은 입출력 폭 C를 g개로 균등 분할하고 커널 k₁,…,k_g를 사용하면 각 분기는 C/g채널에 depthwise를 적용한다. 연결한 결과는 원래 C채널이다. 웨이트 수는 ∑ⱼ(C/g)kⱼ²이며 코드에는 분기별 출력 바이어스 C개도 있다. 홀수 커널과 padding=k//2에서는 공간 크기를 유지한다.

채널별 공간 연산만으로 서로 다른 그룹이 직접 섞이지는 않는다. `MixNetBlock`의 앞뒤 1×1 컨볼루션이 채널 혼합을 수행한다. 커널 크기는 학습으로 바꾸는 선택값이 아니라 구성 시 고정한 구조다. 여러 크기를 쓴다는 사실만으로 입력마다 커널 하나를 선택하거나 모든 채널이 모든 크기를 보는 것은 아니다.

### 일반 인수의 제약

코드는 출력 채널이 커널 개수로 나누어지는지만 assert한다. 입력도 균등 분할과 각 Conv2d의 고정 입력 폭에 맞아야 한다. C_in/g가 양수여야 하고 각 출력 폭은 depthwise 그룹 수의 배수여야 한다. `torch.chunk`가 균등한 기대 폭을 만들지 않으면 분기 입력과 맞지 않을 수 있다. 비어 있는 커널 목록·짝수 커널·작은 채널 등의 경우도 별도 조건이 필요하다.

기본 본체에서는 확장 폭 16×6=96과 24×6=144가 세 분기로 나누어지므로 이 조건을 맞춘다. 이를 임의의 입출력 인수에서도 일반적으로 안전하다는 근거로 삼지는 않는다.

### 블록과 본체의 형태

블록은 1×1 확장→MixConv→1×1 투영이며 최종 투영 뒤 비선형 활성화를 생략한다. 입력과 출력의 전체 형태가 같을 때만 잔차를 더한다. 본체의 첫 16→24블록에는 덧셈이 없고 둘째 24→24블록에는 덧셈이 있다.

64×64 RGB 입력은 stem 뒤 16×32×32가 되고 두 블록 모두 stride1이라 공간을 유지한다. head가 24→1536채널로 바꾸고 전역 풀링 뒤 N×1536을 fc에 전달한다. 분류기 입력은 공간 크기와 무관하게 채널 수로 정해진다. 이 예제는 두 블록만 사용하고 전체 MixNet의 탐색된 단계·SE·스케일링을 재현하지 않는다.

### 비용 비교와 확장

C=96을 3·5·7 커널에 32채널씩 나누면 depthwise 웨이트는 32(9+25+49)=2,656개이고 바이어스는 96개다. 모든 채널에 3×3만 쓰는 864개보다 크지만 7×7만 쓰는 4,704개보다 적다. 표준 컨볼루션이나 전체 확장 블록과 비교할 때는 앞뒤 1×1·BN도 포함해야 한다.

연습의 ModuleList는 정의되지 않은 일반 골격이다. 반복 블록을 구체화하고 채널 분할·잔차·forward를 맞춰야 한다. 실제 지연시간은 다중 커널 호출과 메모리 접근에 영향을 받으므로 웨이트 절감 비와 동일하지 않다. [원 논문](https://arxiv.org/abs/1907.09595)의 전체 모델과 이 예제의 범위를 구분한다. 코드·출력은 실행하지 않고 유지했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 (4,96,32,32), 커널 [3,5,7]인 동일 폭 MixConv의 형태를 추적하라.

</div>

??? success "풀이"
    32채널씩 세 부분으로 나누고 각 부분에 해당 depthwise 커널을 적용한다. 홀수 커널과 같은 크기 패딩으로 분기별 (4,32,32,32)를 유지하며 연결 후 (4,96,32,32)다. 이 연산 자체에 채널 혼합 1×1은 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 입력에서 본체의 두 블록과 fc 입력을 추적하라.

</div>

??? success "풀이"
    stem 뒤 (N,16,32,32), 첫 블록 뒤 (N,24,32,32), 둘째도 같은 형태다. 첫 블록은 채널이 달라 잔차가 없고 둘째는 있다. head 뒤 1536채널, 전역 풀링 뒤 N×1536이므로 기존 fc 폭과 맞는다. 입력 채널은 이미 3이며 fc를 공간 크기에 맞춰 변경할 필요는 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
C=96의 혼합 depthwise 웨이트 수를 단일 3×3·7×7과 비교하라.

</div>

??? success "풀이"
    혼합은 32×(3²+5²+7²)=2,656개, 단일 3×3은 96×9=864개, 7×7은 96×49=4,704개다. 코드의 바이어스 96개는 별도다. 큰 커널의 비용과 여러 공간 범위를 나눠 표현하는 절충이며 표준 컨볼루션의 C² 웨이트 식과 다르다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 반복 골격으로 MixNet을 확장할 때 필요한 제약과 모듈 역할을 설명하라.

</div>

??? success "풀이"

    YourBlock·dim 등이 정의되지 않은 골격이다. 실제 MixNetBlock을 정하고 확장 채널이 분기 수로 나누어지는지, 각 depthwise의 입출력 배수가 맞는지 확인한다. 형태가 같을 때의 잔차와 forward 순회를 구성해야 한다. ModuleList 등록만으로 깊이 조절 가능한 MixConv2d가 완성되지는 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

MixConv는 채널 부분집합마다 다른 depthwise 커널을 적용한다. 입력·출력 분할과 그룹 배수, 홀수 커널 패딩을 확인하고, 전체 블록의 1×1 비용과 축약 본체의 범위를 함께 해석한다.
