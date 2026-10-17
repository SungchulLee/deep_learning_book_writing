# CoordConv

CoordConv는 피처에 좌표 채널을 추가한 뒤 컨볼루션을 적용한다. 위치를 직접 제공해 절대 위치에 의존하는 함수를 더 쉽게 표현하도록 하는 구조다. 좌표를 사용하지 않는 일반 컨볼루션과 다른 귀납적 바이어스며, 모든 이미지 분류 문제에서 유리하다는 보장은 없다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
CoordConv — 컨볼루션 뉴럴 넷트웍에 자리표 더하기
논문: "An Intriguing Failing of Convolutional Neural Networks" (2018)
핵심: 자리표 채널을 더해 자리에 대한 따짐을 돕는다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r
    
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # 좌표 격자를 만든다
        xx_channel = torch.arange(width, dtype=x.dtype, device=x.device).repeat(1, height, 1)
        yy_channel = torch.arange(height, dtype=x.dtype, device=x.device).repeat(1, width, 1).transpose(1, 2)
        
        xx_channel = xx_channel / (width - 1)
        yy_channel = yy_channel / (height - 1)
        
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)
        
        ret = torch.cat([x, xx_channel, yy_channel], dim=1)
        
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            ret = torch.cat([ret, rr], dim=1)
        
        return ret

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x

class CoordConvNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = CoordConv(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.conv2 = CoordConv(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = CoordConvNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 221,288
```

## 2. 논의

### 좌표의 정의와 형태

폭 W의 열 인덱스 j와 높이 H의 행 인덱스 i에서 xⱼ=2j/(W−1)−1, yᵢ=2i/(H−1)−1을 정의한다. 같은 좌표 지도를 배치의 모든 표본에 붙이면 N×C×H×W가 N×(C+2)×H×W가 된다. `with_r=True`이면 r=√(x²+y²)를 더해 C+3채널이다.

r은 정규화된 영상 중심으로부터의 거리다. 모서리는 √2이고 중심의 위치가 실제 격자점에 있어야 0을 취한다. 임의의 학습 가능한 위치 임베딩이나 물체 중심으로부터의 거리가 아니다. `AddCoords` 자체에는 학습 파라메터가 없다.

코드는 좌표를 입력의 장치·자료형에 맞춰 생성하지만 H=1 또는 W=1이면 분모가 0이다. 유한한 부동소수점 입력과 유효한 공간 크기가 필요하며, 정수 입력이나 아주 작은 피처 지도로 일반화하는 별도 처리가 없다.

### 좌표 채널의 비용

그룹 없는 표준 k×k 컨볼루션에서 좌표 두 채널을 추가하면 웨이트는 2C_outk²개 늘고, 반지름까지 추가하면 3C_outk²개 늘어난다. 바이어스 수는 출력 폭이 같으면 변하지 않는다. 좌표 지도 생성·배치 복제·연결과 공간별 연산 비용은 별도다.

이 본체의 첫 3→64, 7×7 CoordConv는 내부적으로 5→64다. 두 번째 64→128, 3×3은 66→128이다. 일반 컨볼루션 두 개와 비교한 추가 웨이트는 2×64×49+2×128×9=8,576개다. 서로 다른 해상도에서 좌표 지도를 새로 생성하므로 두 지도의 간격은 다르지만 각각 자신의 격자에서 [−1,1] 범위를 쓴다.

### 위치 의존성과 전처리

고정 격자의 좌표를 제공하면 내용만 이동할 때 좌표와의 관계가 달라져 일반적인 이동 등변성을 깨고 위치에 의존하는 출력을 만들 수 있다. 반대로 좌표 웨이트를 사용하지 않으면 일반 컨볼루션의 연산을 포함할 수 있다. 추가 채널이 자동으로 위치 불변성을 보장하거나 모든 위치 외삽을 해결하는 것은 아니다. 일반 CNN도 padding·경계 등으로 위치 정보를 얻을 수 있으므로 좌표가 없으면 어떤 위치도 학습하지 못한다고 설명하지 않는다.

크롭·회전·리사이즈를 사용할 때 좌표가 변환된 영상 안의 상대 위치인지 원 영상의 절대 위치인지 정해야 한다. 현재 코드는 각 입력 격자에서 좌표를 다시 만든다. 좌표가 의미하는 기준이 평가와 데이터 어그멘테이션에서 일관되어야 한다.

### 본체와 반복의 범위

64×64 RGB 입력은 첫 CoordConv 뒤 (N,64,32,32), 최대 풀링 뒤 (N,64,16,16), 두 번째 CoordConv 뒤 (N,128,16,16)이다. 전역 풀링 후 N×128이므로 분류기는 공간 크기를 직접 펼치지 않는다. `AddCoords` 단독은 피처 생성 모듈이지 분류 모델이 아니다.

연습의 ModuleList는 정의되지 않은 일반 골격이다. AddCoords를 반복하면 같은 좌표가 여러 번 붙을 뿐 학습 가능한 깊은 피처 변환이 생기지 않는다. 반복할 CoordConv 블록의 채널과 좌표 생성 시점을 구체적으로 정해야 한다. 배경과 실험은 [원 논문](https://arxiv.org/abs/1807.03247)을 참고한다. 코드·출력은 실행하지 않고 유지했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 (4,3,64,64)에 좌표를 붙인 형태와 반지름의 범위를 설명하라.

</div>

??? success "연습문제 1 풀이"
    기본 출력은 (4,5,64,64), with_r=True이면 (4,6,64,64)다. 좌표 x,y는 각 축에서 −1부터1이고 모서리 반지름은√2다. 짝수64 격자에는 정확한 중심점이 없어 r=0인 점이 없다. AddCoords에는 학습 파라메터가 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 입력에서 본체의 fc 입력 폭과 공간 크기 1의 문제를 설명하라.

</div>

??? success "연습문제 2 풀이"
    두 CoordConv와 풀링 뒤 128×16×16이며 전역 평균은128차원이므로 fc 입력을 공간 크기에 맞춰 바꿀 필요는 없다. 그러나 H=1 또는 W=1인 피처에 AddCoords를 적용하면 좌표식의 분모가0이다. 적응형 풀링 지원과 좌표 생성의 유효 범위는 별도 조건이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
일반 컨볼루션 대비 좌표 채널의 추가 웨이트를 계산하라.

</div>

??? success "연습문제 3 풀이"
    두 좌표는 2C_outk², 반지름 포함은3C_outk²개의 웨이트를 더한다. 이 본체의 두 레이어에서는 6,272+2,304=8,576개다. 바이어스 수는 변하지 않으며 좌표 생성·복제와 공간 연산의 비용은 별도로 든다. 이는 depthwise 분리 컨볼루션의 절감 공식과 다른 비교다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 목록 골격으로 AddCoords만 여러 번 적용하는 것이 깊은 CoordConv 모델인가?

</div>

??? success "연습문제 4 풀이"

    아니다. AddCoords는 결정론적 좌표를 붙이고 학습 웨이트가 없다. 반복하면 중복 좌표 채널만 늘어난다. CoordConv와 활성화 등의 실제 변환을 구성하고 각 블록이 받는 채널·해상도를 맞춰야 한다. YourBlock·dim이 정의되지 않은 골격은 이런 구조를 구현하지 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

CoordConv는 명시적인 좌표로 위치 의존 함수를 표현하게 한다. 좌표 기준·이동 성질·추가 비용을 구분하고, 공간 크기1의 나눗셈과 입력 채널 추적을 확인한다. 좌표 추가만 반복하는 것은 깊은 학습 모델이 아니다.
