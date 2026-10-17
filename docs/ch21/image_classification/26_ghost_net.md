# GhostNet

GhostNet은 일부 기본 피처를 표준 컨볼루션으로 만든 뒤 값싼 변환으로 추가 피처를 생성하는 CNN이다. Ghost 모듈은 모든 출력 채널을 독립적인 표준 컨볼루션으로 계산하는 대신 기존 피처를 활용한다. 아래 구현에는 ratio=1 경로의 문제가 있어 전체 모델의 완성된 실행 예제로 읽으면 안 된다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
GhostNet — 값싼 연산으로 피처 더 만들기
논문: "GhostNet: More Features from Cheap Operations" (2020)
핵심: 유령 단원이 파라메터를 덜 쓰고 피처를 더 만든다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3):
        super().__init__()
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, 1, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.ghost1 = GhostModule(in_channels, out_channels)
        
        if stride > 1:
            self.conv_dw = nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=out_channels, bias=False)
            self.bn_dw = nn.BatchNorm2d(out_channels)
        self.stride = stride
        
        self.ghost2 = GhostModule(out_channels, out_channels, ratio=1)
        
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.bn_dw(self.conv_dw(x))
        x = self.ghost2(x)
        return x + self.shortcut(residual)

class GhostNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)
        
        self.blocks = nn.Sequential(
            GhostBottleneck(16, 16),
            GhostBottleneck(16, 24, 2),
            GhostBottleneck(24, 24),
        )
        
        self.conv_head = nn.Conv2d(24, 960, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.act2 = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(960, num_classes)
    
    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x = self.blocks(x)
        x = self.act2(self.bn2(self.conv_head(x)))
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

if __name__ == "__main__":
    model = GhostNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 989,680
```

## 2. 논의

### 기본 피처와 추가 피처

`GhostModule`은 출력 폭 O와 ratio=s에서 m=⌊O/s⌋개의 기본 피처를 만들고, 그 피처에 그룹 컨볼루션을 적용해 O−m개를 더 만든다. 두 결과를 채널로 연결한다. 기본 피처는 공유하지만 서로 다른 변환을 학습할 수 있어 추가 피처가 단순 복사인 것은 아니다. 그렇다고 채널 수가 많아진 만큼 독립적인 정보가 보장되는 것도 아니다.

입력 폭 I, primary 커널 k, cheap 커널 d에서 바이어스·BN을 제외한 웨이트는 Imk²+(O−m)d²다. 두 번째 식은 groups=m이고 출력이 m의 정수배라는 조건을 사용한다. 같은 O출력을 만드는 표준 k×k의 IOk²와 비교할 수 있지만, 실제 비용은 공간 크기·BN·활성화·연결·장치 구현에도 영향을 받는다.

### ratio의 조건과 실제 오류

이 구현에는 ratio를 임의로 바꿔도 유효하도록 하는 채널 올림·슬라이싱 처리가 없다. m>0, O−m>0이고 cheap 출력 O−m이 그룹 수 m의 배수여야 한다. 예를 들어 ratio=2와 짝수 O는 이 조건을 맞추지만 일반 O나 ratio에서는 실패할 수 있다.

`GhostBottleneck.ghost2`는 **ratio=1**을 사용한다. 그러면 m=O, O−m=0이 되어 추가 피처 컨볼루션과 BN이 0채널을 갖는 경로를 만든다. 정상적인 cheap 연산이 아니며 그대로 유효한 순전파를 수행할 수 없다. ratio=1일 때 추가 경로를 생략하는 등의 별도 처리가 필요하지만 요청에 따라 코드는 유지했다. 기존 파라미터 수 출력이 이 경로의 실행 가능성을 확인한 것은 아니다.

### 병목과 축약 본체

병목은 첫 Ghost 모듈, 필요하면 stride depthwise, 두 번째 Ghost 모듈과 지름길을 합한다. 두 모듈 모두 ReLU를 쓰므로 선형 출력 투영을 사용하는 원 병목과도 구분해야 한다. 본체에는 병목 세 개만 있고 전체 GhostNet의 단계·SE·학습 레시피를 재현하지 않는다.

RGB 64×64 입력은 stem에서 16×32×32가 되고, 의도한 채널 흐름은 16→16→24→24이며 stride2 뒤 공간은 16×16이다. head는 960채널과 전역 풀링을 사용하지만 이 의도된 형태가 ratio=1의 실제 오류를 없애지는 않는다. [원 논문](https://openaccess.thecvf.com/content_CVPR_2020/html/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.html)의 결과와 예제 골격을 구분한다.

### 깊이 변경

연습의 ModuleList 코드는 정의되지 않은 `YourBlock`을 사용한다. 실제 병목을 반복하려면 입출력 폭·stride·지름길과 ratio의 유효성을 먼저 정해야 한다. ModuleList는 등록과 순회 구조를 제공할 뿐 잘못된 0채널 연산을 고치지 않는다. 코드·출력은 실행하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 (4,16,32,32), O=24, ratio=2인 GhostModule의 형태를 추적하라.

</div>

??? success "풀이"
    기본 폭 m=12이며 primary 1×1 뒤 (4,12,32,32), cheap depthwise 뒤 같은 형태다. 연결하면 (4,24,32,32)다. BN·ReLU는 채널 폭을 바꾸지 않는다. cheap 출력과 그룹 수의 나눗셈 조건도 맞는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
이 파일의 GhostNet을 64×64 RGB로 순전파할 때 형태 추적만으로 충분한가?

</div>

??? success "풀이"
    충분하지 않다. stem·블록·head의 의도된 폭이 맞아도 ghost2가 ratio=1로 0채널 cheap 연산을 구성한다. 입력 공간 크기를 바꾸거나 fc 폭을 맞추는 것으로 이 오류는 해결되지 않는다. 별도 ratio 분기와 연산 유효성을 먼저 검토해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Ghost 모듈과 표준 컨볼루션의 웨이트를 비교하라.

</div>

??? success "풀이"
    기본 피처 m개와 추가 O−m개를 만들면 바이어스·BN을 제외한 웨이트는 Imk²+(O−m)d²다. 그룹 수 m과 출력의 나눗셈 조건을 가정한다. 표준은 IOk²다. depthwise+pointwise를 순서대로 계산해 전체 출력을 만드는 구조와 달리 기본·추가 피처를 연결하므로 해당 구조의 식을 사용해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 반복 골격을 실제 Ghost 병목으로 확장하기 전에 검토할 조건을 설명하라.

</div>

??? success "풀이"

    ratio=1의 0채널 경로, 일반 ratio의 그룹 나눗셈, 병목별 입출력 폭·stride·지름길을 먼저 정해야 한다. YourBlock과 dim 등은 정의되지 않은 표시자이며 반복 컨테이너만 추가한다고 완성된 GhostNet이나 유효한 cheap 연산이 되는 것은 아니다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

Ghost 모듈은 기본 피처에서 추가 피처를 생성해 연결한다. 비용은 실제 기본 폭과 cheap 변환으로 계산하며, 이 구현의 ratio=1은 0채널 경로를 만들어 별도 처리가 필요하다. 단순 채널 추적과 파라미터 합산이 실행 검증을 대신하지 않는다.
