# SSD의 앵커별 예측

SSD는 여러 해상도의 피처 맵에서 기본 상자(default box)별 클래스와 위치 보정을 예측하는 단일 단계 탐지기다. [원 논문](https://arxiv.org/abs/1512.02325)은 다중 해상도·종횡비의 기본 상자와 매칭·학습 절차를 함께 제시한다. 아래 코드는 한 해상도의 예측 head만 보여 주는 축약 예제다.

## 1. 축약 모델 코드

```python
#!/usr/bin/env python3
'''
SSD — 한 방 여러 상자 알아내개
논문: "SSD: Single Shot MultiBox Detector" (2016)
핵심: 알아내기에 여러 잣수 피처 지도를 쓰며 피드포워드 한 번
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class SSD(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # 바탕 넷트웍(VGG 비슷한 것)
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # 여러 잣수 알아내기를 위한 덧레이어
        self.extras = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 알아내기 머리
        self.loc = nn.Conv2d(256, 4 * 4, 3, padding=1)  # 붙박이 상자 4개 * 자리표 4개
        self.conf = nn.Conv2d(256, 4 * num_classes, 3, padding=1)  # 상자 4개 * 클래스 수
    
    def forward(self, x):
        x = self.base(x)
        x = self.extras(x)
        
        loc = self.loc(x)
        conf = self.conf(x)
        
        return {'loc': loc, 'conf': conf}

if __name__ == "__main__":
    model = SSD()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**기존 출력:**

```
Parameters: 1,375,908
```

`base`는 컨볼루션과 두 번의 풀링으로 피처를 만들고 `extras`는 채널을 256으로 바꾼다. extras의 컨볼루션은 stride1·padding1이므로 새로운 저해상도 단계를 만들지 않는다. loc·conf 두 head는 모두 같은 피처 맵에 적용된다. 따라서 이 코드가 원 SSD의 여러 스케일 예측을 구현한다고 설명할 수 없다.

## 2. 위치와 클래스 출력

입력 4×3×64×64는 첫 두 컨볼루션 뒤에 4×64×64×64, 첫 풀링 뒤에 4×64×32×32가 된다. 다음 두 컨볼루션과 풀링 뒤에는 4×128×16×16, extras 뒤에는 4×256×16×16이다. 기본 클래스 수 21에서 반환 사전의 loc는 4×16×16×16, conf는 4×84×16×16이다. loc의 첫 16은 채널 수이고 마지막 두 16은 공간 크기다.

위치마다 기본 상자 4개를 가정하므로 loc 채널은 4×4, conf 채널은 4×21이다. 이미지당 후보는 4×16×16=1024개이며 실제 객체 수가 아니다. 원래 SSD의 클래스 softmax는 배경 클래스를 포함하므로 21은 배경 1개와 객체 클래스 20개로 해석한다. raw conf에 softmax를 적용하거나 loc를 이미지 좌표로 변환하는 과정은 현재 코드에 없다.

## 3. 기본 상자와 학습·후처리

기본 상자의 중심·크기·종횡비와 출력 채널 순서를 정의해야 loc의 오프셋을 상자로 복원할 수 있다. 학습에는 기본 상자와 정답 매칭, 양성 회귀 로스, 배경을 포함한 분류 로스와 hard negative mining 등이 필요하다. 이 예제에는 그러한 목표 할당·로스·디코딩·NMS가 없다.

다중 해상도 SSD로 확장할 때는 각 피처 단계의 공간 크기·앵커 수에 맞는 head를 만들고 모든 후보를 같은 좌표계로 합친다. 단순히 같은 해상도 컨볼루션을 더 깊게 쌓는 것과는 다르다. 입력 크기가 바뀌면 풀링의 바닥 나눗셈과 기본 상자의 스케일도 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
4×3×64×64 입력에서 extras·loc·conf의 모양과 후보 수를 계산하라.

</div>

??? success "연습문제 1 풀이"

    extras는 4×256×16×16, loc는 4×16×16×16, conf는 4×84×16×16이다. 이미지당 1024개 기본 상자에 대해 네 오프셋과 21클래스 로짓을 출력한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
현재 코드가 다중 스케일 탐지인가? extras를 통과하면 해상도가 줄어드는가?

</div>

??? success "연습문제 2 풀이"

    두 head는 모두 같은 extras 출력에 적용되므로 한 스케일이다. extras는 stride1·padding1의 3×3 컨볼루션 두 개여서 공간 크기를 유지한다. 다중 스케일에는 서로 다른 공간 크기의 피처와 각 피처에 대응하는 head가 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
객체 클래스가 10개이고 배경을 포함하면 loc·conf 채널을 어떻게 정하는가?

</div>

??? success "연습문제 3 풀이"

    기본 상자 수 4를 유지하면 loc는 16채널이고 num_classes=11로 두어 conf는 44채널이다. 클래스 수를 바꿔도 좌표 오프셋 개수는 변하지 않으며 목표 레이블과 분류 로스의 클래스 차원도 함께 맞춘다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList로 SSD를 확장할 때 필요한 피처·head 대응을 설명하라.

</div>

??? success "연습문제 4 풀이"

    YourBlock은 별도 정의가 필요하다. 각 단계에서 사용할 피처를 보관하고 단계별 입력 채널과 기본 상자 수에 맞는 loc·conf head를 등록해야 한다. 후보를 펼칠 때 위치·앵커·클래스 순서를 기본 상자 생성 순서와 맞춘다. 마지막 피처에만 head를 적용하면 다중 스케일 예측이 되지 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

SSD의 핵심은 기본 상자별 분류·회귀와 여러 해상도 피처의 활용이다. 이 축약 코드는 한 피처 맵의 raw 출력을 보여 준다. 후보 수와 객체 수, 배경 포함 클래스 수, 상자 오프셋과 디코딩된 좌표를 구분한다.
