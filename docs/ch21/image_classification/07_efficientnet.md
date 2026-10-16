# EfficientNet

이 페이지의 모델은 EfficientNet에서 사용하는 역병목 연산과 SiLU 활성화를 살피는 축약 예제다. stem 뒤 MBConv 3개와 분류 head만 사용하며, SE·단계별 반복·5×5 커널·복합 스케일링을 생략한다. 따라서 EfficientNet-B0 전체 구현이나 그 정확도를 재현한 모델로 읽으면 안 된다.

## 1. 코드

```python
import torch
import torch.nn as nn


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res = stride == 1 and in_channels == out_channels
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            MBConv(32, 16, expand_ratio=1),
            MBConv(16, 24, stride=2),
            MBConv(24, 40, stride=2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(40, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


if __name__ == "__main__":
    model = EfficientNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,352,888
```

## 2. 논의

### 축약 MBConv의 흐름

확장 비율이 1보다 크면 1×1로 채널을 넓힌 뒤 3×3 depthwise로 공간 피처를 계산하고 1×1로 출력 폭에 투영한다. stride=1이며 채널 수가 같을 때만 잔차를 더한다. 이 코드의 세 블록은 32→16→24→40으로 채널이 달라 모두 잔차 덧셈을 사용하지 않는다.

head는 40채널을 1280채널로 바꾼 뒤 전역 평균 풀링, 드롭아웃과 선형 분류기를 적용한다. 기본 출력은 1,000클래스 로짓이다. 전역 평균 풀링은 공간 피처를 요약하지만 앞단이 생략된 모델을 원 B0와 동등하게 만들지는 않는다.

### SiLU의 성질

SiLU는 f(x)=x·sigmoid(x)이며 도함수는 f′(x)=s(x)+xs(x)(1−s(x))다. 매끄럽고 음수 입력에서 작은 음수 출력을 허용한다. x≈−1.278에서 극솟값이 있고 그 지점의 도함수는 0이므로 ‘모든 음수 입력에서 경사가 0이 아니다’라는 주장은 틀리다. 매우 큰 음수에서는 출력과 경사가 0에 가까워질 수도 있다. ReLU의 0에서의 수학적 미분 불가능성과 자동미분 구현의 선택도 구분한다.

첫 블록의 확장 비율 1과 뒤의 6은 기준 설계의 선택이다. 채널이 적으면 확장이 쓸모없고 뒤에서는 반드시 확장해야 한다는 수학적 필연은 아니다. 폭·해상도별 비용과 표현력의 균형을 실험으로 비교한 설계로 이해한다.

### SE를 추가할 때

연습의 SE 클래스는 전역 평균→채널 축소→SiLU→채널 복원→sigmoid로 게이트를 만든다. 클래스를 정의하는 것만으로 MBConv에 삽입되지는 않는다. depthwise 뒤와 투영 전 등 연결 위치를 명시해야 한다.

hidden=192, reduction=4이면 축소 폭은 48이고 바이어스까지 포함한 웨이트 수는 192×48+48+48×192+192=18,672개다. 이 폭을 hidden 기준으로 잡는 연습은 원 EfficientNet처럼 확장 전 입력 폭을 기준으로 축소 폭을 정하는 구성과 다를 수 있다. 증가 비율은 블록의 확장·출력 폭에 따라 달라지며 일률적으로 1~2%라고 할 수 없다.

더 완전한 SE·스케일링 예제는 [심화 EfficientNet](04_efficientnet_advanced.md)을 참고한다. 기존 코드·출력·그림은 실행하거나 재생성하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
SiLU 도함수와 ReLU를 비교하고 음수 영역의 경사를 설명하라.

</div>

??? success "연습문제 1 풀이"
    f′(x)=s(x)+xs(x)(1−s(x))다. SiLU는 매끄럽지만 비단조이며 약 −1.278에서 도함수가 0이다. 큰 음수에서 경사 크기도 작아질 수 있어 경사 소실을 완전히 막는 함수는 아니다. x=0에서 SiLU의 도함수는 1/2이고 ReLU는 수학적으로 미분 불가능하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
확장 비율과 잔차 사용 조건을 코드에 맞춰 설명하라.

</div>

??? success "연습문제 2 풀이"
    확장 비율이 1이면 확장 1×1을 생략하고 그보다 크면 내부 폭을 늘린다. 잔차는 stride=1과 같은 채널 수라는 두 조건이 필요하다. 이 예제의 32→16, 16→24, 24→40 블록은 모두 채널 수가 달라 잔차를 쓰지 않는다. 비율 선택은 설계값이지 모든 모델에 대한 필연이 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 SE 클래스의 hidden=192, reduction=4일 때 바이어스를 포함한 파라메터 수를 구하고 삽입 조건을 설명하라.

</div>

??? success "연습문제 3 풀이"

    축소 폭은 48이고 두 1×1의 웨이트 18,432개와 바이어스 48+192개를 합해 18,672개다. 블록 비용에 대한 비율은 실제 구조로 계산해야 하며 1~2%로 일반화할 수 없다. 이 클래스 정의만으로 원 모델은 바뀌지 않는다. MBConv의 hidden 피처에 적용할 위치와 채널 수를 맞추고, 축소 폭이 0이 되지 않는지도 확인한다.

    ```python
    class SE(nn.Module):
        def __init__(self, channels, reduction=4):
            super().__init__()
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction, 1),
                nn.SiLU(),
                nn.Conv2d(channels // reduction, channels, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return x * self.se(x)
    ```


## 정리하며

이 축약 모델은 역병목과 SiLU를 설명하는 예제다. 전체 B0의 SE·반복·커널·스케일링을 재현하지 않으며, 잔차 조건과 추가 모듈의 실제 비용을 코드에 맞춰 해석해야 한다.
