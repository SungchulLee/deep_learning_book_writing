# 심화 MobileNet

MobileNet 계열은 모바일·임베디드 환경의 연산 예산을 고려한 CNN이다. MobileNetV1은 depthwise separable convolution을 활용하고, Sandler 외의 MobileNetV2(CVPR 2018)는 역잔차와 선형 병목을 도입했다. 아래 코드는 V2 블록의 핵심 구조와 너비 배율을 보여 준다.

## 1. 코드

```python
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """깊이별로 갈라지는 컨볼루션 = 깊이별 + 점별."""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1,
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class InvertedResidual(nn.Module):
    """뒤집은 잔차 덩이(MobileNetV2)."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        config = [
            [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
            [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        in_channels = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        blocks = []
        for expand, channels, num_blocks, stride in config:
            out_channels = int(channels * width_mult)
            for i in range(num_blocks):
                blocks.append(InvertedResidual(
                    in_channels, out_channels,
                    stride if i == 0 else 1, expand
                ))
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        final_channels = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(final_channels, num_classes)
        )
    
    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


if __name__ == "__main__":
    pass
```

## 2. 논의

### Depthwise와 pointwise 연산

입력 M채널, 출력 N채널, 커널 K×K의 표준 컨볼루션은 바이어스를 제외하면 K²MN개의 웨이트를 갖는다. 채널마다 하나의 필터를 적용하는 depthwise와 M→N의 1×1 pointwise로 나누면 K²M+MN개다. 비는 1/N+1/K²이고, N이 충분히 크며 K=3이면 대략 8~9배 적을 수 있다.

이 수치는 특정 분해의 웨이트·동일 해상도 연산 비교다. 실제 지연시간은 장치의 커널·메모리 접근 등에 영향을 받는다. V2의 확장 1×1까지 포함한 블록을 단순 depthwise 한 레이어와 같은 비용으로 계산해서는 안 된다. 이 파일의 `DepthwiseSeparableConv`는 별도 예시이며 `MobileNetV2` 본체는 `InvertedResidual`을 사용한다.

### 역잔차와 선형 병목

역잔차 블록은 좁은 입력을 넓은 내부 채널로 확장하고 depthwise로 공간을 처리한 뒤 좁은 출력으로 투영한다. 확장과 공간 연산 뒤에는 ReLU6를 사용하지만 최종 투영 뒤에는 ReLU를 두지 않는다. 낮은 차원의 표현에서 추가 비선형 절단을 피하려는 설계이며 정보 로스가 절대 없다는 보장은 아니다. 선형 투영도 차원을 줄이면 정보를 잃을 수 있다.

코드의 투영 뒤에는 BN이 있다. ‘선형 병목’은 이 위치에 ReLU 같은 추가 비선형 활성화를 넣지 않는다는 구조적 표현이지, 학습 모드의 배치 통계 계산까지 전체적으로 고정 선형 사상이라는 뜻은 아니다. stride=1이고 입력·출력 채널 수가 같을 때만 잔차 덧셈을 사용한다.

### 너비 배율과 구현의 범위

코드는 채널에 `width_mult`를 곱해 정수로 절삭한다. 배율이 너무 작으면 0채널이 될 수 있어 유효한 채널 수를 확인해야 한다. 공개 구현에서 사용하는 채널 정렬·반올림 규칙과는 다를 수 있다. 배율≤1에서는 마지막 1280채널 head를 고정하므로 모든 레이어가 동일한 비율로 줄어드는 것도 아니다.

기본 분류기는 10클래스다. 흔히 인용하는 1,000클래스 MobileNetV2의 약 350만 파라메터를 이 코드의 10클래스 모델에 그대로 대입하면 안 된다. 최종 선형 레이어의 입력 폭이 1280이면 클래스 수 1000→10 변경만으로 1280×990+990개의 파라메터가 줄어든다.

너비를 줄이면 표현력과 비용이 바뀌지만 정확도·지연시간이 항상 정해진 비율로 줄어드는 것은 아니다. 아래 연습 코드는 모델을 구성해 파라메터를 세는 용도이며 학습이나 정확도 비교를 수행하지 않는다. 구조의 근거는 [MobileNetV2 원 논문](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html)을 참고한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
256→256, 3×3에서 depthwise+pointwise의 파라메터를 계산하라.

</div>

??? success "연습문제 1 풀이"
    바이어스·BN을 제외하면 표준은 256²×9=589,824개, 분리는 256×9+256²=67,840개다. 표준/분리 비는 약 8.69다. 채널별 multiplier=1과 해당 입출력 폭을 가정하며 V2 블록의 확장레이어까지 포함한 값은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
투영 뒤 ReLU를 생략한 이유와 그 주장의 한계를 설명하라.

</div>

??? success "연습문제 2 풀이"
    좁은 출력 공간에서 음수 활성까지 제거하는 추가 비선형 절단을 피하려는 설계다. 그러나 투영 자체가 정보를 완전히 보존한다는 보장은 아니다. 코드에는 BN도 있으며, 역잔차의 덧셈은 stride와 채널 형태가 맞을 때만 사용한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
너비 배율 비교 코드가 확인하는 것과 확인하지 않는 것을 구분하라.

</div>

??? success "연습문제 3 풀이"

    아래 코드는 10클래스 모델의 학습 파라메터 수만 센다. 정확도·학습시간·장치 지연시간을 측정하지 않는다. 배율≤1에서 head의 1280채널은 고정되고 나머지 채널은 정수 절삭되므로 전체 파라메터가 배율의 제곱에 정확히 비례하지도 않는다. 1,000클래스 모델의 공개 수치와 직접 혼용하지 않는다.

    ```python
    for mult in [0.5, 0.75, 1.0]:
        model = MobileNetV2(num_classes=10, width_mult=mult)
        params = sum(p.numel() for p in model.parameters())
        print(f"width_mult={mult}: {params:,} parameters")
    ```


## 정리하며

MobileNetV2는 넓은 내부 연산과 좁은 잔차 경로, 비선형 활성화를 생략한 출력 병목을 결합한다. 분리 컨볼루션의 절감 비율은 가정한 채널과 포함 연산에 따라 계산하고, 실제 배포 비용과 정확도는 별도로 평가한다.
