# EfficientNet

EfficientNet은 2019년 논문 "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"에서 나왔으며, 깊이, 너비, 해상도의 세 차원을 한꺼번에 스케일링하는 원칙 있는 방법을 제시했다. 복합 스케일링 계수(compound coefficient)를 써서, 앞선 모델보다 파라메터와 FLOPs(부동소수점 연산 횟수)가 훨씬 적으면서도 최고 수준의 정확도를 이룬다. 기본 구조인 EfficientNet-B0는 뉴럴 아키텍처 탐색(NAS)으로 찾아냈다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
EfficientNet - 모델 스케일링 다시 보기
논문: "EfficientNet: Rethinking Model Scaling for CNNs" (2019)
핵심: 깊이·너비·해상도를 함께 스케일링하며 효율이 매우 높다
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

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
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
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
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

if __name__ == "__main__":
    model = EfficientNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,352,888
```

## 2. 논의

MBConv(Mobile Inverted Bottleneck Convolution, 모바일 역 병목 컨볼루션) 블록이 EfficientNet의 기본 구성 요소다. 역 잔차(inverted residual) 구조를 따른다. 먼저 채널 차원을 몇 배(흔히 6배)로 확장하고, 확장한 차원에서 깊이별 컨볼루션을 적용한 뒤, 더 작은 출력 차원으로 사영한다. 입력과 출력의 차원이 같으면 잔차 연결이 블록을 통째로 건너뛰어 그래디언트가 잘 흐르게 한다. ReLU 대신 SiLU(Swish) 활성화 함수를 써서 그래디언트가 더 매끄럽다.

복합 스케일링 방법이 EfficientNet의 핵심 이론적 기여다. 깊이(레이어 늘리기), 너비(채널 늘리기), 해상도(이미지 키우기)를 따로따로 스케일링하는 대신, 복합 계수 $\phi$로 셋을 함께 스케일링한다. 깊이 $d = \alpha^\phi$, 너비 $w = \beta^\phi$, 해상도 $r = \gamma^\phi$이고 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$를 만족시킨다. 이 제약 덕에 $\phi$가 1 오를 때마다 FLOPs가 대략 두 배가 되고, 세 차원 사이의 비율이 B0에서 B7까지 그대로 유지된다.

복합 스케일링의 실제 효과는 크다. EfficientNet-B7은 파라메터 6600만 개로 이미지넷에서 톱-1 정확도 84.3%를 이루는데, GPipe는 같은 84.3%에 5억 5700만 개가 든다. 차원을 균형 있게 스케일링하는 편이 한 차원만 극단적으로 키우는 것보다 훨씬 효율적이라는 뜻이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
복합 스케일링 파라메터 $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$와 $\phi = 2$가 주어졌을 때 스케일링된 깊이, 너비, 해상도의 배율을 계산하여라.

</div>

??? success "풀이"
    깊이 배율: $d = \alpha^\phi = 1.2^2 = 1.44$. 너비 배율: $w = \beta^\phi = 1.1^2 = 1.21$. 해상도 배율: $r = \gamma^\phi = 1.15^2 = 1.3225$. 확인해 보면 $\alpha \cdot \beta^2 \cdot \gamma^2 = 1.2 \times 1.21 \times 1.3225 \approx 1.919 \approx 2$이다. 그러므로 $\phi = 2$이면 기본 모델에 비해 레이어가 약 $1.44 \times$, 채널이 $1.21 \times$ 많고, 입력 해상도가 $1.32 \times$ 크다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
MBConv 블록이 마지막 사영에 또 다른 깊이별 컨볼루션이 아니라 $1 \times 1$ 컨볼루션을 쓰는 까닭은 무엇인가? 선형 병목(linear bottleneck)의 역할을 설명하여라.

</div>

??? success "풀이"
    $1 \times 1$ 사영은 확장된 표현을 비선형 활성화 함수 없이 더 낮은 차원으로 압축하는 "선형 병목" 역할을 한다. 이는 의도된 설계다. 확장된 공간(채널 6배)이 깊이별 컨볼루션으로 피처 사이의 풍부한 상호작용을 담고, 사영이 그 피처를 선형으로 결합한다. 여기에 비선형 활성화 함수를 더하면 MobileNetV2 논문이 보인 대로 낮은 차원의 병목에서 정보가 손실된다. 병목 차원에서 깊이별 컨볼루션을 쓰면 이미 압축된 차원에서 채널마다 따로 공간 필터링을 할 뿐이라, $1 \times 1$ 컨볼루션이 주는 채널 간 혼합을 놓친다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`MBConv` 블록의 깊이별 컨볼루션과 마지막 $1 \times 1$ 사영 사이에 SE(Squeeze-and-Excitation) 모듈을 더하여라.

</div>

??? success "풀이"
    ```python
    class MBConvSE(nn.Module):
        def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1, se_ratio=0.25):
            super().__init__()
            hidden_dim = in_channels * expand_ratio
            self.use_res = stride == 1 and in_channels == out_channels

            layers = []
            if expand_ratio != 1:
                layers.extend([
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True)
                ])
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True),
            ])
            self.pre_se = nn.Sequential(*layers)

            se_ch = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_ch, 1), nn.SiLU(inplace=True),
                nn.Conv2d(se_ch, hidden_dim, 1), nn.Sigmoid(),
            )
            self.project = nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            out = self.pre_se(x)
            out = out * self.se(out)
            out = self.project(out)
            if self.use_res:
                out = out + x
            return out
    ```
    SE 모듈은 깊이별 컨볼루션 뒤, 사영 앞에 놓여 확장된 채널 차원에서 동작한다.

## 정리하며

**다룬 것** — EfficientNet

MBConv(모바일 역 병목 컨볼루션) 블록이 EfficientNet의 기본 구성 요소다.

핵심 클래스는 `MBConv`, `EfficientNet`, `MBConvSE`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
