# 심화 EfficientNet

EfficientNet은 기준 구조를 정한 뒤 깊이·채널 폭·입력 해상도를 함께 조절하는 복합 스케일링을 사용한다. Tan과 Le의 ICML 2019 논문은 탐색으로 얻은 B0과 스케일을 확대한 모델들을 비교한다. 아래 코드는 MBConv와 SE, 폭·깊이 반올림을 보여 주는 교육용 구현이며 원 논문의 학습과 성능을 재현하는 전체 실험은 아니다.

## 1. 코드

```python
import torch
import torch.nn as nn
import math


class SwishActivation(nn.Module):
    """스위시 활성화: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """쥐어짜기-북돋우기 덩이."""
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """손전화용 뒤집은 병목 컨볼루션(MBConv) 덩이."""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                     kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(hidden_dim, se_channels)
        
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x


class EfficientNet(nn.Module):
    """겹친 잣수 맞추기를 쓴 EfficientNet 구조."""
    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=10,
                 dropout=0.2):
        super().__init__()
        
        base_config = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        
        out_channels = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        in_channels = out_channels
        blocks = []
        for expand, channels, num_layers, stride, kernel in base_config:
            out_channels = self._round_filters(channels, width_mult)
            num_layers = self._round_repeats(num_layers, depth_mult)
            for i in range(num_layers):
                blocks.append(MBConvBlock(
                    in_channels, out_channels, kernel,
                    stride if i == 0 else 1, expand
                ))
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        
        final_channels = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(final_channels, num_classes)
        )
    
    def _round_filters(self, filters, width_mult):
        filters *= width_mult
        new_filters = int(filters + 4) // 8 * 8
        new_filters = max(8, new_filters)
        if new_filters < 0.9 * filters:
            new_filters += 8
        return int(new_filters)
    
    def _round_repeats(self, repeats, depth_mult):
        return int(math.ceil(depth_mult * repeats))
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def efficientnet_b0(num_classes=10):
    return EfficientNet(width_mult=1.0, depth_mult=1.0, num_classes=num_classes)


if __name__ == "__main__":
    pass
```

## 2. 논의

### 복합 스케일링의 근사

기준 모델에 깊이 d=αᵠ, 폭 w=βᵠ, 해상도 r=γᵠ를 적용한다. 표준 컨볼루션의 비용을 지배항으로 보면 깊이에 선형, 입출력 채널 폭과 공간 해상도에 각각 제곱으로 비례하므로 비용은 대략 dw²r²배다. αβ²γ²≈2이면 φ가 1 증가할 때 대략 두 배의 비용을 목표로 한다.

이는 채널 반올림, 블록 반복의 올림, depthwise·SE·head 비용 등을 생략한 근사다. 파라메터 수에는 입력 해상도의 제곱 배율이 직접 들어가지 않는다. 더 높은 해상도는 활성 메모리와 연산을 늘리지만 같은 컨볼루션 웨이트를 공유한다. φ 하나가 모든 공개 B모델의 계수와 입력 크기를 정확히 생성하는 것도 아니다.

### MBConv와 SE

코드는 1×1 확장, 채널별 공간 컨볼루션, SE 채널 게이팅, 1×1 선형 투영을 수행한다. SE는 전역 평균 피처를 축소·확장하고 sigmoid 게이트로 원래 피처를 곱한다. 채널 간 중요도를 합이 1인 확률로 정규화하는 연산은 아니다. stride=1이며 입출력 채널 수가 같을 때만 잔차를 더한다.

Depthwise 연산은 채널별 공간 필터링을 맡고 1×1 연산이 채널을 섞는다. 실제 MBConv 비용에는 확장 폭과 SE까지 포함되므로 단순 depthwise 한 레이어의 비용으로 전체 블록 효율을 설명하지 않는다.

### 구현과 평가 범위

너비는 8의 배수로 반올림하고 깊이는 단계별 반복 수를 올림한다. 모델 클래스는 입력 이미지를 자동으로 지정 해상도에 변환하지 않는다. 해상도는 데이터 전처리에서 정해야 한다. 기본 분류기는 10클래스이며 연습 코드의 1,000클래스 모델과 파라메터 수가 다르다. 원 설계의 학습 레시피나 확률적 깊이까지 모두 포함하지는 않는다.

연산 수와 실제 지연시간은 다른 지표다. 메모리 접근·커널 구현·장치·배치 크기에 따라 속도가 달라지므로 B0의 파라미터 수나 논문 정확도만으로 어떤 환경에서도 ResNet보다 빠르고 정확하다고 결론 내리지 않는다. 출판 당시 결과는 [원 논문](https://proceedings.mlr.press/v97/tan19a.html)의 평가 조건과 함께 읽는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
α=1.2, β=1.1, γ=1.15, φ=2의 배율과 기준 224 입력의 계산상 크기를 구하라. 이 값이 공개 B2의 정의인가?

</div>

??? success "연습문제 1 풀이"
    깊이는 1.44, 폭은 1.21, 해상도는 1.3225배다. 224×1.3225=296.24이며 정수 또는 8의 배수로 조정하면 296을 선택할 수 있다. 이는 주어진 가상 규칙의 결과이지 공개 EfficientNet-B2의 실제 입력 크기·계수를 정의하지 않는다. 공간 크기를 8의 배수로 만드는 것도 이 클래스의 자동 처리 규칙은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
128→128, 3×3 컨볼루션을 depthwise+pointwise로 대체할 때 바이어스·BN을 제외한 파라메터 비를 구하라.

</div>

??? success "연습문제 2 풀이"
    표준은 9×128²=147,456개, 분리 연산은 9×128+128²=17,536개다. 비는 약 8.41이다. 일반적으로 분리/표준의 비는 1/C_out+1/K²이며 그 역수가 줄어드는 배수다. K²+1/C_out이라는 식은 잘못된 비다. 이 비교에는 MBConv 확장과 SE가 포함되지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 B3–B7 폭·깊이 설정으로 구성한 모델의 규모를 해석할 때 어떤 조건을 함께 확인해야 하는가?

</div>

??? success "연습문제 3 풀이"

    예제는 1,000클래스와 해당 폭·깊이 반올림을 사용한다. 입력 해상도는 별도로 정해야 하며 파라메터 수만으로 비용과 정확도를 예측할 수 없다. 공개 모델의 대략적인 크기를 참고할 수 있지만 이 교육용 코드의 정확한 수치와 성능을 실행 없이 검증했다고 주장하지 않는다. 동일한 학습·평가·장치 조건으로 비교해야 한다.

    ```python
    configs = {
        'B3': (1.2, 1.4),
        'B4': (1.4, 1.8),
        'B5': (1.6, 2.2),
        'B6': (1.8, 2.6),
        'B7': (2.0, 3.1),
    }

    for name, (width, depth) in configs.items():
        model = EfficientNet(width_mult=width, depth_mult=depth, num_classes=1000)
        params = sum(p.numel() for p in model.parameters())
        print(f"EfficientNet-{name}: {params:,} parameters")
    ```


## 정리하며

복합 스케일링은 깊이·폭·해상도의 균형을 조절하는 설계 규칙이다. 비용의 근사 배율과 실제 반올림·블록 비용을 구분하고, 파라미터 수·연산량·지연시간·정확도를 별도로 평가한다.
