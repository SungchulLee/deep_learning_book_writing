# ConvNeXt

ConvNeXt는 ResNet 계열의 CNN 설계와 학습 방식을 현대화한 구조다. Liu 외의 CVPR 2022 논문은 큰 depthwise 커널, 채널별 LayerNorm, 확장 MLP와 계층적 구성을 비교한다. 아래 코드는 블록의 핵심 연산을 보여 주는 축약형이며 공개 ConvNeXt-Tiny 전체 모델과는 다르다.

## 1. 코드

```python
import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + x


class ConvNeXt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 4, stride=4),
            nn.LayerNorm([96, 56, 56])
        )
        self.stages = nn.Sequential(*[ConvNeXtBlock(96) for _ in range(3)])
        self.norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = x.mean([2, 3])
        x = self.norm(x)
        return self.head(x)


if __name__ == "__main__":
    model = ConvNeXt()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 941,896
```

## 2. 논의

### 블록의 연산과 정규화 축

입력 N×C×H×W에 7×7 depthwise 컨볼루션을 적용한 뒤 N×H×W×C로 바꾸어 `LayerNorm(C)`를 수행한다. 이는 각 위치에서 채널 축을 정규화한다. 이어서 Linear(C,4C)–GELU–Linear(4C,C)를 적용하고 작은 채널별 Layer Scale을 곱한 뒤 원래 입력에 더한다. 공간 위치별 Linear는 같은 웨이트를 공유하는 1×1 컨볼루션과 대응한다.

7×7 depthwise는 국소 공간 연산이며 입력에 따라 웨이트를 계산하는 셀프 어텐션과 같은 연산이 아니다. LayerNorm이 BatchNorm보다 모든 조건에서 더 안정적·빠르다는 보장도 없다. 정규화하는 축과 학습·평가 시 통계의 사용 방식이 달라진다.

### Layer Scale과 stem

블록은 y=x+γ⊙F(x)다. γ를 10⁻⁶처럼 작게 시작하면 잔차 분기의 초기 기여가 작아 블록이 항등 함수에 가깝게 시작한다. 이는 전체 네트워크가 항등 함수라는 뜻은 아니다. stem과 분류기는 다른 변환이며 학습 중 γ가 반드시 단조롭게 커지는 것도 아니다.

예제 stem의 `LayerNorm([96,56,56])`는 채널뿐 아니라 공간 축까지 함께 정규화하고 위치별 affine 파라메터도 가진다. 위 블록의 채널별 LayerNorm과 다르며 stem 출력이 정확히 96×56×56이어야 한다. 224×224 입력은 그 조건을 만족하지만 임의 크기 입력을 자동으로 지원하지는 않는다. 공간 전체를 사용하는 이 stem 정규화 때문에 전체 모델의 입력 의존범위를 블록의 7×7만으로 계산해서도 안 된다.

### 축약 모델의 범위

코드는 96채널 블록 3개만 사용하고 후속 다중 해상도 단계와 단계 사이 다운샘플링을 생략한다. 원 ConvNeXt-Tiny의 전체 계층 구조·확률적 깊이·학습 레시피를 재현한 구현이 아니다. 코드의 파라메터 수와 원 논문의 ImageNet 결과를 직접 연결하지 않는다.

연습의 BN 변형은 NCHW 형식을 유지하고 Linear 대신 1×1 컨볼루션을 쓴다. BN은 채널별로 배치와 공간 축의 통계를 사용하며, 학습 모드에서 N=1이어도 공간 표본이 충분하면 계산 가능하다. 다만 작은 유효 표본 수와 상관, 학습·평가 통계 차이가 영향을 줄 수 있다. 코드와 출력은 재실행하지 않았다. [원 논문](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)의 성능은 전체 모델과 평가 조건에 대한 결과다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
stride=dilation=1인 3×3 컨볼루션 몇 개가 7×7 한 개와 같은 이론적 수용영역을 갖는가?

</div>

??? success "연습문제 1 풀이"
    중간 풀링이 없으면 n개를 쌓은 이론적 수용영역은 1+2n이므로 n=3이다. depthwise와 표준 컨볼루션은 채널 혼합 방식이 다르고 중간 활성화도 있으므로 같은 수용영역이 같은 연산·파라메터 수를 뜻하지는 않는다. 전체 모델에 공간 전체 정규화가 있으면 그 의존성도 별도로 고려한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
작은 Layer Scale 초기값이 블록과 전체 모델에 미치는 영향을 구분하라.

</div>

??? success "연습문제 2 풀이"
    y=x+γF(x)에서 γ가 작으면 초기 잔차 기여가 작아 해당 블록이 항등에 가깝다. F와 γ의 학습 경사에도 배율이 작용한다. 전체 모델의 stem·다운샘플링·분류기는 여전히 비항등이며 γ가 학습 중 반드시 증가한다는 보장도 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 BN 변형이 바꾸는 텐서 형식과 정규화 통계를 설명하라.

</div>

??? success "연습문제 3 풀이"

    NCHW에서 BatchNorm2d(C)를 적용하고 위치별 Linear 두 개를 1×1 컨볼루션 두 개로 바꾸므로 NHWC 순열이 필요 없다. BN은 배치·공간 통계를, 원 블록의 LN은 각 위치의 채널 통계를 사용한다. 원 예제 stem의 CHW LayerNorm까지 같은 연산으로 바뀌는 것은 아니다. 성능·지연시간의 우열은 실제 조건에서 평가해야 한다.

    ```python
    class ConvNeXtBlockBN(nn.Module):
        def __init__(self, dim, layer_scale_init=1e-6):
            super().__init__()
            self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
            self.norm = nn.BatchNorm2d(dim)  # LayerNorm에서 바꿈
            self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)  # 대신 Conv2d 쓰기
            self.act = nn.GELU()
            self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))

        def forward(self, x):
            input = x
            x = self.norm(self.dwconv(x))
            x = self.act(self.pwconv1(x))
            x = self.gamma * self.pwconv2(x)
            return input + x
    ```


## 정리하며

ConvNeXt 블록은 큰 depthwise 커널과 채널 MLP, LayerNorm·Layer Scale을 결합한다. 정규화 축과 데이터 형식이 중요하며 예제의 고정 공간 stem과 3블록 축약형을 원래의 전체 모델과 구분한다.
