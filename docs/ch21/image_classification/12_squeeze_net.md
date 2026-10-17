# SqueezeNet

SqueezeNet은 Fire 모듈을 이용해 웨이트 수를 줄인 CNN이다. Iandola 외의 2016년 논문은 특정 ImageNet 평가에서 AlexNet과 비슷한 정확도를 더 적은 파라메터로 얻었다. 아래 예제에는 일부 Fire 단계가 빠져 분류기의 입력 채널과 맞지 않으므로 전체 모델의 재현으로 읽으면 안 된다.

## 1. 코드

```python
import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels,
                 expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


if __name__ == "__main__":
    model = SqueezeNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 646,344
```

## 2. 논의

### Fire 모듈

squeeze의 1×1 컨볼루션으로 입력 채널을 줄이고, expand의 1×1과 3×3 분기를 병렬 적용해 채널로 연결한다. 출력 폭은 두 expand 폭의 합이다. 3×3 분기에 padding=1을 사용하므로 공간 크기가 같아 연결할 수 있다. squeeze가 3×3의 입력 폭을 줄이는 것이 웨이트 절감의 주요 원리다.

### 예제의 채널 불일치

`features`는 마지막 Fire(256,32,128,128)에서 256채널을 출력한다. 그러나 `classifier`의 첫 컨볼루션은 512채널을 요구하므로 순전파가 실패한다. 누락된 후반 단계를 연결하거나 분류기를 해당 피처에 맞추는 작업이 필요하지만, 이번 편집에서는 코드와 기존 출력을 그대로 유지했다. `__main__`의 파라메터 합산은 순전파 검증이 아니다.

### 파라미터 수·저장 크기·실행비용

원 SqueezeNet의 약 120만 파라메터와 논문 제목의 0.5MB 미만 저장 크기는 같은 표현 정밀도의 수치가 아니다. 후자는 추가적인 가지치기·양자화·부호화 등 압축을 적용한 결과로 구분해야 한다. 약 120만 값을 float32로 저장하면 웨이트만 대략 4.8MB이며 형식의 부가정보도 있을 수 있다. 적은 파라미터는 활성 메모리와 실제 지연시간이 같은 비율로 줄어든다는 뜻도 아니다.

### 우회 연결

Fire 출력과 입력의 배치·채널·공간 크기가 모두 같으면 원소별 덧셈을 사용할 수 있다. 채널만 검사하는 아래 예제는 기본 Fire가 공간 크기를 유지한다는 조건에 의존한다. 투영이 없는 항등 우회는 학습 파라메터를 추가하지 않지만 덧셈 연산은 추가한다. 정확도가 일정 비율 개선된다고 일반화할 수는 없다. [원 논문](https://arxiv.org/abs/1602.07360)의 구조·압축·실험을 각각 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 128, squeeze=16, expand 두 분기 각 64채널의 웨이트·바이어스 수를 계산하라.

</div>

??? success "풀이"
    squeeze는 128×16+16=2,064개, expand 1×1은 16×64+64=1,088개, expand 3×3은 16×64×9+64=9,280개다. 합계는 12,432개다. ReLU와 채널 연결에는 학습 파라메터가 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
Fire 출력 채널과 현재 본체의 분류기 입력을 비교하라.

</div>

??? success "풀이"
    expand가 각각 128채널이면 출력은 256채널이다. 이 파일의 마지막 Fire가 그 폭인데 분류기 컨볼루션은 512입력을 요구한다. 채널을 맞춘 후반 구조나 분류기 수정 없이는 실행 가능한 전체 모델이 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
우회 덧셈의 형태 조건과 비용을 설명하라.

</div>

??? success "풀이"

    입력·출력의 전체 형태가 같아야 한다. 기본 Fire는 공간 크기를 유지하므로 채널까지 같으면 덧셈할 수 있지만, 다른 stride나 padding을 도입하면 공간 조건도 확인해야 한다. 항등 우회는 학습 파라미터를 늘리지 않으나 덧셈·메모리 접근은 추가하고 정확도 개선량은 별도 평가 문제다.

    ```python
    class FireWithBypass(Fire):
        def forward(self, x):
            out = super().forward(x)
            if x.shape[1] == out.shape[1]:
                return out + x
            return out
    ```


## 정리하며

Fire 모듈은 채널 축소와 병렬 확장으로 웨이트 수를 조절한다. 예제의 누락 단계·채널 불일치와 원 모델의 결과를 구분하고, 파라미터 수·압축 저장 크기·실행비용을 같은 지표로 취급하지 않는다.
