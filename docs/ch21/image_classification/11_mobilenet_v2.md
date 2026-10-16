# MobileNet V2

MobileNetV2는 넓은 내부 피처와 좁은 잔차 경로, 선형 투영을 사용하는 모바일 CNN이다. 아래 `InvertedResidual`은 그 블록을 보여 주지만 `MobileNetV2` 본체에는 중간 단계가 빠져 있어 그대로 순전파할 수 없다. 파라메터 개수를 출력하는 것과 모델의 실행 가능성·정확도를 확인하는 것은 다른 일이다.

## 1. 코드

```python
import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.classifier = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        return self.classifier(x)


if __name__ == "__main__":
    model = MobileNetV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,281,928
```

## 2. 논의

### 역잔차와 활성화

블록은 1×1 확장, 채널별 3×3, 1×1 투영으로 이어진다. 최종 투영에는 ReLU6를 두지 않아 좁은 표현의 추가적인 비선형 절단을 피한다. 정보가 완전히 보존된다는 보장은 아니다. stride=1이며 입출력 채널 수가 같을 때만 입력을 더한다.

ReLU6(x)=min(max(x,0),6)는 범위를 제한해 낮은 정밀도의 표현을 다루는 데 유용할 수 있다. 하지만 클리핑으로 정보가 로스될 수 있고 연산이 양자화 가능한 형태라는 사실만으로 장치 속도나 정확도까지 보장되지는 않는다. 양자화 방식·스케일·커널 지원을 함께 고려한다.

### 본체의 누락과 차원 불일치

이 파일의 `MobileNetV2.features`는 3→32채널 stem만 포함한다. `InvertedResidual`을 정의했지만 본체에서 사용하지 않고, `width_mult`도 실제 채널 구성에 사용하지 않는다. 전역 평균 뒤 피처는 N×32인데 분류기는 1280개의 입력을 요구하므로 `forward`에서 차원 불일치가 발생한다.

`__main__`은 파라메터 수만 세며 순전파하지 않으므로 기존 출력이 있어도 이 오류를 검증한 것은 아니다. 공개 1.0배·1,000클래스 MobileNetV2의 약 350만 규모를 이 미완성 본체의 규모나 성능으로 인용하지 않는다. 더 완전한 블록 구성은 [심화 MobileNet](05_mobilenet_advanced.md)을 참고한다.

### 연산량과 후속 활성화

stride=1, 같은 C채널, 확장 비율 6에서 컨볼루션 MAC은 HW(12C²+54C)다. 표준 3×3 두 레이어는 18HWC²다. 전자가 더 작은 조건은 C>9이며, BN·활성화·덧셈을 제외한 비교다. 곱셈과 덧셈을 각각 1 FLOP으로 세면 두 수를 모두 2배한다. 같은 입출력이라는 사실만으로 표현력과 장치 지연시간도 같은 기준이 되지는 않는다.

연습의 HardSwish는 x·ReLU6(x+3)/6이다. sigmoid를 구간별 선형 게이트로 근사하지만 전체 함수는 가운데 구간에서 이차식이다. 이를 정의하는 것만으로 본체의 활성화가 바뀌지 않으며 모든 모델에서 ReLU6보다 정확하거나 빠르다는 보장은 없다. [MobileNetV2 논문](https://arxiv.org/abs/1801.04381)의 구조와 예제의 범위를 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
확장 비율 6인 동일 C채널 역잔차와 3×3 두 레이어의 MAC을 비교하라.

</div>

??? success "연습문제 1 풀이"
    stride=1과 같은 H×W에서 각각 HW(12C²+54C), 18HWC²다. 첫 값이 작으려면 54C<6C², 즉 C>9다. BN·바이어스·활성화·잔차 덧셈을 제외했으며 곱셈과 덧셈을 2 FLOP으로 세면 두 값 모두 2배한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
ReLU6의 범위 제한과 저정밀 추론의 관계를 설명하라.

</div>

??? success "연습문제 2 풀이"
    출력을 [0,6]으로 제한해 표현 범위를 정할 수 있다. 그러나 포화에 따른 정보 로스가 있고 실제 정확도와 속도는 양자화의 스케일·정밀도·커널 지원에 달려 있다. 범위 제한만으로 정밀도가 보존되거나 실행이 빨라지는 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 HardSwish의 수식과 적용 범위를 설명하라.

</div>

??? success "연습문제 3 풀이"

    함수는 x·clamp(x+3,0,6)/6이다. x≤−3에서는 0, −3<x<3에서는 x(x+3)/6, x≥3에서는 x다. 이 클래스는 활성화만 정의하고 본체에 자동 삽입되지 않는다. 교체한 모델의 정확도와 장치 지연시간은 별도 평가가 필요하다.

    ```python
    class HardSwish(nn.Module):
        def forward(self, x):
            return x * torch.clamp(x + 3, 0, 6) / 6
    ```


## 정리하며

역잔차 블록의 설계와 미완성 본체의 상태를 구분한다. 이 파일의 본체는 32차원 피처와 1280입력 분류기가 맞지 않으며 너비 인수도 사용하지 않는다. 구조의 효율은 실제 연산과 장치 조건에 따라 평가해야 한다.
